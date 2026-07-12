#!/usr/bin/env python3
"""
risk_manager.py - Управление рисками.

Архитектура:
  Single Responsibility: только проверки рисков.
  Не хранит позиции, не отправляет ордера.

  Типичный вызов RiskManager:
    risk = RiskManager(capital=100, config=...)
    approval = risk.check_entry(symbol="NEAR/USDT", direction=Direction.SHORT)
    if not approval.approved → пропускаем сделку
    risk.adjust_position_size(decision, approval) → фикс размера

  Риски:
    1. Capital: хватит ли денег, лимиты на 1 позицию
    2. Portfolio: общая экспозиция, дневной PnL guard
    3. Consecutive losses: макс проигрышей подряд
    4. Self-inflicted: buy_lock, stale cleanup
    5. Market: BTC filter (шорты только не up-trend)
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any

from models import (
    Direction, Mode, ApprovalResult, RiskConfig, BtcState,
    Position,
)
from position_manager import PositionManager

logger = logging.getLogger("risk_manager")


class RiskManager:
    """Единый центр проверки рисков."""

    def __init__(self, config: dict,
                 position_manager: Optional[PositionManager] = None,
                 risk_config: Optional[RiskConfig] = None,
                 on_block: Optional[Callable] = None):
        """
        config: api_config_final.json
        position_manager: для запросов текущих позиций
        risk_config: RiskConfig (если None — читает из config)
        on_block: callback(reason) при блокировке сделки
        """
        self.config = config
        self.pm = position_manager
        self.risk_config = risk_config or self._build_risk_config(config)
        self.on_block = on_block

        # 🤫 buy_lock: {symbol: unlock_time}
        self._buy_locks: Dict[str, float] = {}

        # 🧮 Дневной PnL
        self._daily_start_pnl: float = 0.0
        self._daily_peak_pnl: float = 0.0

        # 🛡️ Последнее срабатывание Portfolio Guard
        self._last_portfolio_guard: float = 0.0
        self._portfolio_guard_cooldown: float = 600.0  # 10 мин

        # 🔄 Re-entry cooldown
        self._reentry_cooldowns: Dict[str, float] = {}

        logger.info("RiskManager инициализирован")

    def _build_risk_config(self, config: dict) -> RiskConfig:
        """Собрать RiskConfig из конфига."""
        rm = config.get('risk_management', {})
        tr = config.get('trading', {})
        pm = rm.get('portfolio_management', {})

        max_positions = rm.get('max_open_positions', 25)
        short_max = min(max_positions // 3, 10)

        # Параметры для трейлинга
        trail_act = pm.get('trail_activation_pct', 2.5)
        trail_dist = pm.get('trail_distance_pct', 1.5)

        return RiskConfig(
            max_long_positions=max_positions,
            max_short_positions=short_max,
            max_open_positions=max_positions + short_max,
            long_position_pct=tr.get('max_position_size_percent', 15),
            short_position_pct=10.0,
            long_sl_pct=rm.get('stop_loss_percent', 1.5),
            short_sl_pct=1.5,
            long_tp_pct=rm.get('take_profit_percent', 10.0) or 10.0,
            short_tp_pct=6.0,
            long_trail_act=trail_act,
            long_trail_dist=trail_dist,
            short_trail_act=trail_act * 0.8,
            short_trail_dist=trail_dist * 0.8,
            long_max_consecutive_losses=tr.get('max_consecutive_losses', 3) or 3,
            short_max_consecutive_losses=2,
            max_daily_loss_pct=rm.get('max_daily_loss_percent', 3.0),
            portfolio_guard_drawdown=pm.get('portfolio_guard_drawdown', 5.0),
            max_order_value=rm.get('max_buy_order_usd', 90.0),
            min_position_value=rm.get('min_position_usd', 5.0),
            max_daily_trades=rm.get('max_daily_trades', 100),
            entry_cooldown_seconds=300,
            max_leverage=tr.get('max_leverage', 5.0),
        )

    def set_position_manager(self, pm: PositionManager) -> None:
        """Подключить PositionManager (если создаётся позже)."""
        self.pm = pm

    def update_daily_pnl(self, total_pnl: float) -> None:
        """Обновить дневной PnL для Portfolio Guard."""
        if self._daily_start_pnl == 0.0:
            self._daily_start_pnl = total_pnl
        self._daily_peak_pnl = max(self._daily_peak_pnl, total_pnl)

    # ── Основные проверки ───────────────────────────────────────────────

    def check_entry(self, symbol: str, direction: Direction,
                    price: float, score: float,
                    btc_state: Optional[BtcState] = None,
                    decision: Any = None) -> ApprovalResult:
        """Проверить, можно ли войти в сделку.

        Args:
            symbol: NEAR/USDT
            direction: LONG | SHORT
            price: текущая цена
            score: скор из DecisionEngine
            btc_state: состояние BTC
            decision: Decision объект (для дополнительных полей)

        Returns:
            ApprovalResult(approved=True/False, reason="...")
        """
        # ── 1. Score порог ──────────────────────────────────────────
        if score < self.risk_config.min_confidence_pct:
            return ApprovalResult.reject(f"Score {score:.0f} < min {self.risk_config.min_confidence_pct:.0f}")

        # ── 2. BTC filter ───────────────────────────────────────────
        if btc_state:
            if direction == Direction.SHORT and btc_state.short_blocked:
                return ApprovalResult.reject(f"Шорты заблокированы: BTC trend={btc_state.trend}")
            if direction == Direction.LONG and btc_state.long_blocked:
                return ApprovalResult.reject(f"Лонги заблокированы: BTC trend={btc_state.trend}")

        # ── 3. Уже есть позиция ─────────────────────────────────────
        if self.pm and self.pm.has_position(symbol):
            existing = self.pm.get_position(symbol)
            if existing.direction == direction:
                # DCA разрешён только если PnL в допустимом диапазоне
                if existing.pnl_pct < -0.5:
                    return ApprovalResult.reject(f"Уже {direction.value.upper()} позиция {symbol} в убытке ({existing.pnl_pct:.2f}%)")
                # Двойной вход — DCA
                return ApprovalResult.approve(max_size=self.risk_config.max_order_value)
            else:
                # Противоположная позиция — хедж (блокируем)
                return ApprovalResult.reject(f"Есть противоположная позиция {symbol}")
        elif self.pm:
            # ── 4. Макс позиций по направлению ──────────────────────
            rc = self.risk_config.for_direction(direction)
            if direction == Direction.LONG:
                if self.pm.long_count >= rc['max_positions']:
                    return ApprovalResult.reject(f"Макс лонгов: {rc['max_positions']}")
            else:
                if self.pm.short_count >= rc['max_positions']:
                    return ApprovalResult.reject(f"Макс шортов: {rc['max_positions']}")

            # ── 5. Макс всего ────────────────────────────────────────
            if self.pm.positions_count >= self.risk_config.max_open_positions:
                return ApprovalResult.reject(f"Макс позиций всего: {self.risk_config.max_open_positions}")

        # ── 6. Buy lock ──────────────────────────────────────────────
        lock_until = self._buy_locks.get(symbol, 0)
        if time.time() < lock_until:
            remaining = int(lock_until - time.time())
            return ApprovalResult.reject(f"Buy_lock: ещё {remaining}с")

        # ── 7. Re-entry cooldown ────────────────────────────────────
        cooldown_until = self._reentry_cooldowns.get(symbol, 0)
        if time.time() < cooldown_until:
            remaining = int(cooldown_until - time.time())
            return ApprovalResult.reject(f"Re-entry cooldown: ещё {remaining}с")

        # ── 8. Portfolio guard ──────────────────────────────────────
        if time.time() - self._last_portfolio_guard < self._portfolio_guard_cooldown:
            return ApprovalResult.reject("Portfolio guard активен")

        # ── 9. Consecutive losses ───────────────────────────────────
        if self.pm:
            rc = self.risk_config.for_direction(direction)
            losses = self.pm.get_consecutive_losses(symbol, direction)
            if losses >= rc['max_consecutive_losses']:
                return ApprovalResult.reject(
                    f"Consecutive losses: {losses} >= {rc['max_consecutive_losses']}")

        # ── 10. Дневные сделки ──────────────────────────────────────
        # (проверяет вызывающий код по daily_trades)

        # ── Проходит все проверки ────────────────────────────────────
        # Подсчитываем макс размер
        max_size = self._calc_max_position_size(direction, price)
        return ApprovalResult.approve(max_size=max_size)

    def check_exit(self, symbol: str, direction: Direction,
                   exit_price: float, reason: str) -> ApprovalResult:
        """Проверить, можно ли закрыть позицию (обычно всегда да)."""
        if not self.pm or not self.pm.has_position(symbol):
            return ApprovalResult.reject(f"Нет позиции {symbol}")
        return ApprovalResult.approve()

    def check_trade_frequency(self, side_str: str) -> bool:
        """Проверить лимит дневных сделок."""
        # Вызывается из main_cycle по daily_trades
        return True  # handled by caller

    # ── Buy lock ────────────────────────────────────────────────────────

    def set_buy_lock(self, symbol: str, duration_seconds: int = 300) -> None:
        """Заблокировать повторный вход в символ."""
        self._buy_locks[symbol] = time.time() + duration_seconds
        logger.info(f"🔒 [LOCK] {symbol}: {duration_seconds}с")

    def release_buy_lock(self, symbol: str) -> None:
        """Снять блокировку."""
        self._buy_locks.pop(symbol, None)
        logger.info(f"🔓 [LOCK] {symbol}: снята")

    def is_locked(self, symbol: str) -> bool:
        lock_until = self._buy_locks.get(symbol, 0)
        return time.time() < lock_until

    def set_reentry_cooldown(self, symbol: str) -> None:
        """После SL — запретить вход на 10 мин."""
        self._reentry_cooldowns[symbol] = time.time() + self.risk_config.reentry_cooldown
        logger.info(f"⌛ [COOLDOWN] {symbol}: {self.risk_config.reentry_cooldown:.0f}с")

    def clean_expired_locks(self) -> int:
        """Очистить истёкшие блокировки. Возвращает количество."""
        now = time.time()
        before = len(self._buy_locks)
        self._buy_locks = {k: v for k, v in self._buy_locks.items() if v > now}
        expired = before - len(self._buy_locks)

        # Тоже для cooldowns
        self._reentry_cooldowns = {k: v for k, v in self._reentry_cooldowns.items() if v > now}
        return expired

    # ── Portfolio Guard ────────────────────────────────────────────────

    def check_portfolio_guard(self, daily_pnl: float,
                              current_pnl_drawdown: float) -> bool:
        """Проверить Portfolio Guard.

        Блокирует новые входы если:
        - Дневной PnL просел на portfolio_guard_drawdown от пика
        - И прошло > cooldown с последнего срабатывания

        Returns:
            True если guard не сработал (можно торговать)
            False если guard сработал (блокируем)
        """
        if time.time() - self._last_portfolio_guard < self._portfolio_guard_cooldown:
            return False

        threshold = self.risk_config.portfolio_guard_drawdown
        if current_pnl_drawdown >= threshold:
            self._last_portfolio_guard = time.time()
            reason = (f"⛔ Portfolio Guard: просадка ${current_pnl_drawdown:.2f} "
                      f">= ${threshold:.2f}")
            logger.warning(reason)
            if self.on_block:
                self.on_block(reason)
            return False

        return True

    # ── Размер позиции ─────────────────────────────────────────────────

    def adjust_position_size(self, decision: Any, btc_state: BtcState,
                             approval: ApprovalResult,
                             direction: Optional[Direction] = None) -> float:
        """Скорректировать размер позиции по режиму рынка.

        Args:
            decision: Decision с score и position_size_pct
            btc_state: BtcState с рын. режимом
            approval: ApprovalResult с лимитами
            direction: направление (LONG/SHORT)

        Returns:
            финальный размер в USD
        """
        # Базовый размер
        pct = getattr(decision, 'position_size_pct', 15.0) or 15.0

        # Режим рынка
        if btc_state.trend == 'up_trend':
            pct *= 2.5
        elif btc_state.trend == 'down_trend':
            pct *= 0.7

        # Волатильность
        if btc_state.regime == 'calm':
            pct *= 0.8
        elif btc_state.regime == 'volatile':
            pct *= 0.6

        # Score
        if getattr(decision, 'score', 50) or 0 < 60:
            pct *= 0.5
        elif decision.score >= 80:
            pct *= 1.2

        # Лимиты: макс размер = 1/max_open_positions от капитала
        if not direction and hasattr(decision, 'side'):
            direction = Direction.SHORT if decision.side == 'short' else Direction.LONG
        _max_pos = max(self.risk_config.max_open_positions, 1)
        _max_share_pct = 100.0 / _max_pos  # 33.3% при max_open=3
        pct = min(pct, _max_share_pct)

        if self.pm:
            capital = self.pm.available_capital
        else:
            capital = 100.0  # fallback

        # Пенальти капитала (N позиций → меньше на каждую)
        existing_count = self.pm.positions_count if self.pm else 0
        if existing_count >= _max_pos:
            pct *= 0.0  # больше не входим, если слоты заняты
        elif existing_count >= _max_pos - 1:
            pct *= 0.5  # последний слот — половинный размер

        max_usd = capital * (min(pct, _max_share_pct) / 100)
        max_usd = min(max_usd, self.risk_config.max_order_value or 1000.0)
        if approval.max_position_size > 0:
            max_usd = min(max_usd, approval.max_position_size)

        return round(max_usd, 2)

    # ── PnL targets (SL/TP/трейлинг) ──────────────────────────────────

    def calc_sl_price(self, entry_price: float, direction: Direction,
                      btc_state: Optional[BtcState] = None) -> Optional[float]:
        """Рассчитать цену стоп-лосса."""
        cfg = self.risk_config.for_direction(direction)
        sl_pct = cfg['sl_pct']

        # Расширяем SL в спокойном рынке
        if btc_state and btc_state.regime == 'calm':
            sl_pct += 0.3

        if direction == Direction.LONG:
            return entry_price * (1 - sl_pct / 100)
        else:
            return entry_price * (1 + sl_pct / 100)

    def calc_tp_price(self, entry_price: float, direction: Direction) -> Optional[float]:
        """Рассчитать цену тейк-профита."""
        cfg = self.risk_config.for_direction(direction)
        tp_pct = cfg['tp_pct']

        if direction == Direction.LONG:
            return entry_price * (1 + tp_pct / 100)
        else:
            return entry_price * (1 - tp_pct / 100)

    def calc_trail_params(self, direction: Direction) -> tuple:
        """Рассчитать параметры трейлинга (activation, distance)."""
        cfg = self.risk_config.for_direction(direction)
        return cfg['trail_act'], cfg['trail_dist']

    def calc_max_hold_hours(self, direction: Direction) -> float:
        """Максимальное время удержания."""
        return self.risk_config.max_hold_hours

    # ── Приватные ──────────────────────────────────────────────────────

    def _calc_max_position_size(self, direction: Direction,
                                price: float) -> float:
        """Рассчитать макс размер позиции в USD.

        Делит доступный капитал на max_open_positions, чтобы каждая
        позиция занимала свою долю (например, 1/3 при max_open=3).
        """
        if self.pm:
            capital = self.pm.available_capital
        else:
            capital = 100.0
        _max_pos = max(self.risk_config.max_open_positions, 1)
        max_usd = capital / _max_pos
        max_usd = min(max_usd, self.risk_config.max_order_value or 1000.0)
        return round(max_usd, 2)

    # ── Status ─────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Статус для отчёта."""
        return {
            'config': {
                'max_long_positions': self.risk_config.max_long_positions,
                'max_short_positions': self.risk_config.max_short_positions,
                'max_open_positions': self.risk_config.max_open_positions,
                'long_position_pct': self.risk_config.long_position_pct,
                'short_position_pct': self.risk_config.short_position_pct,
                'long_sl_pct': self.risk_config.long_sl_pct,
                'short_sl_pct': self.risk_config.short_sl_pct,
                'long_tp_pct': self.risk_config.long_tp_pct,
                'short_tp_pct': self.risk_config.short_tp_pct,
                'portfolio_guard_drawdown': self.risk_config.portfolio_guard_drawdown,
                'max_consecutive_losses_long': self.risk_config.long_max_consecutive_losses,
                'max_consecutive_losses_short': self.risk_config.short_max_consecutive_losses,
            },
            'buy_locks': len(self._buy_locks),
            'reentry_cooldowns': len(self._reentry_cooldowns),
            'portfolio_guard_last': self._last_portfolio_guard,
            'daily_start_pnl': self._daily_start_pnl,
            'daily_peak_pnl': self._daily_peak_pnl,
        }
