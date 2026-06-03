#!/usr/bin/env python3
"""
position_manager.py - Управление открытыми позициями.

Архитектура:
  Единое хранилище всех позиций (LONG + SHORT).
  PositionManager не отправляет ордера — он получает их от OrderManager
  через callback on_order_filled.

  Жизненный цикл:
    1. OrderManager.submit_order() → ордер исполнен
    2. OrderManager зовёт callback → PositionManager.on_order_filled()
    3. PositionManager создаёт Position и сохраняет в PG
    4. Каждый цикл: PositionManager.update_prices() + stale_cleanup()
    5. При выходе: PositionManager.close_position() + удаление из PG

  Баги, которые больше не повторятся:
    - buy_lock до проверки баланса (ставят OrderManager + RiskManager)
    - stale_cleanup удаляет не ту позицию (фильтр по direction + sync)
    - self.positions как dict — теперь PositionManager с единым API
"""

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Callable, Any

from models import (
    Direction, Mode, Position, PositionFilter, Order, OrderStatus,
    RiskConfig, TradeOutcome,
)
import db_pg as db

logger = logging.getLogger("position_manager")


class PositionManager:
    """Управление позициями."""

    def __init__(self, exchange, config: dict,
                 on_position_closed: Optional[Callable] = None):
        """
        exchange: CCXT биржа
        config: полный конфиг
        on_position_closed: callback(symbol, direction, pnl_pct) при закрытии
        """
        self.exchange = exchange
        self.config = config
        self.on_position_closed = on_position_closed

        # Единое хранилище: {symbol: Position}
        # Для LONG и SHORT — один словарь. Один символ = одна позиция.
        self._positions: Dict[str, Position] = {}

        # Капитал (следим в реальном времени)
        self._available_capital: float = 0.0
        self._total_capital: float = 0.0

        # Счётчик consecutive losses (по direction)
        self._consecutive_losses: Dict[str, int] = defaultdict(int)

        # Progress SL (для SL→BE отслеживания)
        self._sl_updated: Set[str] = set()

        logger.info("PositionManager инициализирован")

    # ── Свойства ────────────────────────────────────────────────────────

    @property
    def available_capital(self) -> float:
        return self._available_capital

    @property
    def total_capital(self) -> float:
        return self._total_capital

    @property
    def positions_count(self) -> int:
        return len(self._positions)

    @property
    def long_positions(self) -> List[Position]:
        return [p for p in self._positions.values() if p.direction == Direction.LONG]

    @property
    def short_positions(self) -> List[Position]:
        return [p for p in self._positions.values() if p.direction == Direction.SHORT]

    @property
    def long_count(self) -> int:
        return len(self.long_positions)

    @property
    def short_count(self) -> int:
        return len(self.short_positions)

    @property
    def total_exposure(self) -> float:
        """Общая стоимость всех позиций в USD."""
        return sum(p.position_value for p in self._positions.values())

    @property
    def position_symbols(self) -> Set[str]:
        return set(self._positions.keys())

    # ── Управление ──────────────────────────────────────────────────────

    def on_order_filled(self, order: Order) -> Optional[Position]:
        """Callback от OrderManager: ордер исполнился.

        Создаёт позицию. Если позиция уже есть — обновляет (DCA).
        """
        if order.side == 'buy':
            # Открытие LONG или закрытие SHORT
            if order.direction == Direction.LONG:
                return self._open_long(order)
            else:
                # Закрытие SHORT (buy to cover)
                return self._close_short(order)
        else:  # sell
            if order.direction == Direction.LONG:
                # Закрытие LONG
                return self._close_long(order)
            else:
                # Открытие SHORT
                return self._open_short(order)

    def open_position(self, symbol: str, direction: Direction,
                      quantity: float, entry_price: float,
                      mode: Mode = Mode.SPOT,
                      leverage: int = 1,
                      sl_price: Optional[float] = None,
                      tp_price: Optional[float] = None,
                      trail_activation: Optional[float] = None,
                      trail_distance: Optional[float] = None,
                      max_hold_hours: Optional[float] = None,
                      entry_order_id: Optional[str] = None,
                      ) -> Position:
        """Создать позицию (после исполнения ордера)."""
        now = time.time()
        pos = Position(
            symbol=symbol,
            direction=direction,
            mode=mode,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            leverage=leverage,
            sl_price=sl_price,
            tp_price=tp_price,
            trail_activation=trail_activation,
            trail_distance=trail_distance,
            max_hold_hours=max_hold_hours,
            highest_price=entry_price,
            lowest_price=entry_price,
            entry_time=datetime.now(timezone.utc),
            created_at=now,
            entry_order_id=entry_order_id,
            position_id=f"{symbol}_{direction.value}_{int(now)}",
        )

        self._positions[symbol] = pos
        # 💰 Уменьшаем доступный капитал
        cost = pos.entry_cost
        self._available_capital = max(0, self._available_capital - cost)
        self._sl_updated.discard(symbol)

        # Сохраняем в PG
        try:
            db_pos = {k: v for k, v in pos.to_dict().items() if k != 'pnl_pct' and k != 'pnl_usd'}
            db_pos['side'] = direction.value
            db_pos['entry_time'] = pos.entry_time.isoformat()
            db_pos['_sl_price'] = sl_price
            db_pos['_tp_price'] = tp_price
            db_pos['_trail_act'] = trail_activation
            db_pos['_trail_dist'] = trail_distance
            db_pos['_max_hold_h'] = max_hold_hours
            db_pos['_created_at'] = now
            db.update_pos_meta(symbol, db_pos)
        except Exception as e:
            logger.debug(f"[PM] PG save: {e}")

        logger.info(f"📥 [{direction.value.upper()}] {symbol}: {quantity:.4f} @ ${entry_price:.4f} "
                    f"SL={sl_price} TP={tp_price} капитал=${self._available_capital:.2f}")
        return pos

    def close_position(self, symbol: str, exit_price: float,
                       exit_reason: str = "",
                       exit_order_id: Optional[str] = None) -> Optional[TradeOutcome]:
        """Закрыть позицию по текущей цене."""
        pos = self._positions.get(symbol)
        if not pos:
            logger.warning(f"⚠️ [CLOSE] {symbol}: нет в хранилище")
            return None

        # Расчёт PnL
        pnl_usd = pos.pnl_usd
        pnl_pct = pos.pnl_pct
        hold_hours = pos.age_hours

        # Создаём outcome
        outcome = TradeOutcome(
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd,
            hold_hours=hold_hours,
            exit_reason=exit_reason,
            entry_time=pos.entry_time,
        )

        # 💰 Возвращаем капитал
        cost = pos.entry_cost
        self._available_capital += cost + pnl_usd

        # Удаляем из хранилища
        self._positions.pop(symbol, None)
        self._sl_updated.discard(symbol)

        # Удаляем из PG
        try:
            db.remove_position(symbol)
        except Exception:
            pass

        # Записываем в PG trade_history (exit)
        try:
            db.add_trade(
                symbol=symbol,
                side='sell' if pos.direction == Direction.LONG else 'buy',
                price=float(exit_price),
                quantity=float(pos.quantity),
                pnl=float(pnl_usd),
                pnl_pct=float(pnl_pct),
                exit_reason=exit_reason[:200],
            )
        except Exception:
            pass

        # 🧠 Consecutive losses
        was_loss = pnl_usd < 0
        if was_loss:
            key = f"{symbol}_{pos.direction.value}"
            self._consecutive_losses[key] += 1
        else:
            # Если сделка профитная — сбрасываем счётчик для этого символа
            key = f"{symbol}_{pos.direction.value}"
            self._consecutive_losses.pop(key, None)

        # Callback
        if self.on_position_closed and callable(self.on_position_closed):
            try:
                self.on_position_closed(symbol, pos.direction, pnl_pct)
            except Exception as e:
                logger.error(f"[PM] callback error: {e}")

        logger.info(f"📤 [{pos.direction.value.upper()}] {symbol}: PnL={pnl_usd:+.2f}$ ({pnl_pct:+.2f}%) "
                    f"за {hold_hours:.1f}ч | {exit_reason}")
        return outcome

    def get_position(self, symbol: str) -> Optional[Position]:
        """Получить позицию по символу."""
        return self._positions.get(symbol)

    def get_positions(self, direction: Optional[Direction] = None,
                      mode: Optional[Mode] = None) -> List[Position]:
        """Получить позиции с фильтрацией."""
        result = list(self._positions.values())
        if direction:
            result = [p for p in result if p.direction == direction]
        if mode:
            result = [p for p in result if p.mode == mode]
        return result

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def is_long(self, symbol: str) -> bool:
        pos = self._positions.get(symbol)
        return pos is not None and pos.direction == Direction.LONG

    def is_short(self, symbol: str) -> bool:
        pos = self._positions.get(symbol)
        return pos is not None and pos.direction == Direction.SHORT

    def get_consecutive_losses(self, symbol: str, direction: Direction) -> int:
        """Сколько убыточных сделок подряд по символу + направлению."""
        return self._consecutive_losses.get(f"{symbol}_{direction.value}", 0)

    def record_success(self, symbol: str, direction: Direction) -> None:
        """Сбросить счётчик проигрышей при профитной сделке."""
        self._consecutive_losses.pop(f"{symbol}_{direction.value}", None)

    # ── Обновление цен ──────────────────────────────────────────────────

    def update_prices(self, prices: Dict[str, float]) -> int:
        """Обновить цены всех позиций.

        Args:
            prices: {symbol: current_price}

        Returns:
            количество позиций с изменением
        """
        updated = 0
        for symbol, pos in self._positions.items():
            price = prices.get(symbol)
            if price and price > 0:
                pos.update_price(price)
                updated += 1
        return updated

    def activate_trail(self, symbol: str) -> bool:
        """Активировать трейлинг для позиции (SL=entry, после 1.5%)."""
        pos = self._positions.get(symbol)
        if not pos:
            return False
        return pos.update_sl_to_breakeven()

    # ── Синхронизация с биржей ──────────────────────────────────────────

    def sync_with_exchange(self) -> int:
        """Сверить хранимые позиции с реальным балансом биржи.

        Удаляет позиции, которых нет на бирже (stale cleanup).

        Returns:
            количество удалённых позиций
        """
        removed = 0
        try:
            balance = self.exchange.fetch_balance()
            # Обновляем капитал
            free_usdt = balance['free'].get('USDT', 0)
            total_usdt = balance['total'].get('USDT', 0)
            self._available_capital = free_usdt
            self._total_capital = total_usdt + self.total_exposure

            for symbol in list(self._positions.keys()):
                currency = symbol.split('/')[0]
                pos = self._positions[symbol]
                real_qty = balance['total'].get(currency, 0)

                # 🛡️ Защита свежих позиций (первые 30с)
                if time.time() - pos.created_at < 30:
                    continue

                # ⚡ SHORT: позиции имеют нулевой баланс на бирже (монеты заняты и проданы)
                # Не удаляем — SL/TP/трейлинг управляются через PositionManager
                if pos.direction == Direction.SHORT:
                    if real_qty < 0.000001:
                        # Обновляем только цену, не трогаем позицию
                        logger.debug(f"🔄 [SYNC] {symbol}: SHORT, баланс={real_qty} — пропускаем синхронизацию")
                        continue
                    else:
                        # SHORT с ненулевым балансом — частично закрыт?
                        if real_qty < pos.quantity * 0.5:
                            old_qty = pos.quantity
                            pos.quantity = max(real_qty, 0.0001)
                            logger.info(f"🔄 [SYNC] {symbol}: SHORT qty {old_qty:.4f} → {pos.quantity:.4f}")
                        continue

                # На бирже нет актива — удаляем из памяти (только для LONG)
                if real_qty < 0.000001:
                    logger.warning(f"🔄 [SYNC] {symbol}: нет на бирже (qty={real_qty}). Удаляю.")
                    self._positions.pop(symbol, None)
                    db.remove_position(symbol)
                    removed += 1
                    continue

                # Пыль (< $1 или < 1% от исходного)
                pos_value = real_qty * pos.entry_price
                if pos_value < 1.0 or real_qty < pos.quantity * 0.01:
                    logger.warning(f"🔄 [SYNC] {symbol}: пыль ${pos_value:.2f} ({real_qty:.6f} из {pos.quantity:.4f}). Удаляю.")
                    self._positions.pop(symbol, None)
                    db.remove_position(symbol)
                    removed += 1
                    continue

                # Количество сильно меньше — обновляем
                if real_qty < pos.quantity * 0.5:
                    old_qty = pos.quantity
                    pos.quantity = real_qty
                    logger.info(f"🔄 [SYNC] {symbol}: qty {old_qty:.4f} → {real_qty:.4f}")

        except Exception as e:
            logger.warning(f"⚠️ [SYNC] ошибка: {e}")

        return removed

    def load_from_db(self) -> int:
        """Восстановить позиции из PostgreSQL при старте."""
        loaded = 0
        try:
            pg_positions = db.load_open_positions()
            if not pg_positions:
                return 0

            for symbol, pos_data in pg_positions.items():
                if symbol not in self._positions and pos_data.get('quantity', 0) > 0:
                    # ⚡ SHORT: распознаём direction из pos_meta
                    direction_str = pos_data.get('side', 'long').lower()
                    if direction_str not in ('long', 'short'):
                        direction_str = 'long'
                    direction = Direction.LONG if direction_str == 'long' else Direction.SHORT

                    # Восстанавливаем через Position.from_dict
                    try:
                        pos = Position.from_dict({
                            'symbol': symbol,
                            'direction': direction,
                            'mode': 'spot',
                            'quantity': pos_data.get('quantity', 0),
                            'entry_price': pos_data.get('entry_price', 0),
                            'current_price': pos_data.get('entry_price', 0),
                            'sl_price': pos_data.get('_sl_price'),
                            'tp_price': pos_data.get('_tp_price'),
                            'trail_activation': pos_data.get('_trail_act'),
                            'trail_distance': pos_data.get('_trail_dist'),
                            'max_hold_hours': pos_data.get('_max_hold_h'),
                            'created_at': pos_data.get('_created_at', time.time()),
                            'entry_time': pos_data.get('entry_time', datetime.now(timezone.utc).isoformat()),
                            'max_profit_pct': pos_data.get('max_profit', 0),
                        })
                        self._positions[symbol] = pos
                        loaded += 1
                        logger.info(f"   🗄️ {symbol}: восстановлена из PG "
                                    f"(entry=${pos.entry_price:.4f}, direction={direction_str})")
                    except Exception as e:
                        logger.warning(f"   ⚠️ {symbol}: ошибка восстановления: {e}")

            if loaded:
                logger.info(f"🗄️ Восстановлено {loaded} позиций из PostgreSQL")
        except Exception as e:
            logger.warning(f"⚠️ [DB] load: {e}")

        return loaded

    def save_state(self) -> None:
        """Сохранить состояние всех позиций в PG (периодически)."""
        try:
            pos_dicts = {sym: pos.to_dict() for sym, pos in self._positions.items()}
            db.save_all_positions(pos_dicts)
        except Exception as e:
            logger.debug(f"[PM] save state: {e}")

    # ── Status ──────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Статус для отчёта."""
        long_pos = self.long_positions
        short_pos = self.short_positions
        return {
            'count': len(self._positions),
            'long': len(long_pos),
            'short': len(short_pos),
            'exposure': round(self.total_exposure, 2),
            'capital': {
                'total': round(self._total_capital, 2),
                'available': round(self._available_capital, 2),
                'in_positions': round(self.total_exposure, 2),
            },
            'consecutive_losses': dict(self._consecutive_losses),
            'positions': {sym: p.to_dict() for sym, p in self._positions.items()},
        }

    # ── Приватные методы (создание/закрытие) ────────────────────────────

    def _open_long(self, order: Order) -> Position:
        if order.symbol in self._positions:
            existing = self._positions[order.symbol]
            if existing.direction == Direction.LONG:
                # DCA: усредняем цену
                total_qty = existing.quantity + order.filled_qty
                total_cost = existing.quantity * existing.entry_price + order.filled_qty * order.avg_price
                existing.quantity = total_qty
                existing.entry_price = total_cost / total_qty
                logger.info(f"📥 [LONG DCA] {order.symbol}: qty={existing.quantity:.4f} "
                            f"entry=${existing.entry_price:.4f}")
                return existing
            else:
                # Есть шорт — закрываем его этой покупкой
                return self._close_short(order)
        else:
            return self.open_position(
                symbol=order.symbol,
                direction=Direction.LONG,
                quantity=order.filled_qty,
                entry_price=order.avg_price,
                entry_order_id=order.order_id,
                **self._extract_sl_tp_from_metadata(order),
            )

    def _open_short(self, order: Order) -> Position:
        if order.symbol in self._positions:
            existing = self._positions[order.symbol]
            if existing.direction == Direction.SHORT:
                # DCA for short
                total_qty = existing.quantity + order.filled_qty
                total_cost = existing.quantity * existing.entry_price + order.filled_qty * order.avg_price
                existing.quantity = total_qty
                existing.entry_price = total_cost / total_qty
                logger.info(f"📥 [SHORT DCA] {order.symbol}: qty={existing.quantity:.4f} "
                            f"entry=${existing.entry_price:.4f}")
                return existing
            else:
                # Есть лонг — ошибочная ситуация, не обрабатываем
                logger.error(f"❌ [SHORT] {order.symbol}: уже есть LONG! Невозможно открыть шорт.")
                return existing
        else:
            return self.open_position(
                symbol=order.symbol,
                direction=Direction.SHORT,
                quantity=order.filled_qty,
                entry_price=order.avg_price,
                entry_order_id=order.order_id,
                **self._extract_sl_tp_from_metadata(order),
            )

    def _close_long(self, order: Order) -> Optional[TradeOutcome]:
        logger.info(f"📤 [LONG→CLOSE] {order.symbol}: sell {order.filled_qty:.4f} @ ${order.avg_price:.4f}")
        return self.close_position(
            symbol=order.symbol,
            exit_price=order.avg_price,
            exit_reason=order.metadata.get('reason', 'order_filled'),
            exit_order_id=order.order_id,
        )

    def _close_short(self, order: Order) -> Optional[TradeOutcome]:
        logger.info(f"📤 [SHORT→CLOSE] {order.symbol}: buy {order.filled_qty:.4f} @ ${order.avg_price:.4f}")
        return self.close_position(
            symbol=order.symbol,
            exit_price=order.avg_price,
            exit_reason=order.metadata.get('reason', 'order_filled'),
            exit_order_id=order.order_id,
        )

    def _extract_sl_tp_from_metadata(self, order: Order) -> dict:
        """Извлечь SL/TP/трейлинг из metadata ордера."""
        meta = order.metadata or {}
        return {
            'sl_price': meta.get('sl_price'),
            'tp_price': meta.get('tp_price'),
            'trail_activation': meta.get('trail_activation'),
            'trail_distance': meta.get('trail_distance'),
            'max_hold_hours': meta.get('max_hold_hours'),
        }
