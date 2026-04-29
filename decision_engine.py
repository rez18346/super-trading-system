#!/usr/bin/env python3
"""
decision_engine.py — Единый центр принятия торговых решений.

Архитектура:
  DecisionEngine — синглтон, который решает:
    - ВХОДИТЬ ли в позицию (на основе анализа трейдера)
    - ВЫХОДИТЬ ли из позиции (SL, TP, трейл, ML, детектор разворота)
    - КОГДА выходить (приоритезация сигналов)
  
  Все остальные модули (trader, monitor) только собирают данные и передают их сюда.
  Решение принимается в одном месте → выполняется через один executor.

Принципы:
  1. Никакой торговой логики вне этого файла
  2. Все решения логируются с причиной
  3. Приоритеты сигналов: SL > TP > ML exit > структурный разворот > трейл
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import importlib

logger = logging.getLogger('decision_engine')

# Чтобы избежать циклических импортов на уровне модуля,
# все импорты будут сделаны лениво внутри методов.


class SignalType(Enum):
    """Типы сигналов на выход, отсортированные по приоритету (0 = самый высокий)."""
    STOP_LOSS = 0          # Критический: цена ушла ниже SL
    TAKE_PROFIT = 1        # Цель достигнута (выше трейла — фиксируем прибыль)
    ML_EXIT = 2            # ML-модель говорит выйти
    STRUCTURAL_REVERSAL = 3 # Детектор разворота (HH/HL + RSI)
    TRAILING_STOP = 4      # Трейлинг-стоп сработал
    TIMED_EXIT = 5         # Тайм-аут удержания


class Decision:
    """
    Решение: что делать с позицией.
    
    Атрибуты:
      symbol: торговая пара
      action: 'hold' | 'exit' | 'adjust'
      signal_type: SignalType (для exit)
      reason: человекочитаемое описание
      priority: 0-100 (0 = срочно)
      quantity: сколько продавать (для exit)
      metadata: дополнительная информация
    """
    
    def __init__(self, symbol: str, action: str = 'hold',
                 signal_type: SignalType = None,
                 reason: str = '', priority: int = 100,
                 quantity: float = 0, metadata: Dict = None):
        self.symbol = symbol
        self.action = action
        self.signal_type = signal_type
        self.reason = reason
        self.priority = priority
        self.quantity = quantity
        self.metadata = metadata or {}
        self.timestamp = datetime.now(timezone.utc)
    
    def __repr__(self) -> str:
        if self.action == 'hold':
            return f"[HOLD] {self.symbol}: {self.reason}"
        tag = self.signal_type.name if self.signal_type else self.action.upper()
        return (f"[{tag}] {self.symbol}: {self.reason}"
                f" (qty={self.quantity:.4f})")


class DecisionEngine:
    """
    Единый центр принятия решений.
    
    Используется потоками trader и monitor для получения консультаций
    и принятия взвешенных решений.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str = None):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        
        self.config_path = config_path
        self.config = {}
        if config_path:
            import json
            with open(config_path) as f:
                self.config = json.load(f)
        
        # HMM-режим (кэшируется, обновляется из монитора)
        self.hmm_regime = 1  # NORMAL по умолчанию
        
        # Делегаты для внешних функций (чтобы не импортировать модули насильно)
        self.quick_struct_exit_fn: Optional[Callable] = None
        self.ml_exit_check_fn: Optional[Callable] = None
        self.confirm_15m_reversal_fn: Optional[Callable] = None
        
        # Комиссия биржи (для расчёта безубытка)
        self.fee_rate = 0.001  # 0.1%
        
        # Защита от повторных решений (чтобы не спамить sell)
        self._last_decisions: Dict[str, Dict] = {}
        
        # Тайм-аут на повторный вход после продажи (сек)
        self.reentry_cooldown = 3600  # 1 час
        
        # Статистика решений
        self.stats = {
            'total_decisions': 0,
            'exit_decisions': 0,
            'entry_decisions': 0,
        }
        
        logger.debug("🧠 DecisionEngine инициализирован")
    
    # ─── ПАРАМЕТРЫ РЕЖИМА ────────────────────────────────────────────────────
    
    def update_hmm_regime(self, regime: int) -> None:
        """Обновить HMM-режим (вызывается из монитора при каждой проверке)."""
        self.hmm_regime = regime
    
    def get_sl_tp_params(self) -> Tuple[float, float, float, float]:
        """
        Вернуть SL%, TP%, trail_activation%, trail_distance% 
        в зависимости от HMM-режима.
        """
        if self.hmm_regime == 0:  # CALM
            return 3.0, 8.0, 2.0, 1.5
        elif self.hmm_regime == 1:  # NORMAL
            return 5.0, 10.0, 1.5, 1.0
        elif self.hmm_regime == 2:  # VOLATILE
            return 8.0, 14.0, 1.0, 1.5
        else:
            return 5.0, 10.0, 1.5, 1.0
    
    # ─── РЕШЕНИЕ НА ВЫХОД ───────────────────────────────────────────────────
    
    def decide_exit(self, symbol: str, entry_price: float, current_price: float,
                    quantity: float, highest_price: float,
                    pnl_pct: float = None) -> Decision:
        """
        Принять решение о выходе из позиции.
        
        Приоритет (от высшего к низшему):
          1. Стоп-лосс (цена < SL)
          2. Тейк-профит (фиксируем прибыль до того как пробьёт трейл)
          3. ML-модель (вероятность падения высокая)
          4. Структурный разворот (HH/HL + RSI)
          5. Трейлинг-стоп (цена упала от максимума)
        
        Возвращает Decision (action='exit' если нужно выйти, иначе 'hold').
        """
        if pnl_pct is None:
            pnl_pct = (current_price - entry_price) / entry_price * 100
        
        sl_pct, tp_pct, trail_act, trail_dist = self.get_sl_tp_params()
        
        self.stats['total_decisions'] += 1
        
        # 1. СТОП-ЛОСС
        if pnl_pct <= -sl_pct:
            self.stats['exit_decisions'] += 1
            return Decision(
                symbol, 'exit', SignalType.STOP_LOSS,
                f"Стоп-лосс: PnL={pnl_pct:.2f}% (порог: -{sl_pct}%)",
                priority=0, quantity=quantity
            )
        
        # 2. ТЕЙК-ПРОФИТ (выше ML/разворота — фиксируем прибыль)
        if pnl_pct >= tp_pct:
            self.stats['exit_decisions'] += 1
            return Decision(
                symbol, 'exit', SignalType.TAKE_PROFIT,
                f"Тейк-профит: PnL={pnl_pct:.2f}% (порог: +{tp_pct}%)",
                priority=1, quantity=quantity
            )
        
        # 3. ML-ВЫХОД
        ml_signal = self._check_ml_exit(symbol, pnl_pct)
        if ml_signal:
            self.stats['exit_decisions'] += 1
            return Decision(
                symbol, 'exit', SignalType.ML_EXIT,
                f"ML-выход: {ml_signal}",
                priority=2, quantity=quantity
            )
        
        # 4. СТРУКТУРНЫЙ РАЗВОРОТ
        struct_signal = self._check_struct_reversal(symbol, pnl_pct, entry_price, current_price)
        if struct_signal:
            self.stats['exit_decisions'] += 1
            return Decision(
                symbol, 'exit', SignalType.STRUCTURAL_REVERSAL,
                f"Разворот: {struct_signal}",
                priority=3, quantity=quantity,
                metadata={'entry_price': entry_price, 'current_price': current_price}
            )
        
        # 5. ТРЕЙЛИНГ-СТОП
        trail_signal = self._check_trailing_stop(
            symbol, entry_price, current_price, highest_price,
            pnl_pct, trail_act, trail_dist
        )
        if trail_signal:
            self.stats['exit_decisions'] += 1
            return Decision(
                symbol, 'exit', SignalType.TRAILING_STOP,
                f"Трейлинг-стоп: {trail_signal}",
                priority=4, quantity=quantity
            )
        
        # 6. ДЕРЖАТЬ
        return Decision(symbol, 'hold', reason=f"PnL={pnl_pct:.2f}%, SL={sl_pct}%, держим")
    
    def _check_ml_exit(self, symbol: str, pnl_pct: float) -> Optional[str]:
        """Проверить ML-сигнал на выход."""
        if self.ml_exit_check_fn is None:
            try:
                from stop_loss_monitor_v5 import ml_exit_check as fn
                self.ml_exit_check_fn = fn
            except ImportError:
                return None
        
        try:
            result = self.ml_exit_check_fn(symbol, pnl_pct)
            if result and isinstance(result, tuple) and len(result) >= 2:
                prob = result[1] if len(result) > 1 else 0
                if prob < 0.50:
                    return f"ML prob={prob:.2f}"
            return None
        except Exception as e:
            logger.debug(f"[ML.exit] Ошибка: {e}")
            return None
    
    def _check_struct_reversal(self, symbol: str, pnl_pct: float,
                                entry_price: float, current_price: float) -> Optional[str]:
        """Проверить структурный разворот."""
        if self.quick_struct_exit_fn is None:
            try:
                from stop_loss_monitor_v5 import _quick_struct_exit as fn
                self.quick_struct_exit_fn = fn
            except ImportError:
                return None
        
        try:
            detected, reason = self.quick_struct_exit_fn(symbol, pnl_pct, entry_price, current_price)
            if detected:
                return reason
            return None
        except Exception as e:
            logger.debug(f"[STRUCT] Ошибка: {e}")
            return None
    
    def _check_trailing_stop(self, symbol: str, entry_price: float,
                              current_price: float, highest_price: float,
                              pnl_pct: float, trail_act: float,
                              trail_dist: float) -> Optional[str]:
        """Проверить трейлинг-стоп."""
        trail_active_pct = (highest_price - entry_price) / entry_price * 100
        
        if trail_active_pct >= trail_act:
            trail_level = highest_price * (1 - trail_dist / 100.0)
            if current_price <= trail_level:
                return (f"Активирован: H={highest_price:.4f}→{current_price:.4f}"
                        f" (активация: {trail_act}%, дист: {trail_dist}%)")
        
        return None
    
    # ─── РЕШЕНИЕ НА ВХОД ────────────────────────────────────────────────────
    
    def decide_entry(self, symbol: str, confidence: float, trend: str,
                     rsi: float, current_price: float,
                     current_positions_count: int, max_positions: int = 5) -> Decision:
        """
        Принять решение о входе в позицию (ансамблевый подход).
        
        Каждый голос даёт балл 0-100, итоговый финальный скор = сумма с весами.
        ML имеет вес 60%, остальные голоса — по 10-15%.
        
        Голоса:
          1. ML-модель (confidence от 0 до 100) — вес 60%
          2. RSI-голос — вес 15%
          3. Тренд-голос — вес 15%
          4. Волатильность-голос — вес 10%
        
        Жёсткие блокировки:
          - Лимит позиций (>5)
          - Кулдаун повторного входа (1 час)
          - 15М фильтр (не покупать на 2+ красных свечах)
        """
        now = time.time()
        
        # ─── ЖЁСТКИЕ БЛОКИРОВКИ ──────────────────────────────────────────
        
        if current_positions_count >= max_positions:
            return Decision(symbol, 'hold', reason=f"Максимум {max_positions} позиций")
        
        if symbol in self._last_decisions:
            last_exit = self._last_decisions[symbol]
            time_since = now - last_exit.get('exit_time', 0)
            if time_since < self.reentry_cooldown:
                remain = self.reentry_cooldown - time_since
                return Decision(
                    symbol, 'hold',
                    reason=f"Повторный вход через {remain/60:.1f} мин"
                )
        
        if not self._check_15m_filter(symbol):
            return Decision(
                symbol, 'hold',
                reason="15М фильтр: не покупаем на падающих свечах"
            )
        
        # ─── АНСАМБЛЬ ГОЛОСОВ ────────────────────────────────────────────
        
        entry_checks = self._evaluate_entry_ensemble(
            symbol, confidence, trend, rsi, current_price
        )
        
        if entry_checks['approved']:
            self.stats['entry_decisions'] += 1
            return Decision(
                symbol, 'enter', reason=entry_checks['reason'],
                priority=50, metadata=entry_checks
            )
        else:
            return Decision(
                symbol, 'hold', reason=entry_checks['reason']
            )
    
    def _check_15m_filter(self, symbol: str) -> bool:
        """Проверить 15М фильтр (не покупать на падающих свечах)."""
        if self.confirm_15m_reversal_fn is None:
            try:
                from industrial_trader import IndustrialTrader
                # Используем статический метод или создаём временный экземпляр
                # Временно — импортируем функцию напрямую
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "industrial_trader_module",
                    os.path.join(BASE_DIR, "industrial_trader.py")
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                # Берём экземпляр трейдера
                trader = mod.IndustrialTrader(self.config_path)
                self.confirm_15m_reversal_fn = trader._confirm_15m_reversal
            except Exception as e:
                logger.warning(f"[15M] Не удалось загрузить фильтр: {e}")
                return True  # Если не можем проверить — пропускаем (доверяем ML)
        
        try:
            return self.confirm_15m_reversal_fn(symbol)
        except Exception as e:
            logger.debug(f"[15M] Ошибка проверки {symbol}: {e}")
            return True
    
    _ENTRY_THRESHOLD = 65.0  # минимальный итоговый скор для входа (0-100)
    
    def _get_entry_threshold(self) -> float:
        """Адаптивный порог входа.
        
        На CALM рынке — снижаем, чтобы не пропускать входы.
        На VOLATILE — повышаем, защита от ложных сигналов.
        """
        base = self._ENTRY_THRESHOLD
        if self.hmm_regime == 0:  # CALM
            return base - 8  # 57 — легче войти
        elif self.hmm_regime == 2:  # VOLATILE
            return base + 5  # 70 — строже
        return base  # NORMAL = 65
    
    def _evaluate_entry_ensemble(self, symbol: str, confidence: float,
                                   trend: str, rsi: float,
                                   current_price: float) -> Dict:
        """
        Ансамбль голосов: каждый даёт балл 0-100.
        Итоговый скор = взвешенная сумма. Если >= порога — вход.
        
        Веса настроены так, что ML — главный голос (60%).
        Остальные — коррекция.
        
        Порог адаптивный — зависит от HMM-режима.
        """
        
        # ─── 1. ML-ГОЛОС (вес 60%) ─────────────────────────────────────
        # confidence уже 0-100 от ML-модели
        ml_score = min(max(confidence, 0), 100)
        
        # ─── 2. RSI-ГОЛОС (вес 15%) ────────────────────────────────────
        # Идеал: RSI 40-60 (нейтральная зона)
        # Перекупленность (70+): штраф
        # Перепроданность (30-): штраф
        if 40 <= rsi <= 60:
            rsi_score = 100  # идеально
        elif 30 <= rsi <= 70:
            rsi_score = 70   # допустимо
        elif rsi > 85:
            rsi_score = 10   # сильно перекуплен
        elif rsi < 20:
            rsi_score = 20   # сильно перепродан (может быть ловушка)
        else:
            rsi_score = 40   # умеренно
        
        # ─── 3. ТРЕНД-ГОЛОС (вес 15%) ──────────────────────────────────
        # ML уже учла тренд в confidence, но это дополнительная страховка
        if trend in ('strong_bullish',):
            trend_score = 100
        elif trend in ('bullish',):
            trend_score = 80
        elif trend in ('weak_bullish', 'neutral'):
            trend_score = 50
        elif trend in ('bearish',):
            trend_score = 20
        elif trend in ('strong_bearish',):
            trend_score = 0
        else:
            trend_score = 40
        
        # ─── 4. ВОЛАТИЛЬНОСТЬ-ГОЛОС (вес 10%) ─────────────────────────
        # Определяем через HMM-режим
        # CALM = низкая волатильность — осторожнее
        # VOLATILE = высокая — выше риск, но и выше потенциал
        if self.hmm_regime == 0:  # CALM
            vol_score = 50   # осторожно
        elif self.hmm_regime == 1:  # NORMAL
            vol_score = 80   # нормально
        elif self.hmm_regime == 2:  # VOLATILE
            vol_score = 40   # рискованно
        else:
            vol_score = 60
        
        # ─── ИТОГОВЫЙ СКОР ──────────────────────────────────────────────
        weights = {
            'ml': 0.60,
            'rsi': 0.15,
            'trend': 0.15,
            'volatility': 0.10,
        }
        
        final_score = (
            ml_score * weights['ml'] +
            rsi_score * weights['rsi'] +
            trend_score * weights['trend'] +
            vol_score * weights['volatility']
        )
        
        threshold = self._get_entry_threshold()
        
        details = (
            f"ML={ml_score:.0f} RSI={rsi_score:.0f} "
            f"TREND={trend_score:.0f} VOL={vol_score:.0f} "
            f"→ SCORE={final_score:.1f} (threshold={threshold}, regime={self.hmm_regime})"
        )
        
        if final_score >= threshold:
            return {
                'approved': True,
                'reason': f"ВХОД ({details})",
                'final_score': final_score,
                'ml_score': ml_score,
                'rsi_score': rsi_score,
                'trend_score': trend_score,
                'vol_score': vol_score,
                'price': current_price,
            }
        else:
            return {
                'approved': False,
                'reason': f"⏭️ {symbol}: {details}",
                'final_score': final_score,
                'ml_score': ml_score,
                'rsi_score': rsi_score,
                'trend_score': trend_score,
                'vol_score': vol_score,
            }
    
    # ─── Управление повторными входами ──────────────────────────────────────
    
    def record_exit(self, symbol: str, reason: str = "") -> None:
        """Запомнить момент выхода (для кулдауна повторного входа)."""
        self._last_decisions[symbol] = {
            'exit_time': time.time(),
            'reason': reason,
        }
        logger.info(f"🚫 Кулдаун входа: {symbol} на {self.reentry_cooldown/3600:.01f}ч")
    
    def set_reentry_cooldown(self, seconds: int) -> None:
        """Установить тайм-аут на повторный вход."""
        self.reentry_cooldown = max(seconds, 60)
    
    # ─── СТАТИСТИКА ─────────────────────────────────────────────────────────
    
    def get_stats(self) -> Dict:
        """Вернуть статистику решений."""
        return {
            **self.stats,
            'hmm_regime': self.hmm_regime,
            'active_cooldowns': len(self._last_decisions),
        }
    
    def reset_cooldowns(self) -> None:
        """Сбросить все тайм-ауты повторного входа."""
        self._last_decisions.clear()
        logger.info("🔄 Кулдауны сброшены")
