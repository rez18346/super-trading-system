#!/usr/bin/env python3
"""
decision_engine.py — Единый центр принятия торговых решений.

Архитектура:
  DecisionEngine — синглтон, который решает:
    - ВХОДИТЬ ли в позицию (ML-ансамбль: 4 голоса + multi-timeframe)
    - ВЫХОДИТЬ из позиции (SL, TP, трейл, детектор разворота)
  
  Все остальные модули (trader, monitor) только собирают данные.
  Решение принимается в одном месте → выполняется в trader.

Голоса на вход (DecisionEngine._evaluate_entry_ensemble):
  1. MLProfessionalV2 (LightGBM, 27/16 признаков, 5M+1H+4H)  — 35%
  2. MLAdvisor (RandomForest, 9 признаков, паттерны+VWAP+D1) — 20%
  3. HMM Режим + Cогласованность трендов (5M/1H/4H)         — 25%
  4. RSI + Объём + BTC-корреляция                            — 20%
  ──────────────────────────────────────────────────────────
  Порог входа: 65 (адаптивный: CALM=57, NORMAL=65, VOLATILE=70)
  Жёсткое вето: 1H или 4H bearish + BTC падает → блокировка

Приоритеты на выход: SL > TP > трейлинг > 48ч таймаут
"""

import sys
import os
import json
import logging
import time
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

logger = logging.getLogger("decision_engine")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Сигнальные типы ────────────────────────────────────────────────────────

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    HOLD = "HOLD"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


# ─── Решение ────────────────────────────────────────────────────────────────

class Decision:
    """
    Единый объект решения.

    ⚡ ЗАЛОЖЕНО ПОД ШОРТ: поле side='short'.
    ⚡ ЗАЛОЖЕНО ПОД ДИНАМИЧЕСКИЙ РИСК: sl_price, tp_price, trail_act, trail_dist.

      Атрибуты:
      symbol        — тикер (XRP/USDT)
      action        — 'enter' | 'hold' | 'exit'
      side          — 'long' | 'short'
      score         — итоговый скор 0-100 (enter) или None
      position_size — доля от max 2% капитала (0.0-1.0)
      tp_levels     — уровни частичного тейка [(price,pct), ...] или None
      sl_price      — цена стоп-лосса (None = стандартный из конфига)
      tp_price      — цена тейк-профита (None = стандартный)
      trail_act     — % активации трейлинга (None = стандартный)
      trail_dist    — % дистанции трейлинга от пика (None = стандартный)
      max_hold_h    — макс. часов удержания (None = 48)
      reason        — причина решения
      signal_type   — тип сигнала (SignalType)
      priority      — приоритет (50 по умолч.)
      metadata      — полная раскладка голосов
    """
    def __init__(self, symbol: str, action: str = 'hold',
                 side: str = 'long',
                 score: Optional[float] = None,
                 position_size: Optional[float] = None,
                 tp_levels: Optional[List[Tuple[float, float]]] = None,
                 sl_price: Optional[float] = None,
                 tp_price: Optional[float] = None,
                 trail_act: Optional[float] = None,
                 trail_dist: Optional[float] = None,
                 max_hold_h: Optional[float] = None,
                 reason: str = '',
                 signal_type: Optional[SignalType] = None,
                 priority: int = 50,
                 metadata: Optional[Dict] = None):
        self.symbol = symbol
        self.action = action
        self.side = side
        self.score = score
        self.position_size = position_size
        self.tp_levels = tp_levels
        self.sl_price = sl_price
        self.tp_price = tp_price
        self.trail_act = trail_act
        self.trail_dist = trail_dist
        self.max_hold_h = max_hold_h
        self.reason = reason
        self.signal_type = signal_type
        self.priority = priority
        self.metadata = metadata or {}

    @property
    def is_long(self) -> bool:
        return self.side == 'long'

    @property
    def is_short(self) -> bool:
        return self.side == 'short'
    
    def __repr__(self) -> str:
        if self.action == 'hold':
            return f"[HOLD] {self.symbol}: {self.reason}"
        tag = self.signal_type.name if self.signal_type else self.action.upper()
        s = f"({self.side.upper()})" if self.side else ""
        sz = f", size={self.position_size:.2f}" if self.position_size else ""
        tp = f", tp={self.tp_levels}" if self.tp_levels else ""
        return (f"[{tag}] {self.symbol}: {self.reason}"
                f" (score={self.score}{sz}{tp})")


# ─── РЕКОМЕНДУЕМЫЕ ПАРЫ ────────────────────────────────────────────────────

RECOMMENDED_PAIRS = [
    "BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT",
    "SOL/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT", "AVAX/USDT",
    "UNI/USDT", "ATOM/USDT", "ARB/USDT", "OP/USDT", "APT/USDT",
    "NEAR/USDT", "FIL/USDT", "ALGO/USDT", "EGLD/USDT", "FTM/USDT",
    "SAND/USDT", "MANA/USDT",
]


# ═══════════════════════════════════════════════════════════════════════════
# DecisionEngine
# ═══════════════════════════════════════════════════════════════════════════

class DecisionEngine:
    """
    Единый центр принятия решений.
    
    Вход: ML-ансамбль (MLProfessionalV2 + MLAdvisor + RSI + HMM)
    Выход: SL / TP / трейлинг / таймаут
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
            with open(config_path) as f:
                self.config = json.load(f)
        
        # HMM-режим (кэшируется, обновляется из монитора)
        self.hmm_regime = 1  # NORMAL по умолчанию
        
        # Делегаты для внешних функций
        self.quick_struct_exit_fn: Optional[Callable] = None
        self.ml_exit_check_fn: Optional[Callable] = None
        self.confirm_15m_reversal_fn: Optional[Callable] = None
        
        # Комиссия биржи
        self.fee_rate = 0.001  # 0.1%
        
        # Защита от повторных решений
        self._last_decisions: Dict[str, Dict] = {}
        
        # Тайм-аут на повторный вход после продажи (сек)
        self.reentry_cooldown = 3600  # 1 час
        
        # Инициализация ML-модулей (ленивая)
        self._ml_pro_v2 = None
        self._ml_advisor = None
        self._liquidity = None
        
        # Кэшированные данные multi-timeframe
        self._candles_1h: Optional[list] = None
        self._candles_4h: Optional[list] = None
        self._btc_price: Optional[float] = None
        self._candle_cache_time = 0
        self._candle_cache_ttl = 120  # обновлять раз в 2 мин
        
        # Базовая цена BTC для корреляции (загружается при первом вызове)
        self._btc_reference_price: Optional[float] = None
        self._btc_reference_time = 0
        
        # Статистика
        self.stats = {
            'total_decisions': 0,
            'exit_decisions': 0,
            'entry_decisions': 0,
            'ml_v2_skips': 0,
            'ml_v2_weak': 0,
            'ml_v2_buys': 0,
            'ml_v2_errors': 0,
            'advisor_skips': 0,
            'advisor_weak': 0,
            'advisor_goods': 0,
            'veto_btc_drop': 0,
            'veto_mtf_conflict': 0,
            'liquidity_scores': 0,
        }
    
    def _lazy_init_ml(self):
        """Ленивая инициализация ML-модулей."""
        if self._ml_pro_v2 is None:
            try:
                from ml_professional_v2 import MLProfessionalV2
                self._ml_pro_v2 = MLProfessionalV2()
                logger.info(f"[DE] MLProfessionalV2 {'готов' if self._ml_pro_v2.trained else 'ожидает модели'}")
            except Exception as e:
                logger.warning(f"[DE] MLProfessionalV2 не загрузился: {e}")
                self._ml_pro_v2 = None
        
        if self._ml_advisor is None:
            try:
                from ml_advisor import get_advisor
                self._ml_advisor = get_advisor()
                logger.info(f"[DE] MLAdvisor {'готов' if self._ml_advisor.is_trained else 'ожидает обучения'}")
            except Exception as e:
                logger.warning(f"[DE] MLAdvisor не загрузился: {e}")
                self._ml_advisor = None
        
        if self._liquidity is None:
            try:
                from liquidity_cluster import get_liquidity_cluster
                self._liquidity = get_liquidity_cluster()
                logger.info("[DE] LiquidityCluster готов")
            except Exception as e:
                logger.warning(f"[DE] LiquidityCluster не загрузился: {e}")
                self._liquidity = None
    
    def set_multi_tf_data(self, candles_1h: Optional[list] = None,
                          candles_4h: Optional[list] = None,
                          btc_price: Optional[float] = None) -> None:
        """Подать multi-timeframe данные для анализа (из трейдера)."""
        if candles_1h is not None:
            self._candles_1h = candles_1h
        if candles_4h is not None:
            self._candles_4h = candles_4h
        if btc_price is not None:
            self._btc_price = btc_price
        self._candle_cache_time = time.time()

    # ─── Адаптивные SL/TP от HMM-режима ────────────────────────────────────
    # ⚡ ЗАЛОЖЕНО ПОД ШОРТ: side='short' инвертирует уровни

    _SL_TP_BY_REGIME = {
        0: {'sl': 3.0, 'tp': 8.0,   'trail_act': 1.2, 'trail_dist': 0.8},  # CALM
        1: {'sl': 5.0, 'tp': 10.0,  'trail_act': 1.5, 'trail_dist': 1.0},  # NORMAL
        2: {'sl': 8.0, 'tp': 14.0,  'trail_act': 2.0, 'trail_dist': 1.5},  # VOLATILE
    }

    def get_sl_tp_params(self, side: str = 'long') -> dict:
        """
        Вернуть параметры SL/TP/трейлинга под текущий HMM-режим.

        ⚡ ЗАЛОЖЕНО ПОД ШОРТ: для short SL/TP симметричны (SL = рост цены,
        TP = падение), но дистанции те же в %.
        """
        p = self._SL_TP_BY_REGIME.get(self.hmm_regime, self._SL_TP_BY_REGIME[1])
        return {
            'sl_pct': p['sl'],
            'tp_pct': p['tp'],
            'trail_act': p['trail_act'],
            'trail_dist': p['trail_dist'],
            'max_hold_h': 48.0,
            'side': side,
        }
    
    def _get_clean_data(self, symbol: str, is_reference: bool = False) -> Optional[list]:
        """
        Получить чистые свечи для symbol.
        Если не закэшированы — вернуть None (трейдер обновляет раз в N циклов).
        """
        # Для BTC используем set_multi_tf_data
        if symbol == 'BTC/USDT':
            return None
        return self._candles_1h  # fallback на общие данные
    
    def update_hmm_regime(self, regime: int) -> None:
        """Обновить HMM-режим рынка (вызывается из монитора)."""
        self.hmm_regime = regime
        logger.debug(f"[DE] HMM: {regime}")
    
    # ─── ВЫХОД ИЗ ПОЗИЦИИ ──────────────────────────────────────────────────
    
    def decide_exit(self, symbol: str, entry_price: float, current_price: float,
                    highest_price: float, lowest_price: float,
                    entry_time: datetime, pnl_pct: float,
                    sl_pct: float, tp_pct: float, trail_act: float,
                    trail_dist: float, max_hold_hours: float = 48,
                    side: str = 'long') -> Optional[Decision]:
        """
        Принять решение о выходе.
        Приоритет: SL > TP > трейлинг > таймаут.
        
        ⚡ ЗАЛОЖЕНО ПОД ШОРТ: side='short' инвертирует логику SL/TP/трейлинга.
        
        Для long:
          SL = цена падает на N% от входа
          TP = цена растёт на N% от входа
          Трейлинг = цена была выше входа на trail_act%, потом упала на trail_dist%
        
        Для short:
          SL = цена растёт на N% от входа (движение против шорта)
          TP = цена падает на N% от входа
          Трейлинг = цена была ниже входа на trail_act%, потом выросла на trail_dist%
        """
        self.stats['total_decisions'] += 1
        is_short = (side == 'short')
        
        # ─── 1. SL — безусловный стоп-лосс ────────────────────────────────
        # Long: PnL <= -sl_pct (цена упала). Short: PnL <= -sl_pct (цена выросла против шорта)
        if pnl_pct <= -sl_pct:
            self.stats['exit_decisions'] += 1
            return Decision(symbol, 'exit', side=side,
                           reason=f"SL -{pnl_pct:.2f}% (лимит -{sl_pct}%)",
                           signal_type=SignalType.STRONG_SELL, priority=100)
        
        # ─── 2. TP — тейк-профит ──────────────────────────────────────────
        # Long: PnL >= tp_pct (цена выросла). Short: PnL >= tp_pct (цена упала в пользу)
        if pnl_pct >= tp_pct:
            self.stats['exit_decisions'] += 1
            return Decision(symbol, 'exit', side=side,
                           reason=f"TP +{pnl_pct:.2f}% (лимит +{tp_pct}%)",
                           signal_type=SignalType.STRONG_SELL, priority=90)
        
        # ─── 3. Трейлинг-стоп ─────────────────────────────────────────────
        # ⚡ ЗАЛОЖЕНО ПОД ШОРТ: для short трейлинг срабатывает при росте цены после падения
        if is_short:
            # Short: lowest_price (минимум от входа), трейлинг = цена выросла от минимума
            trail_active_pct = (entry_price - lowest_price) / entry_price * 100
            if trail_active_pct >= trail_act:
                trail_level = lowest_price * (1 + trail_dist / 100.0)
                if current_price >= trail_level:
                    self.stats['exit_decisions'] += 1
                    return Decision(
                        symbol, 'exit', side=side, priority=80,
                        reason=(f"Short-трейлинг: L={lowest_price:.4f}→{current_price:.4f}"
                                f" (активация: {trail_act}%, дист: {trail_dist}%)"),
                        signal_type=SignalType.SELL
                    )
        else:
            # Long: highest_price (максимум от входа), трейлинг = цена упала от максимума
            trail_active_pct = (highest_price - entry_price) / entry_price * 100
            if trail_active_pct >= trail_act:
                trail_level = highest_price * (1 - trail_dist / 100.0)
                if current_price <= trail_level:
                    self.stats['exit_decisions'] += 1
                    return Decision(
                        symbol, 'exit', side=side, priority=80,
                        reason=(f"Long-трейлинг: H={highest_price:.4f}→{current_price:.4f}"
                                f" (активация: {trail_act}%, дист: {trail_dist}%)"),
                        signal_type=SignalType.SELL
                    )
        
        # ─── 4. Таймаут удержания (48 часов) ──────────────────────────────
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        hold_hours = (now - entry_time).total_seconds() / 3600
        if hold_hours >= max_hold_hours:
            self.stats['exit_decisions'] += 1
            return Decision(
                symbol, 'exit', side=side, priority=70,
                reason=f"Таймаут {hold_hours:.1f}ч > {max_hold_hours}ч",
                signal_type=SignalType.WEAK_SELL
            )
        
        return None  # держим
    
    def _check_trailing_stop(self, symbol: str, entry_price: float,
                              current_price: float, highest_price: float,
                              pnl_pct: float, trail_act: float,
                              trail_dist: float) -> Optional[str]:
        """Проверить трейлинг-стоп (вспомогательный метод)."""
        trail_active_pct = (highest_price - entry_price) / entry_price * 100
        if trail_active_pct >= trail_act:
            trail_level = highest_price * (1 - trail_dist / 100.0)
            if current_price <= trail_level:
                return (f"Активирован: H={highest_price:.4f}→{current_price:.4f}"
                        f" (активация: {trail_act}%, дист: {trail_dist}%)")
        return None
    
    # ─── РЕШЕНИЕ НА ВХОД ───────────────────────────────────────────────────
    
    def decide_entry(self, symbol: str, confidence: float, trend: str,
                     rsi: float, current_price: float,
                     current_positions_count: int, max_positions: int = 5,
                     candles_5m: Optional[list] = None,
                     candles_1h: Optional[list] = None,
                     candles_4h: Optional[list] = None,
                     side: str = 'long') -> Decision:
        """
        Принять решение о входе в позицию.
        
        ⚡ ЗАЛОЖЕНО ПОД ШОРТ: side='short' инвертирует логику.
        
        AN SAMBЛЬ:
          1. MLProfessionalV2 (LightGBM, 27/16 признаков, 5M+1H+4H) — 35%
          2. MLAdvisor (RandomForest, 9 признаков)                  — 20%
          3. HMM + Мульти-таймфрейм согласованность (5M/1H/4H)     — 25%
          4. RSI + Объём + BTC-корреляция                           — 20%
        
        Vetо:
          - 1H или 4H bearish → блокировка (для long)
          - 1H или 4H bullish → блокировка (для short)
          - BTC падает >1.5% за 4H → блокировка
          - Максимум N позиций
          - Кулдаун повторного входа
        """
        self.stats['total_decisions'] += 1
        is_short = (side == 'short')
        self._lazy_init_ml()
        
        # Используем переданные свечи или закэшированные
        c1h = candles_1h if candles_1h is not None else self._candles_1h
        c4h = candles_4h if candles_4h is not None else self._candles_4h
        btc_p = self._btc_price
        
        now = time.time()
        
        # ═══ ЖЁСТКИЕ БЛОКИРОВКИ ═══════════════════════════════════════════
        
        # 1. Лимит позиций
        if current_positions_count >= max_positions:
            return Decision(symbol, 'hold', side=side, reason=f"Максимум {max_positions} позиций")
        
        # 2. Кулдаун повторного входа
        if symbol in self._last_decisions:
            last_exit = self._last_decisions[symbol]
            time_since = now - last_exit.get('exit_time', 0)
            if time_since < self.reentry_cooldown:
                remain = self.reentry_cooldown - time_since
                return Decision(
                    symbol, 'hold', side=side,
                    reason=f"Повторный вход через {remain/60:.1f} мин"
                )
        
        # 3. BTC-корреляция: не входить если BTC падает
        if btc_p is not None and self._btc_reference_price is not None:
            btc_change_4h = (btc_p - self._btc_reference_price) / self._btc_reference_price * 100
            if btc_change_4h < -1.5:
                self.stats['veto_btc_drop'] += 1
                return Decision(symbol, 'hold', side=side,
                               reason=f"VETO: BTC упал на {btc_change_4h:.1f}% за период")
        
        # 4. Multi-timeframe veto: не входить против старшего ТФ
        # ⚡ ЗАЛОЖЕНО ПОД ШОРТ: для short — блокировка при bullish
        veto_trends_long = ('strong_bearish',)      # Long: не входить на медвежьем ТФ
        veto_trends_short = ('strong_bullish',)     # Short: не входить на бычьем ТФ
        veto_trends = veto_trends_short if is_short else veto_trends_long
        veto_label = 'bullish' if is_short else 'bearish'
        
        if c1h is not None and len(c1h) > 0:
            try:
                trend_1h = self._calc_trend_from_candles(c1h)
                if trend_1h in veto_trends:
                    self.stats['veto_mtf_conflict'] += 1
                    return Decision(symbol, 'hold', side=side,
                                   reason=f"VETO: 1H тренд {veto_label}")
            except:
                pass
        
        if c4h is not None and len(c4h) > 0:
            try:
                trend_4h = self._calc_trend_from_candles(c4h)
                if trend_4h in veto_trends:
                    self.stats['veto_mtf_conflict'] += 1
                    return Decision(symbol, 'hold', side=side,
                                   reason=f"VETO: 4H тренд {veto_label}")
            except:
                pass
        
        # ═══ АНСАМБЛЬ ГОЛОСОВ ═══════════════════════════════════════════════
        
        entry_checks = self._evaluate_entry_ensemble(
            symbol, confidence, trend, rsi, current_price,
            candles_5m=candles_5m, candles_1h=c1h, candles_4h=c4h,
            side=side
        )
        
        if entry_checks['approved']:
            self.stats['entry_decisions'] += 1
            
            # Position size: пропорционально скору
            score = entry_checks.get('final_score', 65)
            # 0.5 при 65, 1.0 при 90+
            position_size = min(1.0, (score - 50) / 45)  # 65→0.33, 80→0.67, 95→1.0
            position_size = max(0.3, min(1.0, position_size))
            
            # SL/TP от HMM-режима
            rp = self.get_sl_tp_params(side)
            
            return Decision(
                symbol, 'enter', side=side,
                score=score,
                position_size=position_size,
                reason=entry_checks['reason'],
                priority=50,
                sl_price=current_price * (1 - rp['sl_pct'] / 100) if side == 'long' else current_price * (1 + rp['sl_pct'] / 100),
                tp_price=current_price * (1 + rp['tp_pct'] / 100) if side == 'long' else current_price * (1 - rp['tp_pct'] / 100),
                trail_act=rp['trail_act'],
                trail_dist=rp['trail_dist'],
                max_hold_h=rp['max_hold_h'],
                metadata=entry_checks
            )
        else:
            return Decision(
                symbol, 'hold', side=side,
                reason=entry_checks['reason']
            )
    
    def _calc_trend_from_candles(self, candles: list) -> str:
        """Определить тренд по списку свечей (быстро по EMAs).
        Поддерживает список списков [ts,o,h,l,c,v] и список словарей {o,h,l,c,v,t}."""
        if not candles or len(candles) < 20:
            return 'neutral'
        
        # Определяем формат: список или словарь
        first = candles[0]
        if isinstance(first, dict):
            closes = [c['c'] for c in candles[-20:]]
        else:
            closes = [c[4] for c in candles[-20:]]
        
        # EMA 7 vs EMA 20
        ema7 = sum(closes[-7:]) / 7
        ema20 = sum(closes[-20:]) / 20
        
        # Процентное изменение
        change = (closes[-1] - closes[-20]) / closes[-20] * 100
        
        if ema7 > ema20 and change > 2:
            return 'strong_bullish'
        elif ema7 > ema20:
            return 'bullish'
        elif ema7 < ema20 and change < -2:
            return 'strong_bearish'
        elif ema7 < ema20:
            return 'bearish'
        return 'neutral'
    
    # ═══════════════════════════════════════════════════════════════════════
    # АНСАМБЛЬ ГОЛОСОВ
    # ═══════════════════════════════════════════════════════════════════════
    
    _ENTRY_THRESHOLD = 65.0
    
    def _get_entry_threshold(self) -> float:
        """Адаптивный порог входа от HMM-режима."""
        base = self._ENTRY_THRESHOLD
        if self.hmm_regime == 0:   # CALM
            return base - 8        # 57
        elif self.hmm_regime == 2: # VOLATILE
            return base + 5        # 70
        return base                # NORMAL = 65
    
    def _evaluate_entry_ensemble(self, symbol: str, confidence: float,
                                   trend: str, rsi: float,
                                   current_price: float,
                                   candles_5m: Optional[list] = None,
                                   candles_1h: Optional[list] = None,
                                   candles_4h: Optional[list] = None,
                                   side: str = 'long') -> Dict:
        """
        Профессиональный ансамбль из 4 голосов.
        
        ⚡ ЗАЛОЖЕНО ПОД ШОРТ: side='short' инвертирует RSI, тренды, BTC-корреляцию.
        ML-модели пока только для long (заглушка для short).
        
        Веса:
          1. MLProfessionalV2 (LightGBM, 27/16 признаков, multi-TF) — 35%
          2. MLAdvisor (RandomForest, 9 признаков, паттерны+VWAP+D1) — 20%
          3. Согласованность трендов 5M/1H/4H + HMM — 25%
          4. RSI + Объём + BTC-корреляция — 20%
        
        Возвращает словарь с approval, score, раскладкой.
        """
        self._lazy_init_ml()
        is_short = (side == 'short')
        
        # Инициализируем результаты голосов
        scores = {
            'ml_v2': {'score': 50, 'weight': 0.30, 'detail': 'N/A'},
            'advisor': {'score': 50, 'weight': 0.15, 'detail': 'N/A'},
            'mtf': {'score': 50, 'weight': 0.25, 'detail': 'N/A'},
            'rsi_vol_btc': {'score': 50, 'weight': 0.15, 'detail': 'N/A'},
            'liquidity': {'score': 50, 'weight': 0.15, 'detail': 'N/A'},
        }
        
        # ═══ ГОЛОС 1: MLProfessionalV2 (35%) ═══════════════════════════════
        try:
            if self._ml_pro_v2 and self._ml_pro_v2.trained:
                ml_decision, ml_prob, ml_feat = self._ml_pro_v2.evaluate(
                    symbol, candles_5m or [], candles_1h or [],
                    candles_4h or [], confidence, trend, rsi
                )
                if ml_decision == 'BUY':
                    ml_score = min(100, confidence + 20)  # усиление
                    scores['ml_v2']['score'] = ml_score
                    scores['ml_v2']['detail'] = f"BUY(prob={ml_prob:.2f})"
                    self.stats['ml_v2_buys'] += 1
                elif ml_decision == 'WEAK_BUY':
                    ml_score = 50 + (ml_prob * 30) if isinstance(ml_prob, (int, float)) else 50
                    scores['ml_v2']['score'] = ml_score
                    scores['ml_v2']['detail'] = f"WEAK(prob={ml_prob:.2f})"
                    self.stats['ml_v2_weak'] += 1
                else:
                    scores['ml_v2']['score'] = 20  # SKIP — штраф
                    scores['ml_v2']['detail'] = f"SKIP(prob={ml_prob:.2f})"
                    self.stats['ml_v2_skips'] += 1
            else:
                # ML не обучен — доверяем confidence
                scores['ml_v2']['score'] = min(max(confidence, 0), 100)
                scores['ml_v2']['detail'] = f"confidence({confidence:.0f})"
        except Exception as e:
            scores['ml_v2']['score'] = 50  # нейтрально при ошибке
            scores['ml_v2']['detail'] = f"error({e})"
            self.stats['ml_v2_errors'] += 1
        
        # ═══ ГОЛОС 2: MLAdvisor (20%) ═══════════════════════════════════════
        try:
            if self._ml_advisor and self._ml_advisor.is_trained:
                # Конвертируем список свечей в pandas DataFrame для MLAdvisor
                adv_df = None
                if candles_5m and len(candles_5m) > 10:
                    try:
                        import pandas as _pd
                        adv_df = _pd.DataFrame(candles_5m)
                    except:
                        adv_df = None
                adv = self._ml_advisor.evaluate(
                    symbol, current_price, rsi, trend, confidence,
                    df=adv_df
                )
                if adv['decision'] == 'GOOD':
                    adv_score = 70 + (adv['confidence'] * 30)
                    scores['advisor']['score'] = min(100, adv_score)
                    scores['advisor']['detail'] = f"GOOD({adv['confidence']:.2f})"
                    self.stats['advisor_goods'] += 1
                elif adv['decision'] == 'WEAK':
                    adv_score = 30 + (adv['confidence'] * 40)
                    scores['advisor']['score'] = min(70, adv_score)
                    scores['advisor']['detail'] = f"WEAK({adv['confidence']:.2f})"
                    self.stats['advisor_weak'] += 1
                else:
                    scores['advisor']['score'] = 15
                    scores['advisor']['detail'] = f"SKIP({adv['confidence']:.2f})"
                    self.stats['advisor_skips'] += 1
            else:
                # MLAdvisor не обучен — нейтральный голос
                scores['advisor']['score'] = 50
                scores['advisor']['detail'] = "not_trained"
        except Exception as e:
            scores['advisor']['score'] = 50
            scores['advisor']['detail'] = f"error({e})"
        
        # ═══ ГОЛОС 3: Multi-timeframe согласованность + HMM (25%) ═════════
        try:
            # 3a. Определяем тренды на 5M, 1H, 4H
            trend_5m_val = self._trend_to_score(trend)  # 0-100
            
            trend_1h_val = 50
            if candles_1h is not None and len(candles_1h) > 0:
                t1h = self._calc_trend_from_candles(candles_1h)
                trend_1h_val = self._trend_to_score(t1h)
            
            trend_4h_val = 50
            if candles_4h is not None and len(candles_4h) > 0:
                t4h = self._calc_trend_from_candles(candles_4h)
                trend_4h_val = self._trend_to_score(t4h)
            
            # Согласованность: штраф за конфликт между таймфреймами
            divergence = max(
                abs(trend_5m_val - trend_1h_val),
                abs(trend_5m_val - trend_4h_val),
                abs(trend_1h_val - trend_4h_val)
            )
            # divergence 0-100 → penalty 0-30
            penalty = min(30, divergence * 0.3)
            
            # 3b. HMM-режим
            hmm_score = 80
            if self.hmm_regime == 0:   # CALM
                hmm_score = 50
            elif self.hmm_regime == 2: # VOLATILE
                hmm_score = 40
            
            # Среднее по всем трендам без штрафа = (5m + 1h + 4h) / 3
            mtf_base = (trend_5m_val + trend_1h_val + trend_4h_val) / 3
            # Итог: база - штраф за конфликт, скорректированная HMM
            mtf_score = max(10, mtf_base + (hmm_score - 50) * 0.3 - penalty)
            
            scores['mtf']['score'] = mtf_score
            scores['mtf']['detail'] = (
                f"5m={trend_5m_val:.0f} 1h={trend_1h_val:.0f} 4h={trend_4h_val:.0f}"
                f" hmm={self.hmm_regime} pen={penalty:.1f}"
            )
        except Exception as e:
            scores['mtf']['score'] = 50
            scores['mtf']['detail'] = f"error({e})"
        
        # ═══ ГОЛОС 4: RSI + Объём + BTC-корреляция (20%) ═════════════════
        # ⚡ ЗАЛОЖЕНО ПОД ШОРТ: для short RSI- и BTC-голоса инвертированы
        try:
            if is_short:
                # Short: низкий RSI = плохо (цена дешёвая, могут откупить),
                # высокий RSI = хорошо (цена дорогая, пора шортить)
                if 40 <= rsi <= 60:
                    rsi_part = 100
                elif 30 <= rsi <= 70:
                    rsi_part = 70
                elif rsi > 85 or rsi < 15:
                    rsi_part = 15  # экстремум против шорта
                elif rsi > 80:
                    rsi_part = 80  # перекупленность = шорт
                elif rsi < 20:
                    rsi_part = 20  # перепроданность = не шорт
                elif rsi > 70:
                    rsi_part = 65
                elif rsi < 30:
                    rsi_part = 30
                else:
                    rsi_part = 50
            else:
                # Long: оригинальная логика
                if 40 <= rsi <= 60:
                    rsi_part = 100
                elif 30 <= rsi <= 70:
                    rsi_part = 70
                elif rsi > 85 or rsi < 15:
                    rsi_part = 15
                elif rsi > 75:
                    rsi_part = 30
                elif rsi < 25:
                    rsi_part = 25
                else:
                    rsi_part = 40
            
            # BTC-корреляция (25%) — для short инвертирована
            btc_part = 50
            if self._btc_price and self._btc_reference_price:
                btc_change = (self._btc_price - self._btc_reference_price) / self._btc_reference_price * 100
                if btc_change > 0.5:
                    btc_part = 80 if not is_short else 20  # BTC растёт → хорошо для long, плохо для short
                elif btc_change < -0.5:
                    btc_part = 20 if not is_short else 80  # BTC падает → плохо для long, хорошо для short
                else:
                    btc_part = 55
            
            # Объём (20%) — пока нейтрально
            vol_part = 50
            
            rsi_vol_btc_score = rsi_part * 0.55 + btc_part * 0.25 + vol_part * 0.20
            
            scores['rsi_vol_btc']['score'] = rsi_vol_btc_score
            scores['rsi_vol_btc']['detail'] = (
                f"rsi={rsi_part:.0f} btc={btc_part:.0f} vol={vol_part:.0f}"
            )
        except Exception as e:
            scores['rsi_vol_btc']['score'] = 50
            scores['rsi_vol_btc']['detail'] = f"error({e})"
        
        # ═══ ГОЛОС 5: LiquidityCluster (15%) ════════════════════════════════
        try:
            if self._liquidity:
                liq = self._liquidity.evaluate(candles_5m, current_price,
                                                candles_1h, candles_4h)
                scores['liquidity']['score'] = liq['score']
                scores['liquidity']['detail'] = liq['detail']
                self.stats['liquidity_scores'] += 1
            else:
                scores['liquidity']['score'] = 50
                scores['liquidity']['detail'] = 'not_loaded'
        except Exception as e:
            scores['liquidity']['score'] = 50
            scores['liquidity']['detail'] = f"error({e})"
        
        # ═══ ИТОГОВЫЙ СКОР ═════════════════════════════════════════════════
        final_score = sum(v['score'] * v['weight'] for v in scores.values())
        
        threshold = self._get_entry_threshold()
        
        # Разбивка голосов для логов/дашборда
        votes_str = (
            f"ML-Pro:{scores['ml_v2']['score']:.0f}({scores['ml_v2']['detail']}) "
            f"Adv:{scores['advisor']['score']:.0f}({scores['advisor']['detail']}) "
            f"MTF:{scores['mtf']['score']:.0f}({scores['mtf']['detail']}) "
            f"RVB:{scores['rsi_vol_btc']['score']:.0f}({scores['rsi_vol_btc']['detail']}) "
            f"Liq:{scores['liquidity']['score']:.0f}({scores['liquidity']['detail']})"
        )
        
        votes = {
            'ml_v2': {'score': scores['ml_v2']['score'], 'detail': scores['ml_v2']['detail']},
            'advisor': {'score': scores['advisor']['score'], 'detail': scores['advisor']['detail']},
            'mtf': {'score': scores['mtf']['score'], 'detail': scores['mtf']['detail']},
            'rsi_vol_btc': {'score': scores['rsi_vol_btc']['score'], 'detail': scores['rsi_vol_btc']['detail']},
            'liquidity': {'score': scores['liquidity']['score'], 'detail': scores['liquidity']['detail']},
        }
        
        if final_score >= threshold:
            return {
                'approved': True,
                'reason': f"ВХОД {votes_str}",
                'final_score': final_score,
                'threshold': threshold,
                'price': current_price,
                'ml_v2_score': scores['ml_v2']['score'],
                'advisor_score': scores['advisor']['score'],
                'mtf_score': scores['mtf']['score'],
                'rsi_vol_btc_score': scores['rsi_vol_btc']['score'],
                'liquidity_score': scores['liquidity']['score'],
                'votes': votes,
            }
        else:
            return {
                'approved': False,
                'reason': f"⏭️ Score={final_score:.0f} < {threshold} {votes_str}",
                'final_score': final_score,
                'threshold': threshold,
                'votes': votes,
            }
    
    def _trend_to_score(self, trend: str) -> float:
        """Преобразовать тренд в скор 0-100."""
        mapping = {
            'strong_bullish': 90,
            'bullish': 75,
            'weak_bullish': 60,
            'neutral': 50,
            'weak_bearish': 35,
            'bearish': 20,
            'strong_bearish': 5,
        }
        return mapping.get(trend, 50)
    
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
    
    def update_btc_reference(self, btc_price: float = None) -> None:
        """Обновить референсную цену BTC для расчёта изменения."""
        if btc_price is not None:
            if self._btc_reference_price is None:
                self._btc_reference_price = btc_price
                self._btc_reference_time = time.time()
            elif time.time() - self._btc_reference_time > 14400:  # каждые 4 часа
                self._btc_reference_price = btc_price
                self._btc_reference_time = time.time()

    def get_stats(self) -> Dict:
        """Вернуть статистику решений."""
        ml_v2_status = "not_loaded"
        adv_status = "not_loaded"
        if self._ml_pro_v2:
            ml_v2_status = f"trained({self._ml_pro_v2.is_27f and '27f' or '16f'})" if self._ml_pro_v2.trained else "pending"
        if self._ml_advisor:
            adv_status = "trained" if self._ml_advisor.is_trained else "pending"

        return {
            **self.stats,
            'ml_v2': ml_v2_status,
            'ml_advisor': adv_status,
            'hmm_regime': self.hmm_regime,
            'regime_name': ['CALM', 'NORMAL', 'VOLATILE'][self.hmm_regime] if self.hmm_regime in (0,1,2) else 'UNKNOWN',
            'active_cooldowns': len(self._last_decisions),
            'entry_threshold': self._get_entry_threshold(),
        }

    def reset_cooldowns(self) -> None:
        """Сбросить все тайм-ауты."""
        self._last_decisions.clear()
        logger.info("🔄 Кулдауны сброшены")
