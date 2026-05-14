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
  1. MLProfessionalV2 (LightGBM, 27/16 признаков, 5M+1H+4H)  — 20%
  2. MLAdvisor (RandomForest, 9 признаков, паттерны+VWAP+D1) — 10%
  3. HMM Режим + Cогласованность трендов (5M/1H/4H)         — 25%
  4. LiquidityCluster v2 (Order Flow/Block/Sweep)            — 25%
  5. Volume/VWAP (VWAP реверсия + Volume Spike)              — 20%
  ──────────────────────────────────────────────────────────
  RSI/Vol/BTC — 0% (отключён, зарезервирован)
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
        self._volume_vwap = None
        
        # Кэшированные данные multi-timeframe
        self._candles_1h: Optional[list] = None
        self._candles_4h: Optional[list] = None
        self._btc_price: Optional[float] = None
        self._candle_cache_time = 0
        self._candle_cache_ttl = 120  # обновлять раз в 2 мин
        
        # BTC Direction Predictor
        self._btc_predictor = None  # Ленивая инициализация
        
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
        """Ленивая инициализация ML-модулей с reload для горячей замены."""
        if self._ml_pro_v2 is None:
            try:
                import importlib
                import ml_professional_v2
                importlib.reload(ml_professional_v2)
                self._ml_pro_v2 = ml_professional_v2.MLProfessionalV2()
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
        
        if self._volume_vwap is None:
            try:
                import volume_vwap
                self._volume_vwap = volume_vwap
                logger.info("[DE] Volume/VWAP модуль готов")
            except Exception as e:
                logger.warning(f"[DE] Volume/VWAP не загрузился: {e}")
                self._volume_vwap = None
    
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
        0: {'sl': 3.0, 'tp': 8.0,   'trail_act': 2.0, 'trail_dist': 1.2},  # CALM
        1: {'sl': 5.0, 'tp': 10.0,  'trail_act': 2.5, 'trail_dist': 1.5},  # NORMAL
        2: {'sl': 8.0, 'tp': 14.0,  'trail_act': 3.0, 'trail_dist': 2.0},  # VOLATILE
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
            self._save_veto_vote(symbol, current_price, f"Максимум {max_positions} позиций", side)
            return Decision(symbol, 'hold', side=side, reason=f"Максимум {max_positions} позиций")
        
        # 2. Кулдаун повторного входа
        if symbol in self._last_decisions:
            last_exit = self._last_decisions[symbol]
            time_since = now - last_exit.get('exit_time', 0)
            if time_since < self.reentry_cooldown:
                remain = self.reentry_cooldown - time_since
                reason = f"Повторный вход через {remain/60:.1f} мин"
                self._save_veto_vote(symbol, current_price, reason, side)
                return Decision(
                    symbol, 'hold', side=side,
                    reason=reason
                )
        
        # 3. BTC-корреляция: не входить если BTC падает
        if btc_p is not None and self._btc_reference_price is not None:
            btc_change_4h = (btc_p - self._btc_reference_price) / self._btc_reference_price * 100
            if btc_change_4h < -1.5:
                reason = f"VETO: BTC упал на {btc_change_4h:.1f}% за период"
                self.stats['veto_btc_drop'] += 1
                self._save_veto_vote(symbol, current_price, reason, side)
                return Decision(symbol, 'hold', side=side, reason=reason)
        
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
                    reason = f"VETO: 1H тренд {veto_label}"
                    self.stats['veto_mtf_conflict'] += 1
                    self._save_veto_vote(symbol, current_price, reason, side)
                    return Decision(symbol, 'hold', side=side, reason=reason)
            except:
                pass
        
        if c4h is not None and len(c4h) > 0:
            try:
                trend_4h = self._calc_trend_from_candles(c4h)
                if trend_4h in veto_trends:
                    reason = f"VETO: 4H тренд {veto_label}"
                    self.stats['veto_mtf_conflict'] += 1
                    self._save_veto_vote(symbol, current_price, reason, side)
                    return Decision(symbol, 'hold', side=side, reason=reason)
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
            position_size = min(1.0, (score - 50) / 30)  # 65→0.50, 80→1.0, 95→1.0
            position_size = max(0.5, min(1.0, position_size))
            
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
    

    def _save_veto_vote(self, symbol: str, current_price: float, veto_reason: str, side: str = 'long') -> None:
        """Save a vote record for VETO cases (dashboard fix)."""
        try:
            now = time.time()
            if not hasattr(self, '_last_vote_ts'):
                self._last_vote_ts = {}
            last_ts = self._last_vote_ts.get(symbol, 0)
            if now - last_ts < 3.0:
                return
            self._last_vote_ts[symbol] = now
            vote_record = {
                'ts': datetime.now(timezone.utc).isoformat(),
                'symbol': symbol,
                'approved': False,
                'final_score': 0.0,
                'threshold': 0,
                'price': round(current_price, 6),
                'veto_reason': veto_reason,
                'votes': {},
                'strong_count': 0,
            }
            vote_log_path = os.path.join(BASE_DIR, 'data', 'vote_history.json')
            os.makedirs(os.path.dirname(vote_log_path), exist_ok=True)
            history = []
            if os.path.exists(vote_log_path):
                try:
                    with open(vote_log_path, 'r') as f:
                        history = json.load(f)
                except:
                    history = []
            history.append(vote_record)
            if len(history) > 10000:
                history = history[-10000:]
            with open(vote_log_path, 'w') as f:
                json.dump(history, f, indent=2, default=str)
        except Exception as e:
            pass  # silently ignore


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
    
    def _get_entry_threshold(self, strong_votes: int = 0, liq_score: float = 0,
                              adv_score: float = 0, vsa_score: float = 0) -> float:
        """Адаптивный порог входа.

        Факторы снижения порога:
        - HMM CALM: -8 (риск-менее агрессивно)
        - 3+ модуля с score >= 75: -4
        - 2 модуля с score >= 75: -2
        - Объёмная тройка (Liq, VV, VSA) средняя >= 75: -5 (доверяем объёму)
        - Liq >= 80 + Adv >= 85: -3

        Returns:
            float — финальный порог (мин 55)
        """
        base = self._ENTRY_THRESHOLD
        if self.hmm_regime == 0:   # CALM
            base -= 8               # 57
        elif self.hmm_regime == 2: # VOLATILE
            base += 5               # 70

        # Снижение при сильных голосах независимо от HMM
        if strong_votes >= 3:
            base -= 4
        elif strong_votes >= 2:
            base -= 2

        # Объёмная тройка (Liq + VSA) сильна — сильно снижаем порог
        if liq_score >= 70 and vsa_score >= 70:
            base -= 5

        # Liq подтверждает Advisor
        if liq_score >= 80 and adv_score >= 85:
            base -= 3

        return max(55.0, base)     # не ниже 55 никогда
    
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
          1. MLProfessionalV2 (LightGBM, 27/16 признаков, multi-TF) — 20%
          2. MLAdvisor (RandomForest, 9 признаков, паттерны+VWAP+D1) — 5%
          3. Согласованность трендов 5M/1H/4H + HMM — 25%
          4. RSI + Объём + BTC-корреляция — 5%
          5. LiquidityCluster v2 (Order Flow/Block/Sweep) — 25%
          6. Volume/VWAP (VWAP реверсия + Volume Spike) — 20%
        
        Возвращает словарь с approval, score, раскладкой.
        """
        self._lazy_init_ml()
        is_short = (side == 'short')
        
        # Инициализируем результаты голосов
        # ⚡ БАЗОВЫЕ ВЕСА
        BASE_WEIGHTS = {
            'ml_v2': 0.20,
            'advisor': 0.10,
            'mtf': 0.20,   # было 0.35 — сумма была 1.15, теперь 1.0
            'rsi_vol_btc': 0.00,
            'liquidity': 0.25,
            'volume_vwap': 0.15,
            'vsa': 0.10,
        }
        # Сумма весов = 1.0 (0.20+0.10+0.20+0.00+0.25+0.15+0.10)

        scores = {}
        for k, w in BASE_WEIGHTS.items():
            scores[k] = {'score': 50, 'weight': w, 'detail': 'N/A'}
        # Веса для бонусов остаются (не входят в BASE_WEIGHTS)
        _bonus_keys = ['_reversal_bonus', '_price_bonus', '_btc_bonus']
        
        # ═══ ГОЛОС 1: MLProfessionalV2 (20%) ═══════════════════════════════
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
            # Если 5M bearish, но 1H и 4H бычьи — это нормальный откат, не штрафуем сильно
            # Полный штраф только когда все ТФ разнонаправлены
            d1 = abs(trend_5m_val - trend_1h_val)
            d2 = abs(trend_5m_val - trend_4h_val)
            d3 = abs(trend_1h_val - trend_4h_val)
            
            # Если старшие ТФ (1H и 4H) согласованы — штраф меньше
            if d3 < 20:
                # 1H и 4H смотрят в одну сторону — ослабляем штраф в 3 раза
                # (5M может быть просто откатом, не стоим за ним стеной)
                divergence = min(d1, d2) * 0.3
            else:
                divergence = max(d1, d2, d3)
            
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
        
        # ═══ ГОЛОС 6: Volume/VWAP (20%) ════════════════════════════════════
        try:
            if self._volume_vwap:
                vv = self._volume_vwap.evaluate(
                    symbol, current_price, candles_5m or [],
                    candles_1h or [], candles_4h
                )
                scores['volume_vwap']['score'] = vv['score']
                scores['volume_vwap']['detail'] = vv['detail']
            else:
                scores['volume_vwap']['score'] = 50
                scores['volume_vwap']['detail'] = 'not_loaded'
        except Exception as e:
            scores['volume_vwap']['score'] = 50
            scores['volume_vwap']['detail'] = f"error({e})"
        
        # ═══ ГОЛОС 7: VSA — Volume Spread Analysis (10%) ═══════════════════
        try:
            from vsa_analyzer import analyze_volume_spread
            if candles_5m and len(candles_5m) > 20:
                vsa = analyze_volume_spread(candles_5m)
                if vsa.signal == 'bullish':
                    vsa_score = 60 + int(vsa.strength * 40)  # 60-100
                elif vsa.signal == 'bearish':
                    vsa_score = 40 - int(vsa.strength * 30)  # 10-40
                else:
                    vsa_score = 50
                scores['vsa']['score'] = vsa_score
                scores['vsa']['detail'] = vsa.detail
        except Exception as e:
            scores['vsa']['score'] = 50
            scores['vsa']['detail'] = f'error({e})'

        # ═══ АДАПТИВНЫЕ ВЕСА ══════════════════════════════════════════════
        # Если Liq сильный (>70) а MTF слабый (<50) — снижаем вес MTF
        liq_score = scores['liquidity']['score']
        mtf_score = scores['mtf']['score']
        adv_score = scores['advisor']['score']
        vsa_score = scores['vsa']['score']

        if liq_score >= 75 and mtf_score < 50:
            # Liq уверен, MTF не подтверждает — не даём MTF душить сигнал
            mtf_shrink = 0.5 if adv_score >= 60 else 0.3
            if vsa_score >= 60:
                mtf_shrink = max(mtf_shrink, 0.4)  # VSA подтверждает — ещё ослабляем MTF
            transfer = BASE_WEIGHTS['mtf'] * mtf_shrink
            scores['mtf']['weight'] = BASE_WEIGHTS['mtf'] - transfer
            scores['liquidity']['weight'] = BASE_WEIGHTS['liquidity'] + transfer * 0.6
            scores['vsa']['weight'] = BASE_WEIGHTS['vsa'] + transfer * 0.4
            logger.debug(f"[DE] Адаптивные веса: MTF {BASE_WEIGHTS['mtf']:.2f}→{scores['mtf']['weight']:.2f}, Liq {BASE_WEIGHTS['liquidity']:.2f}→{scores['liquidity']['weight']:.2f}")
        elif liq_score >= 60 and adv_score >= 80 and mtf_score < 50:
            # Advisor сильный + Liq умеренный — немного ослабляем MTF
            transfer = BASE_WEIGHTS['mtf'] * 0.25
            scores['mtf']['weight'] = BASE_WEIGHTS['mtf'] - transfer
            scores['advisor']['weight'] = BASE_WEIGHTS['advisor'] + transfer * 0.6
            scores['liquidity']['weight'] = BASE_WEIGHTS['liquidity'] + transfer * 0.4

        # ═══ БОНУС ЗА ПОЗИЦИЮ ЦЕНЫ И РАННИЙ РАЗВОРОТ ══════════════════
        # Бонус +0..20 если цена в нижней половине дневного диапазона (отскок)
        # Бонус +0..15 дополнительно если детектор разворота активен
        price_bonus = 0
        reversal_bonus = 0
        try:
            if candles_5m and len(candles_5m) >= 100:
                h24_high = max(c['h'] for c in candles_5m[-288:] if isinstance(c, dict))
                h24_low = min(c['l'] for c in candles_5m[-288:] if isinstance(c, dict))
                if h24_high > h24_low:
                    pos = (current_price - h24_low) / (h24_high - h24_low)
                    # Бонус за позицию на дне
                    if pos < 0.50 and rsi < 72 and trend != 'bearish':
                        bonus_val = max(0, int((0.50 - pos) * 40))  # 0..20 баллов
                        bonus_val = min(bonus_val, 20)  # макс 20
                        price_bonus = bonus_val
                    
                    # ═══ ДЕТЕКТОР РАННЕГО РАЗВОРОТА ═══
                    # Проверяем последние 5 свечей 5M на паттерн разворота
                    try:
                        recent = [c for c in candles_5m[-6:] if isinstance(c, dict)]
                        if len(recent) >= 5:
                            closes = [c['c'] for c in recent]
                            volumes = [c['v'] for c in recent]
                            
                            # 1. Цена растёт последние 3 свечи из 4
                            green_count = sum(1 for i in range(-4, 0) if closes[i] > closes[i-1])
                            
                            # 2. Объём растёт на зелёных свечах
                            last3_vol = sum(volumes[-3:])
                            prev3_vol = sum(volumes[-6:-3])
                            
                            # 3. RSI выходит из зоны (был <= 50, идёт вверх)
                            rsi_recovering = 40 < rsi < 65
                            
                            # 4. Цена не на хаях (позиция < 75% чтобы не покупать топ)
                            not_too_high = pos < 0.75
                            
                            if (green_count >= 2 and 
                                last3_vol > prev3_vol * 1.2 and 
                                rsi_recovering and 
                                not_too_high):
                                # Сила разворота: 0-15 баллов
                                # Чем сильнее зелёных и объём — тем больше бонус
                                strength = min(green_count * 3 + 
                                              int((last3_vol / prev3_vol - 1) * 5), 15)
                                reversal_bonus = min(strength, 15)
                    except Exception:
                        pass
        except Exception:
            pass
        scores['_reversal_bonus'] = {'score': reversal_bonus, 'weight': 1.0}
        scores['_price_bonus'] = {'score': price_bonus, 'weight': 1.0}
        
        # ═══ ИТОГОВЫЙ СКОР ═════════════════════════════════════════════════
        final_score = sum(v['score'] * v['weight'] for v in scores.values())
        
        # ═══ BTC DIRECTION PREDICTOR ═══════════════════════════════════════
        btc_bonus = 0
        try:
            if self._btc_predictor is None:
                from btc_direction import BTCDirectionPredictor
                self._btc_predictor = BTCDirectionPredictor()
            btc_bonus = self._btc_predictor.calculate_bonus(final_score)
            scores['_btc_bonus'] = {'score': btc_bonus, 'weight': 1.0}
        except Exception as e:
            scores['_btc_bonus'] = {'score': 0, 'weight': 1.0}
            logger.debug(f"BTC Direction: ошибка: {e}")
        
        # ═══ БЛОКИРОВКА ОТ BTC ═══════════════════════════════════════════
        # Если btc_bonus == -999 — жёсткое veto на лонги (BTC падает >2% за 6ч)
        if btc_bonus <= -999:
            return {
                'approved': False,
                'reason': f"VETO: BTC падает >2% за 6ч",
                'final_score': 0,
                'threshold': threshold,
                'votes': {},
            }
        
        # Сколько модулей дают сильные сигналы (используем ядро голосов, без бонусов)
        core_votes = [scores['ml_v2'], scores['advisor'], scores['liquidity'],
                      scores['volume_vwap'], scores['vsa']]
        strong_votes = sum(1 for v in core_votes if v['score'] >= 75)
        
        threshold = self._get_entry_threshold(
            strong_votes=strong_votes,
            liq_score=scores['liquidity']['score'],
            adv_score=scores['advisor']['score'],
            vsa_score=scores['vsa']['score']
        )
        
        # Разбивка голосов для логов/дашборда
        votes_str = (
            f"ML-Pro:{scores['ml_v2']['score']:.0f}({scores['ml_v2']['detail']}) "
            f"Adv:{scores['advisor']['score']:.0f}({scores['advisor']['detail']}) "
            f"MTF:{scores['mtf']['score']:.0f}({scores['mtf']['detail']}) "
            f"RVB:{scores['rsi_vol_btc']['score']:.0f}({scores['rsi_vol_btc']['detail']}) "
            f"Liq:{scores['liquidity']['score']:.0f}({scores['liquidity']['detail']}) "
            f"VV:{scores['volume_vwap']['score']:.0f}({scores['volume_vwap']['detail']}) "
            f"VSA:{scores['vsa']['score']:.0f}({scores['vsa']['detail']})"
        )
        
        votes = {
            'ml_v2': {'score': scores['ml_v2']['score'], 'detail': scores['ml_v2']['detail']},
            'advisor': {'score': scores['advisor']['score'], 'detail': scores['advisor']['detail']},
            'mtf': {'score': scores['mtf']['score'], 'detail': scores['mtf']['detail']},
            'rsi_vol_btc': {'score': scores['rsi_vol_btc']['score'], 'detail': scores['rsi_vol_btc']['detail']},
            'liquidity': {'score': scores['liquidity']['score'], 'detail': scores['liquidity']['detail']},
            'volume_vwap': {'score': scores['volume_vwap']['score'], 'detail': scores['volume_vwap']['detail']},
            'vsa': {'score': scores['vsa']['score'], 'detail': scores['vsa']['detail']},
        }
        
        # ═══ СОХРАНЕНИЕ ИСТОРИИ ГОЛОСОВ ════════════════════════════════
        try:
            now = time.time()
            # Не чаще раза в 3 секунды для той же монеты
            if not hasattr(self, '_last_vote_ts'):
                self._last_vote_ts = {}
            last_ts = self._last_vote_ts.get(symbol, 0)
            if now - last_ts < 3.0:
                pass  # слишком часто — пропускаем запись на диск
            else:
                self._last_vote_ts[symbol] = now
                vote_record = {
                    'ts': datetime.now(timezone.utc).isoformat(),
                    'symbol': symbol,
                    'approved': final_score >= threshold,
                    'final_score': round(final_score, 1),
                    'threshold': threshold,
                    'price': round(current_price, 6),
                    'weights': {k: BASE_WEIGHTS.get(k, 0) for k in BASE_WEIGHTS},
                    'bonus': {'price': round(price_bonus, 1), 'reversal': round(reversal_bonus, 1), 'btc': round(btc_bonus, 1)},
                    'votes': votes,
                    'strong_count': strong_votes,
                }
                vote_log_path = os.path.join(BASE_DIR, 'data', 'vote_history.json')
                os.makedirs(os.path.dirname(vote_log_path), exist_ok=True)
                history = []
                if os.path.exists(vote_log_path):
                    try:
                        with open(vote_log_path, 'r') as f:
                            history = json.load(f)
                    except:
                        history = []
                history.append(vote_record)
                # Держим последние 10000 записей
                if len(history) > 10000:
                    history = history[-10000:]
                with open(vote_log_path, 'w') as f:
                    json.dump(history, f, indent=2, default=str)
        except Exception as e:
            logger.debug(f"[DE] vote_history save error: {e}")
        # ════════════════════════════════════════════════════════════════
        
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
                'volume_vwap_score': scores['volume_vwap']['score'],
                'votes': votes,
            }
        else:
            return {
                'approved': False,
                'reason': f"⏭️ Score={final_score:.0f} < {threshold}({strong_votes}≥75) {votes_str} bonus={price_bonus:.0f} rev={reversal_bonus:.0f} btc={btc_bonus:+.0f}",
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
