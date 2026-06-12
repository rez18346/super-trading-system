#!/usr/bin/env python3
"""
decision_engine.py - Единый центр принятия торговых решений.

Архитектура:
  DecisionEngine - синглтон, который решает:
    - ВХОДИТЬ ли в позицию (ML-ансамбль: 4 голоса + multi-timeframe)
    - ВЫХОДИТЬ из позиции (SL, TP, трейл, детектор разворота)

  Все остальные модули (trader, monitor) только собирают данные.
  Решение принимается в одном месте → выполняется в trader.

Голоса на вход (DecisionEngine._evaluate_entry_ensemble):
  1. MLProfessionalV2 (LightGBM, 27/16 признаков, 5M+1H+4H)  - 20%
  2. MLAdvisor (RandomForest, 9 признаков, паттерны+VWAP+D1) - 35%  🏆 усилен (забрал голос RVB)
  3. RSI/Vol/BTC (RSI<30 oversold + объёмный spike)          - 0%   ❌ отключён (дублирует Advisor)
  4. LiquidityCluster v2 (Order Flow/Block/Sweep)            - 25%
  5. Volume/VWAP (VWAP реверсия + Volume Spike)              - 20%
  ──────────────────────────────────────────────────────────
  MTF - 0% (информационно, без веса)
  VSA - 10% (качество движения: дивергенция, накопление/распределение)
  VSA - 10% (качество движения: дивергенция, накопление/распределение)
  Порог входа: 60 (адаптивный: CALM=52, NORMAL=60, VOLATILE=65)
  VETO: макс(1H=-15, 4H=-25) + BTC(до -25) - не складываем таймфреймы
  Override VETO: Liq≥75 + VV≥70 + Adv≥80 → игнорировать VETO

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

from vote_tracker import get_tracker as get_vote_tracker

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
    ⚡ ЗАЛОЖЕНО ПОД SMART EXIT: exit_override, exit_vote.

      Атрибуты:
      symbol        - тикер (XRP/USDT)
      action        - 'enter' | 'hold' | 'exit'
      side          - 'long' | 'short'
      score         - итоговый скор 0-100 (enter) или None
      position_size - доля от max 2% капитала (0.0-1.0)
      tp_levels     - уровни частичного тейка [(price,pct), ...] или None
      sl_price      - цена стоп-лосса (None = стандартный из конфига)
      tp_price      - цена тейк-профита (None = стандартный)
      trail_act     - % активации трейлинга (None = стандартный)
      trail_dist    - % дистанции трейлинга от пика (None = стандартный)
      max_hold_h    - макс. часов удержания (None = 48)
      reason        - причина решения
      signal_type   - тип сигнала (SignalType)
      priority      - приоритет (50 по умолч.)
      metadata      - полная раскладка голосов
      exit_override - 'exit' | 'hold_widen_sl' | 'hold_tighten_sl': как поступить (для exit)
      exit_vote     - полная раскладка exit ensemble (для exit & hold_widen_sl)
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
                 metadata: Optional[Dict] = None,
                 exit_override: Optional[str] = None,
                 exit_vote: Optional[Dict] = None):
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
        self.exit_override = exit_override
        self.exit_vote = exit_vote or {}

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
        self.reentry_cooldown = 14400  # 4 часа (базовый кулдаун после выхода)

        # Счётчик последовательных убытков на символ
        self._consecutive_losses: Dict[str, int] = {}
        self._consecutive_short_losses: Dict[str, int] = {}  # отдельно для шортов
        # Максимум убытков подряд перед блокировкой
        self.max_consecutive_losses = 2  # после 2х убытков подряд - блокировка
        self.max_consecutive_short_losses = 2  # для шортов жёстче
        
        # ⚡ BTC-фильтр для шортов
        self._btc_short_blocked = False  # блокировка шортов если BTC в up-trend
        # Файл для сохранения счётчика (переживает перезагрузки)
        self._losses_file = os.path.join(os.path.dirname(__file__), 'data', 'consecutive_losses.json')
        self._maybe_restore_losses()

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

    def _push_memory_to_ml(self) -> None:
        """Передать память о поведении ММ в MLAdvisor.

        Вызывается после record_exit() и в начале decide() для синхронизации.
        """
        if self._ml_advisor is None:
            return
        # Память о символах: ML больше не использует эти фичи
        pass

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
        0: {'sl': 1.5, 'tp': 8.0,   'trail_act': 2.0, 'trail_dist': 1.2},  # CALM
        1: {'sl': 1.5, 'tp': 10.0,  'trail_act': 2.5, 'trail_dist': 1.5},  # NORMAL
        2: {'sl': 1.5, 'tp': 14.0,  'trail_act': 3.0, 'trail_dist': 2.0},  # VOLATILE
    }

    # ⚡ ШОРТ: отдельные SL/TP параметры (консервативнее)
    _SHORT_SL_TP_BY_REGIME = {
        0: {'sl': 1.5, 'tp': 6.0,   'trail_act': 1.8, 'trail_dist': 1.2},  # CALM (TP ниже)
        1: {'sl': 1.5, 'tp': 6.0,   'trail_act': 2.0, 'trail_dist': 1.5},  # NORMAL (TP 6% vs 10%)
        2: {'sl': 2.0, 'tp': 8.0,   'trail_act': 2.5, 'trail_dist': 2.0},  # VOLATILE
    }

    def get_sl_tp_params(self, side: str = 'long') -> dict:
        """
        Вернуть параметры SL/TP/трейлинга под текущий HMM-режим.

        ⚡ ШОРТ: отдельные параметры (TP=6%, SL=1.5%, консервативнее).
        ⚡ ПЛЕЧО: режимы MARGIN/FUTURES используют те же %, но apply к margin.
        """
        if side == 'short':
            p = self._SHORT_SL_TP_BY_REGIME.get(self.hmm_regime, self._SHORT_SL_TP_BY_REGIME[1])
        else:
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
        Если не закэшированы - вернуть None (трейдер обновляет раз в N циклов).
        """
        # Для BTC используем set_multi_tf_data
        if symbol == 'BTC/USDT':
            return None
        return self._candles_1h  # fallback на общие данные

    def update_hmm_regime(self, regime: int) -> None:
        """Обновить HMM-режим рынка (вызывается из монитора)."""
        self.hmm_regime = regime
        logger.debug(f"[DE] HMM: {regime}")

    # ─── EXIT ENSEMBLE - МОДУЛЬНОЕ ГОЛОСОВАНИЕ НА ВЫХОД ───────────────────

    def _evaluate_exit_ensemble(self, symbol: str,
                                 entry_price: float,
                                 current_price: float,
                                 pnl_pct: float,
                                 highest_price: float = None,
                                 candles_5m: Optional[list] = None,
                                 candles_1h: Optional[list] = None,
                                 entry_vwap_deviation: Optional[float] = None,
                                 cvd_data: Optional[Dict] = None) -> Dict:
        """
        Профессиональный ансамбль для решения о выходе.

        Отвечает на вопрос: «Рынок всё ещё за позицию?»

        Веса (exit-специфичные, отличаются от entry):
          1. VSA (качество движения, распределение/накопление) - 35%
          2. MLAdvisor (уверенность в тренде)                  - 25%
          3. LiquidityCluster (структура ликвидности)          - 20%
          4. Volume/VWAP (реверсия, спайки)                    - 10%
          5. ML-v2 (LightGBM, резерв)                          - 10%

        Дополнительно:
          - Multi-TF тренд (1H): если старший тренд bullish → буст hold_confidence
          - VWAP entry deviation: если цена в зоне накопления → буст

        Возвращает:
        {
            'hold_confidence': 0-100,  # насколько уверенно надо держать
            'approved': bool,          # true = отменить/ослабить SL
            'votes': {...},            # раскладка по модулям
            'reason': str,             # пояснение
            'weights': {...},          # веса модулей
        }
        """
        self._lazy_init_ml()

        EXIT_WEIGHTS = {
            'vsa': 0.25,        # Было 0.35, убавили в пользу CVD
            'advisor': 0.25,
            'liquidity': 0.20,
            'volume_vwap': 0.10,
            'ml_v2': 0.10,
            'cvd': 0.10,         # Новый: CVD — видит buy/sell давление
        }

        scores = {}
        for k, w in EXIT_WEIGHTS.items():
            scores[k] = {'score': 50, 'weight': w, 'detail': 'N/A'}

        # ═══ VSA (35%) - критичен для выхода: видит распределение/накопление
        try:
            from vsa_analyzer import analyze_volume_spread
            vsa_result = analyze_volume_spread(candles_5m or [])
            if vsa_result.signal == 'bullish':
                vsa_score = 60 + vsa_result.strength * 35
                scores['vsa']['detail'] = f"bullish(strength={vsa_result.strength:.2f})"
            elif vsa_result.signal == 'bearish':
                vsa_score = 40 - vsa_result.strength * 35  # 5-40 - надо выходить
                scores['vsa']['detail'] = f"bearish(strength={vsa_result.strength:.2f})"
            else:
                vsa_score = 50
                scores['vsa']['detail'] = f"neutral({vsa_result.detail[:40]})"
            scores['vsa']['score'] = max(0, min(100, int(vsa_score)))
        except Exception as e:
            logger.debug(f"[EXIT] VSA: {e}")
            scores['vsa']['score'] = 50

        # ═══ Advisor (25%) - уверенность в продолжении тренда
        try:
            if self._ml_advisor and self._ml_advisor.is_trained:
                adv_df = None
                if candles_5m and len(candles_5m) > 10:
                    try:
                        import pandas as _pd
                        adv_df = _pd.DataFrame(candles_5m)
                    except Exception:
                        adv_df = None
                adv = self._ml_advisor.evaluate(
                    symbol, current_price, 50, 'neutral', 50, df=adv_df
                )
                if adv['decision'] == 'GOOD':
                    adv_score = 60 + adv['confidence'] * 35
                    scores['advisor']['score'] = min(100, int(adv_score))
                    scores['advisor']['detail'] = f"GOOD({adv['confidence']:.2f})"
                elif adv['decision'] == 'WEAK':
                    scores['advisor']['score'] = 45
                    scores['advisor']['detail'] = f"WEAK({adv['confidence']:.2f})"
                else:
                    scores['advisor']['score'] = 30
                    scores['advisor']['detail'] = f"SKIP({adv['confidence']:.2f})"
            else:
                scores['advisor']['score'] = 50
                scores['advisor']['detail'] = "not_trained"
        except Exception as e:
            logger.debug(f"[EXIT] Advisor: {e}")
            scores['advisor']['score'] = 50

        # ═══ Liquidity (20%) - структура ликвидности всё ещё за?
        try:
            if self._liquidity:
                liq = self._liquidity.evaluate(candles_5m or [], current_price, None, None)
                if liq and isinstance(liq, dict):
                    raw_liq = liq.get('score', 50)
                    scores['liquidity']['score'] = int(raw_liq)
                    scores['liquidity']['detail'] = f"score={raw_liq:.0f}"
                else:
                    scores['liquidity']['score'] = 50
            else:
                scores['liquidity']['score'] = 50
        except Exception as e:
            logger.debug(f"[EXIT] Liquidity: {e}")
            scores['liquidity']['score'] = 50

        # ═══ Volume/VWAP (10%) - спайки, реверсия
        try:
            if self._volume_vwap and candles_5m:
                vv = self._volume_vwap.evaluate(symbol, current_price, candles_5m, None, None)
                if vv and isinstance(vv, dict):
                    vv_score = vv.get('score', 50)
                    vv_signal = vv.get('signal', vv.get('detail', ''))
                    if 'reversion_sell' in str(vv_signal).lower():
                        vv_score = max(10, vv_score - 30)
                    scores['volume_vwap']['score'] = int(vv_score)
                    # Сохраняем сигнал для VV-фильтра
                    vv_raw_signal = str(vv_signal)
                    scores['volume_vwap']['detail'] = f"score={vv_score:.0f} sig={vv_raw_signal[:40]}"
                else:
                    scores['volume_vwap']['score'] = 50
            else:
                scores['volume_vwap']['score'] = 50
        except Exception as e:
            logger.debug(f"[EXIT] VWAP: {e}")
            scores['volume_vwap']['score'] = 50

        # ═══ ML-v2 (10%) - резервный голос
        try:
            if self._ml_pro_v2 and self._ml_pro_v2.trained:
                ml = self._ml_pro_v2.evaluate(symbol, candles_5m or [], [], [], 50, 'neutral', 50)
                ml_dec, ml_prob, _ = ml if len(ml) == 3 else (None, 0.5, None)
                if ml_dec == 'BUY':
                    scores['ml_v2']['score'] = 70
                    scores['ml_v2']['detail'] = f"BUY(prob={ml_prob:.2f})"
                elif ml_dec == 'WEAK_BUY':
                    scores['ml_v2']['score'] = 55
                    scores['ml_v2']['detail'] = f"WEAK(prob={ml_prob:.2f})"
                else:
                    scores['ml_v2']['score'] = 30
                    scores['ml_v2']['detail'] = f"SKIP(prob={ml_prob:.2f})"
            else:
                scores['ml_v2']['score'] = 50
        except Exception as e:
            logger.debug(f"[EXIT] ML-v2: {e}")
            scores['ml_v2']['score'] = 50

        # ═══ CVD (10%) — buy/sell давление ────────────────────────────────
        try:
            if cvd_data and cvd_data.get('minutes', 0) >= 2:
                trend = cvd_data.get('trend', 'neutral')
                buy_pct = cvd_data.get('buy_pct', 50)
                cvd_ratio = cvd_data.get('cvd_ratio', 1.0)
                
                # Основа: buy_pct (50% = нейтрально, 60%+ = бычий, 40%- = медвежий)
                cvd_score = 50 + (buy_pct - 50) * 1.5
                cvd_score = max(10, min(100, cvd_score))
                
                # Коррекция по тренду
                if trend == 'bullish':
                    cvd_score += 10
                elif trend == 'bearish':
                    cvd_score -= 10
                
                # Если ratio сильно несбалансирован
                if cvd_ratio > 1.5:
                    cvd_score += 5  # много покупок
                elif cvd_ratio < 0.67:
                    cvd_score -= 5  # много продаж
                
                cvd_score = max(10, min(100, cvd_score))
                scores['cvd']['score'] = cvd_score
                scores['cvd']['detail'] = f'trend={trend} buy={buy_pct:.0f}% ratio={cvd_ratio:.2f}'
                logger.debug(f"[EXIT→CVD] {symbol}: score={cvd_score:.0f} — {scores['cvd']['detail']}")
            else:
                scores['cvd']['score'] = 50
                scores['cvd']['detail'] = 'no_data'
        except Exception as e:
            logger.debug(f"[EXIT→CVD] {symbol}: {e}")
            scores['cvd']['score'] = 50
            scores['cvd']['detail'] = f'error({e})'

        # ═══ OI LIQUIDATION + FR BOOST ────────────────────────────────────
        # Если цена в зоне ликвидации шортов → потенциал сквиза → держим
        oi_squeeze = 0
        oi_heat = 0
        oi_detail = ''
        try:
            from collect_oi import get_oi_collector
            oi_collector = get_oi_collector()
            oi_levels = oi_collector.get_liq_levels(symbol, current_price)
            oi_heat = oi_levels.get('heat', 0)

            # Если цена в зоне ликвидации шортов
            sz = oi_levels.get('liq_zone_short')
            in_short_zone = False
            if sz and len(sz) == 2:
                if sz[0] <= current_price <= sz[1]:
                    in_short_zone = True
                # Или цена прямо у нижней границы (шорты начнут гореть при пробое)
                elif abs(current_price - sz[0]) / current_price < 0.02:
                    in_short_zone = True

            if oi_heat >= 1 and in_short_zone:
                oi_squeeze = 15 + oi_heat * 5  # 20-30 баллов буста
                oi_detail = f"Liq🔥OI(heat={oi_heat},+{oi_squeeze})"
            elif oi_heat >= 2:
                oi_squeeze = 10
                oi_detail = f"OI(heat={oi_heat},+{oi_squeeze})"
        except Exception:
            pass

        # FR (Funding Rate) - негативный FR = сквиз потенциал
        # Загружаем напрямую из CSV (путь совпадает с btc_direction)
        fr_squeeze = 0
        fr_detail = ''
        try:
            import csv
            import os as _os
            _fr_csv = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'data', 'btc_fr_oi.csv')
            if _os.path.exists(_fr_csv):
                with open(_fr_csv) as _f:
                    _rows = list(csv.DictReader(_f))
                if _rows:
                    _last = _rows[-1]
                    _fr_str = _last.get('funding_rate', '')
                    if _fr_str and _fr_str.strip():
                        last_fr = float(_fr_str)
                        if last_fr < -0.0005:  # FR сильно негативный (< -0.05%)
                            fr_squeeze = 20
                            fr_detail = f"FR🔥({last_fr:.6f},+{fr_squeeze})"
                        elif last_fr < -0.0001:  # FR умеренно негативный
                            fr_squeeze = 10
                            fr_detail = f"FR({last_fr:.6f},+{fr_squeeze})"
        except Exception:
            pass

        # Итоговый буст: OI + FR, максимум 35
        squeeze_bonus = min(35, oi_squeeze + fr_squeeze)
        squeeze_detail = ' | '.join(filter(None, [oi_detail, fr_detail]))

        # ═══ БАЗОВЫЙ SCORE HOLD (ансамбль 5 модулей) ──────────────────────
        total_weight = sum(EXIT_WEIGHTS.values())
        hold_confidence = sum(s['score'] * s['weight'] for s in scores.values()) / total_weight if total_weight > 0 else 50
        hold_confidence = max(0, min(100, int(round(hold_confidence))))

        # Применяем squeeze bonus (OI + FR)
        if squeeze_bonus > 0:
            hold_confidence = min(100, hold_confidence + squeeze_bonus)

        # ═══ MULTI-TF TREND BOOST (1H) - старший тренд сильный → держим ───
        mtf_boost = 0
        mtf_detail = ''
        if candles_1h is not None and len(candles_1h) >= 10:
            try:
                trend_1h = self._calc_trend_from_candles(candles_1h)
                if trend_1h in ('strong_bullish', 'bullish'):
                    mtf_boost = 15 if trend_1h == 'strong_bullish' else 10
                    mtf_detail = f"1H={trend_1h}(+{mtf_boost})"
                elif trend_1h in ('bearish', 'strong_bearish'):
                    mtf_boost = -10
                    mtf_detail = f"1H={trend_1h}({mtf_boost})"
            except Exception as e:
                logger.debug(f"[EXIT] Multi-TF trend: {e}")

        # ═══ VWAP ENTRY DEVIATION - проверка растянутости ────────────────
        vwap_boost = 0
        vwap_detail = ''
        if entry_vwap_deviation is not None:
            # deviation > 0 = цена выше VWAP (растянута)
            # deviation < 0 = цена ниже VWAP (в зоне накопления)
            if entry_vwap_deviation < 1.0:
                # Цена близка к VWAP entry - зона накопления, держим
                vwap_boost = 10
                vwap_detail = f"VWAP_dev={entry_vwap_deviation:+.1f}%(накопление,+{vwap_boost})"
            elif entry_vwap_deviation > 3.0:
                # Цена сильно выше VWAP entry - растянута, снижаем уверенность
                vwap_boost = -10
                vwap_detail = f"VWAP_dev={entry_vwap_deviation:+.1f}%(растянуто,{vwap_boost})"
            else:
                vwap_detail = f"VWAP_dev={entry_vwap_deviation:+.1f}%(норма)"

        # ═══ ИТОГОВЫЙ SCORE HOLD ────────────────────────────────────────
        # Применяем бусты от старшего тренда и VWAP
        if mtf_boost != 0:
            hold_confidence = max(0, min(100, hold_confidence + mtf_boost))
        if vwap_boost != 0:
            hold_confidence = max(0, min(100, hold_confidence + vwap_boost))

        # ═══ ПЕРЕСЧЁТ РЕШЕНИЯ после бустов ───────────────────────────────
        approved = False
        widen_pct = 0.0
        if hold_confidence >= 55:
            if hold_confidence >= 65:
                approved = True
                widen_pct = 0.5
            else:
                approved = True
                widen_pct = 0.3

        # ═══ ОБНОВЛЕНИЕ reason с учётом новых модулей ────────────────────
        details = ' | '.join(f"{k}={v['score']}" for k, v in scores.items())
        reason = f"EXIT-VOTE: hold={hold_confidence}% {'✅' if approved else '❌'} [{details}]"
        if squeeze_detail:
            reason += f" | {squeeze_detail}"
        if mtf_detail:
            reason += f" | {mtf_detail}"
        if vwap_detail:
            reason += f" | {vwap_detail}"

        if approved and widen_pct == 0.3:
            reason += " | → расширение SL 30%"
        elif approved and widen_pct == 0.5:
            reason += " | → отмена SL"

        # ⚡ PnL-контекст: если уже в хорошем плюсе (>3%), ослабляем требования
        if pnl_pct > 3.0 and hold_confidence >= 55:
            approved = True
            reason += " | PnL>3% → ослаблен порог"

        return {
            'hold_confidence': hold_confidence,
            'approved': approved,
            'widen_pct': widen_pct,
            'votes': scores,
            'reason': reason,
            'weights': EXIT_WEIGHTS,
            'squeeze_bonus': squeeze_bonus,
            'oi_heat': oi_heat,
            'mtf_boost': mtf_boost,
            'vwap_boost': vwap_boost,
        }

    # ─── ВЫХОД ИЗ ПОЗИЦИИ ──────────────────────────────────────────────────

    def decide_exit(self, symbol: str, entry_price: float, current_price: float,
                    highest_price: float, lowest_price: float,
                    entry_time: datetime, pnl_pct: float,
                    sl_pct: float, tp_pct: float, trail_act: float,
                    trail_dist: float, max_hold_hours: float = 48,
                    side: str = 'long',
                    candles_1m: Optional[list] = None,
                    candles_1h: Optional[list] = None,
                    entry_vwap_deviation: Optional[float] = None,
                    current_sl_pct: Optional[float] = None,
                    max_sl_pct: float = 7.5,
                    cvd_data: Optional[Dict] = None) -> Optional[Decision]:
        """
        Принять решение о выходе.
        Приоритет: SL > TP > трейлинг > таймаут.

        ⚡ SMART EXIT: перед выходом по SL/TP/трейлингу/таймауту проверяет
           exit-ensemble (VSA + Advisor + Liq + VWAP + ML-v2).
           Если ensemble говорит «держать» (hold_confidence >= 55) -
           возвращает hold_widen_sl с расширенным SL вместо выхода.
           При 55-64: расширение SL на 30%. При ≥65: полное (50% расширение).

        ⚡ SL - ТВЁРДЫЙ УРОВЕНЬ:
           - Пред-SL зона: цена рядом с SL, но не пробила → ensemble расширяет
           - SL пробит: sell unconditionally, ensemble не слушается
           - Расширение от текущего SL, не от базы (прогрессивная защита)

        ⚡ ЗАЛОЖЕНО ПОД ШОРТ: side='short' инвертирует логику SL/TP/трейлинга.

        Параметры:
            sl_pct: базовый SL из конфига (используется если current_sl_pct не передан)
            current_sl_pct: текущий эффективный SL (может быть уже расширен).
                            Если None - используется sl_pct.
            max_sl_pct: максимальный SL, выше которого нельзя расширять
            candles_1m: 1M свечи для exit ensemble (опционально)
            candles_1h: 1H свечи для multi-TF тренд-проверки
            entry_vwap_deviation: отклонение текущей цены от VWAP с момента входа (%)
        """
        self.stats['total_decisions'] += 1
        is_short = (side == 'short')

        # ─── ИСПОЛЬЗУЕМ ТЕКУЩИЙ SL (может быть расширен) ─────────────
        _effective_sl = current_sl_pct if current_sl_pct is not None else sl_pct

        # ═══ ВСПОМОГАТЕЛЬНАЯ: exit ensemble для TP/трейлинга ─────────────
        def _check_exit_ensemble(trigger_type: str, base_reason: str) -> Optional[Decision]:
            """Проверить exit ensemble: если модули говорят «держать» - расширить SL.

            Возвращает:
                Decision с exit_override='hold_widen_sl' - расширить SL
                Decision с exit_override='exit' - выход подтверждён
                None - нет данных для ensemble (выход как обычно)
            """
            if candles_1m is None or len(candles_1m) < 5:
                return None  # нет данных для ensemble → exit как обычно

            exit_vote = self._evaluate_exit_ensemble(
                symbol, entry_price, current_price, pnl_pct,
                highest_price=highest_price, candles_5m=candles_1m,
                candles_1h=candles_1h, entry_vwap_deviation=entry_vwap_deviation,
                cvd_data=cvd_data
            )

            if exit_vote['approved']:
                # Модули говорят «держать» → расширяем SL от текущего уровня
                widen_pct = exit_vote.get('widen_pct', 0.3)  # 0.3 для 30%, 0.5 для 50%
                new_sl_pct = min(_effective_sl * (1 + widen_pct), max_sl_pct)
                action = 'расширение 30%' if widen_pct < 0.5 else 'полное (50%)'
                reason = (f"{trigger_type} {action}: ensemble hold={exit_vote['hold_confidence']}% "
                          f"[{exit_vote.get('reason','')}]"
                          f" | SL {_effective_sl:.2f}%→{new_sl_pct:.2f}%")
                logger.info(f"🧠 [SMART EXIT] {symbol}: {reason}")

                return Decision(
                    symbol, 'hold', side=side, priority=85,
                    reason=reason,
                    signal_type=SignalType.HOLD,
                    sl_price=entry_price * (1 - new_sl_pct / 100.0),
                    exit_override='hold_widen_sl',
                    exit_vote=exit_vote
                )
            else:
                # Модули согласны с выходом
                reason = f"{base_reason} | exit_ensemble hold={exit_vote['hold_confidence']}% - выход подтверждён"
                logger.info(f"🧠 [SMART EXIT] {symbol}: {reason}")
                return Decision(
                    symbol, 'exit', side=side, priority=80 if trigger_type != 'TP' else 90,
                    reason=reason,
                    signal_type=SignalType.SELL,
                    exit_override='exit',
                    exit_vote=exit_vote
                )

        # ─── 1. SL - стоп-лосс ──────────────────────────────────────────
        # 🎯 SL - БЕЗУСЛОВНЫЙ выход. Ensemble не спрашивается.
        #    Чем быстрее закроем убыток — тем лучше.
        if pnl_pct <= -_effective_sl:
            self.stats['exit_decisions'] += 1
            return Decision(symbol, 'exit', side=side,
                           reason=f"SL -{pnl_pct:.2f}% (лимит -{_effective_sl:.2f}%)",
                           signal_type=SignalType.STRONG_SELL, priority=100,
                           exit_override='exit')

        # ─── 2. TP - тейк-профит с exit ensemble ──────────────────────────
        if pnl_pct >= tp_pct:
            self.stats['exit_decisions'] += 1
            ensemble_result = _check_exit_ensemble(
                'TP',
                f"TP +{pnl_pct:.2f}% (лимит +{tp_pct}%)"
            )
            if ensemble_result:
                return ensemble_result
            # Fallback: нет данных для ensemble → exit как обычно
            return Decision(symbol, 'exit', side=side,
                           reason=f"TP +{pnl_pct:.2f}% (лимит +{tp_pct}%)",
                           signal_type=SignalType.STRONG_SELL, priority=90,
                           exit_override='exit')

        # ─── 3. Трейлинг-стоп с exit ensemble ─────────────────────────────
        if is_short:
            trail_active_pct = (entry_price - lowest_price) / entry_price * 100
            if trail_active_pct >= trail_act:
                trail_level = lowest_price * (1 + trail_dist / 100.0)
                if current_price >= trail_level:
                    self.stats['exit_decisions'] += 1
                    ensemble_result = _check_exit_ensemble(
                        'Short-трейлинг',
                        f"Short-трейлинг: L={lowest_price:.4f}→{current_price:.4f}"
                        f" (активация: {trail_act}%, дист: {trail_dist}%)"
                    )
                    if ensemble_result:
                        return ensemble_result
                    return Decision(
                        symbol, 'exit', side=side, priority=80,
                        reason=(f"Short-трейлинг: L={lowest_price:.4f}→{current_price:.4f}"
                                f" (активация: {trail_act}%, дист: {trail_dist}%)"),
                        signal_type=SignalType.SELL,
                        exit_override='exit'
                    )
        else:
            # ⚡ На бычьем 1H тренде трейлинг отключается - держим до ensemble/SL
            _do_trail = True
            if candles_1h and len(candles_1h) >= 20:
                _1h_t = self._calc_trend_from_candles(candles_1h)
                if _1h_t in ('strong_bullish', 'bullish'):
                    _do_trail = False
                    logger.info(f"🌡️ [TRAIL→СКИП] {symbol}: 1H={_1h_t} - трейлинг отключён (bullish trend)")

            if _do_trail:
                trail_active_pct = (highest_price - entry_price) / entry_price * 100
                if trail_active_pct >= trail_act:
                    trail_level = highest_price * (1 - trail_dist / 100.0)
                    if current_price <= trail_level:
                        self.stats['exit_decisions'] += 1
                        ensemble_result = _check_exit_ensemble(
                            'Long-трейлинг',
                            f"Long-трейлинг: H={highest_price:.4f}→{current_price:.4f}"
                            f" (активация: {trail_act}%, дист: {trail_dist}%)"
                        )
                        if ensemble_result:
                            return ensemble_result
                        return Decision(
                            symbol, 'exit', side=side, priority=80,
                            reason=(f"Long-трейлинг: H={highest_price:.4f}→{current_price:.4f}"
                                    f" (активация: {trail_act}%, дист: {trail_dist}%)"),
                            signal_type=SignalType.SELL,
                            exit_override='exit'
                        )

        # ─── 4. Таймаут удержания (48 часов) ──────────────────────────────
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        hold_hours = (now - entry_time).total_seconds() / 3600
        if hold_hours >= max_hold_hours:
            self.stats['exit_decisions'] += 1
            ensemble_result = _check_exit_ensemble(
                'Таймаут',
                f"Таймаут {hold_hours:.1f}ч > {max_hold_hours}ч"
            )
            if ensemble_result:
                return ensemble_result
            return Decision(
                symbol, 'exit', side=side, priority=70,
                reason=f"Таймаут {hold_hours:.1f}ч > {max_hold_hours}ч",
                signal_type=SignalType.WEAK_SELL,
                exit_override='exit'
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
                     current_positions_count: int, max_positions: int = 8,
                     candles_5m: Optional[list] = None,
                     candles_1h: Optional[list] = None,
                     candles_4h: Optional[list] = None,
                     side: str = 'long',
                     last_exit_price: Optional[float] = None,
                     last_exit_time: Optional[float] = None,
                     cvd_data: Optional[Dict] = None,
                     high_24h: Optional[float] = None,
                     low_24h: Optional[float] = None) -> Decision:
        """
        Принять решение о входе в позицию.

        ⚡ ЗАЛОЖЕНО ПОД ШОРТ: side='short' инвертирует логику.

        AN SAMBЛЬ:
          1. MLProfessionalV2 (LightGBM, 27/16 признаков, 5M+1H+4H) - 35%
          2. MLAdvisor (RandomForest, 9 признаков)                  - 20%
          3. HMM + Мульти-таймфрейм согласованность (5M/1H/4H)     - 25%
          4. RSI + Объём + BTC-корреляция                           - 20%

        Vetо:
          - 1H или 4H bearish → блокировка (для long)
          - 1H или 4H bullish → блокировка (для short)
          - BTC падает >1.5% за 4H → блокировка
          - Максимум N позиций
          - Кулдаун повторного входа
          - Anti-FOMO: если цена выше последнего выхода на 3%+ в течение 4ч
        """
        self.stats['total_decisions'] += 1
        is_short = (side == 'short')
        self._lazy_init_ml()

        # Синхронизируем память о поведении ММ с MLAdvisor
        self._push_memory_to_ml()

        # Используем переданные свечи или закэшированные
        c1h = candles_1h if candles_1h is not None else self._candles_1h
        c4h = candles_4h if candles_4h is not None else self._candles_4h
        btc_p = self._btc_price

        now = time.time()

        # ═══ ЖЁСТКИЕ БЛОКИРОВКИ ═══════════════════════════════════════════

        # 1. Anti-FOMO: не входить если цена значительно выше последнего выхода
        #    Защита от покупки на хаях после того, как нас высадило
        if last_exit_price is not None and last_exit_time is not None:
            fomo_window = 14400  # 4 часа после выхода
            fomo_threshold = 1.03  # 3% выше цены выхода

            time_since_exit = now - last_exit_time

            if time_since_exit < fomo_window:
                if not is_short:
                    # Для long: если цена выше last_exit_price * threshold - FOMO
                    if current_price > last_exit_price * fomo_threshold:
                        pct_above = (current_price / last_exit_price - 1) * 100
                        remain = (fomo_window - time_since_exit) / 60
                        reason = (f"🚫 Anti-FOMO: цена +{pct_above:.1f}% выше последнего выхода "
                                  f"(${last_exit_price:.4f}), осталось {remain:.0f}мин")
                        self.stats['veto_fomo'] = self.stats.get('veto_fomo', 0) + 1
                        self._save_veto_vote(symbol, current_price, reason, side)
                        return Decision(symbol, 'hold', side=side, reason=reason)
                else:
                    # Для short: если цена ниже last_exit_price / threshold - FOMO
                    if current_price < last_exit_price / fomo_threshold:
                        pct_below = (1 - current_price / last_exit_price) * 100
                        remain = (fomo_window - time_since_exit) / 60
                        reason = (f"🚫 Anti-FOMO: цена -{pct_below:.1f}% ниже последнего выхода "
                                  f"(${last_exit_price:.4f}), осталось {remain:.0f}мин")
                        self.stats['veto_fomo'] = self.stats.get('veto_fomo', 0) + 1
                        self._save_veto_vote(symbol, current_price, reason, side)
                        return Decision(symbol, 'hold', side=side, reason=reason)

        # 2. Лимит позиций
        if current_positions_count >= max_positions:
            self._save_veto_vote(symbol, current_price, f"Максимум {max_positions} позиций", side)
            return Decision(symbol, 'hold', side=side, reason=f"Максимум {max_positions} позиций")

        # 3. Кулдаун повторного входа (прогрессивный)
        if symbol in self._last_decisions:
            last_exit = self._last_decisions[symbol]
            time_since = now - last_exit.get('exit_time', 0)

            # Прогрессивный кулдаун: чем больше убытков подряд, тем дольше ждём
            losses = self._consecutive_losses.get(symbol, 0)
            if losses >= self.max_consecutive_losses:
                # После N убытков подряд - жёсткая блокировка на N*4 часов
                hard_block_hours = losses * 4
                hard_block_sec = hard_block_hours * 3600
                if time_since < hard_block_sec:
                    remain = hard_block_sec - time_since
                    reason = f"🚫 {losses} убытка подряд, блокировка на {hard_block_hours}ч (осталось {remain/3600:.1f}ч)"
                    self._save_veto_vote(symbol, current_price, reason, side)
                    return Decision(symbol, 'hold', side=side, reason=reason)

            if time_since < self.reentry_cooldown:
                remain = self.reentry_cooldown - time_since
                reason = f"Повторный вход через {remain/60:.1f} мин"
                self._save_veto_vote(symbol, current_price, reason, side)
                return Decision(
                    symbol, 'hold', side=side,
                    reason=reason
                )

        # 3. BTC-корреляция: штраф вместо блокировки
        veto_penalty = 0
        veto_reasons = []

        if btc_p is not None and self._btc_reference_price is not None:
            btc_change_4h = (btc_p - self._btc_reference_price) / self._btc_reference_price * 100
            if btc_change_4h < -1.5:
                penalty = min(25, int(abs(btc_change_4h) * 10))
                veto_penalty += penalty
                veto_reasons.append(f"BTC-{btc_change_4h:.1f}%")
                self.stats['veto_btc_drop'] += 1

        # 4. Multi-timeframe veto: штраф к score вместо блокировки
        # Берём максимум из двух, не складываем (1H=-15, 4H=-25, оба=-25)
        # ⚡ ЗАЛОЖЕНО ПОД ШОРТ: для short - блокировка при bullish
        veto_trends_long = ('strong_bearish',)      # Long: не входить на медвежьем ТФ
        veto_trends_short = ('strong_bullish',)     # Short: не входить на бычьем ТФ
        veto_trends = veto_trends_short if is_short else veto_trends_long
        veto_label = 'bullish' if is_short else 'bearish'

        max_tf_penalty = 0
        if c1h is not None and len(c1h) > 0:
            try:
                trend_1h = self._calc_trend_from_candles(c1h)
                if trend_1h in veto_trends:
                    max_tf_penalty = 15
                    self.stats['veto_mtf_conflict'] += 1
            except Exception as _e:
                logger.debug(f"[VETO] calc_trend 1h: {_e}")

        if c4h is not None and len(c4h) > 0:
            try:
                trend_4h = self._calc_trend_from_candles(c4h)
                if trend_4h in veto_trends:
                    max_tf_penalty = 25
                    self.stats['veto_mtf_conflict'] += 1
            except Exception as _e:
                logger.debug(f"[VETO] calc_trend 4h: {_e}")

        if max_tf_penalty > 0:
            veto_penalty += max_tf_penalty
            veto_reasons.append("TF")

        # BTC штраф остаётся отдельно (до -25)
        # Всего: макс TF(-15..-25) + BTC(0..-25) = до -50

        # ═══ АНСАМБЛЬ ГОЛОСОВ ═══════════════════════════════════════════════

        entry_checks = self._evaluate_entry_ensemble(
            symbol, confidence, trend, rsi, current_price,
            candles_5m=candles_5m, candles_1h=c1h, candles_4h=c4h,
            side=side, cvd_data=cvd_data, high_24h=high_24h, low_24h=low_24h
        )

        # ═══ ПРИМЕНЕНИЕ VETO-ШТРАФА ═══════════════════════════════════════
        if veto_penalty > 0:
            # Override: Liq≥75 + VV≥70 + Adv≥80 → сигнал сильнее VETO
            liq_score = entry_checks.get('liquidity_score', 0)
            adv_score = entry_checks.get('advisor_score', 0)
            # CVD тоже сильный сигнал — если 80+ и Liq≥70, перебивает VETO
            cvd_score = entry_checks.get('cvd_score', 50)
            vv_votes = entry_checks.get('votes', {}).get('volume_vwap', {})
            vv_score = vv_votes.get('score', 50) if isinstance(vv_votes, dict) else 50

            if liq_score >= 75 and vv_score >= 70 and adv_score >= 80:
                logger.info(f"⚠️ [DE→VETO_OVERRIDE] {symbol}: VETO {'/'.join(veto_reasons)} проигнорирован - сильный объёмный сигнал (Liq={liq_score} VV={vv_score} Adv={adv_score})")
            else:
                score = entry_checks.get('final_score', 65)
                score -= veto_penalty
                entry_checks['final_score'] = score
                veto_reason_str = '/'.join(veto_reasons)
                entry_checks['reason'] += f" | VETO({veto_reason_str}): -{veto_penalty}pts score={score:.0f}"
                if score < entry_checks.get('threshold', 65):
                    entry_checks['approved'] = False
                logger.debug(f"[DE→VETO] {symbol}: штраф -{veto_penalty} ({veto_reason_str})")

            # Сохраняем VETO-голос для дашборда
            self._save_veto_vote(symbol, current_price, f"VETO: {'/'.join(veto_reasons)} -{veto_penalty}pts", side)

        if entry_checks['approved']:
            self.stats['entry_decisions'] += 1

            # Position size: пропорционально скору
            score = entry_checks.get('final_score', 65)
            # 0.5 при 65, 1.0 при 90+
            position_size = min(1.0, (score - 50) / 30)  # 65→0.50, 80→1.0, 95→1.0
            position_size = max(0.5, min(1.0, position_size))

            # SL/TP от HMM-режима
            rp = self.get_sl_tp_params(side)

            # 🧠 Сохраняем свечи входа для дообучения ML-Pro v2
            # (store_entry_features сам проверит, есть ли свечи от _evaluate_27f)
            if self._ml_pro_v2:
                self._ml_pro_v2.store_entry_features(symbol)

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
            import tempfile
            history = []
            if os.path.exists(vote_log_path):
                try:
                    with open(vote_log_path, 'r') as f:
                        history = json.load(f)
                except Exception as _e:
                    history = []
            history.append(vote_record)
            if len(history) > 10000:
                history = history[-10000:]
            # Атомарная запись: temp + rename
            tmp_path = vote_log_path + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            os.replace(tmp_path, vote_log_path)
        except Exception as e:
            logger.debug(f"[DE] vote_log save: {e}")


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

    _ENTRY_THRESHOLD = 60.0

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
            float - финальный порог (мин 55)
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

        # Объёмная тройка (Liq + VSA) сильна - сильно снижаем порог
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
                                   side: str = 'long',
                                   cvd_data: Optional[Dict] = None,
                                   high_24h: Optional[float] = None,
                                   low_24h: Optional[float] = None,
                                   btc_state: Optional[Dict] = None) -> Dict:
        """
        Профессиональный ансамбль из 7 голосов + RP (Recovery & Potential).

        ⚡ ШОРТ: все сигналы инвертируются:
           - VSA bearish → boost, VSA bullish → penalty
           - CVD sell → boost, CVD buy → penalty
           - VV reversion_buy → boost
           - ML-Pro BUY prob < 0.45 → short boost
           - Liq: зоны сопротивления (SWH) → boost
           - BTC в up-trend → шорт заблокирован

        ⚡ ПЛЕЧО: leverage не меняет логику голосов, только размер позиции.

        Веса:
          1. MLProfessionalV2 (LightGBM, 27/16 признаков, multi-TF) - 20%
          2. MLAdvisor (RandomForest, 9 признаков, паттерны+VWAP+D1) - 5%
          3. Согласованность трендов 5M/1H/4H + HMM - 25%
          4. RSI + Объём + BTC-корреляция - 5%
          5. LiquidityCluster v2 (Order Flow/Block/Sweep) - 25%
          6. Volume/VWAP (VWAP реверсия + Volume Spike) - 20%
          7. CVD Order Flow (реальный поток агрессивных сделок) - 10% 🆕
          8. RP (Recovery & Potential) — мета-дельта −20..+20 🆕

        Возвращает словарь с approval, score, раскладкой.
        """
        self._lazy_init_ml()
        is_short = (side == 'short')

        # Инициализируем результаты голосов
        # ⚡ БАЗОВЫЕ ВЕСА
        BASE_WEIGHTS = {
            'ml_v2': 0.10,
            'advisor': 0.25,
            'mtf': 0.00,   # информационно, без веса
            'rsi_vol_btc': 0.00,  # отключён - дублирует Advisor
            'liquidity': 0.25,
            'volume_vwap': 0.10,
            'vsa': 0.20,   # VSA - качество движения (повышено для защиты от заходов на вершине)
            'cvd': 0.10,   # 🆕 Order Flow CVD
        }
        # Сумма весов = 1.0 (0.10+0.25+0.00+0.00+0.25+0.10+0.20+0.10)

        scores = {}
        for k, w in BASE_WEIGHTS.items():
            scores[k] = {'score': 50, 'weight': w, 'detail': 'N/A'}

        # 🧠 Динамические веса от VoteTracker (accuracy-based)
        try:
            _vt_weights = get_vote_tracker().get_weights()
            for mod in ['ml_v2', 'advisor', 'liquidity', 'volume_vwap', 'vsa', 'cvd']:
                if mod in scores and mod in _vt_weights:
                    new_w = _vt_weights[mod]
                    if abs(new_w - BASE_WEIGHTS.get(mod, 0)) > 0.001:
                        scores[mod]['weight'] = new_w
                        logger.debug(f"[DE] 🧠 VoteTracker вес {mod}: {BASE_WEIGHTS.get(mod, 0):.2f}→{new_w:.2f}")
        except Exception as _e:
            logger.debug(f"[DE] VoteTracker weight error: {_e}")

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
                    scores['ml_v2']['score'] = 20  # SKIP - штраф
                    scores['ml_v2']['detail'] = f"SKIP(prob={ml_prob:.2f})"
                    self.stats['ml_v2_skips'] += 1
            else:
                # ML не обучен - доверяем confidence
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
                    except Exception as _e:
                        logger.debug(f"[Advisor] df build: {_e}")
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
                # MLAdvisor не обучен - нейтральный голос
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
            # Если 5M bearish, но 1H и 4H бычьи - это нормальный откат, не штрафуем сильно
            # Полный штраф только когда все ТФ разнонаправлены
            d1 = abs(trend_5m_val - trend_1h_val)
            d2 = abs(trend_5m_val - trend_4h_val)
            d3 = abs(trend_1h_val - trend_4h_val)

            # Если старшие ТФ (1H и 4H) согласованы - штраф меньше
            if d3 < 20:
                # 1H и 4H смотрят в одну сторону - ослабляем штраф в 3 раза
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

            # BTC-корреляция (25%) - для short инвертирована
            btc_part = 50
            if self._btc_price and self._btc_reference_price:
                btc_change = (self._btc_price - self._btc_reference_price) / self._btc_reference_price * 100
                if btc_change > 0.5:
                    btc_part = 80 if not is_short else 20  # BTC растёт → хорошо для long, плохо для short
                elif btc_change < -0.5:
                    btc_part = 20 if not is_short else 80  # BTC падает → плохо для long, хорошо для short
                else:
                    btc_part = 55

            # Объём (20%) - пока нейтрально
            vol_part = 50

            rsi_vol_btc_score = rsi_part * 0.55 + btc_part * 0.25 + vol_part * 0.20

            scores['rsi_vol_btc']['score'] = rsi_vol_btc_score
            scores['rsi_vol_btc']['detail'] = (
                f"rsi={rsi_part:.0f} btc={btc_part:.0f} vol={vol_part:.0f}"
            )
        except Exception as e:
            scores['rsi_vol_btc']['score'] = 50
            scores['rsi_vol_btc']['detail'] = f"error({e})"

        # ═══ ГОЛОС 5: LiquidityCluster (25%) ═══════════════════════════════
        try:
            if self._liquidity:
                liq = self._liquidity.evaluate(candles_5m, current_price,
                                                candles_1h, candles_4h)
                liq_score = liq['score']

                # ─── OI LIQ BONUS ────────────────────────────────────────
                # Подмешиваем уровни ликвидаций из OI Delta
                oi_bonus = 0
                oi_heat = 0
                try:
                    from collect_oi import get_oi_collector
                    oi_collector = get_oi_collector()
                    oi_levels = oi_collector.get_liq_levels(symbol, current_price)
                    oi_bonus = oi_levels.get('score_bonus', 0)
                    oi_heat = oi_levels.get('heat', 0)
                    if oi_bonus > 0:
                        liq_score = min(100, liq_score + oi_bonus)
                        if oi_heat >= 2:
                            self.stats['oi_liq_hits'] = self.stats.get('oi_liq_hits', 0) + 1
                except Exception:
                    pass  # OI collector может быть не загружен

                scores['liquidity']['score'] = liq_score
                scores['liquidity']['detail'] = liq['detail']
                if oi_heat > 0:
                    scores['liquidity']['detail'] += f" OI🔥{oi_heat}(+{oi_bonus})"
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
                vv_score_orig = vv['score']
                if is_short:
                    # ⚡ ШОРТ: инверсия VWAP-сигнала
                    # reversion_sell → цены растянуты → шорт
                    # reversion_buy → цены сжаты → шорт это плохая идея
                    detail = vv.get('detail', '')
                    if 'reversion_sell' in detail or 'SPIKE' in detail:
                        vv_score = min(100, vv_score_orig + 25)  # перегрет → шорт
                        vv_detail = f"{detail} (short-boost)"
                    elif 'reversion_buy' in detail:
                        vv_score = max(10, vv_score_orig - 25)  # дешёво → не шорт
                        vv_detail = f"{detail} (short-penalty)"
                    else:
                        # Инвертируем зеркально: 100→0, 50→50, 0→100
                        vv_score = 100 - vv_score_orig
                        vv_detail = f"{detail} (short-inverted)"
                    scores['volume_vwap']['score'] = vv_score
                    scores['volume_vwap']['detail'] = vv_detail
                else:
                    scores['volume_vwap']['score'] = vv_score_orig
                    scores['volume_vwap']['detail'] = vv['detail']
            else:
                scores['volume_vwap']['score'] = 50
                scores['volume_vwap']['detail'] = 'not_loaded'
        except Exception as e:
            scores['volume_vwap']['score'] = 50
            scores['volume_vwap']['detail'] = f"error({e})"

        # ═══ ГОЛОС 7: VSA - Volume Spread Analysis (10%) ═══════════════════
        try:
            from vsa_analyzer import analyze_volume_spread
            if candles_5m and len(candles_5m) > 20:
                vsa = analyze_volume_spread(candles_5m)
                if is_short:
                    # ⚡ ШОРТ: инверсия VSA
                    # bearish → сила продавцов → шорт-сигнал
                    # bullish → накопление → не шорт
                    if vsa.signal == 'bearish':
                        vsa_score = 60 + int(vsa.strength * 40)  # 60-100 (было 10-40)
                        vsa_detail = f"{vsa.detail} (short-boost)"
                    elif vsa.signal == 'bullish':
                        vsa_score = 40 - int(vsa.strength * 30)  # 10-40 (было 60-100)
                        vsa_detail = f"{vsa.detail} (short-penalty)"
                    else:
                        vsa_score = 50
                        vsa_detail = vsa.detail
                    scores['vsa']['score'] = vsa_score
                    scores['vsa']['detail'] = vsa_detail
                else:
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

        # Сохраняем VSA состояние для RP модуля
        _rp_vsa_signal = 'neutral'
        _rp_vsa_strength = 0.0
        _rp_vsa_divergence = 0.0
        try:
            if vsa is not None:
                _rp_vsa_signal = vsa.signal if hasattr(vsa, 'signal') else 'neutral'
                _rp_vsa_strength = vsa.strength if hasattr(vsa, 'strength') else 0.0
                _rp_vsa_divergence = vsa.volume_divergence if hasattr(vsa, 'volume_divergence') else 0.0
        except Exception:
            pass

        # ═══ ГОЛОС 8: CVD Order Flow (10%) — реальный поток агрессивных сделок 🆕 ════
        try:
            cvd_score = 50
            cvd_detail = 'N/A'
            if cvd_data and cvd_data.get('minutes', 0) >= 2:
                trend = cvd_data.get('trend', 'neutral')
                buy_pct = cvd_data.get('buy_pct', 50)
                cvd_ratio = cvd_data.get('cvd_ratio', 1.0)
                
                if is_short:
                    # ⚡ ШОРТ: инверсия CVD
                    # sell-поток (buy_pct < 45%) → шорт-сигнал
                    # buy-поток (buy_pct > 55%) → не шорт
                    # buy_pct 50% = нейтрально, 40% = бычий для шорта
                    cvd_score = 50 + (50 - buy_pct) * 1.5  # 50→50, 40→65, 60→35
                    cvd_score = max(10, min(100, cvd_score))
                    
                    if trend == 'bearish':
                        cvd_score += 10  # bearish trend → подтверждение шорта
                    elif trend == 'bullish':
                        cvd_score -= 10  # bullish trend → против шорта
                    
                    if cvd_ratio < 0.67:
                        cvd_score += 5  # много продаж → шорт
                    elif cvd_ratio > 1.5:
                        cvd_score -= 5  # много покупок → не шорт
                    
                    cvd_score = max(10, min(100, cvd_score))
                    cvd_detail = f'trend={trend} buy={buy_pct:.0f}% ratio={cvd_ratio:.2f} (short-inverted)'
                else:
                    # Основа: buy_pct (50% = нейтрально, 60%+ = бычий, 40%- = медвежий)
                    cvd_score = 50 + (buy_pct - 50) * 1.5
                    cvd_score = max(10, min(100, cvd_score))
                    
                    # Коррекция по тренду
                    if trend == 'bullish':
                        cvd_score += 10
                    elif trend == 'bearish':
                        cvd_score -= 10
                    
                    # Если ratio сильно несбалансирован
                    if cvd_ratio > 1.5:
                        cvd_score += 5  # много покупок
                    elif cvd_ratio < 0.67:
                        cvd_score -= 5  # много продаж
                    
                    cvd_score = max(10, min(100, cvd_score))
                    cvd_detail = f'trend={trend} buy={buy_pct:.0f}% ratio={cvd_ratio:.2f}'
                
                logger.debug(f"[CVD] {symbol}: score={cvd_score:.0f} — {cvd_detail}")
            
            scores['cvd']['score'] = cvd_score
            scores['cvd']['detail'] = cvd_detail
        except Exception as e:
            scores['cvd']['score'] = 50
            scores['cvd']['detail'] = f'error({e})'

        # ═══ RP (Recovery & Potential) — мета-дельта −20..+20 🆕 ════════
        try:
            from rp_analyzer import analyze_recovery_potential
            rp_result = analyze_recovery_potential(
                symbol=symbol,
                current_price=current_price,
                high_24h=high_24h,
                low_24h=low_24h,
                cvd_data=cvd_data,
                vsa_signal=_rp_vsa_signal,
                vsa_strength=_rp_vsa_strength,
                vsa_divergence=_rp_vsa_divergence,
            )
            rp_delta = rp_result['rp_delta']
            scores['rp'] = {
                'score': rp_result['rp_score'],
                'weight': 0,  # RP — дельта, не взвешенный голос
                'delta': rp_delta,
                'detail': rp_result['detail'],
            }
            logger.debug(f"[RP] {symbol}: score={rp_result['rp_score']:.0f} delta={rp_delta:+d} — {rp_result['detail']}")
        except Exception as e:
            logger.debug(f"[RP] {symbol}: error: {e}")
            scores['rp'] = {'score': 50, 'weight': 0, 'delta': 0, 'detail': f'error({e})'}

        # ═══ АДАПТИВНЫЕ ВЕСА ══════════════════════════════════════════════
        # Если Liq сильный (>70) а MTF слабый (<50) — снижаем вес MTF
        liq_score = scores['liquidity']['score']
        mtf_score = scores['mtf']['score']
        adv_score = scores['advisor']['score']
        vsa_score = scores['vsa']['score']

        if liq_score >= 75 and mtf_score < 50:
            # Liq уверен, MTF не подтверждает - не даём MTF душить сигнал
            mtf_shrink = 0.5 if adv_score >= 60 else 0.3
            if vsa_score >= 60:
                mtf_shrink = max(mtf_shrink, 0.4)  # VSA подтверждает - ещё ослабляем MTF
            transfer = BASE_WEIGHTS['mtf'] * mtf_shrink
            scores['mtf']['weight'] = BASE_WEIGHTS['mtf'] - transfer
            scores['liquidity']['weight'] = BASE_WEIGHTS['liquidity'] + transfer * 0.6
            scores['vsa']['weight'] = BASE_WEIGHTS['vsa'] + transfer * 0.4
            logger.debug(f"[DE] Адаптивные веса: MTF {BASE_WEIGHTS['mtf']:.2f}→{scores['mtf']['weight']:.2f}, Liq {BASE_WEIGHTS['liquidity']:.2f}→{scores['liquidity']['weight']:.2f}")
        elif liq_score >= 60 and adv_score >= 80 and mtf_score < 50:
            # Advisor сильный + Liq умеренный - немного ослабляем MTF
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
                                # Чем сильнее зелёных и объём - тем больше бонус
                                strength = min(green_count * 3 +
                                              int((last3_vol / prev3_vol - 1) * 5), 15)
                                reversal_bonus = min(strength, 15)
                    except Exception:
                        pass
        except Exception:
            pass
        scores['_reversal_bonus'] = {'score': reversal_bonus, 'weight': 1.0}
        scores['_price_bonus'] = {'score': price_bonus, 'weight': 1.0}

        # ═══ BTC DIRECTION PREDICTOR ═══════════════════════════════════════
        # Вычисляем ДО final_score, чтобы штраф/бонус BTC влиял на итоговую оценку
        btc_bonus = 0
        threshold = 0  # Инициализация на случай раннего return
        try:
            if self._btc_predictor is None:
                from btc_direction import BTCDirectionPredictor
                self._btc_predictor = BTCDirectionPredictor()
            # Предварительный score (без бонуса) для расчёта масштаба штрафа
            preliminary_score = sum(v['score'] * v['weight'] for v in scores.values())
            btc_bonus = self._btc_predictor.calculate_bonus(preliminary_score)
            scores['_btc_bonus'] = {'score': btc_bonus, 'weight': 1.0}
        except Exception as e:
            scores['_btc_bonus'] = {'score': 0, 'weight': 1.0}
            logger.debug(f"BTC Direction: ошибка: {e}")

        # ═══ ШТРАФ ЗА БЛИЗОСТЬ К 24H ХАЮ ═════════════════════════════════
        if high_24h is not None and high_24h > 0 and current_price > 0:
            _ratio = current_price / high_24h
            if _ratio >= 0.96:
                _penalty = min(25, (_ratio - 0.96) * 625)  # 0.96→0, 1.0→25
                scores['_level_penalty'] = {'score': -_penalty, 'weight': 1.0}
                logger.info(f"📏 [LEVEL PENALTY] {symbol}: цена {_ratio:.1%} от 24h хая — штраф -{_penalty:.0f}pts")
            else:
                scores['_level_penalty'] = {'score': 0, 'weight': 1.0}
                if _ratio >= 0.85:
                    _small_penalty = (_ratio - 0.85) * 66
                    scores['_level_penalty'] = {'score': -_small_penalty, 'weight': 1.0}

        # ═══ ИТОГОВЫЙ СКОР ═════════════════════════════════════════════════
        # Теперь в scores есть все компоненты, включая btc_bonus
        final_score = sum(v['score'] * v['weight'] for v in scores.values())
        # RP дельта добавляется к финальному скору (weight=0, не влияет на сумму)
        _rp_delta = scores.get('rp', {}).get('delta', 0)
        if _rp_delta != 0:
            og_score = final_score
            final_score += _rp_delta
            final_score = max(0, min(150, final_score))
            logger.debug(f"[RP] {symbol}: дельта={_rp_delta:+d}pts ({og_score:.1f}→{final_score:.1f})")

        # ═══ БЛОКИРОВКА ОТ BTC ═══════════════════════════════════════════
        # Если btc_bonus == -999 - жёсткое veto на лонги (BTC падает >2% за 6ч)
        if btc_bonus <= -999:
            return {
                'approved': False,
                'reason': 'VETO: BTC падает >2% за 6ч',
                'final_score': 0,
                'threshold': 0,
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

        logger.debug(f"[threshold] symbol={symbol} threshold={threshold} strong={strong_votes} liq={scores['liquidity']['score']} adv={scores['advisor']['score']} vsa={scores['vsa']['score']} final_score={final_score}")

        # Разбивка голосов для логов/дашборда
        votes_str = (
            f"ML-Pro:{scores['ml_v2']['score']:.0f}({scores['ml_v2']['detail']}) "
            f"Adv:{scores['advisor']['score']:.0f}({scores['advisor']['detail']}) "
            f"MTF:{scores['mtf']['score']:.0f}({scores['mtf']['detail']}) "
            f"RVB:{scores['rsi_vol_btc']['score']:.0f}({scores['rsi_vol_btc']['detail']}) "
            f"Liq:{scores['liquidity']['score']:.0f}({scores['liquidity']['detail']}) "
            f"VV:{scores['volume_vwap']['score']:.0f}({scores['volume_vwap']['detail']}) "
            f"VSA:{scores['vsa']['score']:.0f}({scores['vsa']['detail']}) "
            f"CVD:{scores['cvd']['score']:.0f}({scores['cvd']['detail']}) "
            f"RP:{scores.get('rp', {}).get('delta', 0):+d}({scores.get('rp', {}).get('detail', 'N/A')})"
        )

        votes = {
            'ml_v2': {'score': scores['ml_v2']['score'], 'detail': scores['ml_v2']['detail']},
            'advisor': {'score': scores['advisor']['score'], 'detail': scores['advisor']['detail']},
            'mtf': {'score': scores['mtf']['score'], 'detail': scores['mtf']['detail']},
            'rsi_vol_btc': {'score': scores['rsi_vol_btc']['score'], 'detail': scores['rsi_vol_btc']['detail']},
            'liquidity': {'score': scores['liquidity']['score'], 'detail': scores['liquidity']['detail']},
            'volume_vwap': {'score': scores['volume_vwap']['score'], 'detail': scores['volume_vwap']['detail']},
            'vsa': {'score': scores['vsa']['score'], 'detail': scores['vsa']['detail']},
            'cvd': {'score': scores['cvd']['score'], 'detail': scores['cvd']['detail']},
            'rp': {'score': scores.get('rp', {}).get('score', 50),
                   'delta': scores.get('rp', {}).get('delta', 0),
                   'detail': scores.get('rp', {}).get('detail', 'N/A')},
            'level': {'score': scores.get('_level_penalty', {}).get('score', 0),
                      'detail': f'{current_price/high_24h*100:.1f}% от 24h хая' if high_24h else 'N/A'},
        }

        # 🧠 Сохраняем голоса для VoteTracker (для учёта точности после закрытия сделки)
        if final_score >= threshold:
            try:
                get_vote_tracker().record_entry_votes(
                    symbol, votes, current_price,
                    datetime.now(timezone.utc).isoformat()
                )
            except Exception as _ve:
                logger.debug(f"[DE] VoteTracker entry save error: {_ve}")

        # ═══ СОХРАНЕНИЕ ИСТОРИИ ГОЛОСОВ ════════════════════════════════
        try:
            now = time.time()
            # Не чаще раза в 3 секунды для той же монеты
            if not hasattr(self, '_last_vote_ts'):
                self._last_vote_ts = {}
            last_ts = self._last_vote_ts.get(symbol, 0)
            if now - last_ts < 3.0:
                pass  # слишком часто - пропускаем запись на диск
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
                import tempfile
                history = []
                if os.path.exists(vote_log_path):
                    try:
                        with open(vote_log_path, 'r') as f:
                            history = json.load(f)
                    except Exception as _e:
                        # Файл битый - создаём новый
                        history = []
                history.append(vote_record)
                # Держим последние 10000 записей
                if len(history) > 10000:
                    history = history[-10000:]
                # Атомарная запись: сначала во временный файл, потом rename
                tmp_path = vote_log_path + '.tmp'
                with open(tmp_path, 'w') as f:
                    json.dump(history, f, indent=2, default=str)
                os.replace(tmp_path, vote_log_path)
        except Exception as e:
            logger.debug(f"[DE] vote_history save error: {e}")
        # ════════════════════════════════════════════════════════════════

        # ═══ ЛОГИРОВАНИЕ РЕШЕНИЯ В signal_log ════════════════════════
        try:
            from db_pg import log_signal
            log_signal(
                symbol=symbol,
                decision='buy' if final_score >= threshold else 'hold',
                score=round(final_score, 1),
                threshold=threshold,
                components={
                    'ml_pro': scores['ml_v2']['score'],
                    'advisor': scores['advisor']['score'],
                    'mtf': scores['mtf']['score'],
                    'rsi_vol_btc': scores['rsi_vol_btc']['score'],
                    'liquidity': scores['liquidity']['score'],
                    'volume_vwap': scores['volume_vwap']['score'],
                    'vsa': scores['vsa']['score'],
                    'cvd': scores['cvd']['score'],
                    'bonus': {'price': round(price_bonus, 1), 'reversal': round(reversal_bonus, 1), 'btc': round(btc_bonus, 1)},
                    'final_score': round(final_score, 1),
                    'threshold': threshold,
                    'strong_votes': strong_votes,
                }
            )
        except Exception:
            pass
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
                'vsa_score': scores['vsa']['score'],
                'cvd_score': scores['cvd']['score'],
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

    def record_exit(self, symbol: str, reason: str = "", was_loss: bool = False) -> None:
        """Запомнить момент выхода (для кулдауна повторного входа).

        Args:
            was_loss: True если сделка закрылась в минус
        """
        self._last_decisions[symbol] = {
            'exit_time': time.time(),
            'reason': reason,
        }

        # Обновляем счётчик последовательных убытков
        if was_loss:
            self._consecutive_losses[symbol] = self._consecutive_losses.get(symbol, 0) + 1
            losses = self._consecutive_losses[symbol]
            if losses >= self.max_consecutive_losses:
                block_h = losses * 4
                logger.warning(f"⚠️ {symbol}: {losses} убытков подряд → блокировка на {block_h}ч")
            else:
                logger.info(f"⚠️ {symbol}: {losses}-й убыток подряд")
        else:
            # На прибыли - не сбрасываем полностью, а уменьшаем на 1
            # (чтобы одна удачная сделка не обнуляла историю ММ-паттерна)
            old = self._consecutive_losses.get(symbol, 0)
            if old > 0:
                new_val = max(0, old - 1)
                if new_val == 0:
                    self._consecutive_losses.pop(symbol, None)
                else:
                    self._consecutive_losses[symbol] = new_val
                logger.info(f"✅ {symbol}: серия убытков снижена с {old} до {new_val} (был профит)")

        # Сохраняем счётчик в файл (переживает перезагрузки)
        self._save_losses()

        # Передаём память в MLAdvisor (чтобы XGBoost учился на этом)
        self._push_memory_to_ml()

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

    def _maybe_restore_losses(self) -> None:
        """Загрузить счётчик убытков из файла при перезапуске."""
        try:
            if os.path.exists(self._losses_file):
                with open(self._losses_file, 'r') as f:
                    data = json.load(f)
                now = time.time()
                restored = 0
                for sym, info in data.items():
                    losses = info.get('losses', 0)
                    saved_at = info.get('saved_at', 0)
                    if losses > 0:
                        # Проверяем, не истекла ли блокировка
                        block_hours = losses * 4
                        if now - saved_at < block_hours * 3600:
                            self._consecutive_losses[sym] = losses
                            restored += 1
                if restored > 0:
                    logger.info(f"♻️ Восстановлены счётчики убытков для {restored} символов")
        except Exception as e:
            logger.debug(f"[restore_losses] {e}")

    def _save_losses(self) -> None:
        """Сохранить счётчик убытков в файл."""
        try:
            data = {}
            now = time.time()
            for sym, losses in self._consecutive_losses.items():
                data[sym] = {'losses': losses, 'saved_at': now}
            os.makedirs(os.path.dirname(self._losses_file), exist_ok=True)
            with open(self._losses_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"[save_losses] {e}")

    def cleanup_old_decisions(self, max_age_sec: int = 7200) -> int:
        """Удалить exit-time записи старше max_age_sec (предотвращает утечку)."""
        now = time.time()
        keys = list(self._last_decisions.keys())
        removed = 0
        for sym in keys:
            exit_time = self._last_decisions[sym].get('exit_time', 0)
            if now - exit_time > max_age_sec:
                del self._last_decisions[sym]
                # Также чистим счётчик убытков, если блокировка уже истекла
                losses = self._consecutive_losses.get(sym, 0)
                if losses > 0:
                    hard_block_sec = losses * 4 * 3600
                    if now - exit_time > hard_block_sec:
                        self._consecutive_losses.pop(sym, None)
                removed += 1
        return removed

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
            'consecutive_losses': dict(self._consecutive_losses),
        }

    # ═══════════════════════════════════════════════════════════════════════
    # ⚡ ШОРТ: решение о входе в короткую позицию
    # ═══════════════════════════════════════════════════════════════════════

    def decide_short(self, symbol: str, confidence: float, trend: str,
                      rsi: float, current_price: float,
                      current_positions_count: int, max_short_positions: int = 10,
                      candles_5m: Optional[list] = None,
                      candles_1h: Optional[list] = None,
                      candles_4h: Optional[list] = None,
                      last_exit_price: Optional[float] = None,
                      last_exit_time: Optional[float] = None,
                      cvd_data: Optional[Dict] = None,
                      high_24h: Optional[float] = None,
                      low_24h: Optional[float] = None,
                      btc_state: Optional[Dict] = None) -> Decision:
        """
        Принять решение о входе в короткую позицию.

        ⚡ ШОРТ-ФИЛЬТР 1: BTC не в up-trend/accumulation
           Если BTC в up-trend — шорт заблокирован.
           BTC в down-trend/distribution — шорт разрешён.

        ⚡ ШОРТ-ФИЛЬТР 2: VETO при strong_bullish на 1H/4H
           (делегировано в decide_entry с side='short')

        ⚡ ШОРТ-ФИЛЬТР 3: не шортим на хаях 24h (LVL penalty работает)
           (делегировано в _evaluate_entry_ensemble)

        Использует decide_entry(side='short') с сигнальной инверсией.
        """
        self._lazy_init_ml()

        # ═══ BTC-FILTER: проверяем, можно ли шортить ───────────────────
        btc_blocked = False
        btc_reason = ''
        if btc_state:
            btc_trend = btc_state.get('trend', 'neutral')
            btc_regime = btc_state.get('regime', 'unknown')
            if btc_trend in ('up', 'rising') or btc_regime == 'accumulation':
                btc_blocked = True
                btc_reason = f"BTC {btc_trend}/{btc_regime} — шорт заблокирован"
                logger.info(f"🚫 [SHORT BLOCKED] {symbol}: {btc_reason}")

        # Проверяем также через _btc_price
        if not btc_blocked and self._btc_price and self._btc_reference_price:
            btc_change_4h = (self._btc_price - self._btc_reference_price) / self._btc_reference_price * 100
            if btc_change_4h > 1.0:
                btc_blocked = True
                btc_reason = f"BTC +{btc_change_4h:.1f}% за 4ч — шорт заблокирован"
                logger.info(f"🚫 [SHORT BLOCKED] {symbol}: {btc_reason}")

        if btc_blocked:
            self._save_veto_vote(symbol, current_price, btc_reason, side='short')
            return Decision(symbol, 'hold', side='short', reason=btc_reason)

        # ═══ ЛИМИТ ШОРТ-ПОЗИЦИЙ ───────────────────────────────────────
        if current_positions_count >= max_short_positions:
            reason = f"Максимум {max_short_positions} шорт-позиций"
            self._save_veto_vote(symbol, current_price, reason, side='short')
            return Decision(symbol, 'hold', side='short', reason=reason)

        # ═══ КУЛДАУН для шортов ───────────────────────────────────────
        now = time.time()
        _short_cooldown_key = f"{symbol}_short"
        if _short_cooldown_key in self._last_decisions:
            last_exit = self._last_decisions[_short_cooldown_key]
            time_since = now - last_exit.get('exit_time', 0)

            losses = self._consecutive_short_losses.get(symbol, 0)
            if losses >= self.max_consecutive_short_losses:
                hard_block_hours = losses * 4
                hard_block_sec = hard_block_hours * 3600
                if time_since < hard_block_sec:
                    remain = hard_block_sec - time_since
                    reason = f"🚫 {losses} шорт-убытка подряд, блокировка {hard_block_hours}ч (осталось {remain/3600:.1f}ч)"
                    self._save_veto_vote(symbol, current_price, reason, side='short')
                    return Decision(symbol, 'hold', side='short', reason=reason)

            if time_since < self.reentry_cooldown:
                remain = self.reentry_cooldown - time_since
                reason = f"Повторный шорт через {remain/60:.1f} мин"
                self._save_veto_vote(symbol, current_price, reason, side='short')
                return Decision(symbol, 'hold', side='short', reason=reason)

        # ═══ ВЫЗОВ decide_entry с side='short' для ансамбля ─────────
        return self.decide_entry(
            symbol, confidence, trend, rsi, current_price,
            current_positions_count=current_positions_count,
            max_positions=max_short_positions,
            candles_5m=candles_5m,
            candles_1h=candles_1h,
            candles_4h=candles_4h,
            side='short',
            last_exit_price=last_exit_price,
            last_exit_time=last_exit_time,
            cvd_data=cvd_data,
            high_24h=high_24h,
            low_24h=low_24h
        )

    def reset_cooldowns(self) -> None:
        """Сбросить все тайм-ауты."""
        self._last_decisions.clear()
        self._consecutive_losses.clear()
        self._consecutive_short_losses.clear()
        logger.info("🔄 Кулдауны сброшены")
