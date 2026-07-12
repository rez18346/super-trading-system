#!/usr/bin/env python3
"""
ML-PRO v2 — Профессиональный ML-советник для входа в сделки
=============================================================
Основан на лучших практиках DeepAlpha + Intelligent Trading Bot.
Адаптирован для спот-торговли на Bybit (22 пары, 5M-1H-4H-D1).

Версия 2.7 — мультитаймфреймовый анализ (1H+4H), 25 признаков, 72% accuracy.

Приоритет моделей:
  1. ml_pro_v2_27f.pkl (25 признаков, 1H+4H, 55K+ образцов, 72% acc) ← НОВЫЙ
  2. ml_pro_v2_1h.pkl  (16 признаков, 1H, 526K образцов, 54% acc)
  3. ml_pro_v2.pkl     (16 признаков, 5M, синтетика, fallback)
"""

import json
import os
import pickle
import time
import logging
import numpy as np
from datetime import datetime

log = logging.getLogger("ML_PRO_V2")

# ─── Конфигурация ──────────────────────────────────────────────────────────
SKIP_THRESHOLD = 0.45  # стандартный, может меняться HMM
BUY_THRESHOLD = 0.55   # стандартный, может меняться HMM
MIN_TRAIN_SAMPLES = 50

# Константы для дообучения 27f модели
RETRAIN_INTERVAL_27F = 3600       # раз в час
MIN_SAMPLES_27F = 50              # минимум примеров для ретрейна
MIN_BOTH_CLASSES_27F = 10         # минимум каждого класса

# HMM-режимы (инициализируем лениво)
_regime = None

def _get_regime():
    global _regime
    if _regime is None:
        try:
            from hmm_regime import get_regime
            _regime = get_regime()
            log.info(f"[ML-v2] HMM-режим: {_regime.get_state_name()}")
        except Exception as e:
            log.warning(f"[ML-v2] HMM не подключился: {e}")
            _regime = None
    return _regime

def _get_dynamic_thresholds():
    """Вернуть пороги с учётом HMM-режима."""
    r = _get_regime()
    if r and r.trained:
        t = r.get_thresholds()
        return t['buy'], t['skip']
    return BUY_THRESHOLD, SKIP_THRESHOLD

# ─── Имя для 16-признаковой модели (старая) ──────────────────────────────
FEATURE_NAMES_16 = [
    'rsi', 'trend', 'atr_pct', 'atr_ratio',
    'ema12_dist', 'ema26_dist', 'sma20_dist', 'vwap_dist',
    'vol_ratio', 'mom_3h', 'mom_7h',
    'candle_body', 'pinbar', 'engulfing', 'd_24h_high', 'rsi_div'
]

# ─── Имена для 25-признаковой модели (новая, 27f) ────────────────────────
FEATURE_NAMES_25 = [
    'rsi_1h','rsi_4h',
    'trend_1h','trend_4h','trend_aligned',
    'atr_1h','atr_4h','atr_ratio',
    'pv_ema12','pv_ema26',
    'sma20_d','sma50_d',
    'volr_1h','volr_4h','vwap_1h',
    'mom1','mom3','mom7','mom24',
    'candle_body','pinbar','engulfing','d24h','rsi_div',
    'hour_of_day'
]

# НОВЫЕ признаки (8 шт., индексы 25-32)
FEATURE_NAMES_33 = FEATURE_NAMES_25 + [
    'tf_aligned',        # 25: 1H и 4H тренды совпадают
    'volatility_ratio',  # 26: 4H / 1H волатильность
    'force_index',       # 27: Force Index (объём × Δцены)
    'eom',               # 28: Ease of Movement
    'rsi_div_5',         # 29: RSI дивергенция за 5 свечей
    'vol_spread',        # 30: разброс объёма (max/mean)
    'wick_body_ratio',   # 31: тени / тело
    'ema_slope_6h',      # 32: наклон EMA12 за 6 часов
]


# ============================================================================
# Вспомогательные функции (vectorised для производительности)
# ============================================================================

def _rsi(closes, period=14):
    r = np.full_like(closes, 50.0, dtype=float)
    pc = np.diff(closes, prepend=closes[0])
    for i in range(period, len(closes)):
        g = np.maximum(pc[i - period + 1:i + 1], 0).mean()
        lo = np.maximum(-pc[i - period + 1:i + 1], 0).mean()
        r[i] = 100.0 - (100.0 / (1.0 + g / lo)) if lo != 0 else 100.0
    return r


def _atr(highs, lows, closes, period=14):
    a = np.zeros_like(closes, dtype=float)
    tr = np.zeros_like(closes, dtype=float)
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
    for i in range(period, len(closes)):
        a[i] = tr[i - period + 1:i + 1].mean()
    return a


def _ema(data, period):
    e = np.zeros_like(data, dtype=float)
    e[0] = data[0]
    k = 2.0 / (period + 1)
    for i in range(1, len(data)):
        e[i] = data[i] * k + e[i - 1] * (1 - k)
    return e


def _trend(closes, period=20):
    tr = np.zeros(len(closes), dtype=float)
    for i in range(period, len(closes)):
        sn = closes[i - period + 1:i + 1].mean()
        sp = closes[i - period:i].mean()
        sl = (sn - sp) / sp if sp > 0 else 0
        tr[i] = 1.0 if sl > 0.001 else (-1.0 if sl < -0.001 else 0)
    return tr


def _vwap(highs, lows, closes, volumes, period=14):
    vw = np.zeros(len(closes), dtype=float)
    for i in range(period, len(closes)):
        vs = volumes[i - period:i].sum()
        if vs > 0:
            typ = (highs[i - period:i] + lows[i - period:i] + closes[i - period:i]) / 3
            vw[i] = (typ * volumes[i - period:i]).sum() / vs
    return vw


def _detect_pinbar(op, hi, lo, cl):
    body = abs(cl - op)
    total = hi - lo
    if total == 0 or body == 0:
        return 0.0
    uw = hi - max(op, cl)
    lw = min(op, cl) - lo
    if lw > body * 2 and uw < body * 0.5:
        return 1.0
    if uw > body * 2 and lw < body * 0.5:
        return -1.0
    return 0.0


def _detect_engulfing(o1, c1, o2, c2):
    b1 = abs(c1 - o1)
    b2 = abs(c2 - o2)
    if b1 == 0 or b2 == 0:
        return 0.0
    if c1 < o1 and c2 > o2 and b2 > b1 * 1.3 and o2 < c1 and c2 > o1:
        return 1.0
    if c1 > o1 and c2 < o2 and b2 > b1 * 1.3 and c2 < o1 and o2 > c1:
        return -1.0
    return 0.0


# ============================================================================
# Feature Engineering — 25 признаков напрямую из 1H+4H свечей
# ============================================================================

def _listify(candles):
    """Преобразовать DataFrame в list[dict] (если ещё не list)."""
    if hasattr(candles, 'iloc'):
        return candles.to_dict('records') if hasattr(candles, 'to_dict') else []
    return candles


def _normalize_candle_dict(d):
    """Привести словарь свечи к формату {o,h,l,c,v,t} из любого входа."""
    if 'o' in d:
        return d  # уже нормализован
    return {
        'o': d.get('open', d.get('o', 0)),
        'h': d.get('high', d.get('h', 0)),
        'l': d.get('low', d.get('l', 0)),
        'c': d.get('close', d.get('c', 0)),
        'v': d.get('volume', d.get('v', 0)),
        't': d.get('timestamp', d.get('t', d.get('timestamp', 0))),
    }


def build_features_27f(candles_1h, candles_4h):
    """
    Построить 25 признаков из 1H и 4H свечей.

    candles_1h : list[dict] с {open,high,low,close,volume,timestamp} ИЛИ {o,h,l,c,v,t}
    candles_4h : list[dict] — аналогично

    Возвращает np.ndarray shape (N, 25) — только для полных рядов.
    """
    # Нормализация ключей (из to_dict('records') или Raw OHLCV)
    candles_1h = [_normalize_candle_dict(c) for c in candles_1h]
    candles_4h = [_normalize_candle_dict(c) for c in candles_4h]

    # Синхронизируем по времени: 1H должны покрывать 4H
    if candles_1h and candles_4h:
        t4s = candles_4h[0].get('t', 0)
        t4e = candles_4h[-1].get('t', 0)
        candles_1h = [c for c in candles_1h if c.get('t', 0) >= t4s and c.get('t', 0) <= t4e]
        # 4H обрезаем до 1H
        t1s = candles_1h[0].get('t', 0) if candles_1h else 0
        t1e = candles_1h[-1].get('t', 0) if candles_1h else 0
        candles_4h = [c for c in candles_4h if c.get('t', 0) >= t1s and c.get('t', 0) <= t1e]
    
    if not candles_1h or not candles_4h or len(candles_1h) < 100 or len(candles_4h) < 25:
        return np.zeros((1, 33), dtype=np.float64)

    # Парсинг
    o1 = np.array([c['o'] for c in candles_1h], dtype=float)
    h1 = np.array([c['h'] for c in candles_1h], dtype=float)
    l1 = np.array([c['l'] for c in candles_1h], dtype=float)
    cl1 = np.array([c['c'] for c in candles_1h], dtype=float)
    v1 = np.array([c['v'] for c in candles_1h], dtype=float)
    t1 = np.array([c['t'] for c in candles_1h], dtype=float)

    o4 = np.array([c['o'] for c in candles_4h], dtype=float)
    h4 = np.array([c['h'] for c in candles_4h], dtype=float)
    l4 = np.array([c['l'] for c in candles_4h], dtype=float)
    cl4 = np.array([c['c'] for c in candles_4h], dtype=float)
    v4 = np.array([c['v'] for c in candles_4h], dtype=float)
    t4 = np.array([c['t'] for c in candles_4h], dtype=float)

    # Индикаторы
    rsi1 = _rsi(cl1)
    rsi4 = _rsi(cl4)
    atr1 = _atr(h1, l1, cl1)
    atr4 = _atr(h4, l4, cl4)
    em12 = _ema(cl1, 12)
    em26 = _ema(cl4, 26)
    tr1 = _trend(cl1)
    tr4 = _trend(cl4)
    vw1 = _vwap(h1, l1, cl1, v1)

    # SMA
    sma20 = np.zeros(len(cl1), dtype=float)
    sma50 = np.zeros(len(cl1), dtype=float)
    for i in range(20, len(cl1)):
        sma20[i] = cl1[i - 20:i].mean()
    for i in range(50, len(cl1)):
        sma50[i] = cl1[i - 50:i].mean()

    # Volume MA
    vma1 = np.ones(len(cl1), dtype=float)
    vma4 = np.ones(len(cl4), dtype=float)
    for i in range(20, len(cl1)):
        m = v1[i - 20:i].mean()
        vma1[i] = m if m > 0 else 1.0
    for i in range(20, len(cl4)):
        m = v4[i - 20:i].mean()
        vma4[i] = m if m > 0 else 1.0

    # Индексы 4H → 1H
    idx4 = np.clip(np.searchsorted(t4, t1, side='right') - 1, 0, len(cl4) - 1)

    n = len(cl1)
    X = np.zeros((n, 33), dtype=np.float64)

    for i in range(50, n):
        j4 = idx4[i]
        if j4 < 5:
            continue

        # 0-1: RSI
        X[i, 0] = rsi1[i]
        X[i, 1] = rsi4[j4]

        # 2-4: Тренды
        X[i, 2] = tr1[i]
        X[i, 3] = tr4[j4]
        if X[i, 2] > 0 and X[i, 3] > 0:
            X[i, 4] = 1.0
        elif X[i, 2] < 0 and X[i, 3] < 0:
            X[i, 4] = -1.0

        # 5-7: ATR
        X[i, 5] = atr1[i] / cl1[i] * 100 if cl1[i] > 0 else 0
        X[i, 6] = atr4[j4] / cl4[j4] * 100 if cl4[j4] > 0 else 0
        if i >= 48:
            as_ = atr1[i - 5:i].mean()
            al = atr1[i - 48:i].mean()
            X[i, 7] = as_ / al if al > 0 else 1.0

        # 8-9: EMA
        if em12[i] > 0:
            X[i, 8] = (cl1[i] - em12[i]) / em12[i] * 100
        if em26[j4] > 0:
            X[i, 9] = (cl4[j4] - em26[j4]) / em26[j4] * 100

        # 10-11: SMA
        if sma20[i] > 0:
            X[i, 10] = (cl1[i] - sma20[i]) / sma20[i] * 100
        if sma50[i] > 0:
            X[i, 11] = (cl1[i] - sma50[i]) / sma50[i] * 100

        # 12-14: Объём
        X[i, 12] = v1[i] / vma1[i] if vma1[i] > 0 else 1.0
        X[i, 13] = v4[j4] / vma4[j4] if vma4[j4] > 0 else 1.0
        if vw1[i] > 0:
            X[i, 14] = (cl1[i] - vw1[i]) / vw1[i] * 100

        # 15-18: Моментум
        if i >= 1:
            X[i, 15] = (cl1[i] - cl1[i - 1]) / cl1[i - 1] * 100
        if i >= 3:
            X[i, 16] = (cl1[i] - cl1[i - 3]) / cl1[i - 3] * 100
        if i >= 7:
            X[i, 17] = (cl1[i] - cl1[i - 7]) / cl1[i - 7] * 100
        if i >= 24:
            X[i, 18] = (cl1[i] - cl1[i - 24]) / cl1[i - 24] * 100

        # 19-21: Свечные паттерны
        body = abs(cl1[i] - o1[i])
        tt = h1[i] - l1[i]
        X[i, 19] = body / tt if tt > 0 else 0
        X[i, 20] = _detect_pinbar(o1[i], h1[i], l1[i], cl1[i])
        if i >= 1:
            X[i, 21] = _detect_engulfing(o1[i - 1], cl1[i - 1], o1[i], cl1[i])

        # 22-23: Риск
        h24 = h1[max(0, i - 24):i + 1].max()
        X[i, 22] = (cl1[i] - h24) / h24 * 100 if h24 > 0 else 0
        if i >= 14:
            if cl1[i] > cl1[i - 14] and rsi1[i] < rsi1[i - 14] - 5:
                X[i, 23] = -1.0
            elif cl1[i] < cl1[i - 14] and rsi1[i] > rsi1[i - 14] + 5:
                X[i, 23] = 1.0

        # 24: Час дня (UTC)
        X[i, 24] = datetime.fromtimestamp(t1[i] / 1000).hour

    # ═══════════════════════════════════════════════════════════════════════
    # НОВЫЕ ПРИЗНАКИ (индексы 25-32, напрямую в X)
    # ═══════════════════════════════════════════════════════════════════════
    
    for i in range(50, n):
        j4 = idx4[i]
        if j4 < 5:
            continue
        
        # 25: Корреляция 1H → 4H (тренд совпадает?)
        X[i, 25] = 1.0 if (tr1[i] > 0 and tr4[j4] > 0) or (tr1[i] < 0 and tr4[j4] < 0) else 0.0
        
        # 26: Волатильность 4H / волатильность 1H
        if i >= 54:
            v1h = atr1[i-5:i].std()
            v4h = atr4[j4-1:j4+1].std() if j4 >= 1 else 0
            X[i, 26] = v4h / (v1h + 1e-8)
        
        # 27: Force Index (объём × изменение цены)
        if i >= 13:
            fi_list = [(cl1[k] - cl1[k-1]) * v1[k] for k in range(i-12, i+1) if k >= 1]
            if fi_list:
                fi_ma13 = np.mean(fi_list)
                X[i, 27] = fi_ma13 / (cl1[i] + 1e-8) * 100
        
        # 28: EOM (Ease of Movement)
        if i >= 1:
            mid_move = (h1[i] + l1[i]) / 2 - (h1[i-1] + l1[i-1]) / 2
            tr0 = h1[i] - l1[i]
            box_ratio = v1[i] / (tr0 * 1e6 + 1e-8) if tr0 > 0 else 0
            X[i, 28] = mid_move * box_ratio * 1000
        
        # 29: RSI дивергенция за 5 свечей
        if i >= 19:
            if cl1[i] < cl1[i-5] and rsi1[i] > rsi1[i-5]:
                X[i, 29] = 1.0
            elif cl1[i] > cl1[i-5] and rsi1[i] < rsi1[i-5]:
                X[i, 29] = -1.0
        
        # 30: Volume spread (max/mean за 12ч)
        if i >= 12:
            vol_last_12 = v1[max(0,i-11):i+1]
            vol_spread = vol_last_12.max() / (vol_last_12.mean() + 1e-8)
            X[i, 30] = min(vol_spread, 5.0)
            
        # 31: wick-to-body ratio
        body_i = abs(cl1[i] - o1[i])
        tt_i = h1[i] - l1[i]
        if body_i > 0 and tt_i > 0:
            upper_wick = h1[i] - max(o1[i], cl1[i])
            lower_wick = min(o1[i], cl1[i]) - l1[i]
            X[i, 31] = (upper_wick + lower_wick) / body_i
        
        # 32: Наклон EMA12 за 6 часов
        if i >= 6 and em12[i] > 0:
            X[i, 32] = (em12[i] - em12[i-6]) / em12[i-6] * 100
    
    return X


# ============================================================================
# Класс ML-PRO v2
# ============================================================================

class MLProfessionalV2:
    """Профессиональный ML-советник."""

    def __init__(self, model_path=None):
        self.model = None
        self.trained = False
        self.training_data = []
        self.feature_importance = None
        self.feature_names = []  # Имена признаков текущей модели
        self.is_27f = False     # Флаг: 27-признаковая модель?
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "models", "ml_pro_v2.pkl")

        # 🧠 Буфер для дообучения на реальных сделках (Professional: храним СВЕЧИ, не фичи)
        self._last_candles = None             # (candles_1h, candles_4h) — последние свечи из evaluate
        self._last_candles_symbol = None      # символ последних свечей
        self._entry_buffer = {}               # {symbol: {'candles_1h': [...], 'candles_4h': [...], 'ts': float}}
        self._training_buffer = []            # [{'candles_1h': [...], 'candles_4h': [...], 'label': 1.0/0.0, 'symbol': str, 'ts': float}]
        self._training_buffer_path = os.path.join(os.path.dirname(self.model_path), 'ml_pro_v2_training.pkl')
        self._last_retrain_27f = 0            # timestamp последнего ретрейна
        
        # 🗂️ Загружаем сохранённый training_buffer (включая исторические сделки)
        buf_was_cleaned = self._load_training_buffer()

        # 1. Пробуем 25-признаковую мультиТФ модель (приоритет)
        v27f = os.path.join(os.path.dirname(__file__), "models", "ml_pro_v2_27f.pkl")
        if os.path.exists(v27f):
            try:
                with open(v27f, "rb") as f:
                    data = pickle.load(f)
                self.model = data["model"]
                self.trained = True
                self.is_27f = True
                acc = data.get("acc", 0)
                n = data.get("samples", 0)
                feats = data.get("features", [])
                self.feature_names = feats if feats else FEATURE_NAMES_25
                self.feature_importance = data.get("importance")
                log.info(f"[ML-v2] ✅ 25-признаковая мультиТФ (1H+4H) ({n:,} обр., {acc:.2%} acc)")
                # Если буфер был очищен (старые багнутые данные) — не используем старую модель
                if buf_was_cleaned:
                    log.warning(f"[ML-v2] 🧹 Буфер очищен — старая модель обучена на багнутых данных. Отключаем до накопления чистых.")
                    self.model = None
                    self.trained = False
                    self.is_27f = False
            except Exception as e:
                log.warning(f"[ML-v2] 27f не загрузилась: {e}")

        # 2. Пробуем 1H-модель (16 признаков)
        if not self.trained:
            v2h = os.path.join(os.path.dirname(__file__), "models", "ml_pro_v2_1h.pkl")
            if os.path.exists(v2h):
                try:
                    with open(v2h, "rb") as f:
                        data = pickle.load(f)
                    self.model = data["model"]
                    self.trained = True
                    self.is_27f = False
                    acc = data.get("acc", 0)
                    n = data.get("samples", 0)
                    self.feature_names = data.get("features", FEATURE_NAMES_16)
                    log.info(f"[ML-v2] ✅ 1H-модель ({n:,} обр., {acc:.2%} acc)")
                except Exception as e:
                    log.warning(f"[ML-v2] 1H не загрузилась: {e}")

        # 3. Fallback на старую pkl
        if not self.trained and os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                self.model = data.get("model")
                self.trained = data.get("trained", False)
                self.is_27f = False
                self.feature_names = FEATURE_NAMES_16
                self.feature_importance = data.get("importance")
                log.info(f"[ML-v2] ✅ Fallback-модель из {self.model_path}")
            except Exception as e:
                log.warning(f"[ML-v2] Fallback не загрузилась: {e}")

    # ── Основной метод оценки ──────────────────────────────────────────────

    def evaluate(self, symbol, candles_5m, candles_1h, candles_4h, confidence, trend, rsi):
        """Оценить сделку. Возвращает (decision, prob, features)."""
        if not self.trained or self.model is None:
            return ("BUY" if confidence >= 55 else "SKIP"), confidence / 100.0, {}

        try:
            # 🧠 Сохраняем свечи для дообучения (Professional: source of truth)
            # Делаем это всегда, когда есть 1H+4H, независимо от загруженной модели
            if len(candles_1h or []) >= 100 and len(candles_4h or []) >= 25:
                h1_list = _listify(candles_1h)
                h4_list = _listify(candles_4h)
                self._last_candles = (h1_list[:] if h1_list else [], h4_list[:] if h4_list else [])
                self._last_candles_symbol = symbol

            # Выбираем путь в зависимости от типа модели
            if self.is_27f:
                if candles_1h and len(candles_1h) >= 100:
                    return self._evaluate_27f(symbol, candles_5m, candles_1h, candles_4h, confidence)
                else:
                    log.info(f"[ML-v2] {symbol}: 27f не вызвана — 1H={len(candles_1h or [])}, fallback на legacy")
            else:
                log.info(f"[ML-v2] {symbol}: is_27f=False, legacy 16f модель")

            # Старая модель (16 признаков)
            return self._evaluate_legacy(symbol, candles_5m, candles_1h, candles_4h, confidence, trend, rsi)

        except Exception as e:
            log.error(f"[ML-v2] Ошибка evaluate({symbol}): {e}")
            return "BUY", 0.5, {"error": str(e)}

    # ── Новая оценка через 27-признаковую модель ──────────────────────────

    def _evaluate_27f(self, symbol, candles_5m, candles_1h, candles_4h, confidence):
        """Оценка через 25-признаковую модель (1H+4H)."""
        # candles_1h может быть DataFrame или list[dict]
        if hasattr(candles_1h, 'iloc'):
            # Это DataFrame — конвертируем
            if len(candles_1h) < 100:
                return self._evaluate_legacy(symbol, candles_5m, candles_1h, candles_4h, confidence, 'neutral', 50)
            h1_list = candles_1h.to_dict('records') if hasattr(candles_1h, 'to_dict') else None
        else:
            h1_list = candles_1h

        # Аналогично для 4H
        if hasattr(candles_4h, 'iloc'):
            h4_list = candles_4h.to_dict('records') if hasattr(candles_4h, 'to_dict') else None
        else:
            h4_list = candles_4h

        if h1_list is None or h4_list is None or len(h1_list) < 100 or len(h4_list) < 25:
            log.warning(f"[ML-v2] Недостаточно 1H/4H для 27f-модели ({len(h1_list if h1_list else [])}/{len(h4_list if h4_list else [])})")
            return None, 0.5, {"error": "no_1h_data"}

        try:
            X = build_features_27f(h1_list, h4_list)
            if len(X) < 5:
                return None, 0.5, {"error": "no_features_27f"}

            last_row = X[-1:]
            prob = self._predict_proba(last_row)
            prob = float(prob)

            # HMM-динамические пороги
            buy_thr, skip_thr = _get_dynamic_thresholds()
            regime = _get_regime()
            regime_name = regime.get_state_name() if regime else 'N/A'

            # Решение с учётом режима
            if prob >= buy_thr:
                decision = "BUY"
            elif prob >= skip_thr and confidence >= 50:
                decision = "WEAK_BUY"
            else:
                decision = "SKIP"

            features = {
                "ml_prob": round(prob, 4),
                "model": "27f",
                "regime": regime_name,
                "buy_thr": buy_thr,
                "rsi_1h": round(float(X[-1, 0]), 1),
                "rsi_4h": round(float(X[-1, 1]), 1),
                "trend_1h": round(float(X[-1, 2]), 1),
                "trend_4h": round(float(X[-1, 3]), 1),
                "pv_ema26": round(float(X[-1, 9]), 2),
                "sma50_d": round(float(X[-1, 11]), 2),
                "volr_4h": round(float(X[-1, 13]), 2),
                "atr_ratio": round(float(X[-1, 7]), 3),
                "hour": int(X[-1, 24]),
                "mom24": round(float(X[-1, 18]), 2),
            }
            # 🧠 Сохраняем свечи для возможного ретрейна (Professional: храним source of truth, а не фичи)
            self._last_candles = (h1_list[:] if h1_list else [], h4_list[:] if h4_list else [])
            self._last_candles_symbol = symbol

            return decision, prob, features

        except Exception as e:
            log.error(f"[ML-v2] _evaluate_27f error: {e}")
            return None, 0.5, {"error": str(e)}

    # ── Старая оценка (16 признаков, 5M или 1H) ───────────────────────────

    def _evaluate_legacy(self, symbol, candles_5m, candles_1h, candles_4h, confidence, trend, rsi):
        """Оценка через старые модели (16 признаков)."""
        if hasattr(candles_5m, 'iloc'):
            df = candles_5m
            if len(df) < 60:
                return ("BUY" if confidence >= 55 else "SKIP"), confidence / 100.0, {}

            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            opens = df['open'].values
            vols = df['volume'].values if 'volume' in df.columns else np.ones(len(closes))

            # Быстрые фичи из 5M
            rsi_val = 50.0
            if len(closes) > 14:
                pc = np.diff(closes[-15:], prepend=closes[-15])
                gains = np.maximum(pc[1:], 0).mean()
                losses = np.maximum(-pc[1:], 0).mean()
                rsi_val = 100 - (100 / (1 + gains / losses)) if losses else 100

            tr_val = 1.0 if trend == 'bullish' else (-1.0 if trend == 'bearish' else 0.0)
            atr_pct = (highs[-14:].max() - lows[-14:].min()) / closes[-1] * 100 if len(closes) >= 14 else 0
            mom3 = (closes[-1] - closes[-4]) / closes[-4] * 100 if len(closes) > 4 else 0
            mom7 = (closes[-1] - closes[-8]) / closes[-8] * 100 if len(closes) > 8 else 0
            vr = vols[-1] / vols[-20:].mean() if len(vols) >= 20 else 1.0
            body = abs(closes[-1] - opens[-1]) / (highs[-1] - lows[-1]) if (highs[-1] - lows[-1]) > 0 else 0.5
            d24h = (closes[-1] - highs[-24:].max()) / highs[-24:].max() * 100 if len(highs) >= 24 else 0

            X_row = np.array([[rsi_val, tr_val, atr_pct, 1.0,
                               0, 0, 0, 0,
                               vr, mom3, mom7,
                               body, 0, 0, d24h, 0]], dtype=np.float64)
        elif len(candles_5m) >= 60 and len(candles_1h) >= 30:
            # Старый формат — build_features_5m
            X = build_features_5m(candles_5m, candles_1h, candles_4h)
            if len(X) == 0:
                return "SKIP", 0.5, {"error": "no_features"}
            X_row = X[-1:]
        else:
            return ("BUY" if confidence >= 55 else "SKIP"), confidence / 100.0, {}

        prob = self._predict_proba(X_row)
        prob = float(prob)

        decision = "BUY" if prob >= BUY_THRESHOLD else ("WEAK_BUY" if prob >= SKIP_THRESHOLD and confidence >= 50 else "SKIP")

        return decision, prob, {"ml_prob": round(prob, 4), "model": "16f"}

    # ── Предсказание ──────────────────────────────────────────────────────

    def _predict_proba(self, X_row):
        """Сигмоида для LightGBM Booster."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_row)[0, 1]
        raw = self.model.predict(X_row, predict_disable_shape_check=True)
        prob = 1.0 / (1.0 + np.exp(-raw))
        if hasattr(prob, '__iter__'):
            return float(prob[0])
        return float(prob)

    # ── Дообучение 27f модели на реальных сделках ────────────────────────────

    def store_entry_features(self, symbol: str):
        """Сохранить свечи входа для трейдера (Professional: храним сырые свечи, не фичи)."""
        if self._last_candles is not None and self._last_candles_symbol == symbol:
            h1, h4 = self._last_candles
            self._entry_buffer[symbol] = {
                'candles_1h': h1[:] if h1 else [],
                'candles_4h': h4[:] if h4 else [],
                'ts': time.time()
            }
            log.info(f"[ML-v2] 📝 {symbol}: сохранены свечи входа ({len(h1)}x1H / {len(h4)}x4H)")

    def record_outcome(self, symbol: str, pnl_pct: float):
        """Записать результат сделки для обучения (Professional: сохраняем свечи + outcome)."""
        entry = self._entry_buffer.pop(symbol, None)
        if entry is None:
            log.debug(f"[ML-v2] {symbol}: свечи входа не найдены")
            return
        # Успех: PnL > 0 (даже +0.1% — профит)
        label = 1.0 if pnl_pct > 0 else 0.0
        self._training_buffer.append({
            'candles_1h': entry['candles_1h'],
            'candles_4h': entry['candles_4h'],
            'label': label,
            'symbol': symbol,
            'ts': entry['ts']
        })
        n = len(self._training_buffer)
        good = sum(1 for e in self._training_buffer if e['label'] > 0.5)
        bad = n - good
        log.info(f"[ML-v2] 🏷️ {symbol}: outcome={'✅' if label else '❌'}, PnL={pnl_pct:+.2f}% "
                 f"(буфер: {n}, good={good}, bad={bad})")
        
        # 💾 Сохраняем буфер на диск, чтобы не терять при перезапусках
        self._save_training_buffer()

    def _save_training_buffer(self):
        """Сохранить training_buffer в pickle."""
        try:
            import pickle
            os.makedirs(os.path.dirname(self._training_buffer_path), exist_ok=True)
            with open(self._training_buffer_path, 'wb') as f:
                pickle.dump(self._training_buffer, f)
            log.info(f"[ML-v2] 💾 Буфер сохранён: {len(self._training_buffer)} примеров")
        except Exception as e:
            log.warning(f"[ML-v2] ⚠️ Не удалось сохранить training buffer: {e}")
    
    def _load_training_buffer(self):
        """Загрузить training_buffer из pickle (персистентность через рестарты).
        Поддерживает миграцию из старого формата [(features, label, symbol, ts)] в новый.
        Возвращает: True если буфер был очищен (старый формат), False иначе.
        """
        cleaned = False
        try:
            if os.path.exists(self._training_buffer_path):
                with open(self._training_buffer_path, 'rb') as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, list) and len(loaded) > 0:
                    # Проверка формата: старый = tuple, новый = dict
                    first = loaded[0]
                    if isinstance(first, tuple):
                        # Старый формат [(features, label, symbol, ts)] — дропаем
                        log.warning(f"[ML-v2] 🧹 Обнаружен старый формат training_buffer ({len(loaded)} примеров). Очищаем — данные содержат багнутые константные фичи.")
                        self._training_buffer = []
                        self._save_training_buffer()
                        cleaned = True
                    else:
                        self._training_buffer = loaded
                        log.info(f"[ML-v2] 📂 Загружен training_buffer: {len(loaded)} примеров")
        except Exception as e:
            log.warning(f"[ML-v2] ⚠️ Не удалось загрузить training buffer: {e}")
        return cleaned
    
    def seed_from_db(self, exchange=None):
        """Загрузить исторические сделки из БД (вызов после init, когда модули готовы).
        
        Профессиональная версия: загружает реальные свечи 1H и 4H с биржи
        и сохраняет их как source of truth для ретрейна.
        
        Принимает опциональный exchange (ccxt) для загрузки свечей.
        Вызывать один раз при старте системы, когда всё инициализировано.
        """
        try:
            import db_pg
            trades = db_pg.get_trade_history(limit=500)
        except Exception as e:
            log.warning(f"[ML-v2] seed_from_db: БД недоступна ({e})")
            return 0
        
        if not trades:
            return 0
        
        existing_ts = {e['ts'] for e in self._training_buffer if isinstance(e, dict)}
        log.info(f"[ML-v2] 📜 Загрузка исторических сделок из БД ({len(trades)} всего, {len(existing_ts)} уже есть)...")
        
        # Группируем по символам
        from collections import defaultdict
        by_symbol = defaultdict(list)
        for t in trades:
            sym = t.get('symbol')
            et = t.get('entry_time')
            xt = t.get('exit_time')
            pnl = t.get('pnl')
            pct = t.get('pnl_percent') or 0
            if not all([sym, et, xt, pnl is not None]):
                continue
            try:
                entry_ts = datetime.fromisoformat(et).timestamp()
                if any(abs(entry_ts - ts) < 10 for ts in existing_ts):
                    continue
            except Exception:
                continue
            by_symbol[sym].append({
                'entry_time': et, 'entry_ts': entry_ts,
                'pnl': pnl, 'pnl_pct': pct,
            })
        
        if not by_symbol:
            log.info(f"[ML-v2] 📜 Нет новых исторических сделок")
            return 0
        
        total_trades = sum(len(v) for v in by_symbol.values())
        log.info(f"[ML-v2] 📜 {len(by_symbol)} символов, нужно обработать {total_trades} сделок")
        
        # Используем переданный exchange или создаём новый
        ex = exchange
        if ex is None:
            import ccxt
            ex = ccxt.bybit()
        
        added = 0
        errors = 0
        for sym, strades in by_symbol.items():
            if not strades:
                continue
            try:
                # Загружаем реальные свечи с биржи (Professional: source of truth)
                raw_1h = ex.fetch_ohlcv(sym, '1h', limit=200)
                raw_4h = ex.fetch_ohlcv(sym, '4h', limit=50)
                if not raw_1h or not raw_4h:
                    log.warning(f"[ML-v2] ⚠️ {sym}: свечи не загружены")
                    continue
                
                c1h = [{'o':c[1],'h':c[2],'l':c[3],'c':c[4],'v':c[5],'t':c[0]/1000} for c in raw_1h]
                c4h = [{'o':c[1],'h':c[2],'l':c[3],'c':c[4],'v':c[5],'t':c[0]/1000} for c in raw_4h]
                
                for s in strades:
                    try:
                        # Смещаем свечи к моменту входа
                        ets = s['entry_ts']
                        aligned_1h = [c for c in c1h if c['t'] <= ets][-200:]
                        aligned_4h = [c for c in c4h if c['t'] <= ets][-50:]
                        
                        if len(aligned_1h) < 50 or len(aligned_4h) < 10:
                            errors += 1
                            continue
                        
                        label = 1.0 if s['pnl'] > 0 else 0.0
                        self._training_buffer.append({
                            'candles_1h': aligned_1h,
                            'candles_4h': aligned_4h,
                            'label': label,
                            'symbol': sym,
                            'ts': s['entry_ts']
                        })
                        existing_ts.add(s['entry_ts'])
                        added += 1
                    except Exception:
                        errors += 1
                        continue
            except Exception:
                errors += 1
                continue
        
        log.info(f"[ML-v2] 📜 Исторические: +{added} сделок (ошибок: {errors}). Буфер: {len(self._training_buffer)}")
        if added:
            self._save_training_buffer()
        return added
    
    def retrain_27f(self, force=False):
        """Дообучить 27f модель на накопленных сделках.
        
        Professional: пересчитывает фичи из сырых свечей актуальной версией build_features_27f.
        Это гарантирует, что изменения в фича-инжиниринге не сломают исторические данные.
        """
        import lightgbm as lgb
        now = time.time()
        if not force and (now - self._last_retrain_27f) < RETRAIN_INTERVAL_27F:
            return

        n = len(self._training_buffer)
        if n < MIN_SAMPLES_27F:
            log.info(f"[ML-v2] ⏭️ Ретрейн 27f: мало данных ({n} < {MIN_SAMPLES_27F})")
            return

        # Пересчитываем фичи из свечей (Professional: актуальная версия кода)
        X_list = []
        y_list = []
        for entry in self._training_buffer:
            h1 = entry.get('candles_1h', [])
            h4 = entry.get('candles_4h', [])
            if len(h1) < 50 or len(h4) < 10:
                continue
            try:
                X_row = build_features_27f(h1, h4)
                if X_row is not None and len(X_row) > 0:
                    # Берем все 33 колонки (FEATURE_NAMES = FEATURE_NAMES_33)
                    feat = X_row[-1, :33].copy()
                    X_list.append(feat)
                    y_list.append(int(entry['label']))
            except Exception:
                continue

        if len(X_list) < MIN_SAMPLES_27F:
            log.info(f"[ML-v2] ⏭️ Ретрейн 27f: после пересчёта фичей осталось {len(X_list)} (нужно >= {MIN_SAMPLES_27F})")
            return

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.int32)
        good = int(y.sum())
        bad = len(y) - good

        if good < MIN_BOTH_CLASSES_27F or bad < MIN_BOTH_CLASSES_27F:
            log.info(f"[ML-v2] ⏭️ Ретрейн 27f: мало одного класса (good={good}, bad={bad})")
            return

        log.info(f"[ML-v2] 🔄 Ретрейн 27f на {len(X)} примерах (good={good}, bad={bad})...")

        # Train/val split
        te = int(len(X) * 0.75)
        params = dict(objective='binary', metric='binary_logloss', boosting='gbdt',
                      num_leaves=31, learning_rate=0.03, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=5, min_child_samples=10, verbose=-1)

        td = lgb.Dataset(X[:te], label=y[:te], feature_name=FEATURE_NAMES)
        vd = lgb.Dataset(X[te:], label=y[te:], feature_name=FEATURE_NAMES, reference=td)

        # Обучаем с нуля (на всех доступных данных)
        self.model = lgb.train(params, td, num_boost_round=500, valid_sets=[vd],
                               callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        self.trained = True

        # Оценка
        yp = self.model.predict(X[te:])
        yb = (yp > 0.5).astype(int)
        acc = float(np.mean(yb == y[te:]))
        log.info(f"[ML-v2] 🎯 Ретрейн 27f завершён! Точность: {acc:.1%} ({len(X)} samples)")

        # Feature importance
        try:
            imp = self.model.feature_importance('gain')
            top5 = [(FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f'f{i}', int(imp[i]))
                    for i in np.argsort(imp)[::-1][:5]]
            log.info(f"[ML-v2] 📊 Топ-5 признаков: {top5}")
            self.feature_importance = [{"rank": i+1, "name": n, "gain": g}
                                       for i, (n, g) in enumerate(top5)]
        except Exception:
            pass

        self._last_retrain_27f = now
        # Сохраняем вместе с основным файлом 27f
        v27f_path = os.path.join(os.path.dirname(__file__), "models", "ml_pro_v2_27f.pkl")
        try:
            with open(v27f_path, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "trained": True,
                    "acc": acc,
                    "samples": len(X),
                    "features": FEATURE_NAMES,
                    "importance": self.feature_importance
                }, f)
            log.info(f"[ML-v2] ✅ 27f модель сохранена ({v27f_path})")
        except Exception as e:
            log.error(f"[ML-v2] ❌ Ошибка сохранения 27f модели: {e}")

    # ── Обучение (16-признаковая, legacy) ───────────────────────────────────

    def train(self, X, y):
        """Обучить LightGBM модель (для 16-признаковой)."""
        import lightgbm as lgb
        if len(X) < MIN_TRAIN_SAMPLES:
            return

        n = len(X)
        te = int(n * 0.70)
        va = int(n * 0.85)

        params = dict(objective='binary', metric='binary_logloss', boosting='gbdt',
                      num_leaves=31, learning_rate=0.05, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=5, min_child_samples=20, verbose=-1)

        td = lgb.Dataset(X[:te], label=y[:te], feature_name=FEATURE_NAMES_16)
        vd = lgb.Dataset(X[te:va], label=y[te:va], feature_name=FEATURE_NAMES_16, reference=td)

        self.model = lgb.train(params, td, num_boost_round=1000, valid_sets=[vd],
                               callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        self.trained = True
        self.is_27f = False

        imp = self.model.feature_importance('gain')
        self.feature_importance = [{"rank": i + 1, "name": FEATURE_NAMES_16[j] if j < len(FEATURE_NAMES_16) else f"f{j}",
                                    "gain": int(imp[j])} for i, j in enumerate(np.argsort(imp)[::-1][:10])]

        yp = self.model.predict(X[va:])
        yb = (yp > 0.5).astype(int)
        acc = np.mean(yb == y[va:])
        log.info(f"[ML-v2] Точность на тесте: {acc:.2%} ({len(y[va:])} samples)")
        self._save()

    def _save(self):
        if not self.trained:
            return
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self.model, "trained": self.trained, "importance": self.feature_importance}, f)


# ============================================================================
# Глобальный экземпляр (singleton)
# ============================================================================
_instance = None


def get_ml():
    global _instance
    if _instance is None:
        _instance = MLProfessionalV2()
    return _instance


def ml_pro_v2_evaluate(symbol, candles_5m, candles_1h, candles_4h, confidence, trend, rsi):
    """Упрощённый API для compatibility."""
    return get_ml().evaluate(symbol, candles_5m, candles_1h, candles_4h, confidence, trend, rsi)


# ============================================================================
# Старая build_features_5m (для совместимости со старыми моделями)
# ============================================================================
# (сохранена для обратной совместимости, не используется в 27f)

NUM_FEATURES = 33

FEATURE_NAMES = FEATURE_NAMES_33


def build_features_5m(candles_5m, candles_1h, candles_4h):
    """Построить 27 признаков из 5M/1H/4H (для обратной совместимости)."""
    # Нормализация ключей (из to_dict('records') или Raw OHLCV)
    candles_5m = [_normalize_candle_dict(c) for c in candles_5m]
    candles_1h = [_normalize_candle_dict(c) for c in candles_1h]
    candles_4h = [_normalize_candle_dict(c) for c in candles_4h]

    def _parse(arr):
        o = np.array([c['o'] for c in arr], dtype=float)
        h = np.array([c['h'] for c in arr], dtype=float)
        l = np.array([c['l'] for c in arr], dtype=float)
        cl = np.array([c['c'] for c in arr], dtype=float)
        v = np.array([c['v'] for c in arr], dtype=float)
        return o, h, l, cl, v

    o5, h5, l5, cl5, v5 = _parse(candles_5m)
    o1, h1, l1, cl1, v1 = _parse(candles_1h)
    o4, h4, l4, cl4, v4 = _parse(candles_4h)

    n = len(cl5)
    X = np.zeros((n, NUM_FEATURES), dtype=np.float64)

    rsi_5m = _rsi(cl5)
    trend_5m = _trend(cl5)
    rsi_1h_full = _rsi(cl1)
    atr_1h_full = _atr(h1, l1, cl1)
    ema12_1h = _ema(cl1, 12)
    ema26_1h = _ema(cl1, 26)  # Not used in original but kept
    trend_1h_full = _trend(cl1)
    vwap_1h = _vwap(h1, l1, cl1, v1)

    t5 = np.array([c['t'] for c in candles_5m], dtype=float)
    t1 = np.array([c['t'] for c in candles_1h], dtype=float)
    idx_1h = np.clip(np.searchsorted(t1, t5, side='right') - 1, 0, len(cl1) - 1)

    rsi_4h_full = _rsi(cl4)
    atr_4h_full = _atr(h4, l4, cl4)
    ema26_4h = _ema(cl4, 26)
    sma20_4h = np.zeros(len(cl4), dtype=float)
    for i in range(20, len(cl4)):
        sma20_4h[i] = cl4[i - 20:i].mean()
    trend_4h_full = _trend(cl4)

    t4 = np.array([c['t'] for c in candles_4h], dtype=float)
    idx_4h = np.clip(np.searchsorted(t4, t5, side='right') - 1, 0, len(cl4) - 1)

    sma20_1h = np.zeros(len(cl1), dtype=float)
    for i in range(20, len(cl1)):
        sma20_1h[i] = cl1[i - 20:i].mean()

    vol_ma_1h = np.ones(len(cl1), dtype=float)
    for i in range(20, len(cl1)):
        m = v1[i - 20:i].mean()
        vol_ma_1h[i] = m if m > 0 else 1.0

    vol_ma_4h = np.ones(len(cl4), dtype=float)
    for i in range(20, len(cl4)):
        m = v4[i - 20:i].mean()
        vol_ma_4h[i] = m if m > 0 else 1.0

    # build_features_5m сохраняется для обратной совместимости
    # Фактически не используется — 27-признаковая модель использует build_features_27f
    if n == 0:
        return X
    
    for i in range(max(50, int(idx_1h[:100].max()) + 1 if len(idx_1h) > 100 else 50), n):
        j1 = int(idx_1h[i])
        j4 = int(idx_4h[i])

        if j1 < 14 or j4 < 5:
            continue

        c1_val = cl1[j1]
        c4_val = cl4[j4]

        X[i, 0] = rsi_5m[i]
        X[i, 1] = rsi_1h_full[j1]
        X[i, 2] = rsi_4h_full[j4]
        X[i, 3] = trend_5m[i]
        X[i, 4] = trend_1h_full[j1]
        X[i, 5] = trend_4h_full[j4]
        trends = [X[i, 3], X[i, 4], X[i, 5]]
        if all(t > 0 for t in trends):
            X[i, 6] = 1.0
        elif all(t < 0 for t in trends):
            X[i, 6] = -1.0
        X[i, 7] = atr_1h_full[j1] / c1_val * 100 if c1_val > 0 else 0
        X[i, 8] = atr_4h_full[j4] / c4_val * 100 if c4_val > 0 else 0
        if j1 >= 48:
            atr_short = atr_1h_full[j1 - 5:j1 + 1].mean()
            atr_long = atr_1h_full[j1 - 48:j1 + 1].mean()
            X[i, 9] = atr_short / atr_long if atr_long > 0 else 1.0
        if ema12_1h[j1] > 0 and c1_val > 0:
            X[i, 10] = (c1_val - ema12_1h[j1]) / ema12_1h[j1] * 100
        if ema26_4h[j4] > 0 and c4_val > 0:
            X[i, 11] = (c4_val - ema26_4h[j4]) / ema26_4h[j4] * 100
        if j1 >= 1 and ema12_1h[j1] > 0:
            X[i, 12] = (cl1[j1] - ema12_1h[j1]) / ema12_1h[j1] * 100
        if j4 >= 1 and ema26_4h[j4] > 0:
            X[i, 13] = (cl4[j4] - ema26_4h[j4]) / ema26_4h[j4] * 100
        if sma20_1h[j1] > 0:
            X[i, 14] = (c1_val - sma20_1h[j1]) / sma20_1h[j1] * 100
        if sma20_4h[j4] > 0:
            X[i, 15] = (c4_val - sma20_4h[j4]) / sma20_4h[j4] * 100
        X[i, 16] = v1[j1] / vol_ma_1h[j1] if vol_ma_1h[j1] > 0 else 1.0
        X[i, 17] = v4[j4] / vol_ma_4h[j4] if vol_ma_4h[j4] > 0 else 1.0
        if vwap_1h[j1] > 0 and c1_val > 0:
            X[i, 18] = (c1_val - vwap_1h[j1]) / vwap_1h[j1] * 100
        if i >= 3:
            X[i, 19] = (cl5[i] - cl5[i - 3]) / cl5[i - 3] * 100
        if j1 >= 3:
            X[i, 20] = (cl1[j1] - cl1[j1 - 3]) / cl1[j1 - 3] * 100
        if j1 >= 7:
            X[i, 21] = (cl1[j1] - cl1[j1 - 7]) / cl1[j1 - 7] * 100
        body = abs(cl1[j1] - o1[j1])
        total = h1[j1] - l1[j1]
        X[i, 22] = body / total if total > 0 else 0.0
        X[i, 23] = _detect_pinbar(o1[j1], h1[j1], l1[j1], cl1[j1])
        if j1 >= 1:
            X[i, 24] = _detect_engulfing(o1[j1 - 1], cl1[j1 - 1], o1[j1], cl1[j1])
        if j1 >= 24:
            high_24h = h1[j1 - 24:j1 + 1].max()
            X[i, 25] = (c1_val - high_24h) / high_24h * 100 if high_24h > 0 else 0

    return X