"""
Volume/VWAP модуль — раннее обнаружение разворота.

Идея:
  Профессиональные трейдеры смотрят не на цену, а на объём + VWAP.
  Когда цена касается VWAP (или уходит далеко от него) и объём
  внезапно растёт — это первый признак разворота.

  Алгоритм:
  1. Считаем VWAP на 1H (12 свечей) — где справедливая цена.
  2. Смотрим отклонение текущей цены от VWAP.
  3. Если цена далеко от VWAP (>1.5%) + объём растёт → возможен возврат.
  4. Если цена у VWAP + объём падает → затишье, входа нет.
  5. Если цена пробивает VWAP с объёмом → трендовое движение.

  ⚡ Ключевое: Volume Spike = аномалия > 2σ от среднего объёма.
     Не просто "объём выше среднего", а статистически значимый всплеск.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List, Any
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# КОНФИГУРАЦИЯ
# ──────────────────────────────────────────────────────────────────────────────

VWAP_LOOKBACK_1H = 12           # Скользящий VWAP на 12 часов
VWAP_LOOKBACK_1D = 24           # Дневной VWAP
SPIKE_STD_THRESHOLD = 1.5       # Аномалия: >1.5σ от среднего объёма — чувствительнее к всплескам
VWAP_REVERSION_THRESHOLD = 0.008  # 0.8% от VWAP (было 1.2%) — раньше видим зону реверсии
VWAP_BREAKOUT_THRESHOLD = 0.003  # 0.3% — пробой VWAP (было 0.2%)
VOLUME_LOOKBACK = 20            # Окно среднего объёма
VOLUME_LOW_MARKET_FACTOR = 0.7  # Коэффициент снижения порога на тихом рынке
MIN_CLUSTERS = 3                # Минимум кластеров объёма для анализа
MIN_CANDLES_5M = 30             # Минимум 5М свечей для анализа

# ──────────────────────────────────────────────────────────────────────────────
# ДАТА-КЛАССЫ
# ──────────────────────────────────────────────────────────────────────────────


class VolumeState:
    """Состояние объёмного анализа для одной пары."""
    __slots__ = (
        'price', 'vwap_1h', 'vwap_1d',
        'vwap_deviation',          # отклонение от VWAP 1H в %
        'avg_volume', 'spike_volume', 'volume_ratio',
        'is_spike',                # TRUE если аномальный всплеск
        'volume_trend',            # 'rising' | 'falling' | 'flat'
        'reversion_zone',          # TRUE если цена у границы зоны реверсии
        'vwap_signal',             # 'reversion_buy' | 'reversion_sell' | 'breakout' | 'neutral'
        'signal_strength',         # 0.0 - 1.0
        'detail',                  # для логов
    )

    def __init__(self):
        self.price = 0.0
        self.vwap_1h = 0.0
        self.vwap_1d = 0.0
        self.vwap_deviation = 0.0
        self.avg_volume = 0.0
        self.spike_volume = 0.0
        self.volume_ratio = 1.0
        self.is_spike = False
        self.volume_trend = 'flat'
        self.reversion_zone = False
        self.vwap_signal = 'neutral'
        self.signal_strength = 0.0
        self.detail = ''


# ──────────────────────────────────────────────────────────────────────────────
# VWAP РАСЧЁТЫ
# ──────────────────────────────────────────────────────────────────────────────


def calc_vwap(closes: np.ndarray, highs: np.ndarray,
              lows: np.ndarray, volumes: np.ndarray) -> float:
    """
    Классический VWAP: сумма(типичная_цена * объём) / сумма(объём).
    Типичная цена = (high + low + close) / 3.
    """
    if len(closes) == 0 or volumes.sum() == 0:
        return 0.0
    typical_price = (highs + lows + closes) / 3.0
    return float(np.average(typical_price, weights=volumes))


def calc_moving_vwap(closes: np.ndarray, highs: np.ndarray,
                     lows: np.ndarray, volumes: np.ndarray,
                     window: int) -> np.ndarray:
    """
    Скользящий VWAP на окне window.
    Возвращает массив VWAP для каждой позиции (первые window-1 — NaN).
    """
    if len(closes) < window:
        return np.full(len(closes), np.nan)
    typical = (highs + lows + closes) / 3.0
    # Cumsum для производительности
    cum_typical_vol = np.cumsum(typical * volumes)
    cum_volume = np.cumsum(volumes)
    vwap = np.full(len(closes), np.nan)
    for i in range(window - 1, len(closes)):
        tv = cum_typical_vol[i] - (cum_typical_vol[i - window] if i >= window else 0)
        v = cum_volume[i] - (cum_volume[i - window] if i >= window else 0)
        vwap[i] = tv / v if v > 0 else np.nan
    return vwap


# ──────────────────────────────────────────────────────────────────────────────
# ОБЪЁМНЫЙ АНАЛИЗ
# ──────────────────────────────────────────────────────────────────────────────


def detect_volume_spike(volumes: np.ndarray, lookback: int = VOLUME_LOOKBACK,
                        std_threshold: float = SPIKE_STD_THRESHOLD) -> Tuple[bool, float, float]:
    """
    Обнаружение аномального всплеска объёма.
    На тихом рынке (низкая базовая активность) порог снижается, чтобы
    не пропускать начало движения.
    Возвращает: (is_spike, avg_volume, current_volume_ratio)
    """
    if len(volumes) < lookback + 1:
        return False, 0.0, 1.0

    recent = volumes[-lookback:]
    current = volumes[-1]
    mean_vol = float(np.mean(recent))
    std_vol = float(np.std(recent)) + 1e-10
    z_score = (current - mean_vol) / std_vol
    ratio = current / (mean_vol + 1e-10)

    # Определяем тихий рынок: если медианный объём за lookback ниже
    # среднего за весь период, снижаем порог
    # Это позволяет ловить начало движения, когда рынок просыпается
    if len(volumes) >= lookback * 2:
        longer_volumes = volumes[-(lookback * 2):-lookback]
        median_recent = np.median(recent)
        median_longer = np.median(longer_volumes)
        
        # Если последние 20 свечей тише, чем предыдущие 20 — рынок спит
        # снижаем порог, чтобы проснуться
        if median_recent < median_longer * VOLUME_LOW_MARKET_FACTOR:
            adjusted_threshold = std_threshold * 0.7  # 1.8σ → 1.26σ
            return z_score > adjusted_threshold, mean_vol, ratio

    return z_score > std_threshold, mean_vol, ratio


def detect_volume_trend(volumes: np.ndarray, lookback: int = 10) -> str:
    """
    Тренд объёма: сравниваем первую половину окна со второй.
    Также проверяем последовательный рост 3+ свечей.
    'rising' | 'falling' | 'flat'
    """
    if len(volumes) < lookback:
        return 'flat'
    segment = volumes[-lookback:]
    half = lookback // 2
    first_half = np.mean(segment[:half])
    second_half = np.mean(segment[half:])
    change = (second_half - first_half) / (first_half + 1e-10)
    
    # Дополнительно: последовательный рост объёма
    # Если последние 3 свечи растущие — это ранний признак активности
    consecutive_growth = 0
    if len(volumes) >= 4:
        for i in range(1, 4):
            if volumes[-i] > volumes[-(i+1)]:
                consecutive_growth += 1
            else:
                break
    
    if consecutive_growth >= 3:
        return 'rising'
    
    if change > 0.15:
        return 'rising'
    elif change < -0.15:
        return 'falling'
    return 'flat'


# ──────────────────────────────────────────────────────────────────────────────
# СИГНАЛЫ
# ──────────────────────────────────────────────────────────────────────────────


def detect_vwap_reversion(price: float, vwap_1h: float, vwap_1d: float,
                          is_spike: bool, volume_trend: str,
                          deviation: float) -> Tuple[str, float]:
    """
    Определение сигнала на основе VWAP + объём.

    Логика:
      reversion_buy:
        - Цена значительно ниже VWAP (>1.5%) — перепродано
        - ИЛИ цена касается нижней границы VWAP канала
        - Объём растёт (spike или rising) — крупный игрок заходит
        - Сигнал: разворот вверх

      reversion_sell:
        - Цена значительно выше VWAP (>1.5%) — перекуплено
        - ИЛИ цена касается верхней границы VWAP канала
        - Объём растёт — распределение
        - Сигнал: разворот вниз

      breakout:
        - Цена пробивает VWAP снизу вверх с объёмом — бычий пробой
        - Цена пробивает VWAP сверху вниз с объёмом — медвежий пробой

      neutral: всё остальное
    """
    if vwap_1h <= 0:
        return 'neutral', 0.0

    abs_deviation = abs(deviation)
    strength = 0.0

    # 1. Реверсия: цена далеко от VWAP
    if deviation < -VWAP_REVERSION_THRESHOLD:
        # Цена ниже VWAP — потенциальный отскок
        strength = min(1.0, abs_deviation * 30)  # 1.5%→0.45, 3%→0.9
        if is_spike or volume_trend == 'rising':
            strength = min(1.0, strength * 1.3 + 0.2)
            return 'reversion_buy', strength
        return 'reversion_buy', strength * 0.5  # без объёма — слабее

    elif deviation > VWAP_REVERSION_THRESHOLD:
        # Цена выше VWAP — потенциальное падение
        strength = min(1.0, abs_deviation * 30)
        if is_spike or volume_trend == 'rising':
            strength = min(1.0, strength * 1.3 + 0.2)
            return 'reversion_sell', strength
        return 'reversion_sell', strength * 0.5

    # 2. Пробой VWAP
    elif abs_deviation < VWAP_BREAKOUT_THRESHOLD:
        # Цена у VWAP — смотрим объём
        if is_spike and volume_trend == 'rising':
            # Пробой с объёмом — сильное движение
            strength = 0.7
            # Определяем направление по последнему движению
            # (передаётся через контекст, здесь generic)
            return 'breakout', strength

    return 'neutral', 0.0


# ──────────────────────────────────────────────────────────────────────────────
# ОСНОВНАЯ ФУНКЦИЯ ОЦЕНКИ
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_candles(candles: List) -> List[Dict]:
    """
    Привести свечи к единому формату [{'close':..., 'high':..., 'low':..., 'volume':...}, ...].
    Поддерживает: list of dict, list of tuple (ohlcv).
    """
    if not candles:
        return []
    if isinstance(candles[0], dict):
        # Уже dict
        normalized = []
        for c in candles:
            normalized.append({
                'close': float(c.get('close', c.get('c', 0))),
                'high': float(c.get('high', c.get('h', 0))),
                'low': float(c.get('low', c.get('l', 0))),
                'volume': float(c.get('volume', c.get('v', 0))),
            })
        return normalized
    # Tuple (timestamp, open, high, low, close, volume)
    return [
        {'close': float(c[4]), 'high': float(c[2]),
         'low': float(c[3]), 'volume': float(c[5])}
        for c in candles
    ]


def evaluate(symbol: str, current_price: float,
             candles_5m: List, candles_1h: List,
             candles_4h: Optional[List] = None) -> Dict[str, Any]:
    """
    Главная функция: анализ VWAP + объём для одной пары.

    Параметры:
      candles_5m — последние N свечей 5М (list of dict или tuple OHLCV)
      candles_1h — последние N свечей 1Н
      candles_4h — опционально 4Н (для контекста)

    Возвращает:
      {
        'score': 0-100,
        'signal': 'reversion_buy' | 'reversion_sell' | 'breakout' | 'neutral',
        'strength': 0.0-1.0,
        'state': VolumeState (с деталями),
        'detail': str для логов,
      }
    """
    state = VolumeState()

    # Нормализуем формат свечей
    c5 = _normalize_candles(candles_5m)
    c1 = _normalize_candles(candles_1h)
    c4 = _normalize_candles(candles_4h) if candles_4h else []

    if not c5 or len(c5) < MIN_CANDLES_5M:
        return {'score': 50, 'signal': 'neutral', 'strength': 0.0,
                'state': state, 'detail': 'мало данных'}

    # ── Извлекаем массивы из свечей 5М ──────────────────────────────────
    closes_5m = np.array([c['close'] for c in c5], dtype=float)
    highs_5m = np.array([c['high'] for c in c5], dtype=float)
    lows_5m = np.array([c['low'] for c in c5], dtype=float)
    volumes_5m = np.array([c['volume'] for c in c5], dtype=float)

    # ── VWAP на 1H (основной) ───────────────────────────────────────────
    if c1 and len(c1) >= VWAP_LOOKBACK_1H:
        closes_1h = np.array([c['close'] for c in c1[-VWAP_LOOKBACK_1H:]], dtype=float)
        highs_1h = np.array([c['high'] for c in c1[-VWAP_LOOKBACK_1H:]], dtype=float)
        lows_1h = np.array([c['low'] for c in c1[-VWAP_LOOKBACK_1H:]], dtype=float)
        volumes_1h = np.array([c['volume'] for c in c1[-VWAP_LOOKBACK_1H:]], dtype=float)
        state.vwap_1h = calc_vwap(closes_1h, highs_1h, lows_1h, volumes_1h)
        state.vwap_1h = calc_vwap(closes_1h, highs_1h, lows_1h, volumes_1h)
    else:
        # Фолбэк: VWAP на 5М за 12 периодов (~1 час)
        lookback = min(VWAP_LOOKBACK_1H * 12, len(closes_5m))
        if lookback >= 12:
            state.vwap_1h = calc_vwap(
                closes_5m[-lookback:], highs_5m[-lookback:],
                lows_5m[-lookback:], volumes_5m[-lookback:]
            )

    # ── VWAP на 1D (контекст) ───────────────────────────────────────────
    if c4 and len(c4) >= 6:
        closes_4h = np.array([c['close'] for c in c4[-6:]], dtype=float)
        highs_4h = np.array([c['high'] for c in c4[-6:]], dtype=float)
        lows_4h = np.array([c['low'] for c in c4[-6:]], dtype=float)
        volumes_4h = np.array([c['volume'] for c in c4[-6:]], dtype=float)
        state.vwap_1d = calc_vwap(closes_4h, highs_4h, lows_4h, volumes_4h)
    else:
        # Фолбэк: VWAP на 5М за 288 свечей (~1 день)
        lookback = min(288, len(closes_5m))
        if lookback >= 50:
            state.vwap_1d = calc_vwap(
                closes_5m[-lookback:], highs_5m[-lookback:],
                lows_5m[-lookback:], volumes_5m[-lookback:]
            )

    state.price = current_price

    # ── Отклонение от VWAP ──────────────────────────────────────────────
    if state.vwap_1h > 0:
        state.vwap_deviation = (current_price - state.vwap_1h) / state.vwap_1h
    elif state.vwap_1d > 0:
        state.vwap_deviation = (current_price - state.vwap_1d) / state.vwap_1d

    # ── Объёмный анализ ──────────────────────────────────────────────────
    state.is_spike, state.avg_volume, state.volume_ratio = \
        detect_volume_spike(volumes_5m)
    state.volume_trend = detect_volume_trend(volumes_5m)
    state.spike_volume = float(volumes_5m[-1]) if len(volumes_5m) > 0 else 0.0

    # ── Зона реверсии ───────────────────────────────────────────────────
    state.reversion_zone = abs(state.vwap_deviation) > VWAP_REVERSION_THRESHOLD

    # ── Сигнал ───────────────────────────────────────────────────────────
    signal, strength = detect_vwap_reversion(
        current_price, state.vwap_1h, state.vwap_1d,
        state.is_spike, state.volume_trend,
        state.vwap_deviation
    )
    state.vwap_signal = signal
    state.signal_strength = strength

    # ── Score 0-100 ──────────────────────────────────────────────────────
    score = 50  # нейтрально
    if signal == 'reversion_buy':
        score = int(50 + strength * 40)  # 50-90
    elif signal == 'reversion_sell':
        score = int(50 - strength * 40)  # 10-50
    elif signal == 'breakout':
        score = int(60 + strength * 30)  # 60-90

    # ── Деталь для логов ────────────────────────────────────────────────
    detail_parts = [
        f"VWAP=${state.vwap_1h:.4f}",
        f"откл={state.vwap_deviation*100:+.2f}%",
    ]
    if state.is_spike:
        detail_parts.append(f"SPIKE({state.volume_ratio:.1f}x)")
    detail_parts.append(f"тренд={state.volume_trend}")
    if signal != 'neutral':
        detail_parts.append(f"⚡{signal}({strength:.0%})")

    state.detail = ' '.join(detail_parts)

    return {
        'score': score,
        'signal': signal,
        'strength': strength,
        'state': state,
        'detail': state.detail,
    }
