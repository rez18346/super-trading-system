"""
VSA (Volume Spread Analysis) + Liq Dynamics

Что даёт:
1. VSA-оценка: up/down volume, дивергенция объём-цена
2. Тренд POC: куда смещается кластер ликвидности
3. Моментальный импульс: Acceleration = (Liq trend + Volume impulse + Price impulse) / 3

Встраивается в decision_engine как дополнительный фильтр.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# КОНФИГ
# ──────────────────────────────────────────────────────────────────────────────

UP_THRESHOLD = 0.001      # 0.1% — минимальное движение для "значимой" свечи
POC_TREND_CANDLES = 10    # Скользим по 10 свечам для тренда POC
STRONG_DIVERGENCE = -0.5  # Порог дивергенции (отрицательная = цена↑, объём↓)
LIQ_FORCE_WEIGHT = 0.6    # Вес объёмного анализа в итоговом импульсе


# ──────────────────────────────────────────────────────────────────────────────
# ДАТА-КЛАССЫ
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class VsaResult:
    """Результат VSA-анализа для одной пары."""
    up_volume_ratio: float      # объём вверх / общий объём (0.0 - 1.0)
    down_volume_ratio: float    # объём вниз / общий объём
    volume_divergence: float    # Расхождение: +1 цена и объём вверх, -1 цена↑ объём↓
    accumulation: float         # Признак накопления (0.0 - 1.0)
    distribution: float        # Признак распределения (0.0 - 1.0)
    liq_trend: float           # Тренд POC: +1 up, 0 flat, -1 down
    momentum: float            # Общий импульс (-1.0 до 1.0)
    signal: str                # 'bullish' | 'bearish' | 'neutral'
    strength: float            # Сила сигнала (0.0 - 1.0)
    detail: str                # Для логов


# ──────────────────────────────────────────────────────────────────────────────
# КЭШ (чтобы не пересчитывать если цена не изменилась)
# ──────────────────────────────────────────────────────────────────────────────

_cache: Dict[str, Tuple[float, VsaResult]] = {}
_CACHE_TTL = 60  # 60 секунд


# ──────────────────────────────────────────────────────────────────────────────
# ФУНКЦИИ VSA
# ──────────────────────────────────────────────────────────────────────────────


def analyze_volume_spread(candles_5m: List[Dict]) -> VsaResult:
    """
    VSA на 5-минутных свечах.

    Считаем:
    - Кумулятивный объём вверх/вниз
    - Дивергенцию (цена идёт вверх, а объём падает = слабый рост)
    - Признак накопления (широкий спред + объём на дне)
    - Признак распределения (широкий спред + объём на вершине)
    """
    if not candles_5m or len(candles_5m) < 10:
        return VsaResult(
            up_volume_ratio=0.5, down_volume_ratio=0.5,
            volume_divergence=0.0, accumulation=0.0, distribution=0.0,
            liq_trend=0.0, momentum=0.0,
            signal='neutral', strength=0.0, detail='мало данных'
        )

    # Универсальный доступ: ['c','close','Close'] / ['v','volume','Volume']
    def _get(candles, keys):
        for k in keys:
            val = candles.get(k)
            if val is not None:
                return val
        return 0.0
    closes = np.array([_get(c, ['c','close','Close']) for c in candles_5m], dtype=float)
    highs = np.array([_get(c, ['h','high','High']) for c in candles_5m], dtype=float)
    lows = np.array([_get(c, ['l','low','Low']) for c in candles_5m], dtype=float)
    volumes = np.array([_get(c, ['v','volume','Volume']) for c in candles_5m], dtype=float)

    # ── Up/Down Volume ───────────────────────────────────────────────────
    # Свеча вверх: close >= open. Свеча вниз: close < open
    changes = np.diff(closes)
    # Берём последние 30 свечей
    lookback = min(30, len(changes))

    up_vol = sum(volumes[i+1] for i in range(-lookback, 0) if changes[i] > 0)
    down_vol = sum(volumes[i+1] for i in range(-lookback, 0) if changes[i] <= 0)
    total_vol = up_vol + down_vol

    if total_vol > 0:
        up_ratio = up_vol / total_vol
        down_ratio = down_vol / total_vol
    else:
        up_ratio = 0.5
        down_ratio = 0.5

    # ── Дивергенция ──────────────────────────────────────────────────────
    # Если цена растёт, а объём по up-свечам падает = слабость
    price_change_30 = (closes[-1] - closes[-lookback]) / closes[-lookback] if closes[-lookback] > 0 else 0
    vol_trend_30 = np.polyfit(range(len(volumes[-lookback:])), volumes[-lookback:], 1)[0] if len(volumes) >= lookback else 0
    vol_trend_normalized = vol_trend_30 / (np.mean(volumes[-lookback:]) + 1e-9)

    # Дивергенция: +1 = объём и цена растут (сильный тренд)
    #            -1 = цена растёт, объём падает (слабый, разворот)
    divergence = 0.0
    if abs(price_change_30) > UP_THRESHOLD:
        # Нормализуем знак: положительная дивергенция = объём подтверждает движение
        price_sign = 1 if price_change_30 > 0 else -1
        vol_sign = 1 if vol_trend_normalized > 0 else -1

        if price_sign == vol_sign:
            divergence = min(1.0, abs(price_change_30) * 50) * price_sign
        else:
            # Расхождение: движение цены не подтверждается объёмом
            divergence = -min(0.5, abs(vol_trend_normalized) * 5) * price_sign

    # ── Накопление / Распределение ──────────────────────────────────────
    # Накопление: широкий спред вниз + рост объёма → крупный игрок покупает
    # Распределение: широкий спред вверх + рост объёма → раздача
    spreads = highs - lows
    avg_spread = np.mean(spreads[-lookback:]) if lookback <= len(spreads) else np.mean(spreads)
    last_spread = spreads[-1] if len(spreads) > 0 else 0

    accumulation = 0.0
    distribution = 0.0

    if avg_spread > 0 and last_spread > avg_spread * 1.5:
        last_close = closes[-1]
        last_low = lows[-1]
        last_high = highs[-1]

        # Накопление: цена у низа диапазона, объём выше среднего
        position_in_range = (last_close - last_low) / (last_high - last_low) if (last_high - last_low) > 0 else 0.5
        vol_ratio = volumes[-1] / (np.mean(volumes[-min(lookback, len(volumes)):]) + 1e-9)

        if position_in_range < 0.3 and vol_ratio > 1.5:
            accumulation = min(1.0, vol_ratio / 3.0)
        elif position_in_range > 0.7 and vol_ratio > 1.5:
            distribution = min(1.0, vol_ratio / 3.0)

    # ── Liq Trend (имитация POC через VWAP) ─────────────────────────────
    # Если у нас нет POC от LiquidityCluster, используем скользящий VWAP
    vwap_short = np.average(closes[-12:], weights=volumes[-12:]) if len(closes) >= 12 else closes[-1]
    vwap_medium = np.average(closes[-24:], weights=volumes[-24:]) if len(closes) >= 24 else closes[-1]

    if vwap_medium > 0:
        liq_trend = (vwap_short - vwap_medium) / vwap_medium
        liq_trend = np.clip(liq_trend * 20, -1.0, 1.0)  # нормализация
    else:
        liq_trend = 0.0

    # ── Моментальный импульс ────────────────────────────────────────────
    # Сочетание: VWAP тренд + дивергенция + накопление
    momentum = (
        liq_trend * 0.35 +
        divergence * 0.35 +
        (accumulation - distribution) * 0.3
    )
    momentum = np.clip(momentum, -1.0, 1.0)

    # ── Итоговый сигнал ─────────────────────────────────────────────────
    if momentum > 0.15:
        signal = 'bullish'
        strength = min(1.0, momentum * 1.5)
    elif momentum < -0.15:
        signal = 'bearish'
        strength = min(1.0, abs(momentum) * 1.5)
    else:
        signal = 'neutral'
        strength = 0.0

    # ── Деталь ──────────────────────────────────────────────────────────
    detail_parts = [
        f"up={up_ratio:.0%}",
        f"down={down_ratio:.0%}",
        f"div={divergence:+.2f}",
        f"POCtr={liq_trend:+.2f}",
    ]
    if accumulation > 0.3:
        detail_parts.append(f"📦acc={accumulation:.0%}")
    if distribution > 0.3:
        detail_parts.append(f"📤dist={distribution:.0%}")
    detail_parts.append(f"mom={momentum:+.2f}")
    detail_parts.append(f"→{signal}")

    return VsaResult(
        up_volume_ratio=up_ratio,
        down_volume_ratio=down_ratio,
        volume_divergence=divergence,
        accumulation=accumulation,
        distribution=distribution,
        liq_trend=liq_trend,
        momentum=momentum,
        signal=signal,
        strength=strength,
        detail=' '.join(detail_parts)
    )


# ──────────────────────────────────────────────────────────────────────────────
# МЕТРИКА "ПРИЦЕЛ"
# ──────────────────────────────────────────────────────────────────────────────


def compute_target_metric(
    mu: float,              # рыночное настроение (-1 до 1)
    liq_score: float,       # 0-100 от Liq (сила кластера)
    vv_score: float,        # 0-100 от VV (VWAP/объём)
    vsa_momentum: float,    # -1 до 1 от VSA
    mtf_signal: float,      # 0-100 от MTF
    rvb_signal: float,      # 0-100 от RVB
    adv_signal: float,      # 0-100 от Advisor
) -> Tuple[float, str]:
    """
    Метрика "Прицел" — насколько монета готова к движению.

    Веса:
    - VSA импульс + Liq = 40% (самый важный — объёмные деньги)
    - MTF контекст = 20% (куда смотрит долгий тренд)
    - VV = 15% (VWAP отклонение)
    - Рыночное настроение = 10%
    - RVB = 10%
    - Advisor = 5%

    Возвращает: (score 0-100, reason)
    """

    # liq_score уже 0-100, нормализуем
    liq_norm = liq_score / 100.0

    # VSA нормализуем
    vsa_norm = max(0, min(1, (vsa_momentum + 1) / 2))  # 0.0-1.0

    # Рыночное настроение
    mu_norm = max(0, min(1, (mu + 1) / 2))

    # MTF
    mtf_norm = mtf_signal / 100.0
    vv_norm = vv_score / 100.0
    rvb_norm = rvb_signal / 100.0
    adv_norm = adv_signal / 100.0

    score = (
        (liq_norm * 0.25 + vsa_norm * 0.15) * 40 +   # объёмный блок 40%
        mtf_norm * 20 +                                # MTF 20%
        vv_norm * 15 +                                 # VV 15%
        mu_norm * 10 +                                 # рынок 10%
        rvb_norm * 10 +                                # RVB 10%
        adv_norm * 5                                   # Advisor 5%
    )

    score = min(100, max(0, score))

    # Reason
    parts = []
    if liq_norm > 0.7:
        parts.append(f"Liq={liq_score:.0f}")
    if vsa_norm > 0.6:
        parts.append("VSA↑")
    if mtf_norm > 0.6:
        parts.append(f"MTF={mtf_signal:.0f}")
    if vv_norm > 0.6:
        parts.append("VWAP↑")

    reason = ' '.join(parts) if parts else 'нейтрально'

    return score, reason
