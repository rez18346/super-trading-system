"""
RP (Recovery & Potential) Analyzer

Оценивает потенциал монеты для входа на основе:
  1. Глубины просадки (drawdown от 24h хая)
  2. Аккумуляции vs тихого слива (CVD buy_pct + trend)
  3. Скрытой силы/слабости (VSA)
  4. Волатильности (диапазон 24h как прокси ATR)

Не заменяет существующие модули, а даёт ДЕЛЬТУ к score:
  −20 … +20 баллов в зависимости от качества монеты.
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def analyze_recovery_potential(
    symbol: str,
    current_price: float,
    high_24h: Optional[float] = None,
    low_24h: Optional[float] = None,
    cvd_data: Optional[Dict] = None,
    vsa_signal: str = 'neutral',
    vsa_strength: float = 0.0,
    vsa_divergence: float = 0.0,
) -> Dict:
    """
    Анализирует Recovery & Potential монеты.

    Возвращает:
      {
        'rp_score': 0..100,     # Сырой RP score
        'rp_delta': -20..+20,   # Дельта к финальному score
        'drawdown_score': 0..100,
        'accumulation_score': 0..100,
        'hidden_strength_score': 0..100,
        'volatility_score': 0..100,
        'detail': str,          # Краткое описание для логов
      }
    """
    drawdown_score = _score_drawdown(current_price, high_24h)
    accumulation_score = _score_accumulation(cvd_data)
    hidden_strength_score = _score_hidden_strength(vsa_signal, vsa_strength, vsa_divergence)
    volatility_score = _score_volatility(current_price, high_24h, low_24h)

    # Итоговый RP score: взвешенное среднее
    rp_score = (
        drawdown_score * 0.25 +
        accumulation_score * 0.30 +
        hidden_strength_score * 0.25 +
        volatility_score * 0.20
    )
    rp_score = max(0, min(100, rp_score))

    # Конвертируем RP score в дельту
    rp_delta = _score_to_delta(rp_score)

    # Деталь для логов
    detail = _build_detail(
        drawdown_score, accumulation_score,
        hidden_strength_score, volatility_score,
        rp_score, rp_delta,
        high_24h, current_price,
        cvd_data, vsa_signal
    )

    return {
        'rp_score': round(rp_score, 1),
        'rp_delta': rp_delta,
        'drawdown_score': round(drawdown_score, 1),
        'accumulation_score': round(accumulation_score, 1),
        'hidden_strength_score': round(hidden_strength_score, 1),
        'volatility_score': round(volatility_score, 1),
        'detail': detail,
    }


# ──────────────────────────────────────────────────────────────────────────────
# КОМПОНЕНТ 1: DRAWDDOWN — глубина просадки от 24h хая
# ──────────────────────────────────────────────────────────────────────────────

def _score_drawdown(current_price: Optional[float],
                    high_24h: Optional[float]) -> float:
    """Оценивает просадку от 24h максимума.

    0-2%     → 40 баллов  (плохо — не на чем отскакивать)
    2-5%     → 60         (слабая просадка, может быть вершина)
    5-10%    → 80         ⭐ идеально — глубокая, но не критическая коррекция
    10-20%   → 75         (глубокая просадка — может быть oversold)
    20-40%   → 50         (очень глубоко — может быть проблема)
    40%+     → 20         (мёртвая монета)
    """
    if not current_price or not high_24h or high_24h <= 0:
        return 50  # нейтрально

    drawdown_pct = (high_24h - current_price) / high_24h * 100
    drawdown_pct = max(0, min(100, drawdown_pct))

    if drawdown_pct < 1:
        return 30  # на хае — нет просадки, плохо для входа
    elif drawdown_pct < 3:
        return 45  # мелкая просадка
    elif drawdown_pct < 5:
        return 60  # небольшая, но уже есть потенциал
    elif drawdown_pct < 8:
        return 85  # ⭐ отличная глубина
    elif drawdown_pct < 12:
        return 80  # хорошая
    elif drawdown_pct < 18:
        return 72  # ещё норм
    elif drawdown_pct < 25:
        return 60  # глубже среднего
    elif drawdown_pct < 40:
        return 40  # глубокая — осторожно
    else:
        return 15  # >40% — мёртвая зона


# ──────────────────────────────────────────────────────────────────────────────
# КОМПОНЕНТ 2: ACCUMULATION — накопление vs тихий слив (через CVD)
# ──────────────────────────────────────────────────────────────────────────────

def _score_accumulation(cvd_data: Optional[Dict]) -> float:
    """Анализирует накопление или слив через CVD buy_pct и trend.

    buy_pct > 55%, trend=bullish → накопление (90)
    buy_pct < 40%, trend=bearish → тихий слив (15) 🚫
    buy_pct ~50% → нейтрально (50)
    """
    if not cvd_data:
        return 50  # нет данных — нейтрально

    buy_pct = cvd_data.get('buy_pct', 50)
    trend = cvd_data.get('trend', 'neutral')
    cvd_ratio = cvd_data.get('cvd_ratio', 1.0)

    if buy_pct is None:
        return 50

    # Базовая оценка от buy_pct
    # 50% → 50, 60% → 70, 70% → 90, 40% → 30, 30% → 10
    base_score = 50 + (buy_pct - 50) * 2.0
    base_score = max(10, min(90, base_score))

    # Модификатор от trend
    if trend == 'bullish':
        base_score += 10
    elif trend == 'bearish':
        base_score -= 10

    # CVD ratio: > 1.5 = больше buy-объёма (накопление)
    if cvd_ratio > 1.5:
        base_score += 8
    elif cvd_ratio < 0.67:
        base_score -= 8

    return max(10, min(100, base_score))


# ──────────────────────────────────────────────────────────────────────────────
# КОМПОНЕНТ 3: HIDDEN STRENGTH — скрытая сила/слабость (через VSA)
# ──────────────────────────────────────────────────────────────────────────────

def _score_hidden_strength(signal: str, strength: float,
                           divergence: float) -> float:
    """VSA показывает скрытую силу или слабость.

    bull + divergence положительная → скрытая сила (85)
    bear + divergence отрицательная → подтверждение слабости (20)
    """
    base = 50

    if signal == 'bullish':
        base += 20 + int(strength * 30)  # 50 + 20..50 = 70..100
    elif signal == 'bearish':
        base -= 20 + int(strength * 20)  # 50 - 20..20 = 30..50

    # Дивергенция усиливает
    if divergence < -0.3:
        base -= 10  # bearish divergence — скрытая слабость
    elif divergence > 0.3:
        base += 10  # bullish divergence — скрытая сила

    return max(10, min(100, base))


# ──────────────────────────────────────────────────────────────────────────────
# КОМПОНЕНТ 4: VOLATILITY — волатильность как прокси потенциала движения
# ──────────────────────────────────────────────────────────────────────────────

def _score_volatility(current_price: Optional[float],
                      high_24h: Optional[float],
                      low_24h: Optional[float]) -> float:
    """Чем шире 24h диапазон, тем выше потенциал движения.

    < 2%    → 30 (вялая — мало потенциала)
    2-4%    → 50 (средняя)
    4-8%    → 80 (⭐ хорошая волатильность)
    8-15%   → 70 (высокая — может быть хаотичной)
    > 15%   → 40 (экстремальная — опасна)
    """
    if not all([current_price, high_24h, low_24h]):
        return 50  # нейтрально

    if high_24h <= low_24h or current_price <= 0:
        return 50

    range_pct = (high_24h - low_24h) / low_24h * 100
    range_pct = max(0, min(100, range_pct))

    if range_pct < 1:
        return 25  # почти нет движения
    elif range_pct < 2:
        return 40  # слабая
    elif range_pct < 3:
        return 55  # умеренная
    elif range_pct < 5:
        return 75  # ⭐ хорошая
    elif range_pct < 8:
        return 85  # ⭐ отличная вола
    elif range_pct < 12:
        return 70  # высокая, но ок
    elif range_pct < 18:
        return 55  # очень высокая
    else:
        return 35  # экстремальная


# ──────────────────────────────────────────────────────────────────────────────
# КОНВЕРТАЦИЯ RP SCORE → ДЕЛЬТА
# ──────────────────────────────────────────────────────────────────────────────

def _score_to_delta(rp_score: float) -> int:
    """Конвертирует RP score (0-100) в дельту к финальному score (-20..+20)."""
    if rp_score >= 90:
        return 20
    elif rp_score >= 80:
        return 15
    elif rp_score >= 70:
        return 10
    elif rp_score >= 60:
        return 5
    elif rp_score >= 45:
        return 0
    elif rp_score >= 35:
        return -5
    elif rp_score >= 25:
        return -10
    elif rp_score >= 15:
        return -15
    else:
        return -20


# ──────────────────────────────────────────────────────────────────────────────
# ФОРМИРОВАНИЕ ДЕТАЛИ ДЛЯ ЛОГОВ
# ──────────────────────────────────────────────────────────────────────────────

def _build_detail(dd_score: float, acc_score: float,
                  hs_score: float, vol_score: float,
                  rp_score: float, rp_delta: int,
                  high_24h: Optional[float],
                  current_price: Optional[float],
                  cvd_data: Optional[Dict],
                  vsa_signal: str) -> str:
    """Краткое описание RP для логов."""
    parts = []

    # Drawdown
    if current_price and high_24h and high_24h > 0:
        dd_pct = (high_24h - current_price) / high_24h * 100
        parts.append(f"DD={dd_pct:.1f}%")
    else:
        parts.append("DD=N/A")

    # Accumulation
    if cvd_data:
        buy_pct = cvd_data.get('buy_pct', 50)
        parts.append(f"acc={buy_pct:.0f}%")
    else:
        parts.append("acc=N/A")

    # VSA
    parts.append(f"vsa={vsa_signal[:3]}")

    # RP total
    delta_sign = '+' if rp_delta >= 0 else ''
    parts.append(f"RP={delta_sign}{rp_delta}")

    return ' '.join(parts)
