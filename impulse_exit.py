"""
Impulse Exit Detector — выход на пике движения по микро-свечам (1M)

Что делает:
1. Следит за 1M свечами после входа в позицию
2. Ищет признаки затухания импульса:
   - 📉 Объём падает, а цена ещё растёт (VSA дивергенция)
   - 📏 Тело свечи сокращается (свечи становятся "дожи")
   - 🎯 Длинные верхние тени на зелёных свечах (отторжение)
3. Даёт сигнал на выход ДО того, как трейлинг сработает

Цель: забрать 1.5-2% движения вместо 0.5-1% после прохода пика.
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 🛡️ Глобальный кеш последнего срабатывания (модульный, не привязан к self в трейдере)
_LAST_IMPULSE_TRIGGER: Dict[str, float] = {}  # symbol → timestamp

# ──────────────────────────────────────────────────────────────────────────────
# КОНФИГ
# ──────────────────────────────────────────────────────────────────────────────

LOOKBACK_CANDLES = 15          # Смотрим последние 15 свечей (15 минут)
VOLUME_DROP_THRESHOLD = 0.3    # Объём упал на 30% от пика → подозрительно
BODY_SHRINK_THRESHOLD = 0.4    # Тело свечи уменьшилось на 40% → затухание
WICK_THRESHOLD = 0.6          # Верхняя тень > 60% от всей свечи → отторжение
EXIT_SCORE_THRESHOLD = 65      # Порог для сигнала на выход (0-100)
STRONG_VOLUME_DROP = 0.5       # Объём упал на 50% → сильный сигнал


# ──────────────────────────────────────────────────────────────────────────────
# ДАТА-КЛАССЫ
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ImpulseResult:
    """Результат анализа импульса."""
    score: int                    # 0-100: насколько уверенно надо выходить
    exhaustion: bool              # Есть ли признаки затухания
    volume_divergence: float      # -1 до 1: дивергенция объёма и цены
    body_trend: float            # -1 до 1: тренд размера тел (-1 = сжимаются)
    wick_rejection: bool          # Отторжение на верхней тени
    signal: str                   # 'exit' | 'hold' | 'neutral'
    detail: str                   # Для логов


@dataclass
class Candle:
    """Универсальная свеча."""
    open: float
    high: float
    low: float
    close: float
    volume: float


# ──────────────────────────────────────────────────────────────────────────────
# ОСНОВНАЯ ФУНКЦИЯ
# ──────────────────────────────────────────────────────────────────────────────


def is_cooldown_active(symbol: str, cooldown_sec: int = 60) -> bool:
    """Проверка кулдауна импульсного выхода."""
    now = time.time()
    last = _LAST_IMPULSE_TRIGGER.get(symbol, 0.0)
    return (now - last) < cooldown_sec


def mark_trigger(symbol: str):
    """Запоминаем время срабатывания импульсного выхода."""
    _LAST_IMPULSE_TRIGGER[symbol] = time.time()


def detect_impulse_exhaustion(candles_1m: List[Dict], 
                                entry_price: float = None,
                                current_pnl_pct: float = None) -> ImpulseResult:
    """
    Анализ 1M свечей на признаки затухания импульса.
    
    Параметры:
        candles_1m: список 1M свечей [{o,h,l,c,v,t} или {open,high,low,close,volume}]
        entry_price: цена входа (если есть — для контекста)
        current_pnl_pct: текущий PnL% (если есть)
    
    Возвращает:
        ImpulseResult с решением: выходить или держать
    """
    if not candles_1m or len(candles_1m) < 5:
        return ImpulseResult(
            score=0, exhaustion=False,
            volume_divergence=0.0, body_trend=0.0,
            wick_rejection=False,
            signal='neutral', detail='мало данных'
        )
    
    # Нормализуем свечи
    candles = [_normalize_candle(c) for c in candles_1m[-LOOKBACK_CANDLES:]]
    
    if len(candles) < 5:
        return ImpulseResult(
            score=0, exhaustion=False,
            volume_divergence=0.0, body_trend=0.0,
            wick_rejection=False,
            signal='neutral', detail='мало свечей'
        )
    
    # ─── 1. VSA дивергенция: объём падает, цена растёт ────────────────────
    # Берём последние N свечей и смотрим тренд объёма vs тренд цены
    
    closes = np.array([c.close for c in candles])
    volumes = np.array([c.volume for c in candles])
    bodies = np.array([abs(c.close - c.open) for c in candles])
    ranges = np.array([c.high - c.low for c in candles])
    
    # Тренд объёма (полифит 1й степени: положительный = объём растёт)
    if len(volumes) >= 5:
        vol_slope = np.polyfit(range(len(volumes)), volumes, 1)[0]
        vol_trend = _normalize_slope(vol_slope, volumes.mean())
    else:
        vol_trend = 0.0
    
    # Тренд цены
    price_slope = np.polyfit(range(len(closes)), closes, 1)[0]
    price_trend = _normalize_slope(price_slope, closes.mean())
    
    # Дивергенция: +1 если цена↑ и объём↑ (здоровый рост)
    # -1 если цена↑ но объём↓ (бычий затухающий импульс → медвежий сигнал)
    if price_trend > 0.1:
        # Цена растёт — смотрим на объём
        volume_divergence = -vol_trend  # Отрицательная = объём падает при росте
    else:
        volume_divergence = vol_trend  # Цена не растёт — нейтрально
    
    volume_divergence = np.clip(volume_divergence, -1.0, 1.0)
    
    # ─── 2. Тренд размера тел свечей ───────────────────────────────────────
    # Если тела уменьшаются при росте цены — импульс затухает
    
    if len(bodies) >= 5 and bodies.mean() > 0:
        body_slope = np.polyfit(range(len(bodies)), bodies, 1)[0]
        body_trend = _normalize_slope(body_slope, bodies.mean())
    else:
        body_trend = 0.0
    
    # ─── 3. Отторжение на верхней тени ─────────────────────────────────────
    # Длинная верхняя тень на зелёной свече = продавцы вошли
    
    wick_rejection = False
    wick_score = 0.0
    
    for c in candles[-3:]:  # Смотрим последние 3 свечи
        if c.high > c.low:  # Избегаем деления на 0
            total_range = c.high - c.low
            if c.close >= c.open:  # Зелёная свеча
                upper_wick = c.high - c.close
                lower_wick = c.open - c.low
            else:  # Красная свеча — смотрим верхнюю тень
                upper_wick = c.high - max(c.open, c.close)
                lower_wick = min(c.open, c.close) - c.low
            
            wick_ratio = upper_wick / total_range if total_range > 0 else 0
            
            if wick_ratio > WICK_THRESHOLD:
                wick_rejection = True
                wick_score = max(wick_score, wick_ratio)
    
    # ─── 4. Итоговая оценка ────────────────────────────────────────────────
    
    score_components = []
    
    # VSA дивергенция
    if volume_divergence < -0.3:
        div_score = min(100, int(abs(volume_divergence) * 80))
        score_components.append(('volume_div', div_score))
    
    # Затухание тел
    if body_trend < -0.2 and price_trend > 0.05:
        body_score = min(100, int(abs(body_trend) * 70))
        score_components.append(('body_shrink', body_score))
    
    # Отторжение
    if wick_rejection:
        wick_score_int = min(100, int(wick_score * 90))
        score_components.append(('wick_rej', wick_score_int))
    
    # Мгновенный VSA: сравниваем объём последних 3 свечей с предыдущими 3
    if len(volumes) >= 6:
        vol_last3 = volumes[-3:].mean()
        vol_prev3 = volumes[-6:-3].mean()
        if vol_prev3 > 0 and vol_last3 < vol_prev3 * (1 - VOLUME_DROP_THRESHOLD):
            drop_pct = (1 - vol_last3 / vol_prev3) * 100
            vol_drop_score = min(100, int(drop_pct * 1.5))
            score_components.append(('vol_drop', vol_drop_score))
        
        # Сильное падение объёма
        if vol_prev3 > 0 and vol_last3 < vol_prev3 * (1 - STRONG_VOLUME_DROP):
            score_components.append(('vol_crash', 90))
    
    # Резкое сокращение тела
    if len(bodies) >= 4:
        body_last2 = bodies[-2:].mean()
        body_prev2 = bodies[-4:-2].mean()
        if body_prev2 > 0 and body_last2 < body_prev2 * (1 - BODY_SHRINK_THRESHOLD):
            score_components.append(('body_collapse', 85))
    
    # Если нет компонентов — импульс здоровый
    if not score_components:
        return ImpulseResult(
            score=0, exhaustion=False,
            volume_divergence=volume_divergence,
            body_trend=body_trend,
            wick_rejection=False,
            signal='hold',
            detail=f"импульс здоров: объём={'+' if vol_trend>0 else ''}{vol_trend:.2f} тела={'+' if body_trend>0 else ''}{body_trend:.2f}"
        )
    
    # Итоговый score: берём максимальный компонент
    max_score = max(s[1] for s in score_components)
    details = '+'.join(f"{k}={v}" for k, v in score_components)
    
    exhaustion = max_score >= EXIT_SCORE_THRESHOLD
    signal = 'exit' if exhaustion else ('caution' if max_score >= 40 else 'hold')
    
    return ImpulseResult(
        score=max_score,
        exhaustion=exhaustion,
        volume_divergence=volume_divergence,
        body_trend=body_trend,
        wick_rejection=wick_rejection,
        signal=signal,
        detail=f"score={max_score}: {details}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_candle(c: Dict) -> Candle:
    """Нормализует свечу из любого формата в Candle."""
    o = c.get('o') or c.get('open') or c.get('Open', 0)
    h = c.get('h') or c.get('high') or c.get('High', 0)
    l = c.get('l') or c.get('low') or c.get('Low', 0)
    cl = c.get('c') or c.get('close') or c.get('Close', 0)
    v = c.get('v') or c.get('volume') or c.get('Volume', 0)
    return Candle(open=o, high=h, low=l, close=cl, volume=v)


def _normalize_slope(slope: float, mean_val: float) -> float:
    """Нормализует наклон в диапазон -1..1."""
    if mean_val <= 0 or abs(mean_val) < 1e-9:
        return 0.0
    raw = slope / mean_val
    return np.clip(raw, -1.0, 1.0)


def exits_signals_to_str(result: ImpulseResult) -> str:
    """Краткая строка сигнала для логов."""
    emoji = '🚨' if result.exhaustion else ('⚠️' if result.score >= 40 else '✅')
    return f"{emoji} Импульс: {result.signal} (score={result.score}) {result.detail}"
