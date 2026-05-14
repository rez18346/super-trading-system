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
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# 🛡️ Глобальный кеш последнего срабатывания (модульный, не привязан к self в трейдере)
_LAST_IMPULSE_TRIGGER: Dict[str, float] = {}  # symbol → timestamp

# ──────────────────────────────────────────────────────────────────────────────
# КОНФИГ (дефолтный + переопределение из файла)
# ──────────────────────────────────────────────────────────────────────────────

_CONFIG_FILE = '/tmp/impulse_config.json'

def _get_config():
    """Читает конфиг из файла (меняется без перезапуска).
    
    Формат файла (JSON):
    {
      "exit_score_threshold": 65,    // порог выхода
      "min_confirmations": 1,        // сколько сигналов нужно (1 = любой один, 2 = два одновременно)
      "consecutive_confirmations": 1, // на скольких последовательных свечах
      "micro_trend_filter": false,   // не выходить по импульсу если микро-тренд идёт вверх
      "min_hold_seconds": 0,        // мин. время в позиции перед выходом (сек)
      "lookback_candles": 15,        // глубина анализа
      "volume_drop_threshold": 0.3,  // порог падения объёма
      "wick_threshold": 0.6,         // порог верхней тени
      "strong_volume_drop": 0.5,     // сильное падение объёма
      "body_shrink_threshold": 0.4   // порог сжатия тела
    }
    """
    defaults = {
        'exit_score_threshold': 65,
        'min_confirmations': 1,
        'consecutive_confirmations': 1,
        'micro_trend_filter': False,
        'min_hold_seconds': 0,
        'lookback_candles': 15,
        'volume_drop_threshold': 0.3,
        'wick_threshold': 0.6,
        'strong_volume_drop': 0.5,
        'body_shrink_threshold': 0.4,
    }
    try:
        if os.path.exists(_CONFIG_FILE):
            with open(_CONFIG_FILE) as f:
                overrides = json.load(f)
                defaults.update(overrides)
    except Exception as e:
        logger.warning(f"[IMPULSE] config read error: {e}")
    return defaults


# Для обратной совместимости — синглтон кеша
_cached_config = {}

# ──────────────────────────────────────────────────────────────────────────────
# ДАТА-КЛАССЫ
# ──────────────────────────────────────────────────────────────────────────────


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
    
    # Динамический конфиг (меняется через /tmp/impulse_config.json на лету)
    cfg = _get_config()
    lookback = cfg['lookback_candles']
    vol_drop_thresh = cfg['volume_drop_threshold']
    body_shrink_thresh = cfg['body_shrink_threshold']
    wick_thresh = cfg['wick_threshold']
    exit_thresh = cfg['exit_score_threshold']
    strong_vol_drop = cfg['strong_volume_drop']
    min_confirmations = cfg['min_confirmations']
    min_hold = cfg['min_hold_seconds']
    
    # Нормализуем свечи
    candles = [_normalize_candle(c) for c in candles_1m[-lookback:]]
    
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
            
            if wick_ratio > wick_thresh:
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
        if vol_prev3 > 0 and vol_last3 < vol_prev3 * (1 - vol_drop_thresh):
            drop_pct = (1 - vol_last3 / vol_prev3) * 100
            vol_drop_score = min(100, int(drop_pct * 1.5))
            score_components.append(('vol_drop', vol_drop_score))
        
        # Сильное падение объёма
        if vol_prev3 > 0 and vol_last3 < vol_prev3 * (1 - strong_vol_drop):
            score_components.append(('vol_crash', 90))
    
    # Резкое сокращение тела
    if len(bodies) >= 4:
        body_last2 = bodies[-2:].mean()
        body_prev2 = bodies[-4:-2].mean()
        if body_prev2 > 0 and body_last2 < body_prev2 * (1 - body_shrink_thresh):
            score_components.append(('body_collapse', 85))
    
    # ─── 4b. Проверка последовательных свечей ──────────────────────────────
    # Сигнал должен подтвердиться на нескольких группах свечей подряд
    consecutive_conf = cfg.get('consecutive_confirmations', 1)
    if consecutive_conf > 1 and len(candles) >= 10 and score_components:
        # Разбиваем свечи на две группы: последние 5 и предыдущие 5
        recent5 = candles[-5:]
        prev5 = candles[-10:-5]
        
        def _check_vol_drop(grp):
            """Проверяет падение объёма на группе свечей."""
            if len(grp) < 4:
                return False
            vol_group = np.array([c.volume for c in grp])
            last3 = vol_group[-3:].mean()
            prev3 = vol_group[-6:-3].mean() if len(vol_group) >= 6 else 0
            if len(vol_group) >= 6 and prev3 > 0:
                return last3 < prev3 * (1 - vol_drop_thresh)
            return False
        
        def _check_body_collapse(grp):
            """Проверяет схлопывание тела на группе свечей."""
            if len(grp) < 4:
                return False
            body_group = np.array([abs(c.close - c.open) for c in grp])
            last2 = body_group[-2:].mean()
            prev2 = body_group[-4:-2].mean()
            return prev2 > 0 and last2 < prev2 * (1 - body_shrink_thresh)
        
        def _check_wick(grp):
            """Проверяет верхние тени на группе свечей."""
            for c in grp[-3:]:
                if c.high > c.low:
                    rng = c.high - c.low
                    upper_wick = c.high - max(c.close, c.open) if c.close >= c.open else c.high - c.close
                    if upper_wick / rng > wick_thresh:
                        return True
            return False
        
        # Какие типы сигналов есть в общей картине
        has_vol_signal = any(k in ['vol_drop', 'vol_crash', 'volume_div'] for k, _ in score_components)
        has_body_signal = any(k in ['body_collapse', 'body_shrink'] for k, _ in score_components)
        has_wick_signal = any(k == 'wick_rej' for k, _ in score_components)
        
        # ПРОВЕРКА: подтверждены ли сигналы в ПРЕДЫДУЩЕЙ группе свечей?
        recent_vol = _check_vol_drop(recent5)
        prev_vol = _check_vol_drop(prev5)
        recent_body = _check_body_collapse(recent5)
        prev_body = _check_body_collapse(prev5)
        recent_wick = _check_wick(recent5)
        prev_wick = _check_wick(prev5)
        
        # Собираем неподтверждённые сигналы (есть в recent5, но нет в prev5)
        unconfirmed = []
        if has_vol_signal and recent_vol and not prev_vol:
            unconfirmed.append(f'vol_drop')
        if has_body_signal and recent_body and not prev_body:
            unconfirmed.append(f'body_collapse')
        if has_wick_signal and recent_wick and not prev_wick:
            unconfirmed.append(f'wick_rej')
        
        if unconfirmed:
            # Сигналы только на последней группе — возможно случайность
            # Понижаем уверенность: фильтруем неподтверждённые компоненты
            old_len = len(score_components)
            score_components = [
                s for s in score_components
                if not (
                    (s[0] == 'vol_drop' and not prev_vol and recent_vol) or
                    (s[0] == 'vol_crash' and not prev_vol and recent_vol) or
                    (s[0] == 'body_collapse' and not prev_body and recent_body) or
                    (s[0] == 'wick_rej' and not prev_wick and recent_wick)
                )
            ]
            if len(score_components) < old_len:
                details += f' (отсев {len(unconfirmed)} неподтв.)'
    
    # Если после отсева не осталось компонентов — импульс здоровый
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
    
    # ─── 5. Микро-тренд: не выходим на растущем рынке ────────────────────
    # Если последние свечи идут вверх — импульсное затухание может быть
    # временным дыханием, а не разворотом. Пропускаем такие выходы.
    micro_trend_filter = cfg.get('micro_trend_filter', False)
    if micro_trend_filter and len(closes) >= 8 and max_score >= exit_thresh:
        # Смотрим: последние 5 свечей выше, чем предыдущие 5?
        recent_avg = closes[-4:].mean()
        prev_avg = closes[-8:-4].mean()
        
        # Проверка на восходящий микро-тренд
        if recent_avg > prev_avg * 1.001:  # Хотя бы 0.1% роста
            # Проверяем также последовательность: последние 3 close ВЫШЕ средних 3 из предыдущих?
            last_highs = closes[-3:]
            prev_highs = closes[-6:-3]
            
            # Все последние 3 выше, чем предыдущие?
            if all(l >= prev_highs.mean() for l in last_highs):
                # Микро-тренд ВВЕРХ — отменяем импульсный выход
                suppressed = len([s for s in score_components if s[1] >= exit_thresh])
                exhaustion = False
                signal = 'hold'
                details += f' (↑ тренд, {suppressed} сигн. подавлено)'
                return ImpulseResult(
                    score=max_score,
                    exhaustion=False,
                    volume_divergence=volume_divergence,
                    body_trend=body_trend,
                    wick_rejection=wick_rejection,
                    signal='hold',
                    detail=f"score={max_score}: {details}"
                )
    
    # Проверка: нужно достаточно подтверждений
    if min_confirmations > 1:
        # Требуется несколько сигналов
        strong_signals = [s for s in score_components if s[1] >= exit_thresh]
        exhaustion = len(strong_signals) >= min_confirmations
        if exhaustion:
            details += f' (×{len(strong_signals)} сигналов)'
    else:
        exhaustion = max_score >= exit_thresh
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
