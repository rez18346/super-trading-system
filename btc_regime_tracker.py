#!/usr/bin/env python3
"""
BTC Regime Tracker — модуль определения фазы движения BTC в реальном времени.

Анализирует 5M и 1H свечи BTC, определяет фазу рынка (pump, dump, accumulation,
distribution, recovery) и выдаёт рекомендации: когда входить, когда сидеть,
а когда принудительно фиксировать прибыль.

Зависимости: numpy, pandas (уже установлены).
"""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Пороговые константы
# ──────────────────────────────────────────────
PUMP_THRESHOLD = 1.5       # % роста за 30 мин → pump
DUMP_THRESHOLD = -1.5      # % падения за 30 мин → dump
VOLUME_SIGMA = 2.0         # количество сигм для аномального объёма
DEFAULT_COOLDOWN = 7200    # кулдаун после dump (сек) — по умолч. 2ч
DISTRIBUTION_MIN_BARS = 4  # минимум баров 5M для распознавания distribution


class BTCRegimeTracker:
    """Определяет фазу движения BTC на основе свечных данных."""

    def __init__(self, reentry_cooldown_seconds: int = DEFAULT_COOLDOWN):
        """
        Args:
            reentry_cooldown_seconds:  время в секундах, в течение которого
                                       после dump запрещены покупки.
        """
        self._cooldown_seconds = reentry_cooldown_seconds
        self._cooldown_until: float = 0.0
        self._regime: str = "unknown"
        self._htf_trend: str = "unknown"   # глобальный тренд: 'up', 'down', 'sideways'
        self._structure_trend: str = "unknown"  # 'bullish', 'bearish', 'neutral'
        self._last_result: dict = {}
        # 🆕 ATR-фильтр: если ATR(14)_1h < порога — volatility_block
        self._atr_1h: float = 0.0
        self._volatility_block: bool = False
        # 🆕 Range-блок: BTC в боковике без пробоя — альты не торгуем
        self._range_block: bool = False
        self._btc_current_price: float = 0.0
        # 🆕 Трекер последовательных падений: буфер последних 5 значений change_30m
        self._change_30m_buffer: list = []

    # ── публичный API ────────────────────────

    def update(self,
               btc_5m_candles: pd.DataFrame,
               btc_1h_candles: pd.DataFrame,
               current_price: float = 0.0) -> dict:
        """
        Анализирует свечи BTC и возвращает словарь с фазой, изменениями,
        рекомендацией и человекочитаемым сообщением.

        Args:
            btc_5m_candles:  DataFrame с колонками ['open','high','low','close','volume']
                             в хронологическом порядке (первая строка — самая старая).
            btc_1h_candles:  то же самое для часовых свечей.
            current_price:   Текущая цена BTC для проверки range-блока (опционально).

        Returns:
            dict с полями:
                regime, btc_change_30m, btc_change_1h, btc_change_4h,
                avg_volume_30m, recommendation, cooldown_until, message
        """
        self._validate_input(btc_5m_candles, btc_1h_candles)

        # 1. Считаем изменения цены за разные окна
        change_30m = self._price_change_percent(btc_5m_candles, 6)    # 6 × 5M = 30 мин
        change_1h  = self._price_change_percent(btc_1h_candles, 1)    # 1 час
        change_4h  = self._price_change_percent(btc_1h_candles, 4)    # 4 часа

        # 2. Объёмный анализ (на 5M свечах)
        avg_vol_30m, vol_zscore = self._volume_analysis(btc_5m_candles, window=6)

        # 2.5. ATR(14) на 1H — волатильность
        self._atr_1h = self._calc_atr_14(btc_1h_candles)
        self._volatility_block = self._atr_1h < 200.0
        if self._volatility_block:
            logger.info(f"🔇 Volatility block: ATR(14)_1h={self._atr_1h:.1f} < 200 — альты не трогаем")

        # 2.6. Range-блок — BTC в боковике без BOS (экспертный диапазон $58,450–$60,918)
        self._btc_current_price = current_price if current_price else 0
        self._range_block = self._is_in_sideways_range()
        if self._range_block:
            logger.info(f"🔇 Range block: BTC=${self._btc_current_price:.0f} в боковике $58,450-$60,918 — альты не трогаем")

        # 2.7. Трекер последовательных падений
        # Если 3 проверки подряд показывают падение за 30 минут — bearish_side
        self._change_30m_buffer.append(change_30m)
        if len(self._change_30m_buffer) > 5:
            self._change_30m_buffer.pop(0)
        
        consecutive_negative = sum(1 for v in self._change_30m_buffer if v < 0)
        if len(self._change_30m_buffer) >= 3 and consecutive_negative >= 3:
            # Если все последние 3 проверки отрицательные — это устойчивое падение
            avg_drop = sum(self._change_30m_buffer[-3:]) / 3
            logger.debug(f"Consecutive drops: {len(self._change_30m_buffer)} checks, "
                         f"{consecutive_negative} negative, avg={avg_drop:.2f}%")
            if avg_drop <= -0.3:  # среднее падение > 0.3% за 30 мин — реальный слив
                logger.warning(f"🛡️ {consecutive_negative} последовательных падений подряд (avg={avg_drop:.2f}%) → bearish_side")
                regime = "bearish_side"
                self._regime = regime

        # 3. Определяем фазу
        regime = self._classify_regime(
            change_30m, change_1h, change_4h,
            vol_zscore, btc_5m_candles
        )
        self._regime = regime

        # 4. Глобальный тренд — определяем по старшим таймфреймам
        self._htf_trend = self._detect_htf_trend(btc_1h_candles)
        
        # 5. Анализ структуры (Lower High + Lower Low на 1H)
        self._structure_trend = self._detect_structure(btc_1h_candles)
        
        # 6. 🛡️ ЗАЩИТА ОТ ЛОЖНЫХ РАЗВОРОТОВ
        # Если HTF=down или structure=bearish — БЛОКИРУЕМ любые bullish-режимы
        if self._htf_trend == 'down' and regime in ('accumulation', 'recovery', 'pump'):
            logger.warning(f"🛡️ HTF={self._htf_trend}, regime={regime} → принудительно bearish_side (защита от ложного разворота)")
            regime = 'bearish_side'
            self._regime = regime
        
        if self._structure_trend == 'bearish' and regime in ('accumulation', 'recovery', 'pump'):
            logger.info(f"🛡️ Структура={self._structure_trend}, regime={regime} → bearish_side")
            regime = 'bearish_side'
            self._regime = regime
        
        # 7. Если regime пытается смениться с bearish на что-то — задержка 10 минут
        if hasattr(self, '_prev_regime') and self._prev_regime in ('dump', 'bearish_side', 'distribution'):
            if regime not in ('dump', 'bearish_side', 'distribution'):
                # Смена с медвежьего на бычий — проверяем время
                now = time.time()
                if not hasattr(self, '_last_bearish_change'):
                    self._last_bearish_change = now
                
                if now - self._last_bearish_change < 120:  # 2 минуты
                    logger.warning(f"🛡️ VETO: смена {self._prev_regime}→{regime} заблокирована (прошло {now - self._last_bearish_change:.0f}с, нужно 120с)")
                    regime = self._prev_regime
                    self._regime = regime
                else:
                    self._last_bearish_change = 0
        
        # Запоминаем текущий режим для следующего цикла
        self._prev_regime = regime
        
        if regime in ('dump', 'bearish_side'):
            self._last_bearish_change = time.time()
        
        # 8. Рекомендация
        recommendation = self._get_recommendation(regime)

        self._last_result = {
            "regime":          regime,
            "htf_trend":       self._htf_trend,
            "structure_trend": self._structure_trend,
            "btc_change_30m":  round(change_30m, 2),
            "btc_change_1h":   round(change_1h, 2),
            "btc_change_4h":   round(change_4h, 2),
            "avg_volume_30m":  round(avg_vol_30m, 2),
            "recommendation":  recommendation,
            "cooldown_until":  self._cooldown_until,
            "message":         self._build_message(regime, change_30m, recommendation, change_4h=change_4h),
            "atr_1h":         round(self._atr_1h, 1),
            "volatility_block": self._volatility_block,
            "range_block": self._range_block,
            "btc_price":     round(self._btc_current_price, 1),
        }

        logger.info("Regime=%s, HTF=%s, Структура=%s, 30m=%.2f%%, rec=%s", regime, self._htf_trend, self._structure_trend, change_30m, recommendation)
        return self._last_result

    def is_buy_allowed(self) -> bool:
        """Можно ли сейчас покупать? Проверяет кулдаун, фазу и глобальный тренд."""
        if time.time() < self._cooldown_until:
            return False
        if self._regime in ("pump", "distribution"):
            return False
        if self._regime in ("dump", "bearish_side", "recovery"):
            return False
        # 🛑 HTF down → блокируем ВСЕ лонги (любая локальная фаза в даун-тренде = ловушка)
        if self._htf_trend == "down":
            return False
        # 🛑 Медвежья структура → блокируем лонги
        if self._structure_trend == "bearish":
            return False
        # 🔇 Низкая волатильность → не входим в альты
        if self._volatility_block:
            logger.info(f"🔇 Volatility block: ATR(14)_1h={self._atr_1h:.1f} < 200 — BUY заблокирован")
            return False
        # 🔇 BTC в боковике — не входим в альты (low-volume chop)
        if self._range_block:
            logger.info(f"🔇 Range block: BTC=${self._btc_current_price:.0f} в боковике $58,450-$60,918 — BUY заблокирован")
            return False
        return True

    def _is_in_sideways_range(self, lo=58450, hi=60918) -> bool:
        """Проверяет, находится ли BTC внутри экспертного диапазона low-volume chop."""
        if self._btc_current_price <= 0:
            return False
        return lo <= self._btc_current_price <= hi

    def is_short_allowed(self) -> bool:
        """Можно ли сейчас шортить? Разрешён на медвежьем рынке."""
        if time.time() < self._cooldown_until:
            return False

        # ═══ ГЛОБАЛЬНЫЕ БЛОКИРОВКИ (проверяются до фазовых разрешений) ═══
        # 🔇 BTC в боковике — не входим в альты (low-volume chop, ни LONG ни SHORT)
        if self._range_block:
            logger.info(f"🔇 Range block: BTC=${self._btc_current_price:.0f} в боковике $58,450-$60,918 — SHORT заблокирован")
            return False
        # 🔇 Низкая волатильность → не входим в альты (даже шорт)
        if self._volatility_block:
            logger.info(f"🔇 Volatility block: ATR(14)_1h={self._atr_1h:.1f} < 200 — SHORT заблокирован")
            return False

        # Шорт разрешён на падающих фазах
        if self._regime in ("bearish_side", "distribution", "dump"):
            return True
        # HTF down → шорт разрешён
        if self._htf_trend == "down":
            return True
        # Медвежья структура → шорт разрешён
        if self._structure_trend == "bearish":
            return True
        # В остальных случаях (accumulation, recovery, pump) — шорт опасен
        return False

    def get_regime(self) -> str:
        """Возвращает текущую фазу."""
        return self._regime

    def get_direction(self) -> str:
        """Возвращает направление рынка на основе фазы: 'up', 'down' или 'neutral'."""
        _dir_map = {
            'pump': 'up',
            'dump': 'down',
            'distribution': 'down',
            'bearish_side': 'down',
            'recovery': 'up',
        }
        return _dir_map.get(self._regime, 'neutral')

    def get_structure(self) -> str:
        """Возвращает тренд по структуре: 'bullish', 'bearish', 'neutral'."""
        return self._structure_trend

    # ── внутренние методы ────────────────────

    def _detect_htf_trend(self, df_1h: pd.DataFrame) -> str:
        """
        Определяет глобальный тренд по 1H свечам (100 свечей ≈ 4 дня).
        Использует EMA50 vs EMA100 с процентным отступом.
        
        Returns: 'up', 'down' или 'sideways'
        """
        if df_1h is None or len(df_1h) < 50:
            return 'unknown'

        close = df_1h['close'].values.astype(float)
        
        # EMA50 (≈ 50 часов = 2 дня)
        ema50 = self._ema(close, 50)
        # EMA100 (≈ 100 часов = 4 дня)
        ema100 = self._ema(close, 100)
        
        current_price = close[-1]
        
        # Процентный отступ — чтобы не срабатывать на касании
        pct_from_ema100 = (current_price - ema100[-1]) / ema100[-1] * 100
        ema_cross_gap = (ema50[-1] - ema100[-1]) / ema100[-1] * 100
        
        # Уверенный downtrend: цена минимум на 1.5% ниже EMA100
        if pct_from_ema100 < -1.5 and ema_cross_gap < -0.5:
            return 'down'
        # Уверенный uptrend: цена минимум на 1.5% выше EMA100
        elif pct_from_ema100 > 1.5 and ema_cross_gap > 0.5:
            return 'up'
        else:
            return 'sideways'

    def _detect_structure(self, df_1h: pd.DataFrame) -> str:
        """
        Анализирует структуру тренда на 1H свечах.
        
        Ищет последовательность Lower High + Lower Low (медвежья структура)
        или Higher High + Higher Low (бычья структура).
        
        Returns: 'bullish', 'bearish', или 'neutral'
        """
        if df_1h is None or len(df_1h) < 40:
            return 'unknown'

        high = df_1h['high'].values.astype(float)
        low = df_1h['low'].values.astype(float)
        n = len(high)

        # Ищем swing highs (локальные максимумы) и swing lows (локальные минимумы)
        # на последних 40 свечах (~40 часов)
        lookback = min(40, n)
        
        swing_highs = []
        swing_lows = []
        
        for i in range(n - lookback, n):
            # Swing high: выше 2 соседей слева и 2 соседей справа
            if i >= 2 and i <= n - 3:
                if (high[i] > high[i-1] and high[i] > high[i-2] and
                    high[i] >= high[i+1] and high[i] >= high[i+2]):
                    swing_highs.append({'idx': i, 'val': high[i]})
            
            # Swing low: ниже 2 соседей слева и 2 соседей справа
            if i >= 2 and i <= n - 3:
                if (low[i] < low[i-1] and low[i] < low[i-2] and
                    low[i] <= low[i+1] and low[i] <= low[i+2]):
                    swing_lows.append({'idx': i, 'val': low[i]})

        # Без достаточного количества свингов — нейтрально
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'neutral'

        # Берём последние 2 свинга каждого типа
        last_2_highs = swing_highs[-2:]
        last_2_lows = swing_lows[-2:]

        # Lower High: последний максимум ниже предыдущего
        lower_high = last_2_highs[1]['val'] < last_2_highs[0]['val']
        # Lower Low: последний минимум ниже предыдущего
        lower_low = last_2_lows[1]['val'] < last_2_lows[0]['val']
        
        # Higher High: последний максимум выше предыдущего
        higher_high = last_2_highs[1]['val'] > last_2_highs[0]['val']
        # Higher Low: последний минимум выше предыдущего
        higher_low = last_2_lows[1]['val'] > last_2_lows[0]['val']
        
        # Медвежья структура: Lower High + Lower Low
        if lower_high and lower_low:
            logger.debug(f"\U0001f989 Медвежья структура: LH={last_2_highs[1]['val']:.1f} < {last_2_highs[0]['val']:.1f}, "
                         f"LL={last_2_lows[1]['val']:.1f} < {last_2_lows[0]['val']:.1f}")
            return 'bearish'
        
        # Бычья структура: Higher High + Higher Low
        if higher_high and higher_low:
            logger.debug(f"\U0001f402 Бычья структура: HH={last_2_highs[1]['val']:.1f} > {last_2_highs[0]['val']:.1f}, "
                         f"HL={last_2_lows[1]['val']:.1f} > {last_2_lows[0]['val']:.1f}")
            return 'bullish'
        
        # Неопределённая структура
        return 'neutral'

    @staticmethod
    def _ema(values: np.ndarray, period: int) -> np.ndarray:
        """Простое экспоненциальное скользящее среднее."""
        result = np.zeros_like(values)
        result[:period] = np.mean(values[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(values)):
            result[i] = (values[i] - result[i-1]) * multiplier + result[i-1]
        return result

    @staticmethod
    def _validate_input(df_5m: pd.DataFrame, df_1h: pd.DataFrame) -> None:
        """Проверяет, что в DataFrame есть необходимые колонки и данные."""
        required = {"open", "high", "low", "close", "volume"}
        for name, df in [("5m", df_5m), ("1h", df_1h)]:
            if df.empty:
                raise ValueError(f"DataFrame {name} пуст")
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"В {name} свечах не хватает колонок: {missing}")

    @staticmethod
    def _price_change_percent(df: pd.DataFrame, bars: int) -> float:
        """
        Возвращает % изменения цены close за последние `bars` свечей.
        Если данных меньше, чем `bars`, считает по доступным.
        """
        closes = df["close"].values
        if len(closes) < 2:
            return 0.0
        n = min(bars, len(closes) - 1)
        start_price = closes[-(n + 1)]
        end_price = closes[-1]
        if start_price == 0:
            return 0.0
        return (end_price - start_price) / start_price * 100.0

    def _calc_atr_14(self, df_1h: pd.DataFrame) -> float:
        """
        Рассчитывает ATR(14) на 1H свечах BTC.
        Если данных < 15 свечей — возвращает 0 (фильтр не работает).
        """
        if df_1h is None or df_1h.empty or len(df_1h) < 15:
            return 0.0
        try:
            high = df_1h['high'].values
            low = df_1h['low'].values
            close = df_1h['close'].values
            tr = np.zeros(len(df_1h))
            for i in range(1, len(df_1h)):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr[i] = max(hl, hc, lc)
            atr = np.mean(tr[-14:])
            return float(atr)
        except Exception as e:
            logger.warning(f"[ATR] Ошибка расчёта: {e}")
            return 0.0

    @staticmethod
    def _volume_analysis(df_5m: pd.DataFrame, window: int = 6):
        """
        Анализирует объём на `window` последних свечах 5M.
        Возвращает (avg_volume, zscore) относительно предшествующей истории.
        """
        volumes = df_5m["volume"].values
        if len(volumes) < window + 5:
            return float(np.mean(volumes[-window:])), 0.0

        recent = volumes[-window:]                     # последние window свечей
        history = volumes[:-(window + 1)]              # всё до них (но не пересекаем)

        avg_recent = float(np.mean(recent))
        if len(history) < 5:
            return avg_recent, 0.0

        hist_mean = float(np.mean(history))
        hist_std  = float(np.std(history, ddof=1))
        if hist_std < 1e-9:
            return avg_recent, 0.0

        zscore = (avg_recent - hist_mean) / hist_std
        return avg_recent, zscore

    def _classify_regime(self,
                         change_30m: float,
                         change_1h: float,
                         change_4h: float,
                         vol_zscore: float,
                         df_5m: pd.DataFrame) -> str:
        """
        Классифицирует фазу рынка.

        Приоритет:
            1. Pump / Dump  (резкое движение)
            2. Recovery     (отскок после dump — умеренный рост на фоне глубокого падения)
            3. Distribution (консолидация после pump)
            4. Accumulation (боковик, низкая волатильность)
        """
                # ── Slow bleed: прямое падение за 1ч (проверяем ПЕРЕД recovery)
        # Если BTC падает за 1 час — это не аккумуляция, а медленный слив.
        # Важно: если 30m растёт, а 1h падает — это может быть recovery,
        # поэтому slow bleed проверяем после recovery, но до pump/dump.
        if change_1h <= -0.75:
            logger.debug(f"Slow bleed 1h: 1h={change_1h:.2f}% 30m={change_30m:.2f}% — bearish_side")
            return "bearish_side"

        # ── Recovery — отскок после dump (проверяем ПЕРЕД pump/dump,
        #     чтобы поймать умеренное восстановление) ──
        # Условия: 30м умеренно положительный (0.2–1.2%), 1ч и 4ч всё ещё
        # глубоко отрицательные, и не было свежего pump.
        if (0.2 <= change_30m <= 1.2
                and change_1h < -0.8
                and change_4h < -0.5
                and not self._is_prior_pump(df_5m, lookback=6)):
            logger.debug("Recovery detected: 30m=%.2f%%, 1h=%.2f%%", change_30m, change_1h)
            return "recovery"

        # ── Pump ──
        if change_30m >= PUMP_THRESHOLD:
            logger.debug("Pump detected: 30m=%.2f%%, vol_z=%.2f", change_30m, vol_zscore)
            return "pump"

        # ── Dump ──
        if change_30m <= DUMP_THRESHOLD:
            logger.debug("Dump detected: 30m=%.2f%%, vol_z=%.2f", change_30m, vol_zscore)
            return "dump"

        # ── Distribution — консолидация после pump ──
        # Умеренные изменения, но был предшествующий существенный рост
        if self._is_prior_pump(df_5m, lookback=12):
            # Если сейчас изменений почти нет — distribution
            if abs(change_30m) < PUMP_THRESHOLD * 0.5:
                logger.debug("Distribution detected (prior pump, now flat)")
                return "distribution"

        # ── Accumulation — боковик ──
        if abs(change_30m) < PUMP_THRESHOLD * 0.5:
            # 🩹 Slow bleed (4h): 4h падение > 1.5% — это не аккумуляция
            if change_4h <= -1.5:
                logger.debug(f"Slow bleed 4h: 30m={change_30m:.2f}% but 4h={change_4h:.2f}% — bearish_side")
                return "bearish_side"
            return "accumulation"

        # 🩹 Slow bleed (4h fallback): sustained drop
        if change_4h <= -1.5:
            logger.debug(f"Slow bleed 4h (fallback): 4h={change_4h:.2f}%")
            return "bearish_side"

        # fallback
        return "accumulation"

    @staticmethod
    def _is_prior_pump(df_5m: pd.DataFrame, lookback: int = 12) -> bool:
        """
        Проверяет, был ли pump в окне, предшествующем последним 6 свечам 5M.
        Использует скользящий максимум: если в окне была пара точек (i, j),
        где i < j и рост close[j]/close[i] - 1 >= PUMP_THRESHOLD, то pump был.
        """
        if len(df_5m) < lookback + 6:
            return False
        # окно: от -lookback-6 до -7 (исключаем последние 6 свечей)
        start = max(0, -(lookback + 6))
        end   = -6
        closes = df_5m["close"].values[start:end]
        if len(closes) < 5:
            return False
        # Сканируем все пары i<j в окне
        for i in range(len(closes)):
            for j in range(i + 1, len(closes)):
                pct = (closes[j] - closes[i]) / closes[i] * 100.0
                if pct >= PUMP_THRESHOLD:
                    return True
        return False

    def _get_recommendation(self, regime: str) -> str:
        """Маппинг фазы → рекомендация."""
        mapping = {
            "pump":          "sell_only",
            "dump":          "no_trade",
            "accumulation":  "buy_allowed",
            "distribution":  "sell_only",
            "bearish_side":   "sell_only",
            "recovery":      "buy_priority",
        }
        return mapping.get(regime, "buy_allowed")

    @staticmethod
    def _build_message(regime: str, change_30m: float,
                       recommendation: str, change_4h: float = 0.0) -> str:
        """Формирует человекочитаемое описание."""
        msgs = {
            "pump": (
                f"🚀 PUMP: BTC вырос на {change_30m:.2f}% за 30 мин. "
                "Продажи разрешены, покупки заблокированы. "
                "Принудительный take-profit."
            ),
            "dump": (
                f"📉 DUMP: BTC упал на {abs(change_30m):.2f}% за 30 мин. "
                "Торговля остановлена, ждём дна."
            ),
            "accumulation": (
                f"➡️ Аккумуляция: BTC в боковике ({change_30m:+.2f}% за 30 мин). "
                "Торговля по стандартной стратегии."
            ),
            "distribution": (
                "📊 Дистрибьюция: консолидация после pump. "
                "Только продажи, готовимся к dump."
            ),
            "bearish_side": (
                f"📉 Медленный слив: за 4ч {change_4h:+.2f}%, 30м {change_30m:+.2f}%. "
                "Только продажи, шорты разрешены."
            ),
            "recovery": (
                f"🔄 Recovery: отскок {change_30m:+.2f}% за 30 мин. "
                "Покупки с повышенным приоритетом."
            ),
        }
        return msgs.get(regime, f"Фаза: {regime}, изменение: {change_30m:+.2f}%")


# ══════════════════════════════════════════════
# Демо / самопроверка (python btc_regime_tracker.py)
# ══════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    np.random.seed(42)

    # Генерируем синтетические свечи для демонстрации
    n_5m = 200
    n_1h = 48

    base_price = 65_000.0

    # Сначала медленный рост, потом pump, потом коррекция
    prices_5m = []
    p = base_price
    for i in range(n_5m):
        if 40 <= i < 60:
            p *= 1 + np.random.normal(0.0015, 0.002)   # pump-зона
        elif 60 <= i < 80:
            p *= 1 + np.random.normal(-0.001, 0.002)    # коррекция
        else:
            p *= 1 + np.random.normal(0.0002, 0.001)    # боковик
        prices_5m.append(max(p, 1.0))

    prices_1h = []
    p = base_price
    for i in range(n_1h):
        p *= 1 + np.random.normal(0.0005, 0.003)
        prices_1h.append(max(p, 1.0))

    df_5m = pd.DataFrame({
        "open":   prices_5m,
        "high":   [v * 1.002 for v in prices_5m],
        "low":    [v * 0.998 for v in prices_5m],
        "close":  prices_5m,
        "volume": np.random.exponential(1000, n_5m).tolist(),
    })

    df_1h = pd.DataFrame({
        "open":   prices_1h,
        "high":   [v * 1.005 for v in prices_1h],
        "low":    [v * 0.995 for v in prices_1h],
        "close":  prices_1h,
        "volume": np.random.exponential(10_000, n_1h).tolist(),
    })

    tracker = BTCRegimeTracker()
    result = tracker.update(df_5m, df_1h)

    print("\n=== BTC Regime Tracker ===")
    for k, v in result.items():
        print(f"  {k:25s} = {v}")
    print(f"\n  buy_allowed: {tracker.is_buy_allowed()}")
    print(f"  get_regime:  {tracker.get_regime()}")
