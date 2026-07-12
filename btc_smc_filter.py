#!/usr/bin/env python3
"""
btc_smc_filter.py — Лёгкий SMC-фильтр для анализа структуры BTC.

Анализирует свечи BTC на 15M/1H/4H через SMC-принципы:
  - CHoCH (Change of Character) — смена структуры
  - BOS (Break of Structure) — пробой структуры
  - FVG (Fair Value Gap) — дисбаланс
  - Order Blocks — зоны накопления/распределения
  - Свинг-хаи/лоу + трендовые линии
  - BTCD (доминирование BTC) если данные доступны

Возвращает VETO-голос: -15..0 к score, флаг emergency_close + анализ.

Никаких внешних зависимостей, кроме numpy.
Не вызывает агентов, не ходит по HTTP — только расчёты на свечах.

Integration:
  from btc_smc_filter import BTCSmcFilter
  filter = BTCSmcFilter()
  result = filter.analyze(btc_15m, btc_1h, btc_4h, btcd_data=None)
  // result = { veto: -15, emergency: False, direction: 'bearish', reason: 'CHoCH на 1H', details: {...} }
"""

import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Константы
# ──────────────────────────────────────────────
MAX_VETO = -15        # Максимальный штраф от SMC фильтра
EMERGENCY_THRESHOLD = -20   # Не используется, но зарезервирован

class BTCSmcFilter:
    """SMC-фильтр для анализа структуры BTC.

    При вызове analyze() без аргументов — сам фетчит BTC свечи через ccxt.
    При передаче свечей — использует их (приоритет переданных).
    """

    def __init__(self, verbose: bool = False, exchange=None):
        self._last_result = {
            'veto': 0,
            'emergency': False,
            'direction': 'neutral',
            'reason': 'N/A',
            'details': {}
        }
        self._last_update = 0.0
        self._update_interval = 30.0  # не чаще раза в 30 сек
        self.verbose = verbose

        # ── Собственный сбор данных ──
        self._exchange = exchange  # можно передать при инициализации
        self._own_exchange = None  # создаём свой, если exchange не передан
        self._fetch_cooldown = 300  # 5 мин между fetch для своих свечей
        self._last_fetch_time = 0
        self._cached_candles_1h = None
        self._cached_candles_4h = None
        self._cached_candles_15m = None

    def _ensure_exchange(self):
        """Создаёт ccxt-клиент если не передан и не создан."""
        if self._exchange is not None:
            return self._exchange
        if self._own_exchange is not None:
            return self._own_exchange
        try:
            import ccxt
            self._own_exchange = ccxt.bybit()
            return self._own_exchange
        except Exception:
            return None

    def _fetch_btc_candles(self, timeframe: str, limit: int = 30) -> Optional[list]:
        """Фетчит BTC свечи. Возвращает [{o,h,l,c,v}, ...] или None."""
        ex = self._ensure_exchange()
        if ex is None:
            return None
        try:
            raw = ex.fetch_ohlcv('BTC/USDT', timeframe, limit=limit)
            if not raw:
                return None
            return [{'o': c[1], 'h': c[2], 'l': c[3], 'c': c[4], 'v': c[5]} for c in raw]
        except Exception as e:
            if self.verbose:
                logger.debug(f"[BTCSMC] fetch {timeframe}: {e}")
            return None

    def _auto_fetch(self):
        """Самостоятельно загружает свечи, если не переданы, с кешем."""
        now = time.time()
        if now - self._last_fetch_time < self._fetch_cooldown:
            return  # кеш свежий

        if self._ensure_exchange() is None:
            return  # нет доступа к ccxt — работать будем только с переданными данными

        self._cached_candles_1h = self._fetch_btc_candles('1h', 40)
        self._cached_candles_4h = self._fetch_btc_candles('4h', 16)
        self._cached_candles_15m = self._fetch_btc_candles('15m', 96)
        self._last_fetch_time = now

        if self.verbose:
            logger.info(f"[BTCSMC] fetched: 1H={len(self._cached_candles_1h) if self._cached_candles_1h else 0}"
                        f" 4H={len(self._cached_candles_4h) if self._cached_candles_4h else 0}"
                        f" 15M={len(self._cached_candles_15m) if self._cached_candles_15m else 0}")

    # ─── Публичный API ────────────────────────

    def analyze(self,
                candles_15m: Optional[list] = None,
                candles_1h: Optional[list] = None,
                candles_4h: Optional[list] = None,
                btcd_data: Optional[list] = None) -> dict:
        """
        Главный метод анализа.

        Если свечи не переданы (None) — фетчит сам через ccxt (без pandas).
        Если переданы — использует их (приоритет переданных).

        Args:
            candles_15m: список свечей 15M [{o,h,l,c,v}, ...] (24-96 шт)
            candles_1h:  список свечей 1H  [{o,h,l,c,v}, ...] (24 шт)
            candles_4h:  список свечей 4H  [{o,h,l,c,v}, ...] (12 шт)
            btcd_data:   список BTC.D свечей или уровней (опционально)

        Returns:
            dict с полями:
              veto       - int, -15..0  (штраф к score)
              emergency  - bool, аварийный выход
              direction  - 'bullish' | 'bearish' | 'neutral'
              reason     - str, причина
              details    - dict с деталями каждого компонента
        """
        now = time.time()
        if now - self._last_update < self._update_interval:
            return self._last_result  # кеш
        self._last_update = now

        # ── Авто-загрузка если свечи не переданы ──
        if (candles_1h is None or candles_4h is None or candles_15m is None):
            self._auto_fetch()
            if candles_1h is None:
                candles_1h = self._cached_candles_1h
            if candles_4h is None:
                candles_4h = self._cached_candles_4h
            if candles_15m is None:
                candles_15m = self._cached_candles_15m

        penalty = 0
        reasons = []
        details = {}
        emergency = False

        # ─── 1. CHoCH на 1H (смена структуры) ────────────────
        if candles_1h and len(candles_1h) >= 12:
            choch_1h, choch_detail = self._detect_choch(candles_1h)
            details['choch_1h'] = choch_detail
            if choch_1h == 'bearish':
                penalty += 6
                reasons.append("CHoCH на 1H")
            elif choch_1h == 'bullish':
                penalty -= 3  # бонус
                reasons.append("bullish CHoCH 1H")

        # ─── 2. CHoCH на 4H ─────────────────────────────────
        if candles_4h and len(candles_4h) >= 8:
            choch_4h, choch4_detail = self._detect_choch(candles_4h)
            details['choch_4h'] = choch4_detail
            if choch_4h == 'bearish':
                penalty += 8  # 4H медвежий CHoCH → серьёзнее
                reasons.append("CHoCH на 4H")
            elif choch_4h == 'bullish':
                penalty -= 5
                reasons.append("bullish CHoCH 4H")

        # ─── 3. FVG (дисбаланс) на 1H ────────────────────────
        if candles_1h and len(candles_1h) >= 6:
            fvg_verdict, fvg_detail = self._detect_fvg(candles_1h)
            details['fvg_1h'] = fvg_detail
            if fvg_verdict == 'bearish':
                penalty += 4
                reasons.append("Медвежий FVG")
            elif fvg_verdict == 'bullish':
                penalty -= 2

        # ─── 4. Order Block на 4H ────────────────────────────
        if candles_4h and len(candles_4h) >= 6:
            ob_verdict, ob_detail = self._detect_order_block(candles_4h)
            details['ob_4h'] = ob_detail
            if ob_verdict == 'bearish':
                penalty += 3
                reasons.append("OB на 4H медвежий")
            elif ob_verdict == 'bullish':
                penalty -= 2

        # ─── 5. Свинг-хаи/лоу на 15M ────────────────────────
        if candles_15m and len(candles_15m) >= 20:
            swing_v, swing_detail = self._detect_swing(candles_15m)
            details['swing_15m'] = swing_detail
            if swing_v == 'bearish':
                penalty += 4
                reasons.append("Нисходящие пики на 15M")
            elif swing_v == 'bullish':
                penalty -= 3

        # ─── 6. BTCD (доминирование BTC) ────────────────────
        if btcd_data and len(btcd_data) >= 4:
            btcd_v, btcd_detail = self._detect_btcd(btcd_data)
            details['btcd'] = btcd_detail
            if btcd_v == 'rising':
                penalty += 3
                reasons.append("BTCD растёт")
            elif btcd_v == 'falling':
                penalty -= 2
                reasons.append("BTCD падает")

        # ─── Финал ───────────────────────────────────────────
        penalty = max(MAX_VETO, penalty)  # не ниже -15

        # EMERGENCY_CLOSE: смотрим на 15M — пробой свинг-лоу + объём
        if candles_15m and len(candles_15m) >= 5:
            emergency = self._detect_emergency_break(candles_15m)
            details['emergency_check'] = emergency

        # Итоговое направление
        if penalty <= -8:
            direction = 'bearish'
        elif penalty >= 3:
            direction = 'bullish'
        else:
            direction = 'neutral'

        reason_str = ', '.join(reasons) if reasons else 'OK'

        self._last_result = {
            'veto': penalty,
            'emergency': emergency,
            'direction': direction,
            'reason': reason_str,
            'details': details
        }

        if self.verbose:
            logger.info(f"[BTCSMC] veto={penalty} dir={direction} reason={reason_str}")

        return self._last_result

    # ─── CHoCH (смена структуры) ───────────────
    @staticmethod
    def _detect_choch(candles: list) -> tuple:
        """
        Определяет смену характера (CHoCH) на основе сравнения трендовых сегментов.

        **Медвежий CHoCH:**
        - Был восходящий тренд (серия HH/HL) — проверяем через последовательность
          более высоких high/low
        - Последний high ниже предпоследнего (LH)
        - Цена пробила последний низкий low (LL)

        **Бычий CHoCH:**
        - Был нисходящий тренд (серия LH/LL)
        - Последний low выше предпоследнего (HL)
        - Цена пробила последний высокий high (HH)

        Логика не требует идеальных свингов — работает на разбивке данных
        на сегменты по N свечей (динамически в зависимости от количества данных).

        Returns:
            (str, dict): ('bullish'|'bearish'|'neutral', detail)
        """
        if not candles or len(candles) < 8:
            return ('neutral', {'reason': 'мало данных', 'detected': False})

        highs_v = np.array([c.get('h', c.get('high', 0)) for c in candles[-40:]])
        lows_v = np.array([c.get('l', c.get('low', 0)) for c in candles[-40:]])
        closes_v = np.array([c.get('c', c.get('close', 0)) for c in candles[-40:]])
        vols_v = np.array([c.get('v', c.get('volume', 0)) for c in candles[-40:]])

        n = len(highs_v)
        if n < 8:
            return ('neutral', {'reason': 'мало данных', 'detected': False})

        mean_vol = float(np.mean(vols_v[vols_v > 0])) if np.any(vols_v > 0) else 1
        if mean_vol == 0:
            mean_vol = 1

        # ── 1. Разбиваем на сегменты ──
        # Делим данные на 4 равных сегмента
        seg_size = max(2, n // 4)
        seg_highs = []
        seg_lows = []
        seg_h_idx = []
        seg_l_idx = []

        for s in range(4):
            start = s * seg_size
            end = min(n, (s + 1) * seg_size) if s < 3 else n
            if end - start < 2:
                continue
            seg_h = float(np.max(highs_v[start:end]))
            seg_l = float(np.min(lows_v[start:end]))
            seg_highs.append(seg_h)
            seg_lows.append(seg_l)
            # Индекс середины сегмента
            mid_idx = start + (end - start) // 2
            seg_h_idx.append(int(np.argmax(highs_v[start:end]) + start))
            seg_l_idx.append(int(np.argmin(lows_v[start:end]) + start))

        detail = {
            'segments': len(seg_highs),
            'seg_highs': [f'{h:.1f}' for h in seg_highs],
            'seg_lows': [f'{l:.1f}' for l in seg_lows],
        }

        if len(seg_highs) < 2:
            return ('neutral', detail)

        # ── 2. Медвежий CHoCH ──
        # Последний сегмент high ниже предпоследнего = LH
        # + цена пробила минимум предпоследнего сегмента
        if seg_highs[-1] < seg_highs[-2] * 1.002 and len(seg_lows) >= 2:
            lowest_prev2 = min(seg_lows[-3:-1]) if len(seg_lows) >= 3 else seg_lows[-2]
            if closes_v[-1] < lowest_prev2:
                vol_ratio = float(vols_v[-1] / mean_vol) if mean_vol > 0 else 1
                detail['bearish'] = True
                detail['reason'] = f'LH {seg_highs[-2]:.1f}→{seg_highs[-1]:.1f}, пробой {lowest_prev2:.1f}'
                detail['vol_ratio'] = vol_ratio
                return ('bearish', detail)

        # ── 3. Бычий CHoCH ──
        # Последний сегмент low выше предпоследнего = HL
        # + цена пробила максимум предпоследнего сегмента
        if len(seg_lows) >= 2 and seg_lows[-1] > seg_lows[-2] * 0.998 and len(seg_highs) >= 2:
            highest_prev2 = max(seg_highs[-3:-1]) if len(seg_highs) >= 3 else seg_highs[-2]
            if closes_v[-1] > highest_prev2:
                vol_ratio = float(vols_v[-1] / mean_vol) if mean_vol > 0 else 1
                detail['bullish'] = True
                detail['reason'] = f'HL {seg_lows[-2]:.1f}→{seg_lows[-1]:.1f}, пробой {highest_prev2:.1f}'
                detail['vol_ratio'] = vol_ratio
                return ('bullish', detail)

        return ('neutral', detail)

        detail = {
            'swing_highs': len(swing_highs),
            'swing_lows': len(swing_lows),
            'swing_high_prices': [f'{s["price"]:.1f}' for s in swing_highs[-4:]],
            'swing_low_prices': [f'{s["price"]:.1f}' for s in swing_lows[-4:]],
        }

        # ── 2. Медвежий CHoCH ──
        # Нужно: минимум 2 SWH и 1 SWL после последнего SWH
        if len(swing_highs) >= 2:
            last_sh = swing_highs[-1]
            prev_sh = swing_highs[-2]
            
            # Ищем SWL после last_sh
            lows_after = [sw for sw in swing_lows if sw['idx'] > prev_sh['idx']]
            
            if lows_after:
                target_ll = min(lows_after, key=lambda x: x['price'])  # самый низкий
                target_ll_idx = target_ll['idx']
                
                # Проверяем пробой: close ниже target_ll в любой свече после него
                closes_after_ll = closes_v[target_ll_idx + 1:]
                if closes_after_ll and min(closes_after_ll) < target_ll['price']:
                    vol_ratio = vols_v[-1] / mean_vol if mean_vol > 0 else 1
                    detail['bearish'] = True
                    detail['last_sh'] = last_sh['price']
                    detail['prev_sh'] = prev_sh['price']
                    detail['target_ll'] = target_ll['price']
                    detail['ll_idx'] = target_ll_idx
                    return ('bearish', detail)

            # Альтернатива: если нет SWL после, но цена явно ниже последнего SWL
            if swing_lows:
                last_swl = swing_lows[-1]
                if last_swl['idx'] > prev_sh['idx'] and closes_v[-1] < last_swl['price']:
                    vol_ratio = vols_v[-1] / mean_vol if mean_vol > 0 else 1
                    detail['bearish'] = True
                    detail['last_sh'] = last_sh['price']
                    detail['prev_sh'] = prev_sh['price']
                    detail['target_ll'] = last_swl['price']
                    return ('bearish', detail)

        # ── 3. Бычий CHoCH ──
        if len(swing_lows) >= 2:
            last_sl = swing_lows[-1]
            prev_sl = swing_lows[-2]
            
            highs_after = [sw for sw in swing_highs if sw['idx'] > prev_sl['idx']]
            
            if highs_after:
                target_hh = max(highs_after, key=lambda x: x['price'])
                target_hh_idx = target_hh['idx']
                
                closes_after_hh = closes_v[target_hh_idx + 1:]
                if closes_after_hh and max(closes_after_hh) > target_hh['price']:
                    vol_ratio = vols_v[-1] / mean_vol if mean_vol > 0 else 1
                    detail['bullish'] = True
                    detail['last_sl'] = last_sl['price']
                    detail['prev_sl'] = prev_sl['price']
                    detail['target_hh'] = target_hh['price']
                    return ('bullish', detail)

            if swing_highs:
                last_swh = swing_highs[-1]
                if last_swh['idx'] > prev_sl['idx'] and closes_v[-1] > last_swh['price']:
                    vol_ratio = vols_v[-1] / mean_vol if mean_vol > 0 else 1
                    detail['bullish'] = True
                    detail['last_sl'] = last_sl['price']
                    detail['prev_sl'] = prev_sl['price']
                    detail['target_hh'] = last_swh['price']
                    return ('bullish', detail)

        return ('neutral', detail)

    # ─── FVG (дисбаланс) ────────────────────────
    @staticmethod
    def _detect_fvg(candles: list) -> tuple:
        """
        Определяет Fair Value Gap на 1H.
        FVG бычий: low_current > high_prev (цена не закрыла разрыв).
        FVG медвежий: high_current < low_prev (медвежий разрыв).

        Returns:
            (str, dict): ('bullish'|'bearish'|'neutral', detail)
        """
        if not candles or len(candles) < 3:
            return ('neutral', {})

        highs = np.array([c.get('h', c.get('high', 0)) for c in candles[-6:]])
        lows = np.array([c.get('l', c.get('low', 0)) for c in candles[-6:]])
        closes = np.array([c.get('c', c.get('close', 0)) for c in candles[-6:]])

        if len(highs) < 3:
            return ('neutral', {})

        # Бычий FVG: свеча после импульсной вверх
        for i in range(2, min(6, len(highs))):
            if lows[i] > highs[i-2]:  # gap не закрыт
                gap = lows[i] - highs[i-2]
                gap_pct = gap / highs[i-2] * 100
                if 0.05 <= gap_pct <= 3.0:  # реальный FVG, а не шум
                    return ('bullish', {
                        'detected': True,
                        'gap_pct': float(gap_pct),
                        'gap_price': float(gap),
                        'idx': i
                    })

        # Медвежий FVG: свеча после импульсной вниз
        for i in range(2, min(6, len(highs))):
            if highs[i] < lows[i-2]:  # gap не закрыт
                gap = lows[i-2] - highs[i]
                gap_pct = gap / lows[i-2] * 100
                if 0.05 <= gap_pct <= 3.0:
                    return ('bearish', {
                        'detected': True,
                        'gap_pct': float(gap_pct),
                        'gap_price': float(gap),
                        'idx': i
                    })

        return ('neutral', {'detected': False})

    # ─── Order Block ────────────────────────────
    @staticmethod
    def _detect_order_block(candles: list) -> tuple:
        """
        Определяет Order Block на 4H.
        Бычий OB: последняя медвежья свеча перед импульсом вверх.
        Медвежий OB: последняя бычья свеча перед импульсом вниз.

        Returns:
            (str, dict): ('bullish'|'bearish'|'neutral', detail)
        """
        if not candles or len(candles) < 4:
            return ('neutral', {})

        opens = np.array([c.get('o', c.get('open', 0)) for c in candles[-8:]])
        highs = np.array([c.get('h', c.get('high', 0)) for c in candles[-8:]])
        lows = np.array([c.get('l', c.get('low', 0)) for c in candles[-8:]])
        closes = np.array([c.get('c', c.get('close', 0)) for c in candles[-8:]])
        vols = np.array([c.get('v', c.get('volume', 0)) for c in candles[-8:]])

        if len(opens) < 4:
            return ('neutral', {})

        # Ищем последнюю медвежью свечу (красную), после которой пошёл бычий импульс
        for i in range(2, min(6, len(opens))):
            if closes[i] < opens[i]:  # медвежья
                if i + 1 < len(closes) and closes[i+1] > highs[i]:  # следующий бар выше = импульс
                    vol_ratio = vols[i] / np.mean(vols[:i+1]) if np.mean(vols[:i+1]) > 0 else 1
                    if vol_ratio > 1.2:  # объём подтверждает
                        return ('bullish', {
                            'detected': True,
                            'ob_type': 'бычий',
                            'ob_high': float(highs[i]),
                            'ob_low': float(lows[i]),
                            'vol_ratio': float(vol_ratio)
                        })

        # Ищем последнюю бычью свечу (зелёную), после которой импульс вниз
        for i in range(2, min(6, len(opens))):
            if closes[i] > opens[i]:  # бычья
                if i + 1 < len(closes) and closes[i+1] < lows[i]:  # импульс вниз
                    vol_ratio = vols[i] / np.mean(vols[:i+1]) if np.mean(vols[:i+1]) > 0 else 1
                    if vol_ratio > 1.2:
                        return ('bearish', {
                            'detected': True,
                            'ob_type': 'медвежий',
                            'ob_high': float(highs[i]),
                            'ob_low': float(lows[i]),
                            'vol_ratio': float(vol_ratio)
                        })

        return ('neutral', {'detected': False})

    # ─── Свинг-структура на 15M ────────────────
    @staticmethod
    def _detect_swing(candles: list) -> tuple:
        """
        Анализирует серию свинг-пиков на 15M.
        HH/HL → bullish, LH/LL → bearish.

        Returns:
            (str, dict): ('bullish'|'bearish'|'neutral', detail)
        """
        if not candles or len(candles) < 12:
            return ('neutral', {})

        highs = np.array([c.get('h', c.get('high', 0)) for c in candles[-15:]])
        lows = np.array([c.get('l', c.get('low', 0)) for c in candles[-15:]])

        if len(highs) < 12:
            return ('neutral', {})

        n = len(highs)

        # Ищем локальные пики/впадины
        pivots_high = []
        pivots_low = []
        for i in range(2, n - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                pivots_high.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                pivots_low.append(lows[i])

        # Если нет пиков или впадин — не можем определить структуру
        if len(pivots_high) < 2 and len(pivots_low) < 2:
            return ('neutral', {})

        # Анализ последних 2-3 пиков
        bearish = False
        bullish = False

        # Медвежья структура: последние 2 пика — нисходящие (LH)
        if len(pivots_high) >= 2:
            if pivots_high[-2] > pivots_high[-1] * 1.001:
                bearish = True

        # Медвежья структура: последние 2 впадины — нисходящие (LL)
        if len(pivots_low) >= 2:
            if pivots_low[-2] > pivots_low[-1] * 1.001:
                bearish = True

        # Бычья структура: последние 2 пика — восходящие (HH)
        if len(pivots_high) >= 2:
            if pivots_high[-1] > pivots_high[-2] * 1.001:
                bullish = True

        # Бычья структура: последние 2 впадины — восходящие (HL)
        if len(pivots_low) >= 2:
            if pivots_low[-1] > pivots_low[-2] * 1.001:
                bullish = True

        # Если оба сигнала — берём последний по времени
        if bearish and not bullish:
            return ('bearish', {
                'detected': True,
                'higher_highs': len(pivots_high),
                'lower_lows': len(pivots_low),
                'structure': 'LL/LH'
            })
        elif bullish and not bearish:
            return ('bullish', {
                'detected': True,
                'higher_highs': len(pivots_high),
                'higher_lows': len(pivots_low),
                'structure': 'HH/HL'
            })

        return ('neutral', {
            'detected': False,
            'pivot_highs': [float(x) for x in pivots_high[-3:]] if pivots_high else [],
            'pivot_lows': [float(x) for x in pivots_low[-3:]] if pivots_low else []
        })

    # ─── BTCD (доминирование) ──────────────────
    @staticmethod
    def _detect_btcd(btcd: list) -> tuple:
        """
        Анализирует доминирование BTC.
        BTCD растёт → капитал в BTC → альты слабеют.
        BTCD падает → капитал в альты → альты сильнее.

        Returns:
            (str, dict): ('rising'|'falling'|'stable', detail)
        """
        if not btcd or len(btcd) < 4:
            return ('stable', {})

        if isinstance(btcd[0], dict):
            vals = np.array([c.get('c', c.get('close', c.get('value', 0))) for c in btcd])
        else:
            vals = np.array(btcd)

        if len(vals) < 4:
            return ('stable', {})

        change_4h = (vals[-1] - vals[0]) / vals[0] * 100

        if change_4h > 0.5:
            return ('rising', {'change_4h_pct': float(change_4h), 'current': float(vals[-1])})
        elif change_4h < -0.5:
            return ('falling', {'change_4h_pct': float(change_4h), 'current': float(vals[-1])})
        else:
            return ('stable', {'change_4h_pct': float(change_4h)})

    # ─── Emergency Break Detection ────────────────
    @staticmethod
    def _detect_emergency_break(candles_15m: list) -> bool:
        """
        Аварийный пробой: последняя свеча пробила минимум предыдущих 4+ свечей
        с повышенным объёмом.

        Returns:
            bool: True если аварийный пробой
        """
        if not candles_15m or len(candles_15m) < 5:
            return False

        close = candles_15m[-1].get('c', candles_15m[-1].get('close', 0))
        low_prev = min(
            c.get('l', c.get('low', float('inf')))
            for c in candles_15m[-5:-1]
        )
        if close < low_prev:
            vol_last = candles_15m[-1].get('v', candles_15m[-1].get('volume', 0))
            vol_avg = np.mean([
                c.get('v', c.get('volume', 0)) for c in candles_15m[-6:-1]
            ])
            if vol_last > vol_avg * 1.5:
                return True
        return False
