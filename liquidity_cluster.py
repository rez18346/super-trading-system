"""
LiquidityCluster - анализ микроструктуры рынка.

Определяет зоны ликвидности на основе:
  1. Rolling Volume Profile - кластеризация объёма за N свечей
  2. POC (Point of Control) - точка максимального объёма
  3. Value Area High/Low - границы 70% объёма
  4. Fair Value Gaps (FVG) - дисбалансы (гэпы) между свечами
  5. VWAP 1H / 4H - позиция цены относительно VWAP

Лицензия: MIT
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# ──────────────────────────────────────────────────────────────────────────────
# ДАННЫЕ
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LiquidityState:
    """Текущее состояние ликвидности для символа."""
    poc: float               # Point of Control
    vah: float               # Value Area High
    val: float               # Value Area Low
    fvg_below: List[Tuple[float, float]]  # незакрытые гэпы снизу (price_low, price_high)
    fvg_above: List[Tuple[float, float]]  # незакрытые гэпы сверху
    vwap_1h: Optional[float] = None
    vwap_4h: Optional[float] = None
    cluster_quality: float = 0.0   # 0-1 - насколько чёткие кластеры

    @property
    def in_value_area(self) -> bool:
        """Цена внутри Value Area?"""
        return self.val <= self.poc <= self.vah  # заглушка, обновится при evaluate


class LiquidityCluster:
    """
    Анализатор ликвидности по свечам.

    Потребляет список свечей (список словарей {o,h,l,c,v,t} или
    список списков [ts,o,h,l,c,v]).
    """

    def __init__(self, window: int = 48):
        """
        Args:
            window: размер окна для Volume Profile (число свечей 5M)
        """
        self.window = window

    # ──────────────────────────────────────────────────────────────────────────
    # ПУБЛИЧНЫЙ МЕТОД
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(self, candles_5m: List, current_price: float,
                 candles_1h: Optional[List] = None,
                 candles_4h: Optional[List] = None) -> Dict:
        """
        Получить оценку ликвидности.

        Returns:
            dict с полями:
              - score: int 0-100
              - detail: str для логов
              - signal: 'bullish' | 'bearish' | 'neutral'
              - state: LiquidityState
        """
        if not candles_5m or len(candles_5m) < 20:
            return {'score': 50, 'detail': 'мало данных', 'signal': 'neutral',
                    'state': None}

        closes = self._extract_closes(candles_5m)
        highs = self._extract_highs(candles_5m)
        lows = self._extract_lows(candles_5m)
        volumes = self._extract_volumes(candles_5m)

        if not closes:
            return {'score': 50, 'detail': 'нет цен', 'signal': 'neutral',
                    'state': None}

        # 1. Volume Profile
        poc, vah, val, vol_clusters = self._volume_profile(closes, highs, lows,
                                                            volumes)
        cluster_quality = self._cluster_quality(vol_clusters)

        # 2. FVG
        fvg_below, fvg_above = self._detect_fvg(candles_5m)

        # 3. VWAP из 1H / 4H
        vwap_1h = self._calc_vwap(candles_1h) if candles_1h else None
        vwap_4h = self._calc_vwap(candles_4h) if candles_4h else None

        state = LiquidityState(
            poc=poc, vah=vah, val=val,
            fvg_below=fvg_below, fvg_above=fvg_above,
            vwap_1h=vwap_1h, vwap_4h=vwap_4h,
            cluster_quality=cluster_quality,
        )

        # 4. Итоговая оценка (score 0-100)
        in_val = val <= current_price <= vah
        near_poc = abs(current_price - poc) / (poc + 1e-8) < 0.005  # 0.5%
        near_vah = abs(current_price - vah) / (vah + 1e-8) < 0.005
        near_val = abs(current_price - val) / (val + 1e-8) < 0.005

        score = 50  # нейтрально

        # Если цена внутри Value Area - нейтрально, лёгкий бонус если у POC
        if in_val:
            score = 55
            if near_poc:
                score = 60
            # Если объём растёт у POC - рынок засасывает
            if near_poc and cluster_quality > 0.6:
                score = 65
        else:
            # Цена ВЫШЕ VAH
            if current_price > vah:
                # Если есть незакрытые FVG сверху - цена может вернуться
                if fvg_above:
                    score = 40  # не входим, ждём заполнения гэпа
                elif cluster_quality > 0.5:
                    score = 70  # пробой на объёме - продолжаем
                else:
                    score = 60  # слабый пробой
            # Цена НИЖЕ VAL
            else:
                if fvg_below:
                    score = 40
                elif cluster_quality > 0.5:
                    score = 70
                else:
                    score = 60

        # Бонус от VWAP: если цена у VWAP 1H - значимый уровень
        if vwap_1h and abs(current_price - vwap_1h) / (vwap_1h + 1e-8) < 0.003:
            score = min(90, score + 20)

        # FVG бонус: цена возвращается к гэпу
        if near_vah and fvg_above:
            score = min(80, score + 15)  # сбор ликвидности у VAH
        if near_val and fvg_below:
            score = min(80, score + 15)  # сбор ликвидности у VAL
        
        # ═══ Breakout Trap: ложный пробой VAH/VAL с возвратом ═══════════════
        trap = self._detect_breakout_trap(highs[-20:], lows[-20:], closes[-20:],
                                           volumes[-20:] if len(volumes) >= 20 else [],
                                           vah, val)
        if trap == 'bullish':
            # Цена пробила VAH и вернулась → это сбор ликвидности шортов
            score = min(90, score + 20)
        elif trap == 'bearish':
            # Цена пробила VAL и вернулась → сбор лонгов
            score = min(90, score + 20)
        elif trap == 'fake_breakout':
            # Цена пробила и закрепилась, потом резкий разворот
            score = min(85, score + 15)
        
        signal = 'bullish' if score > 60 else ('bearish' if score < 40 else 'neutral')
        detail = (
            f"POC={poc:.4f} VAH={vah:.4f} VAL={val:.4f} "
            f"q={cluster_quality:.2f} fvg↑={len(fvg_above)} fvg↓={len(fvg_below)} "
            f"{trap or ''}"
        )

        return {'score': int(score), 'detail': detail, 'signal': signal, 'state': state}

    # ──────────────────────────────────────────────────────────────────────────
    # VOLUME PROFILE
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_closes(self, candles: List) -> List[float]:
        if not candles:
            return []
        if isinstance(candles[0], dict):
            # Поддерживаем 'c' (сокращённый) и 'close' (полный) формат
            first = candles[0]
            if 'c' in first:
                return [c['c'] for c in candles]
            if 'close' in first:
                return [c['close'] for c in candles]
            return []
        return [c[4] for c in candles]

    def _extract_highs(self, candles: List) -> List[float]:
        if isinstance(candles[0], dict):
            if 'h' in candles[0]:
                return [c['h'] for c in candles]
            if 'high' in candles[0]:
                return [c['high'] for c in candles]
            return []
        return [c[2] for c in candles]

    def _extract_lows(self, candles: List) -> List[float]:
        if isinstance(candles[0], dict):
            if 'l' in candles[0]:
                return [c['l'] for c in candles]
            if 'low' in candles[0]:
                return [c['low'] for c in candles]
            return []
        return [c[3] for c in candles]

    def _extract_volumes(self, candles: List) -> List[float]:
        if isinstance(candles[0], dict):
            if 'v' in candles[0]:
                return [c['v'] for c in candles]
            if 'volume' in candles[0]:
                return [c['volume'] for c in candles]
            return []
        return [c[5] for c in candles]

    def _volume_profile(self, closes: List[float], highs: List[float],
                        lows: List[float], volumes: List[float]
                        ) -> Tuple[float, float, float, Dict]:
        """
        Биннинг по цене: группируем объём по ценовым уровней.

        Returns:
            poc, vah, val, clusters
        """
        if not closes or not volumes:
            return 0, 0, 0, {}

        # Нормализация шага
        price_range = max(highs[-self.window:]) - min(lows[-self.window:])
        if price_range == 0:
            return closes[-1], closes[-1], closes[-1], {}
        tick_size = max(price_range * 0.002, 0.0001)  # 0.2% от диапазона

        n = min(self.window, len(closes))
        clusters: Dict[float, float] = {}

        for i in range(-n, 0):
            price_bin = round(closes[i] / tick_size) * tick_size
            clusters[price_bin] = clusters.get(price_bin, 0) + volumes[i]

        if not clusters:
            return closes[-1], closes[-1], closes[-1], {}

        # POC - бин с макс объёмом
        poc = max(clusters, key=clusters.get)
        total_vol = sum(clusters.values())

        # Value Area - 70% объёма вокруг POC
        sorted_bins = sorted(clusters.keys())
        poc_idx = sorted_bins.index(poc) if poc in sorted_bins else len(sorted_bins)//2

        cum_vol = 0
        low_idx = poc_idx
        high_idx = poc_idx
        cum_vol += clusters.get(poc, 0)

        while cum_vol < total_vol * 0.7:
            left_val = clusters.get(sorted_bins[low_idx - 1], 0) if low_idx > 0 else 0
            right_val = clusters.get(sorted_bins[high_idx + 1], 0) if high_idx < len(sorted_bins)-1 else 0

            if left_val >= right_val and low_idx > 0:
                low_idx -= 1
                cum_vol += left_val
            elif high_idx < len(sorted_bins)-1:
                high_idx += 1
                cum_vol += right_val
            else:
                break
            if low_idx <= 0 and high_idx >= len(sorted_bins)-1:
                break

        val = sorted_bins[max(0, low_idx)]
        vah = sorted_bins[min(len(sorted_bins)-1, high_idx)]

        return poc, vah, val, clusters

    def _cluster_quality(self, clusters: Dict) -> float:
        """
        Насколько чётко выделен POC.
        0 - плохо (объём размазан), 1 - отлично (один явный пик).
        """
        if not clusters or len(clusters) < 3:
            return 0.3
        max_vol = max(clusters.values())
        if max_vol == 0:
            return 0.3
        mean_vol = sum(clusters.values()) / len(clusters)
        return min(1.0, (max_vol / mean_vol) * 0.15)

    # ──────────────────────────────────────────────────────────────────────────
    # BREAKOUT TRAP
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_breakout_trap(self, highs: List[float], lows: List[float],
                               closes: List[float], volumes: List[float],
                               vah: float, val: float) -> str:
        """
        Детектор ложного пробоя (breakout trap).
        
        Проверяет последние 6 свечей на наличие:
          - Пробой VAH вверх → возврат под VAH (сбор ликвидности шортов)
          - Пробой VAL вниз → возврат над VAL (сбор лонгов)
          - Fake breakout: цена пробила уровень, закрепилась на 2+ свечах,
            потом резкий разворот с увеличенным объёмом
        
        Returns:
            'bullish' — цена сходила выше VAH и вернулась (скоро вверх)
            'bearish' — цена сходила ниже VAL и вернулась (скоро вниз)
            'fake_breakout' — ложный пробой с закреплением и разворотом
            '' — ничего не обнаружено
        """
        if len(highs) < 10 or len(lows) < 10 or not volumes:
            return ''
        
        # Последние 6 свечей для анализа
        n = min(6, len(highs))
        recent_highs = highs[-n:]
        recent_lows = lows[-n:]
        recent_closes = closes[-n:]
        recent_vols = volumes[-n:] if len(volumes) >= n else []
        
        avg_vol = sum(recent_vols[:-1]) / max(len(recent_vols[:-1]), 1)
        
        # 1. Цена сходила ВЫШЕ VAH и вернулась
        broke_vah = any(h > vah * 1.001 for h in recent_highs)
        back_below_vah = recent_closes[-1] < vah
        back_below_vah_2 = recent_closes[-2] < vah if n >= 2 else True
        if broke_vah and back_below_vah and back_below_vah_2:
            # Проверяем что возврат на низком объёме (сбор завершён)
            last_vol_low = len(recent_vols) >= 3 and recent_vols[-1] < avg_vol * 0.8
            if last_vol_low:
                return 'bullish'
        
        # 2. Цена сходила НИЖЕ VAL и вернулась
        broke_val = any(l < val * 0.999 for l in recent_lows)
        back_above_val = recent_closes[-1] > val
        back_above_val_2 = recent_closes[-2] > val if n >= 2 else True
        if broke_val and back_above_val and back_above_val_2:
            last_vol_low = len(recent_vols) >= 3 and recent_vols[-1] < avg_vol * 0.8
            if last_vol_low:
                return 'bearish'
        
        # 3. Fake breakout: пробой + закрепление выше VAH на 2 свечи + резкий
        #    разворот вниз с объёмом
        if n >= 5 and len(recent_vols) >= 5:
            two_above = all(h > vah * 1.001 for h in recent_highs[-4:-2])
            last_two_below = all(
                recent_closes[-i] < vah * 1.001 for i in range(1, 3)
            ) if n >= 2 else False
            if two_above and last_two_below and recent_vols[-1] > avg_vol * 1.3:
                return 'fake_breakout'
        
        return ''

    # ──────────────────────────────────────────────────────────────────────────
    # FAIR VALUE GAPS
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_fvg(self, candles: List) -> Tuple[List, List]:
        """
        Ищем незакрытые гэпы (FVG) за последние 24 свечи.

        Гэп сверху (цена прыгнула вверх → осталась дыра):
          если low[i] > high[i-1]:
            FVG = (high[i-1], low[i])

        Returns:
            (fvg_below, fvg_above) - списки (price_low, price_high)
        """
        fvg_below: List[Tuple[float, float]] = []
        fvg_above: List[Tuple[float, float]] = []

        highs = self._extract_highs(candles)
        lows = self._extract_lows(candles)

        n = min(24, len(highs))
        for i in range(-n + 1, 0):
            # Гэп вверх: low[i] > high[i-1]
            if lows[i] > highs[i - 1]:
                fvg_above.append((highs[i - 1], lows[i]))
            # Гэп вниз: high[i] < low[i-1]
            if highs[i] < lows[i - 1]:
                fvg_below.append((highs[i], lows[i - 1]))

        # Убираем уже заполненные (цена вернулась в диапазон)
        current_high = highs[-1]
        current_low = lows[-1]
        fvg_above = [(lo, hi) for lo, hi in fvg_above if current_high < hi]
        fvg_below = [(lo, hi) for lo, hi in fvg_below if current_low > lo]

        return fvg_below, fvg_above

    # ──────────────────────────────────────────────────────────────────────────
    # VWAP
    # ──────────────────────────────────────────────────────────────────────────

    def _calc_vwap(self, candles: List) -> Optional[float]:
        """
        Расчёт VWAP по списку свечей.
        """
        if not candles or len(candles) < 5:
            return None
        closes = self._extract_closes(candles)
        volumes = self._extract_volumes(candles)
        if not closes or not volumes or sum(volumes) == 0:
            return None
        n = min(len(closes), 24)
        pv = sum(closes[-n:][i] * volumes[-n:][i] for i in range(n))
        v = sum(volumes[-n:])
        return pv / v if v > 0 else None


# Глобальный экземпляр
_liq = None


def get_liquidity_cluster() -> LiquidityCluster:
    global _liq
    if _liq is None:
        _liq = LiquidityCluster()
    return _liq
