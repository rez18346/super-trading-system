"""
LiquidityCluster v2 — анализ микроструктуры + Order Flow / Order Block (Smart Money)

Определяет зоны ликвидности на основе:
  1. Rolling Volume Profile — POC, VAH, VAL (70% объёма)
  2. Fair Value Gaps (FVG) — незакрытые дисбалансы
  3. VWAP 1H / 4H — позиция цены относительно VWAP
  4. Order Flow (OF) — импульсное движение от институциональных входов
  5. Order Block (OB) — зона перед импульсом
  6. IDM → IDM OB → EXT OB — 3 ключевые зоны POI
  7. Mitigation — тест зон (до перекрытия телом)
  8. Liquidity Sweep — захват ликвидности через манипуляцию

Лицензия: MIT
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────────────────────
# ДАННЫЕ
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OrderBlockInfo:
    """Информация об ордерблоке."""
    price_high: float
    price_low: float
    kind: str               # 'idm' | 'idm_ob' | 'ext_ob'
    direction: str          # 'bullish' | 'bearish'
    mitigated: bool = False # перекрыто телом свечи?
    strength: float = 0.5   # 0-1

@dataclass
class OrderFlowInfo:
    """Информация о потоке ордеров."""
    price_start: float      # начало импульса
    price_end: float        # конец импульса
    direction: str          # 'bullish' | 'bearish'
    volume_ratio: float     # объём импульса / средний объём
    candle_count: int       # сколько свечей занял импульс
    sweep: bool = False     # был ли захват ликвидности перед импульсом

@dataclass
class LiquidityState:
    """Текущее состояние ликвидности для символа."""
    poc: float                  # Point of Control
    vah: float                  # Value Area High
    val: float                  # Value Area Low
    fvg_below: List[Tuple[float, float]]   # незакрытые гэпы снизу
    fvg_above: List[Tuple[float, float]]   # незакрытые гэпы сверху
    vwap_1h: Optional[float] = None
    vwap_4h: Optional[float] = None
    cluster_quality: float = 0.0    # 0-1
    # Новое
    order_blocks: List[OrderBlockInfo] = field(default_factory=list)
    order_flows: List[OrderFlowInfo] = field(default_factory=list)
    poi_idm: Optional[float] = None     # IDM уровень
    poi_idm_ob: Optional[Tuple[float, float]] = None  # IDM OB (low, high)
    poi_ext_ob: Optional[Tuple[float, float]] = None  # EXT OB (low, high)

    @property
    def in_value_area(self) -> bool:
        """Цена внутри Value Area?"""
        return self.val <= self.poc <= self.vah


class LiquidityCluster:
    """
    Анализатор ликвидности + Order Flow / Order Block (Smart Money).

    Потребляет список свечей (список словарей {o,h,l,c,v,t} или
    список списков [ts,o,h,l,c,v]).
    """

    def __init__(self, window: int = 48):
        self.window = window
        # Кэш для обнаруженных зон (сбрасывается при каждом evaluate)
        self._cached_of: List[OrderFlowInfo] = []
        self._cached_ob: List[OrderBlockInfo] = []
        # История POC для анализа смещения (для динамики Liq)
        self._poc_history: List[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # ПУБЛИЧНЫЙ МЕТОД
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(self, candles_5m: List, current_price: float,
                 candles_1h: Optional[List] = None,
                 candles_4h: Optional[List] = None) -> Dict:
        """
        Получить оценку ликвидности + Order Flow / Order Block.

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

        # 1. Volume Profile (старый добрый)
        poc, vah, val, vol_clusters = self._volume_profile(closes, highs, lows,
                                                            volumes)
        cluster_quality = self._cluster_quality(vol_clusters)

        # 2. FVG
        fvg_below, fvg_above = self._detect_fvg(candles_5m)

        # 3. VWAP
        vwap_1h = self._calc_vwap(candles_1h) if candles_1h else None
        vwap_4h = self._calc_vwap(candles_4h) if candles_4h else None

        # ─── НОВОЕ: Order Flow / Order Block ────────────────────────────────
        order_flows = self._detect_order_flows(closes, highs, lows, volumes)
        order_blocks = self._detect_order_blocks(closes, highs, lows, volumes,
                                                  order_flows)
        poi_idm, poi_idm_ob, poi_ext_ob = self._classify_poi(order_blocks)

        state = LiquidityState(
            poc=poc, vah=vah, val=val,
            fvg_below=fvg_below, fvg_above=fvg_above,
            vwap_1h=vwap_1h, vwap_4h=vwap_4h,
            cluster_quality=cluster_quality,
            order_blocks=order_blocks,
            order_flows=order_flows,
            poi_idm=poi_idm,
            poi_idm_ob=poi_idm_ob,
            poi_ext_ob=poi_ext_ob,
        )

        # 4. Итоговая оценка (score 0-100) — ТЕПЕРЬ с учётом OF/OB
        # ── Контекст: цена ниже VWAP 4H — медвежий сигнал ───────────────
        bearish_context = False
        if vwap_4h and current_price < vwap_4h * 0.995:
            bearish_context = True
            # Проверяем тренд 4H (цена ниже открытия последней 4H свечи)
            if candles_4h and len(candles_4h) >= 2:
                last_4h = candles_4h[-1]
                if isinstance(last_4h, dict):
                    open_4h = last_4h.get('o', last_4h.get('open', last_4h.get('c', current_price)))
                    close_4h = last_4h.get('c', last_4h.get('close', current_price))
                elif isinstance(last_4h, list) and len(last_4h) >= 5:
                    open_4h = last_4h[1]
                    close_4h = last_4h[4]
                else:
                    open_4h = None
                    close_4h = None
                if open_4h and close_4h and open_4h > close_4h:
                    bearish_context = True  # 4H свеча медвежья

        score = self._calc_score(closes, highs, lows, volumes,
                                  current_price, vah, val, poc,
                                  cluster_quality, fvg_above, fvg_below,
                                  vwap_1h, order_flows, order_blocks,
                                  poi_idm, poi_idm_ob, poi_ext_ob,
                                  bearish_context=bearish_context)

        signal = 'bullish' if score > 60 else ('bearish' if score < 40 else 'neutral')

        # Деталь для логов: добавляем OF/OB статус
        trap = self._detect_breakout_trap(highs[-20:], lows[-20:], closes[-20:],
                                           volumes[-20:] if len(volumes) >= 20 else [],
                                           vah, val)
        ob_details = []
        if order_blocks:
            alive = [ob for ob in order_blocks if not ob.mitigated]
            dead = [ob for ob in order_blocks if ob.mitigated]
            if alive:
                ob_details.append(f"OB{len(alive)}")
                for ob in alive[:2]:
                    ob_details.append(f"{ob.kind[:3]} {ob.direction}")
            if dead:
                ob_details.append(f"OF{len(dead)}")

        # ── Динамика POC ───────────────────────────────────────────────────
        poc_trend_str = ''
        self._poc_history.append(poc)
        if len(self._poc_history) > 12:
            self._poc_history = self._poc_history[-12:]
        if len(self._poc_history) >= 6:
            poc_start = self._poc_history[0]
            poc_end = self._poc_history[-1]
            if poc_start > 0:
                poc_change = (poc_end - poc_start) / poc_start
                if abs(poc_change) > 0.001:
                    poc_trend_str = 'POC↑' if poc_change > 0 else 'POC↓'

        ctx_tag = '⚠️ctx↓' if bearish_context else ''
        detail = (
            f"POC={poc:.4f} VAH={vah:.4f} VAL={val:.4f} "
            f"q={cluster_quality:.2f} fvg↑={len(fvg_above)} fvg↓={len(fvg_below)} "
            f"{' '.join(ob_details)} "
            f"{poc_trend_str} {trap or ''} {ctx_tag}"
        )

        # Кэшируем для следующих вызовов
        # ── Swing Levels (старая ликвидность) ──────────────────────────────
        swing_near, swing_detail = self._detect_swing_levels(
            candles_4h or candles_1h, current_price)
        
        # Если цена у старого уровня ликвидности — корректируем оценку
        if swing_near == 'below':
            # Цена у старого минимума — возможен отскок
            score = min(100, score + 8)
            detail += f' 📉SWL{swing_detail}'
        elif swing_near == 'above':
            # Цена у старого максимума — возможна ловушка
            score = max(0, score - 5)
            detail += f' 📈SWH{swing_detail}'

        self._cached_of = order_flows
        self._cached_ob = order_blocks

        return {'score': int(score), 'detail': detail, 'signal': signal, 'state': state,
                'poc_trend': poc_trend_str}

    # ──────────────────────────────────────────────────────────────────────────
    # ORDER FLOW — ДЕТЕКЦИЯ ИМПУЛЬСОВ
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_order_flows(self, closes: List[float], highs: List[float],
                             lows: List[float], volumes: List[float]
                             ) -> List[OrderFlowInfo]:
        """
        Ищем Order Flow — импульсные движения, вызванные институциональными
        входами.

        Критерии импульса:
          - 3+ последовательных свечи в одном направлении
          - Общий ход > 0.3% от средней цены
          - Объём > 1.5x среднего за 20 свечей
          - (опционально) свечи с малыми тенями — агрессивный вход

        Returns:
            список OrderFlowInfo (только значимые импульсы, не старше 24 свечей)
        """
        if len(closes) < 30:
            return []

        n = min(24, len(closes))
        avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else 1
        avg_price = np.mean(closes[-n:])

        flows: List[OrderFlowInfo] = []
        i = len(closes) - n

        while i < len(closes) - 3:
            # Ищем 3+ бычьих свечи подряд
            bull_count = 0
            bear_count = 0
            j = i
            while j < len(closes):
                if closes[j] > closes[j-1] if j > 0 else False:
                    bull_count += 1
                    bear_count = 0
                elif closes[j] < closes[j-1] if j > 0 else False:
                    bear_count += 1
                    bull_count = 0
                else:
                    bull_count = 0
                    bear_count = 0

                if bull_count >= 3:
                    break
                if bear_count >= 3:
                    break
                j += 1

            if bull_count >= 3:
                # Запомнили импульс
                start = max(0, j - bull_count)
                end = j
                move_pct = abs(closes[end] - closes[start]) / avg_price * 100
                vol_sum = sum(volumes[start:end+1]) if len(volumes) > end else 0
                vol_ratio = vol_sum / (avg_vol * (end - start + 1) + 1e-8)

                if move_pct > 0.3 and vol_ratio > 1.2:
                    # Проверяем был ли sweep перед импульсом
                    sweep = self._check_sweep_before(highs, lows, closes,
                                                      volumes, start, avg_vol,
                                                      direction='bullish')
                    flows.append(OrderFlowInfo(
                        price_start=closes[start],
                        price_end=closes[end],
                        direction='bullish',
                        volume_ratio=vol_ratio,
                        candle_count=end - start + 1,
                        sweep=sweep,
                    ))
                i = end + 1
                continue

            if bear_count >= 3:
                start = max(0, j - bear_count)
                end = j
                move_pct = abs(closes[end] - closes[start]) / avg_price * 100
                vol_sum = sum(volumes[start:end+1]) if len(volumes) > end else 0
                vol_ratio = vol_sum / (avg_vol * (end - start + 1) + 1e-8)

                if move_pct > 0.3 and vol_ratio > 1.2:
                    sweep = self._check_sweep_before(highs, lows, closes,
                                                      volumes, start, avg_vol,
                                                      direction='bearish')
                    flows.append(OrderFlowInfo(
                        price_start=closes[start],
                        price_end=closes[end],
                        direction='bearish',
                        volume_ratio=vol_ratio,
                        candle_count=end - start + 1,
                        sweep=sweep,
                    ))
                i = end + 1
                continue

            i += 1

        # Оставляем только последние 24 свечи
        flows = [f for f in flows if f.candle_count <= 24]
        return flows[:6]  # не болEE 6 потоков

    def _check_sweep_before(self, highs: List[float], lows: List[float],
                              closes: List[float], volumes: List[float],
                              start_idx: int, avg_vol: float,
                              direction: str) -> bool:
        """
        Проверяем, был ли захват ликвидности (sweep) ПЕРЕД импульсом.

        Для bullish: перед импульсом цена должна сходить ниже последнего
                      минимума (sweep стоп-лоссов лонгов).
        Для bearish: перед импульсом цена должна сходить выше последнего
                      максимума (sweep стоп-лоссов шортов).

        Returns:
            True если sweep обнаружен
        """
        if start_idx < 5:
            return False

        lookback = min(10, start_idx)
        before_highs = highs[start_idx - lookback:start_idx]
        before_lows = lows[start_idx - lookback:start_idx]

        if direction == 'bullish':
            min_before = min(before_lows[:-1]) if len(before_lows) > 1 else before_lows[0]
            current_low = before_lows[-1]
            # Бычий sweep: цена ушла ниже недавнего минимума перед ростом
            if current_low < min_before * 0.998:
                # Проверяем что объём на sweep был (сбор ликвидности)
                sweep_vol = volumes[start_idx - 1] if len(volumes) > start_idx - 1 else 0
                if sweep_vol > avg_vol * 0.8:
                    return True

        elif direction == 'bearish':
            max_before = max(before_highs[:-1]) if len(before_highs) > 1 else before_highs[0]
            current_high = before_highs[-1]
            # Медвежий sweep: цена ушла выше недавнего максимума перед падением
            if current_high > max_before * 1.002:
                sweep_vol = volumes[start_idx - 1] if len(volumes) > start_idx - 1 else 0
                if sweep_vol > avg_vol * 0.8:
                    return True

        return False

    # ──────────────────────────────────────────────────────────────────────────
    # ORDER BLOCK — ДЕТЕКЦИЯ БЛОКОВ
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_order_blocks(self, closes: List[float], highs: List[float],
                              lows: List[float], volumes: List[float],
                              order_flows: List[OrderFlowInfo]
                              ) -> List[OrderBlockInfo]:
        """
        Находим Order Blocks — зоны, откуда начался импульс (Order Flow).

        OB = последняя свеча ПЕРЕД началом импульса (или группа свечей).
        Для бычьего OF: OB — свеча с минимальным low перед импульсом
        Для медвежьего OF: OB — свеча с максимальным high перед импульсом

        Классификация:
          - IDM: побуждение (первая значимая зона)
          - IDM OB: ордер блок за IDM
          - EXT OB: экстремальный блок (максимальная ликвидность)
        """
        blocks: List[OrderBlockInfo] = []

        for of in order_flows:
            if of.direction == 'bullish':
                # Ищем свечу перед импульсом с наибольшим низом
                # (институциональный вход)
                ob_start = of.price_start
                ob_idx = self._find_ob_index(closes, lows, ob_start, direction='bullish')

                if ob_idx is not None and ob_idx > 0:
                    ob_low = lows[ob_idx]
                    ob_high = highs[ob_idx]

                    # Плюс одна свеча до (контекст)
                    if ob_idx > 1:
                        ob_low = min(ob_low, lows[ob_idx - 1])
                        ob_high = max(ob_high, highs[ob_idx - 1])

                    # Проверяем mitigation
                    mitigated = self._is_mitigated(closes, highs, lows,
                                                    ob_low, ob_high, ob_idx)
                    blocks.append(OrderBlockInfo(
                        price_high=ob_high,
                        price_low=ob_low,
                        kind='ext_ob' if of.sweep else 'idm_ob',
                        direction='bullish',
                        mitigated=mitigated,
                        strength=min(1.0, of.volume_ratio * 0.4),
                    ))
            else:
                ob_start = of.price_start
                ob_idx = self._find_ob_index(closes, highs, ob_start,
                                              direction='bearish')

                if ob_idx is not None and ob_idx > 0:
                    ob_low = lows[ob_idx]
                    ob_high = highs[ob_idx]

                    if ob_idx > 1:
                        ob_low = min(ob_low, lows[ob_idx - 1])
                        ob_high = max(ob_high, highs[ob_idx - 1])

                    mitigated = self._is_mitigated(closes, highs, lows,
                                                    ob_low, ob_high, ob_idx)

                    blocks.append(OrderBlockInfo(
                        price_high=ob_high,
                        price_low=ob_low,
                        kind='ext_ob' if of.sweep else 'idm_ob',
                        direction='bearish',
                        mitigated=mitigated,
                        strength=min(1.0, of.volume_ratio * 0.4),
                    ))

        # Сортируем по силе
        blocks.sort(key=lambda b: b.strength, reverse=True)
        return blocks[:6]  # не болEE 6 блоков

    def _find_ob_index(self, closes: List[float], prices: List[float],
                        target_price: float, direction: str) -> Optional[int]:
        """
        Найти индекс свечи, которая является ордер блоком.

        Для bullish: ищем локальный минимум ПЕРЕД началом импульса
        Для bearish: ищем локальный максимум ПЕРЕД началом импульса
        """
        # Ищем target_price в closes
        try:
            idx = list(closes).index(target_price)
        except ValueError:
            # Не нашли точную цену — ищем ближайшую
            diffs = [abs(c - target_price) for c in closes]
            idx = diffs.index(min(diffs))

        if idx < 2:
            return None

        if direction == 'bullish':
            # Ищем минимум за 5 свечей ДО импульса
            lookback = min(5, idx)
            segment = closes[idx - lookback:idx + 1]
            min_val = min(segment)
            min_idx = closes.index(min_val) if min_val in closes else idx - lookback
            return max(0, min_idx - 1)
        else:
            lookback = min(5, idx)
            segment = closes[idx - lookback:idx + 1]
            max_val = max(segment)
            max_idx = closes.index(max_val) if max_val in closes else idx - lookback
            return max(0, max_idx - 1)

    def _is_mitigated(self, closes: List[float], highs: List[float],
                       lows: List[float], ob_low: float, ob_high: float,
                       ob_idx: int) -> bool:
        """
        Проверяем, перекрыта ли зона OB телом свечи после её формирования.

        По методологии Ханна: зона считается немитигированной, пока тело
        свечи не перекрыло её. После перекрытия — зона не валидна.
        Однако мы считаем зону валидной ДО полного перекрытия телом.
        """
        # Смотрим свечи ПОСЛЕ OB
        for i in range(ob_idx + 1, len(closes)):
            c = closes[i]
            o = closes[i - 1] if i > 0 else c
            body_high = max(o, c)
            body_low = min(o, c)

            # Тело полностью перекрыло зону OB
            if body_high >= ob_high and body_low <= ob_low:
                return True

        return False

    # ──────────────────────────────────────────────────────────────────────────
    # POI — ЗОНЫ ИНТЕРЕСА (IDM → IDM OB → EXT OB)
    # ──────────────────────────────────────────────────────────────────────────

    def _classify_poi(self, order_blocks: List[OrderBlockInfo]
                       ) -> Tuple[Optional[float],
                                  Optional[Tuple[float, float]],
                                  Optional[Tuple[float, float]]]:
        """
        Классифицируем 3 ключевые зоны POI.

        Returns:
            (idm_level, idm_ob_zone, ext_ob_zone)
        """
        idm_level = None
        idm_ob_zone = None
        ext_ob_zone = None

        alive = [ob for ob in order_blocks if not ob.mitigated]
        if not alive:
            return None, None, None

        # IDM: первая значимая зона (самая слабая)
        if len(alive) >= 1:
            idm_block = alive[-1]  # самая слабая/старая
            idm_level = (idm_block.price_low + idm_block.price_high) / 2
            idm_ob_zone = (idm_block.price_low, idm_block.price_high)

        # EXT OB: сильнейший блок (высокая ликвидность)
        if alive:
            ext_block = max(alive, key=lambda b: b.strength)
            ext_ob_zone = (ext_block.price_low, ext_block.price_high)

        return idm_level, idm_ob_zone, ext_ob_zone

    # ──────────────────────────────────────────────────────────────────────────
    # ИТОГОВАЯ ОЦЕНКА (С УЧЁТОМ OF/OB)
    # ──────────────────────────────────────────────────────────────────────────

    def _calc_score(self, closes, highs, lows, volumes,
                     current_price, vah, val, poc,
                     cluster_quality, fvg_above, fvg_below,
                     vwap_1h, order_flows, order_blocks,
                     poi_idm, poi_idm_ob, poi_ext_ob,
                     bearish_context: bool = False) -> int:
        """Расчёт итогового score с учётом новых метрик."""
        score = 50

        # ── 1. Volume Profile (как было) ──────────────────────────────────
        in_val = val <= current_price <= vah
        near_poc = abs(current_price - poc) / (poc + 1e-8) < 0.005
        near_vah = abs(current_price - vah) / (vah + 1e-8) < 0.005
        near_val = abs(current_price - val) / (val + 1e-8) < 0.005

        if in_val:
            score = 55
            if near_poc:
                score = 60
            if near_poc and cluster_quality > 0.6:
                score = 65
        else:
            if current_price > vah:
                score = 40 if fvg_above else (70 if cluster_quality > 0.5 else 60)
            else:
                score = 40 if fvg_below else (70 if cluster_quality > 0.5 else 60)

        # ── 2. Контекстный штраф ───────────────────────────────────────
        # Если цена ниже VWAP 4H или тренд падающий — ослабляем все
        # буллиш сигналы от OB/POI/POC в 2 раза
        ctx_mult = 0.5 if bearish_context else 1.0

        # ── 2. VWAP бонус ────────────────────────────────────────────────
        if vwap_1h and abs(current_price - vwap_1h) / (vwap_1h + 1e-8) < 0.003:
            score = min(90, score + 20)

        # ── 3. FVG возврат ────────────────────────────────────────────────
        if near_vah and fvg_above:
            score = min(80, score + 15)
        if near_val and fvg_below:
            score = min(80, score + 15)

        # ── 4. Breakout Trap ─────────────────────────────────────────────
        trap = self._detect_breakout_trap(highs[-20:], lows[-20:], closes[-20:],
                                           volumes[-20:] if len(volumes) >= 20 else [],
                                           vah, val)
        if trap == 'bullish':
            score = min(90, score + int(20 * ctx_mult))
        elif trap == 'bearish':
            score = min(90, score + int(20 * ctx_mult))
        elif trap == 'fake_breakout':
            score = min(85, score + int(15 * ctx_mult))

        # ── НОВОЕ: Order Flow бонус ──────────────────────────────────────
        for of in order_flows:
            # Цена рядом с началом OF — потенциальный вход
            dist_to_of_start = abs(current_price - of.price_start) / (of.price_start + 1e-8)
            if dist_to_of_start < 0.005:  # 0.5%
                if of.sweep:
                    # OF со sweep → EXT OB (очень сильный сигнал)
                    score = min(95, score + int(25 * ctx_mult))
                elif of.volume_ratio > 2.0:
                    score = min(90, score + int(20 * ctx_mult))
                else:
                    score = min(80, score + int(15 * ctx_mult))

            # Цена внутри OF (импульс в нашу сторону) — продолжение
            if of.direction == 'bullish' and current_price > of.price_start:
                if of.price_start <= current_price <= of.price_end:
                    score = min(85, score + int(10 * ctx_mult))
            elif of.direction == 'bearish' and current_price < of.price_start:
                if of.price_end <= current_price <= of.price_start:
                    score = min(85, score + int(10 * ctx_mult))

        # ── НОВОЕ: Order Block бонус ──────────────────────────────────────
        for ob in order_blocks:
            if ob.mitigated:
                continue  # не используем mitigated блоки

            in_ob = ob.price_low <= current_price <= ob.price_high
            near_ob = (abs(current_price - ob.price_low) / (ob.price_low + 1e-8) < 0.003 or
                       abs(current_price - ob.price_high) / (ob.price_high + 1e-8) < 0.003)

            if in_ob:
                if ob.kind == 'ext_ob':
                    score = min(95, score + int(25 * ctx_mult))
                elif ob.kind == 'idm_ob':
                    score = min(85, score + int(15 * ctx_mult))
                else:
                    score = min(75, score + int(10 * ctx_mult))
            elif near_ob:
                if ob.kind == 'ext_ob':
                    score = min(90, score + int(20 * ctx_mult))
                elif ob.kind == 'idm_ob':
                    score = min(80, score + int(12 * ctx_mult))

        # ── НОВОЕ: POI зоны ──────────────────────────────────────────────
        if poi_idm_ob:
            lo, hi = poi_idm_ob
            if lo <= current_price <= hi:
                score = min(85, score + int(15 * ctx_mult))
        if poi_ext_ob:
            lo, hi = poi_ext_ob
            if lo <= current_price <= hi:
                score = min(95, score + int(20 * ctx_mult))

        return int(score)

    # ──────────────────────────────────────────────────────────────────────────
    # VOLUME PROFILE (БЕЗ ИЗМЕНЕНИЙ)
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_closes(self, candles: List) -> List[float]:
        if not candles:
            return []
        if isinstance(candles[0], dict):
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

    def _volume_profile(self, closes, highs, lows, volumes):
        if not closes or not volumes:
            return 0, 0, 0, {}

        price_range = max(highs[-self.window:]) - min(lows[-self.window:])
        if price_range == 0:
            return closes[-1], closes[-1], closes[-1], {}
        tick_size = max(price_range * 0.002, 0.0001)

        n = min(self.window, len(closes))
        clusters: Dict[float, float] = {}
        for i in range(-n, 0):
            price_bin = round(closes[i] / tick_size) * tick_size
            clusters[price_bin] = clusters.get(price_bin, 0) + volumes[i]

        if not clusters:
            return closes[-1], closes[-1], closes[-1], {}

        poc = max(clusters, key=clusters.get)
        total_vol = sum(clusters.values())
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
        if not clusters or len(clusters) < 3:
            return 0.3
        max_vol = max(clusters.values())
        if max_vol == 0:
            return 0.3
        mean_vol = sum(clusters.values()) / len(clusters)
        return min(1.0, (max_vol / mean_vol) * 0.15)

    # ──────────────────────────────────────────────────────────────────────────
    # BREAKOUT TRAP (БЕЗ ИЗМЕНЕНИЙ)
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_breakout_trap(self, highs, lows, closes, volumes, vah, val):
        if len(highs) < 10 or len(lows) < 10 or not volumes:
            return ''
        n = min(6, len(highs))
        recent_highs = highs[-n:]
        recent_lows = lows[-n:]
        recent_closes = closes[-n:]
        recent_vols = volumes[-n:] if len(volumes) >= n else []
        avg_vol = sum(recent_vols[:-1]) / max(len(recent_vols[:-1]), 1)

        broke_vah = any(h > vah * 1.001 for h in recent_highs)
        back_below_vah = recent_closes[-1] < vah
        back_below_vah_2 = recent_closes[-2] < vah if n >= 2 else True
        if broke_vah and back_below_vah and back_below_vah_2:
            last_vol_low = len(recent_vols) >= 3 and recent_vols[-1] < avg_vol * 0.8
            if last_vol_low:
                return 'bullish'

        broke_val = any(l < val * 0.999 for l in recent_lows)
        back_above_val = recent_closes[-1] > val
        back_above_val_2 = recent_closes[-2] > val if n >= 2 else True
        if broke_val and back_above_val and back_above_val_2:
            last_vol_low = len(recent_vols) >= 3 and recent_vols[-1] < avg_vol * 0.8
            if last_vol_low:
                return 'bearish'

        if n >= 5 and len(recent_vols) >= 5:
            two_above = all(h > vah * 1.001 for h in recent_highs[-4:-2])
            last_two_below = all(recent_closes[-i] < vah * 1.001 for i in range(1, 3)) if n >= 2 else False
            if two_above and last_two_below and recent_vols[-1] > avg_vol * 1.3:
                return 'fake_breakout'
        return ''

    # ──────────────────────────────────────────────────────────────────────────
    # SWING LEVELS (старая ликвидность)
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_swing_levels(self, candles_htf: Optional[List],
                              current_price: float) -> Tuple[str, str]:
        """
        Ищет ключевые уровни старой ликвидности (swing lows/highs) на 4h.

        Returns:
            (position, detail):
              position = 'below' | 'above' | ''
              detail = '(уровень: цена)'
        """
        if not candles_htf or len(candles_htf) < 15:
            return '', ''

        highs = self._extract_highs(candles_htf)
        lows = self._extract_lows(candles_htf)

        if len(highs) < 15:
            return '', ''

        # Ищем swing lows: минимум, который ниже обоих соседей
        swing_lows = []
        for i in range(3, len(lows) - 3):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i-3] \
               and lows[i] < lows[i+1] and lows[i] < lows[i+2] and lows[i] < lows[i+3]:
                swing_lows.append(lows[i])

        # Ищем swing highs: максимум, который выше обоих соседей
        swing_highs = []
        for i in range(3, len(highs) - 3):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i-3] \
               and highs[i] > highs[i+1] and highs[i] > highs[i+2] and highs[i] > highs[i+3]:
                swing_highs.append(highs[i])

        # Ищем ближайший уровень (снизу или сверху) в радиусе 3%
        best_level = None
        best_dist = current_price * 0.03
        best_pos = ''

        if swing_lows:
            below_levels = [l for l in swing_lows if 0 < current_price - l < current_price * 0.03]
            if below_levels:
                nearest_below = max(below_levels)
                dist = current_price - nearest_below
                if dist < best_dist:
                    best_dist = dist
                    best_level = nearest_below
                    best_pos = 'below'

        if swing_highs:
            above_levels = [h for h in swing_highs if 0 < h - current_price < current_price * 0.03]
            if above_levels:
                nearest_above = min(above_levels)
                dist = nearest_above - current_price
                if dist < best_dist:
                    best_dist = dist
                    best_level = nearest_above
                    best_pos = 'above'

        if best_level:
            dist_pct = best_dist / current_price * 100
            return best_pos, f'{best_level:.4f}({dist_pct:.1f}%)'

        return '', ''

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
            if lows[i] > highs[i - 1]:
                fvg_above.append((highs[i - 1], lows[i]))
            if highs[i] < lows[i - 1]:
                fvg_below.append((highs[i], lows[i - 1]))

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
