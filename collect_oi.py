#!/usr/bin/env python3
"""
collect_oi.py — Сбор Open Interest (OI) для всех торгуемых монет.

Складывает OI Delta в JSON, чтобы Liq модуль мог использовать
для расчёта уровней ликвидаций (аналогично TradingView индикатору).

Запуск: каждые 15 минут из main.py
Данные: ~/data/oi_data.json
"""

import os, sys, time, json, logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger('oi_collector')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OI_PATH = os.path.join(DATA_DIR, 'oi_data.json')

# Монеты, по которым собираем OI (фьючерсные контракты)
OI_SYMBOLS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
    'DOGE/USDT:USDT', 'ADA/USDT:USDT', 'LINK/USDT:USDT', 'DOT/USDT:USDT',
    'AVAX/USDT:USDT', 'APT/USDT:USDT', 'ARB/USDT:USDT', 'OP/USDT:USDT',
    'ATOM/USDT:USDT', 'NEAR/USDT:USDT', 'ALGO/USDT:USDT', 'FIL/USDT:USDT',
    'ROSE/USDT:USDT', 'EGLD/USDT:USDT', 'MNT/USDT:USDT',
]

# Маппинг: фьючерсный символ → спотовый (для поиска цены)
SPOT_MAP = {s.replace(':USDT', ''): s for s in OI_SYMBOLS}


class OICollector:
    """
    Сборщик OI данных.

    Хранит историю OI по каждой монете, считает OI Delta
    (по уникальным часам), скользящую среднюю и определяет
    моменты всплесков → уровни ликвидаций.
    """

    def __init__(self):
        self.data: Dict[str, List[Dict]] = {}
        self._call_count: int = 0
        self._load()

    # ──────────────────────────────────────────────────────────────────────
    # ЗАГРУЗКА / СОХРАНЕНИЕ
    # ──────────────────────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(OI_PATH):
            try:
                with open(OI_PATH) as f:
                    raw = json.load(f)
                self.data = raw.get('symbols', {})
                self._last_save = raw.get('saved_at', 0)
                total = sum(len(v) for v in self.data.values())
                logger.info(f"📂 OI loaded: {len(self.data)} symbols, {total} records")
            except Exception as e:
                logger.warning(f"⚠️ OI load error: {e}")
                self.data = {}
                self._last_save = 0
        else:
            self.data = {}
            self._last_save = 0

    def save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        # Ограничим каждую историю 500 записями (≈5 дней по 15 мин)
        for sym in self.data:
            if len(self.data[sym]) > 500:
                self.data[sym] = self.data[sym][-500:]
        try:
            with open(OI_PATH, 'w') as f:
                json.dump({
                    'saved_at': time.time(),
                    'symbols': self.data,
                }, f)
            self._last_save = time.time()
        except Exception as e:
            logger.warning(f"⚠️ OI save error: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # СБОР OI С БИРЖИ
    # ──────────────────────────────────────────────────────────────────────

    def collect(self, exchange) -> Dict[str, Dict]:
        """
        Собрать OI для всех монет.

        Returns:
            {spot_symbol: {oi, price, oi_delta, ma_delta, heat, levels...}}
        """
        results = {}
        now_ts = int(time.time() * 1000)  # ms

        for symbol in OI_SYMBOLS:
            try:
                oi = exchange.fetch_open_interest(symbol)
                amount = float(oi.get('openInterestAmount', 0))
                ts = oi.get('timestamp', now_ts)
            except Exception as e:
                logger.debug(f"⚠️ OI {symbol}: {e}")
                continue

            spot_sym = symbol.replace(':USDT', '')

            # Получаем цену монеты
            try:
                ticker = exchange.fetch_ticker(spot_sym)
                price = ticker.get('last', 0)
            except:
                price = 0

            record = {'ts': ts, 'oi': amount, 'price': price}
            if spot_sym not in self.data:
                self.data[spot_sym] = []
            self.data[spot_sym].append(record)

            enriched = self._enrich(spot_sym, amount, price)
            results[spot_sym] = enriched

        # Автосохранение раз в 3 вызова
        self._call_count += 1
        if self._call_count % 3 == 0:
            self.save()

        return results

    # ──────────────────────────────────────────────────────────────────────
    # OI DELTA + УРОВНИ ЛИКВИДАЦИЙ (исправлено: по уникальным часам)
    # ──────────────────────────────────────────────────────────────────────

    def _enrich(self, spot_sym: str, current_oi: float, price: float) -> Dict:
        """
        Рассчитать OI Delta и уровни ликвидаций.

        Ключевое отличие: OI обновляется раз в 1m..1h, поэтому дельту
        считаем между УНИКАЛЬНЫМИ часовыми значениями, а не между двумя
        последовательными замерами в пределах одного часа.
        """
        result = {
            'oi': current_oi,
            'price': price,
            'oi_delta': 0,
            'oi_delta_abs': 0,
            'ma_delta': 0,
            'h3_levels': [],
            'h2_levels': [],
            'h1_levels': [],
            'heat': 0,
        }

        hist = self.data.get(spot_sym, [])
        if len(hist) < 2:
            return result

        # ─── Шаг 1: группируем записи по уникальному часу ──────────────
        hourly = {}  # {hour_ts_ms: oi}
        for entry in hist:
            hour_ts = (entry['ts'] // 3600000) * 3600000
            hourly[hour_ts] = entry['oi']

        # Добавляем/обновляем текущий час
        current_hour = (int(time.time() * 1000) // 3600000) * 3600000
        hourly[current_hour] = current_oi

        hours = sorted(hourly.keys())
        if len(hours) < 2:
            return result

        # ─── Шаг 2: OI Delta между последними двумя часами ────────────
        prev_hour_oi = hourly[hours[-2]]
        delta = current_oi - prev_hour_oi
        delta_abs = abs(delta)

        result['oi_delta'] = delta
        result['oi_delta_abs'] = delta_abs

        # ─── Шаг 3: SMA от |OI Delta| (80 периодов, по часам) ─────────
        hourly_deltas = []
        for i in range(1, len(hours)):
            d = hourly[hours[i]] - hourly[hours[i-1]]
            hourly_deltas.append(abs(d))

        n = min(80, len(hourly_deltas))
        if n == 0:
            return result

        ma_delta = sum(hourly_deltas[-n:]) / n
        result['ma_delta'] = ma_delta

        # ─── Шаг 4: пороги и уровни ликвидаций ────────────────────────
        h3_threshold = ma_delta * 3.4
        h2_threshold = ma_delta * 2.2
        h1_threshold = ma_delta * 1.8

        leverages = [0.01, 0.02, 0.04, 0.1, 0.2]
        lev_labels = ['100x', '50x', '25x', '10x', '5x']

        def _build_levels(n_levers: int) -> list:
            return [
                {'level': lev_labels[li],
                 'liq_long': price * (1 - lev),
                 'liq_short': price * (1 + lev)}
                for li, lev in enumerate(leverages[:n_levers])
            ]

        if delta > 0:  # Только рост OI (как в TV индикаторе)
            if delta_abs >= h3_threshold:
                result['h3_levels'] = _build_levels(5)
                result['heat'] = 3
            elif delta_abs >= h2_threshold:
                result['h2_levels'] = _build_levels(4)
                result['heat'] = 2
            elif delta_abs >= h1_threshold:
                result['h1_levels'] = _build_levels(3)
                result['heat'] = 1

        return result

    # ──────────────────────────────────────────────────────────────────────
    # API ДЛЯ LIQ МОДУЛЯ (возвращает уровни для конкретной монеты)
    # ──────────────────────────────────────────────────────────────────────

    def get_liq_levels(self, spot_sym: str, current_price: float) -> Dict:
        """
        Получить уровни ликвидации для монеты.

        Returns:
            {
                'heat': 0-3,
                'score_bonus': 0-35,
                'liq_zone_long': (low, high) or None,
                'liq_zone_short': (low, high) or None,
                'levels': [...]
            }
        """
        result = {'heat': 0, 'score_bonus': 0, 'liq_zone_long': None,
                  'liq_zone_short': None, 'levels': []}

        # Без данных — ничего не даём
        if spot_sym not in self.data or len(self.data[spot_sym]) < 2:
            return result

        last_oi = self.data[spot_sym][-1]['oi']
        enriched = self._enrich(spot_sym, last_oi, current_price)

        levels = enriched.get('h3_levels', []) or enriched.get('h2_levels', []) or enriched.get('h1_levels', [])
        result['levels'] = levels

        heat = enriched.get('heat', 0)
        result['heat'] = heat

        if heat >= 3:
            result['score_bonus'] = 25
            if levels:
                min_long = min(l['liq_long'] for l in levels)
                max_long = max(l['liq_long'] for l in levels)
                min_short = min(l['liq_short'] for l in levels)
                max_short = max(l['liq_short'] for l in levels)
                result['liq_zone_long'] = (min_long, max_long)
                result['liq_zone_short'] = (min_short, max_short)

        elif heat >= 2:
            result['score_bonus'] = 15
            if levels:
                result['liq_zone_long'] = (levels[-1]['liq_long'], levels[0]['liq_long'])
                result['liq_zone_short'] = (levels[0]['liq_short'], levels[-1]['liq_short'])

        elif heat >= 1:
            result['score_bonus'] = 8
            if levels:
                result['liq_zone_long'] = (levels[-1]['liq_long'], levels[0]['liq_long'])
                result['liq_zone_short'] = (levels[0]['liq_short'], levels[-1]['liq_short'])

        # Если цена внутри зоны ликвидации — повышенный бонус
        if result['liq_zone_long'] and current_price >= result['liq_zone_long'][0] and current_price <= result['liq_zone_long'][1]:
            result['score_bonus'] = min(35, result['score_bonus'] + 10)
        if result['liq_zone_short'] and current_price >= result['liq_zone_short'][0] and current_price <= result['liq_zone_short'][1]:
            result['score_bonus'] = min(35, result['score_bonus'] + 10)

        return result


# Глобальный экземпляр
_oi_collector = None


def get_oi_collector() -> OICollector:
    global _oi_collector
    if _oi_collector is None:
        _oi_collector = OICollector()
    return _oi_collector


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    import ccxt
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'api_config_final.json')) as f:
        config = json.load(f)

    exchange = ccxt.bybit(config)
    collector = get_oi_collector()

    # Загружаем историю OI для быстрого старта
    print("Загружаем OI историю (по 5 часов на монету)...")
    for oi_sym in OI_SYMBOLS:
        spot_sym = oi_sym.replace(':USDT', '')
        try:
            hist = exchange.fetch_open_interest_history(oi_sym, '1h', limit=5)
            for o in hist:
                ts = o['timestamp']
                oi_val = o['openInterestAmount']
                if spot_sym not in collector.data:
                    collector.data[spot_sym] = []
                collector.data[spot_sym].append({'ts': ts, 'oi': oi_val, 'price': 0})
        except Exception as e:
            pass
    print(f"  Загружено ≈{sum(len(v) for v in collector.data.values())} записей")

    # Живой сбор
    results = collector.collect(exchange)
    print(f"\n✅ Собрано OI для {len(results)} монет\n")

    # Вывод с тепловой картой
    print(f"{'Монета':10s} | {'OI':>12s} | {'Delta':>10s} | {'MA':>10s} | {'Heat':>5s} | Уровни")
    print("-" * 80)
    for spot_sym in sorted(results.keys()):
        d = results[spot_sym]
        icon = {0:'⚪', 1:'🟡', 2:'🟠', 3:'🔴'}.get(d['heat'], '⚪')
        oi_str = f"{d['oi']:.1f}"
        delta_str = f"{d['oi_delta']:+.1f}" if abs(d['oi_delta']) > 0 else "0"
        ma_str = f"{d['ma_delta']:.1f}" if d['ma_delta'] > 0 else "-"
        levels = d.get('h3_levels', []) or d.get('h2_levels', []) or d.get('h1_levels', [])
        levels_str = " ".join(l['level'] for l in levels[:3]) if levels else "-"

        if d['heat'] > 0:
            print(f"{icon} {spot_sym:10s} | {oi_str:>12s} | {delta_str:>10s} | {ma_str:>10s} | {d['heat']:^5d} | {levels_str}")
        else:
            print(f"{icon} {spot_sym:10s} | {oi_str:>12s} | {delta_str:>10s} | {ma_str:>10s} | {d['heat']:^5d} | {levels_str}")

    # Детально для BTC
    print(f"\n=== Детально BTC ===")
    btc = results.get('BTC/USDT', {})
    for k, v in btc.items():
        if k in ('h3_levels', 'h2_levels', 'h1_levels'):
            if v:
                print(f"  {k}:")
                for l in v:
                    print(f"    {l['level']:5s}: long_liq=${l['liq_long']:.2f} short_liq=${l['liq_short']:.2f}")
        elif k != 'levels':
            print(f"  {k}: {v}")

    collector.save()
    print(f"\n✅ Данные сохранены в {OI_PATH}")
