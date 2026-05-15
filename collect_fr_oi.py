#!/usr/bin/env python3
"""
collect_fr_oi.py — Сбор funding rate + open interest для BTC.

Складывает в CSV, чтобы модель BTC Direction могла использовать
как дополнительные признаки.

Запуск: каждые 1-4 часа (cron).
Данные: ~/data/btc_fr_oi.csv
         timestamp (UTC ms) | funding_rate | open_interest | open_interest_value
"""

import os, sys, time, json, logging
from datetime import datetime

logger = logging.getLogger('fr_oi_collector')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
CSV_PATH = os.path.join(DATA_DIR, 'btc_fr_oi.csv')


def collect(exchange) -> int:
    """Собрать FR и OI, дописать в CSV. Возвращает сколько новых записей."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # ── Загружаем существующие данные ──────────────────────────────
    existing = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('timestamp'):
                    existing.add(line.split(',')[0])  # timestamp как ключ
    
    # ── 1. Funding Rate (Bybit: каждые 8ч, до 200 записей) ─────────
    records = []
    try:
        fr_data = exchange.fetch_funding_rate_history('BTC/USDT:USDT', limit=200)
        for entry in fr_data:
            ts_ms = str(entry['timestamp'])
            rate = entry.get('fundingRate', 0)
            if ts_ms not in existing:
                records.append({
                    'ts': ts_ms,
                    'fr': rate,
                    'oi': '',  # будет заполнено ниже
                    'oi_val': '',
                })
    except Exception as e:
        logger.warning(f"⚠️ FR fetch: {e}")
    
    # ── 2. Open Interest (1h свечи, до 200 записей) ────────────────
    try:
        oi_data = exchange.fetch_open_interest_history('BTC/USDT:USDT', '1h', limit=200)
        oi_map = {}
        for entry in oi_data:
            ts_ms = str(entry['timestamp'])
            oi_map[ts_ms] = {
                'oi': entry.get('openInterestAmount', 0),
                'oi_val': entry.get('openInterestValue', 0),
            }
        
        for entry in oi_data:
            ts_ms = str(entry['timestamp'])
            if ts_ms not in existing:
                records.append({
                    'ts': ts_ms,
                    'fr': '',
                    'oi': entry.get('openInterestAmount', 0),
                    'oi_val': entry.get('openInterestValue', 0),
                })
    except Exception as e:
        logger.warning(f"⚠️ OI fetch: {e}")
    
    if not records:
        return 0
    
    # ── Сортируем по времени и пишем ──────────────────────────────
    records.sort(key=lambda r: r['ts'])
    
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, 'a') as f:
        if write_header:
            f.write('timestamp,funding_rate,open_interest,open_interest_value\n')
        for r in records:
            f.write(f"{r['ts']},{r['fr']},{r['oi']},{r['oi_val']}\n")
    
    # ── Объединяем дубли: если один ts имеет и FR и OI ─────────────
    _deduplicate(CSV_PATH)
    
    return len(records)


def _deduplicate(path: str):
    """Объединить строки с одинаковым timestamp (FR в одной, OI в другой)."""
    rows = {}
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',', 3)
            ts = parts[0]
            if ts not in rows:
                rows[ts] = ['', '', '']
            # funding_rate
            if parts[1]:
                rows[ts][0] = parts[1]
            # open_interest
            if len(parts) > 2 and parts[2]:
                rows[ts][1] = parts[2]
            # open_interest_value
            if len(parts) > 3 and parts[3]:
                rows[ts][2] = parts[3]
    
    with open(path, 'w') as f:
        f.write('timestamp,funding_rate,open_interest,open_interest_value\n')
        for ts in sorted(rows.keys()):
            f.write(f"{ts},{rows[ts][0]},{rows[ts][1]},{rows[ts][2]}\n")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    import ccxt
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'api_config_final.json')) as f:
        config = json.load(f)
    
    exchange = ccxt.bybit(config)
    
    n = collect(exchange)
    print(f"✅ Собрано {n} новых записей")
    print(f"📁 {CSV_PATH}")
