#!/usr/bin/env python3
"""
Скрипт массовой загрузки исторических данных для всех торговых пар.
Скачивает: 1H (3+ года), 4H, D1, 5M (последние 7 дней)
Сохраняет в data/ папку в формате JSON, совместимом с ML-PRO v2.
"""
import json, ccxt, time, os, sys
from datetime import datetime

# ─── Конфигурация ──────────────────────────────────────────────────────────
START_DATE = "2023-01-01T00:00:00Z"  # 3+ года
RATE_LIMIT = 0.3  # задержка между запросами (сек)
TIMEFRAMES = {
    '1h': {'limit': 1000, 'label': '1H'},
    '4h': {'limit': 1000, 'label': '4H'},
    '1d': {'limit': 1000, 'label': 'D1'},
    '5m': {'limit': 1000, 'label': '5M', 'since_days': 7},  # только последние 7 дней
}

# ─── Инициализация ────────────────────────────────────────────────────────
c = json.load(open("config/api_config_final.json"))
e = ccxt.bybit({
    "apiKey": c["bybit"]["api_key"],
    "secret": c["bybit"]["secret"],
    "enableRateLimit": True,
    "options": {"defaultType": "spot"}
})

symbols = c['trading']['enabled_pairs']
os.makedirs("data", exist_ok=True)

def download_pair(symbol, timeframe, limit=1000, since_ts=None):
    """Скачать историю для одной пары/таймфрейма."""
    all_candles = []
    since = since_ts
    total = 0
    
    while True:
        try:
            ohlcv = e.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            
            # Конвертируем в наш формат
            for o in ohlcv:
                all_candles.append({
                    't': o[0],
                    'o': o[1],
                    'h': o[2],
                    'l': o[3],
                    'c': o[4],
                    'v': o[5],
                })
            
            total += len(ohlcv)
            
            # Если получили меньше лимита — достигли конца
            if len(ohlcv) < limit:
                break
            
            # Сдвигаемся вперёд
            since = ohlcv[-1][0] + 1
            time.sleep(RATE_LIMIT)
            
        except Exception as ex:
            print(f"    ⚠️ Ошибка: {ex}")
            time.sleep(2)
            break
    
    return all_candles

# ─── Загрузка ─────────────────────────────────────────────────────────────
total_pairs = len(symbols)
print(f"📥 Загрузка исторических данных: {total_pairs} пар")
print(f"   Таймфреймы: 1H (3+ года), 4H, D1, 5M (7 дней)")
print("=" * 60)

for idx, sym in enumerate(symbols, 1):
    print(f"\n[{idx}/{total_pairs}] {sym}")
    fname = sym.replace('/', '_').lower()
    
    for tf, cfg in TIMEFRAMES.items():
        if tf == '5m':
            # 5M — только последние 7 дней (со стартом от сейчас)
            since = None
            label = f"{cfg['label']} (7 дней)"
        else:
            since = e.parse8601(START_DATE)
            label = cfg['label']
        
        candles = download_pair(sym, tf, cfg['limit'], since)
        
        out_path = f"data/{fname}_{tf}.json"
        with open(out_path, 'w') as f:
            json.dump(candles, f)
        
        # Даты
        if candles:
            first = datetime.fromtimestamp(candles[0]['t'] / 1000)
            last = datetime.fromtimestamp(candles[-1]['t'] / 1000)
            print(f"  {label:10s}: {len(candles):>5} свечей ({first.date()} → {last.date()})")
        else:
            print(f"  {label:10s}: ❌ пусто")

# ─── Итог ─────────────────────────────────────────────────────────────────
total_files = len(os.listdir("data"))
total_size = sum(os.path.getsize(f"data/{f}") for f in os.listdir("data")) / 1024 / 1024
print(f"\n{'=' * 60}")
print(f"✅ Загрузка завершена!")
print(f"   Файлов: {total_files}")
print(f"   Объём: {total_size:.1f} MB")
print(f"   Данные готовы для обучения ML-PRO v2 на истории")
