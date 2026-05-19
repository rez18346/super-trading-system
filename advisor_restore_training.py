#!/usr/bin/env python3
"""
Восстановление training_data для ML-советника из исторических сделок.
Берёт sell-трейды из SQL, для каждого находит вход (buy), 
качает OHLCV с Bybit и считает 14 признаков.

Запуск: python3 advisor_restore_training.py [--max=N] [--force]
"""

import sys
import os
import json
import sqlite3
import time
import logging
import numpy as np
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("ADVISOR_RESTORE")

# Загружаем конфиг Bybit
CONFIG_PATH = os.path.join(BASE_DIR, 'config/api_config_final.json')
config = json.load(open(CONFIG_PATH))

# Подключаем ccxt
import ccxt
exchange = ccxt.bybit({
    'apiKey': config['bybit']['api_key'],
    'secret': config['bybit']['secret'],
    'password': config['bybit']['password'],
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'},
})

# БД
DB_PATH = os.path.join(BASE_DIR, 'data/trading.db')

# Получаем trading_data.json
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "data/training_data.json")

def fetch_ohlcv(symbol: str, since_ts: int, limit: int = 60) -> list:
    """Получить OHLCV (1H) с Bybit начиная с since_ts (ms)"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=since_ts, limit=limit)
        return ohlcv
    except Exception as e:
        log.warning(f"  fetch_ohlcv {symbol}: {e}")
        return []

def calc_rsi(prices, period=14):
    """RSI по закрытиям"""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)

def calc_features(symbol: str, entry_price: float, entry_ts_ms: int):
    """
    Рассчитать 14 признаков Advisor на момент входа.
    """
    # Качаем OHLCV 1H: 60 свечей до входа + 2 после (для расчёта)
    ohlcv = fetch_ohlcv(symbol, entry_ts_ms - 60*60*1000*60, 70)
    if len(ohlcv) < 20:
        log.warning(f"  Мало OHLCV для {symbol}: {len(ohlcv)}")
        return None
    
    closes = [c[4] for c in ohlcv]
    highs = [c[2] for c in ohlcv]
    lows = [c[3] for c in ohlcv]
    volumes = [c[5] for c in ohlcv]
    
    # Находим свечу, соответствующую моменту входа
    entry_idx = None
    for i, c in enumerate(ohlcv):
        if c[0] >= entry_ts_ms:
            entry_idx = max(0, i - 1)  # берём предыдущую свечу
            break
    if entry_idx is None or entry_idx < 15:
        entry_idx = min(len(closes) - 15, 20) if len(closes) > 20 else len(closes) - 5
    
    # RSI на момент входа
    rsi = calc_rsi(closes[:entry_idx+1])
    
    # Trend: RSI-based
    if rsi > 60:
        trend_enc = 1.0  # bullish
    elif rsi < 40:
        trend_enc = 0.0  # bearish
    else:
        trend_enc = 0.5  # neutral
    
    # Volatility: std of returns за 20 свечей
    rets = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(max(1, entry_idx-19), entry_idx+1)]
    volatility = np.std(rets) if rets else 0.01
    
    # Volume ratio
    vol_window = volumes[max(0, entry_idx-10):entry_idx+1]
    avg_vol = np.mean(vol_window) if vol_window else 1.0
    current_vol = volumes[entry_idx] if entry_idx < len(volumes) else 1.0
    volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
    
    # Multi TF: RSI на 4H (эмулируем через массив closes)
    rsi_4h = calc_rsi(closes[:entry_idx+1:4])
    if rsi_4h > 60:
        mtf = 1.0
    elif rsi_4h < 40:
        mtf = 0.0
    else:
        mtf = 0.5
    
    # VWAP distance
    # VWAP = sum(price * volume) / sum(volume)
    vwap = sum(closes[i] * volumes[i] for i in range(max(0, entry_idx-12), entry_idx+1)) / max(1, sum(volumes[max(0, entry_idx-12):entry_idx+1]))
    vwap_dist = (entry_price - vwap) / vwap if vwap > 0 else 0.0
    
    # Candle patterns
    if entry_idx > 0 and entry_idx < len(ohlcv):
        open_p = ohlcv[entry_idx][1]
        high_p = ohlcv[entry_idx][2]
        low_p = ohlcv[entry_idx][3]
        close_p = ohlcv[entry_idx][4]
        body = abs(close_p - open_p)
        wick_u = high_p - max(open_p, close_p)
        wick_l = min(open_p, close_p) - low_p
        total_range = high_p - low_p if high_p > low_p else 0.001
        
        # Doji
        candle_doji = 1.0 if body / total_range < 0.1 else 0.0
        # Hammer
        candle_hammer = 1.0 if wick_l > body * 2 and wick_u < body * 0.5 else 0.0
        # Engulfing
        if entry_idx > 0:
            prev_open = ohlcv[entry_idx-1][1]
            prev_close = ohlcv[entry_idx-1][4]
            prev_body = abs(prev_close - prev_open)
            candle_engulfing = 1.0 if body > prev_body * 1.5 else 0.0
        else:
            candle_engulfing = 0.0
    else:
        candle_doji = 0.0
        candle_hammer = 0.0
        candle_engulfing = 0.0
    
    # BTC change 1h — используем placeholder
    btc_change_1h = 0.0
    
    # Hour of day: sin-coded
    entry_dt = datetime.fromtimestamp(entry_ts_ms / 1000, tz=timezone.utc)
    hour = entry_dt.hour + entry_dt.minute / 60.0
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    
    # Volume momentum
    prev_vol = volumes[entry_idx-1] if entry_idx > 0 else current_vol
    volume_momentum = current_vol / prev_vol if prev_vol > 0 else 1.0
    
    # HL range
    hl_range = (max(closes[max(0, entry_idx-9):entry_idx+1]) - min(closes[max(0, entry_idx-9):entry_idx+1])) / entry_price if entry_price > 0 else 0.01
    
    # Price above SMA20
    sma20 = np.mean(closes[max(0, entry_idx-19):entry_idx+1]) if entry_idx >= 19 else np.mean(closes[:entry_idx+1])
    price_above_sma20 = 1.0 if entry_price > sma20 else 0.0
    
    features = [
        rsi,                        # 0: rsi
        trend_enc,                  # 1: trend
        volatility,                  # 2: volatility
        volume_ratio,                # 3: volume_ratio
        mtf,                         # 4: multi_tf
        vwap_dist,                   # 5: vwap_dist
        candle_doji,                 # 6: candle_doji
        candle_hammer,               # 7: candle_hammer
        candle_engulfing,            # 8: candle_engulfing
        btc_change_1h,               # 9: btc_change_1h
        hour_sin,                    # 10: hour_of_day
        volume_momentum,             # 11: volume_momentum
        hl_range,                    # 12: hl_range
        price_above_sma20,           # 13: price_above_sma20
    ]
    
    return features


def restore(max_trades=500, force=False):
    """Восстановление training_data из истории"""
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Загружаем существующие
    existing = []
    if os.path.exists(TRAINING_DATA_PATH) and not force:
        with open(TRAINING_DATA_PATH, 'r') as f:
            existing = json.load(f)
        log.info(f"Существующие данные: {len(existing)} записей")
    
    # Берём sell-трейды
    sells = conn.execute(
        'SELECT * FROM trade_history WHERE side=? ORDER BY timestamp DESC LIMIT ?',
        ('sell', max_trades)
    ).fetchall()
    log.info(f"Обрабатываем {len(sells)} sell-трейдов...")
    
    # Для каждого sell находим buy
    processed = 0
    added = 0
    for sell in sells:
        sym = sell['symbol']
        entry_ts = sell['timestamp']
        
        # Находим последний buy перед sell для этого символа
        buy = conn.execute(
            'SELECT * FROM trade_history WHERE symbol=? AND side=? AND timestamp < ? ORDER BY timestamp DESC LIMIT 1',
            (sym, 'buy', entry_ts)
        ).fetchone()
        
        if buy is None:
            continue
        
        # Пропускаем если уже есть
        if not force:
            already = any(
                e.get('symbol') == sym and e.get('entry_price') == buy['price'] and abs(e.get('pnl', 0) - sell['pnl_pct']) < 0.1
                for e in existing
            )
            if already:
                continue
        
        # Рассчитываем фичи на момент входа
        entry_price = buy['price']
        entry_ms = int(datetime.fromisoformat(buy['timestamp'].replace('Z', '+00:00')).timestamp() * 1000)
        
        features = calc_features(sym, entry_price, entry_ms)
        if features is None:
            continue
        
        # PnL — метка good (1) / bad (0)
        label = 1 if sell['pnl_pct'] > 0 else 0
        
        record = {
            'features': features,
            'label': label,
            'symbol': sym,
            'pnl': sell['pnl_pct'],
            'entry_price': buy['price'],
            'exit_price': sell['price'],
            'reason': 'restored',
            'time': datetime.now().isoformat()
        }
        existing.append(record)
        added += 1
        
        processed += 1
        if processed % 20 == 0:
            log.info(f"  Прогресс: {processed}/{len(sells)} (добавлено: {added})")
        
        # Лимит на API
        time.sleep(0.3)
    
    conn.close()
    
    # Сохраняем
    with open(TRAINING_DATA_PATH, 'w') as f:
        json.dump(existing, f, default=str)
    
    log.info(f"✅ Готово: {len(existing)} записей (добавлено: {added})")
    return existing


if __name__ == '__main__':
    max_n = 200
    force = '--force' in sys.argv
    for a in sys.argv:
        if a.startswith('--max='):
            max_n = int(a.split('=')[1])
    
    log.info(f"🚀 Восстановление training_data (max={max_n}, force={force})...")
    data = restore(max_n, force)
    
    # Итог
    goods = sum(1 for d in data if d['label'] == 1)
    bads = sum(1 for d in data if d['label'] == 0)
    log.info(f"📊 Итог: good={goods}, bad={bads}, всего={len(data)}")
    
    # Пробуем обучить
    log.info("🧠 Пробуем обучить Advisor...")
    from ml_advisor import ml_train
    ml_train(force=True)
    log.info("✅ Обучение завершено")
