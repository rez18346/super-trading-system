#!/usr/bin/env python3
"""DOT SHORT monitor: SL/TP для ручного шорта."""
import os, sys, time, json

api_key = secret = password = ''
with open(os.path.join(os.path.dirname(__file__), '.env')) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            if '=' in line:
                k, v = line.split('=', 1)
                if k == 'BYBIT_API_KEY': api_key = v
                elif k == 'BYBIT_SECRET': secret = v
                elif k == 'BYBIT_PASSWORD': password = v

import ccxt
exchange = ccxt.bybit({
    'apiKey': api_key, 'secret': secret, 'password': password,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot', 'enableUnifiedAccount': True},
})

ENTRY = 1.093; QTY = 9.9
SL_PRICE = round(ENTRY * 1.035, 4)   # $1.1313
TP_PRICE = round(ENTRY * 0.97, 4)    # $1.0602

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log(f"DOT SHORT Monitor | Entry=${ENTRY} | Qty={QTY}")
log(f"SL: ${SL_PRICE} | TP: ${TP_PRICE}")

last_price = 0
while True:
    try:
        ticker = exchange.fetch_ticker('DOT/USDT')
        price = ticker['last']

        if price != last_price:
            pnl_pct = (ENTRY - price) / ENTRY * 100
            pnl_usd = QTY * (ENTRY - price)
            log(f"DOT=${price:.4f} PnL={pnl_pct:+.2f}%(${pnl_usd:+.2f}) "
                f"SL_dist={((price-SL_PRICE)/SL_PRICE*100):+.1f}%  "
                f"TP_dist={((TP_PRICE-price)/TP_PRICE*100):+.1f}%")
            last_price = price

        if price >= SL_PRICE:
            log(f"🔴 SL! Закрываю шорт...")
            o = exchange.create_order('DOT/USDT','market','buy', QTY, params={'category':'spot','isLeverage':1})
            log(f"✅ Закрыто: {o['id']}")
            break

        if price <= TP_PRICE:
            log(f"🟢 TP! Закрываю шорт...")
            o = exchange.create_order('DOT/USDT','market','buy', QTY, params={'category':'spot','isLeverage':1})
            log(f"✅ Закрыто: {o['id']}")
            break

        time.sleep(10)
    except KeyboardInterrupt:
        log("Остановлен")
        break
    except Exception as e:
        log(f"Ошибка: {e}")
        time.sleep(10)

log("Monitor завершён")
