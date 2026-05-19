#!/usr/bin/env python3
"""
🧹 Очистка мёртвых позиций: ALGO, OP, FIL — держатся >36ч без профита.
Запускается однократно после рестарта трейдера.
"""

import sys, os, json, time, logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('cleanup')

import ccxt
import db

# Загружаем конфиг
with open('config/config.json') as f:
    config = json.load(f)

# Подключаемся к бирже
exchange = ccxt.bybit({
    'apiKey': config['bybit']['api_key'],
    'secret': config['bybit']['secret'],
    'password': config['bybit']['password'],
    'enableRateLimit': config['bybit']['enableRateLimit'],
    'options': {'defaultType': config['bybit']['default_type']}
})

STALE_SYMBOLS = ['ALGO/USDT', 'OP/USDT', 'FIL/USDT']

for symbol in STALE_SYMBOLS:
    try:
        # Проверяем баланс на бирже
        currency = symbol.split('/')[0]
        balance = exchange.fetch_balance()
        total = balance['total'].get(currency, 0)
        free = balance['free'].get(currency, 0)
        
        if total < 0.000001:
            logger.info(f"⏭️ {symbol}: нет на бирже. Чищу БД.")
            db.remove_position(symbol)
            continue
        
        ticker = exchange.fetch_ticker(symbol)
        price = ticker['last']
        value = total * price
        
        if value < 1.0:
            logger.info(f"⏭️ {symbol}: остаток ${value:.2f} < $1. Чищу БД.")
            db.remove_position(symbol)
            continue
        
        # Продаём
        logger.info(f"💰 {symbol}: {total:.6f} @ ${price:.4f} = ${value:.2f}. Продаю...")
        
        # Используем лимитный ордер по рынку
        order = exchange.create_market_sell_order(symbol, free * 0.999)
        logger.info(f"✅ {symbol}: ПРОДАНО! Ордер: {order['id']}")
        
        # Записываем в БД
        db.add_trade(symbol, 'sell', price, free * 0.999)
        db.remove_position(symbol)
        
    except Exception as e:
        logger.error(f"❌ {symbol}: {e}")

logger.info("🧹 Очистка завершена!")
