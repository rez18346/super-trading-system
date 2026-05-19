#!/usr/bin/env python3
"""
ЗАПУСК СУПЕР-СИСТЕМЫ V4
- Основная система с патчем комиссий
- Независимый монитор стоп-лоссов через JSON-трекер
- Автоматическая запись entry price в трекер при каждой сделке
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import logging
import threading
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# PID-файл: защита от двойного запуска
PID_FILE = os.path.join(BASE_DIR, 'data', 'trader.pid')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_log = logging.getLogger('pid')
def _check_pid_file():
    """Проверяет PID-файл, убивает дубля если он есть"""
    import signal
    current_pid = os.getpid()
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE) as f:
                old_pid = int(f.read().strip())
            if old_pid != current_pid:
                try:
                    os.kill(old_pid, 0)
                    _log.warning(f"⚠️ Найден старый процесс PID={old_pid}, убиваю...")
                    os.kill(old_pid, signal.SIGTERM)
                    time.sleep(1)
                except OSError:
                    pass
        except (ValueError, IOError):
            pass
    with open(PID_FILE, 'w') as f:
        f.write(str(current_pid))
    _log.info(f"📝 PID-файл: {current_pid}")

_check_pid_file()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import industrial_trader
from stop_loss_monitor_v5 import StopLossMonitor, save_trade_to_tracker

# ─── ПАТЧ EXECUTE_TRADE ───
original_execute_trade = industrial_trader.IndustrialTrader.execute_trade

def patched_execute_trade(self, symbol, side, quantity, price):
    """Патч: комиссии + трекер позиций"""
    COMMISSION_RATE = 0.001
    pnl_before = None
    
    if side == 'buy':
        quantity = quantity * (1 - COMMISSION_RATE)
    elif side == 'sell':
        # Сохраняем PnL перед продажей для трекера
        if symbol in self.positions:
            entry = self.positions[symbol]['entry_price']
            pnl_before = (price - entry) / entry * 100
        
        quantity = quantity * (1 - COMMISSION_RATE)
        logger.info(f"🔧 Коррекция продажи: {original_qty:.6f} -> {quantity:.6f} (комиссия {COMMISSION_RATE*100}%)")
        
        try:
            balance = self.exchange.fetch_balance()
            currency = symbol.split('/')[0]
            real_qty = balance['total'].get(currency, 0)
            
            if real_qty > 0 and quantity > real_qty:
                safe_qty = real_qty * 0.999
                logger.warning(f"⚠️  Доп. коррекция: реальное {real_qty:.6f} < {quantity:.6f} → {safe_qty:.6f}")
                quantity = safe_qty
        except Exception as e:
            logger.warning(f"Не удалось проверить реальное количество: {e}")
    
    # 👇 Сохраняем оригинальное quantity для коррекции sell (патч выше использует original_qty)
    # Выполняем оригинальную функцию
    result = original_execute_trade(self, symbol, side, quantity, price)
    
    # Запись в трекер после успешной сделки
    if result:
        save_trade_to_tracker(symbol, side, quantity, price, pnl_before)
    
    return result

def patched_execute_trade_safe(self, symbol, side, quantity, price):
    """Безопасная версия с сохранением original_qty"""
    COMMISSION_RATE = 0.001
    original_qty = quantity
    pnl_before = None
    
    if side == 'buy':
        quantity = quantity * (1 - COMMISSION_RATE)
    elif side == 'sell':
        if symbol in self.positions:
            entry = self.positions[symbol]['entry_price']
            pnl_before = (price - entry) / entry * 100
        
        quantity = quantity * (1 - COMMISSION_RATE)
        logger.info(f"🔧 Коррекция продажи: {original_qty:.6f} -> {quantity:.6f} (комиссия {COMMISSION_RATE*100}%)")
        
        try:
            balance = self.exchange.fetch_balance()
            currency = symbol.split('/')[0]
            real_qty = balance['total'].get(currency, 0)
            if real_qty > 0 and quantity > real_qty:
                safe_qty = real_qty * 0.999
                logger.warning(f"⚠️  Доп. коррекция: реальное {real_qty:.6f} < {quantity:.6f} → {safe_qty:.6f}")
                quantity = safe_qty
        except Exception as e:
            logger.warning(f"Не удалось проверить реальное количество: {e}")
    
    result = original_execute_trade(self, symbol, side, quantity, price)
    
    if result:
        save_trade_to_tracker(symbol, side, quantity, price, pnl_before)
    
    return result

# Применяем патч
industrial_trader.IndustrialTrader.execute_trade = patched_execute_trade_safe


def main():
    config_path = os.path.join(BASE_DIR, "config/api_config_final.json")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        return
    
    logger.info("=" * 60)
    logger.info("🚀 ЗАПУСК СУПЕР-СИСТЕМЫ V4")
    logger.info("   ✓ Патч комиссий (0.1%)")
    logger.info("   ✓ JSON-трекер позиций")
    logger.info("   ✓ Независимый монитор стоп-лоссов (5 сек)")
    logger.info("=" * 60)
    
    # 1. Синхронизируем трекер с текущими позициями
    logger.info("\n🔄 СИНХРОНИЗАЦИЯ ТРЕКЕРА С БИРЖЕЙ...")
    try:
        import ccxt
        exchange = ccxt.bybit({
            'apiKey': config['bybit']['api_key'],
            'secret': config['bybit']['secret'],
            'password': config['bybit']['password'],
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        
        # Загружаем текущий трекер
        tracker_path = os.path.join(BASE_DIR, "data/positions_tracker.json")
        os.makedirs(os.path.dirname(tracker_path), exist_ok=True)
        
        tracker = {}
        if os.path.exists(tracker_path):
            try:
                with open(tracker_path, 'r') as f:
                    tracker = json.load(f)
            except:
                pass
        
        balance = exchange.fetch_balance()
        synced_count = 0
        
        for asset, amount in balance['total'].items():
            if asset != 'USDT' and amount > 0.000001:
                symbol = f"{asset}/USDT"
                if symbol in tracker:
                    # Уже есть в трекере
                    continue
                
                # Пытаемся найти entry price
                entry_price_avg = None
                try:
                    orders = exchange.fetch_orders(symbol, limit=20)
                    buy_orders = [o for o in orders if o['side'] == 'buy' and o['status'] == 'closed' and o.get('filled', 0) > 0]
                    if buy_orders:
                        total_cost = sum(o.get('cost', 0) or 0 for o in buy_orders)
                        total_filled = sum(o.get('filled', 0) or 0 for o in buy_orders)
                        if total_filled > 0:
                            entry_price_avg = total_cost / total_filled
                except:
                    pass
                
                if entry_price_avg:
                    tracker[symbol] = {
                        'entry_price': entry_price_avg,
                        'quantity': amount,
                        'entry_time': datetime.now(timezone.utc).isoformat(),
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    }
                    synced_count += 1
                    logger.info(f"   ✅ Синхронизирован: {symbol} @ ${entry_price_avg:.4f} ({amount:.6f})")
        
        # Сохраняем трекер
        with open(tracker_path, 'w') as f:
            json.dump(tracker, f, indent=2, default=str)
        
        logger.info(f"   📝 Трекер синхронизирован: {len(tracker)} позиций (+{synced_count} новых)")
        
    except Exception as e:
        logger.warning(f"⚠️  Ошибка синхронизации трекера: {e}")
    
    # 2. ЗАПУСК ОСНОВНОЙ СИСТЕМЫ
    logger.info("\n📦 ЗАПУСК ОСНОВНОЙ ТОРГОВОЙ СИСТЕМЫ...")
    trader = industrial_trader.IndustrialTrader(config_path)
    
    logger.info(f"✅ Основная система инициализирована")
    logger.info(f"   Лимит позиций: {config['risk_management'].get('max_open_positions', 5)}")
    logger.info(f"   Размер позиции: ${config['risk_management'].get('max_buy_order_usd', 10.0)}")
    logger.info(f"   Порог уверенности: {config['risk_management'].get('min_confidence_percent', 38)}%")
    logger.info(f"   Стоп-лосс: -{config['risk_management']['stop_loss_percent']}%")
    logger.info(f"   Тейк-профит: +{config['risk_management']['take_profit_percent']}%")
    
    trader.start()
    
    # 3. ЗАПУСК МОНИТОРА СТОП-ЛОССОВ
    logger.info("\n🛡️ ЗАПУСК НЕЗАВИСИМОГО МОНИТОРА СТОП-ЛОССОВ...")
    monitor = StopLossMonitor(config_path)
    
    # Запускаем в фоновом потоке
    monitor_thread = threading.Thread(target=monitor.run, daemon=True)
    monitor_thread.start()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ СИСТЕМА V4 ЗАПУЩЕНА ПОЛНОСТЬЮ:")
    logger.info("   1. Основная торговая система")
    logger.info("   2. JSON-трекер позиций (data/positions_tracker.json)")
    logger.info("   3. Монитор стоп-лоссов (проверка каждые 5 сек)")
    logger.info("   🛡️ Любая позиция будет закрыта при -5% или +10%")
    logger.info("=" * 60)
    
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("🛑 Остановка системы")
        trader.running = False
        monitor.running = False


if __name__ == "__main__":
    main()
