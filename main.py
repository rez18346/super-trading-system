#!/usr/bin/env python3
"""
main.py — Единая точка входа в супер-систему.

Архитектура:
  - Один процесс, два потока (trader + monitor)
  - Единая SQLite база данных (db.py)
  - Никаких JSON трекеров, никаких race condition
  
Потоки:
  trader  — анализ рынка, принятие решений о покупке/продаже
  monitor — стоп-лоссы, трейлинг-стопы, ML-выход, защита

Журнал: /tmp/system_v4.log
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

import json
import time
import logging
import signal
import threading
from datetime import datetime, timezone
from typing import Optional

# ─── Настройка логирования ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/system_v4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('main')

# ─── PID-файл: защита от двойного запуска ────────────────────────────────────
PID_FILE = os.path.join(BASE_DIR, "data", "trader.pid")
_log = logging.getLogger('pid')


def _check_pid_file():
    import signal as sig
    current_pid = os.getpid()
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE) as f:
                old_pid = int(f.read().strip())
            if old_pid != current_pid:
                try:
                    os.kill(old_pid, 0)
                    _log.warning(f"⚠️ Найден старый процесс PID={old_pid}, убиваю...")
                    os.kill(old_pid, sig.SIGTERM)
                    time.sleep(1)
                except OSError:
                    pass
        except (ValueError, IOError):
            pass
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
    with open(PID_FILE, 'w') as f:
        f.write(str(current_pid))
    _log.info(f"📝 PID-файл: {current_pid}")


_check_pid_file()

# ─── Импорты ──────────────────────────────────────────────────────────────────
import ccxt
import db
from industrial_trader import IndustrialTrader
from decision_engine import DecisionEngine, SignalType
from error_handler import ErrorHandler, RetryConfig
from data_cache import (PriceCache, OHLCVCache, CachedDataFetcher,
                         get_fetcher,
                         patch_exchange_fetch_ticker, patch_exchange_create_order)
from ws_client import BybitWebSocketClient, set_global_client


class TradingSystem:
    """
    Единая торговая система.
    Запускает трейдера как поток, управляет синхронизацией.
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path) as f:
            self.config = json.load(f)
        
        # Инициализация БД (создаёт таблицы если нет)
        db.init_db()
        
        # Биржа
        self.exchange = None
        self._setup_exchange()
        
        # Кеш данных (инициализируем до трейдера/монитора, делаем глобально доступным)
        self.price_cache = PriceCache()
        self.ohlcv_cache = OHLCVCache()
        self.data_fetcher = CachedDataFetcher(self.exchange, self.price_cache, self.ohlcv_cache)
        
        # Сохраняем как глобальный объект для доступа из трейдера/монитора
        import data_cache as dc_module
        dc_module._global_fetcher = self.data_fetcher
        
        # Monkey-patch: exchange.fetch_ticker() → кеш
        patch_exchange_fetch_ticker(self.exchange)
        
        # Monkey-patch: exchange.create_order() → DecisionEngine
        # Все buy-ордера проверяются DecisionEngine перед исполнением.
        patch_exchange_create_order(self.exchange)
        
        logger.info(f"📦 Кеш данных: ticker_throttle=3s, ohlcv_throttle=60s | fetch_ticker → кеш")
        
        # Поток трейдера
        self.trader_thread: Optional[threading.Thread] = None
        
        # Флаг работы
        self.running = True
        
        logger.info("=" * 50)
        logger.info("🏭 СУПЕР-СИСТЕМА V5 (единая архитектура)")
        logger.info(f"   БД: {db.get_db_path()}")
        logger.info("=" * 50)
        
        # WebSocket-клиент (public данные: цены и свечи)
        # Запускаем ДО синхронизации — не зависит от биржи
        self.ws_client = None
        self._start_websocket()
        
        # Первичная синхронизация
        self._initial_sync()
    
    def _setup_exchange(self):
        """Настройка подключения к бирже (bybit spot)."""
        try:
            bybit_config = self.config['bybit']
            self.exchange = ccxt.bybit({
                'apiKey': bybit_config['api_key'],
                'secret': bybit_config['secret'],
                'password': bybit_config.get('password', ''),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                },
            })
            
            # Проверка соединения
            self.exchange.load_markets()
            logger.info("🔗 Подключение к Bybit установлено")
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к Bybit: {e}")
            raise
    
    def _start_websocket(self):
        """Запустить WebSocket-клиент для цен и свечей."""
        try:
            # Получаем список символов из конфига
            enabled_pairs = self.config.get('trading', {}).get('enabled_pairs', [])
            # Конвертируем SOL/USDT → SOLUSDT (формат Bybit)
            symbols = [p.replace('/', '') for p in enabled_pairs]
            
            if not symbols:
                logger.warning("Нет символов для WebSocket подписки")
                return
            
            # Используем существующий кеш (передаём явно, чтобы был тот же синглтон)
            self.ws_client = BybitWebSocketClient(
                symbols=symbols,
                price_cache=self.price_cache,
                ohlcv_cache=self.ohlcv_cache,
            )
            self.ws_client.start()
            # Сохраняем глобально для доступа из других модулей
            from ws_client import set_global_client
            set_global_client(self.ws_client)
            logger.info(f"🌐 WebSocket запущен: {len(symbols)} символов")
        except Exception as e:
            logger.warning(f"⚠️ WebSocket не запущен (будет REST fallback): {e}")
            self.ws_client = None

    def _initial_sync(self):
        """Первичная синхронизация БД с биржей при старте."""
        try:
            enabled_pairs = self.config.get('trading', {}).get('enabled_pairs', [])
            changes = db.sync_positions_from_exchange(self.exchange, enabled_pairs)
            db.sync_orders_from_exchange(self.exchange)
            pos_count = len(db.get_all_positions())
            logger.info(f"🔄 Начальная синхронизация: {pos_count} позиций, {changes} изменений")
            
        except Exception as e:
            logger.warning(f"[INIT] Ошибка синхронизации: {e}")
    
    def run(self):
        """Основной цикл — запуск и поддержка потоков."""
        logger.info("🚀 Запуск системы...")
        
        # Запуск трейдера в отдельном потоке
        self.trader_thread = threading.Thread(
            target=self._run_trader,
            name='trader',
            daemon=True
        )
        self.trader_thread.start()
        
        logger.info("✅ Трейдер запущен")
        
        # Основной поток: синхронизация БД каждые 2 минуты
        sync_count = 0
        while self.running:
            try:
                time.sleep(30)
                sync_count += 1
                
                # Синхронизация с биржей каждую минуту
                enabled_pairs = self.config.get('trading', {}).get('enabled_pairs', [])
                db.sync_positions_from_exchange(self.exchange, enabled_pairs)
                
                # Полная синхронизация ордеров раз в 5 минут
                if sync_count % 10 == 0:
                    db.sync_orders_from_exchange(self.exchange)
                
                # Статус WebSocket раз в 10 циклов
                if sync_count % 10 == 0 and self.ws_client:
                    try:
                        ws_state = self.ws_client.get_state()
                        logger.info(f"🌐 WebSocket: {ws_state['state']}, msg={ws_state['messages_received']}, rc={ws_state['reconnects']}")
                    except Exception:
                        pass
                
                # Статус раз в 10 циклов (5 минут)
                if sync_count % 10 == 0:
                    self._log_status()
                
                # Проверка потока трейдера
                if not self.trader_thread.is_alive():
                    logger.warning("⚠️ Поток трейдера умер, перезапуск...")
                    self.trader_thread = threading.Thread(
                        target=self._run_trader, name='trader', daemon=True
                    )
                    self.trader_thread.start()
                    
            except Exception as e:
                logger.error(f"[MAIN] Ошибка цикла: {e}")
    
    def _run_trader(self):
        """Запуск трейдера в потоке."""
        try:
            trader = IndustrialTrader(self.config_path)
            trader.running = True
            
            while self.running:
                try:
                    trader.trading_cycle()
                except Exception as e:
                    logger.error(f"[TRADER] Ошибка цикла: {e}")
                    time.sleep(10)
                    
        except Exception as e:
            logger.error(f"[TRADER] Фатальная ошибка: {e}")
    

    
    def _log_status(self):
        """Логирование статуса системы."""
        try:
            stats = db.get_db_stats()
            pnl = db.get_pnl_stats()
            positions = db.get_all_positions()
            
            balance = self.exchange.fetch_balance()
            usdt = balance['total'].get('USDT', 0)
            total = usdt
            
            pos_lines = []
            for sym, pos in positions.items():
                try:
                    ticker = self.exchange.fetch_ticker(sym)
                    val = pos['quantity'] * ticker['last']
                    total += val
                    pnl_pct = (ticker['last'] - pos['entry_price']) / pos['entry_price'] * 100
                    pos_lines.append(f"{sym}: ${val:.2f} @ ${ticker['last']:.4f} ({pnl_pct:+.2f}%)")
                except:
                    pos_lines.append(f"{sym}: ${pos['quantity'] * pos['entry_price']:.2f}")
            
            logger.info("=" * 45)
            logger.info(f"📊 СТАТУС | Капитал: ${total:.2f} | USDT: ${usdt:.2f} | {len(positions)}/5")
            for p in pos_lines:
                logger.info(f"   {p}")
            logger.info(f"   PnL: ${pnl['total_pnl']:.2f} | Сделок: {pnl['total_trades']} | WR: {pnl['win_rate']}%")
            logger.info("=" * 45)
            
        except Exception as e:
            logger.debug(f"[STATUS] Ошибка: {e}")


def main():
    import signal as sig
    
    config_path = os.path.join(BASE_DIR, "config/api_config_final.json")
    
    # Создаём систему
    system = TradingSystem(config_path)
    
    # Обработка сигналов
    def _handle_signal(signum, frame):
        logger.info(f"🛑 Получен сигнал {signum}, завершение...")
        system.running = False
        # Удаляем PID-файл
        try:
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
        except:
            pass
        sys.exit(0)
    
    sig.signal(sig.SIGTERM, _handle_signal)
    sig.signal(sig.SIGINT, _handle_signal)
    
    # Запуск
    try:
        system.run()
    except KeyboardInterrupt:
        _handle_signal(sig.SIGINT, None)
    except Exception as e:
        logger.error(f"❌ Фатальная ошибка: {e}")
        raise


if __name__ == "__main__":
    main()
