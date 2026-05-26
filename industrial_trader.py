#!/usr/bin/env python3
"""
ПРОМЫШЛЕННАЯ ТОРГОВАЯ СИСТЕМА - Industrial Super System
Версия: 1.0.0
Автор: Капитан (Главный координатор системы)
"""

import json
import time
import logging
import os
from datetime import datetime, timezone
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
import signal
import sys
import db_pg as db  # PostgreSQL

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/industrial_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IndustrialTrader:
    """Промышленная торговая система с управлением рисками"""
    
    def __init__(self, config_path: str):
        """Инициализация системы с конфигурацией"""
        self.load_config(config_path)
        self.setup_exchange()
        self.running = False
        self.positions = {}
        self._lock = threading.RLock()
        self.trades_history = []
        self.capital = self.config['trading']['capital']
        self.available_capital = self.capital
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self._current_market_mode = 'neutral'
        
        # Статистика
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # СИНХРОНИЗАЦИЯ С РЕАЛЬНЫМИ ПОЗИЦИЯМИ НА БИРЖЕ
        self.sync_with_exchange_positions()
        
        logger.info(f"Industrial Trader инициализирован с капиталом: ${self.capital}")
    
    def sync_with_exchange_positions(self):
        """Синхронизация с реальными позициями на бирже"""
        try:
            logger.info("🔄 Синхронизация с реальными позициями на бирже...")
            
            # Получаем баланс с биржи
            balance = self.exchange.fetch_balance()
            
            # Рассчитываем реальный доступный капитал
            total_usdt = balance['total'].get('USDT', 0)
            free_usdt = balance['free'].get('USDT', 0)
            
            # Обновляем доступный капитал
            self.available_capital = free_usdt
            logger.info(f"💰 Реальный доступный капитал: ${free_usdt:.2f} (всего USDT: ${total_usdt:.2f})")
            

            
            # Находим открытые позиции (активы кроме USDT)
            open_positions_count = 0
            
            # Загружаем все цены одним запросом (предотвращает rate limit)
            try:
                all_tickers = self.exchange.fetch_tickers()
            except Exception as e:
                logger.warning(f"⚠️ Не удалось загрузить все тикеры: {e}")
                all_tickers = {}
            
            for currency, amount in balance['total'].items():
                if currency != 'USDT' and amount > 0.000001:  # Минимальный порог
                    try:
                        symbol_key = f"{currency}/USDT"
                        # Ищем тикер — сначала spot, потом все
                        ticker = all_tickers.get(symbol_key, {})
                        current_price = ticker.get('last') or 0
                        if current_price <= 0:
                            # fallback: индивидуальный запрос
                            try:
                                ticker = self.exchange.fetch_ticker(symbol_key)
                                current_price = ticker.get('last') or 0
                            except Exception:
                                current_price = 0
                        if current_price <= 0:
                            logger.warning(f"   ⚠️ Нет цены для {symbol_key} — пропускаем")
                            continue
                        value_usdt = amount * current_price
                        
                        if value_usdt > 1.0:  # Позиции больше $1
                            symbol = f"{currency}/USDT"
                            
                            # 🩹 Цена входа: сначала средневзвешенная с биржи (DCA-aware),
                            # fallback на последнюю цену из PG или текущую
                            avg_price, first_trade_time = self.get_average_buy_price(symbol, amount)
                            
                            if avg_price:
                                entry_price = avg_price
                                entry_time = first_trade_time
                                logger.info(f"   📊 Средняя цена покупки {symbol}: ${avg_price:.4f} (DCA weighted)")
                            else:
                                # Если нет истории на бирже — последняя цена из PG
                                trades = db.get_trade_history(symbol, limit=1, side='buy')
                                if trades and trades[0].get('entry_price', 0) > 0:
                                    entry_price = float(trades[0]['entry_price'])
                                    entry_time = trades[0]['entry_time']
                                    logger.info(f"   📊 Последняя цена покупки из PG: {symbol}: ${entry_price:.4f}")
                                else:
                                    entry_price = current_price
                                    entry_time = datetime.now().isoformat()
                                    logger.info(f"   ⚠️  Не удалось получить среднюю цену, используем текущую: ${current_price:.4f}")
                            
                            # При рестарте не пишем в trade_history — данные будут от реальных сделок
                    # (синхронизация с биржей обновит self.positions)
                            
                            # Добавляем позицию в self.positions
                            self.positions[symbol] = {
                                'quantity': amount,
                                'entry_price': entry_price,
                                'entry_time': entry_time,
                                'side': 'long'
                            }
                            
                            open_positions_count += 1
                            logger.info(f"   ✅ Позиция добавлена: {symbol} - {amount:.6f} @ ${entry_price:.4f} (текущая: ${current_price:.4f})")
                            
                            # Рассчитываем текущий P&L
                            if entry_price > 0:
                                pnl_percent = ((current_price - entry_price) / entry_price) * 100
                                logger.info(f"   📈 Текущий P&L: {pnl_percent:+.2f}%")
                                
                                # Проверяем, не пора ли продавать
                                risk_config = self.config['risk_management']
                                take_profit = risk_config['take_profit_percent']
                                
                                if pnl_percent >= take_profit:
                                    logger.warning(f"   🎯 ДОСТИГНУТ ТЕЙК-ПРОФИТ! {pnl_percent:.2f}% >= {take_profit}%")
                                    logger.warning(f"   💡 Рекомендация: продать позицию")
                            
                    except Exception:
                        logger.debug(f"Не удалось обработать {currency}")
            
            if open_positions_count > 0:
                logger.info(f"✅ Синхронизировано {open_positions_count} позиций с биржей")
            else:
                logger.info("ℹ️ Открытых позиций на бирже не найдено")
                
        except Exception as e:
            logger.error(f"❌ Ошибка синхронизации с биржей: {e}")
            logger.error("   Система будет работать без синхронизации позиций")
    
    def get_average_buy_price(self, symbol: str, current_amount: float):
        """Получение средней цены покупки из истории ордеров"""
        try:
            # Получаем историю закрытых ордеров за последние 48 часов (для DCA/усреднения)
            since = self.exchange.milliseconds() - (48 * 60 * 60 * 1000)
            
            # Bybit требует использовать fetch_closed_orders вместо fetch_orders
            orders = self.exchange.fetch_closed_orders(symbol, since=since)
            
            if not orders:
                return None, None
            
            # Фильтруем только покупки (buy orders)
            buy_orders = [o for o in orders if o['side'] == 'buy' and o['status'] == 'closed']
            
            if not buy_orders:
                return None, None
            
            # Сортируем по времени (от старых к новым)
            buy_orders.sort(key=lambda x: x['timestamp'])
            
            # Рассчитываем среднюю цену
            total_cost = 0.0
            total_amount = 0.0
            first_trade_time = None
            
            for order in buy_orders:
                if total_amount >= current_amount:
                    break  # Уже набрали нужное количество
                    
                order_amount = order['amount']
                order_price = order['price']
                order_cost = order_amount * order_price
                
                # Если этот ордер превышает нужное количество, берем только часть
                if total_amount + order_amount > current_amount:
                    needed_amount = current_amount - total_amount
                    order_cost = needed_amount * order_price
                    order_amount = needed_amount
                
                total_cost += order_cost
                total_amount += order_amount
                
                if first_trade_time is None:
                    first_trade_time = datetime.fromtimestamp(order['timestamp'] / 1000).isoformat()
            
            if total_amount > 0:
                avg_price = total_cost / total_amount
                return avg_price, first_trade_time
            else:
                return None, None
                
        except Exception as e:
            logger.warning(f"Не удалось получить среднюю цену для {symbol}: {e}")
            return None, None
    
    def load_config(self, config_path: str):
        """Загрузка конфигурации из JSON файла + переопределение из .env"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Конфигурация загружена из {config_path}")
            
            # 🔐 Переопределение API-ключей из .env (безопаснее, чем в JSON)
            env_path = os.path.join(os.path.dirname(__file__), '.env')
            if os.path.exists(env_path):
                try:
                    with open(env_path) as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('#') or '=' not in line:
                                continue
                            key, val = line.split('=', 1)
                            key, val = key.strip(), val.strip().strip("'\"")
                            if key == 'BYBIT_API_KEY' and val:
                                self.config['bybit']['api_key'] = val
                                logger.debug(f".env: BYBIT_API_KEY загружен")
                            elif key == 'BYBIT_SECRET' and val:
                                self.config['bybit']['secret'] = val
                                logger.debug(f".env: BYBIT_SECRET загружен")
                            elif key == 'BYBIT_PASSWORD' and val:
                                self.config['bybit']['password'] = val
                                logger.debug(f".env: BYBIT_PASSWORD загружен")
                    logger.info("🔐 API-ключи из .env применены")
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка чтения .env: {e}")
            else:
                logger.warning(f"⚠️ .env не найден ({env_path}) — используются ключи из JSON")
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            raise
    
    def setup_exchange(self):
        """Настройка подключения к бирже"""
        try:
            bybit_config = self.config['bybit']
            
            # Создаем экземпляр биржи
            self.exchange = ccxt.bybit({
                'apiKey': bybit_config['api_key'],
                'secret': bybit_config['secret'],
                'password': bybit_config['password'],
                'enableRateLimit': bybit_config['enableRateLimit'],
                'options': {
                    'defaultType': bybit_config['default_type']
                }
            })
            
            # Тестовый режим
            if bybit_config.get('paper_trading', True):
                self.exchange.set_sandbox_mode(True)
                logger.info("Режим бумажной торговли активирован")
            
            logger.info(f"Подключение к Bybit установлено")
            
        except Exception as e:
            logger.error(f"Ошибка настройки биржи: {e}")
            raise
    
    def get_market_data(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Получение рыночных данных"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Ошибка получения данных для {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет технических индикаторов"""
        if df.empty:
            return df
        
        # Простые скользящие средние
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Волатильность (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        return df
    
    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Анализ тренда"""
        if df.empty:
            return {'trend': 'neutral', 'strength': 0, 'confidence': 0}
        
        latest = df.iloc[-1]
        
        # Определение тренда по скользящим средним
        if latest['sma_20'] > latest['sma_50']:
            trend = 'bullish'
            strength = (latest['sma_20'] - latest['sma_50']) / latest['sma_50'] * 100
        elif latest['sma_20'] < latest['sma_50']:
            trend = 'bearish'
            strength = (latest['sma_50'] - latest['sma_20']) / latest['sma_20'] * 100
        else:
            trend = 'neutral'
            strength = 0
        
        # Уверенность на основе RSI и волатильности
        rsi_conf = 1.0 - abs(latest['rsi'] - 50) / 50 if not pd.isna(latest['rsi']) else 0.5
        volatility_conf = min(latest['atr'] / latest['close'] * 100, 1.0) if not pd.isna(latest['atr']) else 0.5
        confidence = (rsi_conf + volatility_conf) / 2
        
        return {
            'trend': trend,
            'strength': strength,
            'confidence': confidence,
            'rsi': latest['rsi'],
            'volatility': latest['atr'] / latest['close'] * 100 if not pd.isna(latest['atr']) else 0
        }
    
    def _notify_trade(self, symbol: str, side: str, price: float, quantity: float):
        """Записать алерт о сделке для дашборда и мониторинга"""
        try:
            alert = {
                'symbol': symbol,
                'side': side,
                'price': price,
                'quantity': quantity,
                'timestamp': time.time(),
                'ts_human': datetime.now().strftime('%H:%M:%S')
            }
            with open('/tmp/trade_alert.json', 'w') as f:
                json.dump(alert, f)
        except Exception as _e:
            logger.debug(f"[_notify_trade] alert save: {_e}")

    def check_risk_limits(self) -> bool:
        """Проверка лимитов риска"""
        risk_config = self.config['risk_management']
        
        # Проверка максимальной дневной потери
        max_loss_abs = (risk_config['max_daily_loss_percent'] / 100.0) * self.capital
        if self.daily_pnl <= -max_loss_abs:
            logger.warning(f"Достигнут лимит дневных потерь: {self.daily_pnl}%")
            return False
        
        # Проверка максимального количества сделок подряд
        if len(self.trades_history) >= 3:
            last_three = self.trades_history[-3:]
            losses = sum(1 for t in last_three if t['pnl'] < 0)
            if losses >= risk_config['max_consecutive_losses']:
                logger.warning(f"Слишком много убыточных сделок подряд: {losses}")
                return False
        
        # Проверка волатильности
        # (здесь можно добавить проверку текущей волатильности рынка)
        
        return True
    
    def calculate_position_size(self, symbol: str, price: float, score: float = 65.0) -> float:
        """Расчет размера позиции (зависит от Score)"""
        risk_config = self.config['risk_management']
        trading_config = self.config['trading']
        
        # База от капитала
        max_position_usd = self.available_capital * (trading_config['max_position_size_percent'] / 100)
        
        # Размер от Score: чем выше уверенность, тем больше вход
        if score >= 80:
            size_before_cap = 90.0
            tier = f"Score>=80"
        elif score >= 75:
            size_before_cap = 60.0
            tier = f"Score>=75"
        elif score >= 70:
            size_before_cap = 45.0
            tier = f"Score>=70"
        else:
            size_before_cap = 30.0
            tier = f"Score<70"
        
        # Ограничение от капитала и конфига
        max_order_limit = risk_config.get('max_buy_order_usd', 60.0)
        position_size = min(max_position_usd, max_order_limit, size_before_cap)
        
        # Минимальный размер
        min_position_usd = risk_config['min_position_usd']
        position_size = max(position_size, min_position_usd)
        
        quantity = position_size / price
        logger.info(f"Размер позиции для {symbol}: ${position_size:.2f} ({tier}, score={score:.0f}), {quantity:.6f} units")
        return quantity
    
    def execute_trade(self, symbol: str, side: str, quantity: float, price: float,
                      sl_price: Optional[float] = None, tp_price: Optional[float] = None,
                      trail_act: Optional[bool] = None, trail_dist: Optional[float] = None,
                      max_hold_h: Optional[int] = None) -> Optional[Dict]:
        """Выполнение торговой операции"""
        try:
            # В режиме бумажной торговли только симулируем
            if self.config['bybit']['paper_trading']:
                trade_id = f"paper_{int(time.time())}"
                timestamp = datetime.now().isoformat()
                
                trade = {
                    'id': trade_id,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'timestamp': timestamp,
                    'paper': True
                }
                
                # 🚀 Алерт о новой сделке — записываем для дашборда
                self._notify_trade(symbol, side, price, quantity)
                
                # Обновляем позиции
                if side == 'buy':
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'entry_price': price,
                        'entry_time': timestamp,
                        'side': 'long',
                        # SL/TP уровни от DecisionEngine (заполнит caller)
                        '_sl_price': sl_price if sl_price is not None else None,
                        '_tp_price': tp_price if tp_price is not None else None,
                        '_trail_act': trail_act if trail_act is not None else None,
                        '_trail_dist': trail_dist if trail_dist is not None else None,
                        '_max_hold_h': max_hold_h if max_hold_h is not None else None,
                        '_created_at': time.time(),  # 🛡️ Для защиты от sync-удаления
                    }
                    self.available_capital -= quantity * price
                    db.add_trade(symbol, 'buy', float(price), float(quantity), ts=timestamp)
                elif side == 'sell' and symbol in self.positions:
                    with self._lock:
                        position = self.positions.pop(symbol)
                    pnl = (price - position['entry_price']) * quantity
                    self.available_capital += quantity * price
                    self.daily_pnl += pnl
                    self.daily_trades += 1
                    
                    trade['pnl'] = pnl
                    trade['pnl_percent'] = (pnl / (position['entry_price'] * quantity)) * 100
                    db.add_trade(symbol, 'sell', float(price), float(quantity), pnl=float(pnl), pnl_pct=float(trade['pnl_percent']), ts=timestamp)
                
                self.trades_history.append(trade)
                logger.info(f"Бумажная сделка: {side} {quantity} {symbol} @ ${price}")
                return trade
            
            # Реальная торговля
            if not self.config['bybit']['paper_trading']:
                # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА БЕЗОПАСНОСТИ
                order_value = quantity * price
                
                # РАЗДЕЛЬНЫЕ ЛИМИТЫ ДЛЯ ПОКУПОК И ПРОДАЖ
                if side == 'buy':
                    max_order_value = self.config['risk_management'].get('max_buy_order_usd', 2.0)
                    limit_type = "покупки"
                else:
                    max_order_value = self.config['risk_management'].get('max_sell_order_usd')
                    limit_type = "продажи"
                
                # ИСКЛЮЧЕНИЕ: при продаже по тейк-профиту/стоп-лоссу продаем ВСЮ позицию
                is_exit_trade = False
                if side == 'sell' and symbol in self.positions:
                    position_quantity = self.positions[symbol]['quantity']
                    if abs(quantity - position_quantity) < 0.000001:  # Продаем всю позицию
                        is_exit_trade = True
                        logger.info(f"🎯 ВЫХОД ИЗ ПОЗИЦИИ: продажа всей позиции {symbol} ({position_quantity:.6f})")
                
                # Применяем лимит только если он задан и это не выход из позиции
                if max_order_value is not None and order_value > max_order_value and not is_exit_trade:
                    logger.warning(f"ПРЕВЫШЕН ЛИМИТ {limit_type}: {order_value} > {max_order_value}")
                    quantity = max_order_value / price
                    logger.info(f"СКОРРЕКТИРОВАННОЕ КОЛИЧЕСТВО: {quantity}")
                
                logger.info(f"🚨 ВНИМАНИЕ: ВЫПОЛНЯЕТСЯ РЕАЛЬНАЯ СДЕЛКА!")
                logger.info(f"   {side.upper()} {quantity:.6f} {symbol} @ ${price:.2f}")
                logger.info(f"   СТОИМОСТЬ: ${order_value:.2f}")
                
                # ПРОВЕРКА РЕАЛЬНОГО БАЛАНСА ПЕРЕД СДЕЛКОЙ
                try:
                    balance = self.exchange.fetch_balance()
                    free_usdt = balance['free'].get('USDT', 0)
                    
                    if side == 'buy' and free_usdt < order_value:
                        logger.error(f"❌ НЕДОСТАТОЧНО USDT: нужно ${order_value:.2f}, доступно ${free_usdt:.2f}")
                        logger.error(f"   СДЕЛКА ОТМЕНЕНА!")
                        return None
                        
                    if side == 'sell':
                        free_asset = balance['free'].get(symbol.split('/')[0], 0)
                        if free_asset < quantity:
                            logger.error(f"❌ НЕДОСТАТОЧНО {symbol.split('/')[0]}: нужно {quantity:.6f}, доступно {free_asset:.6f}")
                            logger.error(f"   СДЕЛКА ОТМЕНЕНА!")
                            return None
                    
                    logger.info(f"✅ Баланс проверен: USDT=${free_usdt:.2f}, нужно=${order_value:.2f}")
                except Exception as e:
                    logger.error(f"❌ Ошибка проверки баланса: {e}")
                    logger.error(f"   Продолжаем сделку без проверки баланса (риск!)")
                
                # Небольшая пауза для подтверждения
                time.sleep(2)
                
                order = self.exchange.create_order(
                    symbol=symbol,
                    type='market',
                    side=side,
                    amount=quantity
                )
                
                logger.info(f"✅ РЕАЛЬНАЯ СДЕЛКА ВЫПОЛНЕНА: ID {order['id']}")
                
                # ОБНОВЛЯЕМ ПОЗИЦИИ ПОСЛЕ УСПЕШНОЙ СДЕЛКИ
                if side == 'buy':
                    # 🩹 Конвертируем numpy → float для PG
                    order_price = order.get('average', order.get('price', price)) or price
                    filled_qty = order.get('filled', quantity) or quantity
                    if hasattr(order_price, 'item'):
                        order_price = order_price.item()
                    if hasattr(filled_qty, 'item'):
                        filled_qty = filled_qty.item()
                    
                    # Запись точной цены в trade_history
                    db.add_trade(
                        symbol=symbol,
                        side='buy',
                        price=float(order_price),
                        quantity=float(filled_qty),
                        order_id=str(order['id']),
                        exchange_id=str(order['id']),
                        ts=datetime.now().isoformat()
                    )
                    
                    # Берём реальное количество из order.filled или quantity
                    actual_qty = filled_qty if filled_qty > 0 else quantity
                    
                    # Добавляем новую позицию
                    # Используем order_price (цена BUY ордера), а не WA из БД
                    # Баг: DB.calculate_weighted_entry хранит старые цены, искажая PnL
                    with self._lock:
                        real_entry = order_price
                        if symbol not in self.positions:
                            self.positions[symbol] = {
                                'quantity': actual_qty,
                                'entry_price': real_entry,
                                'entry_time': datetime.now().isoformat(),
                                'max_profit': 0.0,
                                '_highest_price': real_entry,
                                '_created_at': time.time(),  # 🛡️ Для защиты от sync-удаления
                                '_sl_price': sl_price,
                                '_tp_price': tp_price,
                                '_trail_act': trail_act,
                                '_trail_dist': trail_dist,
                                '_max_hold_h': max_hold_h,
                            }
                        else:
                            self.positions[symbol]['quantity'] = actual_qty
                            self.positions[symbol]['entry_price'] = real_entry
                            self.positions[symbol]['entry_time'] = datetime.now().isoformat()
                            self.positions[symbol]['_created_at'] = time.time()
                            self.positions[symbol]['_sl_price'] = sl_price
                            self.positions[symbol]['_tp_price'] = tp_price
                            self.positions[symbol]['_trail_act'] = trail_act
                            self.positions[symbol]['_trail_dist'] = trail_dist
                            self.positions[symbol]['_max_hold_h'] = max_hold_h
                        # 💰 Обновляем доступный капитал сразу (предотвращает двойные BUY)
                        order_cost = actual_qty * real_entry
                        self.available_capital = max(0, self.available_capital - order_cost)
                    logger.info(f"📥 Обновлена позиция: {symbol} - {actual_qty:.6f} @ ${real_entry:.4f}, капитал: ${self.available_capital:.2f}")
                elif side == 'sell':
                    # Запись sell в trade_history
                    pnl = 0
                    pnl_pct = 0
                    if symbol in self.positions:
                        entry = self.positions[symbol]['entry_price']
                        filled = order.get('filled', quantity) or quantity
                        # 🩹 Конвертируем numpy → float для PG (psycopg2 не умеет в np.float64)
                        if hasattr(filled_price, 'item'):
                            filled_price = filled_price.item()
                        filled = float(filled)
                        
                        pnl = (filled_price - entry) * filled
                        pnl_pct = ((filled_price - entry) / entry) * 100 if entry > 0 else 0
                        
                        db.add_trade(
                            symbol=symbol,
                            side='sell',
                            price=float(filled_price),
                            quantity=float(filled),
                            pnl=float(pnl),
                            pnl_pct=float(pnl_pct),
                            order_id=str(order['id']),
                            exchange_id=str(order['id']),
                            ts=datetime.now().isoformat()
                        )
                    
                    # Удаляем позицию после продажи
                    if symbol in self.positions:
                        entry = self.positions[symbol]['entry_price']
                        qty = self.positions[symbol]['quantity']
                        cost = entry * qty
                        self.positions.pop(symbol, None)
                        db.remove_position(symbol)
                        # 💰 Возвращаем капитал (сколько потратили на вход + PnL)
                        filled = order.get('filled', quantity) or quantity
                        filled_price = order.get('average', order.get('price', price)) or price
                        profit = (filled_price - entry) * filled
                        self.available_capital += cost + profit
                        pnl_str = f", PnL=${pnl:.2f} ({pnl_pct:+.2f}%)" if pnl != 0 else ""
                        logger.info(f"📤 Позиция {symbol} продана{pnl_str}, капитал: ${self.available_capital:.2f}")
                
                return order
            else:
                return None
            
        except Exception as e:
            logger.error(f"Ошибка выполнения сделки: {e}")
            return None
    
    def clean_stale_orders(self):
        """Отменяет лимитные ордера, висящие дольше таймаута (настраивается в конфиге)."""
        try:
            stale_cfg = self.config.get('clean_stale_orders', {})
            sell_timeout = stale_cfg.get('sell_timeout', 300)
            buy_timeout = stale_cfg.get('buy_timeout', 180)
            orders = self.exchange.fetchOpenOrders()
            if orders:
                now = time.time()
                for o in orders:
                    age = now - (o['timestamp'] / 1000)
                    if o['side'] == 'sell':
                        if age > sell_timeout:
                            self.exchange.cancelOrder(o['id'], o['symbol'])
                            logger.info(f"🧹 Отменён sell-ордер ({age:.0f}с): {o['symbol']} {o['amount']} @ ${o['price']}")
                    else:
                        if age > buy_timeout:
                            self.exchange.cancelOrder(o['id'], o['symbol'])
                            logger.info(f"🧹 Отменён buy-ордер ({age:.0f}с): {o['symbol']} {o['amount']} @ ${o['price']}")
        except Exception as e:
            logger.warning(f"Ошибка очистки старых ордеров: {e}")

    def trading_cycle(self):
        """Основной торговый цикл"""
        logger.info("Запуск торгового цикла")
        
        trading_config = self.config['trading']
        symbols = trading_config['enabled_pairs']
        
        while self.running:
            try:
                cycle_start = time.time()
                # 🧹 Очистка старых лимитных ордеров
                self.clean_stale_orders()
                
                # Проверка лимитов риска
                if not self.check_risk_limits():
                    logger.warning("Лимиты риска превышены, пропускаем цикл")
                    time.sleep(60)
                    continue
                
                # Проверка максимального количества сделок в день
                if self.daily_trades >= self.config['system']['max_daily_trades']:
                    logger.info(f"Достигнут лимит дневных сделок: {self.daily_trades}")
                    time.sleep(300)
                    continue
                
                # 🔄 СИНХРОНИЗАЦИЯ С БИРЖЕЙ: сверяем self.positions с реальным балансом
                try:
                    balance = self.exchange.fetch_balance()
                    with self._lock:
                        for symbol in list(self.positions.keys()):
                            currency = symbol.split('/')[0]
                            real_qty = balance['total'].get(currency, 0)
                            mem_qty = self.positions[symbol]['quantity']
                            real_value = real_qty * (balance['free'].get(currency, 0) or 0)
                            
                            # 🛡️ ЗАЩИТА СВЕЖИХ ПОЗИЦИЙ: не синхронизируем первые 30 сек после входа
                            if '_created_at' in self.positions[symbol]:
                                age = time.time() - self.positions[symbol]['_created_at']
                                if age < 30:
                                    continue  # Пропускаем синхронизацию для свежих позиций
                            
                            # Если на бирже нет актива — удаляем из памяти
                            if real_qty < 0.000001:
                                logger.warning(f"🔄 Синхронизация: {symbol} нет на бирже. Удаляю из кеша/БД.")
                                self.positions.pop(symbol, None)
                                db.remove_position(symbol)
                            # Если осталась пыль (кол-во < 1% от исходного) — удаляем
                            if real_qty < mem_qty * 0.01 and real_qty < 0.1:
                                logger.warning(f"🔄 Синхронизация: {symbol} пыль ({real_qty:.6f} от {mem_qty:.4f}). Удаляю.")
                                self.positions.pop(symbol, None)
                                db.remove_position(symbol)
                            # Если актив есть, но сильно меньше закешированного — обновляем
                            elif real_qty < mem_qty * 0.5:
                                self.positions[symbol]['quantity'] = real_qty
                                logger.info(f"🔄 Синхронизация: {symbol} скорректирован с {mem_qty:.4f} до {real_qty:.4f}")
                except Exception as e:
                    logger.warning(f"Ошибка синхронизации: {e}")
                
                # 📊 Загрузка multi-timeframe данных для DecisionEngine
                # (1H, 4H, BTC цена — для ML-голосов: MLProfessionalV2, MLAdvisor, BTC-корреляция)
                # Используем CachedDataFetcher — данные уже кэшируются WebSocket-клиентом
                mtf_candles_1h = {}
                mtf_candles_4h = {}
                btc_price = None
                
                btc_5m_candles = None
                btc_1h_candles = None
                
                try:
                    from data_cache import get_fetcher, get_price
                    fetcher = get_fetcher()
                    if fetcher:
                        # BTC цена из кеша (или API с троттлингом 3с)
                        btc_price = fetcher.get_ticker('BTC/USDT')
                        
                        # BTC свечи для Regime Tracker
                        raw_btc_5m = fetcher.get_ohlcv('BTC/USDT', '5m', 50)
                        raw_btc_1h = fetcher.get_ohlcv('BTC/USDT', '1h', 48)
                        if raw_btc_5m:
                            import pandas as pd
                            btc_5m_candles = pd.DataFrame(
                                [[c[1],c[2],c[3],c[4],c[5]] for c in raw_btc_5m],
                                columns=['open','high','low','close','volume']
                            )
                        if raw_btc_1h:
                            import pandas as pd
                            btc_1h_candles = pd.DataFrame(
                                [[c[1],c[2],c[3],c[4],c[5]] for c in raw_btc_1h],
                                columns=['open','high','low','close','volume']
                            )
                        
                        # Загружаем 1H/4H для каждой пары — через кеш (60с троттлинг)
                        for sym in symbols:
                            raw_1h = fetcher.get_ohlcv(sym, '1h', 100)
                            raw_4h = fetcher.get_ohlcv(sym, '4h', 50)
                            if raw_1h:
                                # Конвертируем [ts,o,h,l,c,v] → {o,h,l,c,v,t}
                                mtf_candles_1h[sym] = [
                                    {'o':c[1],'h':c[2],'l':c[3],'c':c[4],'v':c[5],'t':c[0]}
                                    for c in raw_1h
                                ]
                            if raw_4h:
                                mtf_candles_4h[sym] = [
                                    {'o':c[1],'h':c[2],'l':c[3],'c':c[4],'v':c[5],'t':c[0]}
                                    for c in raw_4h
                                ]
                except Exception as e:
                    logger.debug(f"[MTF] Загрузка через кеш: fallback ({e})")
                    # Fallback: через прямой fetch (без кеша)
                    try:
                        btc_ticker = self.exchange.fetch_ticker('BTC/USDT')
                        btc_price = btc_ticker['last']
                    except Exception as _e:
                        logger.debug(f"[trading_cycle] btc fetch: {_e}")
                
                # Инициализация DecisionEngine (синглтон — один на весь процесс)
                if not hasattr(self, '_de'):
                    from decision_engine import DecisionEngine
                    self._de = DecisionEngine()
                de = self._de
                
                # BTC-корреляция для DE
                if btc_price is not None:
                    de.set_multi_tf_data(btc_price=btc_price)
                    de.update_btc_reference(btc_price)
                
                # BTC Regime Tracker: загружаем/обновляем фазу BTC
                if not hasattr(self, '_btc_regime'):
                    from btc_regime_tracker import BTCRegimeTracker
                    self._btc_regime = BTCRegimeTracker()
                if btc_5m_candles is not None and btc_1h_candles is not None:
                    try:
                        regime_result = self._btc_regime.update(btc_5m_candles, btc_1h_candles)
                        logger.info(f"🧠 BTC Regime: {regime_result['regime']} → rec={regime_result['recommendation']} ({regime_result['message']})")
                        if not self._btc_regime.is_buy_allowed():
                            logger.info(f"⏳ BTC Regime: BUY заблокирован — {regime_result['regime']}. Пропускаем входы.")
                        
                        # 🎯 ДИНАМИЧЕСКОЕ ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ ПО ТРЕНДУ BTC
                        # Определяем режим: up_trend или down_trend
                        regime = regime_result['regime']
                        
                        # Upward regimes — трейлинг, крупный лот, без импульса
                        # Downward regimes — импульс, мелкий лот
                        up_regimes = ('accumulation', 'recovery')
                        down_regimes = ('dump', 'distribution', 'pump')
                        
                        if regime in up_regimes:
                            self._current_market_mode = 'up_trend'
                            # Импульс: только при очень сильных сигналах (почти отключён)
                            impulse_cfg = {
                                'exit_score_threshold': 95,
                                'min_confirmations': 2,
                                'consecutive_confirmations': 3,
                                'micro_trend_filter': True,
                                'min_hold_seconds': 120,
                                'lookback_candles': 15,
                                'volume_drop_threshold': 0.3,
                                'wick_threshold': 0.6,
                                'strong_volume_drop': 0.5,
                                'body_shrink_threshold': 0.4
                            }
                            logger.info(f"📈 РЕЖИМ UP-TREND: трейлинг +50% лота, импульс подавлен")
                        elif regime in down_regimes:
                            self._current_market_mode = 'down_trend'
                            # Импульс: активен, как сейчас
                            impulse_cfg = {
                                'exit_score_threshold': 65,
                                'min_confirmations': 1,
                                'consecutive_confirmations': 1,
                                'micro_trend_filter': False,
                                'min_hold_seconds': 0,
                                'lookback_candles': 15,
                                'volume_drop_threshold': 0.3,
                                'wick_threshold': 0.6,
                                'strong_volume_drop': 0.5,
                                'body_shrink_threshold': 0.4
                            }
                            logger.info(f"📉 РЕЖИМ DOWN-TREND: импульс активен, лот стандартный")
                        else:
                            self._current_market_mode = 'neutral'
                            impulse_cfg = {
                                'exit_score_threshold': 75,
                                'min_confirmations': 2,
                                'consecutive_confirmations': 2,
                                'micro_trend_filter': True,
                                'min_hold_seconds': 60,
                                'lookback_candles': 15,
                                'volume_drop_threshold': 0.3,
                                'wick_threshold': 0.6,
                                'strong_volume_drop': 0.5,
                                'body_shrink_threshold': 0.4
                            }
                            logger.info(f"➡️ РЕЖИМ NEUTRAL: стандартные настройки")
                        
                        # Записываем конфиг (импульс читает его на лету)
                        try:
                            with open('/tmp/impulse_config.json', 'w') as f_ic:
                                json.dump(impulse_cfg, f_ic)
                            logger.info(f"⚙️ Импульс-конфиг обновлён для режима {self._current_market_mode}")
                        except Exception as e_ic:
                            logger.warning(f"⚠️ Ошибка записи impulse_config: {e_ic}")
                    except Exception as e:
                        logger.warning(f"⚠️ BTC Regime Tracker: {e}")
                
                # BTC Direction Predictor: загружаем/обновляем прогноз каждую итерацию
                if hasattr(de, '_btc_predictor'):
                    if de._btc_predictor is None:
                        try:
                            from btc_direction import BTCDirectionPredictor
                            de._btc_predictor = BTCDirectionPredictor(exchange=self.exchange)
                            model_loaded = de._btc_predictor._load_model()
                            if model_loaded:
                                logger.info("🧠 BTC Direction Predictor: модель загружена с диска")
                            else:
                                logger.info("🧠 BTC Direction Predictor: модель не найдена, будет обучена")
                        except Exception as e:
                            logger.warning(f"⚠️ BTC Direction Predictor init: {e}")
                    
                    # Пересчитываем прогноз каждую итерацию цикла
                    if de._btc_predictor is not None:
                        try:
                            signal = de._btc_predictor.predict()
                            logger.info(f"🔮 BTC Direction: {signal['direction']} (conf={signal['confidence']:.0%}, strength={signal['strength']}, up={signal['up_probability']:.0%} down={signal['down_probability']:.0%})")
                        except Exception as e:
                            logger.warning(f"⚠️ BTC Direction Predictor predict: {e}")
                
                # 📊 Обновляем портфельные метрики для MLAdvisor
                try:
                    _day_start = datetime.now(timezone.utc).strftime('%Y-%m-%d') + 'T00:00'
                    _daily_pnl, _daily_trades, _daily_wins, _daily_losses = db.get_daily_stats(_day_start)
                    _total_pos = db.get_total_position_value()
                    _pos_vals = db.get_all_active_position_values()
                    _avg_pos = sum(_pos_vals)/max(len(_pos_vals), 1)
                    _capital = getattr(self, 'capital', 300)
                    _exposure = _total_pos / max(_capital, 1) * 100
                    
                    # Считаем consecutive_profits из последних сделок
                    _recent_pnls = db.get_recent_pnls(20)
                    _cons_profits = 0
                    for _p in _recent_pnls:
                        if _p > 0:
                            _cons_profits += 1
                        else:
                            break
                    _cons_losses = 0
                    for _p in _recent_pnls:
                        if _p < 0:
                            _cons_losses += 1
                        else:
                            break
                    _c2.close()
                    
                    from ml_advisor import get_advisor
                    _adv = get_advisor()
                    _adv.update_portfolio_stats(
                        daily_pnl=_daily_pnl,
                        profit_count=_daily_wins,
                        loss_count=_daily_losses,
                        trade_count=_daily_trades,
                        consecutive_profits=_cons_profits,
                        consecutive_losses_global=_cons_losses,
                        open_positions=_open_pos,
                        exposure_pct=_exposure,
                        avg_position_value=_avg_pos
                    )
                    if _daily_trades % 5 == 0 or _daily_trades < 3:
                        logger.debug(f"📊 Портфель: PnL=\${_daily_pnl:.2f}, сделок={_daily_trades}, "
                                     f"профитных={_daily_wins}, открыто={_open_pos}, "
                                     f"экспозиция={_exposure:.0f}%")
                except Exception as _e:
                    logger.debug(f"[portfolio_stats] {_e}")

                # Анализ каждого символа
                for symbol in symbols:
                    if not self.running:
                        break
                    
                    # Получение и анализ данных
                    # Загрузка мультитаймфреймовых данных для ML-PRO v2
                    df = self.get_market_data(symbol, '5m', 100)
                    if df.empty:
                        continue
                    
                    df = self.calculate_indicators(df)
                    analysis = self.analyze_trend(df)
                    
                    current_price = df['close'].iloc[-1]
                    
                    # Логирование анализа
                    logger.info(f"{symbol}: Цена=${current_price:.2f}, Тренд={analysis['trend']}, "
                              f"Уверенность={analysis['confidence']:.2%}, RSI={analysis['rsi']:.1f}")
                    
                    # ─── ТОРГОВАЯ ЛОГИКА ──────────────────────────────────────
                    
                    # 🛡️ BTC Regime Check: блокируем вход если BTC в pump/dump/distribution
                    if symbol not in self.positions:
                        if hasattr(self, '_btc_regime') and not self._btc_regime.is_buy_allowed():
                            regime_name = self._btc_regime.get_regime()
                            logger.info(f"⏳ [DE→HOLD] {symbol}: BTC Regime {regime_name} — BUY заблокирован")
                            continue
                    
                    # ⚡ DecisionEngine: ЕДИНСТВЕННЫЙ источник решений о входе
                    # DE оценивает рынок ML-ансамблем. Если одобрил — входим.
                    if symbol not in self.positions:
                        try:
                            de_price = current_price
                            de_confidence = analysis['confidence']
                            de_trend = analysis['trend']
                            de_rsi = analysis['rsi']
                            de_positions = len(self.positions)
                            max_pos = self.config['risk_management'].get('max_open_positions', 3)
                            # Multi-timeframe данные для ML-голосов (своя 1H/4H для каждой пары)
                            c5m = df.to_dict('records') if hasattr(df, 'to_dict') else None
                            c1h = mtf_candles_1h.get(symbol, None)
                            c4h = mtf_candles_4h.get(symbol, None)
                            decision = de.decide_entry(symbol, de_confidence, de_trend, de_rsi,
                                                       de_price, de_positions, max_pos,
                                                       candles_5m=c5m, candles_1h=c1h, candles_4h=c4h)
                            
                            if decision.action == 'enter':
                                # 🌙 НОЧНОЙ РЕЖИМ: не входить с 23:00 до 08:00 KRAT (UTC+7)
                                _local_hour = (datetime.now(timezone.utc).hour + 7) % 24
                                if _local_hour >= 23 or _local_hour < 8:
                                    logger.info(f"🌙 {symbol}: ночной режим ({_local_hour}:00 KRAT). Пропускаю.")
                                    continue

                                score = decision.score or 50
                                # Адаптивный размер позиции: от DE или стандартный
                                size_mult = decision.position_size or 0.5
                                
                                # 🎯 РЕЖИМ-ЗАВИСИМЫЙ РАЗМЕР ПОЗИЦИИ
                                # На восходящем тренде входим крупнее (×2-3)
                                # На нисходящем — консервативно (×0.5-1)
                                market_mode = getattr(self, '_current_market_mode', 'neutral')
                                if market_mode == 'up_trend':
                                    regime_mult = 2.5  # $30-50 на сделку
                                    logger.info(f"📈 UP-TREND: множитель позиции {regime_mult}x")
                                elif market_mode == 'down_trend':
                                    regime_mult = 0.7  # $10-15 на сделку
                                    logger.info(f"📉 DOWN-TREND: множитель позиции {regime_mult}x")
                                else:
                                    regime_mult = 1.0  # $15-25 стандарт
                                size_mult = size_mult * regime_mult
                                
                                # Лимиты безопасности
                                max_positions = self.config['risk_management'].get('max_open_positions', 3)
                                if len(self.positions) >= max_positions:
                                    logger.info(f"⚠️ [DE] {symbol}: лимит {len(self.positions)}/{max_positions}")
                                    continue
                                
                                # 🛡️ ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: кулдаун после SL (если DE пропустил)
                                if hasattr(de, '_last_decisions') and symbol in de._last_decisions:
                                    last_exit = de._last_decisions[symbol]
                                    time_since = time.time() - last_exit.get('exit_time', 0)
                                    cooldown = getattr(de, 'reentry_cooldown', 3600)
                                    if time_since < cooldown:
                                        remain = cooldown - time_since
                                        logger.warning(f"🛡️ {symbol}: кулдаун {remain/60:.0f}мин (SL защита). Пропускаю BUY.")
                                        continue
                                
                                # 🛡️ ЗАЩИТА ОТ ДВОЙНЫХ входов: 5 мин кулдаун + БД проверка
                                _buy_lock = getattr(self, '_buy_locks', {})
                                last_buy = _buy_lock.get(symbol, 0.0)
                                if time.time() - last_buy < 300:
                                    remaining = int(300 - (time.time() - last_buy))
                                    logger.warning(f"🔒 {symbol}: блокировка повторного BUY ({remaining}с)")
                                    continue
                                # Дополнительная проверка: есть ли уже позиция в БД
                                try:
                                    in_db = db.position_exists(symbol)
                                    # Проверяем реальную позицию: в БД И в open_positions (биржевой статус)
                                    # Иначе старые записи в БД блокируют ре-вход
                                    if in_db and symbol in self.positions:
                                        logger.warning(f"🛡️ {symbol}: уже есть в БД, пропускаю повторный BUY")
                                        continue
                                    elif in_db and symbol not in self.positions:
                                        logger.info(f"🧹 {symbol}: очистка устаревшей записи в БД (нет на бирже)")
                                        # 🩹 FIX: закрываем с текущей рыночной ценой и PnL
                                        try:
                                            symbol_key = symbol.replace('/', '')
                                            clean_price = 0
                                            try:
                                                ticker = self.exchange.fetch_ticker(symbol_key)
                                                clean_price = ticker.get('last') or 0
                                            except Exception:
                                                pass
                                            db.close_trade(
                                                symbol,
                                                exit_price=float(clean_price) if clean_price else 0,
                                                exit_qty=0,
                                                pnl=0,
                                                pnl_pct=0,
                                                exit_reason='stale_cleanup'
                                            )
                                            logger.info(f"   ✅ {symbol}: stale-запись закрыта @ ${clean_price:.4f}")
                                        except Exception as cleanup_e:
                                            logger.error(f"   ❌ {symbol}: ошибка при очистке БД: {cleanup_e}")
                                except Exception as _e:
                                    logger.debug(f"[buy_lock] DB check: {_e}")
                                logger.info(f"⚡ [DE→BUY] {symbol}: score={score:.0f}/100 size={size_mult:.2f} | ${de_price:.4f} | {decision.reason}")
                                # Размер позиции = стандартный * size_mult от DE, с учётом Score
                                base_qty = self.calculate_position_size(symbol, de_price, score)
                                quantity = base_qty * size_mult
                                if quantity * de_price <= self.available_capital:
                                    # ✅ ЗАПОМИНАЕМ buy_lock ТОЛЬКО при реальной покупке
                                    _buy_lock[symbol] = time.time()
                                    self._buy_locks = _buy_lock
                                    # ✅ ЗАПОМИНАЕМ SL/TP УРОВНИ ОТ DE в позиции
                                    self.execute_trade(symbol, 'buy', quantity, de_price,
                                        sl_price=decision.sl_price, tp_price=decision.tp_price,
                                        trail_act=decision.trail_act, trail_dist=decision.trail_dist,
                                        max_hold_h=decision.max_hold_h)
                                else:
                                    logger.warning(f"⏭️ [DE] {symbol}: недостаточно капитала")
                            else:
                                reason = decision.reason or ''
                                # Всегда логируем отклонённые решения — надо видеть почему DE не входит
                                logger.info(f"⏳ [DE→HOLD] {symbol}: {reason}")
                        except Exception as e_de:
                            logger.warning(f"[DE] {symbol}: ошибка: {e_de}")
                    
                    else:
                        # Управление открытой позицией
                        position = self.positions[symbol]
                        
                        # 🩹 СТАРЫЕ ПОЗИЦИИ: используем _created_at (время в памяти),
                        # а не entry_time (реальное время покупки — может быть вчерашним при DCA)
                        hold_start = position.get('_created_at') or position.get('entry_time', time.time())
                        if isinstance(hold_start, str):
                            pos_time = datetime.fromisoformat(hold_start)
                            if pos_time.tzinfo is not None:
                                pos_time = pos_time.replace(tzinfo=None)
                            hold_hours = (datetime.now() - pos_time).total_seconds() / 3600
                        else:
                            hold_hours = (time.time() - hold_start) / 3600
                        
                        # ✅ ЕДИНЫЙ ЦЕНТР РЕШЕНИЙ: DE решает когда выходить
                        entry_price = position['entry_price']
                        pnl_percent = (current_price - entry_price) / entry_price * 100
                        
                        # 🧹 СТАРЫЕ ПОЗИЦИИ: force-закрытие если больше 36ч и PnL в - или меньше +2%
                        stale_hours = 36
                        stale_sell_reason = None
                        if hold_hours >= stale_hours:
                            if pnl_percent < 2.0:
                                stale_sell_reason = f"Принудительно: позиция {hold_hours:.0f}ч, PnL={pnl_percent:+.2f}%"
                                logger.warning(f"⏰ [STALE] {symbol}: {stale_sell_reason}. Закрываю.")
                        
                        # Параметры HMM-адаптивного риска
                        rp = de.get_sl_tp_params(side='long')
                        
                        # Берём ценовые уровни из position (установлены при входе) или из конфига
                        sl_price = position.get('_sl_price', entry_price * (1 - rp['sl_pct'] / 100))
                        tp_price = position.get('_tp_price', entry_price * (1 + rp['tp_pct'] / 100))
                        trail_act = position.get('_trail_act', rp['trail_act'])
                        trail_dist = position.get('_trail_dist', rp['trail_dist'])
                        max_hold = position.get('_max_hold_h', rp['max_hold_h'])
                        
                        # Апдейт максимума для трейлинга
                        if 'max_profit' not in position:
                            position['max_profit'] = pnl_percent
                        else:
                            position['max_profit'] = max(position['max_profit'], pnl_percent)
                        # Точная максимальная цена (для корректного highest_price в decide_exit)
                        if '_highest_price' not in position or current_price > position['_highest_price']:
                            position['_highest_price'] = current_price
                        
                        # 🎯 РАННИЙ ТРЕЙЛИНГ: цена прошла +1.5% — активируем плотный трейлинг 0.3%
                        # Отличие от стандартного трейлинга: срабатывает раньше и плотнее
                        # Если рост продолжается — остаёмся в позиции и трейлим
                        early_trail_triggered = False
                        early_trail_reason = ""
                        # 🟢 Активация раннего трейлинга (при PnL >= 1.5%)
                        if pnl_percent >= 1.5 and rp['tp_pct'] > 2.0:
                            if '_early_trail_peak' not in position:
                                position['_early_trail_peak'] = current_price
                                logger.info(f"📈 [Ранний трейлинг] {symbol}: активирован при PnL={pnl_percent:+.2f}%, пик={current_price:.4f}")
                            elif current_price > position['_early_trail_peak']:
                                position['_early_trail_peak'] = current_price
                        
                        # 🟢 Проверка отката (всегда, если трейлинг уже активирован)
                        # (исправлено: раньше проверка была внутри pnl_percent >= 1.5,
                        #  из-за чего при падении PnL ниже 1.5% трейлинг переставал проверяться)
                        if '_early_trail_peak' in position and not early_trail_triggered:
                            peak = position['_early_trail_peak']
                            trail_price = peak * (1 - 0.003)
                            if current_price <= trail_price and pnl_percent >= 0.8:
                                early_trail_triggered = True
                                early_trail_reason = f"Ранний трейлинг: пик={peak:.4f}, откат до {current_price:.4f} ({pnl_percent:+.2f}%)"
                                logger.info(f"🎯 [Ранний трейлинг] {symbol}: {early_trail_reason}")
                        
                        # ═══════════════════════════════════════════════════════
                        # ⚡ ИМПУЛЬСНЫЙ ВЫХОД (1M свечи) + EXIT ENSEMBLE (модули)
                        # ═══════════════════════════════════════════════════════
                        impulse_exit_signal = None
                        candles_1m = None
                        vsa_for_impulse = None

                        # 🛡️ Auto-reload
                        try:
                            import importlib
                            import impulse_exit as ie
                            importlib.reload(ie)
                        except Exception:
                            try:
                                import impulse_exit as ie
                            except Exception:
                                ie = None

                        # Получаем 1M свечи в одном месте, чтобы не дёргать API дважды
                        if hasattr(self, 'exchange'):
                            try:
                                raw_1m = self.exchange.fetch_ohlcv(symbol, '1m', limit=25)
                                if raw_1m and len(raw_1m) >= 5:
                                    candles_1m = [
                                        {'o':c[1],'h':c[2],'l':c[3],'c':c[4],'v':c[5]}
                                        for c in raw_1m
                                    ]
                            except Exception as e:
                                logger.debug(f"[FETCH] {symbol}: {e}")

                        # ═══ VSA для импульса: быстрая проверка ────────────────
                        if candles_1m and len(candles_1m) >= 10:
                            try:
                                from vsa_analyzer import analyze_volume_spread
                                vsa_imp = analyze_volume_spread(candles_1m)
                                if vsa_imp:
                                    vsa_for_impulse = {
                                        'signal': vsa_imp.signal,
                                        'strength': vsa_imp.strength,
                                        'label': vsa_imp.detail,
                                    }
                            except Exception as e:
                                logger.debug(f"[VSA] {symbol}: {e}")

                        # ═══ Импульсный выход с VSA-валидацией ─────────────────
                        if ie is not None and not ie.is_cooldown_active(symbol, 60):
                            if candles_1m and len(candles_1m) >= 5:
                                try:
                                    impulse_result = ie.detect_impulse_exhaustion(
                                        candles_1m,
                                        entry_price=entry_price,
                                        current_pnl_pct=pnl_percent,
                                        vsa_result=vsa_for_impulse
                                    )
                                    if impulse_result.exhaustion and pnl_percent > 0.3:
                                        impulse_exit_signal = f"Импульс: {impulse_result.detail}"
                                        logger.info(f"⚡ [IMPULSE EXIT] {symbol}: {impulse_result.detail} (PnL={pnl_percent:+.2f}%)")
                                        ie.mark_trigger(symbol)
                                except Exception as e:
                                    logger.debug(f"[IMPULSE] {symbol}: {e}")

                        # ═══ DecisionEngine: SL/TP/трейлинг/таймаут (с exit ensemble) ─
                        exit_decision = de.decide_exit(
                            symbol, entry_price, current_price,
                            highest_price=position.get('_highest_price', current_price * (1 + position.get('max_profit', 0) / 100)),
                            lowest_price=current_price * (1 + min(0, pnl_percent) / 100),
                            entry_time=pos_time, pnl_pct=pnl_percent,
                            sl_pct=rp['sl_pct'], tp_pct=rp['tp_pct'],
                            trail_act=rp['trail_act'], trail_dist=rp['trail_dist'],
                            max_hold_hours=max_hold, side='long',
                            candles_1m=candles_1m
                        )

                        # ═══ SMART EXIT: проверяем exit_override ──────────────
                        # exit_override может быть:
                        #   None — решение не принято (держать)
                        #   'exit' — выходим
                        #   'hold_widen_sl' — не выходим, расширяем SL
                        # Action: 'exit' (выход) или 'hold' (не выходим)

                        should_sell = False
                        sell_reason = ""
                        exit_widened = False

                        # 1) Early trail — если цена просела, но ensemble говорит держать
                        if early_trail_triggered:
                            # Пробуем ensemble (из decide_exit или свой)
                            et_ensemble_override = (
                                exit_decision and exit_decision.exit_override == 'hold_widen_sl'
                            )
                            # Если decide_exit не дал ensemble (exit_decision=None), делаем свой
                            if not et_ensemble_override:
                                ensemble_candles = None
                                ensemble_label = ""
                                hold_threshold = 65

                                # Сначала пробуем 1м свечи (точнее, порог 65)
                                if candles_1m and len(candles_1m) >= 10:
                                    ensemble_candles = candles_1m
                                    ensemble_label = "1m"
                                    hold_threshold = 65
                                # Fallback на 5м если 1м нет (грубее, порог 70)
                                elif df is not None and len(df) >= 10:
                                    ensemble_candles = df.to_dict('records')
                                    ensemble_label = "5m⚠️"
                                    hold_threshold = 70

                                if ensemble_candles is not None:
                                    try:
                                        et_vote = de._evaluate_exit_ensemble(
                                            symbol, entry_price, current_price, pnl_percent,
                                            highest_price=position.get('_highest_price', current_price),
                                            candles_5m=ensemble_candles
                                        )
                                        hold_conf = et_vote.get('hold_confidence', 0)
                                        et_approved = hold_conf >= hold_threshold
                                        if et_approved:
                                            et_ensemble_override = True
                                            logger.info(f"🧠 [SMART EXIT] {symbol}: ранний трейлинг отменён"
                                                        f" ({ensemble_label} ensemble hold={hold_conf}% ≥ {hold_threshold})")
                                        else:
                                            logger.info(f"🧠 [SMART EXIT] {symbol}: ensemble подтвердил выход"
                                                        f" ({ensemble_label} hold={hold_conf}% < {hold_threshold})")
                                    except Exception as _et_e:
                                        logger.debug(f"[SMART] early trail ensemble: {_et_e}")

                            if et_ensemble_override:
                                sell_reason = ""
                                early_trail_triggered = False
                                exit_widened = True
                            else:
                                should_sell = True
                                sell_reason = early_trail_reason

                        # 2) DecisionEngine exit (SL/TP/трейлинг/таймаут с exit ensemble)
                        if not should_sell and exit_decision is not None:
                            if exit_decision.exit_override == 'hold_widen_sl':
                                # Ensemble говорит «держать» — расширяем SL
                                logger.info(f"🧠 [SMART EXIT] {symbol}: {exit_decision.reason}")
                                exit_widened = True
                            else:
                                # Ensemble подтвердил выход
                                should_sell = True
                                sell_reason = exit_decision.reason

                        # 3) Импульсный выход
                        if not should_sell and impulse_exit_signal is not None:
                            should_sell = True
                            sell_reason = impulse_exit_signal

                        # 4) Stale sell
                        if not should_sell and stale_sell_reason is not None:
                            should_sell = True
                            sell_reason = stale_sell_reason

                        # ═══ SMART SL WIDENING: обновляем SL если ensemble отменил выход ─
                        if exit_widened and exit_decision and exit_decision.sl_price is not None:
                            new_sl = exit_decision.sl_price
                            old_sl = entry_price * (1 - rp['sl_pct'] / 100.0)
                            if new_sl < old_sl:  # SL расширяется (ниже текущего)
                                # Обновляем SL в позиции — ключ _sl_price а не sl_price (см. __init__)
                                position['_sl_price'] = new_sl
                                position['smart_exit_count'] = position.get('smart_exit_count', 0) + 1
                                logger.info(f"🧠 SL РАСШИРЕН: {symbol} {old_sl:.6f}→{new_sl:.6f} "
                                            f"(счётчик: {position.get('smart_exit_count')})")

                        # ═══ RE-ENTRY SUPPRESSION: если только что вышли и снова хотим войти ─
                        reentry_check_cooldown = 300  # 5 мин
                        if should_sell and hasattr(de, '_last_decisions'):
                            last_entry = de._last_decisions.get(symbol, {})
                            last_exit_time = last_entry.get('exit_time', 0)
                            if last_exit_time > 0 and (time.time() - last_exit_time) < reentry_check_cooldown:
                                # Вышли менее 5 мин назад — возможно нас выбило шумом
                                # Вместо продажи: восстанавливаем позицию по цене входа
                                # (на самом деле просто пропускаем sell — позиция остаётся)
                                entry_age = time.time() - last_exit_time
                                logger.info(f"🧠 RE-ENTRY SUPPRESSION: {symbol} вышли {entry_age:.0f}с назад")
                                # Отменяем sell, но сбрасываем таймер выхода
                                # чтобы не попасть в цикл "продажа → вход → продажа"
                                should_sell = False
                                sell_reason = ""
                                de._last_decisions.pop(symbol, None)

                        if not should_sell and not exit_widened:
                            # Статусный лог (раз в цикл для видимых PnL)
                            if abs(pnl_percent) > 1:
                                logger.info(f"📊 {symbol}: PnL={pnl_percent:+.2f}%, время={hold_hours:.0f}ч, трейд={position.get('max_profit', 0):.1f}% пик")
                        
                        if should_sell:
                            logger.info(f"🎯 ВЫХОД ИЗ ПОЗИЦИИ {symbol}: {sell_reason}")
                            quantity = position['quantity']
                            
                            # 🔄 ЕДИНЫЙ ИСТОЧНИК ИСТИНЫ: проверка реального баланса перед sell
                            try:
                                balance = self.exchange.fetch_balance()
                                currency = symbol.split('/')[0]
                                free_asset = balance['free'].get(currency, 0)
                                total_asset = balance['total'].get(currency, 0)
                                pos_value = total_asset * current_price
                            except Exception as e:
                                logger.error(f"❌ Ошибка проверки баланса для {symbol}: {e}")
                                free_asset = 0
                                total_asset = 0
                                pos_value = 0
                            
                            # Если на бирже нет монет — чистим кеш и идём дальше
                            if total_asset < 0.000001:
                                logger.warning(f"⏭️ {symbol}: нет на бирже ({currency}=0). Удаляю из памяти.")
                                if symbol in self.positions:
                                    self.positions.pop(symbol, None)
                                db.remove_position(symbol)
                                continue
                            
                            # Мусорный остаток (< $1) — не продаём, просто чистим кеш
                            MIN_POSITION_VALUE = 1.0
                            if pos_value < MIN_POSITION_VALUE:
                                logger.warning(f"⏭️ {symbol}: остаток ${pos_value:.2f} < ${MIN_POSITION_VALUE:.2f}. Чищу кеш без продажи.")
                                if symbol in self.positions:
                                    self.positions.pop(symbol, None)
                                db.remove_position(symbol)
                                continue
                            
                            # Есть монеты — продаём через биржу
                            safe_qty = free_asset * 0.999 if free_asset < quantity else quantity
                            if safe_qty < quantity * 0.9:
                                logger.warning(f"⚠️ {symbol}: free={free_asset:.6f} < qty={quantity:.6f}. Использую free.")
                            
                            result = self.execute_trade(symbol, 'sell', safe_qty, current_price)

                            # 🚫 Кулдаун re-entry: запомнить время выхода
                            if result is not None:
                                # Определяем убыток: если цена продажи ниже цены входа (для long)
                                pos = self.positions.get(symbol, {})
                                entry_price_pos = pos.get('entry_price', current_price)
                                was_loss = current_price < entry_price_pos
                                de.record_exit(symbol, sell_reason, was_loss=was_loss)
                            
                            # 🧠 ML: обучаем на результате сделки
                            if result is not None and symbol in self.positions:
                                try:
                                    from ml_advisor import ml_add_result, ml_train
                                    pos = self.positions[symbol]
                                    # analysis гарантированно определена (строки 856-857 до if/else)
                                    ml_add_result(symbol, pos['entry_price'], current_price,
                                                  analysis['rsi'], analysis['trend'],
                                                  analysis['confidence'], 0.1, sell_reason)
                                    ml_train()
                                except Exception as e:
                                    logger.warning(f"ML обучение: {e}")
                            
                            # 🛡️ Если sell не удался — принудительно удаляем позицию
                            if result is None and symbol in self.positions:
                                logger.warning(f"⚠️ Sell {symbol} не удался. Принудительно очищаю позицию из памяти.")
                                self.positions.pop(symbol, None)
                                db.remove_position(symbol)
                
                # Периодическая очистка старых триггеров + ретрейн ML (раз в 10 мин)
                if getattr(self, '_last_cleanup_time', 0) < time.time() - 600:
                    try:
                        import impulse_exit as ie_clean
                        by_ie = ie_clean.cleanup_old_triggers(7200)
                        by_de = de.cleanup_old_decisions(7200)
                        if by_ie + by_de > 0:
                            logger.info(f"🧹 Очистка кешей: импульс={by_ie}, решения={by_de}")
                    except Exception:
                        pass
                    # 🧠 Advisor: попытка ретрейна (раз в час, если хватает данных)
                    try:
                        from ml_advisor import ml_train
                        ml_train(force=False)
                    except Exception:
                        pass
                    self._last_cleanup_time = time.time()
                
                # Сохраняем реальный баланс с биржи для дашборда
                self._save_real_balance()
                
                # 📊 Диагностика длительности цикла
                cycle_end = time.time()
                cycle_duration = cycle_end - cycle_start
                if cycle_duration > 60:
                    logger.warning(f"⚠️ Цикл занял {cycle_duration:.0f}с — возможно опаздываем")
                elif cycle_duration > 30:
                    logger.info(f"⌛ Цикл: {cycle_duration:.0f}с")
                
                # Пауза между циклами
                time.sleep(trading_config['check_interval_seconds'])
                
            except Exception as e:
                logger.error(f"Ошибка в торговом цикле: {e}")
                time.sleep(60)
    
    def start(self):
        """Запуск торговой системы"""
        if self.running:
            logger.warning("Система уже запущена")
            return
        
        self.running = True
        logger.info("Запуск промышленной торговой системы")
        
        # Запуск торгового цикла в отдельном потоке
        self.trading_thread = threading.Thread(target=self.trading_cycle, daemon=True)
        self.trading_thread.start()
        
        # Запуск мониторинга
        self.monitor_thread = threading.Thread(target=self.monitor_system, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Система успешно запущена")
    
    def stop(self):
        """Остановка торговой системы"""
        logger.info("Остановка промышленной торговой системы")
        self.running = False
        
        # Закрытие всех открытых позиций
        for symbol, position in list(self.positions.items()):
            try:
                # Получаем текущую цену
                df = self.get_market_data(symbol, '1m', 1)
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    self.execute_trade(symbol, 'sell', position['quantity'], current_price)
            except Exception as e:
                logger.error(f"Ошибка закрытия позиции {symbol}: {e}")
        
        # Ожидание завершения потоков
        if hasattr(self, 'trading_thread'):
            self.trading_thread.join(timeout=10)
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=10)
        
        logger.info("Система остановлена")
    
    def monitor_system(self):
        """Мониторинг состояния системы"""
        while self.running:
            try:
                # Обновление статистики
                self.update_stats()
                
                # Логирование состояния
                logger.info(f"=== СТАТУС СИСТЕМЫ ===")
                logger.info(f"Капитал: ${self.available_capital:.2f} / ${self.capital:.2f}")
                logger.info(f"Дневной P&L: ${self.daily_pnl:.2f} ({self.daily_pnl/self.capital*100:.2f}%)")
                logger.info(f"Дневные сделки: {self.daily_trades}")
                logger.info(f"Открытые позиции: {len(self.positions)}")
                logger.info(f"Всего сделок: {self.stats['total_trades']}")
                logger.info(f"Win Rate: {self.stats['winning_trades']/max(self.stats['total_trades'],1)*100:.1f}%")
                logger.info(f"Общий P&L: ${self.stats['total_pnl']:.2f}")
                
                # Проверка аварийной остановки
                if self.config['system']['emergency_stop_enabled']:
                    if self.daily_pnl <= -self.config['system']['max_loss_daily']:
                        logger.critical(f"Аварийная остановка! Дневные потери: {self.daily_pnl}%")
                        self.stop()
                        break
                
                time.sleep(60)  # Обновление каждую минуту
                
            except Exception as e:
                logger.error(f"Ошибка мониторинга: {e}")
                time.sleep(30)
    
    def _save_real_balance(self):
        """Сохраняет реальный баланс с биржи в /tmp/real_balance.json.
        1 API call: fetch_balance(). Цены позиций берёт из self.positions."""
        try:
            balance = self.exchange.fetch_balance()
            free_usdt = balance['free'].get('USDT', 0)
            
            # Стоимость позиций по последней известной цене (без lock:
            # копируем данные за один проход)
            pos_value = 0
            pos_count = 0
            positions_copy = dict(self.positions)
            for sym, pos in positions_copy.items():
                if isinstance(pos, dict):
                    price = pos.get('current_price', pos.get('entry_price', 0))
                    qty = pos.get('quantity', 0)
                    val = qty * price
                    if val > 1.0:
                        pos_value += val
                        pos_count += 1
            
            data = {
                'total': round(float(free_usdt) + float(pos_value), 2),
                'free_usdt': round(float(free_usdt), 2),
                'in_positions': round(float(pos_value), 2),
                'positions_count': pos_count,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
            with open('/tmp/real_balance.json', 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.info(f"Balance save error: {e}")
    
    def update_stats(self):
        """Обновление статистики"""
        if not self.trades_history:
            return
        
        # Фильтруем завершенные сделки
        closed_trades = [t for t in self.trades_history if 'pnl' in t]
        
        if not closed_trades:
            return
        
        self.stats['total_trades'] = len(closed_trades)
        self.stats['winning_trades'] = sum(1 for t in closed_trades if t.get('pnl', 0) > 0)
        self.stats['losing_trades'] = sum(1 for t in closed_trades if t.get('pnl', 0) < 0)
        self.stats['total_pnl'] = sum(t.get('pnl', 0) for t in closed_trades)
        
        # Расчет максимальной просадки
        if closed_trades:
            cumulative_pnl = 0
            peak = 0
            max_dd = 0
            
            for trade in closed_trades:
                cumulative_pnl += trade.get('pnl', 0)
                peak = max(peak, cumulative_pnl)
                drawdown = peak - cumulative_pnl
                max_dd = max(max_dd, drawdown)
            
            self.stats['max_drawdown'] = max_dd
    
    def get_system_status(self) -> Dict:
        """Получение текущего статуса системы"""
        return {
            'running': self.running,
            'capital': {
                'total': self.capital,
                'available': self.available_capital,
                'used': self.capital - self.available_capital
            },
            'daily': {
                'pnl': self.daily_pnl,
                'pnl_percent': self.daily_pnl / self.capital * 100 if self.capital > 0 else 0,
                'trades': self.daily_trades
            },
            'positions': {
                'count': len(self.positions),
                'details': self.positions
            },
            'statistics': self.stats,
            'config': {
                'paper_trading': self.config['bybit']['paper_trading'],
                'enabled_pairs': self.config['trading']['enabled_pairs'],
                'risk_limits': self.config['risk_management']
            }
        }
    
    def generate_report(self) -> str:
        """Генерация отчета о работе системы"""
        status = self.get_system_status()
        
        report = f"""
{'='*60}
ОТЧЕТ ПРОМЫШЛЕННОЙ ТОРГОВОЙ СИСТЕМЫ
Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

СТАТУС СИСТЕМЫ:
  Запущена: {'ДА' if status['running'] else 'НЕТ'}
  Режим: {'Бумажная торговля' if status['config']['paper_trading'] else 'Реальная торговля'}

КАПИТАЛ:
  Общий: ${status['capital']['total']:.2f}
  Доступный: ${status['capital']['available']:.2f}
  Использовано: ${status['capital']['used']:.2f}

ДНЕВНАЯ СТАТИСТИКА:
  P&L: ${status['daily']['pnl']:.2f} ({status['daily']['pnl_percent']:.2f}%)
  Сделки: {status['daily']['trades']}

ПОЗИЦИИ:
  Открыто: {status['positions']['count']}
"""
        
        for symbol, pos in status['positions']['details'].items():
            report += f"  - {symbol}: {pos['quantity']:.6f} @ ${pos['entry_price']:.2f} ({pos['side']})\n"
        
        report += f"""
ОБЩАЯ СТАТИСТИКА:
  Всего сделок: {status['statistics']['total_trades']}
  Выигрышных: {status['statistics']['winning_trades']}
  Проигрышных: {status['statistics']['losing_trades']}
  Win Rate: {status['statistics']['winning_trades']/max(status['statistics']['total_trades'],1)*100:.1f}%
  Общий P&L: ${status['statistics']['total_pnl']:.2f}
  Макс. просадка: ${status['statistics']['max_drawdown']:.2f}

КОНФИГУРАЦИЯ РИСКОВ:
  Макс. дневная потеря: {status['config']['risk_limits']['max_daily_loss_percent']}%
  Стоп-лосс: {status['config']['risk_limits']['stop_loss_percent']}%
  Тейк-профит: {status['config']['risk_limits']['take_profit_percent']}%
  Макс. позиций: {status['config']['risk_limits']['max_open_positions']}

ТОРГУЕМЫЕ ПАРЫ: {', '.join(status['config']['enabled_pairs'])}
{'='*60}
"""
        return report

# Дополнительные утилиты
class TradingUtils:
    """Утилиты для торговли"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Расчет коэффициента Шарпа"""
        if not returns:
            return 0.0
        
        import numpy as np
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate/252  # Дневная безрисковая ставка
        
        if len(excess_returns) < 2:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Расчет коэффициента Сортино"""
        if not returns:
            return 0.0
        
        import numpy as np
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate/252
        
        # Только отрицательные возвраты
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) < 2:
            return 0.0
        
        sortino = np.mean(excess_returns) / np.std(negative_returns) * np.sqrt(252)
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: List[float]) -> Tuple[float, int, int]:
        """Расчет максимальной просадки"""
        if not equity_curve:
            return 0.0, 0, 0
        
        peak = equity_curve[0]
        max_dd = 0.0
        peak_idx = 0
        trough_idx = 0
        
        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
                peak_idx = i
            
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
                trough_idx = i
        
        return max_dd, peak_idx, trough_idx

if __name__ == "__main__":
    # Тестовый запуск
    trader = IndustrialTrader("config/api_config_final.json")
    print(trader.generate_report())