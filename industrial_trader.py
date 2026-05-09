#!/usr/bin/env python3
"""
ПРОМЫШЛЕННАЯ ТОРГОВАЯ СИСТЕМА - Industrial Super System
Версия: 1.0.0
Автор: Капитан (Главный координатор системы)
"""

import json
import time
import logging
from datetime import datetime
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import threading
import signal
import sys
import db  # SQLite БД вместо JSON-трекера

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
        self.trades_history = []
        self.capital = self.config['trading']['capital']
        self.available_capital = self.capital
        self.daily_pnl = 0.0
        self.daily_trades = 0
        
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
            for currency, amount in balance['total'].items():
                if currency != 'USDT' and amount > 0.000001:  # Минимальный порог
                    try:
                        ticker = self.exchange.fetch_ticker(f"{currency}/USDT")
                        current_price = ticker['last']
                        value_usdt = amount * current_price
                        
                        if value_usdt > 1.0:  # Позиции больше $1
                            symbol = f"{currency}/USDT"
                            
                            # Цена входа: используем последнюю цену покупки из trade_history
                            # (не WA — он искажает PnL из-за старых бумажных сделок)
                            trades = db.get_trade_history(symbol, limit=1, side='buy')
                            if trades and trades[0]['price'] > 0:
                                entry_price = trades[0]['price']
                                entry_time = trades[0]['timestamp']
                                logger.info(f"   📊 Последняя цена покупки из истории: {symbol}: ${entry_price:.4f}")
                            else:
                                # Если нет в истории — пытаемся получить с биржи
                                avg_price, first_trade_time = self.get_average_buy_price(symbol, amount)
                                
                                if avg_price:
                                    entry_price = avg_price
                                    entry_time = first_trade_time
                                    logger.info(f"   📊 Средняя цена покупки {symbol}: ${avg_price:.4f}")
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
                            
                    except Exception as e:
                        logger.warning(f"Не удалось обработать {currency}: {e}")
            
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
            # Получаем историю закрытых ордеров за последние 24 часа
            since = self.exchange.milliseconds() - (24 * 60 * 60 * 1000)
            
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
        """Загрузка конфигурации из JSON файла"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Конфигурация загружена из {config_path}")
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
    
    def check_risk_limits(self) -> bool:
        """Проверка лимитов риска"""
        risk_config = self.config['risk_management']
        
        # Проверка максимальной дневной потери
        if self.daily_pnl <= -risk_config['max_daily_loss_percent']:
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
    
    def execute_trade(self, symbol: str, side: str, quantity: float, price: float) -> Optional[Dict]:
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
                
                # Обновляем позиции
                if side == 'buy':
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'entry_price': price,
                        'entry_time': timestamp,
                        'side': 'long',
                        # SL/TP уровни от DecisionEngine (заполнит caller)
                        '_sl_price': sl_price if 'sl_price' in dir() else None,
                        '_tp_price': tp_price if 'tp_price' in dir() else None,
                        '_trail_act': trail_act if 'trail_act' in dir() else None,
                        '_trail_dist': trail_dist if 'trail_dist' in dir() else None,
                        '_max_hold_h': max_hold_h if 'max_hold_h' in dir() else None,
                    }
                    self.available_capital -= quantity * price
                    db.add_trade(symbol, 'buy', price, quantity, timestamp=timestamp)
                elif side == 'sell' and symbol in self.positions:
                    position = self.positions.pop(symbol)
                    pnl = (price - position['entry_price']) * quantity
                    self.available_capital += quantity * price + pnl
                    self.daily_pnl += pnl
                    self.daily_trades += 1
                    
                    trade['pnl'] = pnl
                    trade['pnl_percent'] = (pnl / (position['entry_price'] * quantity)) * 100
                    db.add_trade(symbol, 'sell', price, quantity, pnl=pnl, pnl_pct=trade['pnl_percent'], timestamp=timestamp)
                
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
                    order_price = order.get('average', order.get('price', price)) or price
                    filled_qty = order.get('filled', quantity) or quantity
                    
                    # Запись точной цены в trade_history
                    db.add_trade(
                        symbol=symbol,
                        side='buy',
                        price=order_price,
                        quantity=filled_qty,
                        order_id=order['id'],
                        exchange_id=order['id'],
                        timestamp=datetime.now().isoformat()
                    )
                    
                    # Берём реальное количество из order.filled или quantity
                    actual_qty = filled_qty if filled_qty > 0 else quantity
                    
                    # Добавляем новую позицию
                    # Используем order_price (цена BUY ордера), а не WA из БД
                    # Баг: DB.calculate_weighted_entry хранит старые цены, искажая PnL
                    real_entry = order_price  # берём цену, по которой реально исполнился BUY
                    if symbol not in self.positions:
                        self.positions[symbol] = {
                            'quantity': actual_qty,
                            'entry_price': real_entry,
                            'entry_time': datetime.now().isoformat(),
                            'max_profit': 0.0
                        }
                        logger.info(f"📥 Добавлена новая позиция: {symbol} - {actual_qty:.6f} @ ${real_entry:.4f}")
                    else:
                        self.positions[symbol]['quantity'] = actual_qty
                        self.positions[symbol]['entry_price'] = real_entry
                        self.positions[symbol]['entry_time'] = datetime.now().isoformat()
                        logger.info(f"📊 Обновлена позиция {symbol}: {actual_qty:.6f} @ ${real_entry:.4f}")
                elif side == 'sell':
                    # Запись sell в trade_history
                    pnl = 0
                    pnl_pct = 0
                    if symbol in self.positions:
                        entry = self.positions[symbol]['entry_price']
                        filled = order.get('filled', quantity) or quantity
                        filled_price = order.get('average', order.get('price', price)) or price
                        pnl = (filled_price - entry) * filled
                        pnl_pct = ((filled_price - entry) / entry) * 100 if entry > 0 else 0
                        
                        db.add_trade(
                            symbol=symbol,
                            side='sell',
                            price=filled_price,
                            quantity=filled,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            order_id=order['id'],
                            exchange_id=order['id'],
                            timestamp=datetime.now().isoformat()
                        )
                    
                    # Удаляем позицию после продажи
                    if symbol in self.positions:
                        del self.positions[symbol]
                        db.remove_position(symbol)
                        pnl_str = f", PnL=${pnl:.2f} ({pnl_pct:+.2f}%)" if pnl != 0 else ""
                        logger.info(f"📤 Позиция {symbol} продана{pnl_str}")
                
                return order
            else:
                return None
            
        except Exception as e:
            logger.error(f"Ошибка выполнения сделки: {e}")
            return None
    
    def clean_stale_orders(self):
        """Отменяет лимитные ордера, висящие дольше таймаута (sell 300c, buy 60c)."""
        try:
            orders = self.exchange.fetchOpenOrders()
            if orders:
                now = time.time()
                for o in orders:
                    age = now - (o['timestamp'] / 1000)
                    # Sell-ордера: живут дольше (5 мин), чтобы не отменять трейлинг/TP
                    if o['side'] == 'sell':
                        if age > 300:
                            self.exchange.cancelOrder(o['id'], o['symbol'])
                            logger.info(f"🧹 Отменён sell-ордер ({age:.0f}с): {o['symbol']} {o['amount']} @ ${o['price']}")
                    else:
                        if age > 60:
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
                    for symbol in list(self.positions.keys()):
                        currency = symbol.split('/')[0]
                        real_qty = balance['total'].get(currency, 0)
                        mem_qty = self.positions[symbol]['quantity']
                        real_value = real_qty * (balance['free'].get(currency, 0) or 0)
                        
                        # Если на бирже нет актива — удаляем из памяти
                        if real_qty < 0.000001:
                            logger.warning(f"🔄 Синхронизация: {symbol} нет на бирже. Удаляю из кеша/БД.")
                            del self.positions[symbol]
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
                
                try:
                    from data_cache import get_fetcher, get_price
                    fetcher = get_fetcher()
                    if fetcher:
                        # BTC цена из кеша (или API с троттлингом 3с)
                        btc_price = fetcher.get_ticker('BTC/USDT')
                        
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
                    except:
                        pass
                
                # Инициализация DecisionEngine (синглтон)
                from decision_engine import DecisionEngine
                de = DecisionEngine()
                
                # BTC-корреляция для DE
                if btc_price is not None:
                    de.set_multi_tf_data(btc_price=btc_price)
                    de.update_btc_reference(btc_price)
                
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
                                score = decision.score or 50
                                # Адаптивный размер позиции: от DE или стандартный
                                size_mult = decision.position_size or 0.5
                                
                                # Лимиты безопасности
                                max_positions = self.config['risk_management'].get('max_open_positions', 3)
                                if len(self.positions) >= max_positions:
                                    logger.info(f"⚠️ [DE] {symbol}: лимит {len(self.positions)}/{max_positions}")
                                    continue
                                
                                logger.info(f"⚡ [DE→BUY] {symbol}: score={score:.0f}/100 size={size_mult:.2f} | ${de_price:.4f}")
                                # Размер позиции = стандартный * size_mult от DE, с учётом Score
                                base_qty = self.calculate_position_size(symbol, de_price, score)
                                quantity = base_qty * size_mult
                                if quantity * de_price <= self.available_capital:
                                    # ✅ ЗАПОМИНАЕМ SL/TP УРОВНИ ОТ DE в позиции
                                    sl_price = decision.sl_price
                                    tp_price = decision.tp_price
                                    trail_act = decision.trail_act
                                    trail_dist = decision.trail_dist
                                    max_hold_h = decision.max_hold_h
                                    self.execute_trade(symbol, 'buy', quantity, de_price)
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
                        pos_time = datetime.fromisoformat(position['entry_time'])
                        if pos_time.tzinfo is not None:
                            pos_time = pos_time.replace(tzinfo=None)
                        hold_time = (datetime.now() - pos_time).total_seconds() / 3600
                        hold_hours = hold_time
                        
                        # ✅ ЕДИНЫЙ ЦЕНТР РЕШЕНИЙ: DE решает когда выходить
                        entry_price = position['entry_price']
                        pnl_percent = (current_price - entry_price) / entry_price * 100
                        
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
                        
                        exit_decision = de.decide_exit(
                            symbol, entry_price, current_price,
                            highest_price=current_price * (1 + position['max_profit'] / 100),
                            lowest_price=current_price * (1 + min(0, pnl_percent) / 100),
                            entry_time=pos_time, pnl_pct=pnl_percent,
                            sl_pct=rp['sl_pct'], tp_pct=rp['tp_pct'],
                            trail_act=rp['trail_act'], trail_dist=rp['trail_dist'],
                            max_hold_hours=max_hold, side='long'
                        )
                        
                        should_sell = (exit_decision is not None)
                        sell_reason = exit_decision.reason if exit_decision else ""
                        
                        if not should_sell:
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
                                    del self.positions[symbol]
                                db.remove_position(symbol)
                                continue
                            
                            # Мусорный остаток (< $1) — не продаём, просто чистим кеш
                            MIN_POSITION_VALUE = 1.0
                            if pos_value < MIN_POSITION_VALUE:
                                logger.warning(f"⏭️ {symbol}: остаток ${pos_value:.2f} < ${MIN_POSITION_VALUE:.2f}. Чищу кеш без продажи.")
                                if symbol in self.positions:
                                    del self.positions[symbol]
                                db.remove_position(symbol)
                                continue
                            
                            # Есть монеты — продаём через биржу
                            safe_qty = free_asset * 0.999 if free_asset < quantity else quantity
                            if safe_qty < quantity * 0.9:
                                logger.warning(f"⚠️ {symbol}: free={free_asset:.6f} < qty={quantity:.6f}. Использую free.")
                            
                            result = self.execute_trade(symbol, 'sell', safe_qty, current_price)
                            
                            # 🧠 ML: обучаем на результате сделки
                            if result is not None and symbol in self.positions:
                                try:
                                    from ml_advisor import ml_add_result, ml_train
                                    pos = self.positions[symbol]
                                    ml_add_result(symbol, pos['entry_price'], current_price,
                                                  analysis['rsi'] if 'analysis' in dir() else 50,
                                                  analysis['trend'] if 'analysis' in dir() else 'neutral',
                                                  analysis['confidence'] if 'analysis' in dir() else 0.5,
                                                  0.1, sell_reason)
                                    ml_train()
                                except Exception as e:
                                    logger.warning(f"ML обучение: {e}")
                            
                            # 🛡️ Если sell не удался — принудительно удаляем позицию
                            if result is None and symbol in self.positions:
                                logger.warning(f"⚠️ Sell {symbol} не удался. Принудительно очищаю позицию из памяти.")
                                del self.positions[symbol]
                                db.remove_position(symbol)
                
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