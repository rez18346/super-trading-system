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
                            
                            # Пытаемся получить цену сначала из БД (trade_history, weighted average)
                            db_entry_price, _ = db.calculate_weighted_entry(symbol)
                            
                            if db_entry_price > 0:
                                entry_price = db_entry_price
                                # Берём время первой покупки из trade_history
                                trades = db.get_trade_history(symbol, limit=1, side='buy')
                                entry_time = trades[0]['timestamp'] if trades else datetime.now().isoformat()
                                logger.info(f"   📊 Weighted average из истории: {symbol}: ${entry_price:.4f}")
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
    
    def _confirm_15m_reversal(self, symbol: str) -> bool:
        """
        Профессиональная проверка: не входить на падающем 15М графике.
        Ждём подтверждение разворота — минимум 2 последовательные зелёные свечи
        или чёткий отскок от минимума.
        """
        try:
            df_15m = self.get_market_data(symbol, '15m', limit=5)
            if df_15m.empty or len(df_15m) < 4:
                return True  # Нет данных — пропускаем проверку
            
            closes = df_15m['close'].values
            opens = df_15m['open'].values
            
            last_3_candles = list(zip(opens[-3:], closes[-3:]))
            
            # Считаем: все красные (close < open) или нет
            red_count = sum(1 for o, c in last_3_candles if c < o)
            # Падение в процентах от первой к последней
            drop_pct = (closes[-4] - closes[-1]) / closes[-4] * 100 if len(closes) >= 4 else 0
            
            # Проверка 1: последняя свеча зелёная и выше предпоследней?
            last_green = closes[-1] > opens[-1]
            last_higher = len(closes) >= 2 and closes[-1] > closes[-2]
            
            # Проверка 2: две подряд зелёные?
            two_green = len(closes) >= 2 and closes[-1] > opens[-1] and closes[-2] > opens[-2]
            
            # Проверка 3: падение > 1.5% за 3 свечи?
            sharp_drop = drop_pct > 1.5
            
            # 🎯 ЛОГИКА РЕШЕНИЯ:
            # - Если падение резкое (>1.5%) — ждём две зелёные свечи
            # - Если падение слабое — достаточно одной зелёной и роста
            # - Если все свечи красные — не входим
            
            if red_count >= 2 and not last_green:
                # Две и более красных, последняя тоже красная — падающий нож
                logger.info(f"📉 {symbol}: 15М падающий нож ({red_count} красных, падение {drop_pct:.1f}%)")
                return False
            
            if sharp_drop and not two_green:
                # Резкое падение, но ещё нет двух зелёных — ждём
                logger.info(f"📉 {symbol}: 15М резкое падение ({drop_pct:.1f}%), жду 2 зелёные")
                return False
            
            if last_green and last_higher:
                logger.debug(f"✅ {symbol}: 15М разворот подтверждён ({drop_pct:.1f}% падения, зелёная)")
            
            return True
            
        except Exception as e:
            logger.warning(f"15М проверка {symbol}: ошибка ({e}), пропускаю")
            return True
    
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
    
    def calculate_position_size(self, symbol: str, price: float) -> float:
        """Расчет размера позиции"""
        risk_config = self.config['risk_management']
        trading_config = self.config['trading']
        
        # Максимальный размер позиции в процентах от капитала
        max_position_usd = min(
            risk_config.get('max_buy_order_usd', 2.0),  # Используем max_buy_order_usd
            self.available_capital * (trading_config['max_position_size_percent'] / 100)
        )
        
        # Минимальный размер позиции
        min_position_usd = risk_config['min_position_usd']
        
        # Расчет на основе доступного капитала и риска
        position_size = min(max_position_usd, self.available_capital * 0.2)
        position_size = max(min_position_usd, position_size)
        
        # Конвертация в количество единиц актива
        quantity = position_size / price
        
        logger.info(f"Размер позиции для {symbol}: ${position_size:.2f} ({quantity:.6f} units)")
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
                        'side': 'long'
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
                    type='limit',
                    side=side,
                    amount=quantity,
                    price=price
                )
                
                logger.info(f"✅ РЕАЛЬНАЯ СДЕЛКА ВЫПОЛНЕНА: ID {order['id']}")
                
                # ОБНОВЛЯЕМ ПОЗИЦИИ ПОСЛЕ УСПЕШНОЙ СДЕЛКИ
                if side == 'buy':
                    order_price = order.get('price', price) or price
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
                    
                    # Пересчитываем weighted average из истории
                    wa_price, wa_qty = db.calculate_weighted_entry(symbol)
                    actual_price = wa_price if wa_price > 0 else order_price
                    actual_qty = wa_qty if wa_qty > 0 else filled_qty
                    
                    # Добавляем новую позицию
                    if symbol not in self.positions:
                        self.positions[symbol] = {
                            'quantity': actual_qty,
                            'entry_price': actual_price,
                            'entry_time': datetime.now().isoformat(),
                            'max_profit': 0.0
                        }
                        logger.info(f"📥 Добавлена новая позиция: {symbol} - {actual_qty:.6f} @ ${actual_price:.4f}")
                    else:
                        self.positions[symbol]['quantity'] = actual_qty
                        self.positions[symbol]['entry_price'] = actual_price
                        self.positions[symbol]['entry_time'] = datetime.now().isoformat()
                        logger.info(f"📊 Обновлена позиция {symbol}: {actual_qty:.6f} @ ${actual_price:.4f}")
                elif side == 'sell':
                    # Запись sell в trade_history
                    pnl = 0
                    pnl_pct = 0
                    if symbol in self.positions:
                        entry = self.positions[symbol]['entry_price']
                        filled = order.get('filled', quantity) or quantity
                        filled_price = order.get('price', price) or price
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
                
                # Анализ каждого символа
                for symbol in symbols:
                    if not self.running:
                        break
                    
                    # ⚡ ПРОВЕРКА DecisionEngine (раз в 5 циклов на символ)
                    # Если DE одобрил — устанавливаем _de_force_entry для исполнения
                    if symbol not in self.positions:
                        de_check_key = f'_de_last_check_{symbol}'
                        de_interval = getattr(self, '_de_check_interval', 5)
                        if not hasattr(self, de_check_key):
                            setattr(self, de_check_key, 0)
                        setattr(self, de_check_key, getattr(self, de_check_key) + 1)
                        
                        if getattr(self, de_check_key) >= de_interval:
                            setattr(self, de_check_key, 0)
                            try:
                                from decision_engine import DecisionEngine
                                de = DecisionEngine()
                                # Берём цену из кеша
                                try:
                                    from data_cache import get_fetcher
                                    fetcher = get_fetcher()
                                    de_price = fetcher.get_ticker(symbol) if fetcher else None
                                except:
                                    de_price = None
                                if de_price:
                                    decision = de.decide_entry(symbol, de_price)
                                    if decision.get('approved', False):
                                        if not hasattr(self, '_de_force_entry'):
                                            self._de_force_entry = {}
                                        self._de_force_entry[symbol] = {
                                            'price': de_price,
                                            'score': decision.get('score', 0),
                                            'ts': time.time(),
                                        }
                                        logger.info(f"🎯 [DE] ВХОД {symbol}: {decision.get('score', 0):.0f}/100")
                            except Exception as e_de:
                                logger.debug(f"[DE] ошибка: {e_de}")
                    
                    # Получение и анализ данных
                    # Загрузка мультитаймфреймовых данных для ML-PRO v2
                    df = self.get_market_data(symbol, '5m', 100)
                    if df.empty:
                        continue
                    
                    df = self.calculate_indicators(df)
                    analysis = self.analyze_trend(df)
                    
                    current_price = df['close'].iloc[-1]
                    
                    # Свечи старших таймфреймов для ML-PRO v2
                    try:
                        df_1h = self.get_market_data(symbol, '1h', 100)
                        df_4h = self.get_market_data(symbol, '4h', 50)
                        if not df_1h.empty:
                            candles_1h = [{'o':r['open'],'h':r['high'],'l':r['low'],'c':r['close'],'v':r['volume'],'t':r.name.timestamp()*1000 if hasattr(r.name,'timestamp') else int(r.name)} for _,r in df_1h.iterrows()]
                        else:
                            candles_1h = []
                        if not df_4h.empty:
                            candles_4h = [{'o':r['open'],'h':r['high'],'l':r['low'],'c':r['close'],'v':r['volume'],'t':r.name.timestamp()*1000 if hasattr(r.name,'timestamp') else int(r.name)} for _,r in df_4h.iterrows()]
                        else:
                            candles_4h = []
                    except Exception as e:
                        logger.warning(f"Не загрузились 1H/4H для {symbol}: {e}")
                        candles_1h = []
                        candles_4h = []
                    
                    # Логирование анализа
                    logger.info(f"{symbol}: Цена=${current_price:.2f}, Тренд={analysis['trend']}, "
                              f"Уверенность={analysis['confidence']:.2%}, RSI={analysis['rsi']:.1f}")
                    
                    # ─── ТОРГОВАЯ ЛОГИКА ──────────────────────────────────────
                    
                    # ⚡ ПРИОРИТЕТ 1: DecisionEngine (ML-ансамбль)
                    # Если DE одобрил вход — входим сразу, без старых фильтров
                    if symbol not in self.positions:
                        de_force = getattr(self, '_de_force_entry', None)
                        if de_force and symbol in de_force:
                            entry = de_force.pop(symbol)  # забираем и удаляем
                            force_price = entry.get('price', current_price)
                            force_score = entry.get('score', 0)
                            
                            # Лимиты позиций всё равно проверяем (безопасность)
                            max_positions = self.config['risk_management'].get('max_open_positions', 3)
                            if len(self.positions) >= max_positions:
                                logger.info(f"⚠️  [DE] {symbol}: лимит позиций {len(self.positions)}/{max_positions}")
                                continue
                            
                            # Проверка лимита по паре (макс 30% портфеля)
                            try:
                                bal = self.exchange.fetch_balance()
                                total_port = 0
                                for asset, amt in bal['total'].items():
                                    if asset != 'USDT' and amt > 0:
                                        try:
                                            t = self.exchange.fetch_ticker(f"{asset}/USDT")
                                            total_port += amt * t['last']
                                        except:
                                            pass
                                total_port += bal['total'].get('USDT', 0)
                                if total_port > 0:
                                    pair_pct = (100 * self.config['risk_management'].get('max_order_value', 2.0)) / total_port
                                else:
                                    pair_pct = 0
                            except:
                                pair_pct = 0
                            
                            logger.info(f"⚡ [DE→EXEC] {symbol}: score={force_score:.0f}/100 | цена={force_price:.4f}")
                            
                            quantity = self.calculate_position_size(symbol, force_price)
                            if quantity * force_price <= self.available_capital:
                                self.execute_trade(symbol, 'buy', quantity, force_price)
                            else:
                                logger.warning(f"⏭️ [DE] {symbol}: недостаточно капитала")
                            continue  # пропускаем старую логику для этого символа
                    
                    # ⚠️  ПРИОРИТЕТ 2: старая логика трейдера (fallback, если DE молчит)
                    if symbol not in self.positions:
                        # ПРОВЕРКА ЛИМИТОВ ПОЗИЦИЙ ПЕРЕД ПОКУПКОЙ
                        # 1. Проверка максимального количества открытых позиций
                        max_positions = self.config['risk_management'].get('max_open_positions', 3)
                        if len(self.positions) >= max_positions:
                            logger.info(f"⚠️  Достигнут лимит открытых позиций: {len(self.positions)}/{max_positions}")
                            continue
                        
                        # 2. Проверка диверсификации (не более 30% в одну пару)
                        try:
                            balance = self.exchange.fetch_balance()
                            total_portfolio = 0
                            for asset, amount in balance['total'].items():
                                if asset != 'USDT' and amount > 0:
                                    try:
                                        ticker = self.exchange.fetch_ticker(f"{asset}/USDT")
                                        total_portfolio += amount * ticker['last']
                                    except:
                                        pass
                            total_portfolio += balance['total'].get('USDT', 0)
                            
                            # Проверяем текущую позицию в этой паре (если есть)
                            current_pair_value = 0
                            for pos_symbol, position in self.positions.items():
                                if pos_symbol == symbol:
                                    current_pair_value = position['quantity'] * current_price
                                    break
                            
                            max_pair_percent = 30  # Макс 30% в одну пару
                            if total_portfolio > 0:
                                pair_percent = (current_pair_value / total_portfolio) * 100
                                if pair_percent >= max_pair_percent:
                                    logger.info(f"⚠️  Превышен лимит по паре {symbol}: {pair_percent:.1f}% > {max_pair_percent}%")
                                    continue
                        except Exception as e:
                            logger.warning(f"Не удалось проверить диверсификацию: {e}")
                        
                        # Сигнал на покупку
                        if (analysis['trend'] == 'bullish' and 
                            analysis['confidence'] > 0.6 and  # СНИЖЕНО С 70% ДО 60%
                            analysis['rsi'] < 70):
                            
                            # ПРОВЕРКА: Уже есть открытая позиция?
                            if symbol in self.positions:
                                logger.info(f"⚠️  Пропускаем покупку {symbol}: позиция уже открыта")
                                continue
                            
                            # 🔍 ЗАЩИТА ОТ ПОКУПКИ НА ХАЕ: не покупаем если цена выросла >2% за последние 4ч
                            try:
                                df_15m = self.get_market_data(symbol, '15m', 16)  # 4 часа по 15 мин
                                if not df_15m.empty:
                                    closes_15m = df_15m['close'].values
                                    price_change_4h = (closes_15m[-1] - closes_15m[0]) / closes_15m[0] * 100
                                    if price_change_4h > 2.0:
                                        logger.info(f"⏸️ {symbol}: пропускаем — цена выросла +{price_change_4h:.1f}% за 4ч (лимит 2%)")
                                        continue
                                    if price_change_4h > 1.5:
                                        logger.info(f"⚠️ {symbol}: цена выросла +{price_change_4h:.1f}% за 4ч, близко к хаю")
                            except Exception as e:
                                logger.warning(f"Не удалось проверить 15M разогрев {symbol}: {e}")
                            
                            # 🧠 ML-PRO v2 СОВЕТНИК: мультитаймфреймовый анализ
                            ml_decision = {'decision': 'GOOD'}  # дефолт, если ML недоступен
                            try:
                                from ml_professional_v2 import ml_pro_v2_evaluate
                                ml_decision, ml_prob, ml_features = ml_pro_v2_evaluate(
                                    symbol,
                                    df if hasattr(df, '__iter__') else [],
                                    candles_1h if len(candles_1h) > 0 else [],
                                    candles_4h if len(candles_4h) > 0 else [],
                                    analysis['confidence'],
                                    analysis['trend'],
                                    analysis['rsi']
                                )
                                if ml_decision == 'SKIP':
                                    logger.info(f"⏭️ {symbol}: ML-PRO v2 отклонил сигнал (prob={ml_prob:.3f})")
                                    continue
                                elif ml_decision == 'WEAK_BUY':
                                    logger.info(f"⚠️ {symbol}: ML-PRO v2 осторожно (prob={ml_prob:.3f})")
                                    continue
                                else:
                                    logger.info(f"✅ {symbol}: ML-PRO v2 одобрил (prob={ml_prob:.3f})")
                            except Exception as e:
                                logger.warning(f"ML-PRO v2 недоступен ({e}), использую old ML")
                                try:
                                    from ml_advisor import ml_evaluate
                                    ml_result = ml_evaluate(symbol, current_price, analysis['rsi'],
                                                            analysis['trend'], analysis['confidence'],
                                                            df)
                                    if ml_result['decision'] == 'SKIP':
                                        logger.info(f"⏭️ {symbol}: old ML отклонил сигнал ({ml_result['reason']})")
                                        continue
                                except Exception as e2:
                                    logger.warning(f"old ML тоже не сработал: {e2}")
                            
                            # 🎯 ПРОВЕРКА: не покупаем на падающем 15М графике
                            if not self._confirm_15m_reversal(symbol):
                                logger.info(f"⏭️ {symbol}: 15М-подтверждение не получено (падающий нож)")
                                continue
                            
                            quantity = self.calculate_position_size(symbol, current_price)
                            if quantity * current_price <= self.available_capital:
                                self.execute_trade(symbol, 'buy', quantity, current_price)
                    
                    else:
                        # Управление открытой позицией
                        position = self.positions[symbol]
                        pos_time = datetime.fromisoformat(position['entry_time'])
                        if pos_time.tzinfo is not None:
                            pos_time = pos_time.replace(tzinfo=None)
                        hold_time = (datetime.now() - pos_time).total_seconds() / 3600
                        
                        # Проверка стоп-лосса и тейк-профита
                        risk_config = self.config['risk_management']
                        entry_price = position['entry_price']
                        pnl_percent = (current_price - entry_price) / entry_price * 100
                        
                        stop_loss = -risk_config['stop_loss_percent']
                        take_profit = risk_config['take_profit_percent']
                        
                        # ✅ ПРОФЕССИОНАЛЬНЫЙ ВЫХОД: только по объективным причинам
                        # 1. Стоп-лосс: рынок против
                        # 2. Фиксированный тейк-профит: цель достигнута
                        # 3. Трейлинг: защита прибыли
                        # 4. Только при 48 часах — принудительный выход (чтобы не блокировать слоты вечно)
                        trailing_config = risk_config.get('trailing_take_profit', {})
                        trailing_activation = trailing_config.get('activation_percent', 3.0)
                        trailing_step = trailing_config.get('trailing_percent', 2.5)
                        
                        should_sell = False
                        sell_reason = ""
                        
                        if pnl_percent <= stop_loss:
                            should_sell = True
                            sell_reason = "стоп-лосс"
                        elif pnl_percent >= take_profit:
                            # Достигли цели — фиксируем прибыль
                            should_sell = True
                            sell_reason = "тейк-профит"
                        elif hold_time > 48:
                            # 48 часов — единственный таймер. Освобождаем слот.
                            should_sell = True
                            sell_reason = "лимит удержания (48ч)"
                        else:
                            # Трейлинг-стоп (всегда активен)
                            if 'max_profit' not in position:
                                position['max_profit'] = pnl_percent
                            else:
                                position['max_profit'] = max(position['max_profit'], pnl_percent)
                            
                            if position['max_profit'] >= trailing_activation:
                                trailing_trigger = position['max_profit'] - trailing_step
                                if pnl_percent <= trailing_trigger:
                                    should_sell = True
                                    sell_reason = f"трейлинг (пик: {position['max_profit']:.1f}%, откат: {pnl_percent:.1f}%)"
                                elif pnl_percent >= trailing_activation * 0.5:
                                    # Логируем трекинг только когда есть смысл
                                    logger.info(f"   🔄 Трейлинг {symbol}: пик={position['max_profit']:.1f}%, тек={pnl_percent:.1f}%, триггер={trailing_trigger:.1f}%")
                        
                        # Если не продали — просто логируем статус (раз в N циклов)
                        if not should_sell and abs(pnl_percent) > 1:
                            logger.info(f"📊 {symbol}: PnL={pnl_percent:+.2f}%, время={hold_time:.0f}ч, трейд={position.get('max_profit', 0):.1f}% пик")
                            activation_percent = trailing_config['activation_percent']
                            trailing_percent = trailing_config['trailing_percent']
                            
                            # Инициализируем максимум прибыли если нужно
                            if 'max_profit' not in position:
                                position['max_profit'] = pnl_percent
                            else:
                                position['max_profit'] = max(position['max_profit'], pnl_percent)
                            
                            # Проверяем активацию скользящего тейк-профита
                            if position['max_profit'] >= activation_percent:
                                trailing_take_profit = position['max_profit'] - trailing_percent
                                
                                if pnl_percent <= trailing_take_profit:
                                    should_sell = True
                                    sell_reason = f"скользящий тейк-профит (макс: {position['max_profit']:.2f}%, текущий: {pnl_percent:.2f}%)"
                                else:
                                    # Логируем отслеживание
                                    logger.info(f"   📈 Отслеживание прибыли {symbol}: макс={position['max_profit']:.2f}%, текущий={pnl_percent:.2f}%, тейк-профит={trailing_take_profit:.2f}%")
                            else:
                                # Используем фиксированный тейк-профит до активации
                                if pnl_percent >= take_profit:
                                    should_sell = True
                                    sell_reason = "фиксированный тейк-профит"
                        else:
                            # Используем фиксированный тейк-профит если скользящий отключен
                            if pnl_percent >= take_profit:
                                should_sell = True
                                sell_reason = "фиксированный тейк-профит"
                        
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