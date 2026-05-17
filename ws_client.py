#!/usr/bin/env python3
"""
ws_client.py — WebSocket-клиент Bybit V5 (public каналы).

Профессиональные принципы:
  1. Авто-переподключение при разрыве (exponential backoff: 1s → 2s → 4s → 8s → max 60s)
  2. Ping/Pong каждые 20 секунд (по спецификации Bybit)
  3. Fallback на REST при недоступности WebSocket
  4. Пишет напрямую в PriceCache / OHLCVCache (data_cache.py)
  5. Один поток на всё — не блокирует main
  6. Мониторинг: считает пропущенные сообщения

Каналы:
  - tickers.{symbol} — цена, bid/ask (spot)
  - kline.{interval}.{symbol} — свечи (spot)

Совместимость:
  - Работает с существующей системой без единой правки в trader/monitor
  - CachedDataFetcher сам перестанет ходить на REST (кеш будет всегда свежий)
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
sys.path.insert(0, BASE_DIR)

import json
import time
import logging
import threading
import ssl
import csv
import pandas as pd
from typing import Dict, List, Optional, Callable, Set
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger('ws_client')

try:
    import websocket
except ImportError:
    logger.warning("websocket-client не установлен. Установи: pip install websocket-client")
    websocket = None

from data_cache import PriceCache, OHLCVCache, get_fetcher


class ConnectionState(Enum):
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    RECONNECTING = 3
    FALLBACK = 4  # WebSocket недоступен, используем REST


class BybitWebSocketClient:
    """
    WebSocket-клиент Bybit V5 для public каналов (spot).
    
    Запускается в отдельном потоке.
    Наполняет PriceCache (цены) и OHLCVCache (свечи).
    При разрыве — автоматическое переподключение с exponential backoff.
    
    Использование:
        client = BybitWebSocketClient()
        client.start()  # запускает поток
        # ... работа системы ...
        client.stop()   # остановка
    """
    
    SPOT_URL = "wss://stream.bybit.com/v5/public/spot"
    PING_INTERVAL = 20  # секунд
    MAX_BACKOFF = 60    # максимальная задержка перед переподключением (сек)
    
    def __init__(self, symbols: List[str] = None,
                 price_cache: PriceCache = None,
                 ohlcv_cache: OHLCVCache = None,
                 kline_intervals: List[str] = None,
                 cvd_collector: Optional['CVDCollector'] = None):
        """
        Args:
            symbols: список символов (SOLUSDT, BTCUSDT и т.д.)
            price_cache: экземпляр PriceCache (или None — создаст сам)
            ohlcv_cache: экземпляр OHLCVCache (или None — создаст сам)
            kline_intervals: таймфреймы для свечей (['60', '240'])
            cvd_collector: экземпляр CVDCollector (или None)
        """
        self.symbols = symbols or []
        self.price_cache = price_cache or PriceCache()
        self.ohlcv_cache = ohlcv_cache or OHLCVCache()
        self.kline_intervals = kline_intervals or ['60', '240']  # 1H, 4H
        self.cvd_collector: Optional['CVDCollector'] = cvd_collector
        
        # Состояние соединения
        self.state = ConnectionState.DISCONNECTED
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._reconnect_count = 0
        self._last_pong = 0
        
        # Статистика
        self.messages_received = 0
        self.messages_skipped = 0
        self.reconnects = 0
        self._last_message_time = 0
        
        # Callback на смену состояния
        self._on_state_change: Optional[Callable] = None
        
        # Auto-restore connection checker
        self._check_timer: Optional[threading.Timer] = None
    
    # ─── ПУБЛИЧНЫЙ API ──────────────────────────────────────────────────
    
    def start(self) -> bool:
        """Запустить WebSocket-клиент в отдельном потоке."""
        if not websocket:
            logger.error("❌ websocket-client не установлен. pip install websocket-client")
            self.state = ConnectionState.FALLBACK
            return False
        
        if self._running:
            logger.warning("WebSocket уже запущен")
            return True
        
        self._running = True
        self._reconnect_count = 0
        self.state = ConnectionState.CONNECTING
        self._on_state_change_cb(self.state)
        
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                         name='ws-client')
        self._thread.start()
        logger.info(f"🌐 WebSocket клиент запущен ({len(self.symbols)} символов)")
        return True
    
    def stop(self):
        """Остановить WebSocket-клиент."""
        self._running = False
        if self._ws:
            self._ws.close()
            self._ws = None
        self.state = ConnectionState.DISCONNECTED
        self._on_state_change_cb(self.state)
        logger.info("🛑 WebSocket клиент остановлен")
    
    def add_symbols(self, symbols: List[str]):
        """Добавить символы для подписки (на лету)."""
        new_symbols = [s for s in symbols if s not in self.symbols]
        if not new_symbols:
            return
        self.symbols.extend(new_symbols)
        if self.state == ConnectionState.CONNECTED and self._ws:
            self._subscribe(new_symbols)
            logger.info(f"➕ Подписан на {len(new_symbols)} новых символов")
    
    def remove_symbols(self, symbols: List[str]):
        """Отписаться от символов."""
        args = self._build_ticker_args(symbols) + self._build_kline_args(symbols)
        if args and self._ws:
            self._ws.send(json.dumps({"op": "unsubscribe", "args": args}))
        self.symbols = [s for s in self.symbols if s not in symbols]
    
    def set_on_state_change(self, callback: Callable):
        """Callback при смене состояния (для мониторинга)."""
        self._on_state_change = callback
    
    def get_state(self) -> Dict:
        """Вернуть состояние клиента."""
        now = time.time()
        return {
            'state': self.state.name,
            'symbols': len(self.symbols),
            'messages_received': self.messages_received,
            'messages_skipped': self.messages_skipped,
            'reconnects': self.reconnects,
            'last_message_ago': f"{now - self._last_message_time:.1f}s" if self._last_message_time else 'never',
            'last_pong_ago': f"{now - self._last_pong:.1f}s" if self._last_pong else 'never',
        }
    
    # ─── ВНУТРЕННЯЯ ЛОГИКА ─────────────────────────────────────────────
    
    def _run_loop(self):
        """Главный цикл соединения (работает в отдельном потоке)."""
        while self._running:
            try:
                self.state = ConnectionState.CONNECTING
                self._on_state_change_cb(self.state)
                
                self._ws = websocket.WebSocketApp(
                    self.SPOT_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_ping=self._on_ping,
                    on_pong=self._on_pong,
                )
                
                # Запускаем соединение (блокирующий вызов)
                # run_forever сама вызывает on_open/on_close/on_error
                self._ws.run_forever(
                    ping_interval=self.PING_INTERVAL,
                    ping_payload="ping",
                    sslopt={"cert_reqs": ssl.CERT_NONE},
                    reconnect=0,  # не встроенный reconnect — делаем сами
                )
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
            
            if not self._running:
                break
            
            # Exponential backoff перед переподключением
            delay = min(2 ** self._reconnect_count, self.MAX_BACKOFF)
            self._reconnect_count += 1
            self.reconnects += 1
            self.state = ConnectionState.RECONNECTING
            
            logger.warning(f"🔄 Переподключение через {delay:.0f}с (попытка {self._reconnect_count})")
            self._on_state_change_cb(self.state)
            
            time.sleep(delay)
        
        self.state = ConnectionState.DISCONNECTED
        self._on_state_change_cb(self.state)
    
    def _on_open(self, ws):
        """Соединение установлено → подписываемся."""
        logger.info("✅ WebSocket соединение установлено")
        self.state = ConnectionState.CONNECTED
        self._reconnect_count = 0
        self._on_state_change_cb(self.state)
        
        # Подписываемся на все каналы
        self._subscribe_all(ws)
    
    def _on_message(self, ws, message: str):
        """Получено сообщение от WebSocket."""
        self.messages_received += 1
        self._last_message_time = time.time()
        
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            self.messages_skipped += 1
            return
        
        # Pong
        if data.get('op') == 'pong':
            self._last_pong = time.time()
            return
        
        # Subscription response
        if data.get('op') == 'subscribe':
            ret = data.get('ret_msg', '')
            if data.get('success'):
                logger.debug(f"📡 Подписка успешна: {ret}")
            else:
                logger.warning(f"⚠️ Ошибка подписки: {ret}")
            return
        
        # Тематическое сообщение (ticker / kline / trade)
        topic = data.get('topic', '')
        if topic.startswith('tickers.'):
            self._handle_ticker(data)
        elif topic.startswith('kline.'):
            self._handle_kline(data)
        elif topic.startswith('publicTrade.'):
            self._handle_trade(data)
    
    def _on_error(self, ws, error):
        """Ошибка WebSocket."""
        logger.error(f"⚠️ WebSocket ошибка: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Соединение закрыто."""
        logger.warning(f"🔌 WebSocket закрыт: {close_status_code} {close_msg}")
        self.state = ConnectionState.DISCONNECTED
        self._ws = None
    
    def _on_ping(self, ws, message):
        """Ping от сервера — отвечаем pong."""
        ws.send(json.dumps({"op": "pong"}))
    
    def _on_pong(self, ws, message):
        """Pong от сервера."""
        self._last_pong = time.time()
    
    # ─── ПОДПИСКА ───────────────────────────────────────────────────────
    
    def _subscribe_all(self, ws):
        """Подписаться на все каналы."""
        args = self._build_ticker_args() + self._build_kline_args()
        
        # Добавляем publicTrade для CVD, если есть коллектор
        if self.cvd_collector:
            trade_args = ['publicTrade.BTCUSDT']
            args += trade_args
        
        if not args:
            logger.warning("Нет символов для подписки")
            return
        
        # Bybit лимит: не более 10 args на сообщение для spot
        batch_size = 10
        for i in range(0, len(args), batch_size):
            batch = args[i:i + batch_size]
            ws.send(json.dumps({
                "op": "subscribe",
                "args": batch
            }))
        
        trade_count = len(trade_args) if self.cvd_collector else 0
        logger.info(f"📡 Подписан на {len(self.symbols)} ticker'ов + "
                     f"{len(self.symbols) * len(self.kline_intervals)} kline'ов + "
                     f"{trade_count} trade(arg)")
    
    def _subscribe(self, symbols: List[str]):
        """Подписаться на новые символы."""
        if not self._ws:
            return
        args = self._build_ticker_args(symbols) + self._build_kline_args(symbols)
        if args:
            for i in range(0, len(args), 10):
                batch = args[i:i + 10]
                self._ws.send(json.dumps({"op": "subscribe", "args": batch}))
    
    def _build_ticker_args(self, symbols: List[str] = None) -> List[str]:
        """Построить список args для подписки на ticker."""
        syms = symbols or self.symbols
        return [f"tickers.{s}" for s in syms]
    
    def _build_kline_args(self, symbols: List[str] = None) -> List[str]:
        """Построить список args для подписки на kline."""
        syms = symbols or self.symbols
        args = []
        for interval in self.kline_intervals:
            for sym in syms:
                args.append(f"kline.{interval}.{sym}")
        return args
    
    # ─── ОБРАБОТКА СООБЩЕНИЙ ───────────────────────────────────────────
    
    def _handle_ticker(self, data: dict):
        """Обработать ticker сообщение."""
        ticker = data.get('data', {})
        symbol = ticker.get('symbol', '')
        if not symbol:
            return
        
        # Bybit spot использует SOLUSDT, нам нужно SOL/USDT
        normalized = self._normalize_symbol(symbol)
        
        last_price = self._safe_float(ticker.get('lastPrice'))
        bid = self._safe_float(ticker.get('bid1Price'))
        ask = self._safe_float(ticker.get('ask1Price'))
        
        if last_price is not None:
            self.price_cache.update_price(normalized, last_price, bid, ask)
    
    def _handle_kline(self, data: dict):
        """Обработать kline сообщение."""
        # В spot kline приходит в data[0] (массив с одним элементом)
        kline_data = data.get('data', [])
        if isinstance(kline_data, list) and len(kline_data) > 0:
            candle = kline_data[0]
        else:
            return
        
        # Извлекаем тему: kline.{interval}.{symbol}
        topic = data.get('topic', '')
        parts = topic.split('.')
        if len(parts) < 3:
            return
        interval = parts[1]
        symbol = parts[2]
        normalized = self._normalize_symbol(symbol)
        
        start = candle.get('start', 0)
        open_p = self._safe_float(candle.get('open'))
        high = self._safe_float(candle.get('high'))
        low = self._safe_float(candle.get('low'))
        close = self._safe_float(candle.get('close'))
        volume = self._safe_float(candle.get('volume'))
        confirm = candle.get('confirm', False)
        
        if open_p is None or close is None:
            return
        
        # Важно: Bybit присылает timestamp в ms, OHLCVCache хранит в ms
        candle_tuple = (start, open_p, high, low, close, volume)
        
        if confirm:
            # Свеча закрыта — записываем в кеш (как завершённую)
            # Для append_candle это новая свеча
            self.ohlcv_cache.append_candle(normalized, interval, candle_tuple)
        else:
            # Свеча ещё не закрыта — это обновление текущей свечи
            # Для append_candle последняя свеча с тем же start обновится
            self.ohlcv_cache.append_candle(normalized, interval, candle_tuple)
    
    # ─── CVD TRADE ─────────────────────────────────────────────────────
    
    def _handle_trade(self, data: dict):
        """Обработать publicTrade сообщение для CVD."""
        if not self.cvd_collector:
            return
        
        try:
            trades = data.get('data', [])
            if not isinstance(trades, list):
                trades = [trades]
            
            for trade in trades:
                if not isinstance(trade, dict):
                    continue
                # Bybit V5 format: T=trade_time_ms, S=Buy/Sell, p=price, v=volume
                ts = int(trade.get('T') or trade.get('timestamp') or 0)
                side = (trade.get('S') or trade.get('side') or 'Buy').upper()
                price = self._safe_float(trade.get('p') or trade.get('price'))
                volume = self._safe_float(trade.get('v') or trade.get('size') or trade.get('volume'))
                
                if price and volume and ts:
                    self.cvd_collector.add_trade(price, volume, side, ts)
        except Exception as e:
            logger.debug(f"[CVD] _handle_trade error: {e}")
    
    # ─── ВСПОМОГАТЕЛЬНЫЕ ───────────────────────────────────────────────
    
    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """SOLUSDT → SOL/USDT."""
        if '/' in symbol:
            return symbol
        # Bybit format: SOLUSDT → SOL/USDT
        if symbol.endswith('USDT'):
            return symbol.replace('USDT', '/USDT')
        if symbol.endswith('USDC'):
            return symbol.replace('USDC', '/USDC')
        if symbol.endswith('BTC'):
            return symbol.replace('BTC', '/BTC')
        return symbol
    
    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Безопасное преобразование в float."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _on_state_change_cb(self, state):
        """Вызвать callback смены состояния."""
        if self._on_state_change:
            try:
                self._on_state_change(state)
            except Exception as e:
                logger.debug(f"State callback error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# CVD Collector — Cumulative Volume Delta на 1-минутных агрегациях
# ═══════════════════════════════════════════════════════════════════════════

CVD_DATA_PATH = os.path.join(DATA_DIR, 'btc_cvd_1m.csv')


class CVDCollector:
    """
    Сбор и агрегация CVD (Cumulative Volume Delta) из publicTrade потока.
    
    Берёт сделки из WebSocket (цена, объём, сторона), агрегирует в 1-минутные
    корзины, сохраняет в CSV для последующей подачи в BTC модель.
    
    Данные доступны:
      - .get_recent(n) — последние n минут CVD
      - .save() — сброс на диск
      - .load() — загрузка с диска
    """
    
    def __init__(self, max_minutes: int = 1440):
        self.max_minutes = max_minutes  # 24 часа данных в памяти
        self.minutes: Dict[int, dict] = {}  # {unixtime_minute: {buy_vol, sell_vol, count}}
        self._last_save_time = 0
        self.load()
    
    def add_trade(self, price: float, volume: float, side: str, timestamp_ms: int):
        """Добавить сделку в корзину."""
        minute_key = (timestamp_ms // 60000) * 60  # округляем до минуты
        
        if minute_key not in self.minutes:
            self.minutes[minute_key] = {
                'ts': minute_key,
                'buy_vol': 0.0,
                'sell_vol': 0.0,
                'buy_usd': 0.0,
                'sell_usd': 0.0,
                'count': 0,
            }
        
        bucket = self.minutes[minute_key]
        volume_usd = price * volume
        
        if side.upper() == 'BUY':
            bucket['buy_vol'] += volume
            bucket['buy_usd'] += volume_usd
        else:
            bucket['sell_vol'] += volume
            bucket['sell_usd'] += volume_usd
        bucket['count'] += 1
        
        # Автосохранение каждые 60 секунд
        if time.time() - self._last_save_time > 60:
            self.save()
        
        # Ограничение размера кеша
        if len(self.minutes) > self.max_minutes * 1.1:
            keys = sorted(self.minutes.keys())
            for k in keys[:len(keys) - self.max_minutes]:
                del self.minutes[k]
    
    def get_cvd(self, minute_key: int) -> Optional[float]:
        """Вернуть чистый CVD (buy - sell) за указанную минуту."""
        bucket = self.minutes.get(minute_key)
        if not bucket:
            return None
        return bucket['buy_usd'] - bucket['sell_usd']
    
    def get_recent(self, n_minutes: int = 60) -> pd.DataFrame:
        """Вернуть последние n минут CVD как DataFrame.
        
        Колонки:
          - ts: timestamp (unix)
          - cvd_net: buy_usd - sell_usd (чистый CVD в USD)
          - cvd_ratio: buy_vol / sell_vol (отношение объёмов)
          - buy_vol, sell_vol: объёмы
          - buy_usd, sell_usd: объёмы в USD
          - trade_count: количество сделок
          - cvd_accum: накопленный CVD с начала сессии
        """
        keys = sorted(self.minutes.keys())
        if not keys:
            return pd.DataFrame()
        
        recent = [self.minutes[k] for k in keys[-n_minutes:]]
        df = pd.DataFrame(recent)
        if df.empty:
            return df
        
        df['cvd_net'] = df['buy_usd'] - df['sell_usd']
        df['cvd_ratio'] = df['buy_vol'] / (df['sell_vol'] + 1e-10)
        df['cvd_accum'] = df['cvd_net'].cumsum()
        df['ts_label'] = pd.to_datetime(df['ts'], unit='s', utc=True)
        return df
    
    def save(self):
        """Сохранить CVD данные в CSV."""
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            keys = sorted(self.minutes.keys())
            if not keys:
                return
            rows = []
            for k in keys:
                b = self.minutes[k]
                rows.append({
                    'ts': k,
                    'buy_vol': round(b['buy_vol'], 6),
                    'sell_vol': round(b['sell_vol'], 6),
                    'buy_usd': round(b['buy_usd'], 2),
                    'sell_usd': round(b['sell_usd'], 2),
                    'count': b['count'],
                })
            df = pd.DataFrame(rows)
            df.to_csv(CVD_DATA_PATH, index=False)
            self._last_save_time = time.time()
            logger.debug(f"[CVD] сохранено {len(rows)} минут")
        except Exception as e:
            logger.debug(f"[CVD] save error: {e}")
    
    def load(self):
        """Загрузить CVD данные из CSV."""
        try:
            if not os.path.exists(CVD_DATA_PATH):
                return
            df = pd.read_csv(CVD_DATA_PATH)
            for _, row in df.iterrows():
                ts = int(row['ts'])
                self.minutes[ts] = {
                    'ts': ts,
                    'buy_vol': float(row['buy_vol']),
                    'sell_vol': float(row['sell_vol']),
                    'buy_usd': float(row['buy_usd']),
                    'sell_usd': float(row['sell_usd']),
                    'count': int(row['count']),
                }
            logger.info(f"[CVD] загружено {len(df)} минут из CSV")
        except Exception as e:
            logger.debug(f"[CVD] load error: {e}")


# ─── УДОБНАЯ ФУНКЦИЯ ДЛЯ ИНТЕГРАЦИИ С main.py ──────────────────────────────

_global_ws_client: Optional[BybitWebSocketClient] = None
_global_cvd: Optional[CVDCollector] = None


def set_global_client(client: BybitWebSocketClient):
    """Установить глобальный WebSocket-клиент (вызывается из main.py)."""
    global _global_ws_client
    _global_ws_client = client


def get_ws_client() -> Optional[BybitWebSocketClient]:
    """Вернуть глобальный WebSocket-клиент."""
    return _global_ws_client


def get_cvd_collector() -> Optional[CVDCollector]:
    """Вернуть глобальный CVD Collector."""
    return _global_cvd


def init_cvd_collector() -> CVDCollector:
    """Создать и сохранить глобальный CVD Collector."""
    global _global_cvd
    _global_cvd = CVDCollector()
    return _global_cvd
