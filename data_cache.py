#!/usr/bin/env python3
"""
data_cache.py — Центральный кеш рыночных данных с архитектурой под WebSocket.

Сейчас: наполняется через REST (с ограничением частоты запросов).
В будущем: WebSocket-клиент будет писать напрямую в этот же кеш.
Менять внешний API не придётся — он просто станет быстрее и дешевле.

Принципы:
  1. Цены обновляются не чаще 1 раза в 3 секунды на символ (REST throttle)
  2. Свечи скачиваются раз в 60-120 секунд (не чаще)
  3. Все модули (trader, monitor) читают из одного кеша
  4. При WebSocket — throttle отключается, данные приходят сами
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import logging
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict

logger = logging.getLogger('data_cache')


class PriceCache:
    """
    Кеш цен тикеров.
    
    Сейчас: наполняется через REST fetch_ticker() с throttle 3 сек.
    WebSocket: заменит fetch_ticker() на подписку, кеш останется тем же.
    
    Thread-safe: threading.Lock.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        
        self._prices: Dict[str, dict] = {}      # symbol -> {price, bid, ask, timestamp}
        self._lock = threading.Lock()
        
        # WebSocket-readiness: можно подключить callback
        self._on_update: Optional[Callable] = None
    
    # ─── ПУБЛИЧНЫЙ API ──────────────────────────────────────────────────
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Вернуть последнюю цену из кеша."""
        with self._lock:
            entry = self._prices.get(symbol)
            if entry:
                return entry['price']
        return None
    
    def get_all(self) -> Dict[str, float]:
        """Вернуть все цены."""
        with self._lock:
            return {k: v['price'] for k, v in self._prices.items()}
    
    def get_timestamp(self, symbol: str) -> Optional[float]:
        """Когда последний раз обновлялась цена."""
        with self._lock:
            entry = self._prices.get(symbol)
            return entry['timestamp'] if entry else None
    
    # ─── ЗАПИСЬ ────────────────────────────────────────────────────────
    
    def update_price(self, symbol: str, price: float, bid: float = None,
                     ask: float = None) -> None:
        """
        Обновить цену в кеше.
        
        WebSocket-ready: WebSocket-клиент будет вызывать этот же метод
        при получении тикера.
        """
        now = time.time()
        with self._lock:
            self._prices[symbol] = {
                'price': price,
                'bid': bid or price,
                'ask': ask or price,
                'timestamp': now,
            }
        
        # Callback для внешних подписчиков (будущее)
        if self._on_update:
            try:
                self._on_update(symbol, price)
            except Exception as e:
                logger.debug(f"[CACHE] Callback error for {symbol}: {e}")
    
    def update_from_ticker(self, ticker: dict) -> None:
        """Обновить из объекта тикера ccxt."""
        symbol = ticker.get('symbol')
        if not symbol:
            return
        self.update_price(
            symbol,
            ticker.get('last', 0),
            ticker.get('bid'),
            ticker.get('ask'),
        )
    
    def set_on_update(self, callback: Callable) -> None:
        """
        Установить callback на обновление цены.
        Для будущей WebSocket-архитектуры.
        """
        self._on_update = callback
    
    # ─── СТАТИСТИКА ─────────────────────────────────────────────────────
    
    def stats(self) -> Dict:
        """Вернуть статистику кеша."""
        with self._lock:
            count = len(self._prices)
            oldest = min((v['timestamp'] for v in self._prices.values()), default=0)
            newest = max((v['timestamp'] for v in self._prices.values()), default=0)
            age = time.time() - oldest if oldest else -1
            return {
                'symbols': count,
                'oldest_age_sec': round(age, 1),
                'newest_age_sec': round(time.time() - newest, 1) if newest else -1,
                'has_websocket': self._on_update is not None,
            }


class OHLCVCache:
    """
    Кеш свечных данных (OHLCV).
    
    Сейчас: скачивается через REST fetch_ohlcv().
    WebSocket: будет подписан на свечные каналы и дописывать новые свечи.
    
    Стратегия кеширования:
      - Каждая свеча хранится как кортеж (timestamp, open, high, low, close, volume)
      - При запросе отдаётся столько свечей сколько запрошено (от newest к oldest)
      - Если свечи устарели (старше max_age_sec) — возвращается пустой список
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, default_cache_seconds: int = 120):
        if hasattr(self, '_initialized') and self._initialized:
            return
        self._initialized = True
        
        # symbol + timeframe -> [(timestamp, o, h, l, c, v), ...]
        self._cache: Dict[str, List[tuple]] = {}
        self._timestamps: Dict[str, float] = {}  # когда обновляли
        self._lock = threading.Lock()
        
        self.default_cache_seconds = default_cache_seconds
    
    def _key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol}:{timeframe}"
    
    # ─── ПУБЛИЧНЫЙ API ──────────────────────────────────────────────────
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h',
                  limit: int = 100, max_age_sec: int = None) -> Optional[List[list]]:
        """
        Вернуть свечи из кеша.
        
        Возвращает None если кеш пуст или устарел.
        Иначе — список свечей [timestamp, o, h, l, c, v], от старых к новым.
        """
        if max_age_sec is None:
            max_age_sec = self.default_cache_seconds * 2
        
        key = self._key(symbol, timeframe)
        with self._lock:
            last_update = self._timestamps.get(key, 0)
            if time.time() - last_update > max_age_sec:
                return None  # кеш устарел
            
            data = self._cache.get(key)
            if not data:
                return None
            
            # Последние `limit` свечей
            return list(data[-limit:])
    
    def has_recent(self, symbol: str, timeframe: str,
                   max_age_sec: int = None) -> bool:
        """Проверить есть ли свежие данные в кеше (без возврата данных)."""
        if max_age_sec is None:
            max_age_sec = self.default_cache_seconds * 2
        key = self._key(symbol, timeframe)
        with self._lock:
            last_update = self._timestamps.get(key, 0)
            return time.time() - last_update <= max_age_sec
    
    def get_last_candle(self, symbol: str, timeframe: str) -> Optional[list]:
        """Вернуть последнюю свечу."""
        key = self._key(symbol, timeframe)
        with self._lock:
            data = self._cache.get(key)
            if data:
                return list(data[-1])
        return None
    
    # ─── ЗАПИСЬ ────────────────────────────────────────────────────────
    
    def update_ohlcv(self, symbol: str, timeframe: str,
                     ohlcv_data: List[list]) -> None:
        """
        Обновить кеш свечей.
        
        WebSocket-ready: будет вызываться WebSocket-клиентом
        при получении новой свечи.
        """
        key = self._key(symbol, timeframe)
        with self._lock:
            self._cache[key] = [tuple(c) for c in ohlcv_data]
            self._timestamps[key] = time.time()
    
    def append_candle(self, symbol: str, timeframe: str,
                       candle: tuple) -> None:
        """
        Добавить одну свечу (для WebSocket).
        Если последняя свеча имеет тот же timestamp — заменяет её.
        """
        key = self._key(symbol, timeframe)
        with self._lock:
            data = self._cache.get(key, [])
            if data and data[-1][0] == candle[0]:
                data[-1] = candle  # обновляем текущую (незакрытую) свечу
            else:
                data.append(candle)
                # Ограничиваем размер (не больше 5000 свечей)
                if len(data) > 5000:
                    data = data[-5000:]
            self._cache[key] = data
            self._timestamps[key] = time.time()
    
    # ─── СТАТИСТИКА ─────────────────────────────────────────────────────
    
    def stats(self) -> Dict:
        """Вернуть статистику кеша."""
        with self._lock:
            keys = list(self._cache.keys())
            return {
                'cached_series': len(keys),
                'keys': keys[:10],  # первые 10
                'oldest_update': min(self._timestamps.values(), default=0),
            }


# ─── УТИЛИТА: REST-загрузчик с кешем ────────────────────────────────────────

class CachedDataFetcher:
    """
    Загрузчик данных с кешем.
    
    Трейдер и монитор используют этот класс вместо прямого fetch_ticker().
    Класс сам решает — взять из кеша или сходить на биржу.
    При WebSocket кеш будет всегда свежий, хождение на биржу отключится.
    """
    
    def __init__(self, exchange, price_cache: PriceCache,
                 ohlcv_cache: OHLCVCache):
        self.exchange = exchange
        self.prices = price_cache
        self.ohlcv = ohlcv_cache
        
        # Throttle: не ходить на биржу чаще чем раз в N секунд
        self.ticker_throttle = 3.0   # секунд между fetch_ticker для одного символа
        self.ohlcv_throttle = 60.0   # секунд между fetch_ohlcv для одного таймфрейма
        
        # Счётчики для статистики
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls = 0
    
    def get_ticker(self, symbol: str, force_fetch: bool = False) -> Optional[float]:
        """
        Получить цену. Сначала кеш, если свежий — возвращаем.
        Если устарел или force_fetch — идём на биржу.
        """
        # 1. Проверяем кеш
        if not force_fetch:
            ts = self.prices.get_timestamp(symbol)
            if ts and time.time() - ts < self.ticker_throttle:
                price = self.prices.get_price(symbol)
                if price is not None:
                    self.cache_hits += 1
                    return price
        
        # 2. Идём на биржу
        self.cache_misses += 1
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            self.api_calls += 1
            self.prices.update_from_ticker(ticker)
            return ticker.get('last')
        except Exception as e:
            logger.debug(f"[FETCH] Ошибка ticker {symbol}: {e}")
            # Возвращаем старый кеш, если есть
            return self.prices.get_price(symbol)
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h',
                   limit: int = 100) -> Optional[List[list]]:
        """
        Получить свечи. Сначала кеш, потом биржа.
        """
        # 1. Проверяем кеш (свечи живут дольше)
        candles = self.ohlcv.get_ohlcv(symbol, timeframe, limit,
                                        max_age_sec=self.ohlcv_throttle * 2)
        if candles is not None:
            self.cache_hits += 1
            return candles
        
        # 2. Идём на биржу
        self.cache_misses += 1
        try:
            data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            self.api_calls += 1
            self.ohlcv.update_ohlcv(symbol, timeframe, data)
            return list(data[-limit:])
        except Exception as e:
            logger.debug(f"[FETCH] Ошибка ohlcv {symbol}: {e}")
            return None
    
    def batch_get_tickers(self, symbols: List[str]) -> Dict[str, float]:
        """
        Получить цены для списка символов.
        Для новых символов — ходим на биржу.
        Для закешированных — возвращаем из кеша.
        """
        result = {}
        fetch_needed = []
        
        for sym in symbols:
            price = self.prices.get_price(sym)
            ts = self.prices.get_timestamp(sym)
            if price is not None and ts and time.time() - ts < self.ticker_throttle:
                result[sym] = price
            else:
                fetch_needed.append(sym)
        
        if fetch_needed:
            for sym in fetch_needed:
                price = self.get_ticker(sym, force_fetch=True)
                if price is not None:
                    result[sym] = price
        
        return result
    
    def stats(self) -> Dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'api_calls': self.api_calls,
        }


# Глобальный fetcher (устанавливается main.py)
_global_fetcher = None


def get_fetcher() -> Optional[CachedDataFetcher]:
    """Вернуть глобальный CachedDataFetcher (если установлен)."""
    return _global_fetcher


def get_price(symbol: str) -> Optional[float]:
    """Удобная функция: получить цену из глобального кеша."""
    f = _global_fetcher
    if f:
        return f.get_ticker(symbol)
    return None


# ─── MONKEY-PATCH: замена exchange.fetch_ticker() на кеш ─────────────────────

# ─── MONKEY-PATCH: кеш для fetch_ticker ──────────────────────────────────────

def patch_exchange_fetch_ticker(exchange) -> None:
    """Monkey-patch exchange.fetch_ticker() — сначала кеш, потом биржа."""
    original_fetch_ticker = exchange.fetch_ticker
    
    def cached_fetch_ticker(symbol: str, params: dict = None):
        fetcher = get_fetcher()
        if fetcher:
            price = fetcher.get_ticker(symbol)
            if price is not None:
                ts = fetcher.prices.get_timestamp(symbol)
                return {
                    'symbol': symbol,
                    'last': price,
                    'bid': price,
                    'ask': price,
                    'timestamp': int(ts * 1000) if ts else 0,
                    'datetime': datetime.now(timezone.utc).isoformat(),
                }
        return original_fetch_ticker(symbol, params or {})
    
    exchange.fetch_ticker = cached_fetch_ticker


# ─── ГЛОБАЛЬНЫЙ DecisionEngine для monkey-patch create_order ─────────────────

_decision_engine_blocked_buys = [0]
_decision_engine_allowed_buys = [0]


def patch_exchange_create_order(exchange) -> None:
    """
    Monkey-patch exchange.create_order() — проверка через DecisionEngine.
    Только buy-ордера. Sell-ордера (стоп-лоссы) проходят без проверки.
    """
    original_create_order = exchange.create_order
    
    def checked_create_order(symbol: str, order_type: str, side: str,
                              amount: float, price: float = None,
                              params: dict = None):
        # Только buy
        if side != 'buy':
            return original_create_order(symbol, order_type, side, amount, price, params or {})
        
        # Цена
        current_price = price
        fetcher = get_fetcher()
        if fetcher:
            cached = fetcher.prices.get_price(symbol)
            if cached:
                current_price = cached
        
        # DecisionEngine
        try:
            from decision_engine import DecisionEngine
            de = DecisionEngine()
            decision = de.decide_entry(symbol, current_price or 0)
            
            if decision.get('approved', False):
                _decision_engine_allowed_buys[0] += 1
                logger.info(f"✅ [DE] buy {symbol} ОДОБРЕН ({decision.get('score', 0):.0f}/100)")
                return original_create_order(symbol, order_type, side, amount, price, params or {})
            else:
                _decision_engine_blocked_buys[0] += 1
                logger.warning(f"🔒 [DE] buy {symbol} ЗАБЛОКИРОВАН | всего: {_decision_engine_blocked_buys[0]}")
                return None
        except Exception as e:
            logger.warning(f"⚠️ [DE] ошибка: {e}. Пропускаем.")
            return original_create_order(symbol, order_type, side, amount, price, params or {})
    
    exchange.create_order = checked_create_order
    logger.info("🔒 Monkey-patch: create_order → DecisionEngine")


def get_de_stats() -> dict:
    """Статистика решений DecisionEngine."""
    return {
        'blocked_buys': _decision_engine_blocked_buys[0],
        'allowed_buys': _decision_engine_allowed_buys[0],
    }
