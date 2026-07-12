#!/usr/bin/env python3
"""
🧠 ML-СОВЕТНИК ДЛЯ ТОРГОВОЙ СИСТЕМЫ (v2 — XGBoost)

Фаза 1: ML-as-Advisor — даёт дополнительную оценку сигналам.
Фаза 2: Накопление примеров → эволюция в автономный сигнал.

Использует XGBoost + усиленные фичи + BTC-корреляцию.

Возвращает: {'decision': 'GOOD'|'WEAK'|'SKIP', 'confidence': 0.0-1.0, 'reason': str}
"""

import numpy as np
import pandas as pd
import json
from collections import deque
import os
import logging
import pickle
import time
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
import xgboost as xgb

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "data/training_data.json")
MODEL_PATH = os.path.join(BASE_DIR, "data/ml_advisor.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "data/ml_scaler.pkl")
BUFFER_PATH = os.path.join(BASE_DIR, "data/ml_advisor_buffer.pkl")
BUFFER_CACHE_PATH = os.path.join(BASE_DIR, "data/ml_advisor_features.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# Кэш для мультитаймфреймовых данных (чтобы не дёргать API каждую секунду)
_TF_CACHE = {}
_TF_CACHE_TIME = 0
_TF_CACHE_TTL = 120  # обновляем раз в 2 минуты

# Глобальная биржа (инициализируется лениво)
_EXCHANGE = None


def _get_exchange():
    global _EXCHANGE
    if _EXCHANGE is None:
        import ccxt
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
        
        # 🔐 Переопределение из .env
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
                            cfg['bybit']['api_key'] = val
                        elif key == 'BYBIT_SECRET' and val:
                            cfg['bybit']['secret'] = val
            except Exception as e:
                logger.warning(f"⚠️ .env load error: {e}")
        
        _EXCHANGE = ccxt.bybit({
            'apiKey': cfg['bybit']['api_key'],
            'secret': cfg['bybit']['secret'],
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
    return _EXCHANGE


def _fetch_tf_data(symbol, timeframe='1h', limit=48):
    """Загрузить данные с таймфрейма с кэшированием"""
    global _TF_CACHE, _TF_CACHE_TIME
    now = time.time()
    cache_key = f"{symbol}_{timeframe}"
    
    if cache_key in _TF_CACHE and (now - _TF_CACHE_TIME) < _TF_CACHE_TTL:
        return _TF_CACHE[cache_key]
    
    try:
        ex = _get_exchange()
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        _TF_CACHE[cache_key] = df
        _TF_CACHE_TIME = now
        return df
    except Exception as e:
        logger.warning(f"Не удалось загрузить {timeframe} для {symbol}: {e}")
        return None


def _fetch_btc_data():
    """Загрузить данные BTC для расчёта корреляции"""
    try:
        ex = _get_exchange()
        ohlcv = ex.fetch_ohlcv('BTC/USDT', '1h', limit=24)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except Exception as e:
        logger.warning(f"BTC data fetch error: {e}")
        return None


# Пороги ML-советника
MIN_TRAINING_SAMPLES = 50        # Минимум для обучения
CONFIDENCE_HIGH = 0.50           # Понижен: давать больше GOOD сигналов
CONFIDENCE_LOW = 0.15            # Понижен: меньше SKIP, больше WEAK
RETRAIN_INTERVAL = 3600          # Переобучение раз в час (в секундах)
SUPPORTED_FEATURES = 13         # 13 работающих фич (18 нулевых удалены после аудита 2025-06-12)


class MLAdvisor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.last_retrain = 0
        # Храним СЫРЫЕ 5m свечи (source of truth), как ML-Pro
        # Каждый элемент: {'candles_5m': [[t,o,h,l,c,v],...], 'label': 1.0/0.0, 'symbol': str, 'ts': float}
        self._candle_buffer: list = []
        self._seeded = False  # чтобы не вызывать seed_from_db дважды
        self.is_trained = False
        # Сглаживание объёмных фич — скользящее среднее за 3 скана
        # (чтобы одиночный всплеск объёма не триггерил ложный вход)
        self._vol_ratio_buffer = {}   # symbol → deque(maxlen=3)
        self._vol_momentum_buffer = {} # symbol → deque(maxlen=3)
        
        # Рыночная память для расчёта shrinking_candles
        self._candle_range_history: Dict[str, deque] = {}  # symbol → deque(range, maxlen=10)

        # Пытаемся загрузить обученную модель
        self._load_model()
        self._load_candle_buffer()  # Загружаем сырые свечи

        # 🌱 Сидирование из БД: исторические сделки
        self.seed_from_db()

        # Форсированный ретрейн при старте
        if len(self._candle_buffer) >= MIN_TRAINING_SAMPLES:
            if not self.is_trained:
                logger.info("🧠 ML: модель не загружена, обучаю с нуля...")
                self.train(force=True)
            else:
                expected = getattr(self.model, 'n_features_in_', 14)
                if expected != SUPPORTED_FEATURES:
                    logger.info(f"🔄 ML: модель обучена на {expected} фич, переобучаю на {SUPPORTED_FEATURES}...")
                    self.train(force=True)

        # Загружаем конфиг для пар
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            self.pairs = cfg['trading']['enabled_pairs']
        except Exception as _e:
            logger.debug("bare except in ml_advisor: %s", _e)
            self.pairs = []

        logger.info(f"🧠 ML-Советник v2 (XGBoost) инициализирован (модель: {'готова' if self.is_trained else 'ожидает обучения'})")

    def _load_candle_buffer(self):
        """Загрузить буфер сырых свечей из pickle"""
        if os.path.exists(BUFFER_PATH):
            try:
                with open(BUFFER_PATH, 'rb') as f:
                    self._candle_buffer = pickle.load(f)
                logger.info(f"📦 Загружено {len(self._candle_buffer)} примеров (сырые свечи)")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось загрузить candle_buffer: {e}")
                self._candle_buffer = []
        else:
            # Fallback: старый training_data.json (на всякий случай)
            self._candle_buffer = []

    def _save_candle_buffer(self):
        """Сохранить буфер сырых свечей в pickle"""
        try:
            os.makedirs(os.path.dirname(BUFFER_PATH), exist_ok=True)
            with open(BUFFER_PATH, 'wb') as f:
                pickle.dump(self._candle_buffer, f)
        except Exception as e:
            logger.warning(f"⚠️ Не удалось сохранить candle_buffer: {e}")

    def _compute_rsi(self, closes, period=14):
        """Простой RSI без pandas_ta"""
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _compute_trend_from_df(self, df):
        """Определить тренд по 5m свечам"""
        if df is None or len(df) < 20:
            return 'neutral'
        closes = df['close'].values
        sma20 = np.mean(closes[-20:])
        sma50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma20
        last = closes[-1]
        # Направление
        momentum = (closes[-1] / closes[-5] - 1) * 100 if len(closes) >= 5 else 0
        if last > sma20 and last > sma50 and momentum > 0.3:
            return 'bullish'
        elif last > sma20 and last > sma50:
            return 'weak_bullish'
        elif last < sma20 and last < sma50 and momentum < -0.3:
            return 'bearish'
        elif last < sma20 and last < sma50:
            return 'weak_bearish'
        return 'neutral'

    def seed_from_db(self):
        """
        Загрузить исторические сделки из БД и сформировать 13-фичевые сэмплы.
        """
        if self._seeded:
            return 0
        self._seeded = True
        try:
            import db_pg
        except Exception as e:
            logger.warning(f"[seed_from_db] db_pg не доступна: {e}")
            return 0

        try:
            all_trades = db_pg.get_trade_history(limit=500)
        except Exception:
            return 0

        if not all_trades:
            return 0

        # Только завершённые (с exit_price)
        completed = [t for t in all_trades if t.get('exit_price') is not None]
        if not completed:
            logger.info("[seed_from_db] Нет завершённых сделок в БД")
            return 0

        logger.info(f"[seed_from_db] Загружено {len(completed)} завершённых сделок")

        # Дедупликация: (symbol, округлённый entry_price)
        existing_keys = set()
        for d in self._candle_buffer:
            sym = d.get('symbol', '')
            ep = d.get('entry_price', d.get('pnl', 0))
            existing_keys.add((sym, round(ep, 6)))

        # Сортируем хронологически по символу для корректной работы аккумуляторов
        completed.sort(key=lambda t: (t['symbol'], t.get('entry_time', '')))

        ex = _get_exchange()
        added = 0
        skipped_dup = 0
        skipped_api = 0

        # Ограничения: макс 100 сделок, макс 30 сек на всё
        MAX_TRADES = 150
        MAX_DURATION = 180  # 3 минуты на всё
        _start_ts = time.time()

        for trade in completed[:MAX_TRADES]:
            # Таймаут: не дольше MAX_DURATION секунд
            if time.time() - _start_ts > MAX_DURATION:
                logger.info(f"[seed_from_db] ⏱ Таймаут {MAX_DURATION}с, обработано {added}/{len(completed)} сделок")
                break

            sym = trade['symbol']
            entry_price = trade['entry_price']
            entry_time_str = trade.get('entry_time', '')
            pnl_pct = trade.get('pnl_percent', 0) or 0
            entry_score = trade.get('entry_score') or 50

            if not entry_time_str:
                continue

            # Дедупликация
            key = (sym, round(entry_price, 6))
            if key in existing_keys:
                skipped_dup += 1
                continue

            # Загрузка 5m свечей на момент входа
            df_5m = None
            try:
                entry_ts = datetime.fromisoformat(entry_time_str).timestamp() * 1000
                since = int(entry_ts) - 14 * 5 * 60 * 1000
                ohlcv = ex.fetch_ohlcv(sym, '5m', since=since, limit=15)
                if ohlcv and len(ohlcv) > 10:
                    df_5m = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            except Exception as e:
                skipped_api += 1
                continue

            if df_5m is None:
                skipped_api += 1
                continue

            # RSI, trend, confidence из свечей и БД
            rsi = self._compute_rsi(df_5m['close'].values)
            trend = self._compute_trend_from_df(df_5m)
            confidence = entry_score / 100.0

            # Полный 31-фичевый вектор
            try:
                features = self._extract_features(sym, entry_price, rsi, trend, confidence, df_5m)
            except Exception as e:
                logger.debug(f"[seed_from_db] {sym}: _extract_features error: {e}")
                continue

            label = 1.0 if pnl_pct > 0.5 else 0.0

            # 🆕 Храним сырые свечи (source of truth)
            self._candle_buffer.append({
                'candles_5m': ohlcv,
                'label': label,
                'symbol': sym,
                'entry_price': entry_price,
                'pnl': pnl_pct,
                'confidence': confidence,
                'ts': entry_ts,
            })
            existing_keys.add(key)
            added += 1

        if added > 0:
            # Сохраняем буфер (сырые свечи)
            self._save_candle_buffer()

            # Принудительный ретрейн
            if len(self._candle_buffer) >= MIN_TRAINING_SAMPLES:
                self.train(force=True)

            logger.info(f"[seed_from_db] ✅ Добавлено {added} сделок (пропущено {skipped_dup} дубликатов, "
                        f"всего {len(self._candle_buffer)} примеров в буфере)")
        else:
            logger.info(f"[seed_from_db] Новых сделок нет (пропущено {skipped_dup} дубликатов)")

        return added

    
    def _load_model(self):
        """Загрузка сохранённой модели"""
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info("✅ ML-модель (XGBoost) загружена")
        except Exception as e:
            logger.warning(f"Не удалось загрузить ML-модель: {e}")
    
    def _save_model(self):
        """Сохранение модели"""
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info("✅ ML-модель (XGBoost) сохранена")
        except Exception as e:
            logger.warning(f"Не удалось сохранить ML-модель: {e}")

    def _extract_features(self, symbol, current_price, rsi, trend, confidence, df=None):
        """
        13 признаков для XGBoost (v3 — очищены от нулевых весов).
        Удалены: rsi, trend, candle_*, price_above_sma20, memory, pf-*, rsi_divergence.
        """
        
        features = {
            'vwap_dist': 0.0,
            'from_24h_high': 0.0,               # близость к 24h хаю (0-1)
            'hour_of_day': 0.0,                 # час дня (0-23, /23.0)
            'volume_momentum': 0.0,              # импульс объёма
            'consecutive_green_5m': 0.0,          # сколько 5m свеч подряд зелёные (0-1)
            'volatility': 0.01,
            'shrinking_candles': 0.0,             # сужение диапазона свечей (0-1)
            'btc_change_1h': 0.0,               # изменение BTC за 1H
            'hl_range': 0.0,                     # High-Low диапазон
            'multi_tf': 0.0,                     # multi-TF D1 против SMA50
            'body_strength': 0.5,                # тело / полный диапазон
            'volume_divergence': 0.0,             # расхождение цена↑ объём↓
            'volume_ratio': 1.0,                 # текущий объём / средний за 5
        }

        features['hour_of_day'] = datetime.now().hour / 23.0

        # 5M DataFrame — основной источник фич
        if df is not None and len(df) > 10:
            closes = df['close'].values
            volumes = df['volume'].values
            highs = df['high'].values
            lows = df['low'].values
            opens_arr = df['open'].values if 'open' in df else np.roll(closes, 1)
            
            # volatility
            features['volatility'] = np.std(closes[-10:] / (np.mean(closes[-10:]) + 0.0001))
            
            # volume_ratio (сглаженный)
            raw_vol_ratio = volumes[-1] / (np.mean(volumes[-5:]) + 0.0001)
            if symbol not in self._vol_ratio_buffer:
                self._vol_ratio_buffer[symbol] = deque(maxlen=3)
            self._vol_ratio_buffer[symbol].append(raw_vol_ratio)
            features['volume_ratio'] = np.mean(self._vol_ratio_buffer[symbol])

            # volume_momentum (сглаженный)
            raw_vol_momentum = np.mean(volumes[-3:]) / (np.mean(volumes[-10:-3]) + 0.0001)
            if symbol not in self._vol_momentum_buffer:
                self._vol_momentum_buffer[symbol] = deque(maxlen=3)
            self._vol_momentum_buffer[symbol].append(raw_vol_momentum)
            features['volume_momentum'] = np.mean(self._vol_momentum_buffer[symbol])
            
            # hl_range
            features['hl_range'] = (highs[-1] - lows[-1]) / (closes[-1] + 0.0001)

            # consecutive_green_5m
            _green_streak = 0
            for i in range(len(closes) - 1, -1, -1):
                if closes[i] > (opens_arr[i] if i < len(opens_arr) else closes[i]):
                    _green_streak += 1
                else:
                    break
            features['consecutive_green_5m'] = min(_green_streak / 10.0, 1.0)

            # body_strength
            _body = abs(closes[-1] - opens_arr[-1])
            _total_range = highs[-1] - lows[-1]
            features['body_strength'] = _body / (_total_range + 0.0001)

            # volume_divergence
            if len(closes) >= 6:
                _price_dir = 1 if closes[-1] > closes[-5] else (-1 if closes[-1] < closes[-5] else 0)
                _vol_trend = np.mean(volumes[-3:]) / (np.mean(volumes[-6:-3]) + 0.0001)
                if _price_dir > 0 and _vol_trend < 0.85:
                    features['volume_divergence'] = min(1.0, (1.0 - _vol_trend) / 0.3)
                elif _price_dir < 0 and _vol_trend > 1.15:
                    features['volume_divergence'] = min(1.0, (_vol_trend - 1.0) / 0.3)
            
            # shrinking_candles
            if symbol not in self._candle_range_history:
                self._candle_range_history[symbol] = deque(maxlen=10)
            self._candle_range_history[symbol].append(_total_range)
            _ranges = list(self._candle_range_history[symbol])
            _shrink_streak = 0
            for i in range(len(_ranges) - 1, -1, -1):
                if i > 0 and _ranges[i] < _ranges[i - 1] * 1.05:
                    _shrink_streak += 1
                else:
                    break
            features['shrinking_candles'] = min(_shrink_streak / 5.0, 1.0)

        # BTC 1H change
        try:
            df_btc = _fetch_btc_data()
            if df_btc is not None and len(df_btc) > 3:
                btc_closes = df_btc['close'].values
                features['btc_change_1h'] = (btc_closes[-1] - btc_closes[-2]) / btc_closes[-2]
        except Exception as _e:
            logger.debug("bare except in ml_advisor: %s", _e)

        # multi_tf (D1 SMA50)
        try:
            df_d1 = _fetch_tf_data(symbol, '1d', 60)
            if df_d1 is not None and len(df_d1) > 50:
                closes_d1 = df_d1['close'].values
                sma50 = np.mean(closes_d1[-50:])
                features['multi_tf'] = (current_price - sma50) / (sma50 + 0.0001)
        except Exception as _e:
            logger.debug("bare except in ml_advisor: %s", _e)

        # vwap_dist (1H)
        try:
            df_1h = _fetch_tf_data(symbol, '1h', 24)
            if df_1h is not None and len(df_1h) > 12:
                closes_1h = df_1h['close'].values
                volumes_1h = df_1h['volume'].values
                vwap = np.sum(closes_1h[-12:] * volumes_1h[-12:]) / (np.sum(volumes_1h[-12:]) + 0.0001)
                features['vwap_dist'] = (current_price - vwap) / vwap
        except Exception as _e:
            logger.debug("bare except in ml_advisor: %s", _e)

        # from_24h_high
        try:
            ex = _get_exchange()
            _ticker = ex.fetch_ticker(symbol)
            _high_24h = _ticker.get('high')
            _low_24h = _ticker.get('low')
            if _high_24h and _low_24h and _high_24h > _low_24h:
                features['from_24h_high'] = (current_price - _low_24h) / (_high_24h - _low_24h)
                features['from_24h_high'] = max(0.0, min(1.0, features['from_24h_high']))
        except Exception as _e:
            logger.debug("bare except in ml_advisor: %s", _e)
        
        return [
            features['vwap_dist'], features['from_24h_high'], features['hour_of_day'],
            features['volume_momentum'], features['consecutive_green_5m'], features['volatility'],
            features['shrinking_candles'], features['btc_change_1h'], features['hl_range'],
            features['multi_tf'], features['body_strength'], features['volume_divergence'],
            features['volume_ratio']
        ]



    def _feature_names(self):
        return ['vwap_dist', 'from_24h_high', 'hour_of_day', 'volume_momentum',
                'consecutive_green_5m', 'volatility', 'shrinking_candles',
                'btc_change_1h', 'hl_range', 'multi_tf', 'body_strength',
                'volume_divergence', 'volume_ratio']

    def add_trade_result(self, symbol, entry_price, exit_price, rsi, trend, confidence,
                         hold_hours, reason, volume_ratio=None):
        """
        Добавляет результат сделки в обучающую выборку.
        Использует _extract_features для полного вектора признаков (13 фич).
        label = 1 если сделка прибыльная > 0.5% (хороший сигнал)
        label = 0 если убыточная или нулевая (плохой сигнал)
        """
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        label = 1.0 if pnl_pct > 0.5 else 0.0

        # Пробуем получить 5m свечи для полного вектора признаков
        df_5m = None
        try:
            import copy
            raw = _fetch_tf_data(symbol, '5m', 12)
            if raw is not None and len(raw) > 10:
                df_5m = raw
        except Exception:
            pass

        # Сырые 5m свечи на момент сделки (source of truth)
        ohlcv_raw = None
        try:
            ex = _get_exchange()
            entry_ts = time.time() * 1000
            since = int(entry_ts) - 14 * 5 * 60 * 1000
            ohlcv_raw = ex.fetch_ohlcv(symbol, '5m', since=since, limit=15)
        except Exception:
            pass

        if ohlcv_raw and len(ohlcv_raw) >= 10:
            self._candle_buffer.append({
                'candles_5m': ohlcv_raw,
                'label': label,
                'symbol': symbol,
                'entry_price': entry_price,
                'pnl': pnl_pct,
                'confidence': confidence,
                'ts': time.time(),
            })
            self._save_candle_buffer()
            logger.info(f"📚 ML: +1 сделка {symbol} (PnL={pnl_pct:+.2f}%, label={int(label)})")
        else:
            logger.warning(f"📚 ML: {symbol} — нет свечей, пропущено")

    def _build_features_from_buffer(self):
        """Пересчитать 13 фич из сырых свечей актуальной версией _extract_features."""
        X_list = []
        y_list = []
        skipped = 0
        for item in self._candle_buffer:
            try:
                candles_5m = item['candles_5m']
                label = item['label']
                symbol = item.get('symbol', 'UNKNOWN')
                entry_price = item.get('entry_price', 0.0)
                confidence = item.get('confidence', 0.5)
                
                if not candles_5m or len(candles_5m) < 10:
                    skipped += 1
                    continue
                
                # OHLCV → DataFrame
                df = pd.DataFrame(candles_5m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                if len(df) < 10:
                    skipped += 1
                    continue
                
                rsi = self._compute_rsi(df['close'].values) if hasattr(self, '_compute_rsi') else 50
                trend = self._compute_trend_from_df(df) if hasattr(self, '_compute_trend_from_df') else 'neutral'
                
                features = self._extract_features(symbol, entry_price, rsi, trend, confidence, df)
                
                if len(features) > SUPPORTED_FEATURES:
                    features = features[:SUPPORTED_FEATURES]
                elif len(features) < SUPPORTED_FEATURES:
                    features = features + [0.0] * (SUPPORTED_FEATURES - len(features))
                
                X_list.append(features)
                y_list.append(label)
            except Exception:
                skipped += 1
                continue
        
        return X_list, y_list, skipped

    def train(self, force=False):
        """
        Обучение/переобучение XGBoost на сырых свечах (13 фич пересчитываются на лету).
        """
        now = time.time()

        if not force and (now - self.last_retrain) < RETRAIN_INTERVAL:
            return

        if len(self._candle_buffer) < MIN_TRAINING_SAMPLES:
            logger.info(f"⏳ ML: ждём данные ({len(self._candle_buffer)}/{MIN_TRAINING_SAMPLES})")
            return

        # 🆕 Пересчитываем фичи из сырых свечей
        X_list, y_list, skipped = self._build_features_from_buffer()

        if len(X_list) < MIN_TRAINING_SAMPLES:
            logger.info(f"⏳ ML: после фильтрации {len(X_list)} сэмплов, нужно {MIN_TRAINING_SAMPLES}")
            return

        X = np.array(X_list)
        y = np.array(y_list)

        if len(np.unique(y)) < 2:
            logger.warning(f"⚠️ ML: все данные одного класса ({int(y[0])}), ждём разнообразия")
            return

        n_bad = int((y == 0).sum())
        n_good = int((y == 1).sum())
        scale_weight = n_bad / max(n_good, 1)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.0,
            reg_alpha=0.5,
            scale_pos_weight=scale_weight,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0
        )
        self.model.fit(X_scaled, y, verbose=False)

        logger.info(f"   Калибровка: отключена (raw XGBoost probs)")

        train_score = self.model.score(X_scaled, y)
        self.is_trained = True
        self.last_retrain = now
        self._save_model()

        logger.info(f"🎯 ML (XGBoost v3): обучен! Точность: {train_score:.1%} | "
                    f"Данных: {len(y)} (good={n_good}, bad={n_bad}, weight={scale_weight:.2f}, skipped={skipped})")

        if hasattr(self.model, 'feature_importances_'):
            names = self._feature_names()
            importances = self.model.feature_importances_
            if len(importances) == len(names):
                top5 = sorted(zip(names, importances), key=lambda x: -x[1])[:5]
                logger.info(f"   Топ-5 фич: {', '.join(f'{n}={v:.2f}' for n,v in top5)}")

    def evaluate(self, symbol, current_price, rsi, trend, confidence, df=None):
        """
        Оценка сигнала ML-советником.
        Возвращает: {'decision': 'GOOD'|'WEAK'|'SKIP', 'confidence': float, 'reason': str}
        """
        if not self.is_trained or self.model is None:
            return {'decision': 'GOOD', 'confidence': 0.5,
                    'reason': 'ML не обучен (доверяем основной системе)'}

        features = self._extract_features(symbol, current_price, rsi, trend, confidence, df)
        
        # Совместимость: модель может ожидать SUPPORTED_FEATURES (13)
        expected_features = getattr(self.model, 'n_features_in_', SUPPORTED_FEATURES)
        if len(features) > expected_features:
            features = features[:expected_features]
        elif len(features) < expected_features:
            # Если вдруг меньше — дополняем нулями
            features = features + [0.0] * (expected_features - len(features))
        
        X = np.array([features])
        X_scaled = self.scaler.transform(X)

        prob = self.model.predict_proba(X_scaled)[0]

        if len(prob) < 2:
            return {'decision': 'GOOD', 'confidence': 0.5,
                    'reason': 'ML: недостаточно классов'}

        good_prob = prob[1] if len(prob) > 1 else 0.5

        if good_prob >= CONFIDENCE_HIGH:
            decision = 'GOOD'
            reason = f'ML: {good_prob:.0%} (сигнал сильный)'
        elif good_prob >= CONFIDENCE_LOW:
            decision = 'WEAK'
            reason = f'ML: {good_prob:.0%} (сигнал средний)'
        else:
            decision = 'SKIP'
            reason = f'ML: {good_prob:.0%} (сигнал слабый)'

        return {'decision': decision, 'confidence': float(good_prob), 'reason': reason}


# Глобальный экземпляр советника
_advisor = None


def get_advisor():
    """Получить глобальный экземпляр ML-советника"""
    global _advisor
    if _advisor is None:
        _advisor = MLAdvisor()
    return _advisor


def ml_evaluate(symbol, current_price, rsi, trend, confidence, df=None):
    """Удобная обёртка для оценки сигнала"""
    advisor = get_advisor()
    return advisor.evaluate(symbol, current_price, rsi, trend, confidence, df)


def ml_add_result(symbol, entry_price, exit_price, rsi, trend, confidence,
                  hold_hours, reason, volume_ratio=None):
    """Удобная обёртка для добавления результата сделки"""
    advisor = get_advisor()
    advisor.add_trade_result(symbol, entry_price, exit_price, rsi, trend,
                              confidence, hold_hours, reason, volume_ratio)


def ml_train(force=False):
    """Удобная обёртка для обучения"""
    advisor = get_advisor()
    advisor.train(force=force)
