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
CONFIG_PATH = os.path.join(BASE_DIR, "config/api_config_final.json")

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
CONFIDENCE_HIGH = 0.65           # XGBoost калиброван лучше, порог можно ниже
CONFIDENCE_LOW = 0.35            # Ниже этого — SKIP
RETRAIN_INTERVAL = 3600          # Переобучение раз в час (в секундах)
SUPPORTED_FEATURES = 31         # 17 базовых + 8 портфельных + 6 рыночных фич


class MLAdvisor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.last_retrain = 0
        self.training_data = []  # Собираем примеры: [features, label]
        self.is_trained = False
        # Сглаживание объёмных фич — скользящее среднее за 3 скана
        # (чтобы одиночный всплеск объёма не триггерил ложный вход)
        self._vol_ratio_buffer = {}   # symbol → deque(maxlen=3)
        self._vol_momentum_buffer = {} # symbol → deque(maxlen=3)
        
        # Память о поведении ММ на символе
        self._consecutive_losses: Dict[str, int] = {}  # symbol → сколько убытков подряд
        self._symbol_trade_count: Dict[str, int] = {}  # symbol → всего сделок
        self._symbol_win_count: Dict[str, int] = {}    # symbol → прибыльных сделок
        self._recent_trades: Dict[str, deque] = {}     # symbol → deque(pnl_pct, maxlen=10)

        # Рыночная память — history для расчёта дивергенций и затухания
        self._rsi_history: Dict[str, deque] = {}       # symbol → deque(rsi, maxlen=5)
        self._candle_range_history: Dict[str, deque] = {}  # symbol → deque(range, maxlen=10)

        # 📊 Портфельные метрики (обновляются трейдером каждый цикл)
        self._pf_daily_pnl = 0.0              # PnL за сегодня
        self._pf_daily_profit_count = 0       # Прибыльных сделок сегодня
        self._pf_daily_loss_count = 0         # Убыточных сделок сегодня
        self._pf_daily_trade_count = 0        # Всего сделок сегодня
        self._pf_consecutive_profits = 0      # Прибыльных подряд
        self._pf_consecutive_losses_global = 0 # Убыточных подряд (глобальный)
        self._pf_open_positions = 0           # Открытых позиций
        self._pf_exposure_pct = 0.0           # % капитала в позициях
        self._pf_avg_position_value = 0.0     # Средний размер позиции

        # Пытаемся загрузить обученную модель
        self._load_model()
        self._load_training_data()  # Загружаем накопленные примеры

        # 🌱 Сидирование из БД: исторические сделки → 31-фичевые сэмплы
        self.seed_from_db()

        # Форсированный ретрейн при старте (если данные есть, но модель старая)
        if self.is_trained and len(self.training_data) >= MIN_TRAINING_SAMPLES:
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

    def _load_training_data(self):
        """Загрузить накопленные данные из JSON"""
        if os.path.exists(TRAINING_DATA_PATH):
            try:
                with open(TRAINING_DATA_PATH, 'r') as f:
                    raw = json.load(f)
                self.training_data = raw
                logger.info(f"📦 Загружено {len(raw)} записей обучения")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось загрузить training_data: {e}")
                self.training_data = []

    def _save_training_data(self):
        """Сохранить накопленные данные в JSON"""
        try:
            os.makedirs(os.path.dirname(TRAINING_DATA_PATH), exist_ok=True)
            with open(TRAINING_DATA_PATH, 'w') as f:
                json.dump(self.training_data, f, default=str)
        except Exception as e:
            logger.warning(f"⚠️ Не удалось сохранить training_data: {e}")

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
        Загрузить исторические сделки из БД и сформировать 31-фичевые сэмплы.
        Решает проблему: новые 6 рыночных признаков не участвуют в модели,
        т.к. training_data содержит только старые 14-фичевые записи.
        """
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
        for d in self.training_data:
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
        MAX_TRADES = 100
        MAX_DURATION = 30
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

            self.training_data.append({
                'features': features,
                'label': label,
                'symbol': sym,
                'entry_price': entry_price,
                'pnl': pnl_pct,
                'reason': 'seed_from_db',
                'time': datetime.now().isoformat()
            })
            existing_keys.add(key)
            added += 1

        if added > 0:
            # Удаляем старые 9-14 фичевые записи (они все замещены историей)
            old_count = len([d for d in self.training_data if len(d.get('features', [])) != SUPPORTED_FEATURES])
            if old_count > 0:
                self.training_data = [d for d in self.training_data if len(d.get('features', [])) == SUPPORTED_FEATURES]

            self._save_training_data()

            # Принудительный ретрейн — теперь на полных 31 фичах
            if len(self.training_data) >= MIN_TRAINING_SAMPLES:
                self.train(force=True)

            logger.info(f"[seed_from_db] ✅ Добавлено {added}, пропущено дубликатов {skipped_dup}, "
                        f"удалено старых {old_count}, всего {len(self.training_data)} сэмплов ({SUPPORTED_FEATURES} фич)")
        else:
            logger.info(f"[seed_from_db] Новых сделок нет (пропущено {skipped_dup} дубликатов)")

        return added

    def update_symbol_memory(self, symbol: str, consecutive_losses: int = 0,
                               trade_result: Optional[float] = None,
                               total_trades: int = 0, wins: int = 0) -> None:
        """
        Обновить память о поведении ММ на символе.
        
        Args:
            symbol: символ (e.g. 'NEAR/USDT')
            consecutive_losses: сколько убытков подряд на этом символе
            trade_result: PnL% последней сделки (если есть)
            total_trades: всего сделок на символе
            wins: прибыльных сделок
        """
        safe_sym = symbol.split('/')[0] if '/' in symbol else symbol
        self._consecutive_losses[safe_sym] = consecutive_losses
        self._symbol_trade_count[safe_sym] = total_trades
        self._symbol_win_count[safe_sym] = wins
        
        if trade_result is not None:
            if safe_sym not in self._recent_trades:
                self._recent_trades[safe_sym] = deque(maxlen=10)
            self._recent_trades[safe_sym].append(trade_result)
    
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

    def _calc_candle_patterns(self, df):
        """Определение свечных паттернов на последней свече"""
        if df is None or len(df) < 2:
            return {}
        
        o, h, l, c = df['open'].values, df['high'].values, df['low'].values, df['close'].values
        last = -1
        
        body = abs(c[last] - o[last])
        upper_wick = h[last] - max(c[last], o[last])
        lower_wick = min(c[last], o[last]) - l[last]
        total_range = h[last] - l[last]
        
        patterns = {
            'doji': body < total_range * 0.1,
            'hammer': lower_wick > body * 2 and upper_wick < body * 0.5,
            'shooting_star': upper_wick > body * 2 and lower_wick < body * 0.5,
            'bullish_candle': c[last] > o[last],
            'bearish_candle': c[last] < o[last],
            'long_body': body > total_range * 0.7,
            'engulfing_bull': last >= 1 and c[last] > o[last] and o[last] < c[last-1] and c[last] > o[last-1],
            'engulfing_bear': last >= 1 and c[last] < o[last] and o[last] > c[last-1] and c[last] < o[last-1],
        }
        return patterns

    def _calc_liquidity_zones(self, df_1h):
        """Определение зон ликвидности"""
        if df_1h is None or len(df_1h) < 10:
            return {}
        
        current_price = df_1h['close'].values[-1]
        volumes = df_1h['volume'].values
        highs = df_1h['high'].values
        lows = df_1h['low'].values
        
        vwap = np.average((highs + lows) / 2, weights=volumes)
        
        max_vol_idx = np.argmax(volumes[-24:]) if len(volumes) >= 24 else np.argmax(volumes)
        liq_high = highs[-(24 - max_vol_idx)] if len(volumes) >= 24 else highs[max_vol_idx]
        liq_low = lows[-(24 - max_vol_idx)] if len(volumes) >= 24 else lows[max_vol_idx]
        
        if current_price < liq_low:
            dist_to_liq = (liq_low - current_price) / current_price * 100
        elif current_price > liq_high:
            dist_to_liq = (current_price - liq_high) / current_price * 100
        else:
            dist_to_liq = 0.0
        
        return {
            'vwap_distance': (current_price - vwap) / vwap * 100,
            'dist_to_big_volume': dist_to_liq,
            'near_big_volume': 1.0 if dist_to_liq < 1.0 else 0.0,
            'price_above_vwap': 1.0 if current_price > vwap else 0.0,
        }

    def _extract_features(self, symbol, current_price, rsi, trend, confidence, df=None):
        """
        17 признаков для XGBoost.
        25 признаков: 17 базовых + 8 портфельных.
        Портфельные метрики обновляются трейдером через update_portfolio_stats().
        """
        safe_sym = symbol.split('/')[0] if '/' in symbol else symbol
        
        features = {
            'rsi': rsi,
            'trend': 0.5,
            'volatility': 0.01,
            'volume_ratio': 1.0,
            'multi_tf': 0.0,
            'vwap_dist': 0.0,
            'candle_doji': 0.0,
            'candle_hammer': 0.0,
            'candle_engulfing': 0.0,
            'btc_change_1h': 0.0,               # Изменение BTC за последний час
            'hour_of_day': 0.0,                 # Час дня (0-23, /23.0)
            'volume_momentum': 0.0,              # Импульс объёма
            'hl_range': 0.0,                     # High-Low диапазон
            'price_above_sma20': 1.0,            # Цена выше SMA20 на 1H
            # Фичи памяти — чтобы XGBoost учился распознавать ММ
            'consecutive_losses': 0.0,            # Убытков подряд (0, 1, 2, 3...)
            'loss_streak_active': 0.0,            # 1 если consecutive_losses >= 2
            'recent_win_rate_10': 0.5,            # % прибыльных из последних 10
            # 📊 Портфельные метрики (одинаковы для всех символов в цикле)
            'pf_daily_pnl': 0.0,                  # $ PnL сегодня
            'pf_daily_profit_ratio': 0.5,          # доля профитных сегодня
            'pf_daily_trade_count': 0.0,           # всего сделок сегодня
            'pf_consecutive_profits': 0.0,         # профитных подряд (глобально)
            'pf_consecutive_losses_global': 0.0,   # убыточных подряд (глобально)
            'pf_open_positions_pct': 0.0,          # кол-во позиций / max (0-1)
            'pf_exposure_pct_norm': 0.0,           # % капитала в позициях / 100
            'pf_avg_position_value_norm': 0.0,     # средний размер позиции / 90
            # 🔬 Рыночные фичи контекста — чтобы отличать «ранний вход» от «погони за хаем»
            'consecutive_green_5m': 0.0,            # сколько 5m свеч подряд зелёные (0-1, /10)
            'from_24h_high': 0.0,                   # насколько близко к 24h хаю (0-1)
            'body_strength': 0.5,                   # отношение тела к теням (0=шипы, 1=сильное тело)
            'volume_divergence': 0.0,               # расхождение цена↑ объём↓ (0=норма, 1=сильное расхождение)
            'rsi_divergence_bear': 0.0,             # цена↑ RSI↓ (0=нет, 1=медвежья дивергенция)
            'shrinking_candles': 0.0,               # сколько свеч подряд диапазон сужается (0-1, /5)
        }

        features['trend'] = (1.0 if trend == 'bullish' else (0.0 if trend == 'bearish' else 0.5))
        features['hour_of_day'] = datetime.now().hour / 23.0  # 0..23 → 0..1

        # 5M DataFrame
        if df is not None and len(df) > 10:
            closes = df['close'].values
            volumes = df['volume'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # Волатильность и объём
            features['volatility'] = np.std(closes[-10:] / (np.mean(closes[-10:]) + 0.0001))
            # Сглаженный volume_ratio — скользящее среднее за 3 скана
            raw_vol_ratio = volumes[-1] / (np.mean(volumes[-5:]) + 0.0001)
            if symbol not in self._vol_ratio_buffer:
                self._vol_ratio_buffer[symbol] = deque(maxlen=3)
            self._vol_ratio_buffer[symbol].append(raw_vol_ratio)
            features['volume_ratio'] = np.mean(self._vol_ratio_buffer[symbol])

            # Сглаженный volume_momentum — скользящее среднее за 3 скана
            raw_vol_momentum = np.mean(volumes[-3:]) / (np.mean(volumes[-10:-3]) + 0.0001)
            if symbol not in self._vol_momentum_buffer:
                self._vol_momentum_buffer[symbol] = deque(maxlen=3)
            self._vol_momentum_buffer[symbol].append(raw_vol_momentum)
            features['volume_momentum'] = np.mean(self._vol_momentum_buffer[symbol])
            features['hl_range'] = (highs[-1] - lows[-1]) / (closes[-1] + 0.0001)

            # Паттерны свечей
            patterns = self._calc_candle_patterns(df)
            features['candle_doji'] = 1.0 if patterns.get('doji') else 0.0
            features['candle_hammer'] = 1.0 if patterns.get('hammer') else 0.0
            features['candle_engulfing'] = 1.0 if (patterns.get('engulfing_bull') or patterns.get('engulfing_bear')) else 0.0

            # 🔬 РЫНОЧНЫЙ КОНТЕКСТ — 6 признаков
            # --- consecutive_green_5m: сколько свеч подряд зелёные
            _green_streak = 0
            for i in range(len(closes) - 1, -1, -1):
                if closes[i] > (df['open'].values[i] if 'open' in df else closes[i]):
                    _green_streak += 1
                else:
                    break
            features['consecutive_green_5m'] = min(_green_streak / 10.0, 1.0)

            # --- body_strength: отношение тела свечи к полному диапазону
            opens_arr = df['open'].values if 'open' in df else np.roll(closes, 1)
            _body = abs(closes[-1] - opens_arr[-1])
            _total_range = highs[-1] - lows[-1]
            features['body_strength'] = _body / (_total_range + 0.0001)  # 0=шипы, 1=сильная свеча

            # --- volume_divergence: расхождение цена↑ объём↓ за 5 свеч
            if len(closes) >= 6:
                _price_dir = 1 if closes[-1] > closes[-5] else (-1 if closes[-1] < closes[-5] else 0)
                _vol_trend = np.mean(volumes[-3:]) / (np.mean(volumes[-6:-3]) + 0.0001)  # >1 = объём растёт
                if _price_dir > 0 and _vol_trend < 0.85:
                    features['volume_divergence'] = min(1.0, (1.0 - _vol_trend) / 0.3)  # 0.85→0.5, 0.7→1.0
                elif _price_dir < 0 and _vol_trend > 1.15:
                    features['volume_divergence'] = min(1.0, (_vol_trend - 1.0) / 0.3)  # падение с объёмом
            
            # --- shrinking_candles: сколько свеч подряд диапазон уменьшается
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

        # --- RSI дивергенция: цена делает новый хай, RSI — нет
        if symbol not in self._rsi_history:
            self._rsi_history[symbol] = deque(maxlen=5)
        self._rsi_history[symbol].append(rsi)
        _rsi_list = list(self._rsi_history[symbol])
        if df is not None and len(df) >= 3 and len(_rsi_list) >= 3:
            closes_arr = df['close'].values
            _price_higher = closes_arr[-1] > closes_arr[-3]
            _rsi_lower = _rsi_list[-1] < _rsi_list[-3] - 2
            if _price_higher and _rsi_lower:
                features['rsi_divergence_bear'] = 1.0
            else:
                # Цена ниже, RSI выше = бычья дивергенция (потенциал роста)
                _price_lower = closes_arr[-1] < closes_arr[-3]
                _rsi_higher = _rsi_list[-1] > _rsi_list[-3] + 2
                if _price_lower and _rsi_higher:
                    features['rsi_divergence_bear'] = -0.5  # бычья дивергенция = небольшой бонус

        # BTC-корреляция
        try:
            df_btc = _fetch_btc_data()
            if df_btc is not None and len(df_btc) > 3:
                btc_closes = df_btc['close'].values
                features['btc_change_1h'] = (btc_closes[-1] - btc_closes[-2]) / btc_closes[-2]
        except Exception as _e:
            logger.debug("bare except in ml_advisor: %s", _e)
            pass

        # MultiTF из D1 и SMA20
        try:
            df_d1 = _fetch_tf_data(symbol, '1d', 60)
            if df_d1 is not None and len(df_d1) > 50:
                closes_d1 = df_d1['close'].values
                sma50 = np.mean(closes_d1[-50:])
                features['multi_tf'] = (current_price - sma50) / (sma50 + 0.0001)
                
                sma20 = np.mean(closes_d1[-20:])
                features['price_above_sma20'] = 1.0 if current_price > sma20 else 0.0
        except Exception as _e:
            logger.debug("bare except in ml_advisor: %s", _e)
            pass

        # VWAP из 1H
        try:
            df_1h = _fetch_tf_data(symbol, '1h', 24)
            if df_1h is not None and len(df_1h) > 12:
                closes_1h = df_1h['close'].values
                volumes_1h = df_1h['volume'].values
                vwap = np.sum(closes_1h[-12:] * volumes_1h[-12:]) / (np.sum(volumes_1h[-12:]) + 0.0001)
                features['vwap_dist'] = (current_price - vwap) / vwap
        except Exception as _e:
            logger.debug("bare except in ml_advisor: %s", _e)
            pass

        # 📏 from_24h_high: близость цены к 24h максимуму
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
            pass

        # 🧠 Память — поведение ММ на символе
        cl = self._consecutive_losses.get(safe_sym, 0)
        features['consecutive_losses'] = min(float(cl), 5.0)  # кап на 5
        features['loss_streak_active'] = 1.0 if cl >= 2 else 0.0
        recent = self._recent_trades.get(safe_sym, deque(maxlen=10))
        if len(recent) > 0:
            wins = sum(1 for r in recent if r > 0)
            features['recent_win_rate_10'] = wins / len(recent)
        
        # 📊 Портфельные метрики (обновляются трейдером)
        features['pf_daily_pnl'] = self._pf_daily_pnl
        tc = max(self._pf_daily_trade_count, 1)
        features['pf_daily_profit_ratio'] = self._pf_daily_profit_count / tc
        features['pf_daily_trade_count'] = min(self._pf_daily_trade_count / 50.0, 1.0)  # норм на 50
        features['pf_consecutive_profits'] = min(self._pf_consecutive_profits / 10.0, 1.0)  # норм на 10
        features['pf_consecutive_losses_global'] = min(self._pf_consecutive_losses_global / 5.0, 1.0)
        features['pf_open_positions_pct'] = self._pf_open_positions / 25.0  # 25 = max позиций
        features['pf_exposure_pct_norm'] = self._pf_exposure_pct / 100.0
        features['pf_avg_position_value_norm'] = min(self._pf_avg_position_value / 90.0, 1.0)
        
        return [features['rsi'], features['trend'], features['volatility'],
                features['volume_ratio'], features['multi_tf'], features['vwap_dist'],
                features['candle_doji'], features['candle_hammer'], features['candle_engulfing'],
                features['btc_change_1h'], features['hour_of_day'],
                features['volume_momentum'], features['hl_range'], features['price_above_sma20'],
                features['consecutive_losses'], features['loss_streak_active'],
                features['recent_win_rate_10'],
                features['pf_daily_pnl'], features['pf_daily_profit_ratio'],
                features['pf_daily_trade_count'], features['pf_consecutive_profits'],
                features['pf_consecutive_losses_global'], features['pf_open_positions_pct'],
                features['pf_exposure_pct_norm'], features['pf_avg_position_value_norm'],
                features['consecutive_green_5m'], features['from_24h_high'],
                features['body_strength'], features['volume_divergence'],
                features['rsi_divergence_bear'], features['shrinking_candles']]

    def update_portfolio_stats(self, daily_pnl=0.0, profit_count=0, loss_count=0,
                                trade_count=0, consecutive_profits=0,
                                consecutive_losses_global=0, open_positions=0,
                                exposure_pct=0.0, avg_position_value=0.0):
        """
        Обновить портфельные метрики (вызывается трейдером раз в цикл).
        Эти фичи будут использованы при следующем evaluate().
        """
        self._pf_daily_pnl = daily_pnl
        self._pf_daily_profit_count = profit_count
        self._pf_daily_loss_count = loss_count
        self._pf_daily_trade_count = trade_count
        self._pf_consecutive_profits = consecutive_profits
        self._pf_consecutive_losses_global = consecutive_losses_global
        self._pf_open_positions = open_positions
        self._pf_exposure_pct = exposure_pct
        self._pf_avg_position_value = avg_position_value

    def _feature_names(self):
        return ['rsi', 'trend', 'volatility', 'volume_ratio', 'multi_tf',
                'vwap_dist', 'candle_doji', 'candle_hammer', 'candle_engulfing',
                'btc_change_1h', 'hour_of_day', 'volume_momentum', 'hl_range', 'price_above_sma20',
                'consecutive_losses', 'loss_streak_active', 'recent_win_rate_10',
                'pf_daily_pnl', 'pf_daily_profit_ratio', 'pf_daily_trade_count',
                'pf_consecutive_profits', 'pf_consecutive_losses_global',
                'pf_open_positions_pct', 'pf_exposure_pct_norm', 'pf_avg_position_value_norm',
                'consecutive_green_5m', 'from_24h_high', 'body_strength',
                'volume_divergence', 'rsi_divergence_bear', 'shrinking_candles']

    def add_trade_result(self, symbol, entry_price, exit_price, rsi, trend, confidence,
                         hold_hours, reason, volume_ratio=None):
        """
        Добавляет результат сделки в обучающую выборку.
        Использует _extract_features для полного вектора признаков (31 фича).
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

        # Полный вектор через _extract_features (с entry_price на момент входа)
        full_features = self._extract_features(symbol, entry_price, rsi, trend, confidence, df_5m)

        sample = {
            'features': full_features,
            'label': label,
            'symbol': symbol,
            'pnl': pnl_pct,
            'reason': reason,
            'time': datetime.now().isoformat()
        }
        self.training_data.append(sample)
        self._save_training_data()

        logger.info(f"📚 ML: обучение на {symbol} (PnL={pnl_pct:+.2f}%, label={int(label)}, фич={len(full_features)})")

    def train(self, force=False):
        """
        Обучение/переобучение XGBoost модели на накопленных данных.
        """
        now = time.time()

        if not force and (now - self.last_retrain) < RETRAIN_INTERVAL:
            return

        if len(self.training_data) < MIN_TRAINING_SAMPLES:
            logger.info(f"⏳ ML: ждём данные ({len(self.training_data)}/{MIN_TRAINING_SAMPLES})")
            return

        # ═══ НОРМАЛИЗАЦИЯ ПРИЗНАКОВ ═══════════════════════════════════
        # Обратная совместимость: старые выборки с 9 признаками → дополняем нулями
        max_features = max(len(d['features']) for d in self.training_data)
        normalized_data = []
        for d in self.training_data:
            f = d['features']
            if len(f) < max_features:
                f = f + [0.0] * (max_features - len(f))
            normalized_data.append({'features': f, 'label': d['label']})

        X = np.array([d['features'] for d in normalized_data])
        y = np.array([d['label'] for d in normalized_data])

        # Проверка: если все метки одного класса — не обучаем
        if len(np.unique(y)) < 2:
            logger.warning(f"⚠️ ML: все данные одного класса ({int(y[0])}), ждём разнообразия")
            return

        # Масштабируем
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # XGBoost — лучше для табличных данных, чем RandomForest
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=1.0,
            reg_alpha=0.5,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0
        )
        self.model.fit(X_scaled, y, verbose=False)

        # Оценка точности
        train_score = self.model.score(X_scaled, y)
        self.is_trained = True
        self.last_retrain = now

        # Сохраняем
        self._save_model()

        n_good = int(y.sum())
        n_bad = len(y) - n_good
        logger.info(f"🎯 ML (XGBoost): обучен! Точность: {train_score:.1%}")
        logger.info(f"   Данных: {len(y)} (good={n_good}, bad={n_bad})")
        
        # Feature importance (топ-5)
        if hasattr(self.model, 'feature_importances_'):
            names = self._feature_names()
            importances = self.model.feature_importances_
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
        
        # Обратная совместимость: старая модель (14 фич) vs новая (17 фич)
        expected_features = getattr(self.model, 'n_features_in_', 14)
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
