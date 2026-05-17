#!/usr/bin/env python3
"""
btc_direction.py — BTC Direction Predictor.

Предсказывает направление BTC на 1-4 часа вперёд.
Главный модуль: анализирует ПРИЧИНУ (BTC), а не следствия (альты).

Архитектура:
  1. Data Pipeline: 1000+ свечей 1H BTC
  2. Feature Engineering: 60+ фич (RSI, MACD, EMA, Volume, HMM, ATR,
     VWAP, MFI, OBV, Time-сессии, свечные паттерны)
  3. LightGBM + XGBoost + RandomForest + ExtraTrees ансамбль
  4. Выход: сигнал силы и направления → DecisionEngine

Интеграция:
  - Тренируется раз в сутки (self._retrain())
  - Прогноз на каждый цикл execution (self.predict(current_btc_data))
  - Сигнал: {'direction': 'up'|'down'|'side', 'confidence': 0.0-1.0,
             'strength': 0-100, 'hours_ahead': 4}
"""

import os
import sys
import json
import time
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from collections import deque

# ML
from lightgbm import LGBMClassifier
from liquidity_cluster import LiquidityCluster
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier

logger = logging.getLogger('btc_direction')

# ─────────────────────────────────────────────
# КОНСТАНТЫ
# ─────────────────────────────────────────────
LOOKBACK_HOURS = 1000          # Сколько свечей 1H для обучения
TARGET_HOURS_AHEAD = 2         # На сколько часов вперёд предсказываем (было 4 — слишком долго для биткоина)
MIN_TRAIN_SAMPLES = 200        # Минимальное количество для обучения
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'btc_direction.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'data', 'btc_scaler.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'data', 'btc_features.json')
PREDICTIONS_LOG = os.path.join(os.path.dirname(__file__), 'data', 'btc_predictions.json')
TRAINING_LOG = os.path.join(os.path.dirname(__file__), 'data', 'btc_training_log.json')

# Параметры для классификации направления
UP_THRESHOLD = 0.005     # +0.5% за TARGET_HOURS_AHEAD = UP
DOWN_THRESHOLD = -0.004  # -0.4% = DOWN
SIDE_ZONE = 0.0015     # ±0.15% вокруг нуля = SIDE (было 0.3% — слишком широкая слепая зона)


class BTCDirectionPredictor:
    """
    BTC Direction Predictor — главный модуль предсказания направления BTC.
    
    Flow:
      1. Сбор свечей 1H BTC через exchange.fetch_ohlcv
      2. Расчёт 30+ технических фич
      3. LightGBM + XGBoost ансамбль
      4. Прогноз на 4 часа вперёд
      5. Сигнал для DecisionEngine
    """
    
    def __init__(self, exchange=None):
        self.exchange = exchange
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.last_train_time = time.time()  # Считаем что загруженная модель уже обучена
        self.retrain_interval = 21600  # Каждые 6 часов (было 24ч — не успевал за рынком)
        self.training_in_progress = False
        
        # Последний прогноз
        self.last_prediction = {
            'direction': 'side',
            'confidence': 0.0,
            'strength': 50,
            'hours_ahead': TARGET_HOURS_AHEAD,
            'timestamp': 0
        }
        
        # Кеш свечей для быстрого доступа
        self._candle_cache = deque(maxlen=LOOKBACK_HOURS + 100)
        self._last_fetch_time = 0
        self._fetch_cooldown = 300  # 5 минут между обновлением кеша
        
        # Кеш для calculate_bonus (HTTP запрос только раз в 60с)
        self._bonus_cache = {'result': 0, 'timestamp': 0}
        
        # Funding Rate / Open Interest (опционально)
        self.fr_oi_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data', 'btc_fr_oi.csv')
        self._fr_oi_data = None  # Кеш FR/OI DataFrame
        
        # Статистика
        self.stats = {
            'predictions': 0,
            'correct': 0,
            'wrong': 0,
            'accuracy': 0.0,
            'last_prediction_time': None,
            'features_calculated': 0
        }
        
        # Пытаемся загрузить существующую модель
        self._load_model()
        
        logger.info("🧠 BTC Direction Predictor инициализирован")
    
    # ════════════════════════════════════════
    # 1. DATA PIPELINE
    # ════════════════════════════════════════
    
    def _fetch_btc_ohlcv(self, limit: int = LOOKBACK_HOURS) -> Optional[pd.DataFrame]:
        """Загрузить свечи BTC 1H с биржи."""
        try:
            now = time.time()
            if now - self._last_fetch_time < self._fetch_cooldown and len(self._candle_cache) > 0:
                return self._candles_to_df(list(self._candle_cache))
            
            if not self.exchange:
                logger.warning("⚠️ BTCDirection: exchange не инициализирован")
                return None
            
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=limit)
            if not ohlcv or len(ohlcv) < 100:
                logger.warning(f"⚠️ BTCDirection: мало данных BTC ({len(ohlcv) if ohlcv else 0})")
                return None
            
            self._candle_cache.clear()
            self._candle_cache.extend(ohlcv)
            self._last_fetch_time = now
            
            df = self._candles_to_df(ohlcv)
            logger.info(f"📊 BTC: загружено {len(df)} свечей 1H")
            return df
            
        except Exception as e:
            logger.error(f"❌ BTCDirection: ошибка загрузки BTC: {e}")
            return None
    
    def _candles_to_df(self, candles: list) -> pd.DataFrame:
        """Преобразовать OHLCV в DataFrame."""
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    
    def _load_fr_oi(self) -> Optional[pd.DataFrame]:
        """Загрузить funding rate + open interest из CSV.
        
        Возвращает DataFrame с колонками:
          fr_rate, oi_amount, oi_value
        Индекс — datetime (UTC).
        """
        if self._fr_oi_data is not None:
            return self._fr_oi_data
        
        if not os.path.exists(self.fr_oi_path):
            logger.warning("⚠️ FR/OI CSV не найден — пропускаем")
            return None
        
        try:
            df = pd.read_csv(self.fr_oi_path)
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Конвертируем в числа (пустые строки → NaN)
            df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
            df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce')
            df['open_interest_value'] = pd.to_numeric(df['open_interest_value'], errors='coerce')
            
            # Переименуем для краткости
            df.rename(columns={
                'funding_rate': 'fr_rate',
                'open_interest': 'oi_amount',
                'open_interest_value': 'oi_value',
            }, inplace=True)
            
            # Сортируем по времени
            df.sort_index(inplace=True)
            
            # Убираем дубли по индексу (если FR и OI пришли отдельно для одного ts)
            df = df[~df.index.duplicated(keep='first')]
            
            self._fr_oi_data = df
            logger.info(f"📊 FR/OI: загружено {len(df)} записей (FR={df['fr_rate'].notna().sum()}, OI={df['oi_amount'].notna().sum()})")
            return df
        except Exception as e:
            logger.warning(f"⚠️ FR/OI загрузка: {e}")
            return None
    
    # ════════════════════════════════════════
    # 2. FEATURE ENGINEERING (30+ фич)
    # ════════════════════════════════════════
    
    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчёт 30+ фич на основе свечей BTC."""
        if df is None or len(df) < 50:
            return None
        
        df = df.copy()
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # ─── Базовые ═══════════════════
        df['returns_1h'] = df['close'].pct_change(1) * 100
        df['returns_4h'] = df['close'].pct_change(4) * 100
        df['returns_12h'] = df['close'].pct_change(12) * 100
        df['returns_24h'] = df['close'].pct_change(24) * 100
        
        df['range_1h'] = (df['high'] - df['low']) / df['close'] * 100
        df['range_4h'] = df['high'].rolling(4).max() - df['low'].rolling(4).min()
        df['range_24h'] = df['high'].rolling(24).max() - df['low'].rolling(24).min()
        
        # ─── RSI ═══════════════════════
        def _rsi(series, period=14):
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        
        df['rsi_14'] = _rsi(df['close'], 14)
        df['rsi_6'] = _rsi(df['close'], 6)
        df['rsi_divergence'] = df['rsi_14'] - df['rsi_6']
        
        # ─── MACD ══════════════════════
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_crossover'] = ((df['macd'] > df['macd_signal']) & 
                                (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        
        # ─── EMA ═══════════════════════
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_100'] = df['close'].ewm(span=100).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
        df['ema_dist_20'] = (df['close'] - df['ema_20']) / df['ema_20'] * 100
        df['ema_dist_50'] = (df['close'] - df['ema_50']) / df['ema_50'] * 100
        df['ema_dist_100'] = (df['close'] - df['ema_100']) / df['ema_100'] * 100
        df['ema_cross'] = ((df['ema_20'] > df['ema_50']) & 
                          (df['ema_20'].shift(1) <= df['ema_50'].shift(1))).astype(int)
        
        # ─── Bollinger Bands ═══════════
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_mid + 2 * bb_std
        df['bb_lower'] = bb_mid - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / bb_mid * 100
        df['bb_position'] = (df['close'] - bb_mid) / (bb_std + 1e-10)
        df['bb_volatility_break'] = (df['bb_width'] > df['bb_width'].rolling(100).mean() * 1.5).astype(int)
        
        # ─── ATR / Волатильность ═══════
        tr = pd.concat([
            pd.Series(high - low, index=df.index),
            pd.Series(abs(high - close), index=df.index),
            pd.Series(abs(low - close), index=df.index)
        ], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr_14'] / close * 100
        df['volatility_24h'] = df['returns_1h'].rolling(24).std() * 100
        
        # ─── Объём ═════════════════════
        df['volume_ma_24'] = volume / df['volume'].rolling(24).mean()
        df['volume_ma_4'] = volume / (df['volume'].rolling(4).mean() + 1e-10)
        df['volume_trend'] = df['volume_ma_4'] / (df['volume_ma_24'] + 1e-10)
        df['volume_vs_avg'] = (volume - df['volume'].rolling(100).mean()) / (df['volume'].rolling(100).std() + 1e-10)
        
        # ─── Ценовые паттерны ══════════
        df['body_ratio'] = abs(close - df['open']) / (high - low + 1e-10)
        df['upper_wick'] = (high - df[['open','close']].max(axis=1)) / (high - low + 1e-10)
        df['lower_wick'] = (df[['open','close']].min(axis=1) - low) / (high - low + 1e-10)
        
        # Зелёные свечи подряд
        df['green_candle'] = (close > df['open']).astype(int)
        df['green_streak'] = df['green_candle'].astype(int).groupby(
            (df['green_candle'] != df['green_candle'].shift()).cumsum()).cumsum()
        df['red_streak'] = (1 - df['green_candle']).groupby(
            (df['green_candle'] == df['green_candle'].shift()).cumsum()).cumsum()
        
        # ─── VWAP — ключевой институциональный уровень ════
        vwap = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).rolling(24).sum() / df['volume'].rolling(24).sum()
        df['vwap'] = vwap
        df['vwap_dist'] = (df['close'] - vwap) / vwap * 100  # % от VWAP
        df['vwap_cross'] = ((df['close'] > vwap) & (df['close'].shift(1) <= vwap.shift(1))).astype(int)
        df['vwap_slope'] = vwap.diff(4) / vwap * 100  # наклон VWAP за 4ч
        
        # ─── MFI — Money Flow Index (объём + цена) ════════
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        mfi_ratio = positive_flow / negative_flow.replace(0, np.nan)
        df['mfi_14'] = 100 - (100 / (1 + mfi_ratio))
        df['mfi_14'] = df['mfi_14'].fillna(50)
        df['mfi_divergence'] = df['mfi_14'] - df['rsi_14']  # расхождение объёма и цены
        
        # ─── OBV — On-Balance Volume тренд ═══════════════
        obv = (df['volume'] * ((df['close'] > df['close'].shift(1)).astype(int) * 2 - 1)).cumsum()
        df['obv'] = obv
        df['obv_ema'] = obv.ewm(span=20).mean()
        df['obv_slope'] = (obv - obv.shift(12)) / obv.shift(12).abs() * 100
        
        # ─── Время — внутридневные паттерны BTC ════════════
        hour = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        # Азиатская/Европейская/Американская сессия
        df['session_asia'] = ((hour >= 0) & (hour < 8)).astype(int)
        df['session_europe'] = ((hour >= 8) & (hour < 16)).astype(int) 
        df['session_usa'] = ((hour >= 16) & (hour < 24)).astype(int)
        
        # ─── HMM режим ═════════════════
        # Простая proxy-оценка волатильности по ценам
        vol_20 = df['returns_1h'].rolling(20).std().fillna(0.5)
        df['hmm_proxy'] = pd.cut(vol_20, bins=[-0.01, 0.5, 1.5, 100], labels=[0, 1, 2]).astype(float).fillna(1).astype(int)
        
        # ─── Лаги ═════════════════════
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # ─── Funding Rate / Open Interest ═══════════
        fr_oi = self._load_fr_oi()
        if fr_oi is not None and len(fr_oi) > 0:
            # Merge_asof: каждой 1H свече — последний известный FR (8H) и OI (1H)
            df_aligned = pd.merge_asof(
                df[[]], fr_oi[['fr_rate', 'oi_amount', 'oi_value']],
                left_index=True, right_index=True,
                direction='backward', tolerance=pd.Timedelta('8h')
            )
            df['fr_rate'] = df_aligned['fr_rate']
            df['oi_amount'] = df_aligned['oi_amount']
            df['oi_value'] = df_aligned['oi_value']
            
            # Forward-fill: FR действителен до следующего funding
            df['fr_rate'] = df['fr_rate'].ffill(limit=8)
            
            # Производные признаки
            df['fr_change_8h'] = df['fr_rate'].diff(8) * 10000  # в б.п.
            df['oi_change_24h'] = df['oi_amount'].pct_change(24) * 100
            df['oi_change_4h'] = df['oi_amount'].pct_change(4) * 100
            
            # Заполняем NaN для старых свечей (где нет OI)
            for col in ['fr_rate', 'oi_amount', 'oi_value', 'fr_change_8h', 'oi_change_24h', 'oi_change_4h']:
                df[col] = df[col].fillna(0)
        else:
            # Нет данных — нулевые значения
            df['fr_rate'] = 0.0
            df['oi_amount'] = 0.0
            df['oi_value'] = 0.0
            df['fr_change_8h'] = 0.0
            df['oi_change_24h'] = 0.0
            df['oi_change_4h'] = 0.0
        
        # ─── Таргет ════════════════════
        future_close = df['close'].shift(-TARGET_HOURS_AHEAD)
        future_return = (future_close - df['close']) / df['close']
        
        df['target'] = 1  # SIDE
        df.loc[future_return > UP_THRESHOLD, 'target'] = 2  # UP
        df.loc[future_return < DOWN_THRESHOLD, 'target'] = 0  # DOWN
        
        # ═══════════════════════════════════
        # Финальная чистка: удаляем NaN
        # ═══════════════════════════════════
        feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume',
                                                           'timestamp', 'target'] 
                       and not c.startswith('close_lag')]
        
        df = df.dropna()
        
        self.feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'target']]
        self.stats['features_calculated'] = len(self.feature_cols)
        
        logger.debug(f"🧮 BTC: расчитано {len(self.feature_cols)} фич на {len(df)} строках")
        
        return df
    
    # ════════════════════════════════════════
    # 3. ML MODEL: LightGBM + XGBoost
    # ════════════════════════════════════════
    
    def train(self) -> bool:
        """Тренировка ансамбля LightGBM + XGBoost на BTC данных."""
        if self.training_in_progress:
            logger.warning("⚠️ BTCDirection: обучение уже идёт")
            return False
        
        self.training_in_progress = True
        try:
            logger.info("🔬 BTC Direction: начало обучения")
            
            df = self._fetch_btc_ohlcv()
            if df is None or len(df) < MIN_TRAIN_SAMPLES:
                logger.warning(f"⚠️ BTC Direction: недостаточно данных ({len(df) if df is not None else 0})")
                return False
            
            df_features = self._calculate_features(df)
            if df_features is None or len(df_features) < MIN_TRAIN_SAMPLES:
                logger.warning(f"⚠️ BTC Direction: недостаточно фич ({len(df_features) if df_features is not None else 0})")
                return False
            
            X = df_features[self.feature_cols].values
            y = df_features['target'].values
            
            # Балансировка классов
            class_counts = pd.Series(y).value_counts()
            logger.info(f"📊 BTC классы: UP={class_counts.get(2, 0)}, DOWN={class_counts.get(0, 0)}, SIDE={class_counts.get(1, 0)}")
            
            # Разделение на train/test — временное (последние 20% по времени)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Масштабирование
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # ─── Без SMOTE — используем оригинальные данные с агрессивными весами ═══
            # SMOTE понижает точность на тесте (тестовые данные не сбалансированы)
            # Вместо этого — сильные веса для UP/DOWN
            X_train_bal, y_train_bal = X_train_scaled, y_train
            
            # ─── LightGBM ═════════════════════
            lgbm = LGBMClassifier(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )
            
            # ─── XGBoost ═════════════════════
            up_ratio = max(class_counts.get(2, 1), 1) / max(class_counts.get(0, 1), 1)
            xgb = XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbosity=0
            )
            
            # ─── Веса классов для повышения чувствительности к UP/DOWN ═══
            # Агрессивные веса: UP=5x, DOWN=3x (UP важнее не пропустить)
            class_weights = {0: 3.0, 1: 1.0, 2: 5.0}
            sample_weight_arr = np.ones(len(X_train_bal))
            for cls, w in class_weights.items():
                sample_weight_arr[y_train_bal == cls] = w
            
            # ─── Ансамбль (LightGBM + XGBoost) ═══════════════
            ensemble = VotingClassifier(
                estimators=[('lgbm', lgbm), ('xgb', xgb)],
                voting='soft'
            )
            
            # Обучение
            ensemble.fit(X_train_bal, y_train_bal, sample_weight=sample_weight_arr)
            
            # Оценка на тесте
            y_pred = ensemble.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Детальный отчёт
            report = classification_report(y_test, y_pred, target_names=['DOWN', 'SIDE', 'UP'],
                                           output_dict=True, zero_division=0)
            up_precision = report.get('UP', {}).get('precision', 0)
            up_recall = report.get('UP', {}).get('recall', 0)
            down_precision = report.get('DOWN', {}).get('precision', 0)
            down_recall = report.get('DOWN', {}).get('recall', 0)
            side_precision = report.get('SIDE', {}).get('precision', 0)
            side_recall = report.get('SIDE', {}).get('recall', 0)
            
            logger.info(f"🎯 BTC Direction: accuracy={accuracy:.1%}, "
                       f"UP precision={up_precision:.1%} recall={up_recall:.1%}, "
                       f"DOWN precision={down_precision:.1%} recall={down_recall:.1%}, "
                       f"SIDE precision={side_precision:.1%} recall={side_recall:.1%}")
            
            # Сохраняем лог обучения
            try:
                training_records = []
                if os.path.exists(TRAINING_LOG):
                    with open(TRAINING_LOG, 'r') as f:
                        training_records = json.load(f)
                training_records.append({
                    'ts': datetime.now().isoformat(),
                    'accuracy': round(accuracy, 3),
                    'up_precision': round(up_precision, 3),
                    'up_recall': round(up_recall, 3),
                    'down_precision': round(down_precision, 3),
                    'down_recall': round(down_recall, 3),
                    'side_precision': round(side_precision, 3),
                    'side_recall': round(side_recall, 3),
                    'features': len(self.feature_cols) if self.feature_cols else 0,
                    'samples': len(df_features),
                    'class_balance': {
                        'up': int(class_counts.get(2, 0)),
                        'down': int(class_counts.get(0, 0)),
                        'side': int(class_counts.get(1, 0)),
                    },
                })
                if len(training_records) > 50:
                    training_records = training_records[-50:]
                with open(TRAINING_LOG, 'w') as f:
                    json.dump(training_records, f, indent=2)
            except Exception as e_log:
                logger.debug(f"[TrainingLog] save: {e_log}")

            # Сохраняем модель
            self.model = ensemble
            self._save_model()
            self.last_train_time = time.time()
            
            # Обновляем статистику
            self.stats['accuracy'] = accuracy
            self.stats['last_train_time'] = datetime.now().isoformat()
            
            logger.info(f"✅ BTC Direction: модель обучена (точность {accuracy:.1%})")
            return True
            
        except Exception as e:
            logger.error(f"❌ BTC Direction: ошибка обучения: {e}", exc_info=True)
            return False
        finally:
            self.training_in_progress = False
    
    # ════════════════════════════════════════
    # 4. ПРОГНОЗ
    # ════════════════════════════════════════
    
    def predict(self, candles_1h: Optional[List] = None) -> Dict:
        """
        Предсказать направление BTC.
        
        Returns:
            {'direction': 'up'|'down'|'side',
             'confidence': 0.0-1.0,
             'strength': 0-100,
             'hours_ahead': int,
             'timestamp': int}
        """
        now = time.time()
        
        # Проверяем, нужно ли переобучить
        if now - self.last_train_time > self.retrain_interval:
            logger.info("🔄 BTC Direction: плановое переобучение")
            self.train()
        
        if self.model is None:
            logger.warning("⚠️ BTC Direction: модель не обучена")
            return self._default_signal()
        
        try:
            # Берём свежие данные
            if candles_1h is not None:
                df = self._candles_to_df(candles_1h)
            else:
                df = self._fetch_btc_ohlcv(limit=200)
            
            if df is None or len(df) < 50:
                return self._default_signal()
            
            # Рассчитываем фичи
            df_features = self._calculate_features(df)
            if df_features is None or len(df_features) < 10:
                return self._default_signal()
            
            # Берём последнюю строку для прогноза
            last_row = df_features.iloc[-1:]
            X = last_row[self.feature_cols].values
            
            if len(X) == 0 or X.shape[1] == 0:
                return self._default_signal()
            
            try:
                X_scaled = self.scaler.transform(X)
            except Exception as _e:
                logger.debug(f"[BTCDirection] scaler.transform: {_e}")
                return self._default_signal()
            
            # Прогноз вероятностей
            probs = self.model.predict_proba(X_scaled)[0]
            
            # proba: классы 0=DOWN, 1=SIDE, 2=UP
            up_prob = probs[2] if len(probs) > 2 else 0.33
            down_prob = probs[0] if len(probs) > 0 else 0.33
            side_prob = probs[1] if len(probs) > 1 else 0.33
            
            # Определяем направление — порог 0.40 даёт меньше сигналов, но выше точность
            # После дообучения с весом UP=8x модель должна быть увереннее
            if up_prob > down_prob and up_prob > side_prob and up_prob > 0.40:
                direction = 'up'
                confidence = up_prob
                strength = min(int(up_prob * 100), 100)
            elif down_prob > up_prob and down_prob > side_prob and down_prob > 0.40:
                direction = 'down'
                confidence = down_prob
                strength = min(int(down_prob * 100), 100)
            else:
                direction = 'side'
                confidence = max(side_prob, max(up_prob, down_prob))
                strength = 50  # нейтрально
            
            # Бонус: смотрим последние 3 свечи
            last_3 = df.iloc[-3:]
            up_candles = sum(1 for _, r in last_3.iterrows() if r['close'] > r['open'])
            price_trend = (last_3['close'].iloc[-1] - last_3['close'].iloc[0]) / last_3['close'].iloc[0] * 100
            
            # Если визуально тренд и прогноз совпадают — усиливаем
            if (direction == 'up' and up_candles >= 2 and price_trend > 0) or \
               (direction == 'down' and up_candles < 2 and price_trend < 0):
                strength = min(int(strength * 1.2), 100)
            
            # 🧊 Liq-бонус: корректировка направления на основе LiquidityCluster (POC/VAH/VAL/FVG)
            liq_boost = self._get_liq_boost(float(df['close'].iloc[-1]))
            if liq_boost:
                boost_dir = liq_boost.get('direction')
                boost_pts = liq_boost.get('points', 0)
                if boost_dir == direction and boost_pts > 0:
                    # Согласование: усиливаем уверенность
                    old_strength = strength
                    strength = min(strength + boost_pts, 100)
                    logger.info(f"🧊 Liq-бонус: {boost_dir}↑ +{boost_pts}pts ({liq_boost['reason']}) → {old_strength}→{strength}")
                elif boost_dir and boost_dir != direction and abs(boost_pts) >= 10:
                    # Противоречие: ослабляем уверенность
                    old_strength = strength
                    penalty = min(abs(boost_pts), 15)
                    strength = max(strength - penalty, 10)
                    logger.info(f"🧊 Liq-пенальти: {boost_dir}≠{direction} -{penalty}pts ({liq_boost['reason']}) → {old_strength}→{strength}")
            
            # 📊 Трекинг предсказаний: сохраняем + валидируем старые
            self._save_prediction(direction, float(df['close'].iloc[-1]), strength)
            self._validate_predictions()
            
            self.last_prediction = {
                'direction': direction,
                'confidence': round(confidence, 3),
                'strength': strength,
                'hours_ahead': TARGET_HOURS_AHEAD,
                'timestamp': now,
                'up_probability': round(float(up_prob), 3),
                'down_probability': round(float(down_prob), 3),
                'side_probability': round(float(side_prob), 3),
                'current_price': float(df['close'].iloc[-1]),
                'price_trend_3h': round(price_trend, 2),
                'features': len(self.feature_cols) if self.feature_cols else 0
            }
            
            self.stats['predictions'] += 1
            self.stats['last_prediction_time'] = datetime.now().isoformat()
            
            logger.info(f"🔮 BTC Direction: {direction} (conf={confidence:.0%}, "
                       f"strength={strength}, up={up_prob:.0%} down={down_prob:.0%} side={side_prob:.0%})")
            
            # 🧊 BTC Liquidity: рассчитываем и логируем для дашборда
            self._log_btc_liquidity()
            
            return self.last_prediction
            
        except Exception as e:
            logger.error(f"❌ BTC Direction: ошибка прогноза: {e}")
            return self._default_signal()
    
    def _default_signal(self) -> Dict:
        """Сигнал по умолчанию — нейтральный."""
        return {
            'direction': 'side',
            'confidence': 0.5,
            'strength': 50,
            'hours_ahead': TARGET_HOURS_AHEAD,
            'timestamp': time.time(),
            'up_probability': 0.33,
            'down_probability': 0.33,
            'side_probability': 0.34,
            'current_price': 0,
            'price_trend_3h': 0,
            'features': 0
        }
    
    def get_signal(self) -> Dict:
        """Получить текущий сигнал (кешированный за последние 5 минут)."""
        if time.time() - self.last_prediction['timestamp'] > 300:
            return self.predict()
        return self.last_prediction
    
    # ════════════════════════════════════════
    # 5. COLD STORAGE
    # ════════════════════════════════════════
    
    def _save_model(self):
        """Сохранить модель и скалер на диск."""
        if self.model is None:
            return
        
        try:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
            with open(SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Сохраняем список фич
            features_info = {
                'feature_cols': self.feature_cols,
                'target_hours': TARGET_HOURS_AHEAD,
                'up_threshold': UP_THRESHOLD,
                'down_threshold': DOWN_THRESHOLD,
                'trained_at': datetime.now().isoformat()
            }
            with open(FEATURES_PATH, 'w') as f:
                json.dump(features_info, f, indent=2)
            
            logger.info(f"💾 BTC Direction: модель сохранена")
        except Exception as e:
            logger.error(f"❌ BTC Direction: ошибка сохранения модели: {e}")
    
    def _load_model(self):
        """Загрузить модель с диска."""
        try:
            if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
                logger.info("ℹ️ BTC Direction: модель не найдена, будет обучена при первом запуске")
                self.last_train_time = 0  # Форсируем обучение при первом predict() — убираем 6ч задержку
                return False
            
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            
            if os.path.exists(FEATURES_PATH):
                with open(FEATURES_PATH) as f:
                    info = json.load(f)
                    self.feature_cols = info.get('feature_cols', [])
            
            logger.info(f"💾 BTC Direction: модель загружена ({len(self.feature_cols) if self.feature_cols else 0} фич)")
            return True
            
        except Exception as e:
            logger.error(f"❌ BTC Direction: ошибка загрузки модели: {e}")
            return False
    
    # ════════════════════════════════════════
    # 6. INTEGRATION с DecisionEngine
    # ════════════════════════════════════════
    
    def calculate_bonus(self, current_score: float) -> float:
        """
        Рассчитать бонус/штраф для DecisionEngine на основе прогноза BTC.
        
        Если BTC растёт — бонус до +15 баллов.
        Если BTC падает — штраф -50 баллов (блокировка).
        Если BTC в боковике — нейтрально.
        
        Args:
            current_score: Текущий Score альт-монеты
            
        Returns:
            Модификация Score (может быть отрицательной)
        """
        # ═══ КЭШ HTTP-ЗАПРОСОВ ═══
        # calculate_bonus вызывается на КАЖДУЮ монету в цикле (20-30 раз за итерацию).
        # Данные BTC меняются раз в 5 минут — нет смысла долбить API каждые 2 секунды.
        now = time.time()
        if now - self._bonus_cache['timestamp'] < 60:
            return self._bonus_cache['result']
        
        # ═══ BTC ДИНАМИЧЕСКИЙ ФИЛЬТР ═══
        # Следим за изменением BTC за разные окна через биржу.
        # Если не можем получить данные — пропускаем (безопасный режим).
        
        try:
            import requests
            import numpy as np
            
            # Берём свечи 6h (72 свечи 5мин) для расчёта изменений
            d = requests.get(
                'https://api.bybit.com/v5/market/kline?category=spot&symbol=BTCUSDT&interval=5&limit=72',
                timeout=5
            ).json()
            
            if 'result' not in d or 'list' not in d['result']:
                self._bonus_cache = {'result': 0, 'timestamp': now}
                return 0
            
            candles = d['result']['list']
            closes = np.array([float(c[4]) for c in candles], dtype=float)
            
            if len(closes) < 72:
                self._bonus_cache = {'result': 0, 'timestamp': now}
                return 0
            
            # Рассчитываем изменения за разные окна
            price_3h = closes[-1]
            price_3h_ago = closes[-36]   # 36 свечей * 5мин = 3 часа
            price_6h_ago = closes[0]      # 72 свечи * 5мин = 6 часов
            
            change_3h = (price_3h - price_3h_ago) / price_3h_ago * 100
            change_6h = (price_3h - price_6h_ago) / price_6h_ago * 100
            
            bonus = 0
            
            # 1. Блокировка при сильном падении
            if change_6h < -2.0:
                # Вето на все лонги — BTC упал >2% за 6 часов
                self._bonus_cache = {'result': -999, 'timestamp': now}
                return -999
            
            # 2. Штраф при падении
            if change_3h < -1.0:
                # Рынок падает — штрафуем лонги
                penalty = min(25, abs(change_3h) * 12)  # -1%→-12, -2%→-24
                bonus = -penalty
            elif change_3h < -0.5:
                bonus = -5  # лёгкое предупреждение (-0.5..-1%)
            
            # 3. Бонус при росте (усилен — чтобы отскоки давали шанс альткоинам)
            elif change_3h > 1.0:
                # Сильный отскок за 3 часа — даём бонус
                bonus = min(15, int(change_3h * 10))  # +1%→+10, +2%→+15
            elif change_3h > 0.5:
                # Умеренный отскок — небольшой бонус
                bonus = min(8, int(change_3h * 12))   # +0.5%→+6, +1%→+8
            elif change_6h > 1.0:
                # Рост за 6 часов — уверенность выше
                bonus = min(12, int(change_6h * 8))   # +1%→+8, +1.5%→+12
            
            # 4. Объёмный мультипликатор: если движение на высоком объёме, бонус сильнее
            if bonus > 0 and len(closes) >= 72:
                volumes = np.array([float(c[5]) for c in candles], dtype=float)[-36:]  # последние 3ч
                avg_vol = np.mean(volumes)
                vol_ratio = volumes[-1] / (avg_vol + 1e-10)
                if vol_ratio > 1.5:
                    bonus = int(bonus * 1.5)  # высокий объём → бонус x1.5
                    logger.debug(f"   ↑ объёмный отскок (x{vol_ratio:.1f}): бонус усилен до {bonus:+.0f}")
            
            # 5. ML модели — дополнительное давление на выход при прогнозе DOWN
            # Используется ТОЛЬКО для усиления продаж (не для входа)
            try:
                if self.model is not None and hasattr(self, 'last_prediction') and self.last_prediction:
                    ml_dir = self.last_prediction.get('direction', 'side')
                    ml_strength = self.last_prediction.get('strength', 50)
                    if ml_dir == 'down' and ml_strength >= 50:
                        ml_penalty = min(-10, -(ml_strength // 5))  # -10 to -20
                        bonus += ml_penalty
                        logger.info(f"🔻 BTC ML exit: ML={ml_dir}({ml_strength}) → add {ml_penalty:+.0f}pts (итог: {bonus:+.0f})")
            except Exception as e_ml:
                logger.debug(f"BTC ML exit error: {e_ml}")
            
            logger.debug(f"BTC bonus: 3h={change_3h:+.1f}% 6h={change_6h:+.1f}% → bonus={bonus:+.0f}")
            self._bonus_cache = {'result': bonus, 'timestamp': now}
            return bonus
        except Exception as e:
            logger.debug(f"BTC bonus error: {e}")
            self._bonus_cache = {'result': 0, 'timestamp': now}
            return 0


# ──────────────────────────────────────────────────────────────────────────────
# BTC Liquidity: расчёт Liq/POC/OB/FVG для дашборда
# ──────────────────────────────────────────────────────────────────────────────

    def _fetch_btc_5m(self, limit: int = 96) -> Optional[List]:
        """Загрузить 5-минутные свечи BTC для расчёта ликвидности."""
        try:
            if not self.exchange:
                return None
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '5m', limit=limit)
            return ohlcv if ohlcv else None
        except Exception as e:
            logger.debug(f"[BTCLiq] fetch 5m: {e}")
            return None

    def _log_btc_liquidity(self):
        """Рассчитать Liq/POC/OB/FVG для BTC и залогировать для дашборда.
        
        Формат лога копирует decision_engine.py для совместимости с control_api.py:
          Liq:score(POC=... VAH=... VAL=... q=... fvg↑=f fvg↓=f ...[OB+]...[POC↑|↓|flat)
        """
        candles_5m = self._fetch_btc_5m(limit=96)  # 8 часов
        if not candles_5m or len(candles_5m) < 20:
            logger.debug("[BTCLiq] нет 5M свечей — пропускаем")
            return
        
        try:
            current_price = candles_5m[-1][4]
            lc = LiquidityCluster(window=48)
            result = lc.evaluate(candles_5m, current_price)
            
            if result['state'] is None:
                logger.debug("[BTCLiq] state is None")
                return
            
            state = result['state']
            score = result['score']
            signal = result['signal']
            
            # Форматируем как в decision_engine: Liq:80(POC=0.1334 VAH=0.1336 VAL=0.1332 q=0.30 fvg↑=0 fvg↓=0)
            poc_trend = '↑' if current_price > state.poc else ('↓' if current_price < state.poc else '')
            ob_str = ''
            for ob in state.order_blocks:
                if ob.kind == 'idm':
                    ob_str = f" OB{1} idm {ob.direction}"
                elif ob.kind == 'idm_ob':
                    ob_str = f" OB{2} idm_ob {ob.direction}"
                elif ob.kind == 'ext_ob':
                    ob_str = f" OB{3} ext_ob {ob.direction}"
            
            detail = (
                f"Liq:{score}("
                f"POC={state.poc:.4f} VAH={state.vah:.4f} VAL={state.val:.4f} "
                f"q={state.cluster_quality:.2f} "
                f"fvg↑={len(state.fvg_above)} fvg↓={len(state.fvg_below)} "
                f"{ob_str}"
                f" POC{poc_trend})"
            )
            logger.info(f"[BTCLiq] BTC/USDT: {detail}")
        except Exception as e:
            logger.debug(f"[BTCLiq] error: {e}")

    def _get_liq_boost(self, current_price: float) -> Optional[Dict]:
        """Рассчитать Liq-бонус для коррекции направления BTC.
        
        Использует LiquidityCluster для определения:
        - Позиция цены относительно POC (выше = бычий сигнал, ниже = медвежий)
        - Ширина Value Area (узкая = зона консолидации перед движением)
        - FVG (незакрытые гэпы как магниты)
        - Ордерблоки (зоны институциональных входов)
        
        Returns:
            {'direction': 'up'|'down'|None, 'points': int, 'reason': str} или None
        """
        candles_5m = self._fetch_btc_5m(limit=96)
        if not candles_5m or len(candles_5m) < 20:
            return None
        
        try:
            lc = LiquidityCluster(window=48)
            result = lc.evaluate(candles_5m, current_price)
            
            if result['state'] is None:
                return None
            
            state = result['state']
            score = result['score']
            
            # Позиция цены относительно POC
            poc_dist_pct = (current_price - state.poc) / state.poc * 100
            val_width = (state.vah - state.val) / state.poc * 100 if state.poc > 0 else 0
            
            reasons = []
            boost_points = 0
            boost_dir = None
            
            # 1. POC-сигнал: цена выше POC = бычий, ниже = медвежий
            if poc_dist_pct > 0.3:
                boost_dir = 'up'
                boost_points = 8
                reasons.append(f"POC↑ +{poc_dist_pct:.2f}%")
            elif poc_dist_pct < -0.3:
                boost_dir = 'down'
                boost_points = 8
                reasons.append(f"POC↓ {poc_dist_pct:.2f}%")
            
            # 2. FVG: незакрытые гэпы — магниты
            if state.fvg_above:
                if boost_dir is None or boost_dir == 'up':
                    boost_points += 5
                    boost_dir = 'up'
                reasons.append(f"FVG↑{len(state.fvg_above)}")
            if state.fvg_below:
                if boost_dir is None or boost_dir == 'down':
                    boost_points += 5
                    boost_dir = 'down'
                reasons.append(f"FVG↓{len(state.fvg_below)}")
            
            # 3. Ордерблоки — институциональные зоны
            bullish_obs = [ob for ob in state.order_blocks if ob.direction == 'bullish' and not ob.mitigated]
            bearish_obs = [ob for ob in state.order_blocks if ob.direction == 'bearish' and not ob.mitigated]
            if bullish_obs:
                boost_points += 5
                if boost_dir is None:
                    boost_dir = 'up'
                reasons.append(f"OB↑{len(bullish_obs)}")
            if bearish_obs:
                boost_points += 5
                if boost_dir is None:
                    boost_dir = 'down'
                reasons.append(f"OB↓{len(bearish_obs)}")
            
            # 4. Узкая Value Area = накопление/распределение
            if val_width < 0.5 and score >= 70:
                boost_points += 5
                reasons.append(f"VA tight({val_width:.2f}%)")
            
            # 5. Качество кластера — чем выше, тем надёжнее сигнал
            if score >= 85:
                boost_points += 3
                reasons.append(f"Liq=85+")
            elif score <= 40:
                boost_points -= 3
                reasons.append(f"Liq={score} слабая")
            
            if not reasons or boost_points == 0:
                return None
            
            return {
                'direction': boost_dir,
                'points': min(boost_points, 25),
                'reason': ', '.join(reasons),
                'score': score,
                'poc_dist': round(poc_dist_pct, 2),
            }
        except Exception as e:
            logger.debug(f"[BTCLiq] boost error: {e}")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # EARN SIGNAL — сигнал для управления BTC в EARN (ручное перезакупы)
    # ──────────────────────────────────────────────────────────────────────────

    def get_earn_signal(self) -> Dict:
        """
        Сигнал для ручного управления BTC в EARN.
        
        Логика:
          - ML DOWN + strength≥50: SELL (выход из BTC в USDT)
          - ML UP + strength≥60: BUY (возврат в BTC из USDT)
          - Иначе: HOLD (без действий)
        
        Усилители:
          - POC trend DOWN → сигнал SELL сильнее
          - POC trend UP → сигнал BUY сильнее
        
        Returns:
            {'signal': 'SELL'|'BUY'|'HOLD', 'confidence': str, 'reason': str,
             'direction': str, 'strength': int, 'hours_ahead': int}
        """
        result = {
            'signal': 'HOLD',
            'confidence': 'neutral',
            'reason': 'Нет уверенного сигнала',
            'direction': 'side',
            'strength': 50,
            'hours_ahead': TARGET_HOURS_AHEAD,
            'validated_accuracy': None,
        }
        
        try:
            # Берём последнее предсказание ML
            if not hasattr(self, 'last_prediction') or not self.last_prediction:
                # Пробуем сделать свежий predict
                pred = self.predict()
            else:
                pred = self.last_prediction
            
            direction = pred.get('direction', 'side')
            strength = pred.get('strength', 50)
            result['direction'] = direction
            result['strength'] = strength
            
            # Статистика валидации для отображения
            validation = self._validate_predictions()
            if validation and validation.get('total', 0) >= 10:
                result['validated_accuracy'] = validation['accuracy']
            
            # Liq/POC данные для усиления
            liq_trend_down = False
            liq_trend_up = False
            for line in self._bonus_cache_log if hasattr(self, '_bonus_cache_log') else []:
                if 'POC↓' in str(line):
                    liq_trend_down = True
                if 'POC↑' in str(line):
                    liq_trend_up = True
            
            # SELL сигнал: ML говорит DOWN с достаточной уверенностью
            if direction == 'down' and strength >= 50:
                result['signal'] = 'SELL'
                reasons = [f"ML=down({strength})"]
                
                if strength >= 70:
                    result['confidence'] = 'high'
                elif strength >= 60:
                    result['confidence'] = 'medium'
                else:
                    result['confidence'] = 'low'
                
                if liq_trend_down:
                    reasons.append("POC↓")
                    if result['confidence'] == 'low':
                        result['confidence'] = 'medium'
                
                result['reason'] = ' | '.join(reasons)
                logger.info(f"🏦 EARN SELL: {result['reason']} (conf={result['confidence']})")
            
            # BUY сигнал: ML говорит UP с достаточной уверенностью
            elif direction == 'up' and strength >= 60:
                result['signal'] = 'BUY'
                reasons = [f"ML=up({strength})"]
                
                if strength >= 80:
                    result['confidence'] = 'high'
                elif strength >= 70:
                    result['confidence'] = 'medium'
                else:
                    result['confidence'] = 'low'
                
                if liq_trend_up:
                    reasons.append("POC↑")
                
                result['reason'] = ' | '.join(reasons)
                logger.info(f"🏦 EARN BUY: {result['reason']} (conf={result['confidence']})")
            
            # HOLD
            else:
                result['signal'] = 'HOLD'
                if direction == 'side':
                    result['reason'] = f'ML=side({strength}) — рынок в боковике'
                elif direction == 'up' and strength < 60:
                    result['reason'] = f'ML=up({strength}) — недостаточно уверенности для BUY'
                elif direction == 'down' and strength < 50:
                    result['reason'] = f'ML=down({strength}) — сигнал слабый, ждём усиления'
                logger.debug(f"🏦 EARN HOLD: {result['reason']}")
            
            return result
        except Exception as e:
            logger.debug(f"[EARN] сигнал: {e}")
            return result

    # ──────────────────────────────────────────────────────────────────────────
    # ТРЕКИНГ ПРЕДСКАЗАНИЙ — накопление статистики точности
    # ──────────────────────────────────────────────────────────────────────────

    def _save_prediction(self, direction: str, price: float, strength: int):
        """Сохранить предсказание для последующей валидации через 2ч."""
        try:
            now_ts = time.time()
            record = {
                'ts': now_ts,
                'direction': direction,
                'price': price,
                'strength': strength,
                'validated': False,
                'correct': None,
                'actual_direction': None,
                'price_change_pct': None,
            }
            records = []
            if os.path.exists(PREDICTIONS_LOG):
                with open(PREDICTIONS_LOG, 'r') as f:
                    records = json.load(f)
            records.append(record)
            # Храним только последние 500 записей
            if len(records) > 500:
                records = records[-500:]
            with open(PREDICTIONS_LOG, 'w') as f:
                json.dump(records, f, indent=2)
        except Exception as e:
            logger.debug(f"[PredTrack] save: {e}")

    def _validate_predictions(self) -> Dict:
        """Проверить созревшие предсказания (старше TARGET_HOURS_AHEAD).
        
        Returns:
            {'total': int, 'correct': int, 'accuracy': float, 'up_accuracy': float, 'down_accuracy': float}
        """
        try:
            if not os.path.exists(PREDICTIONS_LOG):
                return {'total': 0, 'correct': 0, 'accuracy': 0.0}
            
            with open(PREDICTIONS_LOG, 'r') as f:
                records = json.load(f)
            
            now_ts = time.time()
            maturity_sec = TARGET_HOURS_AHEAD * 3600
            validated = 0
            correct = 0
            up_total = 0
            up_correct = 0
            down_total = 0
            down_correct = 0
            
            # Берём текущую цену BTC для валидации
            current_btc = None
            try:
                if self.exchange:
                    ticker = self.exchange.fetch_ticker('BTC/USDT')
                    current_btc = ticker['last']
            except Exception:
                pass
            
            if current_btc is None:
                return {'total': 0, 'correct': 0, 'accuracy': 0.0, 'reason': 'no price'}
            
            changed = False
            for rec in records:
                if not rec.get('validated') and (now_ts - rec['ts']) >= maturity_sec:
                    price_then = rec['price']
                    change_pct = (current_btc - price_then) / price_then * 100
                    rec['price_change_pct'] = round(change_pct, 2)
                    
                    # Определяем фактическое направление
                    if change_pct > UP_THRESHOLD * 100:
                        actual_dir = 'up'
                    elif change_pct < DOWN_THRESHOLD * 100:
                        actual_dir = 'down'
                    else:
                        actual_dir = 'side'
                    rec['actual_direction'] = actual_dir
                    rec['validated'] = True
                    rec['correct'] = rec['direction'] == actual_dir
                    changed = True
                    validated += 1
                    if rec['correct']:
                        correct += 1
                    if rec['direction'] == 'up':
                        up_total += 1
                        if rec['correct']:
                            up_correct += 1
                    elif rec['direction'] == 'down':
                        down_total += 1
                        if rec['correct']:
                            down_correct += 1
            
            if changed:
                with open(PREDICTIONS_LOG, 'w') as f:
                    json.dump(records, f, indent=2)
            
            # Общая статистика по всем (не только за этот раз)
            total_predictions = len([r for r in records if r.get('validated')])
            correct_predictions = len([r for r in records if r.get('correct')])
            total_up = len([r for r in records if r.get('validated') and r['direction'] == 'up'])
            correct_up = len([r for r in records if r.get('correct') and r['direction'] == 'up'])
            total_down = len([r for r in records if r.get('validated') and r['direction'] == 'down'])
            correct_down = len([r for r in records if r.get('correct') and r['direction'] == 'down'])
            total_side = len([r for r in records if r.get('validated') and r['direction'] == 'side'])
            correct_side = len([r for r in records if r.get('correct') and r['direction'] == 'side'])
            
            stats = {
                'total': total_predictions,
                'correct': correct_predictions,
                'accuracy': round(correct_predictions / total_predictions, 3) if total_predictions > 0 else 0,
                'up_total': total_up,
                'up_correct': correct_up,
                'up_accuracy': round(correct_up / total_up, 3) if total_up > 0 else 0,
                'down_total': total_down,
                'down_correct': correct_down,
                'down_accuracy': round(correct_down / total_down, 3) if total_down > 0 else 0,
                'side_total': total_side,
                'side_correct': correct_side,
                'side_accuracy': round(correct_side / total_side, 3) if total_side > 0 else 0,
            }
            
            # Логируем результат
            if validated > 0:
                logger.info(f"🎯 [PredTrack] проверено {validated} предсказаний: "
                           f"{correct}/{validated} верных ({round(correct/validated*100) if validated else 0}%)")
            if stats['total'] > 0:
                logger.info(f"📊 [PredTrack] общая статистика: {stats['total']} предсказаний, "
                           f"{stats['correct']} верных ({stats['accuracy']:.0%}) | "
                           f"UP: {stats['up_accuracy']:.0%} ({stats['up_correct']}/{stats['up_total']}) | "
                           f"DOWN: {stats['down_accuracy']:.0%} ({stats['down_correct']}/{stats['down_total']})")
            
            return stats
        except Exception as e:
            logger.debug(f"[PredTrack] validate: {e}")
            return {'total': 0, 'correct': 0, 'accuracy': 0.0}

# ════════════════════════════════════════
# SELF-TEST
# ════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🧪 BTC Direction Predictor — самодиагностика")
    print("=" * 60)
    
    # Тест 1: Инициализация
    predictor = BTCDirectionPredictor()
    print(f"✅ Инициализация: модель={'есть' if predictor.model else 'нет'}")
    print(f"✅ Кеш свечей: {len(predictor._candle_cache)}")
    
    # Тест 2: feature engineering на синтетике
    print("\n🧪 Тест 2: Feature Engineering (синтетические данные)")
    np.random.seed(42)
    synthetic = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1h'),
        'open': 40000 + np.cumsum(np.random.randn(1000) * 100),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.rand(1000) * 1000
    })
    synthetic['high'] = synthetic['open'] + np.random.rand(1000) * 200
    synthetic['low'] = synthetic['open'] - np.random.rand(1000) * 200
    synthetic['close'] = synthetic['open'] + np.random.randn(1000) * 100
    
    synthetic.set_index('timestamp', inplace=True)
    
    df_features = predictor._calculate_features(synthetic.reset_index().rename(columns={'index': 'timestamp'}).set_index('timestamp'))
    if df_features is not None:
        print(f"✅ Сгенерировано {len(df_features)} строк с {len(predictor.feature_cols)} фичами")
        print(f"   Таргет: UP={sum(df_features['target']==2)}, DOWN={sum(df_features['target']==0)}, SIDE={sum(df_features['target']==1)}")
    else:
        print("❌ Feature engineering не удался")
    
    # Тест 3: Обучение на синтетике
    print("\n🧪 Тест 3: Обучение на синтетических данных")
    if df_features is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            df_features[predictor.feature_cols].values,
            df_features['target'].values,
            test_size=0.2, shuffle=False, random_state=42
        )
        
        predictor.scaler = StandardScaler()
        X_train_s = predictor.scaler.fit_transform(X_train)
        X_test_s = predictor.scaler.transform(X_test)
        
        model = LGBMClassifier(n_estimators=50, max_depth=4, verbose=-1)
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        acc = (preds == y_test).mean()
        print(f"✅ LightGBM accuracy: {acc:.1%}")
    
    # Тест 4: Прогноз
    print("\n🧪 Тест 4: Сигнал по умолчанию")
    signal = predictor.predict()
    print(f"   Direction: {signal['direction']}")
    print(f"   Confidence: {signal['confidence']:.0%}")
    print(f"   Strength: {signal['strength']}")
    
    # Тест 5: Бонусы
    print("\n🧪 Тест 5: Бонус/штраф для DecisionEngine")
    for test_score in [60, 55, 65]:
        bonus = predictor.calculate_bonus(test_score)
        print(f"   Score={test_score} → бонус={bonus:+.0f} → итог={test_score + bonus:.0f}")
    
    print("\n" + "=" * 60)
    print(f"✅ BTC Direction Predictor: {len(predictor.feature_cols) if predictor.feature_cols else 0} фич готовы")
