#!/usr/bin/env python3
"""
🧠 ML-PRO — Профессиональный ML-движок для торговой системы
==============================================================
Заменяет ml_advisor.py (RandomForest 9 признаков).

Архитектура:
  1. Модуль 1: Market Regime Detector — D1 определяет рынок (bull/bear/sideways)
  2. Модуль 2: Entry Filter — 1H + 15M анализ для фильтрации входа
  3. Модуль 3: Position Sizing — предсказывает оптимальный % капитала

Признаки:
  - 15-минутный тренд (изменение за 4ч)
  - Объём на 15M (соотношение к среднему)
  - Расстояние до 24-часового min/max
  - Кластер волатильности (низкая/средняя/высокая)
  - Мультитаймфрейм: D1, 4H, 1H, 15M тренды
  - VWAP и его расстояние
  - RSI на всех таймфреймах
  - Свечные паттерны (дожи, молот, поглощение)

Технологии:
  - XGBoost Classifier (фильтр) + XGBoost Regressor (размер позиции)
  - Feature importance анализ
  - Time-series CV (не перемешиваем время)
  - Online learning (дообучение после каждой сделки)
  - Early stopping против переобучения

Автор: Вася (для Ксюши)
Дата: 2026-04-26
"""

import numpy as np
import pandas as pd
import json
import os
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import xgboost as xgb

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "data/ml_pro_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "data/ml_pro_scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "data/ml_pro_features.csv")
CONFIG_PATH = os.path.join(BASE_DIR, "config/api_config_final.json")

# Пороги
REBUILD_INTERVAL = 3600         # Полное перестроение модели раз в час
MIN_TRAINING_SAMPLES = 50       # Минимум для обучения
PREDICT_PROFIT_THRESHOLD = 0.55 # Порог GOOD/BAD (немного выше 50%)
CONFIDENCE_HIGH = 0.70          # Сильный сигнал
CONFIDENCE_LOW = 0.50           # Слабый, но допустимый

# Веса признаков для importance отчёта
FEATURE_NAMES = [
    # Базовые (3)
    'rsi_15m', 'trend_15m', 'volatility_15m',
    # Объём и ликвидность (4)
    'volume_ratio_15m', 'volume_ratio_1h', 'vwap_dist_1h', 'vwap_dist_d1',
    # Мультитаймфрейм (6)
    'trend_1h', 'trend_4h', 'trend_d1',
    'change_pct_15m', 'change_pct_1h', 'change_pct_4h',
    # Экстремумы (3)
    'dist_to_24h_min', 'dist_to_24h_max', 'dist_to_24h_mid',
    # Свечные паттерны (3)
    'candle_doji', 'candle_hammer', 'candle_engulfing',
    # Рыночный режим (4)
    'regime_bull', 'regime_bear', 'regime_sideways',
    'regime_volatility',
    # Волатильность (2)
    'atr_15m', 'atr_1h',
    # Кластеры (1)
    'volatility_cluster'
]

# === КЭШ ДЛЯ ДАННЫХ (чтобы не стучаться к бирже 100 раз за цикл) ===
_data_cache = {}
_cache_ttl = {}

def _cached_fetch(symbol: str, tf: str, limit: int, force: bool = False) -> Optional[pd.DataFrame]:
    """Кэшированный fetch с TTL 60 секунд"""
    key = f"{symbol}_{tf}_{limit}"
    now = time.time()
    if not force and key in _data_cache and now - _cache_ttl.get(key, 0) < 60:
        return _data_cache[key]
    try:
        import ccxt
        cfg = json.load(open(CONFIG_PATH))
        ex = ccxt.bybit({
            "apiKey": cfg["bybit"]["api_key"],
            "secret": cfg["bybit"]["secret"],
            "enableRateLimit": True,
            "options": {"defaultType": "spot"}
        })
        ohlcv = ex.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        _data_cache[key] = df
        _cache_ttl[key] = now
        return df
    except Exception as _e:
        logger.debug("bare except in ml_professional: %s", _e)
        return None


class MarketRegimeDetector:
    """
    Модуль 1: Определение рыночного режима на D1.
    Возвращает: {regime: 'bull'|'bear'|'sideways', strength: 0-1}
    """
    
    @staticmethod
    def detect(symbol: str) -> Dict:
        df = _cached_fetch(symbol, '1d', 30)
        if df is None or len(df) < 14:
            return {'regime': 'sideways', 'strength': 0.5}
        
        closes = df['close'].values
        volumes = df['volume'].values
        
        # Тренд: SMA7 vs SMA25
        sma7 = np.mean(closes[-7:]) if len(closes) >= 7 else np.mean(closes)
        sma25 = np.mean(closes[-25:]) if len(closes) >= 25 else np.mean(closes)
        
        # Наклон SMA7 за последние 3 дня
        sma7_3d_ago = np.mean(closes[-10:-7]) if len(closes) >= 10 else sma7
        sma_slope = (sma7 - sma7_3d_ago) / sma7_3d_ago * 100
        
        # Волатильность (стд за 14 дней)
        if len(closes) >= 16:
            returns = np.diff(closes[-15:]) / closes[-15:-1]
        elif len(closes) >= 4:
            returns = np.diff(closes[-3:]) / closes[-3:-1]
        else:
            returns = np.array([0.01])
        vol = np.std(returns) * 100
        
        # ADX (направление тренда)
        up_moves = np.maximum(0, np.diff(closes[-15:]))
        down_moves = np.maximum(0, -np.diff(closes[-15:]))
        avg_up = np.mean(up_moves)
        avg_down = np.mean(down_moves)
        adx = abs(avg_up - avg_down) / (avg_up + avg_down + 0.0001) * 100
        
        # Определяем режим
        if sma7 > sma25 * 1.02 and sma_slope > 0 and adx > 20:
            regime = 'bull'
            strength = min(1.0, max(0.5, adx / 50))
        elif sma7 < sma25 * 0.98 and sma_slope < 0 and adx > 20:
            regime = 'bear'
            strength = min(1.0, max(0.5, adx / 50))
        else:
            regime = 'sideways'
            strength = max(0.3, 1.0 - adx / 40)
        
        # Класс волатильности
        if vol < 1.0:
            vol_cluster = 'low'
        elif vol < 2.5:
            vol_cluster = 'medium'
        else:
            vol_cluster = 'high'
        
        return {
            'regime': regime,
            'strength': float(strength),
            'volatility': float(vol),
            'vol_cluster': vol_cluster,
            'sma7': float(sma7),
            'sma25': float(sma25),
            'sma_slope_pct': float(sma_slope),
            'adx': float(adx)
        }


class MLProAdvisor:
    """
    ML-PRO — профессиональный ML-советник.
    
    Методы:
      analyze(symbol, price, rsi, trend, confidence, df_5m) -> Dict
      add_trade_result(...) -> None
      train(force=False) -> None
    
    Возврат analyze:
      {decision: 'GOOD'|'WEAK'|'SKIP', confidence: float, 
       regime: str, position_size_pct: float, reason: str}
    """
    
    def __init__(self):
        self.model = None          # XGBoost Classifier (фильтр)
        self.model_size = None     # XGBoost Regressor (размер позиции)
        self.scaler = StandardScaler()
        self.regime_detector = MarketRegimeDetector()
        
        self.is_trained = False
        self.last_retrain = 0
        self.training_data = []
        
        # Статистика
        self._regime_cache = {}  # symbol -> regime
        self._regime_ttl = {}
        
        self._load_model()
    
    def _load_model(self):
        """Загрузка сохранённой модели"""
        try:
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info("✅ ML-PRO: модель загружена (XGBoost)")
        except Exception as e:
            logger.warning(f"⚠️ ML-PRO: не удалось загрузить модель: {e}")
    
    def _get_regime(self, symbol: str) -> Dict:
        """Получение режима рынка с кэшем (120 сек)"""
        now = time.time()
        if symbol in self._regime_cache and now - self._regime_ttl.get(symbol, 0) < 120:
            return self._regime_cache[symbol]
        regime = self.regime_detector.detect(symbol)
        self._regime_cache[symbol] = regime
        self._regime_ttl[symbol] = now
        return regime
    
    def _extract_features(self, symbol: str, current_price: float, 
                           rsi: float, trend: str, confidence: float,
                           df_5m=None) -> np.ndarray:
        """
        Извлечение 26 признаков для XGBoost.
        """
        features = {}
        
        # === БАЗОВЫЕ (3) ===
        features['rsi_15m'] = rsi
        features['trend_15m'] = 1.0 if trend == 'bullish' else (0.0 if trend == 'bearish' else 0.5)
        features['volatility_15m'] = 0.01
        
        # === 15M ДАННЫЕ ===
        df_15m = _cached_fetch(symbol, '15m', 16)  # 4 часа по 15 мин
        if df_15m is not None and len(df_15m) > 5:
            closes = df_15m['close'].values
            volumes = df_15m['volume'].values
            
            features['volume_ratio_15m'] = volumes[-1] / (np.mean(volumes[-5:]) + 0.0001)
            features['change_pct_15m'] = (closes[-1] - closes[-4]) / closes[-4] * 100
            features['volatility_15m'] = np.std(closes[-10:] / (np.mean(closes[-10:]) + 0.0001))
            
            # ATR на 15M
            h = df_15m['high'].values[-10:]
            l = df_15m['low'].values[-10:]
            c = df_15m['close'].values[-10:]
            tr = np.maximum(h[1:] - l[1:], 
                            np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
            features['atr_15m'] = np.mean(tr) / np.mean(c) * 100
        else:
            features['volume_ratio_15m'] = 1.0
            features['change_pct_15m'] = 0.0
            features['atr_15m'] = 0.5
        
        # === 1H ДАННЫЕ ===
        df_1h = _cached_fetch(symbol, '1h', 24)
        if df_1h is not None and len(df_1h) > 5:
            closes_1h = df_1h['close'].values
            volumes_1h = df_1h['volume'].values
            
            features['volume_ratio_1h'] = volumes_1h[-1] / (np.mean(volumes_1h[-5:]) + 0.0001)
            features['change_pct_1h'] = (closes_1h[-1] - closes_1h[-4]) / closes_1h[-4] * 100
            features['trend_1h'] = 1.0 if closes_1h[-1] > closes_1h[-4] else 0.0
            features['vwap_dist_1h'] = (current_price - np.mean(closes_1h[-12:])) / np.mean(closes_1h[-12:])
            
            # ATR на 1H
            h = df_1h['high'].values[-10:]
            l = df_1h['low'].values[-10:]
            c = df_1h['close'].values[-10:]
            tr = np.maximum(h[1:] - l[1:],
                            np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
            features['atr_1h'] = np.mean(tr) / np.mean(c) * 100
        else:
            features['volume_ratio_1h'] = 1.0
            features['change_pct_1h'] = 0.0
            features['trend_1h'] = 0.5
            features['vwap_dist_1h'] = 0.0
            features['atr_1h'] = 0.5
        
        # === 4H ДАННЫЕ ===
        df_4h = _cached_fetch(symbol, '4h', 12)
        if df_4h is not None and len(df_4h) > 3:
            closes_4h = df_4h['close'].values
            features['trend_4h'] = 1.0 if closes_4h[-1] > closes_4h[-3] else 0.0
            features['change_pct_4h'] = (closes_4h[-1] - closes_4h[-3]) / closes_4h[-3] * 100
        else:
            features['trend_4h'] = 0.5
            features['change_pct_4h'] = 0.0
        
        # === D1 ДАННЫЕ + РЕЖИМ ===
        regime = self._get_regime(symbol)
        features['trend_d1'] = {'bull': 1.0, 'bear': 0.0, 'sideways': 0.5}.get(regime['regime'], 0.5)
        features['regime_bull'] = 1.0 if regime['regime'] == 'bull' else 0.0
        features['regime_bear'] = 1.0 if regime['regime'] == 'bear' else 0.0
        features['regime_sideways'] = 1.0 if regime['regime'] == 'sideways' else 0.0
        features['regime_volatility'] = regime['volatility']
        features['volatility_cluster'] = {'low': 0, 'medium': 1, 'high': 2}.get(regime['vol_cluster'], 1)
        features['vwap_dist_d1'] = (current_price - regime['sma7']) / (regime['sma7'] + 0.0001)
        
        # === ЭКСТРЕМУМЫ (24H) ===
        df_15m_96 = _cached_fetch(symbol, '15m', 96 if df_15m is None else 96)
        if df_15m_96 is not None and len(df_15m_96) > 10:
            all_closes = df_15m_96['close'].values
            min_24h = np.min(all_closes)
            max_24h = np.max(all_closes)
            mid_24h = (min_24h + max_24h) / 2
            features['dist_to_24h_min'] = (current_price - min_24h) / (max_24h - min_24h + 0.0001)
            features['dist_to_24h_max'] = (max_24h - current_price) / (max_24h - min_24h + 0.0001)
            features['dist_to_24h_mid'] = (current_price - mid_24h) / (mid_24h + 0.0001)
        else:
            features['dist_to_24h_min'] = 0.5
            features['dist_to_24h_max'] = 0.5
            features['dist_to_24h_mid'] = 0.0
        
        # === СВЕЧНЫЕ ПАТТЕРНЫ (из 5M если есть) ===
        features['candle_doji'] = 0.0
        features['candle_hammer'] = 0.0
        features['candle_engulfing'] = 0.0
        if df_5m is not None and len(df_5m) > 5:
            try:
                # Простые свечные паттерны на последней свече
                o, h, l, c = df_5m[['open','high','low','close']].iloc[-1]
                body = abs(c - o)
                _range = h - l + 0.000001
                upper_w = h - max(c, o)
                lower_w = min(c, o) - l
                features['candle_doji'] = 1.0 if body / _range < 0.1 else 0.0
                features['candle_hammer'] = 1.0 if (lower_w > body * 2 and upper_w < body * 0.5) else 0.0
                # Engulfing: проверяем две свечи
                if len(df_5m) > 1:
                    o2, c2 = df_5m[['open','close']].iloc[-2]
                    features['candle_engulfing'] = 1.0 if (c > o and o < c2 and c > o2) else 0.0
            except Exception as _e:
                logger.debug("bare except in ml_professional: %s", _e)
                pass
        
        # === УПАКОВКА В ВЕКТОР ===
        return np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=float)
    
    def add_trade_result(self, symbol: str, entry_price: float, exit_price: float,
                          rsi: float, trend: str, confidence: float,
                          hold_hours: float, reason: str, 
                          volume_ratio: float = None, df_5m=None):
        """
        Добавление результата сделки в обучающую выборку.
        На реальном PnL считаем метку:
          - label = 1 если PnL > 0.5% (хороший вход)
          - label = 0 если PnL < -0.3% (плохой вход)
          - mid пропускаем
        """
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        
        # Пропускаем сделки на грани
        if -0.3 <= pnl_pct <= 0.5:
            return
        
        label = 1.0 if pnl_pct > 0.5 else 0.0
        
        features = self._extract_features(symbol, entry_price, rsi, trend, confidence, df_5m)
        
        # Размер позиции: PnL как целевая переменная для регрессии
        position_size_pct = min(6.0, max(1.0, abs(pnl_pct) * 2))  # эвристика
        
        self.training_data.append({
            'features': features,
            'label': label,
            'size_target': position_size_pct if pnl_pct > 0 else 0,
            'symbol': symbol,
            'pnl': pnl_pct,
            'hold_hours': hold_hours,
            'reason': reason,
            'time': datetime.now().isoformat()
        })
        
        logger.info(f"📚 ML-PRO: обучение на {symbol} (PnL={pnl_pct:+.2f}%, label={int(label)})")
    
    def train(self, force: bool = False):
        """
        Обучение XGBoost моделей на накопленных данных.
        Использует TimeSeriesSplit вместо shuffle.
        """
        now = time.time()
        if not force and (now - self.last_retrain) < REBUILD_INTERVAL:
            return
        
        if len(self.training_data) < MIN_TRAINING_SAMPLES:
            logger.info(f"⏳ ML-PRO: ждём данные ({len(self.training_data)}/{MIN_TRAINING_SAMPLES})")
            return
        
        # Подготовка данных
        X = np.array([d['features'] for d in self.training_data])
        y = np.array([d['label'] for d in self.training_data])
        y_size = np.array([d['size_target'] for d in self.training_data])
        
        # Масштабирование
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # TimeSeriesSplit (не перемешиваем!)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # === XGBoost CLASSIFIER (фильтр сигналов) ===
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            eval_metric='logloss',
            early_stopping_rounds=20,
            random_state=42,
            n_jobs=4,
            enable_categorical=False,
            verbosity=0
        )
        
        # Оценка через CV
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_scaled):
            if len(train_idx) < 30 or len(val_idx) < 10:
                continue
            X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            model = xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                eval_metric='logloss', early_stopping_rounds=20,
                random_state=42, verbosity=0, n_jobs=2
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            cv_scores.append(model.score(X_val, y_val))
        
        # Финальное обучение
        self.model.fit(X_scaled, y)
        
        # === XGBoost REGRESSOR (размер позиции) ===
        self.model_size = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mae',
            early_stopping_rounds=15,
            random_state=42,
            n_jobs=4,
            verbosity=0
        )
        # Обучаем только на прибыльных сделках
        profitable = y == 1
        if profitable.sum() > 10:
            y_size_prof = y_size[profitable]
            X_size = X_scaled[profitable]
            self.model_size.fit(X_size, y_size_prof)
        
        self.is_trained = True
        self.last_retrain = now
        
        # Статистика
        train_score = self.model.score(X_scaled, y)
        avg_cv = np.mean(cv_scores) if cv_scores else train_score
        
        # Feature importance
        importance = self.model.feature_importances_
        top_features = sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])[:5]
        
        n_good = int(y.sum())
        n_bad = len(y) - n_good
        
        logger.info(f"🎯 ML-PRO: модель обучена!")
        logger.info(f"   Данных: {len(y)} (good={n_good}, bad={n_bad})")
        logger.info(f"   XGBoost точность (train): {train_score:.1%}")
        if cv_scores:
            logger.info(f"   XGBoost CV (time-series): {avg_cv:.1%} (±{np.std(cv_scores):.1%})")
        logger.info(f"   Топ-5 признаков: {', '.join(f'{n}={v:.1%}' for n,v in top_features)}")
        
        # Сохраняем
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Сохраняем обучающие данные
        df_result = pd.DataFrame([
            {**{'symbol': d['symbol'], 'pnl': d['pnl'], 'hold_hours': d['hold_hours'],
                'reason': d['reason'], 'time': d['time']},
             **{FEATURE_NAMES[i]: d['features'][i] for i in range(len(FEATURE_NAMES))}}
            for d in self.training_data
        ])
        df_result.to_csv(FEATURES_PATH, index=False)
    
    def analyze(self, symbol: str, current_price: float,
                 rsi: float, trend: str, confidence: float,
                 df_5m=None) -> Dict:
        """
        Полный анализ сигнала ML-PRO.
        
        Возвращает:
          decision: 'GOOD' | 'WEAK' | 'SKIP'
          confidence: float (0-1)
          regime: str (bull/bear/sideways)
          regime_strength: float
          position_size_pct: float (оптимальный % капитала)
          reason: str
        """
        # 1. Режим рынка
        regime = self._get_regime(symbol)
        
        # 2. Если режим bear и strength > 0.6 — сразу SKIP
        if regime['regime'] == 'bear' and regime['strength'] > 0.6:
            return {
                'decision': 'SKIP',
                'confidence': 0.0,
                'regime': 'bear',
                'regime_strength': regime['strength'],
                'position_size_pct': 0,
                'reason': f"ML-PRO: режим медвежий ({regime['strength']:.0%}) — пропускаем"
            }
        
        # 3. Если не обучен — доверяем основной системе, но с режимом
        if not self.is_trained or self.model is None:
            base_size = 6.0  # дефолт
            if regime['regime'] == 'sideways':
                base_size = 4.0  # в боковике меньше
            return {
                'decision': 'GOOD',
                'confidence': 0.5,
                'regime': regime['regime'],
                'regime_strength': regime['strength'],
                'position_size_pct': base_size,
                'reason': f"ML-PRO: не обучен, режим={regime['regime']}, размер={base_size}%"
            }
        
        # 4. XGBoost оценка
        try:
            features = self._extract_features(symbol, current_price, rsi, trend, confidence, df_5m)
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            prob = self.model.predict_proba(X_scaled)[0]
            good_prob = prob[1] if len(prob) > 1 else 0.5
            
            # Умный размер позиции
            if self.model_size is not None:
                try:
                    size_pred = self.model_size.predict(X_scaled)[0]
                    # Ограничиваем 3%-10%
                    size_pct = max(3.0, min(10.0, size_pred))
                except Exception as _e:
                    logger.debug("bare except in ml_professional: %s", _e)
                    size_pct = 6.0
            else:
                size_pct = 6.0
            
            # Корректировка на режим рынка
            if regime['regime'] == 'sideways':
                size_pct *= 0.7  # в боковике меньше
            elif regime['regime'] == 'bull' and regime['strength'] > 0.7:
                size_pct *= 1.2  # сильный бычий — чуть больше
            
            size_pct = min(10.0, size_pct)  # макс 10%
            
            # Принятие решения
            if good_prob >= CONFIDENCE_HIGH and regime['regime'] != 'bear':
                decision = 'GOOD'
                reason = f"ML-PRO: {good_prob:.0%} (сильный), режим={regime['regime']}, размер={size_pct:.0f}%"
            elif good_prob >= CONFIDENCE_LOW and regime['regime'] != 'bear':
                decision = 'WEAK'
                reason = f"ML-PRO: {good_prob:.0%} (средний), режим={regime['regime']}"
            else:
                decision = 'SKIP'
                reason = f"ML-PRO: {good_prob:.0%} (слабый), режим={regime['regime']}"
                size_pct = 0
            
            return {
                'decision': decision,
                'confidence': float(good_prob),
                'regime': regime['regime'],
                'regime_strength': regime['strength'],
                'position_size_pct': float(size_pct),
                'reason': reason
            }
            
        except Exception as e:
            logger.warning(f"⚠️ ML-PRO analyze error: {e}")
            return {
                'decision': 'GOOD',
                'confidence': 0.5,
                'regime': regime['regime'],
                'regime_strength': regime['strength'],
                'position_size_pct': 4.0,
                'reason': f"ML-PRO: ошибка, fallback, режим={regime['regime']}"
            }


# === ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР ===
_advisor_pro = None


def get_advisor():
    global _advisor_pro
    if _advisor_pro is None:
        _advisor_pro = MLProAdvisor()
    return _advisor_pro


def ml_pro_evaluate(symbol, current_price, rsi, trend, confidence, df=None):
    advisor = get_advisor()
    return advisor.analyze(symbol, current_price, rsi, trend, confidence, df)


def ml_pro_add_result(symbol, entry_price, exit_price, rsi, trend, confidence,
                       hold_hours, reason, volume_ratio=None, df=None):
    advisor = get_advisor()
    advisor.add_trade_result(symbol, entry_price, exit_price, rsi, trend,
                              confidence, hold_hours, reason, volume_ratio, df)


def ml_pro_train(force=False):
    advisor = get_advisor()
    advisor.train(force=force)


# === ТЕСТОВЫЙ ЗАПУСК ===
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("🧠 ML-PRO: тестовый запуск")
    advisor = MLProAdvisor()
    
    # Тест на BTC
    for sym in ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']:
        try:
            import ccxt
            cfg = json.load(open(CONFIG_PATH))
            ex = ccxt.bybit({"apiKey":cfg["bybit"]["api_key"],"secret":cfg["bybit"]["secret"],"enableRateLimit":True,"options":{"defaultType":"spot"}})
            t = ex.fetch_ticker(sym)
            price = t['last']
            
            reg = advisor._get_regime(sym)
            result = advisor.analyze(sym, price, 50, 'sideways', 0.5)
            print(f"  {sym}: ${price:.2f} → {result}")
        except Exception as e:
            print(f"  {sym}: ошибка - {e}")
    
    print(f"\n✅ ML-PRO готов к работе")
    print(f"   Файл: {__file__}")
