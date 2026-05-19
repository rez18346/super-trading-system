#!/usr/bin/env python3
"""
retrain_ml.py — Переобучение ML-PRO v2 на свежих данных
========================================================
Запускается вручную или по расписанию (раз в 7 дней).
Использует latest 1H/4H данные из JSON-файлов.

Сравнивает accuracy новой модели с текущей.
Сохраняет только если новая acc >= старая + 0.5%.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
from datetime import datetime, timezone
import lightgbm as lgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("RETRAIN_ML")

# 19 пар (та же последовательность что при обучении)
PAIRS = [
    'ADA/USDT', 'ALGO/USDT', 'APT/USDT', 'ARB/USDT',
    'ATOM/USDT', 'AVAX/USDT', 'BTC/USDT', 'DOGE/USDT',
    'DOT/USDT', 'EGLD/USDT', 'ETH/USDT', 'FIL/USDT',
    'LINK/USDT', 'MNT/USDT', 'NEAR/USDT', 'OP/USDT',
    'ROSE/USDT', 'SOL/USDT', 'XRP/USDT',
]

FEATURES_25 = [
    'rsi_1h','rsi_4h',
    'trend_1h','trend_4h','trend_aligned',
    'atr_1h','atr_4h','atr_ratio',
    'pv_ema12','pv_ema26',
    'sma20_d','sma50_d',
    'volr_1h','volr_4h','vwap_1h',
    'mom1','mom3','mom7','mom24',
    'candle_body','pinbar','engulfing','d24h','rsi_div',
    'hour_of_day',
    'tf_aligned','volatility_ratio','force_index','eom',
    'rsi_div_5','vol_spread','wick_body_ratio','ema_slope_6h',
]


def load_current_model():
    """Загрузить текущую модель, вернуть {acc, samples, features} или None."""
    path = os.path.join(MODELS_DIR, 'ml_pro_v2_27f.pkl')
    if not os.path.exists(path):
        log.info("[RETRAIN] Нет существующей модели — будет создана новая")
        return None
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return {
            'acc': data.get('acc', 0),
            'samples': data.get('samples', 0),
            'features': data.get('features', []),
        }
    except Exception as e:
        log.warning(f"[RETRAIN] Не загрузилась: {e}")
        return None


def load_data_for_pair(pair):
    """Загрузить последние 2000 1H и соответствующие 4H для пары."""
    symbol = pair.lower().replace('/', '_').replace('usdt', 'usdt')
    
    # Пробуем разные имена файлов
    candidates = [
        f'{symbol}_1h.json',
        f'{symbol.replace("_usdt", "_usdt")}_1h.json',
        pair.lower().replace('/', '_') + '_1h.json',
    ]
    
    h1_data = None
    for fname in candidates:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    h1_data = json.load(f)
                break
            except:
                continue
    
    h4_data = None
    for fname in [f.replace('_1h', '_4h') for f in candidates]:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    h4_data = json.load(f)
                break
            except:
                continue
    
    if h1_data is None or h4_data is None:
        return None, None
    
    # Берём последние 2000 часов 1H, синхронизируем с 4H
    h1 = h1_data[-2000:]
    h4 = [c for c in h4_data if c['t'] >= h1[0]['t'] and c['t'] <= h1[-1]['t']]
    
    return h1, h4


def build_labels(candles_1h):
    """Метка: 1 если return через +3 часа > +0.3%."""
    cl = np.array([c['c'] for c in candles_1h], float)
    n = len(cl)
    labels = np.zeros(n - 3, dtype=int)
    
    for i in range(n - 3):
        entry = cl[i]
        future = cl[i + 3]
        ret = (future - entry) / entry
        
        if ret >= 0.003:
            labels[i] =  1
        elif ret <= -0.003:
            labels[i] = 0
        else:
            labels[i] = -1  # нейтральные — пропускаем
    
    return labels


def build_dataset(pairs=PAIRS, lookback=1500):
    """Построить X и y для обучения на всех парах."""
    sys.path.insert(0, BASE_DIR)
    from ml_professional_v2 import build_features_27f
    
    all_X = []
    all_y = []
    total_samples = 0
    
    for pair in pairs:
        h1, h4 = load_data_for_pair(pair)
        if h1 is None or len(h1) < lookback:
            log.warning(f"[RETRAIN] {pair}: недостаточно 1H данных ({len(h1) if h1 else 0})")
            continue
        if h4 is None or len(h4) < lookback // 4:
            log.warning(f"[RETRAIN] {pair}: недостаточно 4H данных ({len(h4) if h4 else 0})")
            continue
        
        X = build_features_27f(h1, h4)
        y = build_labels(h1)
        
        # Синхронизируем — метки на 6 короче, фичи тоже
        min_len = min(len(X), len(y))
        if min_len < 100:
            log.warning(f"[RETRAIN] {pair}: слишком мало ({min_len})")
            continue
        
        X = X[-min_len:]
        y = y[-min_len:]
        
        # Убираем неопределённые
        valid = y != -1
        X, y = X[valid], y[valid]
        
        if len(X) < 50:
            continue
        
        all_X.append(X)
        all_y.append(y)
        total_samples += len(X)
        log.info(f"[RETRAIN] {pair}: {len(X)} обр. (bal {y.mean():.1%})")
    
    if not all_X:
        log.error("[RETRAIN] Нет данных!")
        return None, None, 0
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    # Убираем NaN
    valid = ~np.isnan(X).any(axis=1)
    X, y = X[valid], y[valid]
    
    log.info(f"[RETRAIN] 📊 Всего: {len(X):,} обр., баланс {y.mean():.1%}")
    return X, y, total_samples


def train_model(X, y):
    """Обучить LightGBM, вернуть (booster, test_acc)."""
    from sklearn.model_selection import train_test_split
    
    # Split 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    log.info(f"[RETRAIN] Обучение: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=6,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.01,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='binary_logloss',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    
    # Тест
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    
    # Precision и Recall
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Feature importance
    gain = model.booster_.feature_importance(importance_type='gain')
    total_gain = gain.sum()
    top_features = sorted(
        [(FEATURES_25[i], gain[i]) for i in range(len(gain))],
        key=lambda x: x[1], reverse=True
    )[:10]
    
    log.info(f"[RETRAIN] 🎯 Test: {acc:.2%} | Prec: {prec:.1%} | Rec: {rec:.1%}")
    log.info(f"[RETRAIN] 🏆 Топ-10:")
    for name, g in top_features:
        log.info(f"       {name}: gain={g:.0f} ({g/total_gain*100:.1f}%)")
    
    # Booster для pickle
    booster = model.booster_
    
    return booster, acc, top_features


def save_model(booster, acc, samples, top_features):
    """Сохранить модель с метаданными."""
    # Перемещаем старую
    model_path = os.path.join(MODELS_DIR, 'ml_pro_v2_27f.pkl')
    old_path = os.path.join(MODELS_DIR, 'ml_pro_v2_27f_old.pkl')
    
    if os.path.exists(model_path):
        if os.path.exists(old_path):
            os.remove(old_path)
        os.rename(model_path, old_path)
        log.info(f"[RETRAIN] 📦 Старая модель → {old_path}")
    
    data = {
        'model': booster,
        'features': FEATURES_25,  # теперь 33 фичи
        'acc': acc,
        'samples': samples,
        'feature_importance': top_features,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(data, f)
    
    log.info(f"[RETRAIN] 💾 Сохранена: {model_path}")
    log.info(f"[RETRAIN] ✅ {samples:,} обр., {acc:.2%} acc")


if __name__ == "__main__":
    log.info("=" * 50)
    log.info("🧠 ПЕРЕОБУЧЕНИЕ ML-PRO v2")
    log.info(f"Время: {datetime.now(timezone.utc).isoformat()}")
    log.info("=" * 50)
    
    # 1. Текущая модель
    current = load_current_model()
    if current:
        log.info(f"[RETRAIN] Текущая модель: {current['samples']:,} обр., {current['acc']:.2%} acc")
    
    # 2. Собираем данные
    X, y, total = build_dataset()
    if X is None or len(X) < 500:
        log.error("[RETRAIN] Недостаточно данных")
        sys.exit(1)
    
    # 3. Обучаем
    booster, new_acc, top_features = train_model(X, y)
    
    # 4. Решаем сохранять ли
    if current:
        min_improvement = 0.005  # 0.5%
        if new_acc > current['acc'] + min_improvement:
            save_model(booster, new_acc, total, top_features)
            log.info(f"[RETRAIN] 🎉 Улучшение: {current['acc']:.2%} → {new_acc:.2%} ({new_acc - current['acc']:+.2%})")
        else:
            log.info(f"[RETRAIN] ⏭️ Пропуск: {new_acc:.2%} vs {current['acc']:.2%} (нужно +0.5%)")
    else:
        save_model(booster, new_acc, total, top_features)
    
    log.info("[RETRAIN] ✅ Завершено")
