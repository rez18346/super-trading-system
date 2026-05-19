#!/usr/bin/env python3
"""
Обучение ML-PRO v2 на исторических данных (2023-2026)
======================================================
Загружает исторические свечи из data/, считает 27 признаков,
обучает LightGBM с walk-forward валидацией и сохраняет модель.

Результат: models/ml_pro_v2.pkl + отчёт models/training_report.json
"""
import json, os, sys, pickle, logging
import numpy as np
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("TRAIN")

# Подключаем ML-PRO v2
sys.path.insert(0, '.')
from ml_professional_v2 import build_features_5m, MLProfessionalV2, FEATURE_NAMES, NUM_FEATURES

DATA_DIR = "data"
LOOKAHEAD = 3  # прогноз на 3 свечи
BATCH_SIZE = 200  # сколько 5M свечей одновременно используем

# ─── Функции для загрузки ──────────────────────────────────────────────────

def load_candles(pair, timeframe):
    """Загрузить свечи из data/ папки."""
    fname = f"{pair.lower().replace('/', '_')}_{timeframe}.json"
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def build_labels(closes, lookahead=LOOKAHEAD):
    """Создать бинарные метки: 1=цена вырастет, 0=упадёт."""
    y = np.zeros(len(closes))
    for i in range(len(closes) - lookahead):
        y[i] = 1.0 if closes[i + lookahead] > closes[i] else 0.0
    return y

def aggregate_5m_to_1h(candles_5m, chunk_size=12):
    """
    Агрегировать 5M свечи в 1H для синтеза старших ТФ,
    когда нет прямых исторических 1H данных.
    """
    candles_1h = []
    for i in range(0, len(candles_5m), chunk_size):
        chunk = candles_5m[i:i + chunk_size]
        if not chunk:
            continue
        o = chunk[0]['o']
        c = chunk[-1]['c']
        h = max(x['h'] for x in chunk)
        l = min(x['l'] for x in chunk)
        v = sum(x['v'] for x in chunk)
        t = chunk[0]['t']
        candles_1h.append({'o': o, 'h': h, 'l': l, 'c': c, 'v': v, 't': t})
    return candles_1h

# ─── Основной процесс обучения ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("📚 ML-PRO v2 — ОБУЧЕНИЕ НА ИСТОРИИ")
    print("=" * 60)

    all_X = []
    all_y = []
    pair_stats = []

    # Обрабатываем каждую пару
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith("_1h.json"):
            continue
        
        pair = fname.replace("_1h.json", "").replace("_", "/").upper()
        pair_name = fname.replace("_1h.json", "")
        
        # Загружаем 1H данные
        candles_1h = load_candles(pair, "1h")
        if candles_1h is None or len(candles_1h) < 200:
            log.warning(f"  {pair:6s}: недостаточно данных")
            continue
        
        # Для 4H агрегируем из 1H
        candles_4h = []
        for i in range(0, len(candles_1h), 4):
            chunk = candles_1h[i:i + 4]
            if len(chunk) < 4:
                break
            candles_4h.append({
                'o': chunk[0]['o'], 'h': max(c['h'] for c in chunk),
                'l': min(c['l'] for c in chunk), 'c': chunk[-1]['c'],
                'v': sum(c['v'] for c in chunk), 't': chunk[0]['t'],
            })
        
        # Симулируем 5M свечи — для дней где у нас нет 5M, создаём из 1H
        o1 = np.array([c['o'] for c in candles_1h], dtype=float)
        h1 = np.array([c['h'] for c in candles_1h], dtype=float)
        l1 = np.array([c['l'] for c in candles_1h], dtype=float)
        c1 = np.array([c['c'] for c in candles_1h], dtype=float)
        v1 = np.array([c['v'] for c in candles_1h], dtype=float)
        t1 = np.array([c['t'] for c in candles_1h], dtype=float)
        
        total_hours = len(candles_1h)
        pair_candles_5m_synth = []
        
        # Синтезируем 5M из 1H (простая интерполяция)
        for idx in range(total_hours):
            # Каждый час = 12 пятиминутных свечей с интерполяцией
            prev_close = candles_1h[idx - 1]['c'] if idx > 0 else candles_1h[0]['o']
            cur_open = candles_1h[idx]['o']
            cur_high = candles_1h[idx]['h']
            cur_low = candles_1h[idx]['l']
            cur_close = candles_1h[idx]['c']
            cur_volume = candles_1h[idx]['v'] / 12  # делим объём на 12 свечей
            
            for sub in range(12):
                frac = (sub + 1) / 12
                # Интерполяция цены
                sub_close = prev_close + (cur_close - prev_close) * frac if sub > 0 else cur_open
                t_stamp = t1[idx] + sub * 300000  # +5 мин
                pair_candles_5m_synth.append({
                    'o': prev_close if sub == 0 else pair_candles_5m_synth[-1]['c'],
                    'h': max(cur_high, sub_close),
                    'l': min(cur_low, sub_close),
                    'c': sub_close,
                    'v': cur_volume,
                    't': t_stamp,
                })
        
        candles_5m = pair_candles_5m_synth
        closes_5m = np.array([c['c'] for c in candles_5m])
        
        # Строим фичи окнами
        window_size = 200  # 200 пятиминутных свечей = ~16 часов 5M
        step = 20  # шаг между окнами
        
        n_windows = max(0, (len(candles_5m) - window_size) // step)
        X_windows = []
        
        for w in range(n_windows):
            start = w * step
            end = start + window_size
            
            # Определяем границы 1H для этого окна
            half_start_ts = candles_5m[start]['t']
            half_end_ts = candles_5m[end - 1]['t']
            
            idx_1h_start = np.searchsorted(t1, half_start_ts, side='right') - 1
            idx_1h_end = np.searchsorted(t1, half_end_ts, side='right') - 1
            idx_1h_start = max(0, idx_1h_start)
            idx_1h_end = min(len(candles_1h) - 1, idx_1h_end)
            
            # Берём 1H/4H для этого окна
            win_candles_1h = candles_1h[max(0, idx_1h_start - 50):idx_1h_end + 1]
            win_candles_4h = candles_4h[max(0, idx_1h_start // 4 - 10):idx_1h_end // 4 + 1]
            
            if len(win_candles_1h) < 20 or len(win_candles_4h) < 10:
                continue
            
            # Берём только последние 100-200 5M свечей для фич
            win_5m = candles_5m[max(0, end - 150):end]
            
            X_row = build_features_5m(win_5m, win_candles_1h, win_candles_4h)
            if X_row.shape[0] > 0:
                # Берём только последнюю строку (текущий сигнал)
                last_row = X_row[-1:]
                if not np.isnan(last_row).any():
                    X_windows.append(last_row[0])
        
        if len(X_windows) < 100:
            log.warning(f"  {pair:6s}: только {len(X_windows)} валидных окон — пропускаем")
            continue
        
        X_arr = np.array(X_windows)
        
        # Метки: берём цену через LOOKAHEAD*5M от конца окна
        y_arr = np.zeros(len(X_windows))
        for w in range(min(len(X_windows), n_windows)):
            idx_5m = min(w * step + window_size + LOOKAHEAD, len(closes_5m) - 1)
            if idx_5m < len(closes_5m) - 1:
                price_now = closes_5m[w * step + window_size - 1]
                price_future = closes_5m[idx_5m]
                y_arr[w] = 1.0 if price_future > price_now else 0.0
        
        pair_stats.append({
            'pair': pair, 'windows': len(X_arr),
            'balance': f"{y_arr.mean():.1%}",
        })
        all_X.append(X_arr)
        all_y.append(y_arr)
        
        print(f"  {pair:6s}: {len(X_arr):>5} окон, баланс {y_arr.mean():.1%}")

    # Объединяем все данные
    if not all_X:
        print("❌ Нет данных для обучения!")
        return
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"\n{'=' * 60}")
    print(f"📊 Всего: {len(X)} окон, {X.shape[1]} признаков, баланс {y.mean():.1%}")
    
    # Обучаем
    ml = MLProfessionalV2(model_path="models/ml_pro_v2.pkl")
    ml.train(X, y)
    
    # Отчёт
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(X),
        'features': NUM_FEATURES,
        'label_balance': float(y.mean()),
        'pairs': pair_stats,
        'feature_importance': ml.feature_importance,
        'status': 'success',
    }
    
    os.makedirs("models", exist_ok=True)
    with open("models/training_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Обучение завершено!")
    print(f"   Модель: models/ml_pro_v2.pkl")
    print(f"   Отчёт: models/training_report.json")
    print(f"   Пар: {len(pair_stats)}")
    print(f"   Всего окон: {len(X)}")
    
    if ml.feature_importance:
        print(f"\n🏆 Топ-10 признаков:")
        for fi in ml.feature_importance[:10]:
            print(f"   {fi['rank']:2d}. {fi['name']:25s} gain={fi['gain']:,}")

if __name__ == "__main__":
    main()
