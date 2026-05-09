#!/usr/bin/env python3
"""
Retrain ML Professional V2 (27-признаковая модель) на свежих данных.

Скачивает 1H и 4H свечи через Bybit API за последние 14 дней,
строит признаковое пространство через build_features_27f(),
генерирует метки по forward-return (shift 3),
обучает LightGBM, сохраняет модель.
"""

import sys, os, json, time, pickle
import numpy as np
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_professional_v2 import build_features_27f, FEATURE_NAMES_25

# ─── ПАРАМЕТРЫ ──────────────────────────────────────────────────────────────

PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT",
    "XRPUSDT", "LINKUSDT", "AVAXUSDT", "ATOMUSDT", "OPUSDT",
    "MNTUSDT", "NEARUSDT", "FILUSDT", "EGLDUSDT", "ALGOUSDT",
    "APTUSDT", "ARBUSDT", "DOGEUSDT", "ROSEUSDT"
]

DAYS_HISTORY = 14        # сколько дней истории скачать
FORWARD_CANDLES = 3      # на сколько 1H-свечей вперёд смотрим для метки
PROFIT_THRESHOLD = 0.003 # 0.3% — порог профита для положительной метки
LOSS_THRESHOLD = -0.003  # -0.3% — порог убытка для отрицательной

SAVE_PATH = os.path.join(os.path.dirname(__file__), "models", "ml_pro_v2_27f.pkl")
TEMP_MODEL = os.path.join(os.path.dirname(__file__), "models", "ml_pro_v2_27f_new.pkl")

# ─── ЗАГРУЗКА СВЕЧЕЙ ЧЕРЕЗ BYBIT ───────────────────────────────────────────

def fetch_klines(symbol: str, interval: str, limit: int = 500) -> list:
    """
    Скачать свечи через REST API Bybit.
    Возвращает список dict с ключами {t, o, h, l, c, v}
    1H = "60", 4H = "240" (в минутах для Bybit v5)
    """
    import requests
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if data.get("retCode") != 0:
            print(f"  ⚠️  {symbol} {interval}: {data.get('retMsg', 'error')}")
            return []
        rows = []
        for item in data["result"]["list"]:
            rows.append({
                "t": int(item[0]),
                "o": float(item[1]),
                "h": float(item[2]),
                "l": float(item[3]),
                "c": float(item[4]),
                "v": float(item[5])
            })
        rows.reverse()
        return rows
    except Exception as e:
        print(f"  ❌ {symbol} {interval}: {e}")
        return []


def fetch_all_candles(symbol: str, interval: str, max_days: int = 14) -> list:
    """
    Скачать до max_days дней истории через пагинацию (по 200 свечей).
    Bybit использует минутные интервалы: 1H="60", 4H="240"
    """
    candles_per_day = 1440 // int(interval)
    all_rows = []
    cursor = None
    limit = 200
    max_items = candles_per_day * max_days + 50
    attempts = 0

    import requests
    url = "https://api.bybit.com/v5/market/kline"

    while len(all_rows) < max_items and attempts < 5:
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor

        try:
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json()
            if data.get("retCode") != 0:
                print(f"  ⚠️  {symbol} {interval}: {data.get('retMsg', 'error')}")
                break
            items = data["result"]["list"]
            if not items:
                break
            for item in items:
                all_rows.append({
                    "t": int(item[0]),
                    "o": float(item[1]),
                    "h": float(item[2]),
                    "l": float(item[3]),
                    "c": float(item[4]),
                    "v": float(item[5])
                })
            cursor = data["result"].get("nextPageCursor")
            if not cursor:
                break
        except Exception as e:
            print(f"  ❌ {symbol} {interval} page: {e}")
            break
        attempts += 1
        time.sleep(0.3)

    all_rows.sort(key=lambda x: x["t"])
    return all_rows


def make_labels(closes: np.ndarray, forward: int = 3, thr: float = 0.003):
    """
    Генерирует метки для обучения:
      1 — цена выросла на thr% за forward свечей
      0 — иначе (включая падение и флэт)
    Возвращает np.ndarray той же длины (первые forward меток = NaN).
    """
    n = len(closes)
    labels = np.full(n, np.nan)
    for i in range(n - forward):
        ret = (closes[i + forward] - closes[i]) / closes[i]
        labels[i] = 1.0 if ret >= thr else 0.0
    return labels


# ─── ОБУЧЕНИЕ ────────────────────────────────────────────────────────────────

def train_model(X, y, feature_names, acc_baseline=0.50):
    """Обучить LightGBM, вернуть (model, accuracy)."""
    n = len(X)
    te = int(n * 0.70)
    va = int(n * 0.85)

    print(f"  Обучающая выборка: {te} (70%)")
    print(f"  Валидация: {va - te} (15%)")
    print(f"  Тест: {n - va} (15%)")
    print(f"  Баланс меток (1/0): {y[:va].sum():.0f}/{va - y[:va].sum():.0f} "
          f"({y[:va].mean()*100:.1f}% позитивных)")

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'seed': 42,
    }

    td = lgb.Dataset(X[:te], label=y[:te], feature_name=feature_names)
    vd = lgb.Dataset(X[te:va], label=y[te:va], feature_name=feature_names, reference=td)

    model = lgb.train(
        params, td, num_boost_round=1000,
        valid_sets=[vd],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
    )

    # Оценка на тесте
    yp = model.predict(X[va:])
    yb = (yp > 0.5).astype(int)
    acc = float(np.mean(yb == y[va:]))
    print(f"  Точность на тесте: {acc:.2%} ({n - va} samples)")

    # Feature importance
    imp = model.feature_importance('gain')
    top10 = sorted(
        [{"rank": i + 1, "name": feature_names[j] if j < len(feature_names) else f"f{j}",
          "gain": int(imp[j])}
         for i, j in enumerate(np.argsort(imp)[::-1][:10])],
        key=lambda x: x['rank']
    )
    print(f"\n  Топ-10 признаков:")
    for fi in top10:
        print(f"    {fi['rank']}. {fi['name']:20s} gain={fi['gain']}")

    # Если точность не лучше baseline — сохраняем старую модель
    if acc < acc_baseline + 0.02:
        print(f"\n  ⚠️  Точность {acc:.2%} ниже baseline {acc_baseline:.2%} + 2%. Модель не сохранена.")
        return None, acc

    return model, acc


def save_model(model, acc, samples, features, importance, path):
    """Сохранить модель в pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "model": model,
        "trained": True,
        "acc": acc,
        "samples": samples,
        "features": features,
        "importance": importance,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"\n  ✅ Модель сохранена: {path} ({samples:,} обр., {acc:.2%} acc)")
    return path


# ─── ГЛАВНАЯ ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🔄 RETRAIN ML Professional V2 (27f)")
    print(f"   Период: {DAYS_HISTORY} дней")
    print(f"   Метка: forward {FORWARD_CANDLES}H, профит ≥ {PROFIT_THRESHOLD*100:.1f}%")
    print(f"   Пар: {len(PAIRS)}")
    print("=" * 60)

    all_X = []
    all_y = []
    pair_stats = {}

    for pair in PAIRS:
        symbol = pair
        print(f"\n📥 {symbol}: загрузка данных...")

        # Скачиваем 1H (60 мин) и 4H (240 мин) — Bybit v5 использует минуты
        h1 = fetch_all_candles(symbol, "60", max_days=DAYS_HISTORY)
        h4 = fetch_all_candles(symbol, "240", max_days=DAYS_HISTORY)

        if len(h1) < 100 or len(h4) < 25:
            print(f"  ⏭️  Недостаточно данных: 1H={len(h1)}, 4H={len(h4)}")
            continue

        # Строим признаки
        X_pair = build_features_27f(h1, h4)
        if X_pair.shape[0] < 50:
            print(f"  ⏭️  Признаков: {X_pair.shape[0]}")
            continue

        # Метки по close 1H
        closes_h1 = np.array([c['c'] for c in h1], dtype=float)
        y_pair = make_labels(closes_h1, forward=FORWARD_CANDLES, thr=PROFIT_THRESHOLD)

        # Синхронизация длины (build_features_27f возвращает ~len(h1), 
        # но может быть обрезана из-за синхронизации)
        min_len = min(X_pair.shape[0], len(y_pair))
        X_pair = X_pair[:min_len]
        y_pair = y_pair[:min_len]

        # Убираем NaN метки (первые forward точек)
        valid = ~np.isnan(y_pair)
        X_pair = X_pair[valid]
        y_pair = y_pair[valid]

        if len(X_pair) < 20:
            print(f"  ⏭️  После фильтра: {len(X_pair)}")
            continue

        positive = int(y_pair.sum())
        total = len(y_pair)
        print(f"  ✅ Признаков: {X_pair.shape}, меток: {total} (поз: {positive} = {positive/total*100:.1f}%)")

        all_X.append(X_pair)
        all_y.append(y_pair)
        pair_stats[symbol] = {
            "windows": total,
            "balance": f"{positive/total*100:.1f}%"
        }

    if not all_X:
        print("\n❌ Нет данных для обучения")
        return

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    print(f"\n{'='*60}")
    print(f"📊 ИТОГО: {X_all.shape[0]} образцов, {X_all.shape[1]} признаков")
    print(f"   Позитивные: {int(y_all.sum())} / {len(y_all)} ({y_all.mean()*100:.1f}%)")
    print(f"   Пар: {len(pair_stats)}")
    print(f"{'='*60}")

    # FEATURE_NAMES_25 — имена для первых 25 признаков (или 33 — все)
    n_feats = X_all.shape[1]
    if n_feats <= 25:
        f_names = FEATURE_NAMES_25[:n_feats]
    else:
        from ml_professional_v2 import FEATURE_NAMES_33
        f_names = FEATURE_NAMES_33[:n_feats]

    # Обучение
    model, acc = train_model(X_all, y_all, f_names)

    if model is None:
        print("\n⚠️  Модель не улучшила baseline. Старая модель оставлена.")
        # Всё равно сохраняем как новую модель для сравнения
        imp = None
    else:
        imp = model.feature_importance('gain')
        imp_list = [{"rank": i + 1,
                     "name": f_names[j] if j < len(f_names) else f"f{j}",
                     "gain": int(imp[j])}
                    for i, j in enumerate(np.argsort(imp)[::-1][:10])]

    # Всегда сохраняем (даже если точность низкая — для диагностики)
    save_model(model if model else None, acc if model else 0.50,
               len(y_all), f_names, imp_list if model else [],
               TEMP_MODEL)

    # Если точность хорошая — заменяем основную модель
    if model and acc >= 0.52:
        save_model(model, acc, len(y_all), f_names, imp_list, SAVE_PATH)
        print(f"\n🎉 НОВАЯ МОДЕЛЬ АКТИВНА!")
    elif model:
        print(f"\n⚠️  Точность {acc:.2%} < 52%. Новая модель сохранена как {TEMP_MODEL}")
        print("   Основная модель НЕ заменена.")

    # Сохраняем отчёт
    report_path = os.path.join(os.path.dirname(__file__), "models", "training_report.json")
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_samples": int(len(y_all)),
        "features": int(X_all.shape[1]),
        "label_balance": float(y_all.mean()),
        "accuracy": acc if model else 0,
        "pairs": [{"pair": p, "windows": int(s["windows"]), "balance": s["balance"]}
                  for p, s in sorted(pair_stats.items())],
        "status": "success" if model else "low_accuracy"
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📋 Отчёт: {report_path}")
    print("✅ Готово!")


if __name__ == "__main__":
    main()
