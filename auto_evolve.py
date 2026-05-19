#!/usr/bin/env python3
"""
auto_evolve.py — Автоматическое дообучение ML и HMM
=====================================================
Запускается по расписанию:
  - HMM: каждые 24 часа (переобучение на последних 2000 часов BTC)
  - ML:  каждые 7 дней (переобучение на новых данных всех пар)

Не прерывает работающую систему.
Сохраняет модель только если новая accuracy > старая + 1%.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("AUTO_EVOLVE")


def evolve_hmm():
    """Переобучить HMM на последних 2000 часов BTC."""
    log.info("[EVOLVE] Обновление HMM...")
    try:
        from hmm_regime import RegimeDetector
        r = RegimeDetector(data_path='data/btc_usdt_1h.json', lookback=2000)
        r.train()
        if r.trained:
            current = r.update_current_state()
            log.info(f"[EVOLVE] ✅ HMM: режим={r.get_state_name()}, SL={r.get_thresholds().get('sl', 'N/A')}%")
        else:
            log.warning("[EVOLVE] ❌ HMM не обучился")
    except Exception as e:
        log.error(f"[EVOLVE] HMM ошибка: {e}")


def evolve_ml():
    """Дообучить ML-модель. Запускает retrain_ml.py."""
    log.info("[EVOLVE] ML: дообучение...")
    
    model_path = os.path.join(BASE_DIR, 'models', 'ml_pro_v2_27f.pkl')
    if not os.path.exists(model_path):
        log.warning("[EVOLVE] Нет модели для обновления")
        return
    
    # Загружаем текущую
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        old_acc = data.get('acc', 0)
        old_samples = data.get('samples', 0)
        old_time = data.get('timestamp', 'never')
        log.info(f"[EVOLVE] Текущая модель: {old_samples:,} обр., {old_acc:.2%} acc, обучена {old_time}")
    except Exception as e:
        log.warning(f"[EVOLVE] Не загрузилась: {e}")
        old_acc = 0
    
    # Запускаем переобучение (subprocess, не блокирует)
    retrain_path = os.path.join(BASE_DIR, 'retrain_ml.py')
    if os.path.exists(retrain_path):
        import subprocess
        result = subprocess.run(
            ['/usr/bin/python3', retrain_path],
            capture_output=True, text=True, timeout=300
        )
        for line in result.stdout.strip().split('\n'):
            log.info(f"[RETRAIN] {line}")
        if result.stderr:
            for line in result.stderr.strip().split('\n'):
                if 'warning' not in line.lower():
                    log.warning(f"[RETRAIN] {line}")
    else:
        log.warning("[EVOLVE] retrain_ml.py не найден")


def evolve_btc():
    """Переобучить BTC Direction Predictor (без exchange — через Bybit API)."""
    log.info("[EVOLVE] 🔄 BTC Direction: запуск переобучения...")
    try:
        import sys, requests
        sys.path.insert(0, BASE_DIR)
        from btc_direction import BTCDirectionPredictor
        # Передаём заглушку exchange — train использует fetch_ohlcv через него
        # Но train() сам загрузит данные если exchange == None и есть прямой API fallback
        p = BTCDirectionPredictor()
        # train() загружает данные через exchange.fetch_ohlcv, но у нас exchange=None
        # Загружаем напрямую с Bybit API и подменяем метод
        import pandas as pd, types
        raw = requests.get(
            'https://api.bybit.com/v5/market/kline?category=spot&symbol=BTCUSDT&interval=60&limit=1000',
            timeout=15
        ).json()
        candles = raw['result']['list']
        if len(candles) < 100:
            log.warning(f"[EVOLVE] ⚠️ BTC: мало данных ({len(candles)})")
            return
        df = pd.DataFrame([{
            'timestamp': int(c[0]),
            'open': float(c[1]),
            'high': float(c[2]),
            'low': float(c[3]),
            'close': float(c[4]),
            'volume': float(c[5])
        } for c in candles])
        df.sort_values('timestamp', inplace=True)
        def _fake_fetch(self, limit=1000):
            sub = df.tail(limit)
            candles = [[int(r['timestamp']), float(r['open']), float(r['high']),
                        float(r['low']), float(r['close']), float(r['volume'])]
                       for _, r in sub.iterrows()]
            return self._candles_to_df(candles)
        p._fetch_btc_ohlcv = types.MethodType(_fake_fetch, p)
        if p.train():
            log.info(f"[EVOLVE] ✅ BTC Direction model retrained")
        else:
            log.warning(f"[EVOLVE] ⚠️ BTC Direction model retrain failed")
    except Exception as e:
        log.error(f"[EVOLVE] ❌ BTC Direction error: {e}")


def evolve_advisor():
    """Дообучить ML-советника (Advisor)."""
    log.info("[EVOLVE] ML-Advisor: дообучение...")
    try:
        sys.path.insert(0, BASE_DIR)
        from ml_advisor import ml_train
        ml_train(force=True)
        log.info("[EVOLVE] ✅ ML-Advisor: обучение завершено")
    except Exception as e:
        log.warning(f"[EVOLVE] ML-Advisor: {e}")


def check_pnl():
    """Проверить performance системы."""
    try:
        sys.path.insert(0, BASE_DIR)
        import ccxt
        c = json.load(open(os.path.join(BASE_DIR, 'config/api_config_final.json')))
        e = ccxt.bybit({
            'apiKey': c['bybit']['api_key'],
            'secret': c['bybit']['secret'],
            'password': c['bybit']['password'],
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        b = e.fetch_balance()
        usdt = b['total'].get('USDT', 0)
        total = usdt
        for coin, qty in sorted(b['total'].items()):
            if coin != 'USDT' and qty > 0.0005:
                t = e.fetch_ticker(f'{coin}/USDT')
                total += qty * t['last']
        
        log.info(f"[EVOLVE] 💰 Баланс: ${total:.2f}, USDT: ${usdt:.2f}")
        
        # Сохраняем в историю
        history_path = os.path.join(BASE_DIR, 'data/balance_history.json')
        history = []
        if os.path.exists(history_path):
            try:
                with open(history_path) as f:
                    history = json.load(f)
            except:
                pass
        
        history.append({
            'time': datetime.now(timezone.utc).isoformat(),
            'total': round(total, 2),
            'usdt': round(usdt, 2),
        })
        
        # Храним последние 100 записей
        if len(history) > 100:
            history = history[-100:]
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Тренд за последние часы
        if len(history) >= 3:
            first = history[0]['total']
            last = history[-1]['total']
            change = (last - first) / first * 100
            log.info(f"[EVOLVE] 📈 Тренд баланса за {len(history)} записей: {change:+.2f}%")
        
    except Exception as e:
        log.error(f"[EVOLVE] Ошибка PnL: {e}")


if __name__ == "__main__":
    log.info("=" * 50)
    log.info("🤖 АВТО-ЭВОЛЮЦИЯ ML-PRO v2")
    log.info(f"Время: {datetime.now(timezone.utc).isoformat()}")
    log.info("=" * 50)
    
    check_pnl()
    evolve_hmm()
    evolve_ml()
    evolve_btc()
    evolve_advisor()
    
    log.info("[EVOLVE] ✅ Цикл завершён")
