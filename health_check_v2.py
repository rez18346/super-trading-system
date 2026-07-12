#!/usr/bin/env python3
"""
health_check_v2.py — проверка супер-трейдера v2 (new_trader).
Правильная архитектура: FastAPI только на пинг (жив/мёртв),
все данные (баланс, позиции) — напрямую с биржи через ccxt.
"""

import sys, json, urllib.request, os, ccxt, subprocess

BASE_DIR = "/home/ksysha/new_trader"
INITIAL_CAPITAL = 60.0

def get_exchange():
    """Подключение к Bybit из .env (source-совместимый файл)."""
    # Читаем .env вручную, чтобы не зависеть от dotenv
    env_path = os.path.join(BASE_DIR, ".env")
    env = {}
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                env[k.strip()] = v.strip().strip("'\"")
    
    return ccxt.bybit({
        'apiKey': env.get('BYBIT_API_KEY', os.environ.get('BYBIT_API_KEY', '')),
        'secret': env.get('BYBIT_SECRET', os.environ.get('BYBIT_SECRET', '')),
    })

def check():
    results = []
    errors = 0
    criticals = 0
    
    # ── 1. FastAPI пинг ───────────────────────────────────────────────────────
    trader_pid = None
    try:
        req = urllib.request.Request("http://127.0.0.1:8765/api/status")
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read())
        
        if data.get('status') == 'ok' or data.get('running') == True:
            results.append(f"✅ FastAPI: порт 8765")
        else:
            errors += 1
            results.append(f"⚠️ FastAPI: unknown")
        
        trader_pid = data.get('trader_pid')
    except Exception as e:
        criticals += 1
        results.append(f"❌ FastAPI недоступен: {e}")
    
    # ── 2. Трейдер-процесс напрямую ──────────────────────────────────────────
    try:
        out = subprocess.check_output(['pgrep', '-f', 'super_trader.py']).decode().strip()
        pids = out.split('\n')
        screen_pids = subprocess.check_output(['pgrep', '-f', 'SCREEN.*trader']).decode().strip().split('\n')
        real_pids = [p for p in pids if p not in screen_pids]
        if real_pids:
            trader_pid = real_pids[0]
            results.append(f"✅ Трейдер: PID {','.join(real_pids)}")
        else:
            criticals += 1
            results.append(f"❌ super_trader.py не найден в процессах")
    except:
        if trader_pid:
            results.append(f"✅ Трейдер: PID {trader_pid}")
        else:
            criticals += 1
            results.append(f"❌ super_trader.py не найден в процессах")
    
    # ── 3. Баланс — напрямую с биржи ─────────────────────────────────────────
    total_balance = 0
    free_balance = 0
    try:
        exchange = get_exchange()
        bal = exchange.fetch_balance()
        
        total_balance = bal['total'].get('USDT', 0)
        free_balance = bal['free'].get('USDT', 0)
        
        # Суммируем мелкие монеты
        dust_value = 0
        for coin, qty in sorted(bal['total'].items()):
            if qty and qty > 0 and coin != 'USDT':
                try:
                    ticker = exchange.fetch_ticker(coin + '/USDT')
                    dust_value += qty * ticker['last']
                except:
                    pass
        
        total = total_balance + dust_value
        results.append(f"✅ 💳 Баланс: ${total:.2f} (USDT: ${total_balance:.2f} + dust: ${dust_value:.2f})")
        free_balance_for_check = total_balance + dust_value
    except Exception as e:
        errors += 1
        results.append(f"⚠️ Не удалось получить баланс с биржи: {e}")
        free_balance_for_check = 0
    
    # ── 4. Позиции — через API (БД) ─────────────────────────────────────────
    try:
        btc_info = {}
        try:
            req = urllib.request.Request("http://127.0.0.1:8765/api/status")
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read())
            
            pos_count = data.get('positions_count', 0)
            positions = data.get('positions', [])
            results.append(f"✅ 📦 Позиции: {pos_count}")
            for p in positions:
                results.append(f"   {p['symbol']:10s} {p['side']:5s} {p['entry_qty']:.4f} @ {p['entry_price']:.4f}")
            
            btc = data.get('btc', {})
            btc_info = btc
        except:
            pass
        
        results.append(f"✅ BTC: {btc_info.get('regime', '?')} / {btc_info.get('htf_trend', '?')}")
    except Exception as e:
        errors += 1
        results.append(f"⚠️ Ошибка при получении статуса: {e}")
    
    # ── 5. Критическая просадка ───────────────────────────────────────────────
    if free_balance_for_check > 0 and free_balance_for_check < INITIAL_CAPITAL * 0.7:
        criticals += 1
        drawdown_pct = (1 - free_balance_for_check / INITIAL_CAPITAL) * 100
        results.append(f"🔴 Просадка: ${free_balance_for_check:.2f} ({drawdown_pct:.1f}%)")
    
    exit_code = 2 if criticals > 0 else (1 if errors > 0 else 0)
    
    print("\n".join(results))
    print(f"\n{'🔴 КРИТИЧЕСКИ' if criticals else '✅ OK'} — ошибок: {errors}, критических: {criticals}")
    return exit_code

if __name__ == "__main__":
    ec = check()
    print(f"EXIT={ec}")
    sys.exit(ec)
