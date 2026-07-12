#!/usr/bin/env python3
"""
fetch_real_balance.py — Получает реальные данные с Bybit API
и сохраняет в /tmp/real_balance.json для дашборда.

Использует ccxt.bybit с ключами из .env.
Запускается раз в 30 секунд (--daemon) или по требованию.
"""

import json, os, sys, time, traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

def get_exchange():
    """Подключение к Bybit из .env (source-совместимый файл)."""
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
    
    import ccxt
    return ccxt.bybit({
        'apiKey': env.get('BYBIT_API_KEY', os.environ.get('BYBIT_API_KEY', '')),
        'secret': env.get('BYBIT_SECRET', os.environ.get('BYBIT_SECRET', '')),
        'options': {'defaultType': 'spot'},
    })


def fetch_real_data():
    """Получает реальные данные с Bybit и возвращает dict в формате real_balance.json."""
    exchange = get_exchange()
    if not exchange:
        return {'error': 'Не удалось создать exchange'}
    
    result = {
        'total': 0,
        'free_usdt': 0,
        'in_positions': 0,
        'positions_count': 0,
        'positions': {},
        'coins': {},
        'recent_trades': [],
        'open_orders': [],
        'timestamp': time.time(),
    }
    
    try:
        # 1. Баланс
        balance = exchange.fetch_balance()
        
        for coin, qty_raw in balance.get('total', {}).items():
            qty = float(qty_raw or 0)
            if qty <= 0:
                continue
            
            free = float(balance.get('free', {}).get(coin, 0) or 0)
            used = float(balance.get('used', {}).get(coin, 0) or 0)
            
            # Стоимость в USD
            if coin == 'USDT':
                usd_value = qty
            else:
                try:
                    ticker = exchange.fetch_ticker(f"{coin}/USDT")
                    price = float(ticker.get('last', 0) or 0)
                    usd_value = qty * price
                except:
                    usd_value = 0
            
            result['coins'][coin] = {
                'equity': round(qty, 8),
                'walletBalance': round(qty, 8),
                'usdValue': round(usd_value, 6),
                'locked': round(used, 8),
                'free': round(free, 8),
            }
            
            if coin == 'USDT':
                result['free_usdt'] = free
            elif usd_value > 0.01:
                sym = f"{coin}/USDT"
                result['positions'][sym] = {
                    'qty': qty,
                    'entry_price': 0,
                    'current_price': round(usd_value / qty, 6) if qty > 0 else 0,
                    'direction': 'LONG',
                    'usdValue': round(usd_value, 2),
                }
        
        total_usd = sum(float(c.get('usdValue', 0)) for c in result['coins'].values())
        result['total'] = round(total_usd, 2)
        in_pos = sum(p.get('usdValue', 0) for p in result['positions'].values())
        result['in_positions'] = round(in_pos, 2)
        result['positions_count'] = len(result['positions'])
        
        # 2. Открытые ордера
        try:
            orders = exchange.fetch_open_orders()
            for o in orders:
                result['open_orders'].append({
                    'symbol': o['symbol'],
                    'side': o['side'],
                    'type': o['type'],
                    'price': o.get('price'),
                    'amount': o.get('amount'),
                    'filled': o.get('filled'),
                    'remaining': o.get('remaining'),
                    'status': o['status'],
                })
        except:
            pass
        
        # 3. Последние сделки
        try:
            my_trades = []
            ALL_SYMS = ['SOL/USDT', 'NEAR/USDT', 'BTC/USDT', 'FIL/USDT',
                         'ADA/USDT', 'APT/USDT', 'DOT/USDT', 'DOGE/USDT',
                         'EGLD/USDT', 'ALGO/USDT', 'AVAX/USDT', 'OP/USDT',
                         'ATOM/USDT', 'LINK/USDT', 'ARB/USDT', 'XRP/USDT',
                         'MNT/USDT', 'ROSE/USDT', 'ETH/USDT']
            
            # Собираем ВСЕ трейды (до 50 на символ) и отдельно только BUY
            all_buy_trades = []
            for sym in ALL_SYMS:
                try:
                    trades = exchange.fetch_my_trades(sym, limit=50)
                    for t in trades:
                        entry = {
                            'symbol': t['symbol'],
                            'side': t['side'],
                            'price': t['price'],
                            'amount': t['amount'],
                            'cost': t['cost'],
                            'fee': t.get('fee', {}),
                            'timestamp': t['timestamp'],
                            'datetime': t.get('datetime', ''),
                        }
                        my_trades.append(entry)
                        if t['side'] == 'buy':
                            all_buy_trades.append(entry)
                except Exception as e:
                    pass
            
            result['all_buy_trades'] = all_buy_trades
            
            my_trades.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            result['recent_trades'] = my_trades[:30]
        except:
            pass
        
    except Exception as e:
        result['error'] = str(e)
        traceback.print_exc()
    
    return result


if __name__ == '__main__':
    run_daemon = '--daemon' in sys.argv
    single_run = '--oneshot' in sys.argv or not run_daemon
    
    while True:
        try:
            data = fetch_real_data()
            
            with open('/tmp/real_balance.json', 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            if single_run:
                ts = data.get('timestamp', time.time())
                tstr = time.strftime('%H:%M:%S', time.localtime(ts))
                print(f"[{tstr}] ✅ total=${data.get('total',0)} "
                      f"free=${data.get('free_usdt',0):.2f} "
                      f"in_pos=${data.get('in_positions',0):.2f} "
                      f"positions={data.get('positions_count',0)}")
                if data.get('open_orders'):
                    print(f"   Открытых ордеров: {len(data['open_orders'])}")
                if data.get('recent_trades'):
                    print(f"   Последних трейдов: {len(data['recent_trades'])}")
                if data.get('error'):
                    print(f"   ⚠️ Ошибка: {data['error']}")
                break
            
            time.sleep(60)
        except KeyboardInterrupt:
            print("\nОстановлен")
            break
        except:
            traceback.print_exc()
            time.sleep(60)
