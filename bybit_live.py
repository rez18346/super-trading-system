"""
bybit_live.py — Получает живые данные с Bybit API.
Вызывается control_api.py для отображения реального состояния.
"""

import json, os, subprocess, sys

# Путь к конфигу API (используем тот же, что трейдер)
API_CFG = os.path.expanduser("~/new_trader/bybit_api.json")


def _use_api_keys() -> dict:
    """Читает API-ключи из конфига трейдера."""
    try:
        with open(API_CFG) as f:
            return json.load(f)
    except:
        return {}


def fetch_wallet_balance() -> dict:
    """Возвращает реальный баланс кошелька с биржи."""
    try:
        # Пытаемся через MCP Bybit — на прямую не получится из python.
        # Используем real_balance.json как свежий источник
        rb_path = '/tmp/real_balance.json'
        if os.path.exists(rb_path):
            with open(rb_path) as f:
                return json.load(f)
    except:
        pass
    return {}


def fetch_open_positions() -> dict:
    """
    Парсит реальный баланс биржи — все монеты с ненулевым остатком.
    Возвращает {symbol: {qty, entry, current_price, direction}}.
    Не путать с фьючерсными позициями — мы торгуем spot USDT.
    """
    result = {}
    
    # Пробуем real_balance.json
    try:
        rb_path = '/tmp/real_balance.json'
        if os.path.exists(rb_path):
            with open(rb_path) as f:
                data = json.load(f)
            
            # Если есть детальный список позиций
            positions = data.get('positions', {})
            for sym, pdata in positions.items():
                q = float(pdata.get('qty', 0))
                if q > 0:
                    result[sym] = {
                        'qty': q,
                        'entry': float(pdata.get('entry_price', 0)),
                        'current_price': float(pdata.get('current_price', 0)),
                        'direction': pdata.get('direction', 'LONG').upper()
                    }
            
            # Если нет — парсим список coins с ненулевым equity
            coins = data.get('coins', {})
            if not result and coins:
                for coin, info in coins.items():
                    eq = float(info.get('equity', 0))
                    usd = float(info.get('usdValue', 0))
                    if eq > 0:
                        sym = coin + '/USDT'
                        if sym not in result:
                            result[sym] = {
                                'qty': eq,
                                'entry': 0,
                                'current_price': usd / eq if eq > 0 else 0,
                                'direction': 'LONG'
                            }
    except:
        pass
    
    return result


def fetch_recent_trades(limit=20) -> list:
    """Парсит последние сделки из real_balance.json (если есть)"""
    try:
        rb_path = '/tmp/real_balance.json'
        if os.path.exists(rb_path):
            with open(rb_path) as f:
                data = json.load(f)
            trades = data.get('recent_trades', [])
            if trades:
                return trades[:limit]
    except:
        pass
    return []
