#!/usr/bin/env python3
"""
📊 АНАЛИЗАТОР СДЕЛОК СУПЕР-СИСТЕМЫ V4

Парсит лог /tmp/system_v4.log, строит таблицу по каждой закрытой сделке.
Показывает PnL, причину выхода и тренд BTC на D1.

Запуск: python3 analyze_trades.py
"""

import re
import json
from datetime import datetime, timezone
from collections import defaultdict

LOG_FILE = "/tmp/system_v4.log"
TRACKER_FILE = "/home/ksysha/.openclaw/industrial_super_system/data/positions_tracker.json"

# Причины выхода из лога
REASONS = {
    "время удержания": "⏰ время",
    "ТРЕЙЛИНГ": "🔴 трейлинг",
    "ТЕЙК-ПРОФИТ": "🎯 тейк-профит",
    "Стоп-Лосс": "🛑 стоп-лосс",
}


def parse_log():
    """Парсит лог, возвращает список сделок."""
    trades = []
    
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Лог не найден: {LOG_FILE}")
        return trades

    current_trade = {}
    
    for line in lines:
        ts_match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if not ts_match:
            continue
        ts = ts_match.group(1)

        # BUY
        buy = re.search(r"BUY ([\d.]+) ([A-Z]+)/USDT @ \$?([\d.]+)", line)
        if buy:
            current_trade = {
                "pair": buy.group(2),
                "qty": float(buy.group(1)),
                "entry": float(buy.group(3)),
                "entry_time": ts,
                "exit": None,
                "exit_time": None,
                "reason": None,
            }
            continue

        # SELL
        sell = re.search(r"SELL ([\d.]+) ([A-Z]+)/USDT @ \$?([\d.]+)", line)
        if sell and current_trade and current_trade.get("pair") == sell.group(2):
            current_trade["exit"] = float(sell.group(3))
            current_trade["exit_time"] = ts
            continue

        # Причина выхода
        for keyword, reason in REASONS.items():
            if keyword in line and current_trade.get("pair") and keyword in line:
                # Проверяем, что это про нашу пару
                pair_in_line = current_trade.get("pair") in line
                if pair_in_line or keyword == "ТРЕЙЛИНГ" or keyword == "Стоп-Лосс":
                    current_trade["reason"] = reason
                    break

        # Удаление позиции = закрытие сделки
        removed = re.search(r"удален ([A-Z]+)/USDT.*PnL:\s*([+-]?[\d.]+)%", line)
        if removed:
            pair = removed.group(1)
            pnl_str = removed.group(2)
            if current_trade and current_trade.get("pair") == pair and current_trade.get("exit"):
                current_trade["pnl_pct"] = float(pnl_str)
                if current_trade.get("entry") and current_trade.get("exit"):
                    current_trade["pnl_usd"] = (current_trade["exit"] - current_trade["entry"]) * current_trade["qty"]
                trades.append(current_trade)
                current_trade = {}

    return trades


def get_btc_daily_trend():
    """Определяет тренд BTC на D1 через CCXT."""
    try:
        import ccxt
        c = json.load(open("/home/ksysha/.openclaw/industrial_super_system/config/api_config_final.json"))
        e = ccxt.bybit({
            "apiKey": c["bybit"]["api_key"],
            "secret": c["bybit"]["secret"],
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        ohlcv = e.fetch_ohlcv("BTC/USDT", "1d", limit=7)
        if len(ohlcv) >= 2:
            closes = [c[4] for c in ohlcv]
            ema7 = sum(closes) / len(closes)
            ema3 = sum(closes[-3:]) / 3
            if (closes[-1] > ema3 > ema7) or (closes[-1] > closes[-2] > closes[-3]):
                return "🟢 бычий"
            elif (closes[-1] < ema3 < ema7) or (closes[-1] < closes[-2] < closes[-3]):
                return "🔴 медвежий"
            else:
                return "🟡 боковик"
        return "❓ нет данных"
    except Exception as e:
        return f"❌ {str(e)[:30]}"


def main():
    trades = parse_log()
    
    if not trades:
        print("📭 Нет завершённых сделок в логе.")
        return

    btc_trend = get_btc_daily_trend()
    
    print("=" * 90)
    print(f"  📊 АНАЛИЗ СДЕЛОК СУПЕР-СИСТЕМЫ V4")
    print(f"  BTC (D1): {btc_trend}")
    print("=" * 90)
    print(f"{'Пара':<8} {'Вход':<8} {'Выход':<8} {'PnL %':<7} {'PnL $':<8} {'Время удерж.':<14} {'Причина':<16}")
    print("-" * 90)

    total_pnl = 0.0
    trade_count = 0
    by_reason = defaultdict(lambda: {"count": 0, "pnl": 0.0})

    for t in sorted(trades, key=lambda x: x.get("entry_time", "")):
        if not t.get("exit_time"):
            continue
        
        entry_t = datetime.strptime(t["entry_time"], "%Y-%m-%d %H:%M:%S")
        exit_t = datetime.strptime(t["exit_time"], "%Y-%m-%d %H:%M:%S")
        hold = exit_t - entry_t
        hold_str = f"{hold.seconds//3600}ч {(hold.seconds%3600)//60}м"
        
        pnl = t.get("pnl_pct", 0)
        pnl_usd = t.get("pnl_usd", 0)
        reason = t.get("reason", "❓ неизвестно")
        
        total_pnl += pnl_usd
        trade_count += 1
        by_reason[reason]["count"] += 1
        by_reason[reason]["pnl"] += pnl_usd
        
        pnl_str = f"{pnl:+.2f}%"
        usd_str = f"${pnl_usd:+.2f}"
        
        print(f"{t['pair']:<8} ${t['entry']:<6.4f} ${t['exit']:<6.4f} {pnl_str:<7} {usd_str:<8} {hold_str:<14} {reason:<16}")

    print("=" * 90)
    print(f"\n📈 ИТОГО: {trade_count} сделок | Суммарный PnL: ${total_pnl:+.2f}")
    print(f"\n📊 По причинам выхода:")
    for reason, data in sorted(by_reason.items(), key=lambda x: abs(x[1]["pnl"]), reverse=True):
        avg = data["pnl"] / data["count"] if data["count"] else 0
        print(f"  {reason:<16}: {data['count']:>2} шт | сумма ${data['pnl']:+.2f} | средняя ${avg:+.2f}")
    
    # Рынки
    try:
        import ccxt
        c = json.load(open("/home/ksysha/.openclaw/industrial_super_system/config/api_config_final.json"))
        e = ccxt.bybit({
            "apiKey": c["bybit"]["api_key"],
            "secret": c["bybit"]["secret"],
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        btc = e.fetch_ticker("BTC/USDT")
        print(f"\n🌍 РЫНОК СЕЙЧАС:")
        print(f"  BTC/USDT: ${btc['last']:,.2f} (день: {(btc['last']/btc['open']-1)*100:+.2f}%)")
    except:
        pass


if __name__ == "__main__":
    main()
