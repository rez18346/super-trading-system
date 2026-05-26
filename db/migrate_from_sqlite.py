#!/usr/bin/env python3
"""Миграция данных из SQLite → PostgreSQL.

SQLite хранит всё в плоском логе trade_history (6029 строк):
  buy  = открытие позиции
  sell = закрытие (pnl указывает результат)

Переносим:
  - sell с |pnl| > 0.01 → закрытые сделки (entry = exit / (1 + pnl%/100))
  - buy без sell → открытые позиции
  - capital_snapshots → balance_history
"""

import os, sys
import sqlite3
import psycopg2
from datetime import datetime, timezone

SQLITE = os.path.expanduser("~/.openclaw/industrial_super_system/data/trading.db")
PG_DSN = "dbname=trading user=ksysha host=/tmp"

conn_sl = sqlite3.connect(SQLITE)
conn_sl.row_factory = sqlite3.Row
conn_pg = psycopg2.connect(PG_DSN)
cur = conn_pg.cursor()
print("✅ PG + SQLite connected")


def parse_ts(val):
    if not val:
        return None
    s = str(val).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s[:len(fmt)], fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except:
        return None


MIN_PNL = 0.01

# 1. Читаем все записи
rows = [dict(r) for r in conn_sl.execute("SELECT * FROM trade_history ORDER BY id")]
print(f"\n📊 trade_history: {len(rows)} записей")

# Разделяем buys и sells
buys = [r for r in rows if r["side"] == "buy"]
sells = [r for r in rows if r["side"] == "sell"
         and r["pnl"] and abs(float(r["pnl"])) >= MIN_PNL]
dust = [r for r in rows if r["side"] == "sell"
        and r["pnl"] and abs(float(r["pnl"])) < MIN_PNL]

print(f"   buys: {len(buys)}, meaningful sells: {len(sells)}, dust: {len(dust)}")

# 2. Закрытые сделки: sell → находим предшествующую buy
migrated = 0
no_pair = 0

for s in sells:
    sym = s["symbol"]
    st = parse_ts(s["timestamp"])
    sp = float(s["price"])
    sq = float(s["quantity"])
    pnl = float(s["pnl"])
    pnl_pct = float(s["pnl_pct"]) if s.get("pnl_pct") and float(s["pnl_pct"]) != 0 else None
    exit_reason = "sl" if pnl < 0 else "tp"

    entry_price = None
    entry_time = None
    entry_qty = sq

    # Ищем buy до этой sell (FIFO)
    pair = None
    for b in buys:
        if b["symbol"] == sym:
            bt = parse_ts(b["timestamp"])
            if bt and st and bt <= st:
                pair = b
                break

    if pair:
        entry_price = float(pair["price"])
        entry_time = parse_ts(pair["timestamp"])
        entry_qty = float(pair["quantity"])
        buys.remove(pair)  # потребляем
    else:
        # Восстанавливаем entry из PnL%
        if pnl_pct and abs(pnl_pct) > 0.001 and sp > 0:
            entry_price = round(sp / (1 + pnl_pct / 100), 8)
            entry_time = st
        else:
            no_pair += 1
            continue

    try:
        cur.execute("""
            INSERT INTO trades
                (symbol, side, status, account_id,
                 entry_price, entry_qty, entry_time,
                 exit_price, exit_qty, exit_time, exit_reason,
                 pnl, pnl_percent,
                 created_at, updated_at)
            VALUES (%s,%s,%s,1,
                    %s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,
                    NOW(), NOW())
            ON CONFLICT DO NOTHING
        """, (sym, "long", "closed",
              entry_price, entry_qty, entry_time,
              sp, sq, st, exit_reason,
              round(pnl, 4), round(pnl_pct, 2) if pnl_pct else None))
        migrated += 1
    except Exception as e:
        print(f"   ❌ {sym} id={s['id']}: {e}")

print(f"   ✅ закрытых сделок: {migrated}")
print(f"   ⏭️  sells без пары: {no_pair}")

# 3. Оставшиеся buys → открытые позиции
open_count = 0
for b in buys:
    ep = float(b["price"])
    if not ep or ep <= 0:
        continue
    try:
        cur.execute("""
            INSERT INTO trades
                (symbol, side, status, account_id,
                 entry_price, entry_qty, entry_time,
                 created_at, updated_at)
            VALUES (%s,%s,%s,1,
                    %s,%s,%s,
                    NOW(), NOW())
            ON CONFLICT DO NOTHING
        """, (b["symbol"], "long", "open",
              ep, float(b["quantity"]), parse_ts(b["timestamp"])))
        open_count += 1
    except Exception as e:
        print(f"   ❌ open {b['symbol']}: {e}")

print(f"   ✅ открытых позиций: {open_count}")

# 4. Баланс (capital_snapshots → balance_history)
print(f"\n💰 Баланс...")
cur.execute("SELECT COUNT(*) FROM balance_history")
if cur.fetchone()[0] == 0:
    snaps = list(conn_sl.execute(
        "SELECT * FROM capital_snapshots ORDER BY id"))
    print(f"   snapshots: {len(snaps)}")
    batch = []
    for r in snaps:
        batch.append((1, float(r["total"]), float(r["free_usdt"]),
                      None, parse_ts(r["created_at"])))
        if len(batch) >= 1000:
            cur.executemany(
                "INSERT INTO balance_history (account_id,balance,free,equity,recorded_at) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                batch)
            batch = []
    if batch:
        cur.executemany(
            "INSERT INTO balance_history (account_id,balance,free,equity,recorded_at) VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            batch)
    conn_pg.commit()
    print(f"   ✅ записано: {len(snaps)}")
else:
    print("   ⏭️ уже есть")

# 5. Итоговая проверка
print("\n🔍 Проверка:")
for tbl in ("trades", "balance_history"):
    cur.execute(f"SELECT COUNT(*) FROM {tbl}")
    print(f"   {tbl}: {cur.fetchone()[0]}")
cur.execute("SELECT SUM(pnl) FROM trades WHERE status='closed'")
s = cur.fetchone()[0]
print(f"   сумма PnL: {s:.2f} USD" if s else "   сумма PnL: 0")
cur.execute(
    "SELECT symbol, pnl, pnl_percent, exit_time FROM trades WHERE status='closed' ORDER BY exit_time DESC LIMIT 10")
print("   последние закрытые:")
for r in cur.fetchall():
    print(f"     {r[0]:<8}  PnL={r[1]:+.2f}  {r[2]:>6.2f}%  {str(r[3])[:16]}")

conn_sl.close()
conn_pg.close()
print("\n✅ Миграция завершена!")
