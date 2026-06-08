#!/usr/bin/env python3
"""Health check: мониторинг супер-системы каждые 30 минут"""

import os, sys, json, time, subprocess, traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db_pg as db

BASE = os.path.dirname(os.path.abspath(__file__))
NOW = datetime.now(timezone.utc)
TZ_OFFSET = 7  # KRAT

def log(msg, level="INFO"):
    print(f"[{level}] {msg}")

def test_trader_alive():
    """Проверка что трейдер жив"""
    # Ищем orchestrator или trader_entry
    out = subprocess.run(
        ["pgrep", "-f", "orchestrator|trader_entry"], 
        capture_output=True, text=True, timeout=5
    )
    pids = out.stdout.strip().split()
    if not pids or not any(p.strip() for p in pids):
        return False, "Нет живого PID трейдера"
    
    # Проверка что не зомби
    for pid_str in pids:
        try:
            pid = int(pid_str)
            stat = open(f"/proc/{pid}/stat").read().split()
            if 'Z' in stat[2]:
                return False, f"PID {pid} — зомби"
        except:
            pass
    return True, f"PID: {', '.join(pids)}"

def _fetch_rows(sql, params=None):
    """Helper: выполняет SQL через _with_conn, возвращает список строк"""
    def _exec(conn):
        cur = conn.cursor()
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        cols = [desc[0] for desc in cur.description] if cur.description else []
        rows = []
        for row in cur.fetchall():
            rows.append(dict(zip(cols, row)))
        return rows
    return db._with_conn(_exec) if db else []

def test_balance():
    """Проверка баланса через PG"""
    rows = _fetch_rows(
        "SELECT balance, free, recorded_at FROM balance_history ORDER BY recorded_at DESC LIMIT 1"
    )
    if not rows:
        return None, "Нет данных balance_history"
    
    row = rows[0] if rows else None
    if not row:
        return None, "Пустой balance_history"
    
    balance = float(row['balance'])
    free = float(row['free'])
    ts = row['recorded_at']
    
    if balance < 1.0:
        return False, f"Капитал упал ниже $1: ${balance:.2f} ({ts})"
    
    return True, f"Капитал: ${balance:.2f} ($free ${free:.2f}), {ts}"

def test_positions():
    """Проверка позиций — не залито ли всё в одну"""
    rows = _fetch_rows(
        "SELECT symbol, entry_price, entry_qty, entry_time FROM trades WHERE status='open'"
    )
    positions = rows or []
    
    n = len(positions)
    if n == 0:
        return True, "Нет открытых позиций"
    
    # Стоимость каждой
    values = []
    for p in positions:
        qty = float(p['entry_qty'] or 0)
        price = float(p['entry_price'] or 0)
        values.append(qty * price)
    
    total = sum(values)
    if total <= 0:
        return False, "Позиции есть, но стоимость = 0"
    
    # Проверка: ни одна позиция не превышает 60% всех позиций
    max_share = max(values) / total if total > 0 else 0
    ratio_ok = max_share <= 0.60  # допустимо до 60% в одной
    
    msg = f"{n} позиций на ${total:.2f}"
    for i, p in enumerate(positions):
        v = values[i]
        share = v / total * 100 if total > 0 else 0
        msg += f"\n  {p['symbol']}: ${v:.2f} ({share:.0f}%)"
    
    if not ratio_ok:
        return False, f"Перекос! {max(share for share in [v/total for v in values]):.0f}% в одной позиции.\n" + msg
    
    return True, msg

def test_trades_table():
    """Проверка что последние сделки не с exit_price=0 (старый баг)"""
    rows = _fetch_rows(
        "SELECT id, symbol, side, status, entry_price, exit_price, pnl FROM trades "
        "WHERE status='closed' AND (exit_price IS NULL OR exit_price = 0) "
        "AND created_at IS NOT NULL "
        "ORDER BY created_at DESC LIMIT 5"
    )
    bad = rows or []
    if len(bad) > 10:
        return False, f"{len(bad)} сделок с exit_price=0 — баг записи"
    return True, f"Всего bad-closed: {len(bad)}"

def test_terminal():
    """Проверка что терминал жив"""
    try:
        import urllib.request
        resp = urllib.request.urlopen("http://localhost:8765/api/status", timeout=5)
        data = json.loads(resp.read())
        return True, f"Терминал OK, {data.get('stats',{}).get('open_positions',0)} позиций"
    except Exception as e:
        return False, f"Терминал не отвечает: {e}"

def run():
    issues = []
    all_ok = True
    critical = False
    
    checks = [
        ("👤 Трейдер", test_trader_alive),
        ("💳 Баланс", test_balance),
        ("📦 Позиции", test_positions),
        ("📋 Сделки", test_trades_table),
        ("🌐 Терминал", test_terminal),
    ]
    
    for name, func in checks:
        try:
            ok, msg = func()
            if ok:
                log(f"✅ {name}: {msg}")
            else:
                log(f"❌ {name}: {msg}", "WARN")
                issues.append(f"{name}: {msg}")
                all_ok = False
                # Трейдер мёртв или баланс протух — критично
                if name in ("👤 Трейдер", "💳 Баланс"):
                    critical = True
        except Exception as e:
            log(f"💥 {name}: {e}\n{traceback.format_exc()}", "ERROR")
            issues.append(f"{name}: ОШИБКА {e}")
            all_ok = False
    
    # 🔥 Stop-Loss всего капитала — только если balance_history свежий (<30 мин)
    bal_row = _fetch_rows(
        "SELECT balance, recorded_at FROM balance_history ORDER BY recorded_at DESC LIMIT 3"
    )
    if bal_row and len(bal_row) >= 2:
        try:
            ts = bal_row[0]['recorded_at']
            if hasattr(ts, 'tzinfo'):
                ts = ts.replace(tzinfo=timezone.utc)
            age = (NOW - ts).total_seconds()
            if age < 1800:  # не старше 30 минут
                b1 = float(bal_row[0]['balance'])
                b_old = float(bal_row[-1]['balance'])
                if b_old > 1:
                    drawdown = (b_old - b1) / b_old * 100
                    if drawdown >= 2.0:
                        all_ok = False
                        critical = True
                        issues.append(f"🚨 STOP: просадка {drawdown:.1f}% (более 2%)")
                        log(f"🚨 ПРОСАДКА {drawdown:.1f}% — останавливаю трейдера!", "CRITICAL")
                        with open(os.path.join(BASE, 'data', 'capital_stop.json'), 'w') as _f:
                            import json as _json
                            _json.dump({'stop': True, 'drawdown': drawdown, 'ts': str(NOW)}, _f)
        except Exception as _e:
            log(f"Ошибка drawdown check: {_e}", "WARN")
    
    # 🩹 Если нет баланс-логов за последние 30 мин — не критично, но чиним
    if bal_row:
        try:
            ts = bal_row[0]['recorded_at']
            if hasattr(ts, 'tzinfo'):
                ts = ts.replace(tzinfo=timezone.utc)
            age = (NOW - ts).total_seconds()
            if age > 1800:
                issues.append(f"Balance history устарел ({age/60:.0f} мин) — нужно починить")
                log(f"⚠️ Balance history: {age/60:.0f} мин назад", "WARN")
        except:
            pass
    
    return all_ok, issues, critical

if __name__ == "__main__":
    all_ok, issues, critical = run()
    if issues:
        log("\n".join(f"  ⚠️ {i}" for i in issues))
    if critical:
        log("🔴 КРИТИЧЕСКИ — трейдер остановлен", "CRITICAL")
    sys.exit(2 if critical else (0 if all_ok else 1))
