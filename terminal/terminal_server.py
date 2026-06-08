#!/usr/bin/env python3
"""
terminal_server.py — Профессиональный торговый терминал.
FastAPI + PostgreSQL + статика. Запускается отдельно от трейдера.
"""

import json
import os
import sys
import logging
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Пути ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PROJECT_DIR = os.path.dirname(BASE_DIR)  # industrial_super_system

# ── PG ─────────────────────────────────────────────────────────────────────────
import psycopg2
import psycopg2.extras

PG_DSN = "host=/tmp dbname=trading user=ksysha"

log = logging.getLogger("terminal")


def get_db():
    return psycopg2.connect(PG_DSN)


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 Терминал запущен")
    yield
    log.info("🛑 Терминал остановлен")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Trading Terminal", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── API ────────────────────────────────────────────────────────────────────────

@app.get("/api/btc")
def api_btc():
    """BTC анализ — regime, цена, RSI, структура, рекомендация (из meta + signal_log)."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # 1. Сначала из meta (btc_analysis — пишется трейдером раз в 5-10 мин)
        cur.execute("SELECT value FROM meta WHERE key = 'btc_analysis'")
        meta_row = cur.fetchone()
        
        result = {"price": None, "regime": "unknown", "rsi": None,
                   "recommendation": "wait", "structure": "neutral",
                   "components": {}, "signal_time": None, "signal_age_min": None,
                   "meta_time": None, "ema_trend": None}
        
        if meta_row:
            try:
                m = json.loads(meta_row['value'])
                result["meta_time"] = m.get('timestamp', '')
                result["regime"] = m.get('regime', 'unknown')
                result["recommendation"] = m.get('recommendation', 'wait')
                result["rsi"] = round(float(m.get('rsi', 0)), 1) if m.get('rsi') else None
                result["structure"] = m.get('structure', 'neutral')
                result["ema_trend"] = m.get('ema_trend', m.get('htf_trend', 'neutral'))
                
                # Возраст meta-записи
                if m.get('timestamp'):
                    try:
                        meta_time = datetime.fromisoformat(m['timestamp'])
                        age = datetime.now(timezone.utc) - meta_time.replace(tzinfo=timezone.utc)
                        result["signal_age_min"] = round(age.total_seconds() / 60, 1)
                    except:
                        pass
            except Exception as e:
                log.warning(f"Ошибка парсинга btc_analysis meta: {e}")
        
        # 2. Всегда подгружаем компоненты из signal_log (для отображения)
        cur.execute("""
            SELECT symbol, decision, score, threshold, components, created_at
            FROM signal_log
            WHERE symbol = 'BTCUSDT' OR symbol = 'BTC/USDT'
            ORDER BY created_at DESC LIMIT 1
        """)
        btc_signal = cur.fetchone()
        
        if btc_signal:
            comp = btc_signal['components']
            if isinstance(comp, str):
                try:
                    comp = json.loads(comp)
                except:
                    comp = {}
            
            # Если meta не установила regime — берём из signal_log
            if not meta_row or result.get('regime') == 'unknown':
                bonus = comp.get('bonus', {})
                if isinstance(bonus, dict):
                    btc_bonus = float(bonus.get('btc', 0) or 0)
                    if btc_bonus <= -20:
                        result["regime"] = "distribution"
                        result["recommendation"] = "sell_only"
                    elif btc_bonus <= -10:
                        result["regime"] = "bearish_side"
                        result["recommendation"] = "sell_only"
                    elif btc_bonus <= -5:
                        result["regime"] = "dump"
                        result["recommendation"] = "caution"
                    else:
                        result["regime"] = "accumulation"
                        result["recommendation"] = "buy_allowed"
                
                rsi = comp.get('rsi_vol_btc')
                if rsi is not None:
                    result["rsi"] = round(float(rsi), 1)
                
                mtf = comp.get('mtf')
                if mtf is not None:
                    mtf = float(mtf)
                    if mtf > 60: result["structure"] = "bullish"
                    elif mtf < 40: result["structure"] = "bearish"
                    else: result["structure"] = "neutral"
            
            # Компоненты — плоские (всегда)
            flat_comp = {}
            for k, v in comp.items():
                if isinstance(v, dict):
                    if k == 'bonus':
                        for sk, sv in v.items():
                            flat_comp[f"bonus_{sk}"] = sv
                    else:
                        for sk, sv in v.items():
                            if not isinstance(sv, (dict, list)):
                                flat_comp[f"{k}_{sk}"] = sv
                elif not isinstance(v, (dict, list)):
                    flat_comp[k] = v
            result["components"] = flat_comp
            result["score"] = float(btc_signal['score'] or 0)
            result["threshold"] = float(btc_signal['threshold'] or 60)
            result["signal_time"] = str(btc_signal['created_at'])
            
            if btc_signal['created_at'] and not result.get('signal_age_min'):
                try:
                    age = datetime.now(timezone.utc) - btc_signal['created_at'].replace(tzinfo=timezone.utc)
                    result["signal_age_min"] = round(age.total_seconds() / 60, 1)
                except:
                    pass
        
        return result
    finally:
        cur.close()
        conn.close()


@app.get("/api/status")
def api_status():
    """Общий статус: баланс, открытые позиции, equity."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # Текущий баланс
        cur.execute("""
            SELECT balance, free, equity, recorded_at
            FROM balance_history
            ORDER BY recorded_at DESC LIMIT 1
        """)
        balance_row = cur.fetchone()

        # Открытые позиции
        cur.execute("""
            SELECT id, symbol, side, entry_price, entry_qty, entry_time,
                   entry_score, exit_price, exit_qty, exit_time, pnl, pnl_percent,
                   pos_meta, ml_pro, adv_score, mtf_score, rvb_score, liq_score,
                   vv_score, vsa_score
            FROM trades
            WHERE status = 'open'
            ORDER BY entry_time DESC
        """)
        positions = cur.fetchall()

        # Последние закрытые
        cur.execute("""
            SELECT id, symbol, side, entry_price, entry_qty, entry_time,
                   exit_price, exit_qty, exit_time, pnl, pnl_percent, exit_reason,
                   pos_meta
            FROM trades
            WHERE status = 'closed'
            ORDER BY exit_time DESC LIMIT 20
        """)
        closed = cur.fetchall()

        # Подсчёт статистики
        total_pnl = sum(float(p['pnl'] or 0) for p in closed)
        win_count = sum(1 for p in closed if float(p['pnl'] or 0) > 0)
        loss_count = sum(1 for p in closed if float(p['pnl'] or 0) < 0)

        # Последние сигналы
        cur.execute("""
            SELECT symbol, decision, score, threshold, components, created_at
            FROM signal_log
            ORDER BY created_at DESC LIMIT 30
        """)
        signals = cur.fetchall()

        return {
            "balance": {
                "total": float(balance_row['balance']) if balance_row else 0,
                "free": float(balance_row['free']) if balance_row else 0,
                "equity": float(balance_row['equity'] or 0) if balance_row else 0,
                "updated": str(balance_row['recorded_at']) if balance_row else None,
            } if balance_row else {"total": 0, "free": 0, "equity": 0},
            "positions": [_fmt_pos(p) for p in positions],
            "closed": [_fmt_closed(p) for p in closed],
            "signals": [_fmt_signal(s) for s in signals],
            "stats": {
                "total_pnl": round(total_pnl, 2),
                "win_count": win_count,
                "loss_count": loss_count,
                "win_rate": round(win_count / (win_count + loss_count) * 100, 1) if (win_count + loss_count) > 0 else 0,
                "open_positions": len(positions),
            },
        }
    finally:
        cur.close()
        conn.close()


@app.get("/api/positions")
def api_positions():
    """Только открытые позиции — для обновления на фронте."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT id, symbol, side, entry_price, entry_qty, entry_time,
                   entry_score, exit_price, exit_qty, exit_time, pnl, pnl_percent,
                   pos_meta, ml_pro, adv_score, mtf_score, rvb_score, liq_score,
                   vv_score, vsa_score
            FROM trades
            WHERE status = 'open'
            ORDER BY entry_time DESC
        """)
        return {"positions": [_fmt_pos(p) for p in cur.fetchall()]}
    finally:
        cur.close()
        conn.close()


@app.get("/api/balance_history")
def api_balance_history(hours: int = Query(24, description="За сколько часов")):
    """Equity-кривая за последние N часов."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT balance, free, equity, recorded_at
            FROM balance_history
            WHERE recorded_at > now() - interval '%s hours'
            ORDER BY recorded_at ASC
        """, (hours,))
        return {"points": [
            {
                "t": str(row['recorded_at']),
                "balance": float(row['balance']),
                "free": float(row['free']),
                "equity": float(row['equity'] or row['balance']),
            }
            for row in cur.fetchall()
        ]}
    finally:
        cur.close()
        conn.close()


@app.get("/api/signals")
def api_signals(limit: int = Query(30, ge=1, le=200)):
    """Последние сигналы/голосования."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT symbol, decision, score, threshold, components, created_at
            FROM signal_log
            ORDER BY created_at DESC LIMIT %s
        """, (limit,))
        return {"signals": [_fmt_signal(s) for s in cur.fetchall()]}
    finally:
        cur.close()
        conn.close()


@app.get("/api/control/{action}")
def api_control(action: str):
    """Управление: pause / resume / exit_all."""
    # Пока заглушка — управление трейдером через файл состояния
    state_file = os.path.join(PROJECT_DIR, "data", "terminal_control.json")
    try:
        with open(state_file) as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    state["action"] = action
    state["updated"] = datetime.now(timezone.utc).isoformat()
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
    return {"status": "ok", "action": action}


# ── Frontend ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html")) as f:
        return HTMLResponse(f.read())


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt_pos(p):
    return {
        "id": p['id'],
        "symbol": p['symbol'],
        "side": p['side'],
        "entry_price": float(p['entry_price']),
        "entry_qty": float(p['entry_qty']),
        "entry_time": str(p['entry_time']),
        "entry_score": float(p['entry_score'] or 0),
        "pos_meta": p['pos_meta'],
        "pnl": 0.0,  # будет обновляться на фронте по текущей цене
        "pnl_percent": 0.0,
        "scores": {
            "ml_pro": round(float(p['ml_pro'] or 0), 1),
            "adv": round(float(p['adv_score'] or 0), 1),
            "mtf": round(float(p['mtf_score'] or 0), 1),
            "rvb": round(float(p['rvb_score'] or 0), 1),
            "liq": round(float(p['liq_score'] or 0), 1),
            "vv": round(float(p['vv_score'] or 0), 1),
            "vsa": round(float(p['vsa_score'] or 0), 1),
        },
    }


def _fmt_closed(p):
    pnl = float(p['pnl'] or 0)
    pnl_pct = float(p['pnl_percent'] or 0)
    return {
        "id": p['id'],
        "symbol": p['symbol'],
        "side": p['side'],
        "entry_price": float(p['entry_price']),
        "entry_qty": float(p['entry_qty']),
        "entry_time": str(p['entry_time']),
        "exit_price": float(p['exit_price'] or 0),
        "exit_time": str(p['exit_time'] or ''),
        "pnl": round(pnl, 2),
        "pnl_percent": round(pnl_pct, 2),
        "exit_reason": p['exit_reason'] or '—',
    }


def _fmt_signal(s):
    comp = s['components']
    # components — jsonb с вложенными скорами
    if isinstance(comp, str):
        try:
            comp = json.loads(comp)
        except json.JSONDecodeError:
            comp = {}
    return {
        "symbol": s['symbol'],
        "decision": s['decision'],
        "score": round(float(s['score'] or 0), 1),
        "threshold": round(float(s['threshold'] or 0), 1),
        "components": comp,
        "time": str(s['created_at']),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    log.info(f"📊 Терминал стартует на порту {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
