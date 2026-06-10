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


@app.get("/api/signals/latest")
def api_signals_latest():
    """Последний сигнал по каждой монете, сортировка по score (убывание)."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT DISTINCT ON (symbol) symbol, decision, score, threshold, components, created_at
            FROM signal_log
            ORDER BY symbol, created_at DESC
        """)
        signals = [_fmt_signal(s) for s in cur.fetchall()]
        signals.sort(key=lambda x: -x['score'])
        return {"signals": signals}
    finally:
        cur.close()
        conn.close()


@app.get("/api/veto")
def api_veto():
    """Монеты в VETO — score близкий к threshold, но заблокированные."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # Берём последний сигнал по каждой монете
        cur.execute("""
            SELECT DISTINCT ON (symbol) symbol, decision, score, threshold, components, created_at
            FROM signal_log
            ORDER BY symbol, created_at DESC
        """)
        all_signals = cur.fetchall()
        
        veto_list = []
        for s in all_signals:
            score = float(s['score'] or 0)
            threshold = float(s['threshold'] or 60)
            decision = s['decision'].lower() if s['decision'] else 'hold'
            
            # Кандидаты — hold, держатся рядом с threshold
            if decision == 'hold' and threshold > 0 and score >= threshold * 0.7:
                veto_list.append({
                    'symbol': s['symbol'],
                    'score': round(score, 1),
                    'threshold': round(threshold, 1),
                    'gap': round(threshold - score, 1),
                    'gap_pct': round((threshold - score) / threshold * 100, 1),
                    'time': str(s['created_at']),
                })
        
        # Сортируем — кто ближе всего к порогу
        veto_list.sort(key=lambda x: x['gap'])
        
        # Следом идём: монеты с decision=enter или buy — уже почти прошли
        enter_list = []
        for s in all_signals:
            decision = s['decision'].lower() if s['decision'] else 'hold'
            score = float(s['score'] or 0)
            threshold = float(s['threshold'] or 60)
            if decision == 'enter' or decision == 'buy':
                enter_list.append({
                    'symbol': s['symbol'],
                    'score': round(score, 1),
                    'threshold': round(threshold, 1),
                    'decision': decision,
                    'time': str(s['created_at']),
                })
        
        return {
            'veto': veto_list,
            'candidates': enter_list,
            'total_watch': len(veto_list),
        }
    finally:
        cur.close()
        conn.close()


@app.get("/api/liquidation_levels")
def api_liquidation_levels():
    """Уровни ликвидности по открытым позициям — расстояние до liq."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cur.execute("""
            SELECT id, symbol, side, entry_price, entry_qty, pos_meta,
                   ml_pro, adv_score, mtf_score, rvb_score, liq_score,
                   vv_score, vsa_score
            FROM trades
            WHERE status = 'open'
            ORDER BY entry_time DESC
        """)
        positions = cur.fetchall()
        
        result = []
        for p in positions:
            meta = p['pos_meta']
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except:
                    meta = {}
            
            liq_price = meta.get('liquidation')
            last_price = meta.get('last_price') or p['entry_price']
            entry_price = float(p['entry_price'])
            side = p['side'].lower()
            
            liq_dist = None
            liq_dist_pct = None
            if liq_price and float(liq_price) > 0:
                liq_price = float(liq_price)
                if side == 'long':
                    liq_dist = last_price - liq_price
                    liq_dist_pct = (last_price - liq_price) / liq_price * 100 if liq_price > 0 else None
                else:
                    liq_dist = liq_price - last_price
                    liq_dist_pct = (liq_price - last_price) / liq_price * 100 if liq_price > 0 else None
            
            sl_price = meta.get('stopLoss')
            tp_price = meta.get('takeProfit')
            
            # Сумма в позиции
            position_value = entry_price * float(p['entry_qty'])
            
            result.append({
                'id': p['id'],
                'symbol': p['symbol'],
                'side': p['side'],
                'entry_price': round(entry_price, 8),
                'position_value': round(position_value, 2),
                'last_price': round(float(last_price), 8),
                'liquidation': round(liq_price, 8) if liq_price else None,
                'liq_distance': round(liq_dist, 8) if liq_dist is not None else None,
                'liq_distance_pct': round(liq_dist_pct, 2) if liq_dist_pct is not None else None,
                'stop_loss': round(float(sl_price), 8) if sl_price else None,
                'take_profit': round(float(tp_price), 8) if tp_price else None,
                'liq_score': round(float(p['liq_score'] or 0), 1),
            })
        
        return {'levels': result}
    finally:
        cur.close()
        conn.close()


@app.get("/api/control/{action}")
@app.get("/api/analytics")
def api_analytics():
    """Аналитика работы системы: PnL по дням, сделки с голосованием."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # Статистика за сегодня
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE status = 'closed') as total_closed,
                COUNT(*) FILTER (WHERE status = 'closed' AND pnl > 0) as wins,
                COUNT(*) FILTER (WHERE status = 'closed' AND pnl < 0) as losses,
                COALESCE(SUM(pnl) FILTER (WHERE status = 'closed'), 0) as total_pnl,
                COALESCE(AVG(pnl_percent) FILTER (WHERE status = 'closed'), 0) as avg_pnl_pct
            FROM trades
            WHERE entry_time >= CURRENT_DATE
        """)
        today_stats = dict(cur.fetchone())
        
        # Статистика за всё время
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE status = 'closed') as total_closed,
                COUNT(*) FILTER (WHERE status = 'closed' AND pnl > 0) as wins,
                COUNT(*) FILTER (WHERE status = 'closed' AND pnl < 0) as losses,
                COALESCE(SUM(pnl) FILTER (WHERE status = 'closed'), 0) as total_pnl
            FROM trades
        """)
        all_stats = dict(cur.fetchone())
        
        # PnL по дням (последние 14 дней)
        cur.execute("""
            SELECT 
                DATE(entry_time) as day,
                COUNT(*) as trades,
                SUM(pnl) FILTER (WHERE status = 'closed') as day_pnl,
                COUNT(*) FILTER (WHERE status = 'closed' AND pnl > 0) as wins
            FROM trades
            WHERE entry_time >= CURRENT_DATE - INTERVAL '14 days'
            GROUP BY DATE(entry_time)
            ORDER BY day DESC
        """)
        daily = []
        for r in cur.fetchall():
            d = dict(r)
            d['day'] = str(d['day'])
            d['day_pnl'] = round(float(d['day_pnl'] or 0), 2)
            daily.append(d)
        
        # Закрытые сделки (последние 50) с голосованием модулей
        cur.execute("""
            SELECT id, symbol, side, entry_price, entry_qty, entry_time,
                   exit_price, exit_time, pnl, pnl_percent, exit_reason,
                   entry_score, ml_pro, adv_score, mtf_score, rvb_score,
                   liq_score, vv_score, vsa_score, pos_meta
            FROM trades
            WHERE status = 'closed'
            ORDER BY exit_time DESC NULLS LAST
            LIMIT 50
        """)
        closed = []
        for r in cur.fetchall():
            c = _fmt_closed(r)
            c['scores'] = {
                'ml_pro': float(r['ml_pro'] or 0),
                'adv_score': float(r['adv_score'] or 0),
                'mtf_score': float(r['mtf_score'] or 0),
                'rvb_score': float(r['rvb_score'] or 0),
                'liq_score': float(r['liq_score'] or 0),
                'vv_score': float(r['vv_score'] or 0),
                'vsa_score': float(r['vsa_score'] or 0),
                'entry_score': float(r['entry_score'] or 0),
            }
            closed.append(c)
        
        return {
            'today': today_stats,
            'all_time': all_stats,
            'daily': daily,
            'closed': closed,
        }
    finally:
        cur.close()
        conn.close()


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
