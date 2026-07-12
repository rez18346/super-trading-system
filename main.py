#!/usr/bin/env python3
"""
main.py — FastAPI-сервер (дашборд) для супер-трейдера.
Работает в отдельном screen. Трейдер — в screen 'trader'.
Оба используют одну БД.
"""

import os, sys, json, logging, ccxt
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "terminal", "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("api")

def _get_exchange():
    """Bybit exchange из .env."""
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

# ── Database ───────────────────────────────────────────────────────────────────
import db_pg

def _get_db():
    import psycopg2, psycopg2.extras
    return psycopg2.connect(db_pg.PG_DSN)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Super Trader API", version="2.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/api/status")
def api_status():
    """Общий статус системы."""
    try:
        conn = _get_db()
        cur = conn.cursor(cursor_factory=__import__('psycopg2').extras.RealDictCursor)
        
        # Баланс — напрямую с биржи (единый источник правды)
        balance = {'total': 0, 'free': 0}
        try:
            exchange = _get_exchange()
            exchange.timeout = 10000  # 10s макс
            bal = exchange.fetch_balance()
            usdt_total = bal['total'].get('USDT', 0)
            usdt_free = bal['free'].get('USDT', 0)
            balance = {'total': round(usdt_total, 2), 'free': round(usdt_free, 2)}
        except Exception as e:
            log.warning(f"Не удалось получить баланс с биржи: {e}")
        
        # Открытые позиции
        cur.execute("SELECT symbol, side, entry_price, entry_qty, entry_time FROM trades WHERE status = 'open'")
        positions = []
        for r in cur.fetchall():
            positions.append({
                'symbol': r['symbol'], 'side': r['side'],
                'entry_price': float(r['entry_price']), 'entry_qty': float(r['entry_qty']),
                'entry_time': str(r['entry_time']),
            })
        
        # Закрытые сделки (последние 50)
        cur.execute("""
            SELECT id, symbol, side, entry_price, entry_qty, entry_time,
                   exit_price, exit_time, pnl, pnl_percent, exit_reason
            FROM trades WHERE status = 'closed' AND exit_time IS NOT NULL
            ORDER BY exit_time DESC LIMIT 50
        """)
        closed = []
        for r in cur.fetchall():
            d = dict(r)
            d['entry_price'] = float(d['entry_price'] or 0)
            d['exit_price'] = float(d['exit_price'] or 0)
            d['pnl'] = float(d['pnl'] or 0)
            d['pnl_percent'] = float(d['pnl_percent'] or 0)
            d['entry_qty'] = float(d['entry_qty'] or 0)
            d['entry_time'] = str(d['entry_time'])[:19]
            d['exit_time'] = str(d['exit_time'])[:19] if d['exit_time'] else ''
            d['scores'] = {}  # пустые голоса — трейдер не пишет их в БД
            closed.append(d)
        
        cur.close()
        conn.close()
        
        # BTC regime из meta (отдельное подключение, чтобы не зависеть от других курсоров)
        btc_info = {}
        try:
            conn2 = __import__('psycopg2').connect(db_pg.PG_DSN)
            cur2 = conn2.cursor()
            cur2.execute("SELECT value FROM meta WHERE key = 'btc_analysis'")
            row_b = cur2.fetchone()
            if row_b:
                btc_info = json.loads(row_b[0])
            cur2.close()
            conn2.close()
        except Exception as e:
            log.warning(f"Ошибка BTC: {e}")
        
        # Трейдер — проверяем PID
        trader_pid = None
        try:
            import subprocess
            out = subprocess.check_output(['pgrep', '-f', 'super_trader.py']).decode().strip()
            trader_pid = out.split('\n')[0]
        except:
            pass
        
        return {
            'status': 'ok',
            'trader_pid': trader_pid,
            'trader_alive': trader_pid is not None,
            'balance': balance,
            'positions': positions,
            'positions_count': len(positions),
            'closed': closed,
            'closed_count': len(closed),
            'equity': [],  # график equity — пока пусто (нужна отдельная таблица)
            'btc': btc_info,
        }
    except Exception as e:
        log.error(f"Ошибка: {e}")
        return {'status': 'error', 'error': str(e)}

@app.get("/api/btc")
def api_btc():
    try:
        conn = _get_db()
        cur = conn.cursor()
        cur.execute("SELECT value FROM meta WHERE key = 'btc_analysis'")
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return json.loads(row[0])
        return {'regime': 'unknown', 'direction': 'neutral', 'structure': 'unknown'}
    except Exception as e:
        return {'error': str(e)}

@app.get("/api/analytics")
def api_analytics():
    try:
        conn = _get_db()
        cur = conn.cursor(cursor_factory=__import__('psycopg2').extras.RealDictCursor)
        
        cur.execute("""
            SELECT COUNT(*) FILTER (WHERE status = 'closed') as total_closed,
                   COUNT(*) FILTER (WHERE status = 'closed' AND COALESCE(pnl,0) > 0) as wins,
                   COUNT(*) FILTER (WHERE status = 'closed' AND COALESCE(pnl,0) < 0) as losses,
                   COALESCE(SUM(COALESCE(pnl,0)) FILTER (WHERE status = 'closed'), 0) as total_pnl
            FROM trades WHERE entry_time >= CURRENT_DATE
        """)
        today = dict(cur.fetchone())
        today['total_pnl'] = round(float(today['total_pnl']), 2)
        
        cur.execute("""
            SELECT DATE(entry_time) as day, COUNT(*) as trades,
                   SUM(COALESCE(pnl,0)) FILTER (WHERE status = 'closed') as day_pnl,
                   COUNT(*) FILTER (WHERE status = 'closed' AND COALESCE(pnl,0) > 0) as wins
            FROM trades WHERE entry_time >= CURRENT_DATE - INTERVAL '14 days'
            GROUP BY DATE(entry_time) ORDER BY day DESC
        """)
        daily = []
        for r in cur.fetchall():
            d = dict(r)
            d['day'] = str(d['day'])
            d['day_pnl'] = round(float(d['day_pnl'] or 0), 2)
            daily.append(d)
        
        cur.close()
        conn.close()
        return {'today': today, 'daily': daily}
    except Exception as e:
        return {'error': str(e)}

@app.get("/api/signals/latest")
def api_signals_latest():
    """Алиас для /api/signals."""
    return api_signals()

@app.get("/api/signals")
def api_signals():
    """Голосование модулей (из /tmp/system_v4.log)."""
    try:
        from vote_parser import parse_votes
        votes = parse_votes()
        if not votes:
            return {'status': 'idle', 'message': 'Система в режиме ожидания — сигналы не генерируются (BUY заблокированы BTC regime)', 'signals': {}}
        filtered = {}
        for sym, data in votes.items():
            if data.get('score', 0) >= 0:
                filtered[sym] = data
        sorted_votes = dict(sorted(filtered.items(), key=lambda x: x[1].get('score', 0), reverse=True))
        return {'status': 'active', 'signals': sorted_votes}
    except Exception as e:
        log.warning(f"Ошибка /api/signals: {e}")
        return {'status': 'error', 'message': str(e), 'signals': {}}

@app.get("/api/veto")
def api_veto():
    """VETO-сигналы (из БД)."""
    try:
        conn = _get_db()
        cur = conn.cursor(cursor_factory=__import__('psycopg2').extras.RealDictCursor)
        cur.execute("""
            SELECT id, symbol, side, entry_price, 
                   current_price, veto_reason, score
            FROM veto_entries
            ORDER BY created_at DESC LIMIT 50
        """)
        veto_list = []
        for r in cur.fetchall():
            d = dict(r)
            d['entry_price'] = float(d.get('entry_price') or 0)
            d['current_price'] = float(d.get('current_price') or 0)
            d['score'] = float(d.get('score') or 0)
            veto_list.append(d)
        cur.close()
        conn.close()
        return veto_list
    except Exception as e:
        log.warning(f"Ошибка /api/veto: {e}")
        return []

@app.get("/api/control/{action:str}")
def api_control(action: str):
    """Управление трейдером (pause/resume/exit_all)."""
    try:
        if action == 'pause':
            # Сигнал трейдеру через файл
            with open(os.path.join(DATA_DIR, 'pause.flag'), 'w') as f:
                f.write('1')
            return {'status': 'paused'}
        elif action == 'resume':
            pause_file = os.path.join(DATA_DIR, 'pause.flag')
            if os.path.exists(pause_file):
                os.remove(pause_file)
            return {'status': 'resumed'}
        elif action == 'exit_all':
            with open(os.path.join(DATA_DIR, 'exit_all.flag'), 'w') as f:
                f.write('1')
            return {'status': 'exit_all_triggered'}
        return {'error': f'Unknown action: {action}'}
    except Exception as e:
        log.error(f"Ошибка control/{action}: {e}")
        return {'error': str(e)}

@app.get("/api/liquidation_levels")
def api_liquidation_levels():
    """Уровни ликвидации (пусто, если нет позиций)."""
    return {}

@app.get("/")
def index():
    with open(os.path.join(STATIC_DIR, "index.html")) as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    log.info(f"🚀 FastAPI на порту 8765 (PID: {os.getpid()})")
    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
