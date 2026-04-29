#!/usr/bin/env python3
"""
control_api.py — FastAPI дашборд для супер-системы.

HTTP API + SSE (Server-Sent Events) для real-time обновлений.

Эндпоинты:
  GET  /                    — HTML dashboard (single page)
  GET  /api/status          — JSON snapshot состояния
  GET  /api/capital-history  — История капитала (JSON)
  SSE  /api/events          — Real-time поток (цены, позиции, логи)

При старте:
  1. Инициализирует таблицу capital_snapshots
  2. Запускает фоновый поток — снэпшоты капитала раз в 60 сек
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import json
import time
import asyncio
import logging
import threading
import sqlite3
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator
from pathlib import Path

import db
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

log = logging.getLogger('control_api')

# ─── КОНФИГ ──────────────────────────────────────────────────────────────────

SYSTEM_PID_FILE = os.path.join(BASE_DIR, "data", "trader.pid")
SYSTEM_LOG_FILE = "/tmp/system_v4.log"
CAPITAL_DB = db.get_db_path()  # используем ту же БД

app = FastAPI(title="Super System Dashboard", version="1.0.0")


# ─── ИНИЦИАЛИЗАЦИЯ ───────────────────────────────────────────────────────────

def _init_capital_table():
    """Создать таблицу capital_snapshots (безопасно, IF NOT EXISTS)."""
    conn = sqlite3.connect(CAPITAL_DB)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS capital_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total REAL NOT NULL,
                positions_value REAL NOT NULL DEFAULT 0,
                free_usdt REAL NOT NULL DEFAULT 0,
                positions_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cap_ts ON capital_snapshots(created_at)")
        conn.commit()
    finally:
        conn.close()


# Вызываем при импорте
_init_capital_table()


# ─── ФОНОВЫЙ СБОРЩИК КАПИТАЛА ───────────────────────────────────────────────

def _snapshot_worker():
    """Фоновый поток: каждые 60 сек снэпшот капитала."""
    while True:
        try:
            pid = None
            try:
                with open(SYSTEM_PID_FILE) as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)  # проверка живости
            except Exception:
                pid = None

            if pid:
                positions = db.get_all_positions()
                pos_value = sum(p['quantity'] * p['entry_price'] for p in positions.values())
                count = len(positions)

                conn = sqlite3.connect(CAPITAL_DB)
                try:
                    conn.execute(
                        "INSERT INTO capital_snapshots (total, positions_value, free_usdt, positions_count) "
                        "VALUES (?, ?, ?, ?)",
                        (round(pos_value + 270, 2), round(pos_value, 2), 270, count)
                    )
                    conn.commit()
                finally:
                    conn.close()
        except Exception:
            pass
        time.sleep(60)


# Запускаем в daemon-потоке
_snapshot_thread = threading.Thread(target=_snapshot_worker, daemon=True, name='cap-snapshot')
_snapshot_thread.start()


# ─── ВСПОМОГАТЕЛЬНЫЕ ─────────────────────────────────────────────────────────

def read_log_tail(n: int = 50) -> list:
    try:
        with open(SYSTEM_LOG_FILE, 'r') as f:
            lines = f.readlines()
        return lines[-n:]
    except Exception:
        return ["Нет лог-файла"]


def read_log_since(timestamp: float) -> list:
    try:
        mtime = os.path.getmtime(SYSTEM_LOG_FILE)
        if mtime < timestamp:
            return []
        with open(SYSTEM_LOG_FILE, 'r') as f:
            lines = f.readlines()
        result = []
        for line in reversed(lines):
            try:
                parts = line.split(' - ', 1)
                if len(parts) < 2:
                    continue
                dt = datetime.strptime(parts[0].split(',')[0], '%Y-%m-%d %H:%M:%S')
                if dt.timestamp() > timestamp:
                    result.insert(0, line.rstrip())
            except (ValueError, IndexError):
                continue
            if len(result) >= 200:
                break
        return result
    except Exception:
        return []


def get_pid() -> Optional[int]:
    try:
        with open(SYSTEM_PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return pid
    except Exception:
        return None


def get_status_snapshot() -> dict:
    pid = get_pid()
    running = pid is not None

    try:
        positions = db.get_all_positions()
    except Exception:
        positions = {}

    try:
        pnl = db.get_pnl_stats()
    except Exception:
        pnl = {"total_pnl": 0, "total_trades": 0, "win_rate": 0}

    total_in_positions = sum(p['quantity'] * p['entry_price'] for p in positions.values())
    balance_info = {
        'in_positions': round(total_in_positions, 2),
        'positions_count': len(positions),
    }

    trades_history = []
    try:
        conn = sqlite3.connect(CAPITAL_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            'SELECT * FROM trades ORDER BY id DESC LIMIT 50'
        ).fetchall()
        trades_history = [
            {'id': r['id'], 'symbol': r['symbol'], 'side': r['side'],
             'price': r['price'], 'qty': r['quantity'], 'value': r['value'],
             'pnl': r['pnl'], 'pnl_pct': r['pnl_pct'], 'ts': r['timestamp']}
            for r in rows
        ]
        conn.close()
    except Exception:
        pass

    return {
        'running': running,
        'pid': pid,
        'timestamp': time.time(),
        'positions': {
            sym: {'qty': pos['quantity'], 'entry': pos['entry_price'],
                  'entry_time': pos.get('entry_time', '')}
            for sym, pos in positions.items()
        },
        'pnl': pnl,
        'balance': balance_info,
        'trades': trades_history,
        'log_tail': read_log_tail(5),
    }


# ─── ЭНДПОИНТЫ ──────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    return get_status_snapshot()


@app.get("/api/capital-history")
async def api_capital_history():
    """Последние 500 точек для графика капитала."""
    try:
        conn = sqlite3.connect(CAPITAL_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT total, created_at FROM capital_snapshots "
            "ORDER BY id ASC LIMIT 500"
        ).fetchall()
        conn.close()
        return {'points': [{'t': r['created_at'], 'v': r['total']} for r in rows]}
    except Exception as e:
        return {'points': [], 'error': str(e)}


@app.get("/api/events")
async def api_events(request: Request):
    async def event_generator() -> AsyncGenerator[str, None]:
        last_ts = time.time() - 5
        while True:
            if await request.is_disconnected():
                break
            try:
                new_logs = read_log_since(last_ts)
                if new_logs:
                    yield f"data: {json.dumps({'type': 'logs', 'lines': new_logs, 'ts': time.time()})}\n\n"
                    last_ts = time.time()
                status = get_status_snapshot()
                yield f"data: {json.dumps({'type': 'status', 'data': status, 'ts': time.time()})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'msg': str(e)})}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(r"""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>🏭 Супер-Система</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0d1117;color:#c9d1d9;min-height:100vh}
.container{max-width:1200px;margin:0 auto;padding:20px}
h1{font-size:24px;margin-bottom:12px;color:#58a6ff}
.sub{font-size:13px;color:#8b949e;margin-bottom:20px}
.status-bar{display:flex;gap:20px;margin-bottom:24px;flex-wrap:wrap}
.status-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;flex:1;min-width:160px}
.status-card h3{font-size:11px;text-transform:uppercase;color:#8b949e;margin-bottom:8px}
.status-card .value{font-size:26px;font-weight:700}
.green{color:#3fb950}.red{color:#f85149}.yellow{color:#d29922}
.section{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin-bottom:16px}
.section h2{font-size:15px;margin-bottom:12px;color:#58a6ff}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:768px){.grid{grid-template-columns:1fr}}
table{width:100%;border-collapse:collapse}
th{text-align:left;font-size:11px;text-transform:uppercase;color:#8b949e;padding:6px 10px;border-bottom:1px solid #30363d}
td{padding:6px 10px;border-bottom:1px solid #21262d;font-size:13px}
tr:hover td{background:#1c2128}
canvas{width:100%!important;height:200px!important;background:#0d1117;border-radius:4px}
.logs{max-height:300px;overflow-y:auto;font-family:'SF Mono','Consolas',monospace;font-size:11px;line-height:1.5}
.log-line{padding:1px 6px;border-bottom:1px solid #21262d;white-space:nowrap}
.log-line:nth-child(odd){background:#161b22}
.lvl-error{color:#f85149}.lvl-warn{color:#d29922}.lvl-info{color:#58a6ff}
.badge{display:inline-block;padding:1px 6px;border-radius:8px;font-size:10px;font-weight:600}
.bg-green{background:#1b3a1d;color:#3fb950}.bg-red{background:#3d1214;color:#f85149}
</style></head><body>
<div class="container">
<h1>🏭 Супер-Система V5</h1>
<div class="sub" id="subtitle">загрузка...</div>

<div class="status-bar" id="statusBar">
  <div class="status-card"><h3>Статус</h3><div class="value" id="stRunning">❓</div></div>
  <div class="status-card"><h3>Позиции</h3><div class="value" id="stPositions">0/5</div></div>
  <div class="status-card"><h3>Капитал</h3><div class="value" id="stCapital">$0</div></div>
  <div class="status-card"><h3>Сделок</h3><div class="value" id="stTrades">0</div></div>
</div>

<div class="grid">
  <div class="section">
    <h2>📊 Позиции</h2>
    <table><thead><tr><th>Символ</th><th>Кол-во</th><th>Вход</th><th>Стоимость</th></tr></thead>
    <tbody id="tbPos"><tr><td colspan="4" style="text-align:center;color:#8b949e;font-size:13px;padding:20px">Нет позиций</td></tr></tbody></table>
  </div>
  <div class="section">
    <h2>📈 Капитал</h2>
    <canvas id="capChart"></canvas>
  </div>
</div>

<div class="section">
  <h2>📜 История сделок</h2>
  <table><thead><tr><th>Символ</th><th>Тип</th><th>Цена</th><th>Кол-во</th><th>PnL</th><th>Время</th></tr></thead>
  <tbody id="tbTrades"><tr><td colspan="6" style="text-align:center;color:#8b949e;font-size:13px;padding:20px">Нет сделок</td></tr></tbody></table>
</div>

<div class="section">
  <h2>📋 Логи</h2>
  <div class="logs" id="logContainer"><div class="log-line" style="color:#8b949e">Ожидание...</div></div>
</div>
</div>

<script>
const EMO = {BTC:'💰',ETH:'🔷',SOL:'🌞',DOGE:'🐕',XRP:'❌',ADA:'📊',AVAX:'🔺',DOT:'🎯',LINK:'🔗',APT:'🍑',ARB:'🔷',ATOM:'⚛️',ALGO:'🔮',EGLD:'🟢',ROSE:'🌹',MNT:'📐',FIL:'🎞️',NEAR:'🌙',SAND:'🏖️'};
const fmtP=p=>'$'+Number(p).toFixed(p<0.1?4:p<1?3:2);
const fmtQ=q=>Number(q).toFixed(q<.001?6:q<1?4:q<100?2:1);
const em=s=>EMO[s.split('/')[0]]||'📈';

// ─── КАНВАС ГРАФИК КАПИТАЛА ───
let capChart=null;
async function drawCapChart(){
  const r=await fetch('/api/capital-history'); const d=await r.json();
  if(!d.points||d.points.length<2) return;
  const vals=d.points.map(p=>p.v); const max=Math.max(...vals); const min=Math.min(...vals);
  const c=document.getElementById('capChart');
  if(!c) return;
  const ctx=c.getContext('2d'); const w=c.width=960; const h=c.height=200;
  ctx.clearRect(0,0,w,h); const pad=10; const pw=w-pad*2; const ph=h-pad*2; const rng=Math.max(max-min,1);
  ctx.beginPath(); ctx.strokeStyle='#3fb950'; ctx.lineWidth=2;
  for(let i=0;i<vals.length;i++){
    const x=pad+(i/(vals.length-1||1))*pw;
    const y=pad+ph-((vals[i]-min)/rng)*ph;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  } ctx.stroke();
  ctx.fillStyle='#8b949e'; ctx.font='10px sans-serif';
  ctx.fillText('$'+max.toFixed(0),pad,pad+10);
  ctx.fillText('$'+min.toFixed(0),pad,h-5);
}

// ─── ОБНОВЛЕНИЯ ───
function updPos(pos){
  const e=Object.entries(pos);
  document.getElementById('tbPos').innerHTML=e.length
    ?e.map(([s,p])=>'<tr><td>'+em(s)+' '+s+'</td><td>'+fmtQ(p.qty)+'</td><td>'+fmtP(p.entry)+'</td><td>'+fmtP(p.qty*p.entry)+'</td></tr>').join('')
    :'<tr><td colspan="4" style="text-align:center;color:#8b949e;font-size:13px;padding:20px">Нет позиций</td></tr>';
}

function updTrades(tr){
  if(!tr||!tr.length){document.getElementById('tbTrades').innerHTML='<tr><td colspan="6" style="text-align:center;color:#8b949e;font-size:13px;padding:20px">Нет сделок</td></tr>';return;}
  document.getElementById('tbTrades').innerHTML=tr.map(t=>{
    const pnlColor=t.pnl>0?'#3fb950':t.pnl<0?'#f85149':'#8b949e';
    return '<tr><td>'+em(t.symbol)+' '+t.symbol+'</td><td>'+(t.side==='buy'?'🟢 BUY':'🔴 SELL')+'</td><td>'+fmtP(t.price)+'</td><td>'+fmtQ(t.qty)+'</td><td style="color:'+pnlColor+'">'+(t.pnl?fmtP(t.pnl):'—')+'</td><td style="font-size:11px;color:#8b949e">'+(t.ts||'').split('T')[0]+'</td></tr>';
  }).join('');
}

function updBar(d){
  const r=d.running;
  document.getElementById('stRunning').innerHTML=r?'🟢 В работе':'🔴 Остановлена';
  document.getElementById('stRunning').className='value '+(r?'green':'red');
  document.getElementById('stPositions').textContent=Object.keys(d.positions||{}).length+'/5';
  document.getElementById('stCapital').innerHTML=(d.balance?fmtP(d.balance.in_positions+270):'$0')+' <span style="font-size:12px;color:#8b949e">(всего)</span>';
  document.getElementById('stTrades').textContent=d.pnl?.total_trades||0;
  document.getElementById('subtitle').textContent='PID: '+(d.pid||'—')+' | обновлено '+new Date().toLocaleTimeString();
  updPos(d.positions||{});
  if(d.trades) updTrades(d.trades);
}

// ─── SSE ───
const logContainer=document.getElementById('logContainer');
let logBuf=[];
function addLogs(ll){for(const l of ll)logBuf.push(l);if(logBuf.length>200)logBuf=logBuf.slice(-200);logContainer.innerHTML=logBuf.map(l=>{let c='log-line';if(l.includes('ERROR'))c+=' lvl-error';else if(l.includes('WARNING'))c+=' lvl-warn';return '<div class="'+c+'">'+l.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')+'</div>';}).join('');logContainer.scrollTop=logContainer.scrollHeight;}

const es=new EventSource('/api/events');
es.onmessage=function(e){try{const m=JSON.parse(e.data);if(m.type==='status'&&m.data)updBar(m.data);if(m.type==='logs'&&m.lines)addLogs(m.lines);}catch(err){}}

// ─── СТАРТ ───
fetch('/api/status').then(r=>r.json()).then(d=>{updBar(d);drawCapChart()});
setInterval(drawCapChart,30000);
</script>
</body></html>""")


def run_server(host: str = "0.0.0.0", port: int = 8765):
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_server()
