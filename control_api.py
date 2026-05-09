#!/usr/bin/env python3
"""
control_api.py — FastAPI дашборд для супер-системы.

HTTP API + SSE (Server-Sent Events) для real-time обновлений.

Эндпоинты:
  GET  /                    — HTML dashboard (single page)
  GET  /api/status          — JSON snapshot состояния
  GET  /api/capital-history  — История капитала (JSON)
  GET  /api/changelog        — История изменений системы (JSON)
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


def _parse_votes_from_log() -> dict:
    """Парсит последние голоса DecisionEngine из лога для каждой пары.
    Читает весь лог с конца, собирает последнее состояние каждой пары.
    Возвращает {symbol: {...price, score, votes...}}
    """
    import re
    # Список всех отслеживаемых пар
    ALL_SYMBOLS = ['BTC','ETH','SOL','XRP','ADA','AVAX','DOT','DOGE','LINK','LTC','BCH',
                   'ATOM','ALGO','EGLD','APT','ARB','ROSE','MNT','FIL','NEAR','OP','SAND']
    result = {}
    prices = {}
    # Сколько строк прочитали — лимит на проход
    MAX_LINES = 20000
    try:
        with open(SYSTEM_LOG_FILE, 'r') as f:
            # Читаем с конца: taker либо tail, либо mmap
            # Для большого лога читаем не более MAX_LINES с конца
            f.seek(0, 2)  # в конец
            fsize = f.tell()
            # читаем блок с конца
            chunk_size = min(fsize, 512 * 1024)  # макс 512KB
            f.seek(max(0, fsize - chunk_size))
            # если сдвинулись не на начало — пропускаем неполную строку
            if f.tell() > 0:
                f.readline()
            lines = f.readlines()
            # Если в блоке не всё — расширяем
            if len(lines) < MAX_LINES and fsize > chunk_size:
                # Добираем ещё блок
                chunk_size *= 2
                f.seek(max(0, fsize - chunk_size))
                if f.tell() > 0:
                    f.readline()
                lines = f.readlines()
    except (FileNotFoundError, IOError):
        return result

    # Сначала цены — они в каждой строке
    for line in reversed(lines):
        pm = re.search(r'(\w+)/USDT:? Цена=\$([\d.]+)', line)
        if pm:
            sym = pm.group(1)
            if sym not in prices:
                prices[sym] = float(pm.group(2))

    # Собираем последнее состояние каждой пары
    for line in reversed(lines):
        m = re.search(r'\[DE→(HOLD|BUY|SELL)\] (\w+)/USDT:.*Score=(\d+).*ML-Pro:(\d+)\(([^)]+)\).*Adv:(\d+)\(([^)]+)\).*MTF:(\d+).*RVB:(\d+).*Liq:(\d+)\([^)]*\)[^V]*VV:(\d+)\(', line)
        if m:
            sym = m.group(2)
            # Всегда перезаписываем — лог может содержать старые строки без VV
            # Ищем bonus/rev/btc в конце строки
            bonus_match = re.search(r'bonus=([-\d]+) rev=([-\d]+) btc=([-+]\d+)', line)
            result[sym] = {
                'score': int(m.group(3)),
                'signal': m.group(1),
                'mlpro': f"{m.group(4)}({m.group(5)})",
                'adv': f"{m.group(6)}({m.group(7)})",
                'mtf': int(m.group(8)),
                'rvb': int(m.group(9)),
                'liq': int(m.group(10)),
                    'vv': int(m.group(11)),
                    'bonus': int(bonus_match.group(1)) if bonus_match else 0,
                    'rev': int(bonus_match.group(2)) if bonus_match else 0,
                    'btc': int(bonus_match.group(3)) if bonus_match else 0,
                    'price': prices.get(sym, 0),
                }
        # VETO — только если не перезаписана основной строкой
        vm = re.search(r'\[DE→(HOLD|BUY|SELL)\] (\w+)/USDT:.*VETO: (.+)', line)
        if vm:
            sym = vm.group(2)
            if sym not in result or result[sym].get('score') == 0 or 'veto' not in result[sym]:
                result[sym] = {
                    'score': 0,
                    'signal': vm.group(1),
                    'veto': vm.group(3),
                    'price': prices.get(sym, 0),
                }
        # Если собрали все — выходим раньше
        if len(result) >= len(ALL_SYMBOLS):
            break

    return result


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
            'SELECT id, symbol, side, price, quantity, value, pnl, pnl_pct, timestamp, created_at FROM trade_history ORDER BY id DESC LIMIT 50'
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

    votes = _parse_votes_from_log()

    # Обогащаем позиции голосами
    enriched_positions = {}
    for sym, pos in positions.items():
        p = {'qty': pos['quantity'], 'entry': pos['entry_price'],
             'entry_time': pos.get('entry_time', '')}
        if sym in votes:
            p['votes'] = votes[sym]
        enriched_positions[sym] = p

    # Добавляем голоса для всех наблюдаемых пар (из лога, даже без позиций)
    for sym, v in votes.items():
        if sym not in enriched_positions:
            enriched_positions[sym] = {
                'qty': 0, 'entry': 0,
                'current_price': v.get('price', 0),
                'entry_time': '',
                'votes': v
            }

    return {
        'running': running,
        'pid': pid,
        'timestamp': time.time(),
        'positions': enriched_positions,
        'pnl': pnl,
        'balance': balance_info,
        'trades': trades_history,
        'log_tail': read_log_tail(5),
    }


# ─── ЭНДПОИНТЫ ──────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    return get_status_snapshot()


@app.get("/api/trade-history")
async def api_trade_history():
    """История всех сделок с PnL, отсортированная по времени."""
    try:
        conn = sqlite3.connect(CAPITAL_DB)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            'SELECT id, symbol, side, price, quantity, value, pnl, pnl_pct, timestamp, created_at FROM trade_history ORDER BY id DESC LIMIT 500'
        ).fetchall()
        conn.close()
        return {'trades': [
            {'id': r['id'], 'symbol': r['symbol'], 'side': r['side'],
             'price': r['price'], 'qty': r['quantity'], 'value': r['value'],
             'pnl': r['pnl'], 'pnl_pct': r['pnl_pct'], 'ts': r['timestamp']}
            for r in rows
        ]}
    except Exception as e:
        return {'trades': [], 'error': str(e)}


CHANGELOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'workspace', 'memory', '2026-05-01.md')
if not os.path.exists(CHANGELOG_FILE):
    CHANGELOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CHANGELOG.md')


@app.get("/api/changelog")
async def api_changelog():
    """Последние 50 строк истории изменений."""
    try:
        with open(CHANGELOG_FILE, 'r') as f:
            lines = f.readlines()
        # Последние 50 строк
        tail = lines[-50:] if len(lines) > 50 else lines
        return {
            'file': os.path.basename(CHANGELOG_FILE),
            'total_lines': len(lines),
            'lines': ''.join(tail)
        }
    except Exception as e:
        return {'file': None, 'lines': f'Файл изменений не найден: {e}', 'total_lines': 0}


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
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate"><meta http-equiv="Pragma" content="no-cache"><meta http-equiv="Expires" content="0">
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

<div class="section" style="grid-column:1/-1">
    <h2>📊 Мониторинг голосов <span style="font-size:12px;color:#8b949e;font-weight:normal">— вход при Score ≥ 65</span></h2>
    <!-- Легенда голосов -->
    <div style="margin-bottom:12px;padding:8px 10px;background:#1c2128;border-radius:6px;font-size:12px;display:flex;flex-wrap:wrap;gap:6px 16px">
      <span><span style="color:#3fb950;font-weight:700">🔵ML</span>=ML-Pro (20%)</span>
      <span><span style="color:#3fb950;font-weight:700">🟢Ad</span>=ML-Advisor (10%)</span>
      <span><span style="color:#3fb950;font-weight:700">🟡TF</span>=TimeFrame (25%)</span>
      <span><span style="color:#3fb950;font-weight:700">🟣RV</span>=RSI/Vol/BTC</span>
      <span><span style="color:#3fb950;font-weight:700">🟠LQ</span>=Liquidity (25%)</span>
      <span><span style="color:#3fb950;font-weight:700">🔴VV</span>=Volume/VWAP (20%)</span>
      <span><span style="color:#8b949e;font-size:11px">Цвет:</span> 🟢≥90 🟡≥75 ⚪&lt;75</span>
      <span><span style="color:#8b949e;font-size:11px">Пороги:</span> ✅≥65 🔶50–64 ⚪&lt;50 ⛔вето</span>
    </div>
    <table style="width:100%;font-size:11px"><thead><tr>
      <th style="width:18px"></th><th>Символ</th><th style="width:28px">Кол-во</th><th style="width:65px">Вход</th><th style="width:70px">Стоим.</th>
      <th style="text-align:left">Голоса (6 гол.)</th>
      <th style="width:70px">Score</th>
    </tr></thead>
    <tbody id="tbPos"><tr><td colspan="7" style="text-align:center;color:#8b949e;font-size:13px;padding:20px">Нет данных</td></tr></tbody></table>
  </div>

<div class="section">
  <h2>📜 История сделок <span style="font-size:12px;cursor:pointer;color:#58a6ff" onclick="toggleTradeLog()">[полная история]</span></h2>
  <table><thead><tr><th>Символ</th><th>Тип</th><th>Цена</th><th>Кол-во</th><th>PnL</th><th>Время</th></tr></thead>
  <tbody id="tbTrades"><tr><td colspan="6" style="text-align:center;color:#8b949e;font-size:13px;padding:20px">Нет сделок</td></tr></tbody></table>
</div>

<div id="tradeLogModal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:1000;overflow-y:auto;padding:20px">
  <div style="background:var(--bg-card);max-width:1000px;margin:20px auto;border-radius:12px;padding:20px;border:1px solid var(--border)">
    <div style="display:flex;justify-content:space-between;margin-bottom:15px">
      <h2 style="margin:0">📊 Полная история сделок</h2>
      <span style="cursor:pointer;font-size:24px" onclick="document.getElementById('tradeLogModal').style.display='none'">✕</span>
    </div>
    <div id="tradeLogSummary" style="margin-bottom:15px;color:#8b949e;font-size:14px"></div>
    <div style="max-height:600px;overflow-y:auto">
      <table style="width:100%;font-size:13px"><thead><tr><th>Символ</th><th>Тип</th><th>Цена</th><th>Кол-во</th><th>Стоимость</th><th>PnL</th><th>%</th><th>Время</th></tr></thead>
      <tbody id="tbTradeLog"><tr><td colspan="8" style="text-align:center;padding:30px;color:#8b949e">Загрузка...</td></tr></tbody></table>
    </div>
  </div>
</div>

<div class="section" id="changelogSection" style="display:none">
  <h2>📋 История изменений</h2>
  <div style="font-family:'SF Mono','Consolas',monospace;font-size:12px;line-height:1.6;max-height:400px;overflow-y:auto;white-space:pre-wrap" id="changelogContent">Загрузка...</div>
</div>

<div class="section">
  <h2>📋 Логи</h2>
  <div style="margin-bottom:8px;display:flex;gap:8px">
    <button onclick="toggleSection('changelogSection');loadChangelog()" style="background:#21262d;color:#58a6ff;border:1px solid #30363d;border-radius:6px;padding:4px 12px;cursor:pointer;font-size:12px">📋 Изменения</button>
  </div>
  <div class="logs" id="logContainer"><div class="log-line" style="color:#8b949e">Ожидание...</div></div>
</div>
</div>

<script>
const EMO = {BTC:'💰',ETH:'🔷',SOL:'🌞',DOGE:'🐕',XRP:'❌',ADA:'📊',AVAX:'🔺',DOT:'🎯',LINK:'🔗',APT:'🍑',ARB:'🔷',ATOM:'⚛️',ALGO:'🔮',EGLD:'🟢',ROSE:'🌹',MNT:'📐',FIL:'🎞️',NEAR:'🌙',SAND:'🏖️'};
const fmtP=p=>'$'+Number(p).toFixed(p<0.1?4:p<1?3:2);
const fmtQ=q=>Number(q).toFixed(q<.001?6:q<1?4:q<100?2:1);
const em=s=>EMO[s.split('/')[0]]||'📈';

// ─── ОБНОВЛЕНИЯ ───
function updPos(pos){
  const e=Object.entries(pos).sort(([,a],[,b])=>{
    const va=(b.votes||{}).score||0, vb=(a.votes||{}).score||0;
    return va-vb;
  });
  document.getElementById('tbPos').innerHTML=e.length
    ?e.map(([s,p])=>{
      const v=p.votes||{};
      if(p.qty==0&&!v.score) return '';
      // Score / бар
      const barW=Math.min((v.score||0)/65*100,100);
      const barColor=v.score>=65?'#3fb950':v.score>=50?'#d29922':'#f85149';
      // Эмодзи и сигнал
      const isVeto = v.veto;
      const sigEmoji = isVeto ? '⛔' : v.score>=65?'✅':v.score>=50?'🔶':'⚪';
      const signal = isVeto ? v.veto : (v.score||'?');
      // Строка голосов
      function vc(val, thr, thr2) {
        if(val===undefined||val===null) return '#555';
        return val>=thr?'#3fb950':val>=thr2?'#d29922':'#8b949e';
      }
      // Укорачиваем длинные строки голосов: "20(SKIP(prob=0.50))" → "20 SK"
      function shortVote(s) {
        if(!s) return '?';
        const parts = String(s).match(/^(\d+)\((\w+)/);
        if(parts) return parts[1]+' '+parts[2].substring(0,2);
        return String(s).substring(0,5);
      }
      const votesLine = v.mlpro
        ? '<span title="'+v.mlpro+'" style="cursor:help;color:'+vc(parseInt(v.mlpro),0,0)+'">🔵'+shortVote(v.mlpro)+'</span> '+
          '<span title="'+v.adv+'" style="cursor:help;color:'+vc(parseInt(v.adv),0,0)+'">🟢'+shortVote(v.adv)+'</span> '+
          '<span title="'+v.mtf+'" style="cursor:help;color:'+vc(v.mtf,90,75)+'">🟡'+v.mtf+'</span> '+
          '<span title="'+v.rvb+'" style="cursor:help;color:'+vc(v.rvb,90,75)+'">🟣'+v.rvb+'</span> '+
          '<span title="'+v.liq+'" style="cursor:help;color:'+vc(v.liq,90,75)+'">🟠'+v.liq+'</span> '+
          '<span title="'+v.vv+'" style="cursor:help;color:'+vc(v.vv,90,75)+'">🔴'+v.vv+'</span> '+
          '<span style="font-size:10px;color:'+(v.bonus>0?'#3fb950':v.bonus<0?'#f85149':'#555')+'">B'+(v.bonus||0)+'</span> '+
          '<span style="font-size:10px;color:'+(v.rev>0?'#d2d268':v.rev<0?'#f85149':'#555')+'">R'+(v.rev||0)+'</span> '+
          '<span style="font-size:10px;color:'+(v.btc>0?'#3fb950':v.btc<0?'#f85149':'#555')+'">₿'+(v.btc||0)+'</span>'
        : (isVeto ? '<span style="color:#f85149;font-size:11px">⛔ '+v.veto+'</span>' : '<span style="color:#555;font-size:11px">ожидание...</span>');
      return '<tr>'+
        '<td style="font-size:14px;padding:4px 2px">'+em(s.split('/')[0])+'</td>'+
        '<td style="font-weight:600;font-size:13px">'+s.split('/')[0]+'</td>'+
        '<td style="font-size:11px;color:#8b949e">'+(p.qty>0?fmtQ(p.qty):'—')+'</td>'+
        '<td style="font-size:11px;color:#8b949e">'+(p.entry>0?fmtP(p.entry):(p.current_price>0?'<span style="color:#d29922;font-size:10px">'+fmtP(p.current_price)+'</span>':'—'))+'</td>'+
        '<td style="font-size:11px;color:#8b949e">'+(p.qty>0?fmtP(p.qty*p.entry):(p.current_price>0?'<span style="color:#8b949e;font-size:10px">тек.</span>':'—'))+'</td>'+
        '<td style="text-align:left;font-size:12px;white-space:nowrap">'+votesLine+'</td>'+
        '<td style="text-align:center">'+
          '<div style="font-weight:700;font-size:15px;color:'+barColor+'">'+sigEmoji+' '+v.score+'</div>'+
          '<div style="width:55px;height:3px;background:#21262d;border-radius:2px;margin:2px auto 0"><div style="width:'+barW+'%;height:3px;background:'+barColor+';border-radius:2px"></div></div>'+
        '</td>'+
        '</tr>';
    }).filter(Boolean).join('')
    :'<tr><td colspan="7" style="text-align:center;color:#8b949e;font-size:13px;padding:20px">Нет данных</td></tr>';
}

function updTrades(tr){
  if(!tr||!tr.length){document.getElementById('tbTrades').innerHTML='<tr><td colspan="6" style="text-align:center;color:#8b949e;font-size:13px;padding:20px">Нет сделок</td></tr>';return;}
  document.getElementById('tbTrades').innerHTML=tr.map(t=>{
    const pnlColor=t.pnl>0?'#3fb950':t.pnl<0?'#f85149':'#8b949e';
    return '<tr><td>'+em(t.symbol)+' '+t.symbol+'</td><td>'+(t.side==='buy'?'🟢 BUY':'🔴 SELL')+'</td><td>'+fmtP(t.price)+'</td><td>'+fmtQ(t.qty)+'</td><td style="color:'+pnlColor+'">'+(t.pnl?fmtP(t.pnl):'—')+'</td><td style="font-size:11px;color:#8b949e">'+(t.ts||'').split('T')[0]+'</td></tr>';
  }).join('');
}

function toggleSection(id){const el=document.getElementById(id);el.style.display=el.style.display==='none'?'block':'none';}

function loadChangelog(){fetch('/api/changelog').then(r=>r.json()).then(d=>{document.getElementById('changelogContent').textContent=d.lines||'Нет записей';}).catch(()=>{document.getElementById('changelogContent').textContent='Ошибка загрузки';})}

function toggleTradeLog(){
  const m=document.getElementById('tradeLogModal');
  if(m.style.display==='block'){m.style.display='none';return;}
  m.style.display='block';
  fetch('/api/trade-history').then(r=>r.json()).then(d=>{
    const tr=d.trades||[];
    const buys=tr.filter(t=>t.side==='buy').reduce((s,t)=>s+(t.value||0),0);
    const sells=tr.filter(t=>t.side==='sell').reduce((s,t)=>s+(t.value||0),0);
    const pnlTrades=tr.filter(t=>t.side==='sell'&&t.pnl);
    const totalPnl=pnlTrades.reduce((s,t)=>s+(t.pnl||0),0);
    const wins=pnlTrades.filter(t=>t.pnl>0).length;
    document.getElementById('tradeLogSummary').innerHTML='Куплено: <b>'+fmtP(buys)+'</b> | Продано: <b>'+fmtP(sells)+'</b> | PnL: <b style="color:'+(totalPnl>=0?'#3fb950':'#f85149')+'">'+fmtP(totalPnl)+'</b> | Сделок: '+tr.length+' | Успешных: '+wins+'/'+pnlTrades.length;
    document.getElementById('tbTradeLog').innerHTML=tr.map(t=>{
      const pnlColor=t.pnl>0?'#3fb950':t.pnl<0?'#f85149':'#8b949e';
      const sideColor=t.side==='buy'?'#3fb950':'#f85149';
      return '<tr><td>'+em(t.symbol)+' '+t.symbol+'</td><td style="color:'+sideColor+'">'+t.side+'</td><td>'+fmtP(t.price)+'</td><td>'+fmtQ(t.qty)+'</td><td>'+fmtP(t.value||t.qty*t.price)+'</td><td style="color:'+pnlColor+'">'+(t.pnl!=null?fmtP(t.pnl):'—')+'</td><td style="color:'+pnlColor+'">'+(t.pnl_pct?(t.pnl_pct>=0?'+':'')+Number(t.pnl_pct).toFixed(2)+'%':'')+'</td><td style="color:#8b949e;font-size:11px">'+(t.ts||'').slice(0,19)+'</td></tr>';
    }).join('')||'<tr><td colspan="8" style="text-align:center;padding:30px;color:#8b949e">Нет сделок</td></tr>';
  }).catch(()=>{document.getElementById('tbTradeLog').innerHTML='<tr><td colspan="8" style="text-align:center;padding:30px;color:#f85149">Ошибка загрузки</td></tr>'});
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
fetch('/api/status').then(r=>r.json()).then(d=>{updBar(d)});
</script>
</body></html>""")


def run_server(host: str = "0.0.0.0", port: int = 8765):
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_server()
