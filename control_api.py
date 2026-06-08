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
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator
from pathlib import Path
from vote_parser import parse_votes

import re
import traceback
import db_pg as db
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

log = logging.getLogger('control_api')

# ─── КОНФИГ ──────────────────────────────────────────────────────────────────

SYSTEM_PID_FILE = os.path.join(BASE_DIR, "data", "orchestrator.pid")
SYSTEM_LOG_FILE = "/tmp/system_v4.log"
PG_DSN = db.get_db_path()  # PostgreSQL DSN

app = FastAPI(title="Super System Dashboard", version="1.0.0")


# ─── ГЛОБАЛЬНЫЙ ОБРАБОТЧИК ОШИБОК ────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Перехватывает все необработанные исключения, логирует и возвращает generic-сообщение."""
    log.error(f"Unhandled exception ({request.url.path}): {exc}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"error": "Внутренняя ошибка сервера"})


# ─── ИНИЦИАЛИЗАЦИЯ ───────────────────────────────────────────────────────────

def _init_capital_table():
    """Таблица capital_snapshots уже создана в PG через schema.sql."""
    db.init_db()
    pass


# Вызываем при импорте
_init_capital_table()


# ─── ФОНОВЫЙ СБОРЩИК КАПИТАЛА ───────────────────────────────────────────────

def _snapshot_worker():
    """Фоновый поток: каждые 60 сек снэпшот капитала.
    Берёт реальные данные из /tmp/real_balance.json (пишется трейдером раз в 60 сек).
    """
    while True:
        try:
            pid = None
            try:
                with open(SYSTEM_PID_FILE) as f:
                    pid = int(f.read().strip())
                os.kill(pid, 0)
            except Exception:
                pid = None

            if pid:
                # Читаем реальный баланс с биржи
                try:
                    with open('/tmp/real_balance.json') as f:
                        real = json.load(f)
                    total = real['total']
                    pos_value = real['in_positions']
                    free_usdt = real['free_usdt']
                    count = real['positions_count']
                except Exception:
                    # Фолбэк: считаем по entry_price (неточно)
                    positions = db.get_all_positions()
                    pos_value = sum(p['quantity'] * p['entry_price'] for p in positions.values())
                    count = len(positions)
                    total = 0
                    free_usdt = 0

                db.save_capital_snapshot(round(total, 2), round(pos_value, 2), round(free_usdt, 2), count)
        except Exception:
            pass
        time.sleep(60)


# Запускаем в daemon-потоке
_snapshot_thread = threading.Thread(target=_snapshot_worker, daemon=True, name='cap-snapshot')
_snapshot_thread.start()


# ─── ВСПОМОГАТЕЛЬНЫЕ ─────────────────────────────────────────────────────────

def read_log_tail(n: int = 50) -> list:
    """Читает последние N строк лога с конца файла."""
    try:
        with open(SYSTEM_LOG_FILE, 'rb') as f:
            f.seek(0, 2)  # конец
            fsize = f.tell()
            chunk = min(fsize, 100 * 1024)  # читаем до 100KB с конца
            f.seek(max(0, fsize - chunk))
            # Отбрасываем первую неполную строку
            data = f.read()
            lines = data.decode('utf-8', errors='replace').splitlines()
            if len(lines) > n:
                return lines[-n:]
            # Если не хватило — читаем больше
            if chunk < fsize:
                f.seek(0)
                all_lines = f.read().decode('utf-8', errors='replace').splitlines()
                return all_lines[-n:]
            return lines
    except Exception:
        return ["Нет лог-файла"]


def read_log_since(timestamp: float) -> list:
    try:
        mtime = os.path.getmtime(SYSTEM_LOG_FILE)
        if mtime < timestamp:
            return []
        with open(SYSTEM_LOG_FILE, 'r', errors='replace') as f:
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
    # Если PID-файла нет — создаём из текущего процесса (защита рестарта)
    if not os.path.exists(SYSTEM_PID_FILE):
        try:
            os.makedirs(os.path.dirname(SYSTEM_PID_FILE), exist_ok=True)
            with open(SYSTEM_PID_FILE, 'w') as f:
                f.write(str(os.getpid()))
        except Exception:
            pass
    try:
        with open(SYSTEM_PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return pid
    except Exception:
        return None


def _parse_votes_from_log() -> dict:
    """Парсит последние голоса DecisionEngine из лога."""
    return parse_votes()


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
    
    # Реальный баланс с биржи (если доступен)
    real_balance = None
    try:
        with open('/tmp/real_balance.json') as f:
            real_balance = json.load(f)
    except Exception:
        pass
    
    if real_balance:
        balance_info = {
            'in_positions': round(real_balance['in_positions'], 2),
            'positions_count': real_balance['positions_count'],
            'free_usdt': round(real_balance['free_usdt'], 2),
            'total': round(real_balance['total'], 2),
        }
    else:
        balance_info = {
            'in_positions': round(total_in_positions, 2),
            'positions_count': len(positions),
        }

    trades_history = []
    try:
        rows = db.get_trade_history(limit=100)
        trades_history = []
        for r in rows:
            is_closed = r.get('exit_price') is not None
            trades_history.append({
                'id': r['id'], 'symbol': r['symbol'],
                'side': 'sell' if is_closed else 'buy',
                'price': float(r['exit_price'] if is_closed else r['entry_price'] or 0),
                'qty': float(r['exit_qty'] or r['entry_qty'] or 0),
                'value': (float(r['exit_price'] or r['entry_price'] or 0) * float(r['exit_qty'] or r['entry_qty'] or 0)),
                'pnl': float(r['pnl']) if r.get('pnl') else 0,
                'pnl_pct': float(r['pnl_percent']) if r.get('pnl_percent') else 0,
                'ts': str(r.get('exit_time', '') or r.get('entry_time', ''))}
            )
        trades_history = trades_history[:50]
    except Exception as e:
        log.error(f"trade-history: {e}")

    votes = _parse_votes_from_log()

    # Обогащаем позиции голосами
    enriched_positions = {}
    # Текущие цены: сначала из PriceCache (WS), fallback на highest_price из PG
    from data_cache import PriceCache
    pxc = PriceCache()
    for sym, pos in positions.items():
        cur_price = pxc.get_price(sym) or pos.get('highest_price', 0) or 0
        p = {'qty': pos['quantity'], 'entry': pos['entry_price'],
             'current_price': cur_price,
             'entry_time': pos.get('entry_time', ''),
             'direction': pos.get('side', 'long').upper()}
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
                'direction': v.get('direction', 'LONG'),
                'votes': v
            }
        else:
            # Обновляем направление из голосов, если нет позиции
            if enriched_positions[sym].get('qty', 0) == 0:
                enriched_positions[sym]['direction'] = v.get('direction', 'LONG')

    # BTC Regime: парсим из лога последнее сообщение
    btc_regime = {}
    try:
        log_lines = read_log_tail(2000)
        for line in reversed(log_lines):
            # Формат: BTC Regime: ... → rec=... (из industrial_trader)
            # Формат: Regime=..., rec=... (из btc_regime_tracker)
            # Формат: "BTC Regime <regime> — ..." (из HOLD сообщений)
            rm = re.search(r'(?:BTC )?Regime\s*(?:[:=]|\s+)(\w+)', line)
            if rm:
                regime_name = rm.group(1).lower()
                # Нормализуем названия
                rec_map = {
                    'pump': 'sell_only',
                    'dump': 'no_trade',
                    'distribution': 'sell_only',
                    'accumulation': 'buy_allowed',
                    'recovery': 'buy_priority',
                    'unknown': 'buy_allowed',
                }
                # Попробуем найти rec= в той же строке
                rec_match = re.search(r'rec=(\w+)', line)
                recommendation = rec_match.group(1) if rec_match else rec_map.get(regime_name, 'buy_allowed')
                btc_regime = {
                    'regime': regime_name,
                    'recommendation': recommendation,
                }
                break
    except Exception:
        pass
    
    # BTC Direction: парсим направление от ML-предиктора
    btc_direction = {}
    try:
        log_lines = read_log_tail(1000)
        for line in reversed(log_lines):
            # 🔮 BTC Direction: side (conf=50%, strength=50, up=33% down=33%) или с side=
            dm = re.search(r'BTC Direction: (\w+) \(conf=(\d+)%, strength=(\d+), up=(\d+)% down=(\d+)%', line)
            if dm:
                btc_direction = {
                    'direction': dm.group(1),
                    'confidence': int(dm.group(2)),
                    'strength': int(dm.group(3)),
                    'up_probability': int(dm.group(4)),
                    'down_probability': int(dm.group(5)),
                }
                break
    except Exception:
        pass
    
    # BTC цена/RSI из последнего анализа
    btc_analysis = {}
    try:
        log_lines = read_log_tail(1000)
        for line in reversed(log_lines):
            am = re.search(r'BTC/USDT: Цена=\$([\d.]+), Тренд=(\w+), Уверенность=([\d.]+)%, RSI=([\d.]+)', line)
            if am:
                btc_analysis = {
                    'price': float(am.group(1)),
                    'trend': am.group(2),
                    'confidence': float(am.group(3)),
                    'rsi': float(am.group(4)),
                }
                break
    except Exception:
        pass
    
    # Liquidity — ордерблоки и Order Flow из лога
    # Сначала ищем BTCLiq (реальные BTC данные от btc_direction._log_btc_liquidity),
    # если нет — берём последний доступный Liq альткоина как proxy.
    btc_liquidity = {}
    try:
        log_lines = read_log_tail(1000)
        
        # Сначала ищем BTCLiq — реальные BTC данные
        for line in reversed(log_lines):
            if 'BTCLiq' in line and 'Liq:' in line:
                lm = re.search(r'Liq:(\d+)\(POC=([\d.]+) VAH=([\d.]+) VAL=([\d.]+) q=([\d.]+) fvg↑=(\d+) fvg↓=(\d+)', line)
                if lm:
                    btc_liquidity = {
                        'symbol': 'BTC',
                        'score': int(lm.group(1)),
                        'poc': float(lm.group(2)),
                        'vah': float(lm.group(3)),
                        'val': float(lm.group(4)),
                        'quality': float(lm.group(5)),
                        'fvg_above': int(lm.group(6)),
                        'fvg_below': int(lm.group(7)),
                        'source': 'btc direct',
                    }
                    # Добавляем POC тренд из BTCLiq
                    if 'POC↑' in line:
                        btc_liquidity['poc_trend'] = 'up'
                    elif 'POC↓' in line:
                        btc_liquidity['poc_trend'] = 'down'
                    else:
                        btc_liquidity['poc_trend'] = 'flat'
                    break
        
        # Если BTCLiq не нашли — пробуем proxy с альткоинов
        if not btc_liquidity:
            for line in reversed(log_lines):
                if 'Liq:' in line:
                    lm = re.search(r'Liq:(\d+)\(POC=([\d.]+) VAH=([\d.]+) VAL=([\d.]+) q=([\d.]+) fvg↑=(\d+) fvg↓=(\d+)', line)
                    if lm:
                        sym_lm = re.search(r'([A-Z]+)/USDT', line)
                        src_symbol = sym_lm.group(1) if sym_lm else '?'
                        btc_liquidity = {
                            'symbol': src_symbol,
                            'score': int(lm.group(1)),
                            'poc': float(lm.group(2)),
                            'vah': float(lm.group(3)),
                            'val': float(lm.group(4)),
                            'quality': float(lm.group(5)),
                            'fvg_above': int(lm.group(6)),
                            'fvg_below': int(lm.group(7)),
                            'source': 'altcoin proxy',
                        }
                        # Добавляем OB и POC тренд
                        ob_match = re.search(r'OB(\d+) (\w+) (\w+)', line)
                        if ob_match:
                            btc_liquidity['ob_count'] = int(ob_match.group(1))
                            btc_liquidity['ob_type'] = ob_match.group(2)
                            btc_liquidity['ob_direction'] = ob_match.group(3)
                        if 'POC↑' in line:
                            btc_liquidity['poc_trend'] = 'up'
                    elif 'POC↓' in line:
                        btc_liquidity['poc_trend'] = 'down'
                    else:
                        btc_liquidity['poc_trend'] = 'flat'
                    break
    except Exception:
        pass
    
    # EARN сигнал: на основе BTC Direction для ручного управления BTC в EARN
    earn_signal = {'signal': 'HOLD', 'confidence': 'neutral', 'reason': 'Нет данных'}
    if btc_direction:
        direction = btc_direction.get('direction', 'side')
        strength = btc_direction.get('strength', 50)
        if direction == 'down' and strength >= 50:
            earn_signal = {
                'signal': 'SELL',
                'confidence': 'medium' if strength >= 60 else 'low',
                'reason': f'BTC ML=down({strength})'
            }
        elif direction == 'up' and strength >= 60:
            earn_signal = {
                'signal': 'BUY',
                'confidence': 'medium' if strength >= 70 else 'low',
                'reason': f'BTC ML=up({strength})'
            }
        else:
            dir_text = {'up': 'UP', 'down': 'DOWN', 'side': 'SIDE'}.get(direction, '—')
            earn_signal = {
                'signal': 'HOLD',
                'confidence': 'neutral',
                'reason': f'BTC {dir_text}({strength}) — нет уверенного сигнала'
            }
    
    result = {
        'running': running,
        'pid': pid,
        'timestamp': time.time(),
        'positions': enriched_positions,
        'pnl': pnl,
        'balance': balance_info,
        'trades': trades_history,
        'btc_regime': btc_regime,
        'btc_direction': btc_direction,
        'btc_analysis': btc_analysis,
        'btc_liquidity': btc_liquidity,
        'earn_signal': earn_signal,
        'log_tail': read_log_tail(5),
    }
    
    # Добавляем OI уровни ликвидаций
    try:
        from collect_oi import get_oi_collector
        oi = get_oi_collector()
        oi_levels = {}
        for sym in oi.data:
            # Цена из OI данных (последняя запись), затем из позиций/голосов
            cur_price = 0
            entries = oi.data.get(sym, [])
            if entries:
                cur_price = entries[-1].get('price', 0)
            if not cur_price:
                base = sym.replace('/USDT', '')
                cur_price = enriched_positions.get(sym, {}).get('entry', 0)
            if not cur_price:
                cur_price = enriched_positions.get(base, {}).get('entry', 0)
            if not cur_price and sym in votes:
                cur_price = votes[sym].get('price', 0)
            levels = oi.get_liq_levels(sym, cur_price or 0)
            heat = levels.get('heat', 0)
            if heat > 0:
                liq_long = levels.get('liq_zone_long', None)
                liq_short = levels.get('liq_zone_short', None)
                liq_long_min = liq_long[0] if liq_long and len(liq_long) == 2 else 0
                liq_long_max = liq_long[1] if liq_long and len(liq_long) == 2 else 0
                liq_short_min = liq_short[0] if liq_short and len(liq_short) == 2 else 0
                liq_short_max = liq_short[1] if liq_short and len(liq_short) == 2 else 0
                oi_levels[sym] = {
                    'heat': heat,
                    'bonus': levels.get('score_bonus', 0),
                    'current_price': cur_price,
                    'liq_long_min': liq_long_min,
                    'liq_long_max': liq_long_max,
                    'liq_short_min': liq_short_min,
                    'liq_short_max': liq_short_max,
                    'price_at_zone_long': (liq_long_min and cur_price >= liq_long_min and cur_price <= liq_long_max) if cur_price else False,
                    'price_at_zone_short': (liq_short_min and cur_price >= liq_short_min and cur_price <= liq_short_max) if cur_price else False,
                }
                if levels.get('levels'):
                    oi_levels[sym]['levels_detail'] = []
                    for lev in levels['levels']:
                        oi_levels[sym]['levels_detail'].append({
                            'name': lev.get('level', '?'),
                            'long': lev.get('liq_long', 0),
                            'short': lev.get('liq_short', 0),
                        })
        result['oi_liquidation_levels'] = oi_levels if oi_levels else {}
    except Exception as e:
        log.error(f"[OI] ошибка: {e}")
        log.debug(traceback.format_exc())
        result['oi_liquidation_levels'] = {}
    
    # CVD — Cumulative Volume Delta (реальный поток агрессивных сделок)
    try:
        from ws_client import get_ws_client
        ws = get_ws_client()
        if ws and hasattr(ws, 'get_symbol_cvd_summary'):
            cvd_status = {}
            for sym in list(result.get('positions', {}).keys())[:6]:
                cvd = ws.get_symbol_cvd_summary(sym, n_minutes=30)
                if cvd:
                    cvd_status[sym] = {
                        'trend': cvd.get('trend'),
                        'buy_pct': cvd.get('buy_pct'),
                        'cvd_net': cvd.get('cvd_net'),
                        'volume_usd': cvd.get('volume_usd'),
                        'minutes': cvd.get('minutes'),
                    }
            result['cvd'] = {
                'symbols_tracked': len(ws._cvd_data),
                'positions': cvd_status,
            }
    except Exception:
        pass
    
    # 🧠 VoteTracker: динамические веса модулей
    try:
        from vote_tracker import get_tracker as _vt
        _vt_inst = _vt()
        result['vote_tracker'] = {
            'summary': _vt_inst.get_summary(),
            'pending': _vt_inst.get_pending_count(),
            'weights': _vt_inst.get_weights(),
        }
    except Exception:
        pass

    return result


# ─── ЭНДПОИНТЫ ──────────────────────────────────────────────────────────────

@app.get("/api/status")
async def api_status():
    return get_status_snapshot()


@app.get("/api/trade-history")
async def api_trade_history():
    """История всех сделок с PnL, отсортированная по времени."""
    try:
        rows = db.get_trade_history(limit=500)
        return {'trades': [
            {'id': r['id'], 'symbol': r['symbol'], 
             'side': 'sell' if r.get('status') == 'closed' else 'buy',
             'price': (float(r['exit_price']) if r.get('exit_price') else 0) if r.get('status') == 'closed' else (float(r['entry_price']) if r.get('entry_price') else 0),
             'qty': float(r['entry_qty']) if r.get('entry_qty') else 0,
             'value': (float(r['entry_price']) * float(r['entry_qty'])) if r.get('entry_price') and r.get('entry_qty') else 0,
             'pnl': float(r['pnl']) if r.get('pnl') else 0,
             'pnl_pct': float(r['pnl_percent']) if r.get('pnl_percent') else 0,
             'ts': str(r.get('exit_time', '') or r.get('entry_time', ''))}
            for r in rows
        ]}
    except Exception as e:
        log.error(f"trade-history: {e}")
        log.debug(traceback.format_exc())
        return {'trades': [], 'error': 'Internal server error'}


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
        log.error(f"changelog: {e}")
        log.debug(traceback.format_exc())
        return {'file': None, 'lines': 'Файл изменений не найден', 'total_lines': 0}


@app.get("/api/capital-history")
async def api_capital_history():
    """Последние 500 точек для графика капитала."""
    try:
        rows = db.get_capital_history(limit=500)
        rows.reverse()
        return {'points': [{'t': str(r['created_at']), 'v': r['total']} for r in rows]}
    except Exception as e:
        log.error(f"capital-history: {e}")
        log.debug(traceback.format_exc())
        return {'points': [], 'error': 'Internal server error'}


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
  <div class="status-card"><h3>BTC Режим</h3><div class="value" id="stBtcRegime">загрузка...</div></div>
  <div class="status-card" style="grid-column:span 2"><h3>BTC Направление</h3><div class="value" id="stBtcDirection" style="font-size:13px">загрузка...</div></div>
  <div class="status-card" style="grid-column:span 2"><h3>BTC Анализ</h3><div class="value" id="stBtcAnalysis">загрузка...</div></div>
  <div class="status-card" style="grid-column:span 2;border:1px solid #f0c040"><h3>🏦 EARN Сигнал</h3><div class="value" id="stEarnSignal" style="font-size:14px">загрузка...</div></div>
  <div class="status-card" style="grid-column:span 2"><h3>📊 BTC Order Flow</h3><div class="value" id="stBtcLiquidity" style="font-size:12px">загрузка...</div></div>
  <div class="status-card" style="grid-column:span 2"><h3>🔥 OI Liquidation Zones</h3><div class="value" id="stOiLevels" style="font-size:12px;line-height:1.6;max-height:280px;overflow-y:auto">загрузка...</div></div>
  <div class="status-card" style="grid-column:span 2"><h3>🚀 Последняя сделка</h3><div class="value" id="stLastTrade" style="font-size:12px">загрузка...</div></div>
</div>

<div class="section" style="grid-column:1/-1">
    <h2>📊 Мониторинг голосов <span style="font-size:12px;color:#8b949e;font-weight:normal">— вход при Score ≥ 65</span></h2>
    <!-- Легенда голосов -->
    <div style="margin-bottom:12px;padding:8px 10px;background:#1c2128;border-radius:6px;font-size:12px;display:flex;flex-wrap:wrap;gap:6px 16px">
      <span><span style="color:#3fb950;font-weight:700">🔵ML</span>=ML-Pro (10%)</span>
      <span><span style="color:#3fb950;font-weight:700">🟢Ad</span>=ML-Advisor (25%)</span>
      <span><span style="color:#3fb950;font-weight:700">🟡TF</span>=MTF (0%)</span>
      <span><span style="color:#3fb950;font-weight:700">🟣RV</span>=RSI/Vol (0%)</span>
      <span><span style="color:#3fb950;font-weight:700">🟠LQ</span>=Liquidity (25%)</span>
      <span><span style="color:#3fb950;font-weight:700">🔴VV</span>=Volume/VWAP (10%)</span>
      <span><span style="color:#3fb950;font-weight:700">🧊CV</span>=CVD OrderFlow (10%) 🆕</span>
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

<div class="section" style="grid-column:1/-1">
  <h2>⛔ VETO-логи <span style="font-size:12px;color:#8b949e;font-weight:normal">— последние 15 отказов</span></h2>
  <div id="tbVeto" style="font-family:'SF Mono','Consolas',monospace;font-size:12px;line-height:1.7;max-height:250px;overflow-y:auto">
    <div style="text-align:center;color:#8b949e;padding:15px">Загрузка...</div>
  </div>
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

<div class="section" style="font-size:12px">
  <h2>📐 Легенда сигналов</h2>
  <div style="display:flex;flex-wrap:wrap;gap:12px 24px">
    <div><span style="display:inline-block;width:14px;text-align:center">🔵</span> ML-Pro <span style="color:#8b949e">(ML)</span></div>
    <div><span style="display:inline-block;width:14px;text-align:center">🟢</span> Advisor <span style="color:#8b949e">(фундамент)</span></div>
    <div><span style="display:inline-block;width:14px;text-align:center">🟡</span> MTF <span style="color:#8b949e">(таймфреймы)</span></div>
    <div><span style="display:inline-block;width:14px;text-align:center">🟣</span> Reversal <span style="color:#8b949e">(разворот)</span></div>
    <div><span style="display:inline-block;width:14px;text-align:center">🟠</span> Liquidity <span style="color:#8b949e">(POC/OB)</span></div>
    <div><span style="display:inline-block;width:14px;text-align:center">🔴</span> Volume/VWAP</div>
    <div><span style="display:inline-block;width:14px;text-align:center">B</span> Бонус <span style="color:#8b949e">(+BTC)</span></div>
    <div><span style="display:inline-block;width:14px;text-align:center">R</span> Реверсивный</div>
    <div><span style="display:inline-block;width:14px;text-align:center">₿</span> Маяк BTC</div>
  </div>
  <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:12px 24px;border-top:1px solid #30363d;padding-top:8px">
    <span style="font-weight:600">▫IDM</span> <span style="color:#8b949e">—</span> <span style="color:#d29922">жетый</span> = индукция, направление неясно
    <span style="font-weight:600">▫IDM↑</span> <span style="color:#8b949e">—</span> <span style="color:#e06c00">тёмно-оранж</span> = ликвидность ✦снизу✦ → цель вверх 🟢
    <span style="font-weight:600">▫IDM↓</span> <span style="color:#8b949e">—</span> <span style="color:#e06c00">тёмно-оранж</span> = ликвидность ✦сверху✦ → цель вниз 🔴
    <span style="font-weight:600">▫OB↑</span> <span style="color:#8b949e">—</span> <span style="color:#3fb950">зелёный</span> = бычий ордерблок (поддержка) 🟢
    <span style="font-weight:600">▫OB↓</span> <span style="color:#8b949e">—</span> <span style="color:#f85149">красный</span> = медвежий ордерблок (сопротивление) 🔴
  </div>
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
      // Direction: Long 🟢 / Short 🔴
      const isShort = p.direction === 'SHORT';
      const dirBadge = isShort ? '<span style="color:#f85149;font-size:10px;font-weight:600">🔴 SHORT</span>' : '<span style="color:#3fb950;font-size:10px;font-weight:600">🟢 LONG</span>';
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
          '<span title="CVD: '+v.cvd+'" style="cursor:help;color:'+vc(parseInt(v.cvd),85,60)+'">🧊'+shortVote(v.cvd)+'</span> '+
          '<span style="font-size:10px;color:'+(v.bonus>0?'#3fb950':v.bonus<0?'#f85149':'#555')+'">B'+(v.bonus||0)+'</span> '+
          '<span style="font-size:10px;color:'+(v.rev>0?'#d2d268':v.rev<0?'#f85149':'#555')+'">R'+(v.rev||0)+'</span> '+
          '<span style="font-size:10px;color:'+(v.btc>0?'#3fb950':v.btc<0?'#f85149':'#555')+'">₿'+(v.btc||0)+'</span>'+
          (v.oi_heat>0?' <span style="font-size:10px;color:#d29922" title="OI heat='+v.oi_heat+' +'+v.oi_bonus+'pts">🔥'+(v.oi_heat||0)+'</span>':'')+
          (v.idm?' <span style="font-size:11px;font-weight:600;color:'+(v.idm==='IDM'||v.idm==='IDM↑'?'#d29922':v.idm==='IDM↓'?'#e06c00':v.idm==='OB↑'?'#3fb950':v.idm==='OB↓'?'#f85149':'#8b949e')+'" title="'+(v.idm==='IDM'?'Индукция (Inducement) — сбор ликвидности, направление не определено':v.idm==='IDM↑'?'Индукция бычья (Inducement ↑) — ликвидность собрана снизу, цель вверх':v.idm==='IDM↓'?'Индукция медвежья (Inducement ↓) — ликвидность собрана сверху, цель вниз':v.idm==='OB↑'?'Ордерблок бычий (Order Block ↑) — поддержка, зона набора позиции':v.idm==='OB↓'?'Ордерблок медвежий (Order Block ↓) — сопротивление, зона набора шортов':'Неизвестно')+'">▫'+v.idm+'</span>':'')
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
  document.getElementById('stCapital').innerHTML=(d.balance?fmtP(d.balance.total):'$0')+' <span style="font-size:12px;color:#8b949e">(всего)</span>';
  document.getElementById('stTrades').textContent=d.pnl?.total_trades||0;
  const btcR=d.btc_regime||{regime:'—',recommendation:''};
  const regimeEl=document.getElementById('stBtcRegime');
  if(regimeEl){regimeEl.textContent=btcR.regime+' / '+btcR.recommendation;regimeEl.className='value '+(btcR.recommendation==='buy_allowed'||btcR.recommendation==='buy_priority'?'green':'red');}
  const btcDir=d.btc_direction||{};
  const dirEl=document.getElementById('stBtcDirection');
  if(dirEl){const dirName={up:'📈',down:'📉',side:'➡️',unknown:'❓'}[btcDir.direction]||'—';dirEl.innerHTML=(btcDir.direction?'<b>'+dirName+' '+btcDir.direction+'</b> | conf='+btcDir.confidence+'% | up='+btcDir.up_probability+'% down='+btcDir.down_probability+'%':'загрузка...');dirEl.className='value '+(btcDir.direction==='up'?'green':btcDir.direction==='down'?'red':'');}
  const btcA=d.btc_analysis||{};
  const aEl=document.getElementById('stBtcAnalysis');
  if(aEl){aEl.innerHTML=btcA.price?'$'+fmtP(btcA.price)+' | '+btcA.trend+' | RSI='+btcA.rsi:'загрузка...';aEl.className='value '+(btcA.trend==='bullish'?'green':btcA.trend==='bearish'?'red':'');}
  
  // 🏦 EARN сигнал
  const earnEl=document.getElementById('stEarnSignal');
  const earnSig=d.earn_signal||{signal:'HOLD',confidence:'neutral',reason:'—'};
  if(earnEl){
    const signalIcons={SELL:'🔴 SELL',BUY:'🟢 BUY',HOLD:'⚪ HOLD'};
    const signalCls={SELL:'red',BUY:'green',HOLD:'value'};
    const confStars={high:'★★★',medium:'★★☆',low:'★☆☆',neutral:'★☆☆'};
    earnEl.innerHTML='<b>'+(signalIcons[earnSig.signal]||'❓')+'</b> | '+confStars[earnSig.confidence]+' | '+earnSig.reason;
    earnEl.className='value '+(signalCls[earnSig.signal]||'');
  }
  
  // 🧊 BTC Order Flow / Liquidity
  const btcL=d.btc_liquidity||{};
  const lEl=document.getElementById('stBtcLiquidity');
  if(lEl){
    if(btcL.score!=null){
      const liqColor=btcL.score>=75?'green':btcL.score>=50?'#d29922':'red';
      const obInfo=btcL.ob_count?((btcL.ob_direction==='bullish'?'🟢':'🔴')+' OB'+btcL.ob_count+' '+btcL.ob_type+' '+btcL.ob_direction):'—';
      const pocDir={'up':'▲','down':'▼','flat':'→'}[btcL.poc_trend]||'→';
      const fvgInfo=(btcL.fvg_above||0)+'↑ '+(btcL.fvg_below||0)+'↓';
      const srcBadge=btcL.source==='btc direct'?' <span style="font-size:10px;color:#3fb950">●BTC</span>':' <span style="font-size:10px;color:#8b949e">proxy:'+btcL.symbol+'</span>';
      lEl.innerHTML='<b style="color:'+liqColor+'">Liq Score: '+btcL.score+'</b> | POC=$'+fmtP(btcL.poc)+' '+pocDir+
        ' | VAH=$'+fmtP(btcL.vah)+' | VAL=$'+fmtP(btcL.val)+
        ' | FVG: '+fvgInfo+
        ' | OB: '+obInfo+
        srcBadge;
      lEl.className='value';
    }else{
      lEl.innerHTML='Нет данных — BTC не в активной оценке';
      lEl.className='value';
    }
  }
  
  // 🔥 OI Liquidation Zones
  const oiLvls=d.oi_liquidation_levels||{};
  const oiEl=document.getElementById('stOiLevels');
  if(oiEl){
    const keys=Object.keys(oiLvls);
    if(keys.length>0){
      let html='';
      keys.forEach(sym=>{
        const o=oiLvls[sym];
        const heatEmojis=['','🟡','🟠','🔴'];
        const heatStr=heatEmojis[o.heat]||'🟡';
        const inLong=o.price_at_zone_long?' ⚠️LONG':'';
        const inShort=o.price_at_zone_short?' ⚠️SHORT':'';
        const warnLong=o.price_at_zone_long?'color:#3fb950;font-weight:bold':'color:#3fb950';
        const warnShort=o.price_at_zone_short?'color:#f85149;font-weight:bold':'color:#f85149';
        html+='<div style="margin-bottom:4px;padding:3px 5px;background:#1c2128;border-radius:4px;font-size:11px">';
        html+='<b>'+sym.split('/')[0]+'</b> '+heatStr+' heat='+o.heat+' (+'+o.bonus+'pts) ';
        html+='<span style="color:#8b949e">L:<span style="'+warnLong+'">$'+fmtP(o.liq_long_min)+'</span>/S:<span style="'+warnShort+'">$'+fmtP(o.liq_short_min)+'</span> ∼$'+fmtP(o.current_price)+'</span>';
        if(o.levels_detail && o.levels_detail.length>0){
          html+=' <span style="font-size:10px;color:#555">('+o.levels_detail.map(l=>l.name.split(' ')[0]).join('/')+')</span>';
        }
        html+=inLong+inShort+'</div>';
      });
      oiEl.innerHTML=html;
    }else{
      oiEl.innerHTML='<span style="color:#555">Нет активных OI-уровней</span>';
    }
    oiEl.className='value';
  }
  
  // 🚀 Последняя сделка
  fetch('/api/last-trade').then(r=>r.json()).then(t=>{
    const ltEl=document.getElementById('stLastTrade');
    if(!ltEl)return;
    if(t.symbol){
      const s=t.side==='buy'?'🟢 BUY':'🔴 SELL';
      const c=t.side==='buy'?'#3fb950':'#f85149';
      ltEl.innerHTML='<span style="color:'+c+'">'+s+'</span> '+t.symbol.split('/')[0]+' @ $'+fmtP(t.price)+' | '+t.ts_human;
    }else{
      ltEl.innerHTML='<span style="color:#555">—</span>';
    }
  }).catch(()=>{});
  
  document.getElementById('subtitle').textContent='PID: '+(d.pid||'—')+' | обновлено '+new Date().toLocaleTimeString();
  updPos(d.positions||{});
  if(d.trades) updTrades(d.trades);
  updVeto();
}

function updVeto(){
  fetch('/api/veto-history').then(r=>r.json()).then(d=>{
    const el=document.getElementById('tbVeto');
    if(!el||!d.entries||!d.entries.length){if(el)el.innerHTML='<div style="text-align:center;color:#8b949e;padding:15px">Нет VETO</div>';return;}
    el.innerHTML=d.entries.map(e=>{
      const ts=(e.ts||'').slice(11,19);
      const sym=e.symbol.replace('/USDT','');
      const reason=e.veto_reason||'';
      return '<div style="display:flex;gap:10px;padding:2px 0"><span style="color:#8b949e;min-width:50px">'+ts+'</span><span style="color:#f85149;font-weight:600;min-width:50px">'+sym+'</span><span style="color:#d29922">'+reason+'</span></div>';
    }).join('');
  }).catch(()=>{});
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


@app.get("/api/vote-history")
@app.get("/api/veto-history")
async def api_veto_history(limit: int = 15):
    """Последние VETO-записи из vote_history."""
    vote_path = os.path.join(BASE_DIR, "data", "vote_history.json")
    if not os.path.exists(vote_path):
        return {"entries": [], "total": 0}
    try:
        with open(vote_path, "r") as f:
            history = json.load(f)
        # Фильтр: только veto_entries (approved=False, есть veto_reason)
        veto = [e for e in history if not e.get("approved", True) and e.get("veto_reason") and e.get("veto_reason", "").startswith("VETO")]
        vetol = veto[-limit:]
        return {"entries": list(reversed(vetol)), "total": len(vetol)}
    except Exception as e:
        log.error(f"veto-history: {e}")
        log.debug(traceback.format_exc())
        return {"error": "Internal server error", "entries": [], "total": 0}


@app.get("/api/vote-history")
async def api_vote_history(symbol: str = "", limit: int = 50):
    """История голосов DE."""
    vote_path = os.path.join(BASE_DIR, "data", "vote_history.json")
    if not os.path.exists(vote_path):
        return {"entries": [], "total": 0}
    try:
        with open(vote_path, "r") as f:
            history = json.load(f)
        if symbol:
            history = [e for e in history if e.get("symbol", "").startswith(symbol.upper())]
        history = history[-limit:]
        return {"entries": list(reversed(history)), "total": len(history)}
    except Exception as e:
        log.error(f"vote-history: {e}")
        log.debug(traceback.format_exc())
        return {"error": "Internal server error", "entries": [], "total": 0}


@app.get("/api/last-trade")
async def api_last_trade():
    """Последняя сделка (BUY/SELL) для алертов"""
    alert_path = "/tmp/trade_alert.json"
    if os.path.exists(alert_path):
        try:
            with open(alert_path, "r") as f:
                data = json.load(f)
            # Не стираем — дашборд сам решит когда обновить
            return data
        except Exception as _e:
            logger.debug("bare except in control_api: %s", _e)
            return {"symbol": None, "side": None, "price": 0, "timestamp": 0}
    return {"symbol": None, "side": None, "price": 0, "timestamp": 0}


# ════════════════════════════════════════════════════════════════════
# ⏯️ CONTROL API: управление трейдером (stop/start/sell-all/status)
# ════════════════════════════════════════════════════════════════════

_CONTROL_FILE = '/tmp/trading_control.json'
_control_ack = 0


def _write_control(mode: str = 'running', sell_all: bool = False) -> dict:
    global _control_ack
    _control_ack += 1
    payload = {
        'mode': mode,
        'sell_all': sell_all,
        'ack': _control_ack,
        'ts': datetime.now(timezone.utc).isoformat(),
    }
    try:
        os.makedirs(os.path.dirname(_CONTROL_FILE), exist_ok=True)
        _tmp = _CONTROL_FILE + '.tmp'
        with open(_tmp, 'w') as f:
            json.dump(payload, f)
        os.rename(_tmp, _CONTROL_FILE)
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
    return {'status': 'ok', 'mode': mode, 'ack': _control_ack}


def _read_control_status() -> dict:
    try:
        if os.path.exists(_CONTROL_FILE):
            with open(_CONTROL_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {'mode': 'running', 'sell_all': False, 'ack': 0}


@app.post("/api/control/stop")
async def api_control_stop():
    """Остановить новые входы. Текущие позиции не закрываются."""
    result = _write_control(mode='stopped')
    log.warning(f"[API] Торговля остановлена (stop)")
    return result


@app.post("/api/control/start")
async def api_control_start():
    """Возобновить входы в новые позиции."""
    result = _write_control(mode='running')
    log.info(f"[API] Торговля возобновлена (start)")
    return result


@app.post("/api/control/sell-all")
async def api_control_sell_all():
    """Продать все позиции и остановить торговлю."""
    result = _write_control(mode='stopped', sell_all=True)
    log.warning(f"[API] Sell-All активирован")
    return {'status': 'ok', 'mode': 'stopped', 'sell_all': True, 'ack': _control_ack}


@app.get("/api/control/status")
async def api_control_status():
    """Текущий статус управления трейдером."""
    status = _read_control_status()
    # Информация о PID, позициях и балансе
    pid = get_pid()
    running = pid is not None
    try:
        positions = db.get_all_positions()
        pos_count = len(positions)
    except Exception:
        positions = {}
        pos_count = 0
    try:
        balance_json = '/tmp/real_balance.json'
        if os.path.exists(balance_json):
            with open(balance_json) as f:
                balance_data = json.load(f)
        else:
            balance_data = {'total': 0, 'free_usdt': 0}
    except Exception:
        balance_data = {'total': 0, 'free_usdt': 0}
    return {
        'mode': status.get('mode', 'unknown'),
        'sell_all': status.get('sell_all', False),
        'ack': status.get('ack', 0),
        'trader_running': running,
        'positions_open': pos_count,
        'total_balance': balance_data.get('total', 0),
        'free_usdt': balance_data.get('free_usdt', 0),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


def run_server(host: str = "0.0.0.0", port: int = 8765):
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    # Если запущен с --standalone — пропускаем проверку оркестратора
    # (оркестратор сам решает когда запускать дашборд)
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--port', type=int, default=8765)
    _parser.add_argument('--standalone', action='store_true')
    _args, _ = _parser.parse_known_args()
    
    if not _args.standalone:
        # Проверка: если оркестратор жив — выходим
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        _clog = logging.getLogger('control_api.startup')
        try:
            with open(SYSTEM_PID_FILE) as f:
                pid = int(f.read().strip())
            os.kill(pid, 0)
            _clog.warning(f"⚠️ Оркестратор PID={pid} жив — дашборд запускается из оркестратора. Выход.")
            sys.exit(0)
        except (FileNotFoundError, ValueError, ProcessLookupError):
            _clog.info("🚀 Control API (standalone)")
    
    run_server(port=_args.port)
