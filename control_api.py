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
import subprocess
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
    # Ищем живой процесс super_trader.py через ps
    try:
        result = subprocess.run(['ps', 'ax', '-o', 'pid=,comm=,args='], capture_output=True, text=True, timeout=3)
        for line in result.stdout.strip().split('\n'):
            parts = line.strip().split(None, 2)
            if len(parts) < 3:
                continue
            pid_str, comm, args = parts
            if 'python' in comm and ('super_trader.py' in args or 'trader_entry.py' in args or 'industrial_trader.py' in args):
                pid = int(pid_str)
                os.kill(pid, 0)
                return pid
    except Exception:
        pass
    
    # Fallback: PID-файл (старый способ)
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
    """Возвращает JSON-снимок состояния, используя реальные данные с Bybit."""
    pid = get_pid()
    running = pid is not None

    # ─── ЖИВЫЕ ДАННЫЕ С БИРЖИ ───
    from bybit_live import fetch_wallet_balance, fetch_open_positions, fetch_recent_trades
    
    wallet = fetch_wallet_balance()
    bybit_positions = fetch_open_positions()
    
    balance_info = {
        'in_positions': round(float(wallet.get('in_positions', 0)), 2),
        'positions_count': int(wallet.get('positions_count', len(bybit_positions))),
        'free_usdt': round(float(wallet.get('free_usdt', 0)), 2),
        'total': round(float(wallet.get('total', 0)), 2),
    }
    
    # Если real_balance.json не заполнен — падаем на PG
    if not bybit_positions:
        try:
            pg_pos = db.get_all_positions()
            for sym, pos in pg_pos.items():
                q = float(pos['quantity'])
                if q > 0.0001:
                    bybit_positions[sym] = {
                        'qty': q,
                        'entry': float(pos['entry_price']),
                        'current_price': float(pos.get('highest_price', 0)),
                        'direction': 'LONG'
                    }
        except:
            pass

    # ─── ИСТОРИЯ ТРЕЙДОВ ───
    trades_history = []
    try:
        rows = db.get_trade_history(limit=50)
        trades_history = _calc_pnl_for_trades(rows)[:30]
    except Exception as e:
        log.error(f"trade-history: {e}")

    votes = _parse_votes_from_log()

    # ─── БАЛАНС С БИРЖИ (все монеты с ненулевым остатком) ───
    # Добавляем свежие покупки даже если их нет в PG
    enriched_positions = {}
    from data_cache import PriceCache
    pxc = PriceCache()
    
    for sym, pos in bybit_positions.items():
        cur_price = pxc.get_price(sym) or pos.get('current_price', 0) or 0
        p = {'qty': pos['qty'], 'entry': pos['entry'],
             'current_price': cur_price,
             'entry_time': '',
             'direction': pos['direction']}
        if sym in votes:
            p['votes'] = votes[sym]
        enriched_positions[sym] = p

    # Добавляем сигналы из трекера (без позиций)
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
            if enriched_positions[sym].get('qty', 0) == 0:
                enriched_positions[sym]['direction'] = v.get('direction', 'LONG')

    # BTC Regime: парсим из лога
    btc_regime = {}
    try:
        log_lines = read_log_tail(2000)
        for line in reversed(log_lines):
            rm = re.search(r'(?:BTC )?Regime\s*(?:[:=]|\s+)(\w+)', line)
            if rm:
                regime_name = rm.group(1).lower()
                rec_map = {
                    'pump': 'sell_only', 'dump': 'no_trade',
                    'distribution': 'sell_only', 'accumulation': 'buy_allowed',
                    'recovery': 'buy_priority', 'unknown': 'buy_allowed',
                }
                rec_match = re.search(r'rec=(\w+)', line)
                recommendation = rec_match.group(1) if rec_match else rec_map.get(regime_name, 'buy_allowed')
                btc_regime = {'regime': regime_name, 'recommendation': recommendation}
                break
    except:
        pass

    # BTC Direction
    btc_direction = {}
    try:
        log_lines = read_log_tail(1000)
        for line in reversed(log_lines):
            dm = re.search(r'BTC Direction: (\w+) \(conf=(\d+)%, strength=(\d+), up=(\d+)% down=(\d+)%', line)
            if dm:
                btc_direction = {
                    'direction': dm.group(1), 'confidence': int(dm.group(2)),
                    'strength': int(dm.group(3)),
                    'up_probability': int(dm.group(4)), 'down_probability': int(dm.group(5)),
                }
                break
    except:
        pass

    # BTC цена/RSI
    btc_analysis = {}
    try:
        log_lines = read_log_tail(1000)
        for line in reversed(log_lines):
            am = re.search(r'BTC/USDT: Цена=\, Тренд=(\w+), Уверенность=([\d.]+)%, RSI=([\d.]+)', line)
            if am:
                btc_analysis = {
                    'price': float(am.group(1)), 'trend': am.group(2),
                    'confidence': float(am.group(3)), 'rsi': float(am.group(4)),
                }
                break
    except:
        pass

    # Liquidity
    btc_liquidity = {}
    try:
        log_lines = read_log_tail(1000)
        for line in reversed(log_lines):
            if 'BTCLiq' in line and 'Liq:' in line:
                lm = re.search(r'Liq:(\d+)\(POC=([\d.]+) VAH=([\d.]+) VAL=([\d.]+) q=([\d.]+) fvg↑=(\d+) fvg↓=(\d+)', line)
                if lm:
                    btc_liquidity = {
                        'symbol': 'BTC', 'score': int(lm.group(1)),
                        'poc': float(lm.group(2)), 'vah': float(lm.group(3)),
                        'val': float(lm.group(4)), 'quality': float(lm.group(5)),
                        'fvg_above': int(lm.group(6)), 'fvg_below': int(lm.group(7)),
                        'source': 'btc direct',
                    }
                    if 'POC↑' in line:
                        btc_liquidity['poc_trend'] = 'up'
                    break
    except:
        pass

    # CVD
    cvd_status = {}
    try:
        log_lines = read_log_tail(500)
        for line in reversed(log_lines):
            if 'CVD:' in line:
                cm = re.search(r'CVD:(\d+\.?\d*).*Delta:([-\d]+\.?\d*)', line)
                if cm:
                    cvd_status = {
                        'cvd': float(cm.group(1)), 'delta': float(cm.group(2)),
                        'source': 'log'
                    }
                    # Парсим направления
                    for sym in list(enriched_positions.keys())[:6]:
                        if sym in line:
                            cvd_status[sym] = line
                    break
    except:
        pass

    # VoteTracker веса
    vote_tracker_weights = {'liquidity': 1.0}
    try:
        from vote_tracker import get_tracker as _vt
        wt = _vt()
        vote_tracker_weights = {
            'liquidity': wt.get('liquidity', 1.0),
        }
    except:
        pass

    return {
        'running': running,
        'pid': pid,
        'balance': balance_info,
        'positions': enriched_positions,
        'trades_history': trades_history,
        'trades': trades_history,  # алиас для совместимости HTML
        'pnl': {'total_pnl': 0, 'total_trades': len(trades_history), 'win_rate': 0},  # алиас для совместимости HTML
        'btc_regime': btc_regime,
        'btc_direction': btc_direction,
        'btc_analysis': btc_analysis,
        'btc_liquidity': btc_liquidity,
        'cvd': cvd_status,
        'vote_tracker_weights': vote_tracker_weights,
        'ts': time.time(),
    }

@app.get("/api/status")
async def api_status():
    return get_status_snapshot()


def _load_fees_from_exchange() -> dict:
    """Загружает все fee (buy+sell) по каждому трейду с биржи для расчёта net PnL.
    Возвращает {symbol: {'buy_fee': float, 'sell_fee': float}}"""
    rb_path = '/tmp/real_balance.json'
    result = {}
    if not os.path.exists(rb_path):
        return result
    try:
        with open(rb_path) as f:
            rb = json.load(f)
        all_trades = rb.get('all_buy_trades', []) + rb.get('recent_trades', [])
        # Дедуплицируем по (timestamp, side, price, amount)
        seen = set()
        for t in all_trades:
            sym = t['symbol']
            key = (t['timestamp'], t['side'], round(t['price'], 6), round(t['amount'], 6))
            if key in seen:
                continue
            seen.add(key)
            fee = t.get('fee', {})
            fee_cost = float(fee.get('cost', 0)) if isinstance(fee, dict) else 0
            if sym not in result:
                result[sym] = {'buy_fee': 0.0, 'sell_fee': 0.0}
            if t['side'] == 'buy':
                result[sym]['buy_fee'] += fee_cost
            elif t['side'] == 'sell':
                result[sym]['sell_fee'] += fee_cost
        return result
    except Exception as e:
        log.debug(f"_load_fees: {e}")
        return result


def _load_sell_prices_from_exchange() -> dict:
    """Загружает продажи с биржи (symbol→[(qty, price, ts)]).
    Используется для closed-сделок без exit_price в PG."""
    rb_path = '/tmp/real_balance.json'
    sym_sells = {}
    if not os.path.exists(rb_path):
        return sym_sells
    try:
        with open(rb_path) as f:
            rb = json.load(f)
        from collections import defaultdict
        sym_sells = defaultdict(list)
        for t in rb.get('recent_trades', []):
            if t.get('side') == 'sell':
                sym = t['symbol']
                sym_sells[sym].append({
                    'price': float(t['price']),
                    'qty': float(t['amount']),
                    'ts': t.get('datetime', ''),
                })
    except:
        pass
    return dict(sym_sells)


def _supplement_buys_from_exchange(trades: list):
    """Дополняет список трейдов buy-записями из real_balance.json (all_buy_trades).
    Добавляет для каждого символа, у которого нет buy в PG, все buy-трейды с биржи."""
    rb_path = '/tmp/real_balance.json'
    if not os.path.exists(rb_path):
        return
    try:
        with open(rb_path) as f:
            rb = json.load(f)
        all_buy = rb.get('all_buy_trades', [])
        if not all_buy:
            all_buy = [t for t in rb.get('recent_trades', []) if t.get('side') == 'buy']
        
        pg_buy_syms = set()
        for t in trades:
            if t['side'] == 'buy':
                pg_buy_syms.add(t['symbol'])
        
        from collections import defaultdict
        exchange_buys_by_sym = defaultdict(list)
        for et in all_buy:
            sym = et['symbol']
            exchange_buys_by_sym[sym].append(et)
        
        added = 0
        for sym, buy_list in exchange_buys_by_sym.items():
            if sym in pg_buy_syms:
                continue
            for et in buy_list:
                trades.append({
                    'id': 0,
                    'symbol': sym,
                    'side': 'buy',
                    'price': float(et['price']),
                    'qty': float(et['amount']),
                    'value': float(et['price']) * float(et['amount']),
                    'pnl': 0.0,
                    'pnl_pct': 0.0,
                    'fee': float(et.get('fee', {}).get('cost', 0)) if isinstance(et.get('fee'), dict) else 0,
                    'ts': et.get('datetime', '') or str(et.get('timestamp', 0)),
                })
                added += 1
        
        if added:
            log.info(f"_calc_pnl: добавлено {added} buy-трейдов с биржи для {len(exchange_buys_by_sym) - len(pg_buy_syms)} символов")
    except Exception as e:
        log.debug(f"_supplement_buys: {e}")


def _calc_pnl_for_trades(rows: list) -> list:
    """Сопоставляет sell с предыдущим buy того же символа и считает PnL на лету.
    Учитывает комиссии (fee) из биржевых данных."""
    # Сначала парсим все записи
    trades = []
    for r in rows:
        status = r.get('status', 'open')
        is_closed = status == 'closed'
        
        entry_price = float(r['entry_price']) if r.get('entry_price') else 0
        exit_price = float(r['exit_price']) if r.get('exit_price') else 0
        entry_qty = float(r['entry_qty']) if r.get('entry_qty') else 0
        exit_qty = float(r['exit_qty']) if r.get('exit_qty') else 0
        
        pg_exit_price = float(r['exit_price']) if r.get('exit_price') else 0
        
        if is_closed:
            side = 'sell'
            # Если exit_price не записан в PG — помечаем, чтобы PnL не считали
            if pg_exit_price > 0:
                price = pg_exit_price
            else:
                price = entry_price  # временно, заменится из exchange_sells
            qty = exit_qty if exit_qty else entry_qty
            ts = str(r.get('exit_time', '') or r.get('entry_time', ''))
        else:
            side = 'buy'
            price = entry_price
            qty = entry_qty
            ts = str(r.get('entry_time', ''))
        
        trades.append({
            'id': r['id'],
            'symbol': r['symbol'],
            'side': side,
            'price': price,
            'qty': qty,
            'value': price * qty,
            'pnl': 0.0,
            'pnl_pct': 0.0,
            'fee': 0.0,
            'ts': ts,
            '_pg_exit_price_missing': is_closed and pg_exit_price == 0,
        })
    
    # Сортируем по времени
    trades.sort(key=lambda t: t['ts'])
    
    # Для closed-сделок где exit_price не записан в PG — пытаемся найти sell-цену с биржи
    exchange_sells = _load_sell_prices_from_exchange()
    if exchange_sells:
        for t in trades:
            if t['side'] != 'sell' or not t.get('_pg_exit_price_missing'):
                continue
            sym = t['symbol']
            if sym in exchange_sells:
                sells = exchange_sells[sym]
                best = None
                for s in sells:
                    if best is None or abs(t['qty'] - s['qty']) < abs(t['qty'] - best['qty']):
                        best = s
                if best:
                    t['price'] = best['price']
                    t['value'] = t['price'] * t['qty']
                    t['_exchange_price'] = True
    
    # Дополняем buy-записи из real_balance.json (реальные трейды с биржи)
    _supplement_buys_from_exchange(trades)
    
    # Загружаем реальные fee с биржи
    exchange_fees = _load_fees_from_exchange()
    
    # Матчим sell → buy:
    # Сначала собираем среднюю цену покупки по каждому символу
    from collections import defaultdict
    buy_prices = defaultdict(list)  # symbol → [price, ...]
    
    for t in trades:
        if t['side'] == 'buy':
            buy_prices[t['symbol']].extend([t['price']] * int(max(1, t['qty'] * 100)))
    
    # Средняя цена по каждому символу
    avg_buy = {}
    for sym, prices in buy_prices.items():
        if prices:
            avg_buy[sym] = sum(prices) / len(prices)
    
    for t in trades:
        if t['side'] == 'sell':
            sym = t['symbol']
            # Если exit_price не был записан и с биржи не нашли — не считаем PnL
            if t.get('_pg_exit_price_missing') and not t.get('_exchange_price'):
                t['pnl'] = 0
                t['gross_pnl'] = 0
                continue
            if sym in avg_buy:
                buy_price = avg_buy[sym]
                gross_pnl = (t['price'] - buy_price) * t['qty']
                pnl_pct = ((t['price'] - buy_price) / buy_price) * 100 if buy_price > 0 else 0
                
                # Вычитаем комиссию
                # Buy fee уже учтён в средней цене (средняя включает все buy с учётом fee? нет)
                # Точнее: fee для sell = sell_price × qty × 0.001 (спот комиссия Bybit)
                sell_fee = t['value'] * 0.001
                # Для buy — fee уже реальная, берём из exchange_fees (доля от всех buy символа)
                sym_fees = exchange_fees.get(sym, {})
                all_buy_fee = sym_fees.get('buy_fee', 0)
                # Каждая продажа платит свою долю buy fee: пропорционально qty
                # Но у нас нет общего qty по символу. Используем упрощение: buy_fee на продажу = all_buy_fee / кол-во продаж символа
                # Ещё проще: 0.1% и на вход, и на выход — это стандарт для спота
                buy_fee_per_trade = t['value'] * 0.001
                total_fee = buy_fee_per_trade + sell_fee
                
                t['pnl'] = round(gross_pnl - total_fee, 2)
                t['pnl_pct'] = round(pnl_pct, 2)
                t['fee'] = round(total_fee, 4)
                t['gross_pnl'] = round(gross_pnl, 2)
    
    # Сортируем обратно — новые сверху
    trades.sort(key=lambda t: t['ts'], reverse=True)
    return trades


@app.get("/api/trade-history")
async def api_trade_history():
    """История всех сделок с PnL, отсортированная по времени."""
    try:
        rows = db.get_trade_history(limit=500)
        trades = _calc_pnl_for_trades(rows)
        return {'trades': trades}
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
          '<div style="font-size:9px;color:#8b949e;line-height:1.2">L='+Math.round(v.long_score||0)+' S='+Math.round(v.short_score||0)+'</div>'+
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
      return '<tr><td>'+em(t.symbol)+' '+t.symbol+'</td><td style="color:'+sideColor+'">'+t.side+'</td><td>'+fmtP(t.price)+'</td><td>'+fmtQ(t.qty)+'</td><td>'+fmtP(t.value||t.qty*t.price)+'</td><td style="color:'+pnlColor+'">'+(t.pnl!==0&&t.pnl!=null?fmtP(t.pnl):'—')+'</td><td style="color:'+pnlColor+'">'+(t.pnl_pct?(t.pnl_pct>=0?'+':'')+Number(t.pnl_pct).toFixed(2)+'%':'')+'</td><td style="color:#8b949e;font-size:11px">'+(t.ts||'').slice(0,19)+'</td></tr>';
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
