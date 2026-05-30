#!/usr/bin/env python3
"""
Загрузка реальных свечей из БД + биржи для ML-Pro v2 training_buffer.
Professional: сохраняем source of truth (свечи), а не готовые фичи.

Запуск: python3 load_ml_v2_candles.py
"""

import os, sys, time, pickle, logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
log = logging.getLogger("ML_V2_LOADER")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUFFER_PATH = os.path.join(BASE_DIR, 'models', 'ml_pro_v2_training.pkl')

def load_trades_from_db():
    """Загрузить закрытые сделки из БД за последние 7 дней."""
    import psycopg2
    conn = psycopg2.connect("host=/tmp dbname=trading user=ksysha")
    c = conn.cursor()
    cutoff = datetime.now() - timedelta(days=8)
    c.execute("""
        SELECT symbol, entry_time, exit_time, 
               COALESCE(CAST(pnl AS double precision), 0) as pnl,
               COALESCE(CAST(pnl_percent AS double precision), 0) as pnl_pct,
               status
        FROM trades
        WHERE entry_time >= %s AND status = 'closed'
        ORDER BY symbol, entry_time
    """, (cutoff,))
    rows = c.fetchall()
    conn.close()
    log.info(f"Загружено {len(rows)} сделок из БД")
    return rows

def fetch_candles_for_symbol(exchange, symbol):
    """Загрузить 1H и 4H свечи для символа."""
    import numpy as np
    try:
        raw_1h = exchange.fetch_ohlcv(symbol, '1h', limit=200)
        raw_4h = exchange.fetch_ohlcv(symbol, '4h', limit=55)
        
        if not raw_1h or not raw_4h:
            log.warning(f"{symbol}: пустой ответ")
            return None, None
        
        c1h = [{'o':c[1],'h':c[2],'l':c[3],'c':c[4],'v':c[5],'t':c[0]/1000} for c in raw_1h]
        c4h = [{'o':c[1],'h':c[2],'l':c[3],'c':c[4],'v':c[5],'t':c[0]/1000} for c in raw_4h]
        
        log.info(f"{symbol}: {len(c1h)}x1H / {len(c4h)}x4H загружено")
        return c1h, c4h
    except Exception as e:
        log.error(f"{symbol}: ошибка загрузки свечей: {e}")
        return None, None

def align_candles_to_trades(trades_by_symbol, symbol_candles):
    """Для каждой сделки найти свечи на момент входа, создать entries."""
    entries = []
    skipped = {'no_candles': 0, 'toast': 0, 'future': 0, 'too_short': 0}
    
    for sym, strades in trades_by_symbol.items():
        c1h, c4h = symbol_candles.get(sym, (None, None))
        if c1h is None or c4h is None:
            skipped['no_candles'] += len(strades)
            continue
        
        for s in strades:
            ets = s['entry_ts']
            last_candle_ts = c1h[-1]['t'] if c1h else 0
            
            # Свечи должны покрывать момент входа
            if ets > last_candle_ts + 7200:  # +2ч запас
                skipped['future'] += 1
                continue
            
            # Алигним свечи к моменту входа
            aligned_1h = [c for c in c1h if c['t'] <= ets][-200:]
            aligned_4h = [c for c in c4h if c['t'] <= ets][-55:]
            
            if len(aligned_1h) < 50 or len(aligned_4h) < 10:
                skipped['too_short'] += 1
                continue
            
            label = 1.0 if s['pnl'] > 0 else 0.0
            entries.append({
                'candles_1h': aligned_1h,
                'candles_4h': aligned_4h,
                'label': label,
                'symbol': sym,
                'ts': ets
            })
    
    for reason, count in skipped.items():
        if count:
            log.info(f"  Пропущено ({reason}): {count}")
    
    return entries

def restore_old_buffer():
    """Загрузить существующий training_buffer (чтобы не потерять уже накопленное)."""
    existing = []
    if os.path.exists(BUFFER_PATH):
        try:
            with open(BUFFER_PATH, 'rb') as f:
                loaded = pickle.load(f)
            if isinstance(loaded, list):
                if loaded and isinstance(loaded[0], dict):
                    existing = loaded
                    log.info(f"Загружен существующий буфер: {len(existing)} примеров")
                else:
                    log.info(f"Старый формат буфера ({len(loaded)} примеров) — пропускаем (коррумпированные данные)")
            else:
                log.info(f"Неизвестный формат буфера")
        except Exception as e:
            log.warning(f"Ошибка загрузки буфера: {e}")
    return existing

def main():
    import numpy as np
    import psycopg2
    import ccxt
    
    # 1. Загружаем существующий буфер
    existing = restore_old_buffer()
    existing_ts = {e['ts'] for e in existing}
    log.info(f"Уже есть: {len(existing)} записей")
    
    # 2. Загружаем сделки из БД
    trades_raw = load_trades_from_db()
    
    # Группируем по символам, фильтруем дубликаты
    from collections import defaultdict
    by_symbol = defaultdict(list)
    for r in trades_raw:
        sym = r[0]
        et = r[1]
        try:
            entry_ts = et.timestamp() if hasattr(et, 'timestamp') else datetime.fromisoformat(str(et)).timestamp()
        except:
            continue
        if any(abs(entry_ts - ts) < 10 for ts in existing_ts):
            continue
        by_symbol[sym].append({
            'entry_time': et, 'entry_ts': entry_ts,
            'pnl': float(r[3]), 'pnl_pct': float(r[4])
        })
    
    total_new = sum(len(v) for v in by_symbol.values())
    log.info(f"Новых сделок для обработки: {total_new} (из {len(trades_raw)} всего)")
    
    if total_new == 0:
        log.info(f"Новых данных нет. Буфер: {len(existing)}")
        return len(existing)
    
    # 3. Загружаем свечи с биржи
    log.info(f"Загрузка свечей для {len(by_symbol)} символов...")
    ex = ccxt.bybit()
    
    symbol_candles = {}
    for i, sym in enumerate(sorted(by_symbol.keys())):
        log.info(f"  [{i+1}/{len(by_symbol)}] {sym}...")
        c1h, c4h = fetch_candles_for_symbol(ex, sym)
        if c1h and c4h:
            symbol_candles[sym] = (c1h, c4h)
        time.sleep(0.3)  # rate limit
    
    # 4. Алигним и создаём entries
    new_entries = align_candles_to_trades(by_symbol, symbol_candles)
    log.info(f"Новых чистых записей: {len(new_entries)}")
    
    if not new_entries:
        log.info("Нет новых записей")
        return len(existing)
    
    # 5. Сохраняем
    full_buffer = existing + new_entries
    log.info(f"Добавлено: good={sum(1 for e in new_entries if e['label']>0.5)}, "
             f"bad={sum(1 for e in new_entries if e['label']<0.5)}")
    log.info(f"Буфер всего: {len(full_buffer)} "
             f"(good={sum(1 for e in full_buffer if e['label']>0.5)}, "
             f"bad={sum(1 for e in full_buffer if e['label']<0.5)})")
    
    os.makedirs(os.path.dirname(BUFFER_PATH), exist_ok=True)
    with open(BUFFER_PATH, 'wb') as f:
        pickle.dump(full_buffer, f)
    log.info(f"✅ Буфер сохранён: {BUFFER_PATH}")
    
    return len(full_buffer)

if __name__ == '__main__':
    n = main()
    print(f"\nИтого в буфере: {n} примеров")
