"""
PostgreSQL backend for the trading system.
Parallel implementation of db.py API using psycopg2.
Target: replace db.py after testing.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import psycopg2
import psycopg2.pool
import psycopg2.extras

# --- Конфигурация ---
PG_DSN = os.environ.get("PG_DSN", "dbname=trading user=ksysha host=/tmp")

# Пул соединений (thread-safe, 2-10 коннектов для main + control_api)
_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=2, maxconn=10, dsn=PG_DSN)
    return _pool


def _with_conn(fn, *args, **kwargs):
    """Выполняет функцию с соединением из пула."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        result = fn(conn, *args, **kwargs)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def _as_dict(cursor) -> List[Dict]:
    """Преобразует результаты в список словарей (RealDictCursor-style).
    Автоматически конвертирует Decimal → float, дата/время → str.
    """
    from decimal import Decimal
    if not cursor.description:
        return []
    cols = [d[0] for d in cursor.description]
    result = []
    for row in cursor.fetchall():
        d = dict(zip(cols, row))
        # Конвертируем Decimal → float для безопасной JSON-сериализации
        for k, v in d.items():
            if isinstance(v, Decimal):
                d[k] = float(v)
            elif isinstance(v, datetime):
                d[k] = v.isoformat()
        result.append(d)
    return result


def import_partitions_create(cur, conn):
    """Создать партиции trades на текущий + 2 месяца вперёд, если их нет."""
    from datetime import datetime, timedelta
    now = datetime.now()
    for offset in [0, 1, 2]:
        dt = now.replace(day=1) + timedelta(days=offset * 32)
        dt = dt.replace(day=1)
        ym = dt.strftime("%Y_%m")
        part_name = "trades_" + ym
        cur.execute("SELECT EXISTS (SELECT FROM pg_class WHERE relname = %s)", (part_name,))
        if cur.fetchone()[0]:
            continue
        if dt.month == 12:
            next_dt = dt.replace(year=dt.year + 1, month=1)
        else:
            next_dt = dt.replace(month=dt.month + 1)
        from_bound = dt.strftime("%Y-%m-%d")
        to_bound = next_dt.strftime("%Y-%m-%d")
        try:
            sql = "CREATE TABLE " + part_name + " PARTITION OF trades FOR VALUES FROM (%s) TO (%s)"
            cur.execute(sql, (from_bound, to_bound))
            conn.commit()
            print("[db_pg] OK partition " + part_name)
        except Exception as e:
            if "already exists" not in str(e):
                print("[db_pg] WARN " + part_name + ": " + str(e))


# ========================
# ИНИЦИАЛИЗАЦИЯ
# ========================

def init_db(db_path=None):
    """
    Создаёт таблицы, если их нет.
    db_path игнорируется — используется PG_DSN.
    Вызывается при старте main.py / control_api.py.
    """
    def _init(conn):
        cur = conn.cursor()

        # Проверяем, есть ли уже схема
        cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name='trades')")
        if cur.fetchone()[0]:
            # Схема есть — только обновляем партиции
            print("[db_pg] Схема уже создана")
            import_partitions_create(cur, conn)
            return

        # Создаём из schema.sql
        schema_path = os.path.join(os.path.dirname(__file__), "db", "schema.sql")
        if os.path.exists(schema_path):
            with open(schema_path) as f:
                cur.execute(f.read())
            conn.commit()
            print("[db_pg] Схема создана из schema.sql")
        else:
            print(f"[db_pg] ❌ schema.sql не найден в {schema_path}")

        # Создаём партиции (текущий + 2 месяца вперёд)
        import_partitions_create(cur, conn)

    return _with_conn(_init)


def ensure_future_partitions():
    """Создаёт партиции trades на текущий + 2 месяца вперёд, если их нет.
    Вызывается при старте системы."""
    def _ensure(conn):
        cur = conn.cursor()
        import_partitions_create(cur, conn)
    return _with_conn(_ensure)


# ========================
# СДЕЛКИ (trades)
# ========================

def add_trade(symbol: str, side: str, price: float, quantity: float,
              pnl: float = 0.0, pnl_pct: float = 0.0,
              exit_reason: str = None, order_id: str = None, exchange_id: str = None,
              ts: str = None, db_path=None,
              scores: Optional[Dict] = None) -> int:
    """
    Добавляет сделку в таблицу trades.
    Аналог add_trade из db.py, но для новой схемы.
    scores: dict с ключами ml_pro, advisor, mtf, vsa, liquidity, rsi_vol_btc, volume_vwap, final_score
    """
    def _add(conn):
        cur = conn.cursor()

        # Дедупликация: проверяем notes на наличие exchange_id
        if exchange_id:
            cur.execute("SELECT id FROM trades WHERE notes LIKE %s", (f"%exch_id={exchange_id}%",))
            existing = cur.fetchone()
            if existing:
                return existing[0]

        _ts = ts or datetime.now(timezone.utc).isoformat()
        _exit_ts = _ts  # сохраняем реальное время выхода

        entry_price = price if side == 'buy' else None
        exit_price = price if side == 'sell' else None
        status = 'open' if side == 'buy' else 'closed'

        # 🩹 Для sell находим entry_price/entry_time из buy-записи (партиция trades_2026_05 NOT NULL)
        if side == 'sell' and entry_price is None:
            try:
                cur.execute("SELECT entry_price, entry_time FROM trades WHERE symbol=%s AND side='long' AND entry_price IS NOT NULL ORDER BY id DESC LIMIT 1", (symbol,))
                row = cur.fetchone()
                if row:
                    entry_price = float(row[0]) if hasattr(row[0], 'item') else float(row[0])
                    if len(row) > 1 and row[1]:
                        _ts = str(row[1])
            except Exception:
                entry_price = 0.0

        notes_val = None
        if order_id or exchange_id:
            parts = []
            if order_id:
                parts.append(f"order_id={order_id}")
            if exchange_id:
                parts.append(f"exch_id={exchange_id}")
            notes_val = ", ".join(parts)

        # Извлекаем скоринг из scores (только для buy)
        s_ml = None
        s_adv = None
        s_mtf = None
        s_vsa = None
        s_liq = None
        s_rvb = None
        s_vv = None
        s_entry = None
        if scores and side == 'buy':
            # Поддержка двух форматов: плоский (signal_log) и вложенный (entry_checks)
            s_ml = scores.get('ml_pro') or (scores.get('ml_v2') or {}).get('score')
            s_adv = scores.get('advisor') or (scores.get('advisor') or {}).get('score')
            s_mtf = scores.get('mtf') or (scores.get('mtf') or {}).get('score')
            s_vsa = scores.get('vsa') or (scores.get('vsa') or {}).get('score')
            s_liq = scores.get('liquidity') or (scores.get('liquidity') or {}).get('score')
            s_rvb = scores.get('rsi_vol_btc') or (scores.get('rsi_vol_btc') or {}).get('score')
            s_vv = scores.get('volume_vwap') or (scores.get('volume_vwap') or {}).get('score')
            s_entry = scores.get('final_score')

        # 🩹 Приводим numpy-типы к float (psycopg2 не умеет в np.float64)
        def _to_float(v): return float(v) if v is not None else None
        
        s_ml = _to_float(s_ml)
        s_adv = _to_float(s_adv)
        s_mtf = _to_float(s_mtf)
        s_vsa = _to_float(s_vsa)
        s_liq = _to_float(s_liq)
        s_rvb = _to_float(s_rvb)
        s_vv = _to_float(s_vv)
        s_entry = _to_float(s_entry)

        cur.execute("""
            INSERT INTO trades
                (symbol, side, status, account_id,
                 entry_price, entry_qty, entry_time,
                 exit_price, exit_qty, exit_time,
                 pnl, pnl_percent,
                 exit_reason,
                 notes,
                 entry_score, ml_pro, adv_score, mtf_score,
                 vsa_score, liq_score, rvb_score, vv_score,
                 created_at, updated_at)
            VALUES (%s, %s, %s, 1,
                    %s, %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s,
                    %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    NOW(), NOW())
            RETURNING id
        """, (
            symbol, _convert_side(side), status,
            entry_price, quantity, _ts,
            exit_price, quantity, _exit_ts if side == 'sell' else None,
            pnl, pnl_pct if pnl_pct else None,
            exit_reason if side == 'sell' else None,
            notes_val,
            s_entry, s_ml, s_adv, s_mtf,
            s_vsa, s_liq, s_rvb, s_vv,
        ))
        return cur.fetchone()[0]

    return _with_conn(_add)


def close_trade(symbol: str, exit_price: float, exit_qty: float,
                pnl: float = 0.0, pnl_pct: float = 0.0,
                exit_reason: str = 'manual', order_id: str = None,
                db_path=None):
    """
    Закрывает открытую позицию по символу.
    Обновляет последний open trade: exit_price, exit_time, pnl.
    """
    def _close(conn):
        cur = conn.cursor()
        cur.execute("""
            SELECT id, entry_price, entry_qty FROM trades
            WHERE symbol = %s AND status = 'open'
            ORDER BY entry_time DESC LIMIT 1
        """, (symbol,))
        row = cur.fetchone()
        if not row:
            print(f"[db_pg] close_trade: no open position {symbol}")
            return None

        trade_id = row[0]
        notes_val = f"order_id={order_id}" if order_id else None

        cur.execute("""
            UPDATE trades SET
                status = 'closed',
                exit_price = %s,
                exit_qty = %s,
                exit_time = NOW(),
                exit_reason = %s,
                pnl = %s,
                pnl_percent = %s,
                notes = COALESCE(notes, '') || %s,
                updated_at = NOW()
            WHERE id = %s
        """, (exit_price, exit_qty, exit_reason,
              pnl, pnl_pct if pnl_pct else None,
              f"; {notes_val}" if notes_val else '',
              trade_id))
        return trade_id

    return _with_conn(_close)


# Несовместимость: db.py.add_trade принимает side='buy'|'sell',
# а новая схема trades ожидает side='long'. Сохраняем API.
def _convert_side(side: str) -> str:
    if side in ('sell', 'close'):
        return 'long'
    return 'long'


# ========================
# ПОЗИЦИИ (через trades c status='open')
# ========================

def get_all_positions(db_path=None) -> Dict[str, Dict]:
    """Возвращает {symbol: position_dict} из открытых trades."""
    def _get(conn):
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT ON (symbol)
                symbol,
                entry_price,
                entry_qty AS quantity,
                GREATEST(entry_price, COALESCE(exit_price, 0)) AS highest_price,
                entry_time,
                updated_at
            FROM trades
            WHERE status = 'open'
            ORDER BY symbol, entry_time DESC
        """)
        result = {}
        for row in cur.fetchall():
            result[row[0]] = {
                'symbol': row[0],
                'entry_price': float(row[1]),
                'quantity': float(row[2]),
                'highest_price': float(row[3]),
                'entry_time': str(row[4]) if row[4] else '',
                'updated_at': str(row[5]) if row[5] else '',
            }
        return result

    return _with_conn(_get)


def get_position(symbol: str, db_path=None) -> Optional[Dict]:
    """Возвращает позицию по символу."""
    def _get(conn):
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM trades
            WHERE symbol = %s AND status = 'open'
            ORDER BY entry_time DESC LIMIT 1
        """, (symbol,))
        row = cur.fetchone()
        if row:
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))
        return None

    return _with_conn(_get)


def upsert_position(symbol: str, entry_price: float, quantity: float,
                    highest_price: float = 0.0, entry_time: str = '',
                    db_path=None):
    """Создаёт или обновляет позицию."""
    def _upsert(conn):
        cur = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        ts = entry_time or now

        # Проверяем, есть ли открытая позиция
        cur.execute(
            "SELECT id, entry_price, entry_qty FROM trades WHERE symbol = %s AND status = 'open' LIMIT 1",
            (symbol,))
        existing = cur.fetchone()

        if existing:
            cur.execute("""
                UPDATE trades SET
                    entry_price = %s,
                    entry_qty = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (entry_price, quantity, existing[0]))
        else:
            cur.execute("""
                INSERT INTO trades
                    (symbol, side, status, account_id,
                     entry_price, entry_qty, entry_time,
                     created_at, updated_at)
                VALUES (%s, 'long', 'open', 1,
                        %s, %s, %s,
                        NOW(), NOW())
            """, (symbol, entry_price, quantity, ts))

    return _with_conn(_upsert)


def remove_position(symbol: str, db_path=None) -> Optional[Dict]:
    """Закрывает позицию (меняет status='closed'). Возвращает данные."""
    def _remove(conn):
        cur = conn.cursor()
        pos = get_position(symbol)
        if pos:
            cur.execute("""
                UPDATE trades SET status='closed', updated_at=NOW()
                WHERE id = %s
            """, (pos['id'],))
        return pos

    return _with_conn(_remove)


# ========================
# ИСТОРИЯ СДЕЛОК
# ========================

def get_trade_history(symbol: str = None, limit: int = 50,
                      side: str = None, db_path=None) -> List[Dict]:
    """Возвращает историю trades."""
    def _get(conn):
        cur = conn.cursor()
        conditions = []
        params = []

        if symbol:
            conditions.append("t.symbol = %s")
            params.append(symbol)
        if side:
            pg_side = _convert_side(side)
            conditions.append("t.side = %s")
            params.append(pg_side)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions) + " AND (t.exit_reason IS NULL OR t.exit_reason != 'stale_cleanup')"
        else:
            where = "WHERE (t.exit_reason IS NULL OR t.exit_reason != 'stale_cleanup')"

        cur.execute(f"""
            SELECT t.*, a.name as account_name
            FROM trades t
            LEFT JOIN accounts a ON a.id = t.account_id
            {where}
            ORDER BY COALESCE(t.exit_time, t.entry_time) DESC
            LIMIT %s
        """, params + [limit])

        return _as_dict(cur)

    return _with_conn(_get)


def get_position_trades(symbol: str, db_path=None) -> List[Dict]:
    """Все сделки по символу (старые → новые)."""
    def _get(conn):
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM trades
            WHERE symbol = %s
            ORDER BY entry_time ASC
        """, (symbol,))
        return _as_dict(cur)

    return _with_conn(_get)


def calculate_weighted_entry(symbol: str, db_path=None) -> Tuple[float, float]:
    """
    Weighted average entry price из открытых long-позиций.
    Аналог db.py.calculate_weighted_entry.
    """
    def _calc(conn):
        cur = conn.cursor()
        # Сумма всех entry_qty по открытым позициям
        cur.execute("""
            SELECT
                SUM(entry_qty) AS total_qty,
                SUM(entry_price * entry_qty) AS total_cost
            FROM trades
            WHERE symbol = %s AND status = 'open'
        """, (symbol,))
        row = cur.fetchone()
        if row and row[0] and row[0] > 0:
            return (float(row[1] / row[0]), float(row[0]))
        return (0.0, 0.0)

    return _with_conn(_calc)


# ========================
# ОРДЕРА
# ========================

def upsert_order(order_id: str, symbol: str, side: str, price: float,
                 quantity: float, filled_qty: float = 0.0,
                 status: str = 'open', db_path=None):
    """Создаёт или обновляет ордер."""
    def _upsert(conn):
        cur = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cur.execute("""
            INSERT INTO orders (order_id, symbol, side, price, quantity, filled_qty, status, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (order_id) DO UPDATE SET
                filled_qty = EXCLUDED.filled_qty,
                status = EXCLUDED.status,
                updated_at = EXCLUDED.updated_at
        """, (order_id, symbol, side, price, quantity, filled_qty, status, now, now))

    return _with_conn(_upsert)


def sync_orders_from_exchange(exchange, db_path=None):
    """Синхронизирует открытые ордера с биржи."""
    import logging
    logger = logging.getLogger('db_pg')
    count = 0
    try:
        open_orders = exchange.fetch_open_orders()
        for order in open_orders:
            order_id = str(order.get('id', ''))
            if not order_id:
                continue
            upsert_order(
                order_id,
                order.get('symbol', ''),
                order.get('side', 'buy'),
                float(order.get('price', 0) or 0),
                float(order.get('amount', 0) or 0),
                float(order.get('filled', 0) or 0),
                order.get('status', 'open'),
                db_path,
            )
            count += 1
        if count > 0:
            logger.info(f"[db_pg] Синхронизировано {count} ордеров с биржи")
    except Exception as e:
        logger.warning(f"[db_pg] Ошибка синхронизации ордеров: {e}")
    return count


# ========================
# БАЛАНС
# ========================

def save_capital_snapshot(total: float, positions_value: float,
                          free_usdt: float, positions_count: int,
                          db_path=None):
    """Сохраняет снэпшот капитала."""
    def _save(conn):
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO balance_history
                (account_id, balance, free, recorded_at)
            VALUES (1, %s, %s, NOW())
        """, (total, free_usdt))

    return _with_conn(_save)


def get_capital_history(limit: int = 1000, db_path=None) -> List[Dict]:
    """Возвращает историю капитала для графика."""
    def _get(conn):
        cur = conn.cursor()
        cur.execute("""
            SELECT
                id,
                balance AS total,
                0 AS positions_value,
                COALESCE(free, 0) AS free_usdt,
                0 AS positions_count,
                recorded_at AS created_at
            FROM balance_history
            ORDER BY recorded_at DESC
            LIMIT %s
        """, (limit,))
        return _as_dict(cur)

    return _with_conn(_get)


# ========================
# PNL СТАТИСТИКА
# ========================

def get_pnl_stats(db_path=None) -> Dict:
    """Статистика по закрытым сделкам."""
    def _get(conn):
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COALESCE(SUM(pnl), 0) AS total_pnl,
                COUNT(*) AS total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS win_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) AS loss_trades
            FROM trades
            WHERE status = 'closed'
              AND (exit_reason IS NULL OR exit_reason != 'stale_cleanup')
        """)
        row = cur.fetchone()
        if not row or row[1] == 0:
            return {'total_pnl': 0.0, 'total_trades': 0, 'win_trades': 0, 'loss_trades': 0, 'win_rate': 0.0}

        total = int(row[1])
        wins = int(row[2] or 0)
        return {
            'total_pnl': float(round(row[0], 2)),
            'total_trades': total,
            'win_trades': wins,
            'loss_trades': int(row[3] or 0),
            'win_rate': float(round(wins / total * 100, 1)),
        }

    return _with_conn(_get)


# ========================
# META
# ========================

def set_meta(key: str, value: str, db_path=None):
    """Сохраняет мета-данные."""
    def _set(conn):
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO meta (key, value) VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """, (key, str(value)))

    return _with_conn(_set)


def get_meta(key: str, default=None, db_path=None):
    """Читает мета-данные."""
    def _get(conn):
        cur = conn.cursor()
        cur.execute("SELECT value FROM meta WHERE key = %s", (key,))
        row = cur.fetchone()
        return row[0] if row else default

    return _with_conn(_get)


# ========================
# СИНХРОНИЗАЦИЯ ПОЗИЦИЙ С БИРЖЕЙ
# ========================

def sync_positions_from_exchange(exchange, enabled_pairs: List[str],
                                  db_path=None) -> Dict[str, Dict]:
    """
    Синхронизирует позиции с биржей (аналог db.py).
    """
    import ccxt

    def _sync(conn):
        cur = conn.cursor()
        try:
            balance = exchange.fetch_balance()
        except Exception as e:
            print(f"[db_pg] Ошибка fetch_balance: {e}")
            return get_all_positions(db_path)

        active_positions = {}
        now = datetime.now(timezone.utc).isoformat()

        # 🩹 Загружаем цены bulk для всех пар перед циклом
        ticker_prices = {}
        try:
            tickers = exchange.fetch_tickers()
            for pair in enabled_pairs:
                t = tickers.get(pair, {})
                ticker_prices[pair] = t.get('last') or t.get('ask') or 0
        except Exception:
            # fallback: по одному
            pass

        for pair in enabled_pairs:
            base = pair.split('/')[0]
            free = balance.get(base, {}).get('free', 0)
            used = balance.get(base, {}).get('used', 0)
            total = free + used

            # Текущая цена (из bulk если есть, иначе fetch)
            current_price = ticker_prices.get(pair, 0)
            if current_price == 0:
                try:
                    ticker = exchange.fetch_ticker(pair)
                    current_price = ticker.get('last') or ticker.get('ask') or 0
                except Exception:
                    current_price = 0

            # 🛡️ SHORT GLOBAL GUARD: если в БД есть открытый SHORT для этого символа — не трогать
            _cur2 = conn.cursor()
            _cur2.execute("SELECT id, pos_meta->>'side' FROM trades WHERE symbol=%s AND status='open' LIMIT 1", (pair,))
            _existing = _cur2.fetchone()
            _cur2.close()
            if _existing and _existing[1] and _existing[1].lower() == 'short':
                # Уже есть открытый SHORT — пропускаем, не перезаписываем
                logger.debug(f"🛡️ {pair}: пропускаю sync, активный SHORT id={_existing[0]} в БД")
                continue

            if total <= 0:
                # 🛡️ SHORT: пропускаем (нулевой/отрицательный баланс — норма для шорта)
                cur.execute("SELECT pos_meta->>'side' FROM trades WHERE symbol=%s AND status='open' LIMIT 1", (pair,))
                _side_row = cur.fetchone()
                if _side_row and _side_row[0] and _side_row[0].lower() == 'short':
                    # Шорт — не закрываем, баланс заимствованных монет ~0
                    continue
                # 🩹 FIX: закрываем с текущей ценой и PnL
                if current_price > 0:
                    cur.execute("""
                        UPDATE trades SET
                            status='closed',
                            exit_price=%s,
                            exit_qty=entry_qty,
                            exit_time=NOW(),
                            pnl=(%s - entry_price) * entry_qty,
                            pnl_percent=((%s - entry_price) / NULLIF(entry_price, 0)) * 100,
                            exit_reason='exchange_sold',
                            updated_at=NOW()
                        WHERE symbol=%s AND status='open'
                    """, (current_price, current_price, current_price, pair))
                else:
                    cur.execute(
                        "UPDATE trades SET status='closed', exit_time=NOW(), updated_at=NOW() WHERE symbol=%s AND status='open'",
                        (pair,))
                continue

            # 🩹 FIX: Пропускаем активы стоимостью < $1 (пылевые остатки)
            # 🛡️ SHORT: пропускаем (нулевой/отрицательный баланс — норма для шорта)
            import logging as _lg
            if current_price > 0 and total * current_price < 1.0:
                # Проверяем, не шорт ли это
                cur.execute("SELECT pos_meta->>'side' FROM trades WHERE symbol=%s AND status='open' LIMIT 1", (pair,))
                _side_row2 = cur.fetchone()
                if _side_row2 and _side_row2[0] and _side_row2[0].lower() == 'short':
                    # Шорт — не чистим, баланс ~0 это нормально
                    continue
                cur.execute("""
                    UPDATE trades SET
                        status='closed',
                        exit_price=%s,
                        exit_qty=entry_qty,
                        exit_time=NOW(),
                        pnl=(%s - entry_price) * entry_qty,
                        pnl_percent=((%s - entry_price) / NULLIF(entry_price, 0)) * 100,
                        exit_reason='dust_cleanup',
                        updated_at=NOW()
                    WHERE symbol=%s AND status='open'
                """, (current_price, current_price, current_price, pair))
                if cur.rowcount > 0:
                    _lg.getLogger('db_pg').info(f"🧹 {pair}: пыль (${total * current_price:.2f} < $1), закрыта")
                else:
                    _lg.getLogger('db_pg').debug(f"🧹 {pair}: пыль (${total * current_price:.2f}) — уже закрыта")

                continue

            entry_price, _ = calculate_weighted_entry(pair)
            entry_price = float(entry_price)

            # Если нет entry_price — берём текущую
            if entry_price == 0:
                entry_price = current_price

            # Проверяем существующую позицию в PG
            cur.execute(
                "SELECT id, entry_price, entry_qty FROM trades WHERE symbol=%s AND status='open' LIMIT 1",
                (pair,))
            existing = cur.fetchone()

            if existing:
                old_cost = float(existing[1] * existing[2])
                new_cost = entry_price * total
                if total > 0 and entry_price > 0:
                    entry_price = (old_cost + new_cost * 0.001) / (float(existing[2]) + total * 0.001)
                else:
                    entry_price = float(existing[1])

            upsert_position(pair, entry_price, total, current_price, str(now))
            active_positions[pair] = {
                'symbol': pair,
                'entry_price': entry_price,
                'quantity': total,
                'highest_price': current_price,
                'entry_time': now,
            }

        return active_positions

    return _with_conn(_sync)


# ========================
# МИГРАЦИЯ ПОЗИЦИЙ
# ========================

def _migrate_legacy_trades_to_history(db_path=None):
    """
    Заглушка. Миграция уже выполнена скриптом migrate_from_sqlite.py.
    """
    print("[db_pg] Миграция legacy позиций: пропускаем (уже выполнена)")

# ========================
# ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ (для совместимости с industrial_trader.py)
# ========================

# Константа для обратной совместимости
DB_PATH = PG_DSN

def get_db_path():
    """Возвращает строку подключения (аналог db.DB_PATH)."""
    return PG_DSN


def get_daily_stats(day_start: str) -> tuple:
    """
    Дневная статистика: (total_pnl, total_trades, wins, losses).
    Аналог сырого SQLite-запроса из industrial_trader.py.
    """
    def _get(conn):
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COALESCE(SUM(pnl), 0),
                COUNT(*),
                COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END), 0)
            FROM trades
            WHERE status = 'closed'
              AND exit_time >= %s::timestamptz
              AND ABS(pnl) > 0.0001
        """, (day_start,))
        row = cur.fetchone()
        return (float(row[0]), row[1], row[2], row[3])
    return _with_conn(_get)


def get_total_position_value() -> float:
    """Суммарная стоимость открытых позиций."""
    def _get(conn):
        cur = conn.cursor()
        cur.execute("""
            SELECT COALESCE(SUM(entry_price * entry_qty), 0)
            FROM trades WHERE status = 'open'
        """)
        return float(cur.fetchone()[0])
    return _with_conn(_get)


def get_recent_pnls(limit: int = 20) -> list:
    """Последние N значений PnL (для расчёта consecutive_profits)."""
    def _get(conn):
        cur = conn.cursor()
        cur.execute("""
            SELECT pnl FROM trades
            WHERE status = 'closed' AND ABS(pnl) > 0.0001
            ORDER BY exit_time DESC LIMIT %s
        """, (limit,))
        return [float(r[0]) for r in cur.fetchall()]
    return _with_conn(_get)


def log_signal(symbol: str, decision: str, score: float = None, threshold: float = None,
               components: dict = None, db_path=None):
    """
    Логирует решение торгового движка в signal_log.
    decision: 'hold' | 'buy' | 'sell'
    components: словарь со скорами всех модулей (ml_pro, advisor, mtf, rvb, liq, vv, vsa)
    """
    def _log(conn):
        cur = conn.cursor()
        import json
        cur.execute(
            """INSERT INTO signal_log (symbol, decision, score, threshold, components)
               VALUES (%s, %s, %s, %s, %s::jsonb)""",
            (symbol, decision, score, threshold,
             json.dumps(components) if components else None)
        )
        conn.commit()
    try:
        _with_conn(_log)
    except Exception as e:
        logger.debug(f"[db_pg] log_signal error: {e}")


def position_exists(symbol: str) -> bool:
    """Проверяет, есть ли открытая позиция по символу."""
    def _get(conn):
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM trades WHERE symbol=%s AND status='open'",
            (symbol,))
        return cur.fetchone()[0] > 0
    return _with_conn(_get)


def update_pos_meta(symbol: str, meta: dict) -> None:
    """Сохраняет мета-данные открытой позиции (SL/TP/пик/трейлинг и т.д.)"""
    def _upd(conn):
        cur = conn.cursor()
        cur.execute(
            "UPDATE trades SET pos_meta = %s::jsonb, updated_at = NOW() WHERE symbol=%s AND status='open'",
            (json.dumps(meta), symbol))
        conn.commit()
    _with_conn(_upd)


def save_all_positions(positions: dict) -> None:
    """Пакетное сохранение всех открытых позиций с мета-данными."""
    def _save(conn):
        cur = conn.cursor()
        for symbol, pos in positions.items():
            meta = {
                'max_profit': pos.get('max_profit', 0),
                '_highest_price': pos.get('_highest_price', 0),
                '_created_at': pos.get('_created_at', 0),
                '_sl_price': pos.get('_sl_price'),
                '_tp_price': pos.get('_tp_price'),
                '_trail_act': pos.get('_trail_act'),
                '_trail_dist': pos.get('_trail_dist'),
                '_max_hold_h': pos.get('_max_hold_h'),
                '_early_trail_peak': pos.get('_early_trail_peak'),
                'smart_exit_count': pos.get('smart_exit_count', 0),
                'entry_time': pos.get('entry_time', ''),
                'entry_price': pos.get('entry_price', 0),
                'quantity': pos.get('quantity', 0),
                'side': pos.get('direction', pos.get('side', 'long')),
            }
            cur.execute(
                "UPDATE trades SET pos_meta = %s::jsonb, updated_at = NOW() WHERE symbol=%s AND status='open'",
                (json.dumps(meta), symbol))
        conn.commit()
    _with_conn(_save)


def load_open_positions() -> dict:
    """Загружает все открытые позиции с полными мета-данными из pos_meta."""
    def _load(conn):
        cur = conn.cursor()
        cur.execute(
            "SELECT symbol, pos_meta, entry_price, entry_qty, entry_time FROM trades WHERE status='open'")
        result = {}
        for r in cur.fetchall():
            symbol = r[0]
            meta_raw = r[1] or {}
            if isinstance(meta_raw, str):
                meta = json.loads(meta_raw)
            else:
                meta = meta_raw
            entry_price = float(r[2]) if r[2] else (meta.get('entry_price') or 0)
            quantity = float(r[3]) if r[3] else (meta.get('quantity') or 0)
            entry_time = meta.get('entry_time', str(r[4]) if r[4] else '')
            position = {
                'quantity': quantity,
                'entry_price': entry_price,
                'entry_time': entry_time,
                'side': meta.get('side', 'long').lower(),
                'max_profit': meta.get('max_profit', 0),
                '_highest_price': meta.get('_highest_price', entry_price),
                '_created_at': meta.get('_created_at', time.time()),
                '_sl_price': meta.get('_sl_price'),
                '_tp_price': meta.get('_tp_price'),
                '_trail_act': meta.get('_trail_act'),
                '_trail_dist': meta.get('_trail_dist'),
                '_max_hold_h': meta.get('_max_hold_h'),
                '_early_trail_peak': meta.get('_early_trail_peak'),
                'smart_exit_count': meta.get('smart_exit_count', 0),
            }
            # Очищаем None значения (не перезаписываем конфиг умолчаниями)
            position = {k: v for k, v in position.items() if v is not None}
            result[symbol] = position
        return result
    return _with_conn(_load)


def get_db_stats() -> dict:
    """Статистика базы данных (для control_api)."""
    def _get(conn):
        cur = conn.cursor()
        # Total trades
        cur.execute("SELECT COUNT(*) FROM trades")
        total = cur.fetchone()[0]
        # Open positions
        cur.execute("SELECT COUNT(*) FROM trades WHERE status='open'")
        opened = cur.fetchone()[0]
        # Closed trades
        cur.execute("SELECT COUNT(*) FROM trades WHERE status='closed'")
        closed = cur.fetchone()[0]
        # Balance snapshots
        cur.execute("SELECT COUNT(*) FROM balance_history")
        snapshots = cur.fetchone()[0]
        # DB size estimate
        cur.execute("""
            SELECT pg_database_size(current_database()) / 1048576.0
        """)
        size_mb = float(cur.fetchone()[0])

        return {
            'total_trades': total,
            'open_positions': opened,
            'closed_trades': closed,
            'capital_snapshots': snapshots,
            'db_size_mb': round(size_mb, 1),
            'engine': 'postgresql',
            'db_path': PG_DSN,
        }
    return _with_conn(_get)

def get_all_active_position_values() -> list:
    """Все стоимости активных позиций (entry_price * entry_qty >= 1)."""
    def _get(conn):
        cur = conn.cursor()
        cur.execute("""
            SELECT entry_price * entry_qty FROM trades
            WHERE status = 'open' AND entry_price * entry_qty >= 1
        """)
        return [float(r[0]) for r in cur.fetchall()]
    return _with_conn(_get)
