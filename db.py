"""
Database module for the trading system.
SQLite with WAL mode for concurrent reads.
Provides position, trade, order, and capital snapshot management.
"""

import sqlite3
import os
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

# Путь к БД по умолчанию
DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, 'data', 'trading.db')

# Ручка соединения (thread-local через функции, не глобальный объект)
_connection = None

# === Вспомогательные ===

def get_db_path(db_path=None):
    return db_path or DB_PATH

def _get_connection(db_path=None):
    """Возвращает соединение с БД (одно на процесс для WAL)"""
    global _connection
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if _connection is None:
        _connection = sqlite3.connect(path, check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA journal_mode=WAL")
        _connection.execute("PRAGMA busy_timeout=5000")
    return _connection


def init_db(db_path=None):
    """Создаёт таблицы, если их нет. Безопасно вызывать многократно."""
    conn = _get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS positions (
            symbol          TEXT PRIMARY KEY,
            entry_price     REAL NOT NULL DEFAULT 0.0,
            quantity        REAL NOT NULL DEFAULT 0.0,
            highest_price   REAL NOT NULL DEFAULT 0.0,
            entry_time      TEXT NOT NULL DEFAULT '',
            updated_at      TEXT NOT NULL DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS trade_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT NOT NULL,
            side        TEXT NOT NULL,       -- 'buy' | 'sell'
            price       REAL NOT NULL,
            quantity    REAL NOT NULL,
            value       REAL NOT NULL,       -- price * quantity
            pnl         REAL DEFAULT 0.0,
            pnl_pct     REAL DEFAULT 0.0,
            order_id    TEXT,
            exchange_id TEXT,                 -- ID ордера с биржи для дедупликации
            timestamp   TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS orders (
            order_id    TEXT PRIMARY KEY,
            symbol      TEXT NOT NULL,
            side        TEXT NOT NULL,
            price       REAL,
            quantity    REAL NOT NULL,
            filled_qty  REAL DEFAULT 0.0,
            status      TEXT DEFAULT 'open',
            created_at  TEXT NOT NULL,
            updated_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS capital_snapshots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            total           REAL NOT NULL,
            positions_value REAL NOT NULL DEFAULT 0,
            free_usdt       REAL NOT NULL DEFAULT 0,
            positions_count INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        -- Индексы для быстрых запросов по trade_history
        CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol);
        CREATE INDEX IF NOT EXISTS idx_trade_history_timestamp ON trade_history(timestamp);
        CREATE INDEX IF NOT EXISTS idx_trade_history_order ON trade_history(order_id);
    """)
    conn.commit()


# === POSITIONS ===

def get_all_positions(db_path=None) -> Dict[str, Dict]:
    """Возвращает словарь {symbol: position_dict}"""
    conn = _get_connection(db_path)
    rows = conn.execute("SELECT * FROM positions WHERE quantity > 0").fetchall()
    result = {}
    for row in rows:
        result[row['symbol']] = dict(row)
    return result


def get_position(symbol: str, db_path=None) -> Optional[Dict]:
    """Возвращает одну позицию или None"""
    conn = _get_connection(db_path)
    row = conn.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,)).fetchone()
    return dict(row) if row else None


def upsert_position(symbol: str, entry_price: float, quantity: float,
                    highest_price: float = 0.0, entry_time: str = '',
                    db_path=None):
    """
    Создаёт или обновляет позицию.
    entry_price — weighted average цена, рассчитанная из trade_history.
    """
    conn = _get_connection(db_path)
    now = datetime.now(timezone.utc).isoformat()
    existing = conn.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,)).fetchone()

    if existing:
        conn.execute("""
            UPDATE positions
            SET entry_price = ?, quantity = ?, highest_price = ?,
                updated_at = ?
            WHERE symbol = ?
        """, (entry_price, quantity, max(highest_price, existing['highest_price']), now, symbol))
    else:
        conn.execute("""
            INSERT INTO positions (symbol, entry_price, quantity, highest_price, entry_time, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (symbol, entry_price, quantity, highest_price, entry_time or now, now))

    conn.commit()


def remove_position(symbol: str, db_path=None) -> Optional[Dict]:
    """Удаляет позицию и возвращает её данные (для архивации)"""
    conn = _get_connection(db_path)
    pos = get_position(symbol, db_path)
    if pos:
        conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
        conn.commit()
    return pos


def sync_positions_from_exchange(exchange, enabled_pairs: List[str],
                                  db_path=None) -> Dict[str, Dict]:
    """
    Синхронизирует позиции с биржей.
    Возвращает словарь {symbol: position_dict} для текущих позиций.
    """
    import ccxt
    conn = _get_connection(db_path)

    try:
        balance = exchange.fetch_balance()
    except Exception as e:
        print(f"[DB] Ошибка fetch_balance: {e}")
        return get_all_positions(db_path)

    active_positions = {}
    now = datetime.now(timezone.utc).isoformat()

    for pair in enabled_pairs:
        base_currency = pair.split('/')[0]
        free = balance.get(base_currency, {}).get('free', 0)
        used = balance.get(base_currency, {}).get('used', 0)
        total = free + used

        if total <= 0:
            existing = conn.execute("SELECT * FROM positions WHERE symbol = ?", (pair,)).fetchone()
            if existing:
                conn.execute("DELETE FROM positions WHERE symbol = ?", (pair,))
                conn.commit()
            continue

        # Получаем цену с биржи
        try:
            ticker = exchange.fetch_ticker(pair)
            current_price = ticker['last'] or ticker['ask'] or 0
        except Exception as _e:
            logger and logger.debug("bare except in db: %s", _e) if "logger" in dir() else None
            current_price = 0

        # Считаем weighted average из trade_history
        entry_price, _ = calculate_weighted_entry(pair, db_path)

        if entry_price == 0:
            entry_price = current_price

        entry_time = now
        existing = conn.execute("SELECT * FROM positions WHERE symbol = ?", (pair,)).fetchone()
        if existing:
            entry_time = existing['entry_time']
            if existing['quantity'] > 0:
                # Если в БД уже была позиция — используем старую weighted average
                old_cost = existing['entry_price'] * existing['quantity']
                new_cost = entry_price * total
                if total > 0 and entry_price > 0:
                    entry_price = (old_cost + new_cost * 0.001) / (existing['quantity'] + total * 0.001)
                else:
                    entry_price = existing['entry_price']

        upsert_position(pair, entry_price, total, current_price, entry_time, db_path)
        active_positions[pair] = {
            'symbol': pair,
            'entry_price': entry_price,
            'quantity': total,
            'highest_price': current_price,
            'entry_time': entry_time,
        }

    return active_positions


# === TRADE HISTORY (weighted average core) ===

def add_trade(symbol: str, side: str, price: float, quantity: float,
              pnl: float = 0.0, pnl_pct: float = 0.0,
              order_id: str = None, exchange_id: str = None,
              timestamp: str = None, db_path=None) -> int:
    """
    Добавляет сделку в trade_history.

    Возвращает id новой записи.
    exchange_id — используется для дедупликации (ID ордера с биржи).
    """
    conn = _get_connection(db_path)

    # Дедупликация: проверяем, нет ли уже такой сделки
    if exchange_id:
        existing = conn.execute(
            "SELECT id FROM trade_history WHERE exchange_id = ?", (exchange_id,)
        ).fetchone()
        if existing:
            return existing['id']

    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    value = price * quantity

    conn.execute("""
        INSERT INTO trade_history (symbol, side, price, quantity, value, pnl, pnl_pct, order_id, exchange_id, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (symbol, side, price, quantity, value, pnl, pnl_pct, order_id, exchange_id, timestamp))

    conn.commit()

    # После добавления сделки — пересчитываем weighted average позиции
    _recalculate_position_from_history(symbol, db_path)

    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def calculate_weighted_entry(symbol: str, db_path=None) -> Tuple[float, float]:
    """
    Рассчитывает weighted average entry price из истории buy-сделок.

    Returns:
        (weighted_avg_price, total_bought_quantity)
        (0, 0) если buy-сделок нет
    """
    conn = _get_connection(db_path)
    buy_trades = conn.execute("""
        SELECT price, quantity FROM trade_history
        WHERE symbol = ? AND side = 'buy'
        ORDER BY timestamp ASC
    """, (symbol,)).fetchall()

    if not buy_trades:
        return 0.0, 0.0

    # Рассчитываем total buy quantity с учётом sell (частичных закрытий)
    sell_trades = conn.execute("""
        SELECT price, quantity FROM trade_history
        WHERE symbol = ? AND side = 'sell'
        ORDER BY timestamp ASC
    """, (symbol,)).fetchall()

    total_buy_qty = sum(t['quantity'] for t in buy_trades)
    total_sell_qty = sum(t['quantity'] for t in sell_trades)
    net_qty = total_buy_qty - total_sell_qty

    if net_qty <= 0:
        return 0.0, 0.0

    # Weighted average: только те buy, которые ещё не проданы
    # Используем FIFO: первые купленные считаются первыми проданными
    remaining = net_qty
    total_cost = 0.0
    total_weighted_qty = 0.0

    for t in buy_trades:
        qty = min(t['quantity'], remaining)
        if qty <= 0:
            break
        total_cost += qty * t['price']
        total_weighted_qty += qty
        remaining -= qty

    if total_weighted_qty > 0:
        return total_cost / total_weighted_qty, total_weighted_qty

    return 0.0, 0.0


def _recalculate_position_from_history(symbol: str, db_path=None):
    """
    Пересчитывает entry_price позиции на основе trade_history.
    Вызывается после каждой новой сделки.
    """
    conn = _get_connection(db_path)
    pos = conn.execute("SELECT * FROM positions WHERE symbol = ?", (symbol,)).fetchone()

    if not pos:
        return

    entry_price, qty = calculate_weighted_entry(symbol, db_path)

    if qty <= 0:
        # Позиция полностью закрыта — удаляем
        conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
        conn.commit()
        return

    old_qty = pos['quantity']
    # Берём highest_price из trade_history buy
    best_buy = conn.execute("""
        SELECT MAX(price) as max_price FROM trade_history
        WHERE symbol = ? AND side = 'buy'
    """, (symbol,)).fetchone()
    highest_price = pos['highest_price']
    if best_buy and best_buy['max_price']:
        highest_price = max(highest_price, best_buy['max_price'])

    if old_qty != qty:
        # Количество изменилось — синхронизируем
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            UPDATE positions
            SET entry_price = ?, quantity = ?, highest_price = ?, updated_at = ?
            WHERE symbol = ?
        """, (entry_price, qty, highest_price, now, symbol))
        conn.commit()


def get_trade_history(symbol: str = None, limit: int = 50,
                      side: str = None, db_path=None) -> List[Dict]:
    """
    Возвращает историю сделок.

    Args:
        symbol: фильтр по паре (None = все)
        limit: количество записей
        side: 'buy' | 'sell' | None (все)
    """
    conn = _get_connection(db_path)

    conditions = []
    params = []

    if symbol:
        conditions.append("symbol = ?")
        params.append(symbol)
    if side:
        conditions.append("side = ?")
        params.append(side)

    where = ""
    if conditions:
        where = "WHERE " + " AND ".join(conditions)

    rows = conn.execute(f"""
        SELECT * FROM trade_history
        {where}
        ORDER BY timestamp DESC
        LIMIT ?
    """, params + [limit]).fetchall()

    return [dict(r) for r in rows]


def get_position_trades(symbol: str, db_path=None) -> List[Dict]:
    """
    Возвращает все сделки по конкретной паре (от старых к новым).
    """
    conn = _get_connection(db_path)
    rows = conn.execute("""
        SELECT * FROM trade_history
        WHERE symbol = ?
        ORDER BY timestamp ASC
    """, (symbol,)).fetchall()
    return [dict(r) for r in rows]


# === ORDERS ===

def sync_orders_from_exchange(exchange, db_path=None):
    """
    Синхронизирует открытые ордера с биржи в локальную БД.
    Используется в main.py каждые 5 минут.
    """
    conn = _get_connection(db_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0
    try:
        open_orders = exchange.fetch_open_orders()
        for order in open_orders:
            order_id = str(order.get('id', ''))
            if not order_id:
                continue
            symbol = order.get('symbol', '')
            side = order.get('side', 'buy')
            price = float(order.get('price', 0) or 0)
            amount = float(order.get('amount', 0) or 0)
            filled = float(order.get('filled', 0) or 0)
            status = order.get('status', 'open')
            upsert_order(order_id, symbol, side, price, amount, filled, status, db_path)
            count += 1
        logger = None
        try:
            import logging
            logger = logging.getLogger('db')
        except Exception as _e:
            logger and logger.debug("bare except in db: %s", _e) if "logger" in dir() else None
            pass
        if logger and count > 0:
            logger.info(f"[DB] Синхронизировано {count} ордеров с биржи")
    except Exception as e:
        try:
            import logging
            logging.getLogger('db').warning(f"[DB] Ошибка синхронизации ордеров: {e}")
        except Exception as _e:
            logger and logger.debug("bare except in db: %s", _e) if "logger" in dir() else None
            pass
    return count


def upsert_order(order_id: str, symbol: str, side: str, price: float,
                 quantity: float, filled_qty: float = 0.0,
                 status: str = 'open', db_path=None):
    """Создаёт или обновляет ордер"""
    conn = _get_connection(db_path)
    now = datetime.now(timezone.utc).isoformat()
    existing = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()

    if existing:
        conn.execute("""
            UPDATE orders SET filled_qty=?, status=?, updated_at=?
            WHERE order_id=?
        """, (filled_qty, status, now, order_id))
    else:
        conn.execute("""
            INSERT INTO orders (order_id, symbol, side, price, quantity, filled_qty, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (order_id, symbol, side, price, quantity, filled_qty, status, now, now))
    conn.commit()


# === CAPITAL SNAPSHOTS ===

def save_capital_snapshot(total: float, positions_value: float,
                          free_usdt: float, positions_count: int,
                          db_path=None):
    """Сохраняет снэпшот капитала для графика"""
    conn = _get_connection(db_path)
    conn.execute("""
        INSERT INTO capital_snapshots (total, positions_value, free_usdt, positions_count)
        VALUES (?, ?, ?, ?)
    """, (total, positions_value, free_usdt, positions_count))
    conn.commit()


def get_capital_history(limit: int = 1000, db_path=None) -> List[Dict]:
    """Возвращает историю капитала для графика"""
    conn = _get_connection(db_path)
    rows = conn.execute("""
        SELECT * FROM capital_snapshots ORDER BY created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


# === PNL ===

def get_pnl_stats(db_path=None) -> Dict:
    """Возвращает статистику по всем сделкам"""
    conn = _get_connection(db_path)
    total_pnl = conn.execute("""
        SELECT COALESCE(SUM(pnl), 0) FROM trade_history WHERE side = 'sell'
    """).fetchone()[0]
    total_trades = conn.execute("""
        SELECT COUNT(*) FROM trade_history WHERE side = 'sell'
    """).fetchone()[0]
    win_trades = conn.execute("""
        SELECT COUNT(*) FROM trade_history WHERE side = 'sell' AND pnl > 0
    """).fetchone()[0]
    loss_trades = conn.execute("""
        SELECT COUNT(*) FROM trade_history WHERE side = 'sell' AND pnl < 0
    """).fetchone()[0]
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0

    return {
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'win_trades': win_trades,
        'loss_trades': loss_trades,
        'win_rate': round(win_rate, 1),
    }


# === META ===

def set_meta(key: str, value: str, db_path=None):
    """Сохраняет мета-данные (key-value)"""
    conn = _get_connection(db_path)
    conn.execute("""
        INSERT INTO meta (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
    """, (key, str(value)))
    conn.commit()


def get_meta(key: str, default=None, db_path=None):
    """Читает мета-данные"""
    conn = _get_connection(db_path)
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row['value'] if row else default


# === МИГРАЦИЯ: старые записи → trade_history ===
# Вызывается один раз при первом запуске новой версии

def _migrate_legacy_trades_to_history(db_path=None):
    """
    Переносит существующие позиции в trade_history как buy-сделки.
    Безопасно: проверяет, есть ли уже записи.
    """
    conn = _get_connection(db_path)
    existing = conn.execute("SELECT COUNT(*) FROM trade_history").fetchone()[0]
    if existing > 0:
        return  # Уже есть данные

    positions = get_all_positions(db_path)
    now = datetime.now(timezone.utc).isoformat()
    count = 0

    for symbol, pos in positions.items():
        if pos['quantity'] <= 0 or pos['entry_price'] <= 0:
            continue
        conn.execute("""
            INSERT INTO trade_history (symbol, side, price, quantity, value, pnl, pnl_pct, order_id, exchange_id, timestamp)
            VALUES (?, 'buy', ?, ?, ?, 0, 0, 'migration', ?, ?)
        """, (symbol, pos['entry_price'], pos['quantity'],
              pos['entry_price'] * pos['quantity'],
              f"mig_{symbol.replace('/', '_')}",
              pos.get('entry_time', now)))
        count += 1

    if count > 0:
        conn.commit()
        print(f"[DB] Мигрировано {count} legacy позиций в trade_history")
