-- ======= PostgreSQL Schema для торговой системы =======
-- Спроектировано с учётом: long/short, JSONB-расширения, партиций

-- Установка расширений
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- для поиска по символам
CREATE EXTENSION IF NOT EXISTS pgcrypto; -- для генерации ID при необходимости

-- ======= 1. АККАУНТЫ =======
CREATE TABLE IF NOT EXISTS accounts (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    exchange    TEXT NOT NULL DEFAULT 'bybit',
    api_key     TEXT,
    is_active   BOOLEAN DEFAULT TRUE,
    meta        JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE accounts IS 'Биржевые аккаунты (на будущее — несколько)';
COMMENT ON COLUMN accounts.exchange IS 'bybit / binance / mock';
COMMENT ON COLUMN accounts.meta IS 'JSONB: testnet, subaccount, permissions';

-- Вставка дефолтного аккаунта
INSERT INTO accounts (name, exchange) VALUES ('Bybit USDT', 'bybit')
ON CONFLICT (name) DO NOTHING;

-- ======= 2. СДЕЛКИ (партицированные по месяцам) =======
CREATE TABLE IF NOT EXISTS trades (
    id              BIGSERIAL,
    account_id      INT REFERENCES accounts(id) DEFAULT 1,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL CHECK (side IN ('long', 'short')),
    status          TEXT NOT NULL DEFAULT 'open'
                        CHECK (status IN ('open', 'closed', 'cancelled')),

    -- Вход
    entry_price     NUMERIC(18,8) NOT NULL,
    entry_qty       NUMERIC(18,8) NOT NULL,
    entry_time      TIMESTAMPTZ NOT NULL,
    entry_reason    JSONB,                   -- весь разбор сигнала на входе
    entry_score     REAL,                    -- общий score
    ml_pro          REAL,
    adv_score       REAL,
    mtf_score       REAL,
    rvb_score       REAL,
    liq_score       REAL,
    vv_score        REAL,
    vsa_score       REAL,

    -- Выход
    exit_price      NUMERIC(18,8),
    exit_qty        NUMERIC(18,8),
    exit_time       TIMESTAMPTZ,
    exit_reason     TEXT,                    -- SL / TP / early_trail / ensemble / manual
    exit_ensemble_hold REAL,                 -- % hold при выходе
    exit_details    JSONB,                   -- детали выхода (trail_peak, откат и т.д.)

    -- Результат
    pnl             NUMERIC(18,8),           -- в USDT
    pnl_percent     NUMERIC(8,4),            -- в %

    -- Технические поля
    tags            TEXT[],                  -- теги: 'night', 'reentry', 'aggressive'
    notes           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

COMMENT ON TABLE trades IS 'Все сделки: открытые + закрытые, long/short';
COMMENT ON COLUMN trades.entry_reason IS 'JSONB: полный разбор сигнала на входе';
COMMENT ON COLUMN trades.exit_details IS 'JSONB: peak, trail, ensemble breakdown';
COMMENT ON COLUMN trades.tags IS 'Массив тегов для фильтрации';

-- Создаём партицию на текущий месяц
DO $$
DECLARE
    partition_name TEXT;
    start_date TEXT;
    end_date TEXT;
BEGIN
    partition_name := 'trades_' || to_char(NOW(), 'YYYY_MM');
    start_date := to_char(date_trunc('month', NOW()), 'YYYY-MM-DD');
    end_date := to_char(date_trunc('month', NOW()) + INTERVAL '1 month', 'YYYY-MM-DD');

    IF NOT EXISTS (
        SELECT 1 FROM pg_class WHERE relname = partition_name
    ) THEN
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF trades FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
        RAISE NOTICE 'Created partition: % (% → %)', partition_name, start_date, end_date;
    END IF;
END
$$;

-- Индексы
CREATE INDEX IF NOT EXISTS idx_trades_symbol    ON trades (symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status    ON trades (status);
CREATE INDEX IF NOT EXISTS idx_trades_side      ON trades (side);
CREATE INDEX IF NOT EXISTS idx_trades_entry     ON trades (entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_exit      ON trades (exit_time) WHERE status = 'closed';

-- ======= 3. ИСТОРИЯ БАЛАНСА =======
CREATE TABLE IF NOT EXISTS balance_history (
    id          BIGSERIAL PRIMARY KEY,
    account_id  INT REFERENCES accounts(id) DEFAULT 1,
    balance     NUMERIC(18,8) NOT NULL,
    free        NUMERIC(18,8) NOT NULL,
    equity      NUMERIC(18,8),               -- если будет отличаться от balance
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_balance_time ON balance_history (recorded_at);
CREATE INDEX IF NOT EXISTS idx_balance_account ON balance_history (account_id);

COMMENT ON TABLE balance_history IS 'Снимки баланса для графиков';

-- ======= 4. ЛОГ СИГНАЛОВ (для анализа решений) =======
CREATE TABLE IF NOT EXISTS signal_log (
    id          BIGSERIAL PRIMARY KEY,
    symbol      TEXT NOT NULL,
    decision    TEXT NOT NULL CHECK (decision IN ('hold', 'buy', 'sell')),
    score       REAL,
    threshold   REAL,
    components  JSONB,                    -- полный разбор (ml_pro, mtf, rvb, liq, vv, vsa...)
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signal_time ON signal_log (created_at);
CREATE INDEX IF NOT EXISTS idx_signal_symbol ON signal_log (symbol);
CREATE INDEX IF NOT EXISTS idx_signal_decision ON signal_log (decision);

COMMENT ON TABLE signal_log IS 'Лог всех решений торгового движка';
COMMENT ON COLUMN signal_log.components IS 'JSONB: все модули со своими скорами';

-- ======= 5. МЕТАДАННЫЕ СИСТЕМЫ =======
CREATE TABLE IF NOT EXISTS meta (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE meta IS 'Системные метаданные (версия схемы, последний запуск и т.д.)';

-- Версионирование схемы
INSERT INTO meta (key, value) VALUES ('schema_version', '1.0')
ON CONFLICT (key) DO UPDATE SET value = '1.0', updated_at = NOW();

-- ======= 6. КОНФИГУРАЦИЯ (версионированная) =======
CREATE TABLE IF NOT EXISTS config (
    key         TEXT PRIMARY KEY,
    value       JSONB NOT NULL,
    description TEXT,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON TABLE config IS 'Версионированные настройки';

-- ======= 7. ТЕГИ ДЛЯ СДЕЛОК (нормализованные) =======
CREATE TABLE IF NOT EXISTS trade_tags (
    trade_id    BIGINT NOT NULL,
    tag         TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (trade_id, tag)
);

COMMENT ON TABLE trade_tags IS 'Теги сделок (нормализованная форма)';

-- ======= СОЗДАНИЕ СЛЕДУЮЩЕЙ ПАРТИЦИИ (автоматизация) =======
CREATE OR REPLACE FUNCTION create_next_trade_partition()
RETURNS void AS $$
DECLARE
    next_month DATE;
    partition_name TEXT;
    start_date TEXT;
    end_date TEXT;
BEGIN
    next_month := date_trunc('month', NOW()) + INTERVAL '1 month';
    partition_name := 'trades_' || to_char(next_month, 'YYYY_MM');
    start_date := to_char(next_month, 'YYYY-MM-DD');
    end_date := to_char(next_month + INTERVAL '1 month', 'YYYY-MM-DD');

    IF NOT EXISTS (
        SELECT 1 FROM pg_class WHERE relname = partition_name
    ) THEN
        EXECUTE format(
            'CREATE TABLE %I PARTITION OF trades FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
    END IF;
END;
$$ LANGUAGE plpgsql;
