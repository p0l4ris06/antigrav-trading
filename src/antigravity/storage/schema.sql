-- ============================================================
-- Antigravity — ClickHouse Tick Persistence Schema
-- Engine: MergeTree (append-only, optimized for time-series)
-- ============================================================

CREATE DATABASE IF NOT EXISTS antigravity;

-- -----------------------------------------------------------
-- Raw tick table
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS antigravity.ticks
(
    symbol       LowCardinality(String),    -- e.g. 'BTCUSDT', bounded cardinality
    timestamp    DateTime64(6, 'UTC'),      -- microsecond precision
    bid_price    Float64,
    ask_price    Float64,
    bid_size     Float64,
    ask_size     Float64,
    last_price   Float64,
    last_size    Float64,
    trade_id     UInt64                      DEFAULT 0
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp)
SETTINGS index_granularity = 8192;


-- -----------------------------------------------------------
-- Materialized view: 1-minute OHLCV candles
-- Computed at ingest time — zero read-time aggregation cost
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS antigravity.ohlcv_1m_store
(
    symbol        LowCardinality(String),
    window_start  DateTime('UTC'),
    open          AggregateFunction(argMin, Float64, DateTime64(6, 'UTC')),
    high          AggregateFunction(max, Float64),
    low           AggregateFunction(min, Float64),
    close         AggregateFunction(argMax, Float64, DateTime64(6, 'UTC')),
    volume        AggregateFunction(sum, Float64)
)
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(window_start)
ORDER BY (symbol, window_start);

CREATE MATERIALIZED VIEW IF NOT EXISTS antigravity.ohlcv_1m_mv
TO antigravity.ohlcv_1m_store
AS SELECT
    symbol,
    toStartOfMinute(timestamp) AS window_start,
    argMinState(last_price, timestamp) AS open,
    maxState(last_price) AS high,
    minState(last_price) AS low,
    argMaxState(last_price, timestamp) AS close,
    sumState(last_size) AS volume
FROM antigravity.ticks
GROUP BY symbol, window_start;


-- -----------------------------------------------------------
-- Skipping index on trade_id for deduplication queries
-- -----------------------------------------------------------
ALTER TABLE antigravity.ticks
    ADD INDEX IF NOT EXISTS idx_trade_id trade_id
    TYPE minmax GRANULARITY 4;
