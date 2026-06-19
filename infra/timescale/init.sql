CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS sales_events (
    time        TIMESTAMPTZ     NOT NULL,
    item_id     TEXT            NOT NULL,
    dept_id     TEXT            NOT NULL,
    cat_id      TEXT            NOT NULL,
    store_id    TEXT            NOT NULL,
    state_id    TEXT            NOT NULL,
    sales_qty   INTEGER         NOT NULL,
    day         TEXT            NOT NULL,
    event_id    TEXT            NOT NULL,
    UNIQUE (event_id)
);

SELECT create_hypertable('sales_events', by_range('time'), if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_sales_store_item ON sales_events (store_id, item_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_sales_cat        ON sales_events (cat_id, time DESC);

-- Rolling window features produced by Spark Structured Streaming
CREATE TABLE IF NOT EXISTS sales_features (
    time            TIMESTAMPTZ NOT NULL,
    store_id        TEXT        NOT NULL,
    item_id         TEXT        NOT NULL,
    rolling_avg_7d  FLOAT,
    rolling_sum_7d  BIGINT,
    event_count_7d  BIGINT,
    max_qty_7d      BIGINT,
    min_qty_7d      BIGINT,
    UNIQUE (time, store_id, item_id)
);

SELECT create_hypertable('sales_features', by_range('time'), if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_features_store_item ON sales_features (store_id, item_id, time DESC);
