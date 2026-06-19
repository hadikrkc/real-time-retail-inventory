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
