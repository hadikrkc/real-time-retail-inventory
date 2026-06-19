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
    inserted_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (time, store_id, item_id)
);

SELECT create_hypertable('sales_features', by_range('time'), if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_features_store_item ON sales_features (store_id, item_id, time DESC);

-- Anomaly alerts produced by streaming Isolation Forest (Faz 4)
CREATE TABLE IF NOT EXISTS anomaly_alerts (
    alert_id             TEXT        NOT NULL,
    detected_at          TIMESTAMPTZ NOT NULL,
    feature_time         TIMESTAMPTZ NOT NULL,
    store_id             TEXT        NOT NULL,
    item_id              TEXT        NOT NULL,
    anomaly_score        FLOAT,
    rolling_avg_7d       FLOAT,
    rolling_sum_7d       BIGINT,
    event_count_7d       BIGINT,
    max_qty_7d           BIGINT,
    min_qty_7d           BIGINT,
    detection_latency_ms BIGINT,
    PRIMARY KEY (alert_id, detected_at)
);

SELECT create_hypertable('anomaly_alerts', by_range('detected_at'), if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_alerts_store_item ON anomaly_alerts (store_id, item_id, detected_at DESC);

-- Demand forecasts produced by streaming forecaster (Faz 5)
CREATE TABLE IF NOT EXISTS forecast_results (
    forecast_id     TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL,
    store_id        TEXT        NOT NULL,
    item_id         TEXT        NOT NULL,
    horizon_day     INTEGER     NOT NULL,
    forecast_date   TIMESTAMPTZ NOT NULL,
    predicted_qty   FLOAT       NOT NULL,
    feature_source  TEXT        NOT NULL,
    PRIMARY KEY (forecast_id, created_at)
);

SELECT create_hypertable('forecast_results', by_range('created_at'), if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_forecast_store_item ON forecast_results (store_id, item_id, created_at DESC);
