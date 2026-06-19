"""
Phase 5 — Streaming demand forecaster.

Polls sales_features for rolling window features, fits ExponentialSmoothing
per (store_id, item_id), and publishes 7-day forecasts to the
retail.forecast.results Kafka topic and the forecast_results DB table.

RQ2 comparison:
  Batch     : scripts/batch_baseline.py — full historical series, one-shot ESM
  Streaming : this script               — 7-day rolling avg, continuously updated

Local usage:
  python pipeline/stream_processor/demand_forecaster.py \
    --db-host localhost --kafka-server localhost:9092 --store CA_1

All stores:
  python pipeline/stream_processor/demand_forecaster.py \
    --db-host localhost --kafka-server localhost:9092 --store ALL
"""

import argparse
import json
import time
import uuid
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")

import numpy as np
np.seterr(all="ignore")
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from confluent_kafka import Producer

HORIZON       = 7    # days ahead
MIN_HISTORY   = 7    # minimum number of windows required to forecast
POLL_INTERVAL = 60   # seconds between polls — forecasting is compute-heavy


# ── Helpers ───────────────────────────────────────────────────────────────────

def connect_db(host: str) -> psycopg2.extensions.connection:
    return psycopg2.connect(
        host=host, port=5432, dbname="retail",
        user="retail", password="retail",
    )


def make_producer(kafka_server: str) -> Producer:
    return Producer({
        "bootstrap.servers": kafka_server,
        "acks": "1",
        "linger.ms": 20,
        "compression.type": "lz4",
    })


def list_stores(conn: psycopg2.extensions.connection) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT store_id FROM sales_features ORDER BY store_id")
        return [r[0] for r in cur.fetchall()]


# ── Feature loading ───────────────────────────────────────────────────────────

def fetch_item_histories(
    conn: psycopg2.extensions.connection,
    store_id: str,
) -> dict[str, list]:
    """Return rolling_avg_7d history for all items in the given store."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT item_id, time, rolling_avg_7d
            FROM sales_features
            WHERE store_id = %s AND rolling_avg_7d IS NOT NULL
            ORDER BY item_id, time ASC
        """, (store_id,))
        rows = cur.fetchall()

    histories: dict[str, list] = defaultdict(list)
    for row in rows:
        histories[row["item_id"]].append(row)
    return dict(histories)


# ── Forecasting ───────────────────────────────────────────────────────────────

def forecast_item(history: list) -> list[float] | None:
    """Fit ExponentialSmoothing on rolling_avg_7d series, return HORIZON-step forecast."""
    if len(history) < MIN_HISTORY:
        return None

    series = np.array([float(r["rolling_avg_7d"]) for r in history], dtype=np.float64)
    series = np.clip(series, 0, None)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(
                series, trend="add", seasonal=None,
            ).fit(optimized=True)
        preds = fit.forecast(HORIZON)
        return [max(0.0, float(p)) for p in preds]
    except Exception:
        last = float(series[-1])
        return [last] * HORIZON


# ── Publish ───────────────────────────────────────────────────────────────────

def publish_forecasts(
    store_id: str,
    item_id: str,
    history: list,
    preds: list[float],
    producer: Producer,
    conn: psycopg2.extensions.connection,
    created_at: datetime,
) -> int:
    last_time: datetime = history[-1]["time"]
    records = []

    for day_offset, qty in enumerate(preds, start=1):
        forecast_date = last_time + timedelta(days=day_offset)
        fid = str(uuid.uuid4())
        msg = {
            "forecast_id":    fid,
            "created_at":     created_at.isoformat(),
            "store_id":       store_id,
            "item_id":        item_id,
            "horizon_day":    day_offset,
            "forecast_date":  forecast_date.isoformat(),
            "predicted_qty":  qty,
            "feature_source": "streaming",
        }
        producer.produce(
            topic="retail.forecast.results",
            key=f"{store_id}_{item_id}".encode(),
            value=json.dumps(msg).encode(),
        )
        records.append((
            fid, created_at, store_id, item_id,
            day_offset, forecast_date, qty, "streaming",
        ))

    _persist_forecasts(conn, records)
    return len(records)


def _persist_forecasts(conn, records: list) -> None:
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO forecast_results
                (forecast_id, created_at, store_id, item_id,
                 horizon_day, forecast_date, predicted_qty, feature_source)
            VALUES %s
            ON CONFLICT (forecast_id, created_at) DO NOTHING
        """, records)
    conn.commit()


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_store(
    store_id: str,
    conn: psycopg2.extensions.connection,
    producer: Producer,
) -> tuple[int, int]:
    """Run one forecasting pass for a single store. Returns (item_count, forecast_count)."""
    histories = fetch_item_histories(conn, store_id)
    if not histories:
        return 0, 0

    created_at    = datetime.now(timezone.utc)
    item_count    = 0
    forecast_count = 0

    for item_id, history in histories.items():
        preds = forecast_item(history)
        if preds is None:
            continue
        n = publish_forecasts(
            store_id, item_id, history, preds, producer, conn, created_at
        )
        forecast_count += n
        item_count     += 1

    producer.flush()
    return item_count, forecast_count


def main():
    parser = argparse.ArgumentParser(description="Streaming demand forecaster — Faz 5")
    parser.add_argument("--db-host",       default="localhost")
    parser.add_argument("--kafka-server",  default="localhost:9092")
    parser.add_argument("--store",         default="CA_1",
                        help="Store ID or ALL (default: CA_1)")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL)
    parser.add_argument("--once",          action="store_true",
                        help="Run one pass and exit (useful for testing)")
    args = parser.parse_args()

    conn     = connect_db(args.db_host)
    producer = make_producer(args.kafka_server)

    print(f"DB            : {args.db_host}:5432/retail")
    print(f"Kafka         : {args.kafka_server}")
    print(f"Store filter  : {args.store}")
    print(f"Horizon       : {HORIZON} gun")
    print(f"Poll interval : {args.poll_interval}s")
    print(f"Topic out     : retail.forecast.results\n")

    if args.store == "ALL":
        stores = list_stores(conn)
        print(f"Stores: {stores}\n")
    else:
        stores = [args.store]

    total_items     = 0
    total_forecasts = 0
    run_count       = 0

    try:
        while True:
            t0 = time.time()
            run_count += 1
            run_items = run_forecasts = 0

            for store_id in stores:
                items, forecasts = run_store(store_id, conn, producer)
                run_items     += items
                run_forecasts += forecasts

            total_items     += run_items
            total_forecasts += run_forecasts
            elapsed = time.time() - t0

            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            print(
                f"[{ts}] Run #{run_count} | items={run_items:,} | "
                f"forecasts={run_forecasts:,} | elapsed={elapsed:.1f}s | "
                f"cumulative forecasts={total_forecasts:,}"
            )

            if args.once:
                break

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print(f"\nStopped.")
        print(f"Total runs      : {run_count}")
        print(f"Total forecasts : {total_forecasts:,}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
