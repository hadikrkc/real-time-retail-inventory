"""
Phase 4 — Streaming anomaly detector.

Polls sales_features, applies Isolation Forest, publishes anomalies to the
retail.anomaly.alerts Kafka topic, and persists results with detection latency
to the anomaly_alerts table (used for RQ1 metrics).

Local usage:
  python pipeline/stream_processor/anomaly_detector.py \
    --db-host localhost --kafka-server localhost:9092

Inside Docker:
  docker exec spark python3 /opt/pipeline/stream_processor/anomaly_detector.py \
    --db-host timescaledb --kafka-server kafka1:19092
"""

import argparse
import json
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from sklearn.ensemble import IsolationForest
from confluent_kafka import Producer

FEATURE_COLS = [
    "rolling_avg_7d", "rolling_sum_7d", "event_count_7d",
    "max_qty_7d", "min_qty_7d",
]
TRAIN_LIMIT   = 100_000
POLL_INTERVAL = 5       # seconds
BATCH_LIMIT   = 10_000  # max rows per poll


# ── DB / Kafka helpers ────────────────────────────────────────────────────────

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


# ── Model training ────────────────────────────────────────────────────────────

def train_model(conn: psycopg2.extensions.connection) -> IsolationForest:
    print(f"Training Isolation Forest on up to {TRAIN_LIMIT:,} samples ...")
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT {', '.join(FEATURE_COLS)}
            FROM sales_features
            WHERE rolling_avg_7d IS NOT NULL
            ORDER BY time DESC
            LIMIT %s
        """, (TRAIN_LIMIT,))
        rows = cur.fetchall()

    if not rows:
        return None

    X = np.array([[float(v or 0) for v in row] for row in rows], dtype=np.float32)
    model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X)
    print(f"  Trained on {len(X):,} samples.  Ready.\n")
    return model


# ── Polling ───────────────────────────────────────────────────────────────────

def get_watermark(conn: psycopg2.extensions.connection) -> datetime:
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(time) FROM sales_features")
        return cur.fetchone()[0]


def poll_features(conn: psycopg2.extensions.connection, after: datetime) -> list:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT time, store_id, item_id, inserted_at, {', '.join(FEATURE_COLS)}
            FROM sales_features
            WHERE time > %s
            ORDER BY time ASC
            LIMIT %s
        """, (after, BATCH_LIMIT))
        return cur.fetchall()


# ── Detection & publishing ────────────────────────────────────────────────────

def detect_and_publish(
    model: IsolationForest,
    rows: list,
    producer: Producer,
    conn: psycopg2.extensions.connection,
) -> tuple[int, int]:
    X = np.array(
        [[float(r[c] or 0) for c in FEATURE_COLS] for r in rows],
        dtype=np.float32,
    )
    preds  = model.predict(X)        # -1 = anomaly, 1 = normal
    scores = model.score_samples(X)

    detected_at   = datetime.now(timezone.utc)
    alerts_to_save = []
    anomaly_count  = 0

    for row, pred, score in zip(rows, preds, scores):
        if pred != -1:
            continue

        feature_time = row["time"]
        # inserted_at = wall-clock time Spark wrote this row; None for old rows
        inserted_at  = row.get("inserted_at") or detected_at
        latency_ms   = int(
            (detected_at - inserted_at).total_seconds() * 1000
        )
        alert_id = str(uuid.uuid4())

        alert = {
            "alert_id":             alert_id,
            "detected_at":          detected_at.isoformat(),
            "feature_time":         feature_time.isoformat(),
            "store_id":             row["store_id"],
            "item_id":              row["item_id"],
            "anomaly_score":        float(score),
            "rolling_avg_7d":       float(row["rolling_avg_7d"] or 0),
            "rolling_sum_7d":       int(row["rolling_sum_7d"]   or 0),
            "event_count_7d":       int(row["event_count_7d"]   or 0),
            "max_qty_7d":           int(row["max_qty_7d"]       or 0),
            "min_qty_7d":           int(row["min_qty_7d"]       or 0),
            "detection_latency_ms": latency_ms,
        }

        producer.produce(
            topic="retail.anomaly.alerts",
            key=f"{row['store_id']}_{row['item_id']}".encode(),
            value=json.dumps(alert).encode(),
        )
        alerts_to_save.append(alert)
        anomaly_count += 1

    producer.flush()

    if alerts_to_save:
        _persist_alerts(conn, alerts_to_save)

    return anomaly_count, len(rows)


def _persist_alerts(conn, alerts: list) -> None:
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO anomaly_alerts
                (alert_id, detected_at, feature_time, store_id, item_id,
                 anomaly_score, rolling_avg_7d, rolling_sum_7d,
                 event_count_7d, max_qty_7d, min_qty_7d, detection_latency_ms)
            VALUES %s
            ON CONFLICT (alert_id, detected_at) DO NOTHING
        """, [
            (a["alert_id"], a["detected_at"], a["feature_time"],
             a["store_id"], a["item_id"], a["anomaly_score"],
             a["rolling_avg_7d"], a["rolling_sum_7d"],
             a["event_count_7d"], a["max_qty_7d"], a["min_qty_7d"],
             a["detection_latency_ms"])
            for a in alerts
        ])
    conn.commit()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Streaming anomaly detector — Faz 4")
    parser.add_argument("--db-host",       default="localhost")
    parser.add_argument("--kafka-server",  default="localhost:9092")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL,
                        help="Seconds between polls (default: 5)")
    args = parser.parse_args()

    conn     = connect_db(args.db_host)
    producer = make_producer(args.kafka_server)

    print(f"DB            : {args.db_host}:5432/retail")
    print(f"Kafka         : {args.kafka_server}")
    print(f"Poll interval : {args.poll_interval}s")
    print(f"Topic out     : retail.anomaly.alerts\n")

    model = None
    while model is None:
        model = train_model(conn)
        if model is None:
            print("  sales_features bos, 10s sonra tekrar deneniyor...")
            time.sleep(10)

    watermark = get_watermark(conn)

    print(f"Watermark start : {watermark}")
    print("Anomaly detector running. Ctrl+C to stop.\n")

    total_checked = 0
    total_alerts  = 0

    try:
        while True:
            rows = poll_features(conn, watermark)

            if rows:
                n_alerts, n_checked = detect_and_publish(
                    model, rows, producer, conn
                )
                total_checked += n_checked
                total_alerts  += n_alerts
                watermark       = rows[-1]["time"]

                ts   = datetime.now(timezone.utc).strftime("%H:%M:%S")
                rate = n_alerts / n_checked * 100 if n_checked else 0
                print(
                    f"[{ts}] checked={n_checked:>6,} | alerts={n_alerts:>5,} ({rate:4.1f}%)"
                    f" | cumulative checked={total_checked:,}  alerts={total_alerts:,}"
                )

            time.sleep(args.poll_interval)

    except KeyboardInterrupt:
        print(f"\nStopped.")
        print(f"Total checked : {total_checked:,}")
        print(f"Total alerts  : {total_alerts:,}")
        if total_checked:
            print(f"Alert rate    : {total_alerts/total_checked*100:.2f}%")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
