"""
Kafka → TimescaleDB sink.

Reads from retail.sales.events, batch-inserts into sales_events hypertable,
and writes throughput + latency metrics to pipeline_metrics (RQ3 data).

Usage:
  python sink.py
  python sink.py --batch-size 2000 --from-beginning

Docker:
  ENV KAFKA_BROKERS=kafka1:19092  DB_HOST=timescaledb  FROM_BEGINNING=true
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from confluent_kafka import Consumer, KafkaError, TopicPartition

TOPIC         = "retail.sales.events"
KAFKA_BROKERS = os.environ.get("KAFKA_BROKERS", "localhost:9092")
DB_HOST       = os.environ.get("DB_HOST",  "localhost")
DB_PORT       = int(os.environ.get("DB_PORT", "5432"))
DB_NAME       = os.environ.get("DB_NAME",  "retail")
DB_USER       = os.environ.get("DB_USER",  "retail")
DB_PASS       = os.environ.get("DB_PASS",  "retail")
FROM_BEGINNING = os.environ.get("FROM_BEGINNING", "false").lower() == "true"

INSERT_SQL = """
    INSERT INTO sales_events
        (time, item_id, dept_id, cat_id, store_id, state_id, sales_qty, day, event_id, ingested_at)
    VALUES %s
    ON CONFLICT DO NOTHING
"""

METRICS_SQL = """
    INSERT INTO pipeline_metrics
        (time, events_per_sec, processing_lag_ms, kafka_lag, db_row_count)
    VALUES (%s, %s, %s, %s, %s)
"""


def make_consumer(from_beginning: bool) -> Consumer:
    conf = {
        "bootstrap.servers": KAFKA_BROKERS,
        "group.id": "timescale-sink",
        "auto.offset.reset": "earliest" if from_beginning else "latest",
        "enable.auto.commit": False,
    }
    consumer = Consumer(conf)
    consumer.subscribe([TOPIC])
    print(f"Kafka consumer ready  (offset={'earliest' if from_beginning else 'latest'}, broker={KAFKA_BROKERS})")
    return consumer


def make_db_conn():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASS,
    )
    conn.autocommit = False
    print(f"TimescaleDB connected  ({DB_HOST}:{DB_PORT}/{DB_NAME})")
    return conn


def flush_batch(conn, batch: list[tuple]) -> None:
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, INSERT_SQL, batch, page_size=500)
    conn.commit()


def write_metrics(
    events_per_sec: float,
    processing_lag_ms: float,
    kafka_lag: int,
    db_row_count: int,
) -> None:
    # Fresh connection per write to avoid any shared transaction state
    try:
        with psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            dbname=DB_NAME, user=DB_USER, password=DB_PASS,
        ) as mconn:
            with mconn.cursor() as cur:
                cur.execute(METRICS_SQL, (
                    datetime.now(timezone.utc),
                    events_per_sec,
                    processing_lag_ms,
                    kafka_lag,
                    db_row_count,
                ))
    except Exception as exc:
        print(f"[metrics] write error: {exc}")


def get_kafka_lag(consumer: Consumer) -> int:
    """Sum of (high_watermark - committed_offset) across all assigned partitions."""
    try:
        partitions = consumer.assignment()
        if not partitions:
            return -1
        total = 0
        committed = consumer.committed(partitions, timeout=3)
        for tp, committed_tp in zip(partitions, committed):
            _, high = consumer.get_watermark_offsets(tp, timeout=2, cached=False)
            committed_offset = max(committed_tp.offset, 0) if committed_tp.offset >= 0 else 0
            total += max(0, high - committed_offset)
        return total
    except Exception:
        return -1


def event_to_row(event: dict, ingested_at: datetime) -> tuple:
    return (
        event["timestamp"],
        event["item_id"],
        event["dept_id"],
        event["cat_id"],
        event["store_id"],
        event["state_id"],
        event["sales_qty"],
        event["day"],
        event["event_id"],
        ingested_at,
    )


def main():
    parser = argparse.ArgumentParser(description="Kafka → TimescaleDB sink")
    parser.add_argument("--batch-size",    type=int, default=1000)
    parser.add_argument("--from-beginning", action="store_true",
                        default=FROM_BEGINNING)
    parser.add_argument("--metrics-interval", type=float, default=10.0,
                        help="Seconds between pipeline_metrics writes")
    args = parser.parse_args()

    consumer = make_consumer(args.from_beginning)
    conn     = make_db_conn()

    batch: list[tuple] = []
    lag_samples: list[float] = []       # processing_lag_ms per message in window
    total       = 0
    t0          = time.perf_counter()
    last_report = t0
    last_metrics = t0

    print(f"Sink running (batch_size={args.batch_size}, metrics_interval={args.metrics_interval}s). Ctrl+C to stop.\n")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                if batch:
                    flush_batch(conn, batch)
                    total += len(batch)
                    batch.clear()
                    consumer.commit(asynchronous=False)
                # Still check timers even when idle (consumer caught up to producer)
                now = time.perf_counter()
                if now - last_report >= 5.0:
                    elapsed = now - t0
                    rate    = total / elapsed if elapsed > 0 else 0
                    print(f"[{elapsed:6.1f}s]  {total:>10,} rows written  |  {rate:>8,.0f} rows/s  (waiting for new events)")
                    last_report = now
                if now - last_metrics >= args.metrics_interval:
                    elapsed    = now - t0
                    rate       = total / elapsed if elapsed > 0 else 0
                    avg_lag_ms = sum(lag_samples) / len(lag_samples) if lag_samples else 0.0
                    kafka_lag  = get_kafka_lag(consumer)
                    write_metrics(rate, avg_lag_ms, kafka_lag, total)
                    lag_samples.clear()
                    last_metrics = now
                continue

            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"Consumer error: {msg.error()}")
                continue

            # RQ3: measure Kafka produce → DB write latency
            ts_type, ts_ms = msg.timestamp()
            if ts_ms and ts_ms > 0:
                lag_ms = time.time() * 1000 - ts_ms
                lag_samples.append(lag_ms)

            ingested_at = datetime.now(timezone.utc)
            event = json.loads(msg.value().decode("utf-8"))
            batch.append(event_to_row(event, ingested_at))

            if len(batch) >= args.batch_size:
                flush_batch(conn, batch)
                total += len(batch)
                batch.clear()
                consumer.commit(asynchronous=False)

            now = time.perf_counter()

            if now - last_report >= 5.0:
                elapsed = now - t0
                rate    = total / elapsed if elapsed > 0 else 0
                print(f"[{elapsed:6.1f}s]  {total:>10,} rows written  |  {rate:>8,.0f} rows/s")
                last_report = now

            if now - last_metrics >= args.metrics_interval:
                elapsed    = now - t0
                rate       = total / elapsed if elapsed > 0 else 0
                avg_lag_ms = sum(lag_samples) / len(lag_samples) if lag_samples else 0.0
                kafka_lag  = get_kafka_lag(consumer)
                write_metrics(rate, avg_lag_ms, kafka_lag, total)
                lag_samples.clear()
                last_metrics = now

    except KeyboardInterrupt:
        if batch:
            flush_batch(conn, batch)
            total += len(batch)
        print(f"\nShutdown. Total rows written: {total:,}")
    finally:
        consumer.close()
        conn.close()


if __name__ == "__main__":
    main()
