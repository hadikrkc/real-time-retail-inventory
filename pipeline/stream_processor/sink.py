"""
Kafka → TimescaleDB sink.

Reads from retail.sales.events and batch-inserts into the sales_events hypertable.

Usage:
  python sink.py
  python sink.py --batch-size 2000 --from-beginning
"""

import argparse
import json
import time

import psycopg2
import psycopg2.extras
from confluent_kafka import Consumer, KafkaError

TOPIC = "retail.sales.events"
BOOTSTRAP_SERVERS = "localhost:9092"

DB_DSN = "host=localhost port=5432 dbname=retail user=retail password=retail"

INSERT_SQL = """
    INSERT INTO sales_events
        (time, item_id, dept_id, cat_id, store_id, state_id, sales_qty, day, event_id)
    VALUES %s
    ON CONFLICT DO NOTHING
"""


def make_consumer(from_beginning: bool) -> Consumer:
    conf = {
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "group.id": "timescale-sink",
        "auto.offset.reset": "earliest" if from_beginning else "latest",
        "enable.auto.commit": False,
    }
    consumer = Consumer(conf)
    consumer.subscribe([TOPIC])
    print(f"Kafka consumer ready (offset={'earliest' if from_beginning else 'latest'})")
    return consumer


def make_db_conn():
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = False
    print("TimescaleDB connected")
    return conn


def flush_batch(conn, batch: list[tuple]) -> None:
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, INSERT_SQL, batch, page_size=500)
    conn.commit()


def event_to_row(event: dict) -> tuple:
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
    )


def main():
    parser = argparse.ArgumentParser(description="Kafka → TimescaleDB sink")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of events per DB insert batch")
    parser.add_argument("--from-beginning", action="store_true")
    args = parser.parse_args()

    consumer = make_consumer(args.from_beginning)
    conn = make_db_conn()

    batch: list[tuple] = []
    total = 0
    t0 = time.perf_counter()
    last_report = t0

    print(f"Sink running (batch_size={args.batch_size}). Ctrl+C to stop.\n")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                if batch:
                    flush_batch(conn, batch)
                    total += len(batch)
                    batch.clear()
                continue

            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"Consumer error: {msg.error()}")
                continue

            event = json.loads(msg.value().decode("utf-8"))
            batch.append(event_to_row(event))

            if len(batch) >= args.batch_size:
                flush_batch(conn, batch)
                total += len(batch)
                batch.clear()
                consumer.commit(asynchronous=False)

            now = time.perf_counter()
            if now - last_report >= 5.0:
                elapsed = now - t0
                print(f"[{elapsed:6.1f}s]  {total:>10,} rows written  |  {total/elapsed:>8,.0f} rows/s")
                last_report = now

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
