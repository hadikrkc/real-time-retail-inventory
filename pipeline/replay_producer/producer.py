"""
Replay producer: reads M5 historical CSV data and publishes daily sales events
to a Kafka topic in chronological order, simulating a real-time stream.

Replay speed: --speed controls seconds to sleep between days.
  speed=1.0  → 1 real second per dataset day (~32 min for full 1941 days)
  speed=0.0  → publish as fast as possible (throughput test)

Usage:
  python producer.py --speed 0.0 --start-day 1 --end-day 30
"""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer
from confluent_kafka.error import KafkaException

try:
    _default_data = str(Path(__file__).parents[2] / "data" / "m5")
except IndexError:
    _default_data = "/data/m5"
DATA_DIR = Path(os.environ.get("M5_DATA_DIR", _default_data))
TOPIC = "retail.sales.events"
BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BROKERS", "localhost:9092")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading M5 calendar...")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", usecols=["date", "d"])
    calendar["date"] = pd.to_datetime(calendar["date"])

    print("Loading M5 sales (this takes ~10s)...")
    sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")
    return sales, calendar


def iter_day_events(sales: pd.DataFrame, calendar: pd.DataFrame, start_day: int, end_day: int):
    """Yields (date, list_of_events) for each day in [start_day, end_day]."""
    day_cols = [f"d_{i}" for i in range(start_day, end_day + 1)]
    day_cols = [c for c in day_cols if c in sales.columns]

    date_map = calendar.set_index("d")["date"].to_dict()

    meta_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    meta = sales[meta_cols]

    for day_col in day_cols:
        event_date = date_map.get(day_col)
        if event_date is None:
            continue

        day_sales = sales[day_col]
        events = []
        for idx, qty in day_sales.items():
            row = meta.iloc[idx]
            events.append({
                "event_id": str(uuid.uuid4()),
                "timestamp": event_date.isoformat(),
                "item_id": row["item_id"],
                "dept_id": row["dept_id"],
                "cat_id": row["cat_id"],
                "store_id": row["store_id"],
                "state_id": row["state_id"],
                "sales_qty": int(qty),
                "day": day_col,
            })
        yield event_date, events


def make_producer() -> Producer:
    conf = {
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "compression.type": "lz4",
        "batch.size": 256 * 1024,
        "linger.ms": 20,
        "acks": "1",
        "queue.buffering.max.messages": 100_000,
        "queue.buffering.max.kbytes": 512 * 1024,
    }
    producer = Producer(conf)
    print(f"Connected to Kafka at {BOOTSTRAP_SERVERS}")
    return producer


def delivery_report(err, msg):
    if err:
        print(f"Delivery failed: {err}")


def main():
    parser = argparse.ArgumentParser(description="M5 → Kafka replay producer")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Seconds to sleep between days (0 = max throughput)")
    parser.add_argument("--start-day", type=int, default=1)
    parser.add_argument("--end-day", type=int, default=1941)
    parser.add_argument("--topic", default=TOPIC)
    args = parser.parse_args()

    sales, calendar = load_data()
    producer = make_producer()

    total_events = 0
    t0 = time.perf_counter()

    for event_date, events in iter_day_events(sales, calendar, args.start_day, args.end_day):
        for event in events:
            producer.produce(
                topic=args.topic,
                key=f"{event['store_id']}_{event['item_id']}",
                value=json.dumps(event).encode("utf-8"),
                on_delivery=delivery_report,
            )
        producer.flush()
        total_events += len(events)

        elapsed = time.perf_counter() - t0
        rate = total_events / elapsed if elapsed > 0 else 0
        print(f"{event_date.date()}  |  {len(events):,} events  |  total {total_events:,}  |  {rate:,.0f} ev/s")

        if args.speed > 0:
            time.sleep(args.speed)

    producer.flush()
    elapsed = time.perf_counter() - t0
    print(f"\nDone. {total_events:,} events in {elapsed:.1f}s  ({total_events/elapsed:,.0f} ev/s avg)")


if __name__ == "__main__":
    main()
