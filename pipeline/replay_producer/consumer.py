"""
Smoke-test consumer: reads from retail.sales.events and prints throughput stats.

Usage:
  python consumer.py                     # read from latest offset
  python consumer.py --from-beginning    # replay all stored messages
  python consumer.py --sample 5          # print first N events in full, then stats only
"""

import argparse
import json
import time
from collections import defaultdict

from confluent_kafka import Consumer, KafkaError

TOPIC = "retail.sales.events"
BOOTSTRAP_SERVERS = "localhost:9092"


def make_consumer(from_beginning: bool) -> Consumer:
    offset_reset = "earliest" if from_beginning else "latest"
    conf = {
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "group.id": "smoke-test-consumer",
        "auto.offset.reset": offset_reset,
        "enable.auto.commit": True,
    }
    consumer = Consumer(conf)
    consumer.subscribe([TOPIC])
    print(f"Connected to Kafka. Listening on '{TOPIC}' (offset={offset_reset})")
    return consumer


def main():
    parser = argparse.ArgumentParser(description="Kafka smoke-test consumer")
    parser.add_argument("--from-beginning", action="store_true",
                        help="Read from the earliest offset")
    parser.add_argument("--sample", type=int, default=3,
                        help="Number of full events to print (rest: stats only)")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Seconds to wait for messages before exiting")
    args = parser.parse_args()

    consumer = make_consumer(args.from_beginning)

    total = 0
    printed = 0
    store_counts: dict[str, int] = defaultdict(int)
    t0 = time.perf_counter()
    last_report = t0
    idle_since = time.perf_counter()

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                if time.perf_counter() - idle_since > args.timeout:
                    print(f"No messages for {args.timeout}s, exiting.")
                    break
                continue

            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"Consumer error: {msg.error()}")
                continue

            idle_since = time.perf_counter()
            event = json.loads(msg.value().decode("utf-8"))
            total += 1
            store_counts[event.get("store_id", "?")] += 1

            if printed < args.sample:
                print(f"\n--- Event #{total} ---")
                print(json.dumps(event, indent=2))
                printed += 1

            now = time.perf_counter()
            if now - last_report >= 5.0:
                elapsed = now - t0
                rate = total / elapsed
                print(f"[{elapsed:6.1f}s]  {total:>10,} events  |  {rate:>10,.0f} ev/s  |  stores: {dict(store_counts)}")
                last_report = now

    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

    elapsed = time.perf_counter() - t0
    rate = total / elapsed if elapsed > 0 else 0
    print(f"\n{'='*60}")
    print(f"Total consumed : {total:,} events")
    print(f"Elapsed        : {elapsed:.1f}s")
    print(f"Avg throughput : {rate:,.0f} events/sec")
    print(f"Stores seen    : {dict(store_counts)}")


if __name__ == "__main__":
    main()
