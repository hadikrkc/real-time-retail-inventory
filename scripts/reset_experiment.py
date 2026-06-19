"""
Resets the experiment environment for a clean run.

Default (--level=soft):
  - Truncate anomaly_alerts
  - Truncate sales_features
  - Delete Spark feature checkpoint

--level=hard (includes all of the above, plus):
  - Truncate sales_events
  - Reset Kafka consumer group offsets to earliest

Usage:
  python scripts/reset_experiment.py              # soft reset
  python scripts/reset_experiment.py --level hard # hard reset (includes events)
  python scripts/reset_experiment.py --dry-run    # show what would be done, no changes
"""

import argparse
import subprocess
import sys

import psycopg2

DB_PARAMS = dict(host="localhost", port=5432, dbname="retail",
                 user="retail", password="retail")

SPARK_CONTAINER   = "spark"
CHECKPOINT_DIR    = "/tmp/spark-checkpoints"

KAFKA_CONTAINER   = "kafka1"
KAFKA_SERVER      = "kafka1:19092"
CONSUMER_GROUPS   = ["timescale-sink"]


def run(cmd: str, dry: bool, label: str) -> None:
    print(f"  {'[DRY]' if dry else '[RUN]'} {label}")
    if not dry:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"         WARN: {result.stderr.strip()}")
        elif result.stdout.strip():
            print(f"         {result.stdout.strip()}")


def truncate_table(cur, table: str, dry: bool) -> None:
    print(f"  {'[DRY]' if dry else '[RUN]'} TRUNCATE {table}")
    if not dry:
        cur.execute(f"TRUNCATE {table}")


def main():
    parser = argparse.ArgumentParser(description="Reset experiment state")
    parser.add_argument("--level", choices=["soft", "hard"], default="soft",
                        help="soft: features+alerts+checkpoint | hard: +events+kafka")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without doing it")
    args = parser.parse_args()

    dry = args.dry_run
    print(f"\n=== Experiment Reset  [level={args.level}{'  DRY RUN' if dry else ''}] ===\n")

    # ── 1. DB tables ──────────────────────────────────────────────────────────
    print("[1] Truncating TimescaleDB tables...")
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = False
    with conn.cursor() as cur:
        truncate_table(cur, "anomaly_alerts", dry)
        truncate_table(cur, "sales_features",  dry)
        if args.level == "hard":
            truncate_table(cur, "sales_events", dry)
    if not dry:
        conn.commit()
    conn.close()
    print("  Done.\n")

    # ── 2. Spark checkpoint ───────────────────────────────────────────────────
    print("[2] Deleting Spark checkpoint...")
    run(
        f'docker exec {SPARK_CONTAINER} rm -rf {CHECKPOINT_DIR}',
        dry,
        f"docker exec {SPARK_CONTAINER} rm -rf {CHECKPOINT_DIR}",
    )
    print("  Done.\n")

    # ── 3. Kafka consumer group offsets (hard only) ───────────────────────────
    if args.level == "hard":
        print("[3] Resetting Kafka consumer group offsets...")
        for group in CONSUMER_GROUPS:
            run(
                f'docker exec {KAFKA_CONTAINER} kafka-consumer-groups '
                f'--bootstrap-server {KAFKA_SERVER} '
                f'--group {group} --reset-offsets --to-earliest --all-topics --execute',
                dry,
                f"reset offsets: group={group} to earliest",
            )
        print("  Done.\n")

    print("=== Reset complete ===\n")
    print("Next steps:")
    print("  1. Start Spark feature pipeline (stop it first if already running):")
    print("       docker exec spark spark-submit \\")
    print("         --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \\")
    print("         /opt/pipeline/stream_processor/feature_pipeline.py \\")
    print("         --kafka-server kafka1:19092 --db-host timescaledb --starting-offsets latest")
    print()
    print("  2. Start anomaly detector:")
    print("       python pipeline/stream_processor/anomaly_detector.py \\")
    print("         --db-host localhost --kafka-server localhost:9092")
    print()
    print("  3. Send data via producer:")
    print("       python pipeline/replay_producer/producer.py --start-day 1 --end-day 10")
    print()
    print("  4. Check detection latency:")
    print("       docker exec timescaledb psql -U retail -d retail -c \\")
    print('         "SELECT COUNT(*), ROUND(AVG(detection_latency_ms)) AS avg_ms,')
    print("          MIN(detection_latency_ms) AS min_ms, MAX(detection_latency_ms) AS max_ms")
    print('          FROM anomaly_alerts;"')


if __name__ == "__main__":
    main()
