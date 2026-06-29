"""
#8 Evaluation: Kafka → TimescaleDB pipeline latency (ms-level, RQ3).

Two data sources:
  1. pipeline_metrics.processing_lag_ms  — written by sink.py, measures
     Kafka message timestamp → DB write wall-clock time in milliseconds.
  2. sales_events (ingested_at column) — arrival time distribution analysis.

If pipeline_metrics is empty, this script re-ingests nothing; it tells you
to re-run the producer + sink.py so sink can populate the table.

Results saved to: evaluation/experiments/streaming_latency_results.json

Usage:
  python evaluation/metrics/streaming_metrics.py
  python evaluation/metrics/streaming_metrics.py --db-host localhost
"""

import argparse
import json
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

DB_PARAMS = dict(host="localhost", port=5432, dbname="retail",
                 user="retail", password="retail")
OUT_DIR = Path(__file__).parents[1] / "experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def connect(host: str):
    p = dict(DB_PARAMS)
    p["host"] = host
    return psycopg2.connect(**p)


# ── pipeline_metrics (Kafka produce → DB write latency) ───────────────────────

def query_pipeline_metrics(conn) -> dict:
    """
    Reads processing_lag_ms from pipeline_metrics table.
    Populated by sink.py: lag_ms = time.time()*1000 - kafka_message_timestamp_ms
    """
    print("[1] Querying pipeline_metrics.processing_lag_ms ...")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT
                COUNT(*)                                                                AS n,
                ROUND(AVG(processing_lag_ms)::numeric, 2)                              AS avg_ms,
                ROUND(MIN(processing_lag_ms)::numeric, 2)                              AS min_ms,
                ROUND(MAX(processing_lag_ms)::numeric, 2)                              AS max_ms,
                ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
                      (ORDER BY processing_lag_ms)::numeric, 2)                        AS p50_ms,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
                      (ORDER BY processing_lag_ms)::numeric, 2)                        AS p75_ms,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
                      (ORDER BY processing_lag_ms)::numeric, 2)                        AS p95_ms,
                ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP
                      (ORDER BY processing_lag_ms)::numeric, 2)                        AS p99_ms
            FROM pipeline_metrics
            WHERE processing_lag_ms IS NOT NULL
              AND processing_lag_ms > 0
              AND processing_lag_ms < 30000
        """)
        row = cur.fetchone()

    if not row or not row["n"]:
        print("  WARN: pipeline_metrics is empty.")
        print("  → Re-run the producer + sink.py to populate this table.")
        print("    sink.py writes processing_lag_ms every 10 seconds during ingestion.")
        return {"available": False, "reason": "pipeline_metrics is empty"}

    n = int(row["n"])
    result = {
        "available":  True,
        "source":     "pipeline_metrics.processing_lag_ms",
        "metric":     "Kafka message timestamp → TimescaleDB write (wall-clock ms)",
        "n_samples":  n,
        "avg_ms":     float(row["avg_ms"]),
        "min_ms":     float(row["min_ms"]),
        "max_ms":     float(row["max_ms"]),
        "p50_ms":     float(row["p50_ms"]),
        "p75_ms":     float(row["p75_ms"]),
        "p95_ms":     float(row["p95_ms"]),
        "p99_ms":     float(row["p99_ms"]),
    }

    print(f"  n={n:,}  avg={result['avg_ms']:.1f}ms  "
          f"P50={result['p50_ms']:.1f}ms  "
          f"P95={result['p95_ms']:.1f}ms  "
          f"P99={result['p99_ms']:.1f}ms")
    return result


# ── sales_events.ingested_at gap analysis ─────────────────────────────────────

def query_ingestion_gaps(conn) -> dict:
    """
    Estimates per-event ingestion speed from sales_events.ingested_at.
    Computes inter-arrival gaps as a proxy for batch write latency.
    Not the same as Kafka→DB latency, but gives distribution shape.
    """
    print("[2] Analysing sales_events.ingested_at inter-arrival gaps ...")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) AS n,
                MIN(ingested_at) AS first_at,
                MAX(ingested_at) AS last_at,
                EXTRACT(EPOCH FROM (MAX(ingested_at) - MIN(ingested_at))) AS span_sec
            FROM sales_events
            WHERE ingested_at IS NOT NULL
        """)
        row = cur.fetchone()

    if not row or not row[0]:
        print("  WARN: no ingested_at data in sales_events.")
        return {"available": False}

    n        = int(row[0])
    span_sec = float(row[3]) if row[3] else 0
    avg_gap_ms = (span_sec / n * 1000) if n > 1 else 0

    print(f"  Events: {n:,}")
    print(f"  Ingestion span: {span_sec:.1f} s")
    print(f"  Avg inter-arrival gap: {avg_gap_ms:.3f} ms/event")

    return {
        "available":          True,
        "source":             "sales_events.ingested_at",
        "metric":             "inter-arrival time between DB writes (proxy for batch size / write speed)",
        "n_events":           n,
        "ingestion_span_sec": round(span_sec, 2),
        "avg_gap_ms":         round(avg_gap_ms, 4),
    }


# ── Row count timeline (events/sec from ingested_at) ──────────────────────────

def query_throughput_from_ingested_at(conn) -> dict:
    """Peak and sustained event throughput based on wall-clock ingestion time."""
    print("[3] Computing throughput from ingested_at timestamps ...")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT DATE_TRUNC('second', ingested_at) AS ts_sec,
                   COUNT(*) AS events
            FROM sales_events
            WHERE ingested_at IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """)
        rows = cur.fetchall()

    if not rows:
        print("  WARN: no ingested_at data.")
        return {"available": False}

    counts = [r["events"] for r in rows]
    peak   = int(max(counts))
    # Top-10% average as "sustained"
    sorted_desc = sorted(counts, reverse=True)
    top10       = sorted_desc[:max(1, len(sorted_desc) // 10)]
    sustained   = float(np.mean(top10))

    print(f"  Active seconds : {len(counts)}")
    print(f"  Peak           : {peak:,} ev/s")
    print(f"  Sustained (p90): {sustained:,.0f} ev/s")

    return {
        "available":             True,
        "n_active_seconds":      len(counts),
        "peak_events_per_sec":   peak,
        "sustained_events_per_sec": round(sustained, 1),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RQ3 streaming latency metrics — Kafka→DB ms-level"
    )
    parser.add_argument("--db-host", default="localhost")
    args = parser.parse_args()

    print("\n=== Streaming Pipeline Latency (ms-level) ===\n")

    conn = connect(args.db_host)
    with conn:
        pipeline_lag = query_pipeline_metrics(conn)
        ingestion    = query_ingestion_gaps(conn)
        throughput   = query_throughput_from_ingested_at(conn)
    conn.close()

    results = {
        "pipeline_lag_ms":  pipeline_lag,
        "ingestion_gaps":   ingestion,
        "throughput":       throughput,
        "notes": {
            "pipeline_lag_ms": (
                "Kafka message timestamp → TimescaleDB write. "
                "Populated by sink.py during ingestion. "
                "Re-run producer + sink.py if empty."
            ),
            "ingestion_gaps": (
                "Proxy metric derived from sales_events.ingested_at timestamps. "
                "Captures DB write speed, not true Kafka→DB latency."
            ),
        },
    }

    out_path = OUT_DIR / "streaming_latency_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved: {out_path}")

    # Human-readable summary
    print("\n── Summary ───────────────────────────────────────────────────")
    if pipeline_lag.get("available"):
        print("Kafka → DB latency (from pipeline_metrics):")
        print(f"  P50 = {pipeline_lag['p50_ms']:.1f} ms")
        print(f"  P95 = {pipeline_lag['p95_ms']:.1f} ms")
        print(f"  P99 = {pipeline_lag['p99_ms']:.1f} ms")
    else:
        print("pipeline_metrics: NOT AVAILABLE — re-run sink.py to populate")

    if throughput.get("available"):
        print(f"\nThroughput (from ingested_at):")
        print(f"  Peak      = {throughput['peak_events_per_sec']:,} ev/s")
        print(f"  Sustained = {throughput['sustained_events_per_sec']:,} ev/s")


if __name__ == "__main__":
    main()
