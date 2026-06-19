"""
RQ3 Evaluation: Architecture reliability and throughput.

Measures end-to-end characteristics of the event-driven streaming platform:
  - Detection latency percentiles (P50/P75/P95/P99) from anomaly_alerts
  - Feature pipeline throughput (rows/min) from sales_features
  - Event ingestion rate (events/sec) from sales_events
  - Anomaly alert rate (alerts / features checked)
  - Data volume summary

Results are saved to evaluation/experiments/rq3_results.json (overwritten each run).

Usage:
  python evaluation/evaluate_rq3.py
  python evaluation/evaluate_rq3.py --db-host localhost
"""

import argparse
import json
from datetime import timezone
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

OUT_FILE = Path("evaluation/experiments/rq3_results.json")


def connect(host: str):
    return psycopg2.connect(
        host=host, port=5432, dbname="retail", user="retail", password="retail"
    )


def query_one(cur, sql: str):
    cur.execute(sql)
    row = cur.fetchone()
    return dict(row) if row else {}


def query_all(cur, sql: str):
    cur.execute(sql)
    return [dict(r) for r in cur.fetchall()]


# ── Latency percentiles ────────────────────────────────────────────────────────

def measure_latency(cur) -> dict:
    print("[1] Measuring detection latency percentiles...")
    row = query_one(cur, """
        SELECT
            COUNT(*)                                                              AS total_alerts,
            ROUND(AVG(detection_latency_ms)::numeric)                            AS avg_ms,
            ROUND(MIN(detection_latency_ms)::numeric)                            AS min_ms,
            ROUND(MAX(detection_latency_ms)::numeric)                            AS max_ms,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
                  (ORDER BY detection_latency_ms)::numeric)                      AS p50_ms,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
                  (ORDER BY detection_latency_ms)::numeric)                      AS p75_ms,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
                  (ORDER BY detection_latency_ms)::numeric)                      AS p95_ms,
            ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP
                  (ORDER BY detection_latency_ms)::numeric)                      AS p99_ms
        FROM anomaly_alerts
        WHERE detection_latency_ms IS NOT NULL
          AND detection_latency_ms > 0
          AND detection_latency_ms < 600000
    """)
    if not row or not row.get("total_alerts"):
        print("  WARN: no valid latency rows found.")
        return {}
    result = {k: int(v) for k, v in row.items() if v is not None}
    print(f"  alerts={result['total_alerts']:,}  "
          f"avg={result['avg_ms']:,}ms  "
          f"P50={result['p50_ms']:,}ms  "
          f"P95={result['p95_ms']:,}ms  "
          f"P99={result['p99_ms']:,}ms")
    return result


# ── Feature pipeline throughput ────────────────────────────────────────────────

def measure_feature_throughput(cur) -> dict:
    print("[2] Measuring feature pipeline throughput...")

    # Rows per minute during active window
    rows = query_all(cur, """
        SELECT DATE_TRUNC('minute', inserted_at) AS minute,
               COUNT(*) AS rows_per_min
        FROM sales_features
        WHERE inserted_at IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)
    if not rows:
        print("  WARN: no timestamped feature rows.")
        return {}

    counts = [r["rows_per_min"] for r in rows]
    active_minutes = len(counts)
    total_rows = sum(counts)
    peak_rows_per_min = max(counts)
    avg_rows_per_min = total_rows / active_minutes if active_minutes else 0

    result = {
        "total_feature_rows": int(total_rows),
        "active_minutes": int(active_minutes),
        "avg_rows_per_min": round(float(avg_rows_per_min), 1),
        "peak_rows_per_min": int(peak_rows_per_min),
    }
    print(f"  total={total_rows:,}  active_minutes={active_minutes}  "
          f"avg={avg_rows_per_min:.0f}/min  peak={peak_rows_per_min}/min")
    return result


# ── Event ingestion throughput ─────────────────────────────────────────────────

def measure_event_throughput(cur) -> dict:
    print("[3] Measuring event ingestion throughput...")

    row = query_one(cur, "SELECT COUNT(*) AS total FROM sales_events")
    total_events = int(row.get("total") or 0)

    # Events per second bucketed by second using the time column
    rows = query_all(cur, """
        SELECT DATE_TRUNC('second', time) AS second,
               COUNT(*) AS events_per_sec
        FROM sales_events
        GROUP BY 1
        ORDER BY 1
    """)
    if not rows:
        print(f"  total_events={total_events:,}  (no per-second data)")
        return {"total_events": total_events}

    counts = [r["events_per_sec"] for r in rows]
    peak_eps = max(counts)
    # Use top 10% of seconds as "sustained" throughput
    sorted_counts = sorted(counts, reverse=True)
    top10_count = max(1, len(sorted_counts) // 10)
    sustained_eps = sum(sorted_counts[:top10_count]) / top10_count

    result = {
        "total_events": total_events,
        "peak_events_per_sec": int(peak_eps),
        "sustained_events_per_sec": round(float(sustained_eps), 1),
    }
    print(f"  total={total_events:,}  peak={peak_eps}/s  sustained={sustained_eps:.0f}/s")
    return result


# ── Alert rate ─────────────────────────────────────────────────────────────────

def measure_alert_rate(cur) -> dict:
    print("[4] Measuring anomaly alert rate...")

    r_feat = query_one(cur, "SELECT COUNT(*) AS total FROM sales_features")
    r_alert = query_one(cur, "SELECT COUNT(*) AS total FROM anomaly_alerts")

    features = int(r_feat.get("total") or 0)
    alerts   = int(r_alert.get("total") or 0)
    rate_pct = round(alerts / features * 100, 2) if features else 0.0

    result = {
        "features_checked": features,
        "alerts_generated": alerts,
        "alert_rate_pct": rate_pct,
    }
    print(f"  features={features:,}  alerts={alerts:,}  rate={rate_pct}%")
    return result


# ── Data volume summary ────────────────────────────────────────────────────────

def measure_data_volume(cur) -> dict:
    print("[5] Summarising data volume...")

    r_ev   = query_one(cur, "SELECT COUNT(*) AS c FROM sales_events")
    r_feat = query_one(cur, "SELECT COUNT(*) AS c FROM sales_features")
    r_al   = query_one(cur, "SELECT COUNT(*) AS c FROM anomaly_alerts")
    r_fc   = query_one(cur, "SELECT COUNT(*) AS c FROM forecast_results")

    result = {
        "sales_events":    int(r_ev.get("c")   or 0),
        "sales_features":  int(r_feat.get("c") or 0),
        "anomaly_alerts":  int(r_al.get("c")   or 0),
        "forecast_results": int(r_fc.get("c")  or 0),
    }
    for k, v in result.items():
        print(f"  {k}: {v:,}")
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RQ3 architecture metrics evaluation")
    parser.add_argument("--db-host", default="localhost")
    args = parser.parse_args()

    print("\n=== RQ3 Evaluation: Architecture Reliability & Throughput ===\n")

    conn = connect(args.db_host)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        latency   = measure_latency(cur)
        feat_tp   = measure_feature_throughput(cur)
        event_tp  = measure_event_throughput(cur)
        alert_rate = measure_alert_rate(cur)
        volume    = measure_data_volume(cur)
    conn.close()

    results = {
        "latency_ms":           latency,
        "feature_throughput":   feat_tp,
        "event_throughput":     event_tp,
        "alert_rate":           alert_rate,
        "data_volume":          volume,
        "batch_baseline_ms":    87400,
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists():
        OUT_FILE.unlink()
    OUT_FILE.write_text(json.dumps(results, indent=2, default=str))

    print(f"\n=== Results saved to {OUT_FILE} ===\n")

    # Human-readable summary
    if latency:
        print("── Latency ──────────────────────────────────────────")
        print(f"  Batch baseline : 87,400 ms")
        print(f"  Streaming avg  : {latency.get('avg_ms', '?'):,} ms")
        print(f"  P50            : {latency.get('p50_ms', '?'):,} ms")
        print(f"  P95            : {latency.get('p95_ms', '?'):,} ms")
        print(f"  P99            : {latency.get('p99_ms', '?'):,} ms")
    if event_tp:
        print("── Throughput ───────────────────────────────────────")
        print(f"  Peak           : {event_tp.get('peak_events_per_sec', '?'):,} ev/s")
        print(f"  Sustained      : {event_tp.get('sustained_events_per_sec', '?'):,} ev/s")
    if alert_rate:
        print("── Alert rate ───────────────────────────────────────")
        print(f"  {alert_rate.get('alert_rate_pct', '?')}%  "
              f"({alert_rate.get('alerts_generated', '?'):,} / "
              f"{alert_rate.get('features_checked', '?'):,})")
    print()


if __name__ == "__main__":
    main()
