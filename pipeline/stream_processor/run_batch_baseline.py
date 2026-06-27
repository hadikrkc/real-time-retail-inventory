"""
One-shot batch baseline for RQ1.

Simulates a traditional daily-batch anomaly detection job:
  1. Load ALL sales data from TimescaleDB
  2. Compute daily aggregate per (store, item)
  3. Train Isolation Forest and predict anomalies on the full dataset
  4. Record wall-clock time for each phase

Result is saved to --out-dir/batch_baseline_<timestamp>.json
Run via:  docker compose run batch-baseline
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
np.seterr(all="ignore")

import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.ensemble import IsolationForest

DB_HOST = os.environ.get("DB_HOST", "localhost")


def connect(host: str):
    return psycopg2.connect(
        host=host, port=5432, dbname="retail",
        user="retail", password="retail",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-host", default=DB_HOST)
    parser.add_argument("--out-dir", default="evaluation/experiments")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    metrics = {"run_id": run_id, "db_host": args.db_host}

    print(f"\n=== Batch Baseline RQ1  [{run_id}] ===\n")
    conn = connect(args.db_host)

    # ── Phase 1: Load data ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    print("[1/3] Loading sales data...")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT store_id, item_id,
                   DATE_TRUNC('day', time)::date AS day,
                   SUM(sales_qty)               AS daily_sales
            FROM sales_events
            GROUP BY store_id, item_id, day
            ORDER BY store_id, item_id, day
        """)
        rows = cur.fetchall()

    load_sec = time.perf_counter() - t0
    metrics["load_sec"]     = round(load_sec, 3)
    metrics["total_rows"]   = len(rows)
    print(f"  {len(rows):,} daily aggregates loaded in {load_sec:.2f}s")

    if not rows:
        print("ERROR: no data in sales_events. Run the producer first.")
        return

    # ── Phase 2: Build feature matrix ─────────────────────────────────────────
    print("[2/3] Building feature matrix...")
    t1 = time.perf_counter()

    from collections import defaultdict
    import pandas as pd

    df = pd.DataFrame(rows)
    df["daily_sales"] = df["daily_sales"].astype(float)

    # Rolling 7-day features per (store_id, item_id) — mirrors streaming pipeline
    feature_rows = []
    for (store, item), grp in df.groupby(["store_id", "item_id"]):
        series = grp.sort_values("day")["daily_sales"]
        roll   = series.rolling(7, min_periods=1)
        feat_df = pd.DataFrame({
            "rolling_avg_7d": roll.mean().values,
            "rolling_sum_7d": roll.sum().values,
            "event_count_7d": roll.count().values,
            "max_qty_7d":     roll.max().values,
            "min_qty_7d":     roll.min().values,
        })
        feature_rows.append(feat_df)

    X = pd.concat(feature_rows, ignore_index=True).fillna(0).values.astype(np.float32)
    feat_sec = time.perf_counter() - t1
    metrics["feature_sec"] = round(feat_sec, 3)
    print(f"  {X.shape[0]:,} feature vectors built in {feat_sec:.2f}s")

    # ── Phase 3: Isolation Forest ─────────────────────────────────────────────
    print("[3/3] Running Isolation Forest...")
    t2 = time.perf_counter()

    sample_size = min(100_000, X.shape[0])
    idx = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[idx]

    model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X_sample)
    preds = model.predict(X)
    n_anomalies = int((preds == -1).sum())

    detect_sec = time.perf_counter() - t2
    metrics["detect_sec"]    = round(detect_sec, 3)
    metrics["n_anomalies"]   = n_anomalies
    metrics["anomaly_rate"]  = round(n_anomalies / len(preds) * 100, 2)
    metrics["sample_size"]   = sample_size

    total_sec = load_sec + feat_sec + detect_sec
    metrics["total_batch_latency_ms"] = round(total_sec * 1000, 0)

    print(f"  {n_anomalies:,} anomalies detected in {detect_sec:.2f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = out_dir / f"batch_baseline_{run_id}.json"
    out_path.write_text(json.dumps(metrics, indent=2))

    print(f"\n{'='*50}")
    print(f"Total batch latency : {total_sec:.2f}s  ({metrics['total_batch_latency_ms']:,.0f} ms)")
    print(f"Results saved       : {out_path}")
    print(f"{'='*50}\n")

    conn.close()


if __name__ == "__main__":
    main()
