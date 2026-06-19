"""
Batch baseline — simulates a daily cron job that runs on all historical data.

This is the CONTROL GROUP for RQ1 (anomaly latency) and RQ2 (forecast accuracy).
It reads from TimescaleDB, runs Isolation Forest + AutoETS, and records wall-clock time.

Usage:
  python scripts/batch_baseline.py --as-of-date 2011-02-28
  python scripts/batch_baseline.py --as-of-date 2011-02-28 --store CA_1
"""

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
np.seterr(all="ignore")

import pandas as pd
import psycopg2
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DB_DSN = "host=localhost port=5432 dbname=retail user=retail password=retail"
RESULTS_DIR = Path(__file__).parent.parent / "evaluation" / "experiments"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sales(conn, as_of_date: str, store: str | None) -> pd.DataFrame:
    store_filter = "AND store_id = %s" if store else ""
    sql = f"""
        SELECT time, item_id, store_id, sales_qty
        FROM sales_events
        WHERE time <= %s
        {store_filter}
        ORDER BY time
    """
    params = (as_of_date, store) if store else (as_of_date,)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=cols)
    df["time"] = pd.to_datetime(df["time"])
    return df


# ── Anomaly detection ─────────────────────────────────────────────────────────

def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Isolation Forest on daily sales per item.
    Returns DataFrame with anomaly flag (-1 = anomaly, 1 = normal).
    """
    pivot = (
        df.groupby(["item_id", pd.Grouper(key="time", freq="D")])["sales_qty"]
        .sum()
        .unstack(level=0)
        .fillna(0)
    )

    clf = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    labels = clf.fit_predict(pivot)

    result = pd.DataFrame({
        "time": pivot.index,
        "anomaly": labels,
    })
    result["is_anomaly"] = result["anomaly"] == -1
    return result


# ── Demand forecasting ────────────────────────────────────────────────────────

def run_forecasting(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """
    Holt-Winters ExponentialSmoothing per item via statsmodels.
    Returns forecast DataFrame with predicted sales for next `horizon` days.
    """
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert(None)
    daily = (
        df.groupby(["item_id", pd.Grouper(key="time", freq="D")])["sales_qty"]
        .sum()
        .reset_index()
    )

    records = []
    for item_id, grp in daily.groupby("item_id"):
        series = grp.set_index("time")["sales_qty"].asfreq("D").fillna(0)
        if len(series) < 10:
            continue
        try:
            model = ExponentialSmoothing(series, trend="add", seasonal=None)
            fit = model.fit(optimized=True)
            preds = fit.forecast(horizon)
            for step, (date, val) in enumerate(preds.items(), 1):
                records.append({"item_id": item_id, "horizon_day": step,
                                 "forecast_date": date, "forecast_qty": round(val, 2)})
        except Exception:
            continue

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch baseline cron job")
    parser.add_argument("--as-of-date", required=True,
                        help="Run the job as if today is this date (YYYY-MM-DD)")
    parser.add_argument("--store", default=None,
                        help="Limit to a single store_id (e.g. CA_1)")
    parser.add_argument("--horizon", type=int, default=7,
                        help="Forecast horizon in days")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics = {"run_id": run_id, "as_of_date": args.as_of_date, "store": args.store}

    print(f"Batch baseline | as_of={args.as_of_date} | store={args.store or 'ALL'}\n")

    conn = psycopg2.connect(DB_DSN)

    # ── Load ──
    t_load = time.perf_counter()
    df = load_sales(conn, args.as_of_date, args.store)
    conn.close()
    metrics["load_sec"] = round(time.perf_counter() - t_load, 3)
    print(f"Loaded {len(df):,} rows in {metrics['load_sec']}s")

    # ── Anomaly detection ──
    t_anom = time.perf_counter()
    anomalies = run_anomaly_detection(df)
    metrics["anomaly_sec"] = round(time.perf_counter() - t_anom, 3)
    n_anomalies = anomalies["is_anomaly"].sum()
    print(f"Anomaly detection: {n_anomalies} anomalous days detected in {metrics['anomaly_sec']}s")

    # ── Forecasting ──
    t_fc = time.perf_counter()
    forecasts = run_forecasting(df, horizon=args.horizon)
    metrics["forecast_sec"] = round(time.perf_counter() - t_fc, 3)
    print(f"Forecasting: {len(forecasts):,} forecasts generated in {metrics['forecast_sec']}s")

    # ── Total batch latency ──
    metrics["total_sec"] = round(
        metrics["load_sec"] + metrics["anomaly_sec"] + metrics["forecast_sec"], 3
    )

    # ── Save results ──
    out_path = RESULTS_DIR / f"batch_baseline_{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    anomalies.to_csv(RESULTS_DIR / f"anomalies_{run_id}.csv", index=False)
    forecasts.to_csv(RESULTS_DIR / f"forecasts_{run_id}.csv", index=False)

    print(f"\n{'='*50}")
    print(f"Total batch latency : {metrics['total_sec']}s")
    print(f"Results saved to    : {RESULTS_DIR}")


if __name__ == "__main__":
    import sys
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
