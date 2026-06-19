"""
RQ2 Evaluation: Streaming features vs Batch (static) forecast accuracy.

Both approaches use the same model (ExponentialSmoothing) and the same test
period. The only difference is the input features:
  Streaming : sales_features.rolling_avg_7d  (Spark 7-day rolling window)
  Batch     : raw daily sales series from M5 CSV (full historical series)

Usage:
  python evaluation/evaluate_forecasts.py \
    --store CA_1 --test-days 7
"""

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

np.seterr(all="ignore")

DB_PARAMS = dict(host="localhost", port=5432, dbname="retail",
                 user="retail", password="retail")
DATA_DIR  = Path(__file__).parents[1] / "data" / "m5"
OUT_DIR   = Path(__file__).parent / "experiments"
OUT_DIR.mkdir(exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_m5_series(store_id: str) -> pd.DataFrame:
    """Load daily sales series for a given store from the M5 CSV files."""
    print("Loading M5 CSV...")
    sales    = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", usecols=["date", "d"])
    calendar["date"] = pd.to_datetime(calendar["date"])

    store_sales = sales[sales["store_id"] == store_id].copy()
    day_cols = [c for c in store_sales.columns if c.startswith("d_")]

    records = []
    for _, row in store_sales.iterrows():
        for dcol in day_cols:
            records.append({
                "item_id":   row["item_id"],
                "day":       dcol,
                "sales_qty": row[dcol],
            })

    df = pd.DataFrame(records)
    df = df.merge(calendar.rename(columns={"d": "day"}), on="day")
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    print(f"  {len(df):,} rows loaded ({store_sales.shape[0]} items).")
    return df


def load_streaming_features(store_id: str) -> pd.DataFrame:
    """Load rolling window features from TimescaleDB sales_features table."""
    conn = psycopg2.connect(**DB_PARAMS)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT item_id, time::date AS date, rolling_avg_7d
            FROM sales_features
            WHERE store_id = %s AND rolling_avg_7d IS NOT NULL
            ORDER BY item_id, time ASC
        """, (store_id,))
        rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    print(f"  {len(df):,} feature rows loaded.")
    return df


# ── Forecasting ───────────────────────────────────────────────────────────────

def esm_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    if len(series) < 3 or series.sum() == 0:
        return np.full(horizon, series[-1] if len(series) > 0 else 0.0)
    try:
        fit = ExponentialSmoothing(
            series, trend="add", seasonal=None,
        ).fit(optimized=True)
        preds = fit.forecast(horizon)
        return np.clip(preds, 0, None)
    except Exception:
        return np.full(horizon, float(series[-1]))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    mask  = denom > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_batch(m5_df: pd.DataFrame, test_start_date, horizon: int) -> dict:
    """Batch approach: fit ESM on raw daily sales series, forecast test period."""
    print("\n[Batch] ESM on raw daily sales series...")

    train_df = m5_df[m5_df["date"] < test_start_date]
    test_df  = m5_df[(m5_df["date"] >= test_start_date)]

    items     = train_df["item_id"].unique()
    all_mape  = []
    all_smape = []
    n_skipped = 0

    for i, item_id in enumerate(items, 1):
        if i % 500 == 0:
            print(f"  {i}/{len(items)} items processed...")
        train_series = train_df[train_df["item_id"] == item_id]["sales_qty"].values
        test_series  = test_df[test_df["item_id"] == item_id]["sales_qty"].values[:horizon]

        if len(test_series) < horizon:
            n_skipped += 1
            continue

        preds = esm_forecast(train_series.astype(float), horizon)
        m = mape(test_series.astype(float), preds)
        s = smape(test_series.astype(float), preds)
        if not np.isnan(m):
            all_mape.append(m)
        if not np.isnan(s):
            all_smape.append(s)

    result = {
        "method":       "batch",
        "items":        len(items),
        "skipped":      n_skipped,
        "avg_mape":     float(np.mean(all_mape))  if all_mape  else None,
        "median_mape":  float(np.median(all_mape)) if all_mape  else None,
        "avg_smape":    float(np.mean(all_smape)) if all_smape else None,
        "horizon_days": horizon,
    }
    print(f"  Items     : {result['items']:,}")
    print(f"  Avg MAPE  : {result['avg_mape']:.2f} %")
    print(f"  Avg sMAPE : {result['avg_smape']:.2f} %")
    return result


def evaluate_streaming(
    feat_df: pd.DataFrame,
    m5_df:   pd.DataFrame,
    test_start_date,
    horizon: int,
) -> dict:
    """Streaming approach: fit ESM on rolling_avg_7d feature series, forecast test period."""
    print("\n[Streaming] ESM on rolling_avg_7d feature series...")

    train_feat = feat_df[feat_df["date"] < test_start_date]
    test_df    = m5_df[(m5_df["date"] >= test_start_date)]

    items     = train_feat["item_id"].unique()
    all_mape  = []
    all_smape = []
    n_skipped = 0

    for i, item_id in enumerate(items, 1):
        if i % 500 == 0:
            print(f"  {i}/{len(items)} items processed...")
        feat_series = train_feat[train_feat["item_id"] == item_id]["rolling_avg_7d"].values
        test_series = test_df[test_df["item_id"] == item_id]["sales_qty"].values[:horizon]

        if len(feat_series) < 3 or len(test_series) < horizon:
            n_skipped += 1
            continue

        preds = esm_forecast(feat_series.astype(float), horizon)
        m = mape(test_series.astype(float), preds)
        s = smape(test_series.astype(float), preds)
        if not np.isnan(m):
            all_mape.append(m)
        if not np.isnan(s):
            all_smape.append(s)

    result = {
        "method":       "streaming",
        "items":        len(items),
        "skipped":      n_skipped,
        "avg_mape":     float(np.mean(all_mape))  if all_mape  else None,
        "median_mape":  float(np.median(all_mape)) if all_mape  else None,
        "avg_smape":    float(np.mean(all_smape)) if all_smape else None,
        "horizon_days": horizon,
    }
    print(f"  Items     : {result['items']:,}")
    print(f"  Avg MAPE  : {result['avg_mape']:.2f} %")
    print(f"  Avg sMAPE : {result['avg_smape']:.2f} %")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RQ2 evaluation — streaming vs batch MAPE")
    parser.add_argument("--store",       default="CA_1")
    parser.add_argument("--test-days",   type=int, default=7,
                        help="Number of days to hold out for testing (default: 7)")
    args = parser.parse_args()

    print(f"=== RQ2 Evaluation: {args.store} | test_horizon={args.test_days} days ===\n")

    m5_df = load_m5_series(args.store)

    print("Loading streaming features...")
    feat_df = load_streaming_features(args.store)

    if feat_df.empty:
        print("ERROR: sales_features is empty. Run feature_pipeline.py first.")
        return

    last_feat_date  = feat_df["date"].max()
    test_start_date = last_feat_date - pd.Timedelta(days=args.test_days)

    print(f"\nTrain : M5 start — {test_start_date.date()}")
    print(f"Test  : {test_start_date.date()} — {last_feat_date.date()} ({args.test_days} days)\n")

    batch_result     = evaluate_batch(m5_df, test_start_date, args.test_days)
    streaming_result = evaluate_streaming(feat_df, m5_df, test_start_date, args.test_days)

    print("\n" + "="*55)
    print(f"{'Metric':<20} {'Batch':>15} {'Streaming':>15}")
    print("-"*55)
    print(f"{'Avg MAPE (%)':<20} {batch_result['avg_mape']:>15.2f} {streaming_result['avg_mape']:>15.2f}")
    print(f"{'Median MAPE (%)':<20} {batch_result['median_mape']:>15.2f} {streaming_result['median_mape']:>15.2f}")
    print(f"{'Avg sMAPE (%)':<20} {batch_result['avg_smape']:>15.2f} {streaming_result['avg_smape']:>15.2f}")
    print(f"{'Items':<20} {batch_result['items']:>15,} {streaming_result['items']:>15,}")
    print("="*55)

    out = {
        "store":      args.store,
        "test_days":  args.test_days,
        "test_start": str(test_start_date.date()),
        "batch":      batch_result,
        "streaming":  streaming_result,
    }
    out_path = OUT_DIR / f"rq2_{args.store}_horizon{args.test_days}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
