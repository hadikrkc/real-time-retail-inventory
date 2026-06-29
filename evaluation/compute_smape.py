"""Compute sMAPE alongside MAPE for batch and streaming forecasts (RQ2 supplement)."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import psycopg2
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DB_PARAMS = dict(host="timescaledb", port=5432, dbname="retail", user="retail", password="retail")
DATA_DIR = Path("/data/m5")

STORE_ID = "CA_1"
TEST_DAYS = 7


def connect():
    return psycopg2.connect(**DB_PARAMS)


def esm_forecast(series, horizon):
    if len(series) < 3 or series.sum() == 0:
        return np.full(horizon, series[-1] if len(series) else 0.0)
    try:
        fit = ExponentialSmoothing(series, trend="add", seasonal=None).fit(optimized=True)
        return np.clip(fit.forecast(horizon), 0, None)
    except Exception:
        return np.full(horizon, float(series[-1]))


def smape(actual, predicted):
    denom = np.abs(actual) + np.abs(predicted)
    mask = denom > 0
    if mask.sum() == 0:
        return None
    return float(np.mean(2 * np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def mape_nonzero(actual, predicted):
    mask = actual > 0
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100)


print(f"Loading M5 CSV for store={STORE_ID}...")
sales = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")
calendar = pd.read_csv(DATA_DIR / "calendar.csv", usecols=["date", "d"])
calendar["date"] = pd.to_datetime(calendar["date"])

store_sales = sales[sales["store_id"] == STORE_ID].copy()
day_cols = [c for c in store_sales.columns if c.startswith("d_")]

records = []
for _, row in store_sales.iterrows():
    for dcol in day_cols:
        records.append({"item_id": row["item_id"], "day": dcol, "sales_qty": row[dcol]})

m5_df = pd.DataFrame(records)
m5_df = m5_df.merge(calendar.rename(columns={"d": "day"}), on="day")
m5_df = m5_df.sort_values(["item_id", "date"])

print(f"Loading streaming features from DB...")
conn = connect()
with conn.cursor() as cur:
    cur.execute("""
        SELECT item_id, time, rolling_avg_7d
        FROM sales_features
        WHERE store_id = %s AND rolling_avg_7d IS NOT NULL
        ORDER BY item_id, time
    """, (STORE_ID,))
    sf_rows = cur.fetchall()
conn.close()

sf_df = pd.DataFrame(sf_rows, columns=["item_id", "date", "rolling_avg_7d"])
sf_df["date"] = pd.to_datetime(sf_df["date"]).dt.tz_localize(None)

# Get test period end date
max_date = sf_df["date"].max()
test_end = max_date
test_start = test_end - pd.Timedelta(days=TEST_DAYS - 1)
train_end = test_start - pd.Timedelta(days=1)

print(f"Items: {store_sales['item_id'].nunique()}  |  test period: {test_start.date()} → {test_end.date()}")

batch_mapes, stream_mapes = [], []
batch_smapes, stream_smapes = [], []
n_skipped = 0

items = store_sales["item_id"].unique()
for i, item_id in enumerate(items):
    if i % 500 == 0 and i > 0:
        print(f"  {i}/{len(items)} items ...")

    item_m5 = m5_df[m5_df["item_id"] == item_id].sort_values("date")
    item_sf = sf_df[sf_df["item_id"] == item_id].sort_values("date")

    if item_m5.empty or item_sf.empty:
        n_skipped += 1
        continue

    # Test period actual values
    test_mask = (item_m5["date"] >= test_start) & (item_m5["date"] <= test_end)
    actual = item_m5[test_mask]["sales_qty"].values

    if len(actual) != TEST_DAYS:
        n_skipped += 1
        continue

    # Batch: full history train → ESM forecast
    train_mask = item_m5["date"] <= train_end
    train_series = item_m5[train_mask]["sales_qty"].values
    batch_pred = esm_forecast(train_series, TEST_DAYS)

    # Streaming: last known rolling_avg_7d → ESM forecast
    sf_train = item_sf[item_sf["date"] <= train_end]
    if sf_train.empty:
        n_skipped += 1
        continue

    stream_series = sf_train["rolling_avg_7d"].values
    stream_pred = esm_forecast(stream_series, TEST_DAYS)

    bm = mape_nonzero(actual, batch_pred)
    sm = mape_nonzero(actual, stream_pred)
    bs = smape(actual, batch_pred)
    ss = smape(actual, stream_pred)

    if bm is not None and sm is not None and bs is not None and ss is not None:
        batch_mapes.append(bm)
        stream_mapes.append(sm)
        batch_smapes.append(bs)
        stream_smapes.append(ss)

n_valid = len(batch_mapes)
print(f"Valid pairs: {n_valid}  skipped: {n_skipped}")
print()
print(f"=== MAPE ===")
print(f"Batch   avg MAPE  : {np.mean(batch_mapes):.2f}%  (median {np.median(batch_mapes):.2f}%)")
print(f"Streaming avg MAPE: {np.mean(stream_mapes):.2f}%  (median {np.median(stream_mapes):.2f}%)")
print(f"Gap     : {np.mean(stream_mapes) - np.mean(batch_mapes):.2f} pp")
print()
print(f"=== sMAPE ===")
print(f"Batch   avg sMAPE  : {np.mean(batch_smapes):.2f}%  (median {np.median(batch_smapes):.2f}%)")
print(f"Streaming avg sMAPE: {np.mean(stream_smapes):.2f}%  (median {np.median(stream_smapes):.2f}%)")
print(f"Gap     : {np.mean(stream_smapes) - np.mean(batch_smapes):.2f} pp")
