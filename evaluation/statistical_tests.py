"""
#1 Evaluation: Statistical significance tests for RQ1 and RQ2.

RQ1 — Anomaly detection latency (Wilcoxon signed-rank):
  H0: median streaming detection latency = batch baseline (87,400 ms)
  H1: median streaming latency < batch baseline
  Data: anomaly_alerts.detection_latency_ms from TimescaleDB

RQ2 — Forecast accuracy (Wilcoxon + Diebold-Mariano):
  H0: no significant difference in per-item MAPE between batch and streaming
  H1: MAPE difference is significant
  Data: per-item MAPE recomputed from sales_features (DB) + M5 CSV

Results saved to: evaluation/experiments/statistical_tests_results.json

Usage:
  python evaluation/statistical_tests.py
  python evaluation/statistical_tests.py --store CA_1 --test-days 7
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
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing

np.seterr(all="ignore")

DB_PARAMS = dict(host="localhost", port=5432, dbname="retail",
                 user="retail", password="retail")
DATA_DIR  = Path(__file__).parents[1] / "data" / "m5"
OUT_DIR   = Path(__file__).parent / "experiments"
OUT_DIR.mkdir(exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def connect():
    return psycopg2.connect(**DB_PARAMS)


def measure_batch_baseline_ms() -> tuple[float, dict]:
    """
    Actually measures RQ1 batch latency: load sales_features from DB,
    train IsolationForest on a sample, predict on all rows.
    Returns (total_ms, detail_dict).
    """
    import time as _time
    from sklearn.ensemble import IsolationForest

    print("  [batch] Loading features from DB...")
    t0 = _time.perf_counter()
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT rolling_avg_7d, rolling_sum_7d, event_count_7d,
                   max_qty_7d, min_qty_7d
              FROM sales_features
             WHERE rolling_avg_7d IS NOT NULL
        """)
        rows = cur.fetchall()
    conn.close()
    load_sec = _time.perf_counter() - t0

    if not rows:
        raise RuntimeError("sales_features is empty — run the replay + Spark pipeline first.")

    X = np.array(rows, dtype=np.float32)
    X = np.nan_to_num(X)
    print(f"  [batch] {len(X):,} rows loaded in {load_sec:.2f}s")

    t1 = _time.perf_counter()
    sample = min(100_000, len(X))
    idx = np.random.choice(len(X), sample, replace=False)
    model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X[idx])
    train_sec = _time.perf_counter() - t1

    t2 = _time.perf_counter()
    preds = model.predict(X)
    predict_sec = _time.perf_counter() - t2

    total_ms = (load_sec + train_sec + predict_sec) * 1000
    print(f"  [batch] load={load_sec:.2f}s  train={train_sec:.2f}s  predict={predict_sec:.2f}s")
    print(f"  [batch] Total batch latency: {total_ms:,.0f} ms")

    return total_ms, {
        "n_rows":      len(X),
        "sample_size": sample,
        "load_sec":    round(load_sec, 3),
        "train_sec":   round(train_sec, 3),
        "predict_sec": round(predict_sec, 3),
        "total_ms":    round(total_ms, 1),
    }


def esm_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    if len(series) < 3 or series.sum() == 0:
        return np.full(horizon, series[-1] if len(series) else 0.0)
    try:
        fit = ExponentialSmoothing(series, trend="add", seasonal=None).fit(optimized=True)
        return np.clip(fit.forecast(horizon), 0, None)
    except Exception:
        return np.full(horizon, float(series[-1]))


def item_mape(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    mask = actual > 0
    if mask.sum() == 0:
        return None
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100)


# ── RQ1: Wilcoxon on streaming vs batch latency ────────────────────────────────

def test_rq1_wilcoxon(batch_baseline_ms: float | None = None) -> dict:
    print("\n=== RQ1: Latency Significance Test (Wilcoxon signed-rank) ===")

    batch_detail = {}
    if batch_baseline_ms is None:
        print("  Measuring batch baseline now (load + train + predict)...")
        batch_baseline_ms, batch_detail = measure_batch_baseline_ms()

    print(f"  H0: median streaming latency = {batch_baseline_ms:,.0f} ms (batch baseline)")
    print(f"  H1: median streaming latency < {batch_baseline_ms:,.0f} ms\n")

    conn = connect()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT detection_latency_ms
            FROM anomaly_alerts
            WHERE detection_latency_ms IS NOT NULL
              AND detection_latency_ms > 0
              AND detection_latency_ms < 600000
        """)
        rows = cur.fetchall()
    conn.close()

    if not rows:
        print("  ERROR: No latency data found in anomaly_alerts.")
        return {"error": "no data"}

    latencies = np.array([r[0] for r in rows], dtype=float)
    n = len(latencies)
    median_ms = float(np.median(latencies))
    mean_ms   = float(np.mean(latencies))

    print(f"  Samples          : {n:,}")
    print(f"  Batch baseline   : {batch_baseline_ms:,} ms  ({batch_baseline_ms/1000:.1f} s)")
    print(f"  Streaming mean   : {mean_ms:,.0f} ms  ({mean_ms/1000:.1f} s)")
    print(f"  Streaming median : {median_ms:,.0f} ms  ({median_ms/1000:.1f} s)")

    # One-sample Wilcoxon signed-rank test: H0: median = batch_baseline_ms
    # Shift data to test against the constant
    diffs = latencies - batch_baseline_ms
    stat, p_value = stats.wilcoxon(diffs, alternative="less")

    # Also run t-test for comparison
    t_stat, t_p = stats.ttest_1samp(latencies, batch_baseline_ms, alternative="less")

    significant = bool(p_value < 0.05)
    print(f"\n  Wilcoxon statistic : {stat:.2f}")
    print(f"  p-value            : {p_value:.4e}")
    print(f"  t-test p-value     : {t_p:.4e}")
    print(f"  Significant (α=0.05): {'YES — streaming is significantly faster' if significant else 'NO'}")

    # Effect size: rank-biserial correlation
    r_effect = 1 - (2 * stat) / (n * (n + 1))

    return {
        "n_samples":           n,
        "batch_baseline_ms":   round(batch_baseline_ms, 1),
        "batch_detail":        batch_detail,
        "streaming_mean_ms":   round(mean_ms, 1),
        "streaming_median_ms": round(median_ms, 1),
        "speedup_factor":      round(batch_baseline_ms / median_ms, 2) if median_ms > 0 else None,
        "wilcoxon_statistic":  round(float(stat), 2),
        "p_value":             float(p_value),
        "t_statistic":         round(float(t_stat), 4),
        "t_p_value":           float(t_p),
        "effect_size_r":       round(float(r_effect), 4),
        "significant":         significant,
        "alpha":               0.05,
        "alternative":         "streaming < batch",
    }


# ── RQ2: Per-item MAPE computation ────────────────────────────────────────────

def compute_per_item_mape(store_id: str, test_days: int) -> tuple[list, list, object]:
    """
    Returns (batch_mapes, streaming_mapes, meta) — one entry per item
    with valid test data in both methods.
    """
    print(f"\n[RQ2] Computing per-item MAPE for store={store_id}, horizon={test_days} days ...")

    # Load M5 CSV
    print("  Loading M5 CSV...")
    sales    = pd.read_csv(DATA_DIR / "sales_train_evaluation.csv")
    calendar = pd.read_csv(DATA_DIR / "calendar.csv", usecols=["date", "d"])
    calendar["date"] = pd.to_datetime(calendar["date"])

    store_sales = sales[sales["store_id"] == store_id].copy()
    day_cols    = [c for c in store_sales.columns if c.startswith("d_")]

    records = []
    for _, row in store_sales.iterrows():
        for dcol in day_cols:
            records.append({
                "item_id":   row["item_id"],
                "day":       dcol,
                "sales_qty": row[dcol],
            })
    m5_df = pd.DataFrame(records)
    m5_df = m5_df.merge(calendar.rename(columns={"d": "day"}), on="day")
    m5_df = m5_df.sort_values(["item_id", "date"]).reset_index(drop=True)

    # Load streaming features from DB
    print("  Loading streaming features from DB...")
    conn = connect()
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT item_id, time::date AS date, rolling_avg_7d
            FROM sales_features
            WHERE store_id = %s AND rolling_avg_7d IS NOT NULL
            ORDER BY item_id, time ASC
        """, (store_id,))
        rows = cur.fetchall()
    conn.close()
    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        raise RuntimeError("sales_features is empty for this store.")
    feat_df["date"] = pd.to_datetime(feat_df["date"])

    last_date       = feat_df["date"].max()
    test_start_date = last_date - pd.Timedelta(days=test_days)
    test_end_date   = last_date

    train_m5   = m5_df[m5_df["date"] < test_start_date]
    test_m5    = m5_df[(m5_df["date"] >= test_start_date) & (m5_df["date"] <= test_end_date)]
    train_feat = feat_df[feat_df["date"] < test_start_date]

    items = sorted(set(train_m5["item_id"].unique()) & set(train_feat["item_id"].unique()))
    print(f"  Items: {len(items):,}  |  test period: {test_start_date.date()} → {test_end_date.date()}")

    batch_mapes    = []
    streaming_mapes = []
    n_skipped = 0

    for i, item_id in enumerate(items, 1):
        if i % 500 == 0:
            print(f"  {i}/{len(items)} items ...")

        # Batch: raw M5 daily sales as training series
        batch_train = train_m5[train_m5["item_id"] == item_id]["sales_qty"].values
        test_actual = test_m5[test_m5["item_id"] == item_id]["sales_qty"].values[:test_days]

        if len(test_actual) < test_days:
            n_skipped += 1
            continue

        # Streaming: rolling_avg_7d feature series as proxy training series
        feat_train = train_feat[train_feat["item_id"] == item_id]["rolling_avg_7d"].values
        if len(feat_train) < 3:
            n_skipped += 1
            continue

        batch_preds    = esm_forecast(batch_train.astype(float), test_days)
        streaming_preds = esm_forecast(feat_train.astype(float), test_days)

        bm = item_mape(test_actual.astype(float), batch_preds)
        sm = item_mape(test_actual.astype(float), streaming_preds)

        if bm is not None and sm is not None:
            batch_mapes.append(bm)
            streaming_mapes.append(sm)

    meta = {
        "store": store_id,
        "test_days": test_days,
        "test_start": str(test_start_date.date()),
        "test_end": str(test_end_date.date()),
        "n_items": len(batch_mapes),
        "n_skipped": n_skipped,
    }
    print(f"  Valid pairs: {len(batch_mapes):,}  skipped: {n_skipped}")
    return batch_mapes, streaming_mapes, meta


# ── RQ2: Wilcoxon signed-rank on paired MAPE differences ─────────────────────

def test_rq2_wilcoxon(batch_mapes: list, streaming_mapes: list) -> dict:
    print("\n=== RQ2: Forecast Accuracy (Wilcoxon signed-rank, paired) ===")
    print("  H0: no difference in per-item MAPE (batch vs streaming)")
    print("  H1: MAPE distributions differ\n")

    b = np.array(batch_mapes)
    s = np.array(streaming_mapes)
    d = b - s   # positive = batch is worse

    print(f"  n items          : {len(b):,}")
    print(f"  Batch avg MAPE   : {np.mean(b):.2f}%  (median {np.median(b):.2f}%)")
    print(f"  Streaming avg    : {np.mean(s):.2f}%  (median {np.median(s):.2f}%)")
    print(f"  Diff (b-s) median: {np.median(d):+.4f}%")

    stat, p_value = stats.wilcoxon(d, alternative="two-sided")
    significant   = bool(p_value < 0.05)

    print(f"\n  Wilcoxon statistic : {stat:.2f}")
    print(f"  p-value (two-sided): {p_value:.4e}")
    print(f"  Significant (α=0.05): {'YES' if significant else 'NO — batch ≈ streaming (no penalty)'}")

    # Effect size: rank-biserial correlation
    n = len(d)
    r_effect = 1 - (2 * stat) / (n * (n + 1))

    return {
        "n_items":             n,
        "batch_avg_mape":      round(float(np.mean(b)), 4),
        "batch_median_mape":   round(float(np.median(b)), 4),
        "streaming_avg_mape":  round(float(np.mean(s)), 4),
        "streaming_median_mape": round(float(np.median(s)), 4),
        "diff_median":         round(float(np.median(d)), 4),
        "diff_mean":           round(float(np.mean(d)), 4),
        "wilcoxon_statistic":  round(float(stat), 2),
        "p_value":             float(p_value),
        "effect_size_r":       round(float(r_effect), 4),
        "significant":         significant,
        "alpha":               0.05,
        "alternative":         "two-sided",
        "interpretation":      (
            "Significant difference in forecast accuracy."
            if significant else
            "No significant MAPE difference — streaming features do not degrade accuracy."
        ),
    }


# ── RQ2: Diebold-Mariano test ─────────────────────────────────────────────────

def test_rq2_diebold_mariano(batch_mapes: list, streaming_mapes: list) -> dict:
    """
    Simplified DM test treating per-item MAPE as the loss series.
    d_i = L(batch_i) - L(streaming_i)   where L = absolute MAPE
    DM statistic: mean(d) / sqrt(Var(d)/n)  ~ N(0,1) for large n
    H0: E[d] = 0  (methods equally accurate)
    """
    print("\n=== RQ2: Forecast Accuracy (Diebold-Mariano test) ===")

    b = np.array(batch_mapes)
    s = np.array(streaming_mapes)
    d = b - s   # loss differential (positive = batch error is larger)
    n = len(d)

    mean_d = np.mean(d)
    # Long-run variance estimate (Newey-West with h=1, i.e., no autocorr for cross-section)
    var_d  = np.var(d, ddof=1)
    se_d   = np.sqrt(var_d / n)

    if se_d == 0:
        return {"error": "zero variance in loss differential"}

    dm_stat = mean_d / se_d
    # p-value: two-sided N(0,1)
    p_value = float(2 * stats.norm.sf(abs(dm_stat)))
    significant = bool(p_value < 0.05)

    print(f"  mean(d)    = {mean_d:+.4f}%  (positive = batch is worse)")
    print(f"  DM stat    = {dm_stat:.4f}")
    print(f"  p-value    = {p_value:.4e}")
    print(f"  Significant (α=0.05): {'YES' if significant else 'NO'}")

    return {
        "n_items":          n,
        "mean_loss_diff":   round(float(mean_d), 4),
        "std_loss_diff":    round(float(np.sqrt(var_d)), 4),
        "dm_statistic":     round(float(dm_stat), 4),
        "p_value":          p_value,
        "significant":      significant,
        "alpha":            0.05,
        "alternative":      "two-sided",
        "interpretation":   (
            "Significant accuracy difference detected."
            if significant else
            "No significant accuracy difference (DM confirms streaming ≈ batch)."
        ),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests — RQ1 + RQ2")
    parser.add_argument("--store",     default="CA_1")
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--db-host",   default="localhost")
    args = parser.parse_args()

    global DB_PARAMS
    DB_PARAMS["host"] = args.db_host

    print("=" * 60)
    print("Statistical Significance Tests")
    print("=" * 60)

    # ── RQ1 ──
    rq1_result = test_rq1_wilcoxon()

    # ── RQ2 per-item MAPE ──
    batch_mapes, streaming_mapes, meta = compute_per_item_mape(args.store, args.test_days)

    # ── RQ2 tests ──
    rq2_wilcoxon = test_rq2_wilcoxon(batch_mapes, streaming_mapes)
    rq2_dm       = test_rq2_diebold_mariano(batch_mapes, streaming_mapes)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nRQ1 Wilcoxon  p={rq1_result.get('p_value', 'N/A'):.2e}  "
          f"→ {'SIGNIFICANT (streaming faster)' if rq1_result.get('significant') else 'not significant'}")
    print(f"RQ2 Wilcoxon  p={rq2_wilcoxon.get('p_value', 'N/A'):.2e}  "
          f"→ {'SIGNIFICANT' if rq2_wilcoxon.get('significant') else 'not significant (streaming ≈ batch)'}")
    print(f"RQ2 DM test   p={rq2_dm.get('p_value', 'N/A'):.2e}  "
          f"→ {'SIGNIFICANT' if rq2_dm.get('significant') else 'not significant (confirms Wilcoxon)'}")

    # ── Save ──
    results = {
        "rq1_wilcoxon":       rq1_result,
        "rq2_meta":           meta,
        "rq2_wilcoxon":       rq2_wilcoxon,
        "rq2_diebold_mariano": rq2_dm,
    }
    out_path = OUT_DIR / "statistical_tests_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
