"""
#2 Evaluation: Anomaly injection with synthetic ground truth labels.

Loads sales_features from TimescaleDB, splits into clean train / test sets,
injects three types of synthetic anomalies into the test set, then evaluates
Isolation Forest Precision / Recall / F1.

Anomaly types injected into test set:
  stockout : rolling_avg_7d = 0, rolling_sum_7d = 0, event_count_7d = 0
  spike    : rolling_avg_7d × 5, rolling_sum_7d × 5, max_qty_7d × 5
  drift    : rolling_avg_7d × 2.5, rolling_sum_7d × 2.5  (sustained demand shift)

Design:
  - Model trained on CLEAN train split (no injected anomalies) — same as production
  - Anomalies injected ONLY into test split → fair evaluation with known ground truth
  - contamination parameter matches injection_rate so IF knows expected anomaly fraction

Usage:
  python evaluation/anomaly_injection.py
  python evaluation/anomaly_injection.py --store CA_1 --injection-rate 0.05 --seed 42
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score, classification_report,
    confusion_matrix,
)

FEATURE_COLS = [
    "rolling_avg_7d", "rolling_sum_7d", "event_count_7d",
    "max_qty_7d", "min_qty_7d",
]
OUT_DIR = Path(__file__).parent / "experiments"

DB_PARAMS = dict(host="localhost", port=5432, dbname="retail",
                 user="retail", password="retail")


# ── Data loading ───────────────────────────────────────────────────────────────

def load_features(store_id: str | None, limit: int) -> np.ndarray:
    conn = psycopg2.connect(**DB_PARAMS)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        if store_id:
            cur.execute(f"""
                SELECT {', '.join(FEATURE_COLS)}
                FROM sales_features
                WHERE store_id = %s
                  AND rolling_avg_7d IS NOT NULL
                ORDER BY inserted_at ASC NULLS LAST, time ASC
                LIMIT %s
            """, (store_id, limit))
        else:
            cur.execute(f"""
                SELECT {', '.join(FEATURE_COLS)}
                FROM sales_features
                WHERE rolling_avg_7d IS NOT NULL
                ORDER BY inserted_at ASC NULLS LAST, time ASC
                LIMIT %s
            """, (limit,))
        rows = cur.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError(
            "sales_features is empty. Run feature_pipeline.py first."
        )

    X = np.array(
        [[float(r[c] or 0) for c in FEATURE_COLS] for r in rows],
        dtype=np.float64,
    )
    print(f"  Loaded {len(X):,} feature rows  (cols: {FEATURE_COLS})")
    return X


# ── Anomaly injection ──────────────────────────────────────────────────────────

def inject_anomalies(
    X_test: np.ndarray,
    injection_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inject synthetic anomalies into X_test rows.
    Returns:
      X_mixed     — test array with injected anomalies replacing selected rows
      y_true      — 1 = anomaly, 0 = normal (ground truth)
      anomaly_type — integer type label (0=stockout, 1=spike, 2=drift, -1=normal)
    """
    n = len(X_test)
    n_inject = max(1, int(n * injection_rate))
    n_per_type = n_inject // 3
    remainder  = n_inject - n_per_type * 3  # distribute remainder into stockout

    # Sample injection indices without replacement
    idx_all   = rng.choice(n, size=n_inject, replace=False)
    idx_stockout = idx_all[:n_per_type + remainder]
    idx_spike    = idx_all[n_per_type + remainder : 2*n_per_type + remainder]
    idx_drift    = idx_all[2*n_per_type + remainder:]

    X_mixed       = X_test.copy()
    y_true        = np.zeros(n, dtype=int)
    anomaly_type  = np.full(n, -1, dtype=int)   # -1 = normal

    # rolling_avg_7d=0, rolling_sum_7d=1, event_count_7d=2, max_qty_7d=3, min_qty_7d=4

    # Stockout: zero sales across all features
    X_mixed[idx_stockout, 0] = 0.0  # rolling_avg_7d
    X_mixed[idx_stockout, 1] = 0.0  # rolling_sum_7d
    X_mixed[idx_stockout, 2] = 0.0  # event_count_7d
    X_mixed[idx_stockout, 3] = 0.0  # max_qty_7d
    X_mixed[idx_stockout, 4] = 0.0  # min_qty_7d
    y_true[idx_stockout]        = 1
    anomaly_type[idx_stockout]  = 0

    # Demand spike: multiply average & sum by 5
    X_mixed[idx_spike, 0] *= 5.0   # rolling_avg_7d
    X_mixed[idx_spike, 1] *= 5.0   # rolling_sum_7d
    X_mixed[idx_spike, 3] *= 5.0   # max_qty_7d
    y_true[idx_spike]        = 1
    anomaly_type[idx_spike]  = 1

    # Drift: sustained 2.5× shift in rolling average
    X_mixed[idx_drift, 0] *= 2.5   # rolling_avg_7d
    X_mixed[idx_drift, 1] *= 2.5   # rolling_sum_7d
    y_true[idx_drift]        = 1
    anomaly_type[idx_drift]  = 2

    return X_mixed, y_true, anomaly_type


# ── Evaluation ─────────────────────────────────────────────────────────────────

def metrics_for_type(y_true: np.ndarray, y_pred: np.ndarray,
                     anomaly_type: np.ndarray, type_id: int) -> dict:
    """Compute P/R/F1 for a single anomaly type vs all normal rows."""
    mask = (anomaly_type == type_id) | (anomaly_type == -1)
    yt = y_true[mask]
    yp = y_pred[mask]
    if yt.sum() == 0:
        return {"precision": None, "recall": None, "f1": None, "n_injected": 0}
    return {
        "precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
        "recall":    round(float(recall_score(yt, yp, zero_division=0)), 4),
        "f1":        round(float(f1_score(yt, yp, zero_division=0)), 4),
        "n_injected": int(yt.sum()),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Anomaly injection evaluation — F1 / Precision / Recall"
    )
    parser.add_argument("--store",          default="CA_1",
                        help="Store to evaluate (default: CA_1)")
    parser.add_argument("--injection-rate", type=float, default=0.05,
                        help="Fraction of test rows to replace with synthetic anomalies (default: 0.05)")
    parser.add_argument("--train-split",    type=float, default=0.80,
                        help="Fraction of data used for clean training (default: 0.80)")
    parser.add_argument("--limit",          type=int,   default=500_000,
                        help="Max feature rows to load (default: 500,000)")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--db-host",        default="localhost")
    args = parser.parse_args()

    global DB_PARAMS
    DB_PARAMS["host"] = args.db_host

    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Anomaly Injection Evaluation ===")
    print(f"  Store          : {args.store}")
    print(f"  Injection rate : {args.injection_rate:.0%}")
    print(f"  Train split    : {args.train_split:.0%}")
    print(f"  Random seed    : {args.seed}\n")

    # ── Load ──
    print("[1] Loading features from TimescaleDB...")
    t0 = time.perf_counter()
    X = load_features(args.store, args.limit)
    print(f"  Done ({time.perf_counter()-t0:.1f}s)\n")

    # ── Split ──
    n_train = int(len(X) * args.train_split)
    X_train = X[:n_train]
    X_test  = X[n_train:]
    print(f"[2] Split: train={len(X_train):,}  test={len(X_test):,}\n")

    # ── Inject anomalies into test set ──
    print("[3] Injecting synthetic anomalies into test set...")
    X_mixed, y_true, anomaly_type = inject_anomalies(X_test, args.injection_rate, rng)
    n_injected = y_true.sum()
    n_normal   = (y_true == 0).sum()
    print(f"  Normal rows   : {n_normal:,}")
    print(f"  Stockout      : {(anomaly_type==0).sum():,}")
    print(f"  Spike         : {(anomaly_type==1).sum():,}")
    print(f"  Drift         : {(anomaly_type==2).sum():,}")
    print(f"  Total injected: {n_injected:,}  ({n_injected/len(y_true):.2%})\n")

    # ── Train IF on CLEAN training set ──
    print(f"[4] Training Isolation Forest on {len(X_train):,} clean samples ...")
    t1 = time.perf_counter()
    model = IsolationForest(
        contamination=args.injection_rate,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_train)
    print(f"  Training done ({time.perf_counter()-t1:.1f}s)\n")

    # ── Predict on test set with injected anomalies ──
    print("[5] Predicting on test set...")
    t2 = time.perf_counter()
    raw_preds = model.predict(X_mixed)        # -1 = anomaly, 1 = normal
    y_pred    = (raw_preds == -1).astype(int)  # 1 = anomaly
    print(f"  Prediction done ({time.perf_counter()-t2:.1f}s)")
    print(f"  Predicted anomalies: {y_pred.sum():,}  "
          f"({y_pred.sum()/len(y_pred):.2%})\n")

    # ── Overall metrics ──
    p   = precision_score(y_true, y_pred, zero_division=0)
    r   = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("── Overall Metrics ───────────────────────────────────────────")
    print(f"  Precision : {p:.4f}  ({p*100:.1f}%)")
    print(f"  Recall    : {r:.4f}  ({r*100:.1f}%)")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # ── Per-type metrics ──
    type_names = {0: "stockout", 1: "spike", 2: "drift"}
    by_type = {}
    print("\n── Per-Type Metrics ──────────────────────────────────────────")
    for tid, tname in type_names.items():
        m = metrics_for_type(y_true, y_pred, anomaly_type, tid)
        by_type[tname] = m
        print(f"  {tname:<10} P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  F1={m['f1']:.4f}  "
              f"(n={m['n_injected']:,})")

    # ── Save results ──
    results = {
        "store":          args.store,
        "n_train":        int(len(X_train)),
        "n_test":         int(len(X_test)),
        "injection_rate": args.injection_rate,
        "n_injected":     int(n_injected),
        "n_normal_test":  int(n_normal),
        "contamination":  args.injection_rate,
        "seed":           args.seed,
        "overall": {
            "precision": round(float(p),  4),
            "recall":    round(float(r),  4),
            "f1":        round(float(f1), 4),
            "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn),
        },
        "by_type": by_type,
    }

    out_path = OUT_DIR / "anomaly_injection_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
