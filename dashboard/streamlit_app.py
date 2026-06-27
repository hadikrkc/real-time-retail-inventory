import glob
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sqlalchemy import create_engine, text

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Inventory Platform",
    page_icon="🛒",
    layout="wide",
)

DSN          = os.environ.get("POSTGRES_DSN", "postgresql://retail:retail@localhost:5432/retail")
PRODUCER_API = os.environ.get("PRODUCER_API_URL", "http://localhost:8000")
EVAL_DIR     = Path("/evaluation/experiments")   # mounted from ../evaluation/experiments

_GREEN_BTN_CSS = """
<style>
[data-testid="baseButton-primary"] {
    background-color: #28a745 !important;
    border-color:     #28a745 !important;
    color:            white   !important;
}
</style>
"""

# ── DB helpers ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    return create_engine(DSN, pool_pre_ping=True)


def db_ok() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def query(sql: str) -> pd.DataFrame:
    try:
        return pd.read_sql(sql, get_engine())
    except Exception:
        return pd.DataFrame()


# ── RQ2 analysis (runs inside Streamlit, DB-only, no CSV needed) ──────────────

def compute_streaming_mape(store_id: str, horizon: int = 7) -> dict | None:
    """Join forecast_results with sales_events to compute streaming forecast MAPE."""
    df = query(f"""
        SELECT fr.item_id,
               fr.horizon_day,
               fr.predicted_qty,
               COALESCE(agg.actual_qty, 0) AS actual_qty
          FROM (
              SELECT item_id, horizon_day,
                     AVG(predicted_qty)                 AS predicted_qty,
                     DATE_TRUNC('day', forecast_date)   AS forecast_day
                FROM forecast_results
               WHERE store_id = '{store_id}'
                 AND feature_source = 'streaming'
                 AND horizon_day <= {horizon}
               GROUP BY item_id, horizon_day, forecast_day
          ) fr
          LEFT JOIN (
              SELECT item_id,
                     DATE_TRUNC('day', time)  AS day,
                     SUM(sales_qty)           AS actual_qty
                FROM sales_events
               WHERE store_id = '{store_id}'
               GROUP BY item_id, day
          ) agg ON fr.item_id = agg.item_id
               AND fr.forecast_day = agg.day
         WHERE agg.actual_qty > 0
         LIMIT 200000
    """)
    if df.empty or len(df) < 10:
        return None
    df["ape"] = (df["actual_qty"] - df["predicted_qty"]).abs() / df["actual_qty"]
    df = df[df["ape"].notna() & (df["ape"] < 10)]   # remove extreme outliers
    return {
        "method":        "streaming",
        "avg_mape":      round(float(df["ape"].mean()   * 100), 2),
        "median_mape":   round(float(df["ape"].median() * 100), 2),
        "avg_smape":     round(float(
            (2 * (df["actual_qty"] - df["predicted_qty"]).abs()
             / (df["actual_qty"].abs() + df["predicted_qty"].abs())).mean() * 100
        ), 2),
        "n_forecasts":   int(len(df)),
        "n_items":       int(df["item_id"].nunique()),
    }


def compute_batch_mape(store_id: str, horizon: int = 7, max_items: int = 200) -> dict | None:
    """
    Batch baseline for RQ2 — computed entirely from DB, no CSV needed.
    Trains ExponentialSmoothing on each item's full historical daily series
    and forecasts the last `horizon` days.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    df = query(f"""
        SELECT item_id,
               DATE_TRUNC('day', time)::date AS day,
               SUM(sales_qty)               AS sales_qty
          FROM sales_events
         WHERE store_id = '{store_id}'
         GROUP BY item_id, day
         ORDER BY item_id, day
    """)
    if df.empty:
        return None

    df["day"]       = pd.to_datetime(df["day"])
    df["sales_qty"] = df["sales_qty"].astype(float)

    items = df["item_id"].unique()[:max_items]
    all_mape  = []
    all_smape = []

    for item_id in items:
        series = (
            df[df["item_id"] == item_id]
            .set_index("day")["sales_qty"]
            .asfreq("D")
            .fillna(0)
        )
        if len(series) < horizon + 3:
            continue
        train = series.iloc[:-horizon]
        test  = series.iloc[-horizon:].values

        if train.sum() == 0:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit   = ExponentialSmoothing(train, trend="add", seasonal=None).fit(optimized=True)
                preds = np.clip(fit.forecast(horizon), 0, None)
        except Exception:
            preds = np.full(horizon, float(train.iloc[-1]))

        mask = test > 0
        if mask.sum() == 0:
            continue
        ape   = np.abs(test[mask] - preds[mask]) / test[mask]
        denom = (np.abs(test[mask]) + np.abs(preds[mask])) / 2
        sape  = np.where(denom > 0, np.abs(test[mask] - preds[mask]) / denom, np.nan)

        all_mape.append(float(np.nanmean(ape)))
        all_smape.append(float(np.nanmean(sape)))

    if not all_mape:
        return None
    return {
        "method":      "batch",
        "avg_mape":    round(float(np.mean(all_mape))   * 100, 2),
        "median_mape": round(float(np.median(all_mape)) * 100, 2),
        "avg_smape":   round(float(np.mean(all_smape))  * 100, 2),
        "n_items":     len(all_mape),
        "items_sampled": int(len(items)),
    }


def compute_batch_anomaly_latency(sample_size: int = 100_000) -> dict | None:
    """
    RQ1 batch baseline — computed from DB, no CSV needed.
    Loads feature vectors from sales_features, trains IsolationForest,
    predicts on all rows, and records wall-clock time for each phase.
    """
    from sklearn.ensemble import IsolationForest

    # Phase 1: load
    t0 = time.perf_counter()
    df = query("""
        SELECT rolling_avg_7d, rolling_sum_7d, event_count_7d,
               max_qty_7d, min_qty_7d
          FROM sales_features
         WHERE rolling_avg_7d IS NOT NULL
    """)
    load_sec = time.perf_counter() - t0

    if df.empty:
        return None

    X = df.fillna(0).values.astype(np.float32)

    # Phase 2: train on sample
    t1 = time.perf_counter()
    idx = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
    model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X[idx])
    train_sec = time.perf_counter() - t1

    # Phase 3: predict on all rows
    t2 = time.perf_counter()
    preds = model.predict(X)
    predict_sec = time.perf_counter() - t2

    total_sec = load_sec + train_sec + predict_sec
    n_anomalies = int((preds == -1).sum())

    return {
        "load_sec":           round(load_sec, 2),
        "train_sec":          round(train_sec, 2),
        "predict_sec":        round(predict_sec, 2),
        "total_sec":          round(total_sec, 2),
        "total_ms":           round(total_sec * 1000, 0),
        "n_rows":             len(X),
        "sample_size":        len(idx),
        "n_anomalies":        n_anomalies,
        "anomaly_rate_pct":   round(n_anomalies / len(X) * 100, 2),
    }


def load_batch_baseline() -> dict | None:
    """Return the most recent batch_baseline_*.json result, or None."""
    if not EVAL_DIR.exists():
        return None
    files = sorted(EVAL_DIR.glob("batch_baseline_*.json"), reverse=True)
    for f in files:
        try:
            return json.loads(f.read_text())
        except Exception:
            continue
    return None


def _file_rq2_for_store(store_id: str) -> dict | None:
    """Look for a saved rq2_<store>_*.json from a previous CLI run."""
    if not EVAL_DIR.exists():
        return None
    files = sorted(EVAL_DIR.glob(f"rq2_{store_id}_*.json"), reverse=True)
    if not files:
        files = sorted(EVAL_DIR.glob("rq2_*.json"), reverse=True)
    for f in files:
        try:
            return json.loads(f.read_text())
        except Exception:
            continue
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
_sidebar_placeholder = st.sidebar.empty()
with _sidebar_placeholder.container():
    st.title("🛒 Retail Platform")
    st.caption("Anomaly Detection · Demand Forecasting")
    st.divider()

    # Fetch producer status first — drives button appearance
    try:
        ps       = requests.get(f"{PRODUCER_API}/status", timeout=2).json()
        running  = ps.get("running", False)
        total_ev = ps.get("total_events", 0)
        ev_sec   = ps.get("events_per_sec", 0.0)
        cur_date = ps.get("current_date", "—")
        api_ok   = True
    except Exception:
        running = False; api_ok = False
        total_ev = ev_sec = 0; cur_date = "—"

    st.subheader("Pipeline Control")
    speed = st.slider("Replay speed (days/min)", 1, 60, 10, key="speed")

    if running:
        st.markdown(_GREEN_BTN_CSS, unsafe_allow_html=True)
        if st.button("⏸ Pause", type="primary", use_container_width=True):
            try:
                requests.post(f"{PRODUCER_API}/stop", timeout=5)
                st.toast("Pipeline stopped.", icon="⏹")
                st.rerun()
            except Exception as e:
                st.toast(f"Connection failed: {e}", icon="🔴")
    else:
        if st.button("▶ Start", type="primary", use_container_width=True):
            try:
                requests.post(
                    f"{PRODUCER_API}/start",
                    params={"speed_days_per_min": speed},
                    timeout=5,
                )
                st.toast(f"Started — {speed} days/min", icon="▶")
                st.rerun()
            except Exception as e:
                st.toast(f"Connection failed: {e}", icon="🔴")

    if api_ok:
        st.caption(f"Pipeline: {'🟢 Running' if running else '⚪ Idle'}")
        if running:
            st.caption(f"{cur_date} · {total_ev:,} events · {ev_sec:,.0f} ev/s")
    else:
        st.caption("Pipeline: 🔴 API unreachable")

    st.divider()

    # ── Reset controls ────────────────────────────────────────────────────────
    st.subheader("Reset")
    reset_level = st.radio(
        "Reset level",
        options=["soft", "hard"],
        captions=["Analytics only (anomalies, features)", "Full reset (+sales_events, Kafka topics)"],
        horizontal=False,
    )
    if "reset_confirm" not in st.session_state:
        st.session_state["reset_confirm"] = False

    # Poll /reset-status on every page refresh to track background reset progress
    _reset_status = None
    try:
        _r = requests.get(f"{PRODUCER_API}/reset-status", timeout=2)
        if _r.ok:
            _reset_status = _r.json()
    except Exception:
        pass

    _rs_running = False
    if _reset_status:
        _rs_running = _reset_status.get("running", False)
        _rs_done    = _reset_status.get("done",    False)
        _rs_error   = _reset_status.get("error")
        _rs_level   = _reset_status.get("level",   "")
        if _rs_running:
            st.info(f"⏳ Reset in progress ({_rs_level})…")
            st.session_state.pop("_reset_done_at", None)  # clear timer for next completion
        elif _rs_done and not _rs_error:
            # Show success message for 10 seconds, then let it disappear
            if "_reset_done_at" not in st.session_state:
                st.session_state["_reset_done_at"] = time.time()
            if time.time() - st.session_state["_reset_done_at"] < 10:
                st.success(f"✅ Reset complete ({_rs_level})")
        elif _rs_error:
            st.error(f"❌ Reset failed: {_rs_error}")

    # Disable the reset button while a reset is already running
    if not st.session_state["reset_confirm"]:
        if st.button("🗑 Reset", use_container_width=True, disabled=_rs_running):
            st.session_state["reset_confirm"] = True
            st.rerun()
    else:
        st.warning(f"**{reset_level.upper()} reset** — irreversible! Confirm?")
        col_y, col_n = st.columns(2)
        if col_y.button("✅ Yes", use_container_width=True):
            st.session_state["reset_confirm"] = False
            try:
                requests.post(
                    f"{PRODUCER_API}/reset",
                    params={"level": reset_level},
                    timeout=5,
                )
                st.toast(f"Reset started ({reset_level}) — running in background…", icon="🗑")
                if reset_level == "hard":
                    st.info(
                        "After hard reset completes, restart Spark:\n"
                        "```\ndocker compose restart spark-streaming anomaly-detector demand-forecaster\n```"
                    )
            except Exception as e:
                st.toast(f"Reset error: {e}", icon="🔴")
            st.rerun()
        if col_n.button("❌ Cancel", use_container_width=True):
            st.session_state["reset_confirm"] = False
            st.rerun()

    st.divider()
    auto_refresh = st.toggle("Auto-refresh (5 s)", value=True, key="auto_refresh")

    st.divider()
    st.subheader("Connections")
    st.write(f"TimescaleDB: {'🟢 OK' if db_ok() else '🔴 Offline'}")
    st.write(f"Kafka:       {'🟢 OK' if api_ok else '🔴 —'}")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Real-Time Retail Inventory Platform")
st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# ── KPIs ──────────────────────────────────────────────────────────────────────
ev = query(
    "SELECT COUNT(*) AS n FROM sales_events "
    "WHERE ingested_at > NOW() - INTERVAL '1 minute'"
)
al = query(
    "SELECT COUNT(*) AS n FROM anomaly_alerts "
    "WHERE detected_at > NOW() - INTERVAL '1 minute'"
)
mx = query(
    "SELECT events_per_sec, kafka_lag FROM pipeline_metrics "
    "ORDER BY time DESC LIMIT 1"
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Events (last 1 min)",    int(ev["n"].iloc[0])                  if not ev.empty else "—")
k2.metric("Anomalies (last 1 min)", int(al["n"].iloc[0])                  if not al.empty else "—")
k3.metric("Events / sec",           f"{mx['events_per_sec'].iloc[0]:.0f}" if not mx.empty else "—")
k4.metric("Kafka Lag",              int(mx["kafka_lag"].iloc[0])           if not mx.empty else "—")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_flow, tab_alerts, tab_forecast, tab_rq3, tab_paper = st.tabs(
    ["📈 Inventory Flow", "🚨 Anomalies · RQ1", "🔮 Forecasts · RQ2", "📊 RQ3 Metrics", "📋 Paper Results"]
)

# ── Tab 1 — Inventory Flow ────────────────────────────────────────────────────
with tab_flow:
    df = query("""
        SELECT time_bucket('30 seconds', ingested_at) AS bucket,
               store_id,
               SUM(sales_qty) AS total_sales
          FROM sales_events
         WHERE ingested_at > NOW() - INTERVAL '10 minutes'
         GROUP BY bucket, store_id
         ORDER BY bucket
    """)
    if df.empty:
        st.info("⏳ Waiting for replay producer — press **▶ Start** in the sidebar.")
    else:
        st.plotly_chart(
            px.line(df, x="bucket", y="total_sales", color="store_id",
                    title="Sales Events by Store (30 s buckets, last 10 min)"),
            use_container_width=True,
        )

    with st.expander("Raw events (last 50)"):
        st.dataframe(
            query(
                "SELECT time, store_id, item_id, sales_qty, day, ingested_at "
                "FROM sales_events ORDER BY ingested_at DESC NULLS LAST LIMIT 50"
            ),
            use_container_width=True,
        )


# ── Tab 2 — Anomaly Alerts + RQ1 ─────────────────────────────────────────────
with tab_alerts:
    df = query("""
        SELECT detected_at, store_id, item_id,
               anomaly_score, detection_latency_ms
          FROM anomaly_alerts
         ORDER BY detected_at DESC
         LIMIT 200
    """)
    if df.empty:
        st.info("⏳ Anomalies will appear here once the Spark feature pipeline starts.")
    else:
        st.dataframe(df, use_container_width=True)
        df["score_abs"] = df["anomaly_score"].abs()
        st.plotly_chart(
            px.scatter(
                df, x="detected_at", y="anomaly_score",
                size="score_abs", color="store_id",
                hover_data=["item_id", "detection_latency_ms"],
                title="Anomaly Score Timeline",
            ),
            use_container_width=True,
        )

    # ── RQ1: Streaming vs Batch latency comparison ────────────────────────────
    st.divider()
    st.subheader("RQ1 — Anomaly Detection Latency: Streaming vs Batch")
    st.caption(
        "**Streaming**: `detection_latency_ms` = time from Spark feature write to anomaly detection.  \n"
        "**Batch**: total wall-clock time to retrain IsolationForest and predict on all rows in `sales_features`."
    )

    lat = query("""
        SELECT
            ROUND(AVG(detection_latency_ms)::numeric)                                          AS avg_ms,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY detection_latency_ms)::numeric) AS p50_ms,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY detection_latency_ms)::numeric) AS p95_ms,
            ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY detection_latency_ms)::numeric) AS p99_ms,
            COUNT(*) AS total_alerts
        FROM anomaly_alerts
        WHERE detection_latency_ms > 0 AND detection_latency_ms < 600000
    """)

    # Batch baseline: session_state (button) or file (docker compose run)
    file_baseline = load_batch_baseline()
    rq1_baseline  = st.session_state.get("rq1_batch_result") or file_baseline

    # Controls
    col_run1, col_clear1 = st.columns([2, 1])
    run_rq1   = col_run1.button("📊 Run Batch Baseline", use_container_width=True, key="rq1_run")
    clear_rq1 = col_clear1.button("🗑 Clear", use_container_width=True, key="rq1_clear")

    if clear_rq1:
        st.session_state.pop("rq1_batch_result", None)
        st.rerun()

    if run_rq1:
        with st.spinner("Running batch baseline… (loading features + training IsolationForest)"):
            result = compute_batch_anomaly_latency()
            if result:
                st.session_state["rq1_batch_result"] = result
                rq1_baseline = result
                # Persist result so future sessions can load it without recomputing
                EVAL_DIR.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                (EVAL_DIR / f"batch_baseline_{ts}.json").write_text(
                    json.dumps({"total_batch_latency_ms": result["total_ms"],
                                "total_rows": result["n_rows"],
                                **result}, indent=2)
                )
            else:
                st.warning("sales_features is empty — start the producer and wait for Spark to generate features.")
        st.rerun()

    # ── Display results ───────────────────────────────────────────────────────
    if not lat.empty and lat["total_alerts"].iloc[0]:
        s_avg = int(lat["avg_ms"].iloc[0])
        s_p50 = int(lat["p50_ms"].iloc[0])
        s_p95 = int(lat["p95_ms"].iloc[0])
        s_p99 = int(lat["p99_ms"].iloc[0])

        col_s, col_b = st.columns(2)
        with col_s:
            st.markdown("**Streaming (incremental, per-feature)**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg",  f"{s_avg:,} ms")
            m2.metric("P50",  f"{s_p50:,} ms")
            m3.metric("P95",  f"{s_p95:,} ms")
            m4.metric("P99",  f"{s_p99:,} ms")
            st.caption(f"from {int(lat['total_alerts'].iloc[0]):,} alerts")

        with col_b:
            st.markdown("**Batch (full re-run on all data)**")
            if rq1_baseline:
                b_ms   = int(rq1_baseline.get("total_ms") or rq1_baseline.get("total_batch_latency_ms", 0))
                n_rows = int(rq1_baseline.get("n_rows") or rq1_baseline.get("total_rows", 0))
                speedup = round(b_ms / s_avg, 1) if s_avg > 0 else "—"
                b1, b2, b3 = st.columns(3)
                b1.metric("Total time",  f"{b_ms:,} ms")
                b2.metric("Row count",   f"{n_rows:,}")
                b3.metric("Speedup",     f"{speedup}×")
                if rq1_baseline.get("load_sec"):
                    # Support both key naming conventions:
                    # dashboard button  → train_sec / predict_sec
                    # run_batch_baseline.py CLI → feature_sec / detect_sec
                    _train   = rq1_baseline.get("train_sec")   or rq1_baseline.get("feature_sec")
                    _predict = rq1_baseline.get("predict_sec") or rq1_baseline.get("detect_sec")
                    parts = [f"Load: {rq1_baseline['load_sec']}s"]
                    if _train   is not None: parts.append(f"Train: {_train}s")
                    if _predict is not None: parts.append(f"Predict: {_predict}s")
                    st.caption(" · ".join(parts))
            else:
                st.info("Press **📊 Run Batch Baseline** above.")

        if rq1_baseline:
            b_ms    = int(rq1_baseline.get("total_ms") or rq1_baseline.get("total_batch_latency_ms", 0))
            speedup = round(b_ms / s_avg, 1) if s_avg > 0 else 0

            fig = go.Figure(data=[
                go.Bar(name="Streaming (avg per feature)",
                       x=["Anomaly Detection Latency"], y=[s_avg],
                       marker_color="#1f77b4",
                       text=[f"{s_avg:,} ms"], textposition="outside"),
                go.Bar(name="Batch (full dataset re-run)",
                       x=["Anomaly Detection Latency"], y=[b_ms],
                       marker_color="#ff7f0e",
                       text=[f"{b_ms:,} ms"], textposition="outside"),
            ])
            fig.update_layout(
                barmode="group",
                title="RQ1: Streaming vs Batch Anomaly Detection Latency",
                yaxis_title="Milliseconds (lower is better)",
                yaxis_type="log",
            )
            st.plotly_chart(fig, use_container_width=True)

            reduction = round((b_ms - s_avg) / b_ms * 100, 1) if b_ms > 0 else 0
            if speedup > 1:
                st.success(
                    f"Streaming is **{speedup}× faster** than batch "
                    f"(latency reduction: **{reduction}%**)."
                )
    else:
        st.info("⏳ Waiting for anomaly alerts — start the producer, let Spark generate features, and ensure anomaly-detector is running.")


# ── Tab 3 — Forecasts + RQ2 ──────────────────────────────────────────────────
with tab_forecast:
    df = query("""
        SELECT created_at, store_id, item_id,
               horizon_day, forecast_date, predicted_qty, feature_source
          FROM forecast_results
         ORDER BY created_at DESC
         LIMIT 500
    """)
    if df.empty:
        st.info("⏳ Forecasts will appear here once the Spark feature pipeline starts.")
    else:
        stores = df["store_id"].unique().tolist()
        col_s, col_m = st.columns([1, 3])
        if len(stores) > 1:
            store = col_s.selectbox("Store", stores)
        else:
            store = stores[0]
            col_s.caption(f"Store: **{store}**")
        mode  = col_m.radio("Feature mode", ["streaming", "batch", "both"], horizontal=True)

        d = df[df["store_id"] == store]
        if mode != "both":
            d = d[d["feature_source"] == mode]

        st.plotly_chart(
            px.line(d, x="forecast_date", y="predicted_qty",
                    color="feature_source", line_dash="horizon_day",
                    title=f"Demand Forecast — Store {store}"),
            use_container_width=True,
        )

    # ── RQ2: Streaming vs Batch forecast accuracy ─────────────────────────────
    st.divider()
    st.subheader("RQ2 — Forecast Accuracy: Streaming vs Batch")
    st.caption(
        "**Streaming**: `forecast_results` predictions vs actual `sales_events`.  \n"
        "**Batch**: same DB historical series → ExponentialSmoothing → last 7 days as test set."
    )

    # Store selector and compute button
    avail_stores = query("SELECT DISTINCT store_id FROM forecast_results ORDER BY store_id")
    store_list   = avail_stores["store_id"].tolist() if not avail_stores.empty else ["CA_1"]
    col_sel, col_btn, col_clear = st.columns([2, 1, 1])
    if len(store_list) > 1:
        rq2_store = col_sel.selectbox("Store", store_list, key="rq2_store")
    else:
        rq2_store = store_list[0]
        col_sel.caption(f"Store: **{rq2_store}**")
    max_items   = col_sel.slider("Items sampled for batch", 50, 500, 150, step=50,
                                  key="rq2_max_items")

    run_clicked   = col_btn.button("📊 Compute", use_container_width=True, key="rq2_run")
    clear_clicked = col_clear.button("🗑 Clear", use_container_width=True, key="rq2_clear")

    if clear_clicked:
        st.session_state.pop("rq2_result", None)
        st.rerun()

    if run_clicked:
        st.session_state["_rq2_computing"] = True
        with st.spinner(f"Computing… (store={rq2_store}, batch items={max_items})"):
            s_res = compute_streaming_mape(rq2_store)
            b_res = compute_batch_mape(rq2_store, max_items=max_items)
            st.session_state["rq2_result"] = {
                "store":     rq2_store,
                "streaming": s_res,
                "batch":     b_res,
            }
        st.session_state.pop("_rq2_computing", None)
        st.rerun()

    # If nothing computed yet, try loading from file (CLI run)
    rq2_res = st.session_state.get("rq2_result") or _file_rq2_for_store(rq2_store)

    if rq2_res:
        s = rq2_res.get("streaming") or {}
        b = rq2_res.get("batch")     or {}

        col_s2, col_b2 = st.columns(2)
        with col_s2:
            st.markdown("**Streaming (rolling_avg_7d → ESM)**")
            st.metric("Avg MAPE",    f"{s.get('avg_mape',  '—')}" + ("%" if s.get("avg_mape")  else ""))
            st.metric("Median MAPE", f"{s.get('median_mape','—')}" + ("%" if s.get("median_mape") else ""))
            st.metric("Avg sMAPE",   f"{s.get('avg_smape', '—')}" + ("%" if s.get("avg_smape")  else ""))
            if s.get("n_forecasts"):
                st.caption(f"{s['n_forecasts']:,} forecast points · {s.get('n_items','')} items")
        with col_b2:
            st.markdown("**Batch (full historical series → ESM)**")
            st.metric("Avg MAPE",    f"{b.get('avg_mape',  '—')}" + ("%" if b.get("avg_mape")  else ""))
            st.metric("Median MAPE", f"{b.get('median_mape','—')}" + ("%" if b.get("median_mape") else ""))
            st.metric("Avg sMAPE",   f"{b.get('avg_smape', '—')}" + ("%" if b.get("avg_smape")  else ""))
            if b.get("n_items"):
                st.caption(f"{b['n_items']} items · {b.get('items_sampled','')} sampled")

        if s.get("avg_mape") and b.get("avg_mape"):
            diff = round(s["avg_mape"] - b["avg_mape"], 2)
            if abs(diff) < 3.0:
                st.success(
                    f"Streaming MAPE is close to batch (**{diff:+.2f}pp**) — "
                    f"streaming features are competitive with full historical series."
                )
            elif diff > 0:
                st.warning(
                    f"Streaming MAPE is **{diff:.2f}pp** higher than batch — "
                    f"streaming trades some accuracy for real-time availability."
                )
            else:
                st.success(
                    f"Streaming MAPE is **{abs(diff):.2f}pp** lower than batch — "
                    f"rolling features capture recent trends better."
                )

            fig = go.Figure(data=[
                go.Bar(name="Streaming", x=["Avg MAPE", "Median MAPE", "Avg sMAPE"],
                       y=[s.get("avg_mape"), s.get("median_mape"), s.get("avg_smape")],
                       marker_color="#1f77b4"),
                go.Bar(name="Batch",     x=["Avg MAPE", "Median MAPE", "Avg sMAPE"],
                       y=[b.get("avg_mape"), b.get("median_mape"), b.get("avg_smape")],
                       marker_color="#ff7f0e"),
            ])
            fig.update_layout(
                barmode="group",
                title=f"RQ2: Streaming vs Batch Forecast Error — {rq2_res.get('store', rq2_store)}",
                yaxis_title="Error % (lower is better)",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        if avail_stores.empty:
            st.info("⏳ `forecast_results` is empty — start the producer first.")
        else:
            st.info("Press **📊 Compute** above to run the comparison.")


# ── Tab 4 — RQ3 Architecture Metrics ─────────────────────────────────────────
with tab_rq3:
    df = query("""
        SELECT time, events_per_sec, processing_lag_ms, kafka_lag
          FROM pipeline_metrics
         ORDER BY time DESC
         LIMIT 300
    """)
    if df.empty:
        st.info("⏳ Metrics will accumulate here once the stream consumer starts.")
    else:
        c_a, c_b = st.columns(2)
        with c_a:
            st.plotly_chart(
                px.line(df, x="time", y="events_per_sec",
                        title="Throughput (events / sec)"),
                use_container_width=True,
            )
        with c_b:
            st.plotly_chart(
                px.line(df, x="time", y="processing_lag_ms",
                        title="Processing Lag (ms)  ← RQ3"),
                use_container_width=True,
            )

        p = query("""
            SELECT
                percentile_cont(0.50) WITHIN GROUP (ORDER BY processing_lag_ms) AS p50,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY processing_lag_ms) AS p95,
                percentile_cont(0.99) WITHIN GROUP (ORDER BY processing_lag_ms) AS p99
              FROM pipeline_metrics
        """)
        if not p.empty:
            st.subheader("Latency Percentiles (target: P50<6s · P95<12s · P99<30s)")
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("P50", f"{p['p50'].iloc[0]:.0f} ms")
            pc2.metric("P95", f"{p['p95'].iloc[0]:.0f} ms")
            pc3.metric("P99", f"{p['p99'].iloc[0]:.0f} ms")

        st.plotly_chart(
            px.line(df, x="time", y="kafka_lag", title="Kafka Consumer Lag"),
            use_container_width=True,
        )

        # Data volume summary
        st.subheader("Data Volume")
        vol = query("""
            SELECT
                (SELECT COUNT(*) FROM sales_events)    AS sales_events,
                (SELECT COUNT(*) FROM sales_features)  AS sales_features,
                (SELECT COUNT(*) FROM anomaly_alerts)  AS anomaly_alerts,
                (SELECT COUNT(*) FROM forecast_results) AS forecast_results
        """)
        if not vol.empty:
            v = vol.iloc[0]
            vc1, vc2, vc3, vc4 = st.columns(4)
            vc1.metric("Sales Events",    f"{int(v['sales_events']):,}")
            vc2.metric("Features",        f"{int(v['sales_features']):,}")
            vc3.metric("Anomaly Alerts",  f"{int(v['anomaly_alerts']):,}")
            vc4.metric("Forecast Results",f"{int(v['forecast_results']):,}")


# ── Tab 5 — Paper Results ─────────────────────────────────────────────────────
with tab_paper:
    st.subheader("Research Results Summary")
    st.caption(
        "All RQ metrics in one place. "
        "Running the same pipeline on any machine produces the same structure. "
        "Use **Export** to download JSON for the paper."
    )

    # ── Collect all metrics ────────────────────────────────────────────────────
    # RQ1 — detection latency
    rq1_stream = query("""
        SELECT
            COUNT(*)                                                              AS n_alerts,
            ROUND(AVG(detection_latency_ms)::numeric)                            AS avg_ms,
            ROUND(MIN(detection_latency_ms)::numeric)                            AS min_ms,
            ROUND(MAX(detection_latency_ms)::numeric)                            AS max_ms,
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY detection_latency_ms)::numeric) AS p50_ms,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY detection_latency_ms)::numeric) AS p95_ms,
            ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY detection_latency_ms)::numeric) AS p99_ms
        FROM anomaly_alerts
        WHERE detection_latency_ms > 0 AND detection_latency_ms < 600000
    """)
    rq1_batch = (
        st.session_state.get("rq1_batch_result")
        or load_batch_baseline()
    )

    # RQ2 — forecast accuracy
    rq2_cached = st.session_state.get("rq2_result")

    # RQ3 — pipeline throughput & lag
    rq3_lag = query("""
        SELECT
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY processing_lag_ms)::numeric) AS p50,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY processing_lag_ms)::numeric) AS p95,
            ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY processing_lag_ms)::numeric) AS p99,
            ROUND(AVG(processing_lag_ms)::numeric)                                           AS avg_ms,
            ROUND(MAX(events_per_sec)::numeric)                                              AS peak_eps,
            ROUND(AVG(events_per_sec)::numeric)                                              AS avg_eps,
            COUNT(*)                                                                         AS n_samples
        FROM pipeline_metrics
        WHERE processing_lag_ms IS NOT NULL AND processing_lag_ms > 0
    """)
    rq3_vol = query("""
        SELECT
            (SELECT COUNT(*) FROM sales_events)     AS sales_events,
            (SELECT COUNT(*) FROM sales_features)   AS sales_features,
            (SELECT COUNT(*) FROM anomaly_alerts)   AS anomaly_alerts,
            (SELECT COUNT(*) FROM forecast_results) AS forecast_results,
            (SELECT COUNT(*) FROM pipeline_metrics) AS pipeline_metrics
    """)

    # ── RQ1 block ─────────────────────────────────────────────────────────────
    st.markdown("### RQ1 — Anomaly Detection Latency")
    st.caption("Streaming: Spark feature write → detection | Batch: full re-run on entire dataset")

    if not rq1_stream.empty and rq1_stream["n_alerts"].iloc[0]:
        r1s = rq1_stream.iloc[0]
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Streaming**")
            t = pd.DataFrame({
                "Metric": ["Avg", "Min", "Max", "P50", "P95", "P99", "Alert count"],
                "Value":  [
                    f"{int(r1s['avg_ms']):,} ms",
                    f"{int(r1s['min_ms']):,} ms",
                    f"{int(r1s['max_ms']):,} ms",
                    f"{int(r1s['p50_ms']):,} ms",
                    f"{int(r1s['p95_ms']):,} ms",
                    f"{int(r1s['p99_ms']):,} ms",
                    f"{int(r1s['n_alerts']):,}",
                ],
            })
            st.table(t.set_index("Metric"))

        with col_b:
            st.markdown("**Batch**")
            if rq1_batch:
                b_ms   = int(rq1_batch.get("total_ms") or rq1_batch.get("total_batch_latency_ms", 0))
                n_rows = int(rq1_batch.get("n_rows")   or rq1_batch.get("total_rows", 0))
                speedup = round(b_ms / int(r1s["avg_ms"]), 1) if int(r1s["avg_ms"]) > 0 else "—"
                t2 = pd.DataFrame({
                    "Metric": ["Total time", "  Load", "  Train", "  Predict",
                               "Row count", "Speedup vs streaming"],
                    "Value": [
                        f"{b_ms:,} ms",
                        f"{rq1_batch.get('load_sec', '—')} s",
                        f"{rq1_batch.get('train_sec') or rq1_batch.get('feature_sec', '—')} s",
                        f"{rq1_batch.get('predict_sec') or rq1_batch.get('detect_sec', '—')} s",
                        f"{n_rows:,}",
                        f"{speedup}×",
                    ],
                })
                st.table(t2.set_index("Metric"))
            else:
                st.info("Run **📊 Run Batch Baseline** from the Anomalies · RQ1 tab.")
    else:
        st.info("⏳ `anomaly_alerts` is empty.")

    st.divider()

    # ── RQ2 block ─────────────────────────────────────────────────────────────
    st.markdown("### RQ2 — Demand Forecast Accuracy")
    st.caption("Streaming: rolling_avg_7d → ESM | Batch: full historical series → ESM")

    if rq2_cached and rq2_cached.get("streaming") and rq2_cached.get("batch"):
        s2 = rq2_cached["streaming"]
        b2 = rq2_cached["batch"]
        store_label = rq2_cached.get("store", "—")
        diff_mape   = round((s2.get("avg_mape", 0) or 0) - (b2.get("avg_mape", 0) or 0), 2)

        t3 = pd.DataFrame({
            "Metric":      ["Avg MAPE (%)", "Median MAPE (%)", "Avg sMAPE (%)", "Item count"],
            "Streaming":   [s2.get("avg_mape"), s2.get("median_mape"), s2.get("avg_smape"), s2.get("n_items")],
            "Batch":       [b2.get("avg_mape"), b2.get("median_mape"), b2.get("avg_smape"), b2.get("n_items")],
            "Diff (S−B)":  [diff_mape, None, None, None],
        })
        st.caption(f"Store: **{store_label}**")
        st.table(t3.set_index("Metric"))

        if abs(diff_mape) < 3.0:
            st.success(f"Streaming MAPE difference **{diff_mape:+.2f}pp** — competitive accuracy.")
        elif diff_mape > 0:
            st.warning(f"Streaming is **{diff_mape:.2f}pp** higher error — acceptable trade-off for real-time availability.")
        else:
            st.success(f"Streaming is **{abs(diff_mape):.2f}pp** lower error.")
    else:
        st.info("Run **📊 Compute** from the Forecasts · RQ2 tab.")

    st.divider()

    # ── RQ3 block ─────────────────────────────────────────────────────────────
    st.markdown("### RQ3 — Architecture Performance")
    st.caption("Kafka → TimescaleDB write latency and throughput")

    if not rq3_lag.empty and rq3_lag["n_samples"].iloc[0]:
        r3 = rq3_lag.iloc[0]
        t4 = pd.DataFrame({
            "Metric": ["Processing Lag P50", "Processing Lag P95", "Processing Lag P99",
                       "Processing Lag Avg", "Peak Events/sec", "Avg Events/sec", "Sample count"],
            "Value": [
                f"{int(r3['p50']):,} ms",
                f"{int(r3['p95']):,} ms",
                f"{int(r3['p99']):,} ms",
                f"{int(r3['avg_ms']):,} ms",
                f"{int(r3['peak_eps']):,}",
                f"{int(r3['avg_eps']):,}",
                f"{int(r3['n_samples']):,}",
            ],
            "Target": ["< 6,000 ms", "< 12,000 ms", "< 30,000 ms",
                       "—", "—", "—", "—"],
            "Status": [
                "✅" if int(r3["p50"]) < 6000  else "❌",
                "✅" if int(r3["p95"]) < 12000 else "❌",
                "✅" if int(r3["p99"]) < 30000 else "❌",
                "—", "—", "—", "—",
            ],
        })
        st.table(t4.set_index("Metric"))
    else:
        st.info("⏳ `pipeline_metrics` is empty.")

    # Data volume
    st.markdown("#### Data Volume")
    if not rq3_vol.empty:
        v = rq3_vol.iloc[0]
        tv = pd.DataFrame({
            "Table": ["sales_events", "sales_features", "anomaly_alerts",
                      "forecast_results", "pipeline_metrics"],
            "Rows":  [f"{int(v['sales_events']):,}",   f"{int(v['sales_features']):,}",
                      f"{int(v['anomaly_alerts']):,}",  f"{int(v['forecast_results']):,}",
                      f"{int(v['pipeline_metrics']):,}"],
        })
        st.table(tv.set_index("Table"))

    st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("### Export")

    def _safe_int(val):
        try:
            return int(val) if val is not None else None
        except Exception:
            return None

    snapshot = {
        "exported_at": datetime.now().isoformat(),
        "rq1": {
            "streaming": {
                k: _safe_int(rq1_stream.iloc[0][k]) if not rq1_stream.empty else None
                for k in ["n_alerts", "avg_ms", "min_ms", "max_ms", "p50_ms", "p95_ms", "p99_ms"]
            } if not rq1_stream.empty else {},
            "batch": rq1_batch or {},
        },
        "rq2": rq2_cached or {},
        "rq3": {
            "processing_lag": {
                k: _safe_int(rq3_lag.iloc[0][k]) if not rq3_lag.empty else None
                for k in ["p50", "p95", "p99", "avg_ms", "peak_eps", "avg_eps", "n_samples"]
            } if not rq3_lag.empty else {},
            "data_volume": {
                k: _safe_int(rq3_vol.iloc[0][k]) if not rq3_vol.empty else None
                for k in ["sales_events", "sales_features", "anomaly_alerts",
                          "forecast_results", "pipeline_metrics"]
            } if not rq3_vol.empty else {},
        },
    }

    col_dl, col_save = st.columns(2)

    col_dl.download_button(
        label="⬇ Download JSON",
        data=json.dumps(snapshot, indent=2, default=str),
        file_name=f"paper_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )

    if col_save.button("💾 Save to Server", use_container_width=True):
        try:
            EVAL_DIR.mkdir(parents=True, exist_ok=True)
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = EVAL_DIR / f"paper_results_{ts}.json"
            path.write_text(json.dumps(snapshot, indent=2, default=str))
            st.success(f"Saved: `evaluation/experiments/paper_results_{ts}.json`")
        except Exception as e:
            st.error(f"Save failed: {e}")

    st.markdown("---")
    st.markdown(
        "**Reproducibility steps:**\n"
        "1. `docker compose up -d` — start all services\n"
        "2. Dashboard → **▶ Start** — begin M5 replay, wait for full data ingestion\n"
        "3. **Anomalies · RQ1** → 📊 Run Batch Baseline\n"
        "4. **Forecasts · RQ2** → select store → 📊 Compute\n"
        "5. Return to this tab → **⬇ Download JSON**"
    )


# ── Auto-refresh ──────────────────────────────────────────────────────────────
if st.session_state.get("auto_refresh", True):
    time.sleep(5)
    st.rerun()
