"""
Phase 6 — Streamlit dashboard.

Real-time overview of the retail inventory streaming platform.
Shows system status, RQ1 anomaly detection metrics, RQ2 forecast accuracy,
and RQ3 architecture throughput.

Usage:
  streamlit run dashboard/app.py
"""

import psycopg2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from psycopg2.extras import RealDictCursor

# ── Config ────────────────────────────────────────────────────────────────────

DB_PARAMS = dict(host="localhost", port=5432, dbname="retail",
                 user="retail", password="retail")

st.set_page_config(
    page_title="Retail Inventory Platform",
    page_icon="📦",
    layout="wide",
)

# ── DB helpers ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def query(sql: str, params=None) -> pd.DataFrame:
    conn = psycopg2.connect(**DB_PARAMS)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.close()
    return pd.DataFrame(rows)


def scalar(sql: str, params=None):
    conn = psycopg2.connect(**DB_PARAMS)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        result = cur.fetchone()
    conn.close()
    return result[0] if result else None


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("📦 Retail Inventory Platform")
st.sidebar.markdown("Real-Time Data Engineering — DSC01")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "RQ1 — Anomaly Detection", "RQ2 — Demand Forecasting", "RQ3 — Architecture"],
)

st.sidebar.divider()
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

# ── Page: Overview ────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("System Overview")
    st.caption("Row counts update every 10 seconds.")

    col1, col2, col3, col4 = st.columns(4)

    sales_events  = scalar("SELECT COUNT(*) FROM sales_events")  or 0
    sales_features = scalar("SELECT COUNT(*) FROM sales_features") or 0
    anomaly_alerts = scalar("SELECT COUNT(*) FROM anomaly_alerts") or 0
    forecast_results = scalar("SELECT COUNT(*) FROM forecast_results") or 0

    col1.metric("Sales Events", f"{sales_events:,}")
    col2.metric("Feature Rows", f"{sales_features:,}")
    col3.metric("Anomaly Alerts", f"{anomaly_alerts:,}")
    col4.metric("Forecast Rows", f"{forecast_results:,}")

    st.divider()

    # Feature rows over time
    st.subheader("Feature Pipeline Activity")
    feat_time = query("""
        SELECT DATE_TRUNC('hour', inserted_at) AS hour,
               COUNT(*) AS rows_inserted
        FROM sales_features
        WHERE inserted_at IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)
    if not feat_time.empty:
        fig = px.bar(feat_time, x="hour", y="rows_inserted",
                     title="Feature rows inserted per hour",
                     labels={"hour": "Time", "rows_inserted": "Rows"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feature data with timestamps yet. Run the producer and feature pipeline.")

    # Recent anomalies
    st.subheader("Recent Anomaly Alerts")
    recent = query("""
        SELECT detected_at, store_id, item_id,
               ROUND(anomaly_score::numeric, 4) AS score,
               detection_latency_ms
        FROM anomaly_alerts
        ORDER BY detected_at DESC
        LIMIT 20
    """)
    if not recent.empty:
        st.dataframe(recent, use_container_width=True)
    else:
        st.info("No anomaly alerts yet.")

# ── Page: RQ1 ────────────────────────────────────────────────────────────────

elif page == "RQ1 — Anomaly Detection":
    st.title("RQ1 — Anomaly Detection Latency")
    st.markdown(
        "**Research Question:** Does stream processing offer lower anomaly detection "
        "latency than batch processing for retail inventory data?"
    )

    # Key metrics
    latency = query("""
        SELECT
            COUNT(*) AS total_alerts,
            ROUND(AVG(detection_latency_ms)::numeric)  AS avg_ms,
            ROUND(MIN(detection_latency_ms)::numeric)  AS min_ms,
            ROUND(MAX(detection_latency_ms)::numeric)  AS max_ms,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY detection_latency_ms)::numeric) AS p50_ms,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY detection_latency_ms)::numeric) AS p95_ms
        FROM anomaly_alerts
    """)

    if not latency.empty and latency["total_alerts"].iloc[0]:
        row = latency.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Alerts", f"{int(row['total_alerts']):,}")
        c2.metric("Avg Latency", f"{int(row['avg_ms']):,} ms")
        c3.metric("P50 Latency", f"{int(row['p50_ms']):,} ms")
        c4.metric("P95 Latency", f"{int(row['p95_ms']):,} ms")
        c5.metric("Batch Baseline", "87,400 ms", delta=f"{int(row['avg_ms'])-87400:,} ms")
    else:
        st.info("No anomaly alert data yet.")

    st.divider()

    # Batch vs streaming comparison bar chart
    st.subheader("Batch vs Streaming — Detection Latency")
    avg_ms = scalar("SELECT ROUND(AVG(detection_latency_ms)) FROM anomaly_alerts") or 0
    fig = go.Figure(go.Bar(
        x=["Batch baseline", "Streaming (this platform)"],
        y=[87400, int(avg_ms)],
        marker_color=["#EF553B", "#00CC96"],
        text=[f"{87400:,} ms", f"{int(avg_ms):,} ms"],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis_title="Latency (ms)",
        title="Anomaly detection latency: batch vs streaming",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Latency over time
    st.subheader("Detection Latency Over Time")
    lat_time = query("""
        SELECT detected_at, detection_latency_ms, store_id
        FROM anomaly_alerts
        ORDER BY detected_at DESC
        LIMIT 500
    """)
    if not lat_time.empty:
        fig2 = px.scatter(lat_time, x="detected_at", y="detection_latency_ms",
                          color="store_id", opacity=0.6,
                          labels={"detected_at": "Time", "detection_latency_ms": "Latency (ms)"},
                          title="Detection latency per alert")
        st.plotly_chart(fig2, use_container_width=True)

    # Top anomalous items
    st.subheader("Most Flagged Items")
    top_items = query("""
        SELECT store_id, item_id, COUNT(*) AS alert_count,
               ROUND(AVG(anomaly_score)::numeric, 4) AS avg_score
        FROM anomaly_alerts
        GROUP BY store_id, item_id
        ORDER BY alert_count DESC
        LIMIT 15
    """)
    if not top_items.empty:
        fig3 = px.bar(top_items, x="item_id", y="alert_count", color="store_id",
                      title="Items with most anomaly alerts")
        st.plotly_chart(fig3, use_container_width=True)

# ── Page: RQ2 ────────────────────────────────────────────────────────────────

elif page == "RQ2 — Demand Forecasting":
    st.title("RQ2 — Demand Forecasting Accuracy")
    st.markdown(
        "**Research Question:** Do streaming features improve demand forecast accuracy "
        "compared to static batch features?"
    )

    # Summary results (hardcoded from evaluation run)
    st.subheader("Evaluation Results — CA_1, 7-day horizon")
    col1, col2, col3 = st.columns(3)
    col1.metric("Batch Avg MAPE", "77.54%")
    col2.metric("Streaming Avg MAPE", "78.51%", delta="+0.97%", delta_color="off")
    col3.metric("Difference", "<1%", help="Negligible difference — streaming matches batch accuracy")

    st.divider()

    # MAPE comparison chart
    fig = go.Figure(go.Bar(
        x=["Batch (raw daily series)", "Streaming (rolling_avg_7d)"],
        y=[77.54, 78.51],
        marker_color=["#EF553B", "#00CC96"],
        text=["77.54%", "78.51%"],
        textposition="outside",
    ))
    fig.update_layout(
        yaxis_title="MAPE (%)",
        title="Forecast accuracy: batch vs streaming features (lower is better)",
        yaxis_range=[0, 100],
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Recent forecasts from DB
    st.subheader("Recent Forecasts")
    store_filter = st.selectbox("Store", ["CA_1", "CA_2", "CA_3", "CA_4",
                                           "TX_1", "TX_2", "TX_3",
                                           "WI_1", "WI_2", "WI_3"])
    forecasts = query("""
        SELECT created_at, item_id, horizon_day, forecast_date,
               ROUND(predicted_qty::numeric, 2) AS predicted_qty
        FROM forecast_results
        WHERE store_id = %s
        ORDER BY created_at DESC, item_id, horizon_day
        LIMIT 200
    """, (store_filter,))

    if not forecasts.empty:
        items = forecasts["item_id"].unique()[:10]
        fig2 = px.line(
            forecasts[forecasts["item_id"].isin(items)],
            x="forecast_date", y="predicted_qty",
            color="item_id", line_dash="item_id",
            title=f"7-day forecasts — {store_filter} (sample items)",
            labels={"forecast_date": "Forecast Date", "predicted_qty": "Predicted Qty"},
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(forecasts.head(50), use_container_width=True)
    else:
        st.info(f"No forecast data for {store_filter}. Run demand_forecaster.py first.")

# ── Page: RQ3 ────────────────────────────────────────────────────────────────

elif page == "RQ3 — Architecture":
    st.title("RQ3 — Architecture Reliability & Throughput")
    st.markdown(
        "**Research Question:** What reliability and throughput characteristics does "
        "the event-driven architecture provide for retail supply chain visibility?"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Peak Throughput", "~13,700 ev/s", help="Measured during producer replay")
    col2.metric("Steady Throughput", "~6,000 ev/s", help="Sustained during multi-day replay")
    col3.metric("Kafka Partitions", "10", help="1 per store, enables parallel consumption")

    st.divider()

    # End-to-end latency distribution
    st.subheader("End-to-End Detection Latency Distribution")
    lat_dist = query("""
        SELECT detection_latency_ms
        FROM anomaly_alerts
        WHERE detection_latency_ms < 200000
        LIMIT 5000
    """)
    if not lat_dist.empty:
        fig = px.histogram(
            lat_dist, x="detection_latency_ms", nbins=40,
            title="Detection latency distribution (ms)",
            labels={"detection_latency_ms": "Latency (ms)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Latency percentiles
    st.subheader("Latency Percentiles")
    pct = query("""
        SELECT
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY detection_latency_ms)::numeric) AS p50,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY detection_latency_ms)::numeric) AS p75,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY detection_latency_ms)::numeric) AS p95,
            ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY detection_latency_ms)::numeric) AS p99
        FROM anomaly_alerts
    """)
    if not pct.empty and pct["p50"].iloc[0]:
        row = pct.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("P50", f"{int(row['p50']):,} ms")
        c2.metric("P75", f"{int(row['p75']):,} ms")
        c3.metric("P95", f"{int(row['p95']):,} ms")
        c4.metric("P99", f"{int(row['p99']):,} ms")

    # Feature pipeline throughput
    st.divider()
    st.subheader("Feature Pipeline — Rows Written Over Time")
    feat = query("""
        SELECT DATE_TRUNC('minute', inserted_at) AS minute,
               COUNT(*) AS rows
        FROM sales_features
        WHERE inserted_at IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """)
    if not feat.empty:
        fig2 = px.area(feat, x="minute", y="rows",
                       title="Feature rows written per minute",
                       labels={"minute": "Time", "rows": "Rows"})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No timestamped feature data yet.")
