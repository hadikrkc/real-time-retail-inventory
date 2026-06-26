# Evaluation Methodology

Covers the three research questions and how each is measured.
Implementation lives in `metrics/`, `experiments/`, and `anomaly_injection.py`.

---

## RQ1 — Stream vs Batch Anomaly Detection

**Question:** What latency–accuracy trade-offs does stream processing offer over batch processing for anomaly detection on retail inventory data?

| Metric | Unit | Notes |
|---|---|---|
| Detection latency | ms → s | Event emit to alert written in TimescaleDB |
| Precision / Recall / F1 | % | Against synthetic anomaly ground truth |
| Throughput | events/s | At saturation load |
| Resource cost | CPU % · RAM MB | Optional: cloud cost projection |

**Control group:** same models run as daily cron jobs on static data (`rq1_stream_vs_batch.py`).

---

## RQ2 — Streaming Features vs Static Features

**Question:** How much better do demand forecasting models perform when fed by a streaming feature pipeline?

| Metric | Horizon | Notes |
|---|---|---|
| MASE | short (1–7 d) · mid (28 d) | Scale-independent, preferred for M5 |
| WAPE | short · mid | Weighted absolute % error |
| sMAPE | short · mid | Symmetric — avoids asymmetry bias |
| Stockout / overstock quality | — | Fill-rate, holding cost proxy |

**Control group:** same models with static (batch) features (`rq2_features_comparison.py`).

---

## RQ3 — Architecture Reliability & Throughput

**Question:** What reliability and throughput characteristics can an event-driven reference architecture deliver for end-to-end retail supply chain visibility?

### Latency measurement

**Measurement window:** `t₀` (event emitted by Replay Producer) → `t₁` (result written to TimescaleDB).

```
Replay Producer
      │  t₀ — timestamp embedded in event payload
      ▼
Kafka broker            +1–5 ms   (network + serialization)
      ▼
Spark micro-batch       +5 s      (trigger interval, configurable)
      ▼
Feature Engineering     +10–50 ms (window computation)
      ▼
Model inference         +5–20 ms  (Isolation Forest / forecast)
      ▼
TimescaleDB write       +1–5 ms
      │  t₁ — persisted
      ▼
  latency = t₁ − t₀
```

### Target thresholds (Spark micro-batch @ 5 s trigger)

| Percentile | Target | What it represents |
|---|---|---|
| **P50** | < 6 s | Typical event — one micro-batch cycle end-to-end |
| **P95** | < 12 s | Slight backpressure or larger window computation |
| **P99** | < 30 s | Tail latency — GC pause, Kafka rebalance, checkpoint write |

> Batch jobs run daily, so even P99 = 30 s is a ~3000× improvement. For retail inventory (not HFT), sub-minute detection is the target.

### Reliability metrics

| Metric | Target | How measured |
|---|---|---|
| Event loss % | < 0.01 % | Kafka offset reconciliation |
| Exactly-once delivery | verified | Spark checkpointing + idempotent sink |
| Throughput at saturation | ≥ 10 k events/s | Replay Producer ramp test |
| Checkpoint recovery time | < 60 s | Kill + restart test |

### Implementation

```
evaluation/
├── metrics/
│   ├── streaming_metrics.py   ← P50/P95/P99, throughput, event loss
│   ├── anomaly_metrics.py     ← Precision/Recall/F1, detection latency
│   └── forecasting_metrics.py ← MASE, WAPE, sMAPE, Pinball
├── statistical_tests.py       ← Diebold-Mariano, Wilcoxon, Friedman
├── anomaly_injection.py       ← synthetic stockout · spike · drift
└── experiments/
    ├── rq1_stream_vs_batch.py
    ├── rq2_features_comparison.py
    └── rq3_architecture_bench.py
```
