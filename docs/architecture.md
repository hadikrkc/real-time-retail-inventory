# System Architecture & Data Flow

Last updated: 2026-06-28

```mermaid
flowchart TD
    subgraph SOURCES["📦 Data Source"]
        M5["M5 Forecasting Dataset\nWalmart · 30,490 series · 5 yr\ndata/m5/"]
    end

    subgraph INGESTION["⚙️ Ingestion — pipeline/replay_producer/"]
        RP["Replay Producer\napi.py + producer.py\nFastAPI :8000\n20 days/min default"]
    end

    subgraph KAFKA["📨 Message Bus — infra/kafka/"]
        K1[/"retail.sales.events\n10 partitions"/]
    end

    subgraph STREAM["🔄 Stream Processing — pipeline/stream_processor/"]
        SC["Stream Consumer\nsink.py\nKafka → sales_events\n+ pipeline_metrics"]
        SP["Spark Feature Pipeline\nfeature_pipeline.py\n7-day sliding window\nwatermark 1 day · trigger 10 s"]
    end

    subgraph STORAGE["🗄️ TimescaleDB — infra/timescale/"]
        SE[("sales_events\nhypertable")]
        SF[("sales_features\nhypertable")]
        AA[("anomaly_alerts")]
        FR[("forecast_results")]
        PM[("pipeline_metrics\nthroughput + lag")]
    end

    subgraph MODELS["🤖 Model Services — pipeline/stream_processor/"]
        AD["Anomaly Detector\nanomaly_detector.py\nIsolation Forest\npoll every 5 s"]
        DF["Demand Forecaster\ndemand_forecaster.py\nExponentialSmoothing\npoll every 60 s"]
    end

    subgraph EVAL["📊 Evaluation — evaluation/"]
        RQ1["RQ1\nstatistical_tests.py\nbatch baseline timing\nWilcoxon · DM test"]
        RQ2["RQ2\nstatistical_tests.py\nMAPE: batch vs streaming\nWilcoxon · DM test"]
        RQ3["RQ3\nmetrics/streaming_metrics.py\nP50 · P95 · P99\nthroughput"]
        ANO["Anomaly F1\nanomaly_injection.py\nstockout · spike · drift\nPrecision · Recall · F1"]
    end

    subgraph OUT["🖥️ Output"]
        DASH["Streamlit Dashboard\ndashboard/streamlit_app.py\n:8501"]
        JSON["evaluation/experiments/\n*_60day.json\n*_365day.json"]
    end

    M5 --> RP
    RP -->|"JSON events"| K1
    K1 --> SC
    K1 --> SP
    SC --> SE
    SC --> PM
    SP -->|"upsert"| SF
    SF --> AD
    SF --> DF
    AD --> AA
    DF --> FR
    SE --> RQ1
    SF --> RQ1
    SF --> RQ2
    PM --> RQ3
    SF --> ANO
    AA --> ANO
    RQ1 --> JSON
    RQ2 --> JSON
    RQ3 --> JSON
    ANO --> JSON
    SE --> DASH
    SF --> DASH
    AA --> DASH
    FR --> DASH
```

## Services (docker-compose)

| Container | Image / Build | Port | Role |
|---|---|---|---|
| `kafka1` | `confluentinc/cp-kafka` (KRaft) | 9092 | Message broker, single node |
| `kafka-init` | cp-kafka (one-shot) | — | Creates topics with 10 partitions |
| `timescaledb` | `timescale/timescaledb:latest-pg16` | 5432 | Primary storage (hypertables), 3 g mem |
| `spark` | `./infra/spark/Dockerfile` | 4040 (UI) | Spark master |
| `spark-streaming` | same Spark image | — | Runs `feature_pipeline.py` continuously |
| `replay-producer` | `pipeline/replay_producer/` | 8000 (FastAPI) | M5 CSV → Kafka |
| `stream-consumer` | `pipeline/stream_processor/` | — | `sink.py`: Kafka → `sales_events` + `pipeline_metrics` |
| `anomaly-detector` | `pipeline/stream_processor/` | — | `anomaly_detector.py`: polls every 5 s |
| `demand-forecaster` | `pipeline/stream_processor/` | — | `demand_forecaster.py`: polls every 60 s |
| `streamlit-dashboard` | `dashboard/` | 8501 | Live monitoring UI |

## TimescaleDB Tables

| Table | Hypertable Column | Purpose |
|---|---|---|
| `sales_events` | `time` (M5 date) | Raw Kafka events; `ingested_at` for RQ3 |
| `sales_features` | `time` | 7-day rolling features from Spark / SQL |
| `anomaly_alerts` | `detected_at` | IsolationForest output; `detection_latency_ms` for RQ1 |
| `forecast_results` | `created_at` | ESM forecasts; `feature_source` for RQ2 comparison |
| `pipeline_metrics` | `time` | Throughput + processing lag; P50/P95/P99 for RQ3 |

## Research Questions → Measurement Sources

| RQ | Summary | Measurement Source |
|---|---|---|
| RQ1 | Streaming vs batch anomaly detection latency | `statistical_tests.py` → batch baseline ms vs 5 s poll |
| RQ2 | Streaming vs batch feature forecast accuracy | `statistical_tests.py` → per-item MAPE, Wilcoxon, DM |
| RQ3 | Architecture reliability and throughput | `pipeline_metrics` → `streaming_metrics.py` |
