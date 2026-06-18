# Real-Time Data Engineering Platform for Retail Inventory

**Anomaly Detection and Demand Forecasting at Stream Scale**

[![Status](https://img.shields.io/badge/status-in_progress-yellow.svg)]()
[![Field](https://img.shields.io/badge/field-Data_Engineering-green.svg)]()

> A streaming-first reference architecture combining Kafka, Spark Structured Streaming, and TimescaleDB with a hybrid anomaly detection and three-tier forecasting evaluation on the M5 dataset.

---

## Research Questions

- **RQ1.** What latency–accuracy trade-offs does a stream-processing approach offer over batch processing for anomaly detection on retail inventory data?
- **RQ2.** How much better do demand forecasting models perform when fed by a streaming feature pipeline compared to a static (batch) feature pipeline?
- **RQ3.** What reliability and throughput characteristics can an event-driven reference architecture deliver for end-to-end retail supply chain visibility?

---

## Architecture Overview

```
[Replay Producer] ──► [Kafka topics] ──► [Spark Structured Streaming] ──► [TimescaleDB]
   (M5 → events)                                  │                            │
                                                  ▼                            ▼
                                       [Model Service]                  [Dashboard]
                                       - Forecasting (3 tier)           (Grafana / Streamlit)
                                       - Anomaly (IF + LSTM-AE)
                                       - Conformal calibration
```

---

## Repository Layout

```
.
├── infra/            # Docker Compose, Kafka topics, Timescale init SQL
├── pipeline/         # Replay producer, stream processor, feature engineering
├── models/           # Forecasting + anomaly detection + conformal
├── evaluation/       # Metrics, statistical tests, anomaly injection
├── notebooks/        # EDA + analysis
├── dashboard/        # Grafana / Streamlit
├── paper/            # Conference paper draft + figures
├── tests/            # Unit + integration tests
├── scripts/          # Setup, data download, run-all
└── data/             # Gitignored — see data/README.md
```

> Detailed roadmap, methodology, reference bibliography, and per-paper notes
> are maintained as internal working documents and are not part of the public
> repository.

---

## Getting Started

> **Note:** Code is under active development. The instructions below are a placeholder for the final repo structure.

```bash
# 1. Clone the repo
git clone <repo-url>
cd real-time-retail-inventory

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate     # Windows
pip install -r requirements.txt

# 3. Download datasets (see data/README.md for details)
bash scripts/download_data.sh

# 4. Start infrastructure (Kafka + Timescale)
docker-compose -f infra/docker-compose.yml up -d

# 5. Run the replay producer
python -m pipeline.replay_producer --dataset m5 --speed 60

# 6. Start the stream processor
python -m pipeline.stream_processor
```

---

## Datasets

| Dataset | Role | Source |
|---|---|---|
| **M5 Forecasting Accuracy** | Primary — forecasting + anomaly evaluation | [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy) |
| **Online Retail II (UCI)** | Secondary — anomaly motif sampling | [UCI](https://archive.ics.uci.edu/dataset/502) |

Datasets are **not** committed to the repo. See [`data/README.md`](data/README.md) for download instructions.

---

## Documentation

- [data/README.md](data/README.md) — Dataset download instructions

Implementation documentation will be added here as code lands.

---

## Contributing (team only)

We use a 1-week sprint cadence:
- **Monday:** sprint planning (45 min)
- **Wed/Fri:** async stand-up
- **Friday:** sprint review + retrospective

Branch naming: `<initials>/<sprint>-<short-description>`, e.g. `hk/s3-kafka-replay`
PRs target `main`, require one review.

---

## Citation

When the conference paper is published, citation info will appear here.

---

## License

Code: MIT License (see [LICENSE](LICENSE) when added).
Documentation: CC BY 4.0.
Datasets: respective original licenses (see `data/README.md`).
