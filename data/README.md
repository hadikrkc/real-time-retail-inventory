# Datasets

⚠️ **The `data/` folder contents are NOT committed to git** (see `.gitignore`). This file IS committed and explains how to obtain the datasets.

## Required: M5 Forecasting Accuracy (Walmart)

**Role in project:** Primary evaluation dataset for both forecasting (RQ2) and anomaly detection (RQ1).

**Size:** ~450 MB (CSV files).

**Source:** [Kaggle competition page](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).

**License:** Walmart/M5 organizers — free for academic use, citation required.

### Download via Kaggle CLI (recommended)

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Place kaggle.json (API token) in the right location:
#    Linux/Mac: ~/.kaggle/kaggle.json
#    Windows:   %USERPROFILE%\.kaggle\kaggle.json
#    (Get token from kaggle.com → Profile → Account → "Create New API Token")

# 3. Download into this folder
cd data
kaggle competitions download -c m5-forecasting-accuracy
unzip m5-forecasting-accuracy.zip -d m5/
rm m5-forecasting-accuracy.zip
```

### Manual download

1. Go to https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
2. Accept competition rules ("I understand and accept")
3. Click "Download All"
4. Extract the zip into `data/m5/`

### Expected files in `data/m5/`

| File | Size | Description |
|---|---|---|
| `sales_train_evaluation.csv` | ~129 MB | 30,490 series × 1,941 days |
| `sales_train_validation.csv` | ~123 MB | Validation split |
| `sell_prices.csv` | ~123 MB | Weekly prices per item per store |
| `calendar.csv` | ~103 KB | Dates + holidays + SNAP flags |
| `sample_submission.csv` | ~5 MB | Format template |

---

## Optional: Online Retail II (UCI)

**Role in project:** Secondary — source for realistic anomaly motifs (returns, fraud-like patterns) to use as templates for synthetic anomaly injection on M5.

**Size:** ~50 MB.

**Source:** [Kaggle mirror](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) or [UCI ML Repository](https://archive.ics.uci.edu/dataset/502).

**License:** CC BY 4.0 — citation required.

### Download

```bash
cd data
kaggle datasets download -d mashlyn/online-retail-ii-uci
unzip online-retail-ii-uci.zip -d online_retail/
rm online-retail-ii-uci.zip
```

### Citation

```
Chen, D. (2015). Online Retail II [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C5BW33
```

---

## Folder Layout (after downloads)

```
data/
├── README.md                ← committed
├── m5/                      ← gitignored
│   ├── calendar.csv
│   ├── sales_train_evaluation.csv
│   ├── sales_train_validation.csv
│   ├── sample_submission.csv
│   └── sell_prices.csv
└── online_retail/           ← gitignored
    └── online_retail_II.csv
```

---

## Troubleshooting

**"403 Forbidden" on Kaggle:** Make sure you've accepted the competition rules on the M5 page first.

**Kaggle CLI not finding token:** Check `kaggle.json` permissions (Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`).

**Disk space:** Reserve ~600 MB for both datasets combined plus working space.
