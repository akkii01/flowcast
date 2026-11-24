# FlowCast — AI-Powered Smart Traffic Flow Prediction Platform

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/Classical%20ML-XGBoost-1f77b4.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Pytest-5%2F5%20Passed-10b981.svg)](#automated-testing)

**Repository:** https://github.com/akkii01/flowcast  
**Author:** Akshit Raj  
**Programme:** Meridian Mobility Systems — Northline Corridor (25 Arterial Segments)

---

## Executive Summary

FlowCast is an end-to-end urban traffic intelligence platform that ingests multi-source loop/radar sensor telemetry, hourly weather station feeds, and corridor calendar signals to forecast short-horizon **traffic volume**, **travel time**, **4-class congestion severity**, and **binary accident risk** across 25 arterial road segments (30-minute aggregation, Jan–May 2025).

### Key Engineering Highlights
- **Zero Silent Drops (M1–M2):** Validated 3 linked CSV feeds (`178,468` traffic rows, `10,872` hourly weather rows, `151` calendar rows), quarantined `2,479` defective records to an audit log, removed `1,767` duplicates, repaired sensor outliers, reconstructed `4,499` dropped 30-min windows onto a `181,200`-row clean grid, harmonised mixed weather labels/date formats, and derived missing congestion bands via `V/C = volume / (capacity / 2)`.
- **Leakage-Free Feature Store (M3):** Engineered `85` strictly past features (cyclical `sin/cos` time encodings, short-term `t-1..t-4`, daily `t-48`, and weekly `t-336` lags, shifted rolling statistics, weather interactions, and train-only segment baselines) across strictly chronological `70% / 15% / 15%` splits (`120,950` train / `25,925` val / `25,925` test).
- **Explicit Mathematics (PRD Section 10):** Implemented Linear Regression via **from-scratch NumPy Gradient Descent** (`NumPyLinearRegressionGD`) alongside Min-Max + Z-score dual scaling and validation-calibrated 90% empirical prediction intervals.
- **Classical ML vs. PyTorch Deep Learning (M5–M6):** Benchmarked Linear Regression, Decision Tree, Random Forest, XGBoost, and Calibrated LinearSVC against scratch-trained **PyTorch Multi-Task LSTM & BiLSTM** sequence forecasters on the exact same held-out test window.
- **Interactive Operations Command Center (M7):** Built a dark glassmorphic Streamlit dashboard exposing all **9 required operational views** plus interactive 30–120 min multi-horizon simulation, CSV batch prediction upload, model cards, and insights report export.

---

## Held-Out Test Benchmark (25,925 Unseen Segment-Windows)

### 1. Traffic Volume & Travel Time Regression

| Family | Task | Model | Test RMSE | Test MAE | Test MAPE | Test R² |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| Classical ML | Traffic Volume | **XGBoost** | **52.59** | **35.37** | **8.20%** | **0.9665** |
| Deep Learning (PyTorch) | Traffic Volume | BiLSTM (Multi-Task) | 52.87 | 35.97 | 8.77% | 0.9661 |
| Classical ML | Traffic Volume | Random Forest | 52.90 | 35.55 | 8.29% | 0.9661 |
| Deep Learning (PyTorch) | Traffic Volume | LSTM (Multi-Task) | 53.24 | 36.41 | 9.10% | 0.9656 |
| Classical ML | Traffic Volume | Decision Tree | 54.45 | 36.57 | 8.57% | 0.9641 |
| Classical ML | Traffic Volume | Linear Regression (sklearn) | 55.27 | 37.98 | 9.65% | 0.9630 |
| Classical ML (Math) | Traffic Volume | Linear Regression (NumPy GD) | 56.75 | 39.10 | 10.03% | 0.9610 |
| Classical ML | Travel Time | **XGBoost (Two-Stage)** | **1.03** | **0.40** | **8.48%** | **0.8444** |
| Classical ML | Travel Time | Random Forest | 1.03 | 0.41 | 8.39% | 0.8441 |

*PRD Volume Target (`MAPE <= 12%`) surpassed across both Classical ML (`8.20%`) and Deep Learning (`8.77%`).*

### 2. Congestion (4-Class) & Accident Risk (Binary) Classification

| Family | Task | Model | Macro-F1 | Accuracy | ROC-AUC |
| :--- | :--- | :--- | ---: | ---: | ---: |
| Deep Learning (PyTorch) | Congestion (4-Class) | **LSTM (Multi-Task)** | **0.7763** | **88.44%** | — |
| Deep Learning (PyTorch) | Congestion (4-Class) | BiLSTM (Multi-Task) | 0.7748 | 88.29% | — |
| Classical ML | Congestion (4-Class) | XGBoost (Val-Calibrated) | 0.7748 | 88.19% | — |
| Classical ML | Congestion (4-Class) | Random Forest | 0.7715 | 88.30% | — |
| Classical ML | Congestion (4-Class) | SVM Baseline (Calibrated LinearSVC) | 0.7256 | 86.43% | — |
| Classical ML | Accident Risk (Binary) | **XGBoost (Regularized)** | — | — | **0.6602** |
| Classical ML | Accident Risk (Binary) | SVM Baseline (Calibrated LinearSVC) | — | — | 0.6548 |
| Classical ML | Accident Risk (Binary) | Logistic Regression (L2) | — | — | 0.6479 |
| Classical ML | Accident Risk (Binary) | Random Forest | — | — | 0.6383 |

*See `reports/final_technical_report.md` for full error analysis, boundary overlap breakdown, and rare-event signal-ceiling discussion.*

---

## Repository Structure

```text
flowcast/
├── data/
│   ├── raw/                  # Raw CSV feeds (traffic, weather, calendar; ignored in git)
│   ├── interim/              # Quarantine defect log, audit JSON, cleaned parquet
│   └── processed/            # Leakage-free engineered feature store & metadata
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory analysis & weather/diurnal insights
│   ├── 02_cleaning.ipynb     # Data-quality defects & deterministic cleaning audit
│   └── 03_modelling.ipynb    # Classical ML vs PyTorch DL benchmark narrative
├── src/
│   ├── ingest.py             # M1: Schema validation & defect quarantine
│   ├── clean.py              # M2: Cleaning, alignment, merge & quality report
│   ├── features.py           # M3: Lags (incl. t-336), rolling stats & splits
│   ├── ml_models.py          # M5: NumPy GD + Tree Ensembles + XGBoost + SVM
│   ├── dl_model.py           # M6: Scratch PyTorch Multi-Task LSTM & BiLSTM
│   ├── evaluate.py           # Metrics, error analysis & JSON model cards
│   ├── build_reports.py      # M4: Figures, notebooks & final technical report
│   └── utils.py              # Logging, YAML config & deterministic seeding
├── models/                   # Persisted models (.joblib, .pt), scoreboards & model cards
├── reports/
│   ├── data_quality.md       # Full data-quality audit report
│   ├── final_technical_report.md
│   └── figures/              # 5 EDA & benchmark PNG visualisations
├── dashboard/
│   └── app.py                # M7: Streamlit Command Center (all 9 PRD views)
├── tests/
│   └── test_pipeline.py      # Pytest suite (data quality, leakage, GD math, artefacts)
├── run_pipeline.py           # One-command full pipeline runner (M1 -> M6)
├── config.yaml               # Central pipeline & hyperparameter configuration
├── requirements.txt          # Pinned Python dependencies
└── README.md                 # Project documentation
```

---

## Quickstart & One-Command Reproduction

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/akkii01/flowcast.git
cd flowcast
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

*(On macOS/Linux, activate using `source venv/bin/activate`.)*

### 2. Place Raw Data & Run Full End-to-End Pipeline

Place `calendar_events.csv`, `weather_observations.csv`, and `traffic_sensor_log.csv` inside `data/raw/`, then reproduce all cleaned data, features, classical models, deep learning models, and reports:

```bash
python run_pipeline.py
python src/build_reports.py
```

*(Optional fast run skipping PyTorch sequence training: `python run_pipeline.py --skip-dl`)*

### 3. Launch Interactive Analytics Dashboard

```bash
streamlit run dashboard/app.py
```

Once the local server starts, open your browser and visit:

```text
http://localhost:8501
```

### 4. Automated Testing

```bash
pytest tests/test_pipeline.py -v
```