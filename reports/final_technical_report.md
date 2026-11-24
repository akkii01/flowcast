# FlowCast v1.0 — Final Technical Report

**Programme:** Meridian Mobility Systems — Northline Corridor (25 Arterial Segments)  
**Author:** Akshit Raj  
**Scope:** End-to-End Traffic Volume, Congestion, Travel Time & Accident Risk Prediction Platform

---

## 1. Executive Summary
FlowCast transforms raw half-hourly inductive-loop/radar telemetry, hourly weather station feeds, and corridor calendar signals into reproducible 30-to-120-minute traffic forecasts and operational analytics. Across the unseen chronological test period (25,925 segment-windows), the system delivers:
- **Traffic Volume Forecast:** XGBoost achieves **RMSE = 52.59**, **MAPE = 8.20%** (surpassing the PRD `<= 12%` target), and **$R^2 = 0.9665$**.
- **Deep Learning Sequence Forecasting:** Scratch-trained PyTorch Multi-Task **BiLSTM** reaches **RMSE = 52.87** (`MAPE = 8.77%`, beating Random Forest, Decision Tree, and Linear Regression), while **LSTM** achieves the highest 4-class congestion score (**Macro-F1 = 0.7763**, **Accuracy = 88.44%**), beating every classical classifier.
- **Travel Time Forecasting:** Two-stage XGBoost achieves **RMSE = 1.0271 min**, **MAPE = 8.48%**, and **$R^2 = 0.8444$**.
- **Accident Risk Ranking:** Regularized XGBoost achieves **ROC-AUC = 0.6602** on a highly imbalanced rare-event target (0.92% positive rate), accurately elevating patrol priority during Rain/Fog and Severe congestion windows.

---

## 2. Data Engineering & Leakage-Free Feature Store
- **Ingestion & Quarantine (M1):** Validated 3 linked CSV sources (`178,468` traffic rows, `10,872` hourly weather rows, `151` calendar rows) and quarantined `2,479` defective rows to `data/interim/quarantine_traffic_defects.csv` with zero silent drops.
- **Deterministic Cleaning (M2):** Removed `1,767` key duplicates, repaired negative volume / `>200 km/h` speed / `>100%` occupancy sensor faults, reconstructed `4,499` dropped 30-minute windows onto the complete `181,200`-row grid, harmonised inconsistent weather strings into `['Clear', 'Cloudy', 'Fog', 'Rain']`, aligned `DD/MM/YYYY` hourly weather onto 30-min traffic windows, and derived missing congestion labels via `V/C = volume / (capacity / 2)`.
- **Feature Engineering & Temporal Discipline (M3):** Built 85 strictly leakage-free features including cyclical `sin/cos` time encodings, per-segment short-term (`t-1..t-4`), daily (`t-48`), and weekly (`t-336`) lags, shifted rolling mean/std windows (`4, 8, 48`), and train-only `(road_id, time_slot, is_weekend)` baselines. Data was split strictly chronologically (`70% Train / 15% Val / 15% Test`) with zero temporal overlap.

---

## 3. Explicit Mathematics Implemented (PRD Section 10)
- **From-Scratch NumPy Gradient Descent (`NumPyLinearRegressionGD`):** Implemented matrix-vector forward prediction ($\hat{y} = Xw + b$), L2-regularized MSE loss, and analytical gradients ($\nabla_w = \frac{2}{n}X^T(\hat{y}-y) + 2\lambda w$) over 800 epochs. Achieved **RMSE = 56.75** ($R^2 = 0.9610$), closely matching scikit-learn's closed-form OLS (`RMSE = 55.27`).
- **Dual Feature Scaling:** Applied **Min-Max Normalisation** (`[0, 1]`) to bounded physical sensors (`occupancy`, `visibility`) and **Z-Score Standardisation** ($\mu=0, \sigma=1$) to unbounded telemetry (`traffic_volume`, `avg_speed`, `travel_time`), strictly fit on the training split.
- **Empirical Prediction Confidence (FR-11):** Constructed calibrated 90% prediction intervals from validation absolute residuals, achieving ~90% empirical coverage on the held-out test set.

---

## 4. Classical ML vs. Deep Learning Head-to-Head Benchmark (PRD Section 12.4)

| Family | Task | Model | Test RMSE | Test MAPE (%) | Test $R^2$ | Macro-F1 | Accuracy |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Classical ML | Traffic Volume | Linear Regression (NumPy GD) | 56.75 | 10.03% | 0.9610 | — | — |
| Classical ML | Traffic Volume | XGBoost | **52.59** | **8.20%** | **0.9665** | — | — |
| Deep Learning (PyTorch) | Traffic Volume | LSTM (Multi-Task) | 53.24 | 9.10% | 0.9656 | — | — |
| Deep Learning (PyTorch) | Traffic Volume | BiLSTM (Multi-Task) | 52.87 | 8.77% | 0.9661 | — | — |
| Classical ML | Congestion (4-Class) | XGBoost (Val-Calibrated) | — | — | — | 0.7256 | 86.43% |
| Deep Learning (PyTorch) | Congestion (4-Class) | LSTM (Multi-Task) | — | — | — | **0.7763** | **88.44%** |

### Honest Model Selection & Signal-Ceiling Analysis
1. **Did Deep Learning earn its added complexity?** For **4-class congestion classification**, the PyTorch Multi-Task LSTM outperformed every classical baseline (`0.7763` Macro-F1 / `88.44%` accuracy), proving that multi-step 6-hour trajectories capture build-up transitions well. For **volume regression**, BiLSTM (`52.87` RMSE) beat Random Forest, Decision Tree, and Linear Regression and came within **0.28 vehicles (0.5%)** of XGBoost (`52.59` RMSE). Given XGBoost's ~30× faster CPU training and instant inference, XGBoost is recommended for real-time volume serving while LSTM/BiLSTM strengthens sequence-based congestion classification.
2. **Congestion Boundary Overlap (`0.776` vs `0.80` target):** Nearly all misclassifications occur strictly between adjacent V/C threshold boundaries (`Moderate <-> Heavy` and `Heavy <-> Severe`) due to natural 30-minute stochastic demand noise around the `0.80` and `1.00` cutoffs.
3. **Accident Risk Rare-Event Ceiling (`0.660` vs `0.75` target):** Only `0.92%` of corridor windows contain an incident, and the strongest individual feature correlation (`visibility`) is `-0.057`. Without leaking concurrent target-window sensor drops, `~0.66 ROC-AUC` reflects the true stochastic signal ceiling of pre-incident telemetry. Operational patrol deployment should combine risk scores with live rain/fog alerts where incident rates rise ~4× (`3.4%` vs `0.8%`).

---

## 5. Error Analysis & Operational Recommendations
- **Residual Distribution:** Mean residual = `1.4035`, Std = `55.2533`, 5th–95th percentile band = `[-84.316, 90.714]` (unbiased, near-symmetric residuals).
- **Peak-Hour Signal Pre-Emption:** Morning (08:00–10:00) and evening (17:30–19:30) peaks exhibit both the highest volume and highest forecast RMSE; operations analysts should trigger adaptive green-time extensions 30 minutes prior to predicted `V/C >= 0.80` transitions.
- **Weather-Triggered Patrol Staging:** Because Rain and Fog windows quadruple incident probability, incident coordinators should pre-position rapid-response units along high-risk segments (`NL-016`, `NL-025`, `NL-006`) whenever visibility drops below `2,000 m` or hourly rainfall exceeds `2.0 mm`.
