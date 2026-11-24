"""M5: Classical ML Engine (From-Scratch NumPy GD + Stacked Multi-Task Ensembles + SVM)."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluate import (
    compute_binary_metrics,
    compute_error_analysis,
    compute_multiclass_metrics,
    compute_regression_metrics,
    find_best_f1_threshold,
    save_model_card,
)
from src.utils import load_config, set_global_seed, setup_logger

logger = setup_logger("M5_ClassicalML")


class NumPyLinearRegressionGD:
    """From-scratch Linear Regression trained via Gradient Descent (PRD Section 10)."""

    def __init__(
        self, learning_rate: float = 0.01, epochs: int = 800, l2_reg: float = 0.01
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.weights: np.ndarray = np.array([])
        self.bias: float = 0.0
        self.loss_history: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumPyLinearRegressionGD":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = float(np.mean(y))
        self.loss_history = []

        for _ in range(self.epochs):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y
            mse = float(np.mean(error**2))
            self.loss_history.append(mse)

            dw = (2.0 / n_samples) * (X.T @ error) + 2.0 * self.l2_reg * self.weights
            db = (2.0 / n_samples) * float(np.sum(error))

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


def extract_feature_importance(model, feature_cols: List[str]) -> pd.DataFrame:
    """Extract sorted feature importance dataframe from fitted model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(np.ravel(model.coef_))[: len(feature_cols)]
    else:
        importances = np.zeros(len(feature_cols))

    return (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def tune_multiclass_multipliers(
    y_val: np.ndarray, val_probs: np.ndarray
) -> np.ndarray:
    """Tune class probability multipliers on validation set to maximize Macro-F1."""
    best_mult = np.ones(4, dtype=np.float64)
    best_f1, _ = compute_multiclass_metrics(y_val, np.argmax(val_probs, axis=1))
    best_score = best_f1["Macro_F1"]

    for _ in range(3):
        for cls_idx in range(4):
            for factor in np.linspace(0.70, 1.50, 33):
                cand = best_mult.copy()
                cand[cls_idx] = factor
                preds = np.argmax(val_probs * cand, axis=1)
                m, _ = compute_multiclass_metrics(y_val, preds)
                if m["Macro_F1"] > best_score:
                    best_score = m["Macro_F1"]
                    best_mult = cand
    return np.round(best_mult, 4)


def vc_to_congestion_id(vc_values: np.ndarray) -> np.ndarray:
    """Map predicted V/C ratio to PRD Data Dictionary congestion bands."""
    return np.select(
        [vc_values < 0.50, vc_values < 0.80, vc_values < 1.00],
        [0, 1, 2],
        default=3,
    ).astype(int)


def train_classical_models(config_path: str = "config.yaml") -> pd.DataFrame:
    """Train, validate, benchmark, and export all M5 classical ML models."""
    start_time = time.time()
    config = load_config(config_path)
    seed = config["project"]["random_seed"]
    set_global_seed(seed)

    feat_path = Path(config["paths"]["processed_features"])
    meta_path = feat_path.parent / "feature_metadata.json"
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading processed feature store from {feat_path}...")
    df = pd.read_parquet(feat_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols: List[str] = metadata["feature_columns"]
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_window = [metadata["splits"]["train"]["start"], metadata["splits"]["train"]["end"]]
    test_window = [metadata["splits"]["test"]["start"], metadata["splits"]["test"]["end"]]

    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float64)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)

    # Fit Stage-1 StandardScaler strictly on train split (PRD Leakage Guard)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, models_dir / "feature_scaler.joblib")
    logger.info("Fitted Stage-1 StandardScaler on train split only -> models/feature_scaler.joblib")

    scoreboard_rows: List[Dict] = []

    # =========================================================================
    # TASK 1: TRAFFIC VOLUME REGRESSION (STAGE 1)
    # =========================================================================
    logger.info("=== Task 1/4: Traffic Volume Regression ===")
    y_tr_vol = train_df["target_volume"].to_numpy(dtype=np.float64)
    y_va_vol = val_df["target_volume"].to_numpy(dtype=np.float64)
    y_te_vol = test_df["target_volume"].to_numpy(dtype=np.float64)

    logger.info("Training Linear Regression (NumPy GD from scratch)...")
    gd_model = NumPyLinearRegressionGD(learning_rate=0.01, epochs=800, l2_reg=0.01)
    gd_model.fit(X_train_scaled, y_tr_vol)
    gd_val_m = compute_regression_metrics(y_va_vol, gd_model.predict(X_val_scaled))
    gd_test_m = compute_regression_metrics(y_te_vol, gd_model.predict(X_test_scaled))
    pd.DataFrame(
        {"epoch": range(1, len(gd_model.loss_history) + 1), "train_mse": gd_model.loss_history}
    ).to_csv(models_dir / "numpy_gd_loss_history.csv", index=False)
    scoreboard_rows.append(
        {"Task": "Traffic Volume", "Model": "Linear Regression (NumPy GD)", **gd_test_m}
    )

    vol_candidates = {
        "Linear Regression (sklearn)": (LinearRegression(), True),
        "Decision Tree": (
            DecisionTreeRegressor(max_depth=12, min_samples_leaf=10, random_state=seed),
            False,
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=80,
                max_depth=14,
                min_samples_leaf=5,
                max_samples=0.35,
                n_jobs=-1,
                random_state=seed,
            ),
            False,
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=700,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.5,
                tree_method="hist",
                early_stopping_rounds=25,
                n_jobs=-1,
                random_state=seed,
            ),
            False,
        ),
    }

    best_vol_name, best_vol_model, best_vol_val_rmse = (
        "Linear Regression (NumPy GD)",
        gd_model,
        gd_val_m["RMSE"],
    )
    best_vol_val_m, best_vol_test_m, best_vol_scaled = gd_val_m, gd_test_m, True

    for name, (model, use_scaled) in vol_candidates.items():
        logger.info(f"Training Volume {name}...")
        Xtr = X_train_scaled if use_scaled else X_train
        Xva = X_val_scaled if use_scaled else X_val
        Xte = X_test_scaled if use_scaled else X_test

        if name == "XGBoost":
            model.fit(Xtr, y_tr_vol, eval_set=[(Xva, y_va_vol)], verbose=False)
        else:
            model.fit(Xtr, y_tr_vol)

        va_m = compute_regression_metrics(y_va_vol, model.predict(Xva))
        te_m = compute_regression_metrics(y_te_vol, model.predict(Xte))
        scoreboard_rows.append({"Task": "Traffic Volume", "Model": name, **te_m})

        if va_m["RMSE"] < best_vol_val_rmse:
            best_vol_val_rmse = va_m["RMSE"]
            best_vol_name = name
            best_vol_model = model
            best_vol_val_m = va_m
            best_vol_test_m = te_m
            best_vol_scaled = use_scaled

    joblib.dump(best_vol_model, models_dir / "volume_best_model.joblib")
    extract_feature_importance(best_vol_model, feature_cols).to_csv(
        models_dir / "volume_feature_importance.csv", index=False
    )

    vol_val_preds = np.clip(
        best_vol_model.predict(X_val_scaled if best_vol_scaled else X_val), 0.0, None
    )
    vol_test_preds = np.clip(
        best_vol_model.predict(X_test_scaled if best_vol_scaled else X_test), 0.0, None
    )
    vol_q90_halfwidth = float(np.percentile(np.abs(y_va_vol - vol_val_preds), 90))

    save_model_card(
        model_name="volume_classical_winner",
        task_name="Traffic Volume Regression",
        algorithm=best_vol_name,
        train_window=train_window,
        test_window=test_window,
        features=feature_cols,
        hyperparameters={
            "winner": best_vol_name,
            "val_q90_halfwidth": round(vol_q90_halfwidth, 3),
        },
        val_metrics=best_vol_val_m,
        test_metrics=best_vol_test_m,
        selection_rationale=f"Selected {best_vol_name} based on lowest chronological validation RMSE ({best_vol_val_m['RMSE']}).",
        models_dir=str(models_dir),
    )

    # =========================================================================
    # STAGE-1 -> STAGE-2 LEAKAGE-FREE OUT-OF-FOLD STACKING FEATURES
    # =========================================================================
    logger.info("Computing out-of-fold Stage-1 predicted volume & V/C ratio for Stage-2 tasks...")
    oof_vol_train = np.zeros(len(train_df), dtype=np.float64)
    kf = KFold(n_splits=4, shuffle=False)
    for tr_idx, hold_idx in kf.split(X_train):
        oof_model = XGBRegressor(
            n_estimators=220,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
        oof_model.fit(X_train[tr_idx], y_tr_vol[tr_idx])
        oof_vol_train[hold_idx] = np.clip(oof_model.predict(X_train[hold_idx]), 0.0, None)

    cap_tr = train_df["road_capacity"].to_numpy(dtype=np.float64) / 2.0
    cap_va = val_df["road_capacity"].to_numpy(dtype=np.float64) / 2.0
    cap_te = test_df["road_capacity"].to_numpy(dtype=np.float64) / 2.0

    pred_vc_tr = oof_vol_train / cap_tr
    pred_vc_va = vol_val_preds / cap_va
    pred_vc_te = vol_test_preds / cap_te

    stage2_extra_cols = [
        "pred_vol_s1",
        "pred_vc_s1",
        "pred_headroom_s1",
        "rain_x_pred_vc_s1",
        "fog_x_pred_vc_s1",
    ]
    stage2_features = feature_cols + stage2_extra_cols

    def _build_s2(X_base: np.ndarray, df_ref: pd.DataFrame, p_vol: np.ndarray, p_vc: np.ndarray, cap: np.ndarray) -> np.ndarray:
        extra = np.column_stack(
            [
                p_vol,
                p_vc,
                cap - p_vol,
                df_ref["is_rain"].to_numpy(dtype=np.float64) * p_vc,
                df_ref["is_fog"].to_numpy(dtype=np.float64) * p_vc,
            ]
        )
        return np.hstack([X_base, extra])

    X_tr_s2 = _build_s2(X_train, train_df, oof_vol_train, pred_vc_tr, cap_tr)
    X_va_s2 = _build_s2(X_val, val_df, vol_val_preds, pred_vc_va, cap_va)
    X_te_s2 = _build_s2(X_test, test_df, vol_test_preds, pred_vc_te, cap_te)

    scaler_s2 = StandardScaler()
    X_tr_s2_sc = scaler_s2.fit_transform(X_tr_s2)
    X_va_s2_sc = scaler_s2.transform(X_va_s2)
    X_te_s2_sc = scaler_s2.transform(X_te_s2)
    joblib.dump(scaler_s2, models_dir / "feature_scaler_s2.joblib")

    # =========================================================================
    # TASK 2: TRAVEL TIME REGRESSION (STAGE 2)
    # =========================================================================
    logger.info("=== Task 2/4: Travel Time Regression ===")
    y_tr_tt = train_df["target_travel_time"].to_numpy(dtype=np.float64)
    y_va_tt = val_df["target_travel_time"].to_numpy(dtype=np.float64)
    y_te_tt = test_df["target_travel_time"].to_numpy(dtype=np.float64)

    tt_candidates = {
        "Linear Regression": (LinearRegression(), True),
        "Decision Tree": (
            DecisionTreeRegressor(max_depth=12, min_samples_leaf=10, random_state=seed),
            False,
        ),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=80,
                max_depth=14,
                min_samples_leaf=5,
                max_samples=0.35,
                n_jobs=-1,
                random_state=seed,
            ),
            False,
        ),
        "XGBoost": (
            XGBRegressor(
                n_estimators=600,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.5,
                tree_method="hist",
                early_stopping_rounds=25,
                n_jobs=-1,
                random_state=seed,
            ),
            False,
        ),
    }

    best_tt_name, best_tt_model, best_tt_val_rmse = "", None, float("inf")
    best_tt_val_m, best_tt_test_m, best_tt_scaled = {}, {}, False

    for name, (model, use_scaled) in tt_candidates.items():
        logger.info(f"Training Travel Time {name}...")
        Xtr = X_tr_s2_sc if use_scaled else X_tr_s2
        Xva = X_va_s2_sc if use_scaled else X_va_s2
        Xte = X_te_s2_sc if use_scaled else X_te_s2

        if name == "XGBoost":
            model.fit(Xtr, y_tr_tt, eval_set=[(Xva, y_va_tt)], verbose=False)
        else:
            model.fit(Xtr, y_tr_tt)

        va_m = compute_regression_metrics(y_va_tt, model.predict(Xva))
        te_m = compute_regression_metrics(y_te_tt, model.predict(Xte))
        scoreboard_rows.append({"Task": "Travel Time", "Model": name, **te_m})

        if va_m["RMSE"] < best_tt_val_rmse:
            best_tt_val_rmse = va_m["RMSE"]
            best_tt_name = name
            best_tt_model = model
            best_tt_val_m = va_m
            best_tt_test_m = te_m
            best_tt_scaled = use_scaled

    joblib.dump(best_tt_model, models_dir / "travel_time_best_model.joblib")
    extract_feature_importance(best_tt_model, stage2_features).to_csv(
        models_dir / "travel_time_feature_importance.csv", index=False
    )
    tt_test_preds = np.clip(
        best_tt_model.predict(X_te_s2_sc if best_tt_scaled else X_te_s2), 0.1, None
    )

    save_model_card(
        model_name="travel_time_classical_winner",
        task_name="Travel Time Regression",
        algorithm=best_tt_name,
        train_window=train_window,
        test_window=test_window,
        features=stage2_features,
        hyperparameters={"winner": best_tt_name},
        val_metrics=best_tt_val_m,
        test_metrics=best_tt_test_m,
        selection_rationale=f"Selected {best_tt_name} using Stage-1 flow predictions based on lowest validation RMSE ({best_tt_val_m['RMSE']}).",
        models_dir=str(models_dir),
    )

    # =========================================================================
    # TASK 3: CONGESTION LEVEL CLASSIFICATION (4-CLASS, STAGE 2)
    # =========================================================================
    logger.info("=== Task 3/4: Congestion Level Classification (4-Class) ===")
    y_tr_cong = train_df["target_congestion_id"].to_numpy(dtype=int)
    y_va_cong = val_df["target_congestion_id"].to_numpy(dtype=int)
    y_te_cong = test_df["target_congestion_id"].to_numpy(dtype=int)

    # Benchmark 3A: Direct V/C Rule-Banding from Stage-1 Volume Forecast
    vc_va_m, _ = compute_multiclass_metrics(y_va_cong, vc_to_congestion_id(pred_vc_va))
    vc_te_m, _ = compute_multiclass_metrics(y_te_cong, vc_to_congestion_id(pred_vc_te))
    scoreboard_rows.append(
        {"Task": "Congestion (4-Class)", "Model": "Stage-1 V/C Rule Banding", **vc_te_m}
    )

    cong_candidates = {
        "SVM Baseline (Calibrated LinearSVC)": (
            CalibratedClassifierCV(
                LinearSVC(C=0.1, tol=1e-3, max_iter=1500, class_weight="balanced", random_state=seed),
                cv=3,
            ),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=seed),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=80,
                max_depth=14,
                min_samples_leaf=5,
                max_samples=0.35,
                n_jobs=-1,
                random_state=seed,
            ),
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=600,
                learning_rate=0.04,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=4,
                eval_metric="mlogloss",
                tree_method="hist",
                early_stopping_rounds=25,
                n_jobs=-1,
                random_state=seed,
            ),
            False,
        ),
    }

    best_cong_name, best_cong_model, best_cong_val_f1 = "", None, -1.0
    best_cong_val_m, best_cong_test_m, best_cong_cm, best_cong_scaled = {}, {}, None, False
    best_cong_mult = np.ones(4, dtype=np.float64)

    for name, (model, use_scaled) in cong_candidates.items():
        logger.info(f"Training Congestion {name}...")
        Xtr = X_tr_s2_sc if use_scaled else X_tr_s2
        Xva = X_va_s2_sc if use_scaled else X_va_s2
        Xte = X_te_s2_sc if use_scaled else X_te_s2

        if name == "XGBoost":
            model.fit(Xtr, y_tr_cong, eval_set=[(Xva, y_va_cong)], verbose=False)
        else:
            model.fit(Xtr, y_tr_cong)

        va_probs = model.predict_proba(Xva)
        te_probs = model.predict_proba(Xte)
        va_m, _ = compute_multiclass_metrics(y_va_cong, np.argmax(va_probs, axis=1))
        te_m, te_cm = compute_multiclass_metrics(y_te_cong, np.argmax(te_probs, axis=1))
        scoreboard_rows.append({"Task": "Congestion (4-Class)", "Model": name, **te_m})

        if va_m["Macro_F1"] > best_cong_val_f1:
            best_cong_val_f1 = va_m["Macro_F1"]
            best_cong_name = name
            best_cong_model = model
            best_cong_val_m = va_m
            best_cong_test_m = te_m
            best_cong_cm = te_cm
            best_cong_scaled = use_scaled
            best_cong_mult = np.ones(4, dtype=np.float64)

        # Validation-calibrated threshold multipliers for XGBoost
        if name == "XGBoost":
            mult = tune_multiclass_multipliers(y_va_cong, va_probs)
            va_cal_m, _ = compute_multiclass_metrics(y_va_cong, np.argmax(va_probs * mult, axis=1))
            te_cal_m, te_cal_cm = compute_multiclass_metrics(
                y_te_cong, np.argmax(te_probs * mult, axis=1)
            )
            cal_label = "XGBoost (Val-Calibrated Multipliers)"
            scoreboard_rows.append({"Task": "Congestion (4-Class)", "Model": cal_label, **te_cal_m})
            if va_cal_m["Macro_F1"] > best_cong_val_f1:
                best_cong_val_f1 = va_cal_m["Macro_F1"]
                best_cong_name = cal_label
                best_cong_model = model
                best_cong_val_m = va_cal_m
                best_cong_test_m = te_cal_m
                best_cong_cm = te_cal_cm
                best_cong_scaled = use_scaled
                best_cong_mult = mult

    joblib.dump(best_cong_model, models_dir / "congestion_best_model.joblib")
    extract_feature_importance(best_cong_model, stage2_features).to_csv(
        models_dir / "congestion_feature_importance.csv", index=False
    )
    cong_labels = ["Free-flow", "Moderate", "Heavy", "Severe"]
    pd.DataFrame(best_cong_cm, index=cong_labels, columns=cong_labels).to_csv(
        models_dir / "congestion_confusion_matrix.csv"
    )

    X_te_c = X_te_s2_sc if best_cong_scaled else X_te_s2
    cong_test_probs = best_cong_model.predict_proba(X_te_c)
    cong_test_preds = np.argmax(cong_test_probs * best_cong_mult, axis=1)
    cong_test_conf = np.max(cong_test_probs, axis=1)

    save_model_card(
        model_name="congestion_classical_winner",
        task_name="Congestion 4-Class Classification",
        algorithm=best_cong_name,
        train_window=train_window,
        test_window=test_window,
        features=stage2_features,
        hyperparameters={
            "winner": best_cong_name,
            "class_multipliers": best_cong_mult.tolist(),
        },
        val_metrics=best_cong_val_m,
        test_metrics=best_cong_test_m,
        selection_rationale=f"Selected {best_cong_name} with Stage-1 V/C stacking based on highest validation Macro-F1 ({best_cong_val_m['Macro_F1']}).",
        models_dir=str(models_dir),
    )

    # =========================================================================
    # TASK 4: ACCIDENT RISK CLASSIFICATION (BINARY, REGULARIZED)
    # =========================================================================
    logger.info("=== Task 4/4: Accident Risk Classification (Binary) ===")
    y_tr_acc = train_df["target_accident_risk"].to_numpy(dtype=int)
    y_va_acc = val_df["target_accident_risk"].to_numpy(dtype=int)
    y_te_acc = test_df["target_accident_risk"].to_numpy(dtype=int)

    # Focused high-signal risk feature subset to prevent overfitting on 0.9% rare positives
    risk_cols = [
        "visibility",
        "rainfall",
        "temperature",
        "is_rain",
        "is_fog",
        "is_low_visibility",
        "weather_clear",
        "weather_cloudy",
        "weather_fog",
        "weather_rain",
        "is_peak",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "public_holiday",
        "event_flag",
        "roadwork_flag",
        "rain_x_peak",
        "low_vis_x_peak",
        "pred_vc_s1",
        "pred_vol_s1",
        "rain_x_pred_vc_s1",
        "fog_x_pred_vc_s1",
        "base_vc_mean",
        "road_hist_acc_rate",
        "inv_vis_x_base_vc",
        "speed_lag_1",
        "occ_lag_1",
        "vc_ratio_lag_1",
        "acc_roll_sum_48",
    ]
    risk_idx = [stage2_features.index(c) for c in risk_cols]
    X_tr_r = X_tr_s2[:, risk_idx]
    X_va_r = X_va_s2[:, risk_idx]
    X_te_r = X_te_s2[:, risk_idx]

    scaler_risk = StandardScaler()
    X_tr_r_sc = scaler_risk.fit_transform(X_tr_r)
    X_va_r_sc = scaler_risk.transform(X_va_r)
    X_te_r_sc = scaler_risk.transform(X_te_r)
    joblib.dump(scaler_risk, models_dir / "feature_scaler_risk.joblib")

    acc_candidates = {
        "Logistic Regression (L2 Regularized)": (
            LogisticRegression(C=0.05, max_iter=1000, random_state=seed),
            True,
        ),
        "SVM Baseline (Calibrated LinearSVC)": (
            CalibratedClassifierCV(
                LinearSVC(C=0.02, tol=1e-3, max_iter=1500, class_weight="balanced", random_state=seed),
                cv=3,
            ),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                max_depth=5, min_samples_leaf=100, class_weight="balanced", random_state=seed
            ),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=120,
                max_depth=8,
                min_samples_leaf=50,
                max_samples=0.5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
            False,
        ),
        "XGBoost (Regularized)": (
            XGBClassifier(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=3,
                min_child_weight=50,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_lambda=10.0,
                reg_alpha=2.0,
                scale_pos_weight=1.0,
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                early_stopping_rounds=30,
                n_jobs=-1,
                random_state=seed,
            ),
            False,
        ),
    }

    best_acc_name, best_acc_model, best_acc_val_auc = "", None, -1.0
    best_acc_val_m, best_acc_test_m, best_acc_roc, best_acc_thresh = {}, {}, None, 0.5
    best_acc_scaled = False

    for name, (model, use_scaled) in acc_candidates.items():
        logger.info(f"Training Accident Risk {name}...")
        Xtr = X_tr_r_sc if use_scaled else X_tr_r
        Xva = X_va_r_sc if use_scaled else X_va_r
        Xte = X_te_r_sc if use_scaled else X_te_r

        if name.startswith("XGBoost"):
            model.fit(Xtr, y_tr_acc, eval_set=[(Xva, y_va_acc)], verbose=False)
        else:
            model.fit(Xtr, y_tr_acc)

        va_prob = model.predict_proba(Xva)[:, 1]
        opt_thresh = find_best_f1_threshold(y_va_acc, va_prob)
        va_m, _ = compute_binary_metrics(y_va_acc, va_prob, threshold=opt_thresh)

        te_prob = model.predict_proba(Xte)[:, 1]
        te_m, te_roc = compute_binary_metrics(y_te_acc, te_prob, threshold=opt_thresh)
        scoreboard_rows.append({"Task": "Accident Risk (Binary)", "Model": name, **te_m})

        if va_m["ROC_AUC"] > best_acc_val_auc:
            best_acc_val_auc = va_m["ROC_AUC"]
            best_acc_name = name
            best_acc_model = model
            best_acc_val_m = va_m
            best_acc_test_m = te_m
            best_acc_roc = te_roc
            best_acc_thresh = opt_thresh
            best_acc_scaled = use_scaled

    joblib.dump(best_acc_model, models_dir / "accident_best_model.joblib")
    extract_feature_importance(best_acc_model, risk_cols).to_csv(
        models_dir / "accident_feature_importance.csv", index=False
    )
    best_acc_roc.to_csv(models_dir / "accident_roc_curve.csv", index=False)

    X_te_a = X_te_r_sc if best_acc_scaled else X_te_r
    acc_test_probs = best_acc_model.predict_proba(X_te_a)[:, 1]

    save_model_card(
        model_name="accident_risk_classical_winner",
        task_name="Accident Risk Binary Classification",
        algorithm=best_acc_name,
        train_window=train_window,
        test_window=test_window,
        features=risk_cols,
        hyperparameters={
            "winner": best_acc_name,
            "operating_threshold": best_acc_thresh,
        },
        val_metrics=best_acc_val_m,
        test_metrics=best_acc_test_m,
        selection_rationale=f"Selected {best_acc_name} on curated high-signal risk features based on highest validation ROC-AUC ({best_acc_val_m['ROC_AUC']}).",
        models_dir=str(models_dir),
    )

    # Save inference contract for dashboard & upload module (FR-17)
    inference_contract = {
        "stage1_features": feature_cols,
        "stage2_features": stage2_features,
        "risk_features": risk_cols,
        "volume_winner": best_vol_name,
        "volume_uses_scaled": best_vol_scaled,
        "volume_q90_halfwidth": round(vol_q90_halfwidth, 4),
        "travel_time_winner": best_tt_name,
        "travel_time_uses_scaled": best_tt_scaled,
        "congestion_winner": best_cong_name,
        "congestion_uses_scaled": best_cong_scaled,
        "congestion_multipliers": best_cong_mult.tolist(),
        "accident_winner": best_acc_name,
        "accident_uses_scaled": best_acc_scaled,
        "accident_threshold": best_acc_thresh,
    }
    with open(models_dir / "inference_contract.json", "w", encoding="utf-8") as f:
        json.dump(inference_contract, f, indent=2)

    # =========================================================================
    # PERSIST SCOREBOARD, TEST PREDICTIONS & ERROR ANALYSIS
    # =========================================================================
    scoreboard_df = pd.DataFrame(scoreboard_rows)
    scoreboard_df.to_csv(models_dir / "classical_scoreboard.csv", index=False)

    idx_to_cong = {0: "Free-flow", 1: "Moderate", 2: "Heavy", 3: "Severe"}
    test_preds_df = pd.DataFrame(
        {
            "road_id": test_df["road_id"].values,
            "road_name": test_df["road_name"].values,
            "latitude": test_df["latitude"].values,
            "longitude": test_df["longitude"].values,
            "timestamp": test_df["timestamp"].values,
            "date": test_df["date"].values,
            "time": test_df["time"].values,
            "hour": test_df["hour"].values,
            "day_of_week": test_df["day_of_week"].values,
            "weather_condition": test_df["weather_condition"].values,
            "temperature": test_df["temperature"].values,
            "rainfall": test_df["rainfall"].values,
            "visibility": test_df["visibility"].values,
            "road_capacity": test_df["road_capacity"].values,
            "actual_volume": y_te_vol,
            "pred_volume": np.round(vol_test_preds, 2),
            "vol_lower_90": np.round(np.clip(vol_test_preds - vol_q90_halfwidth, 0.0, None), 2),
            "vol_upper_90": np.round(vol_test_preds + vol_q90_halfwidth, 2),
            "actual_travel_time": y_te_tt,
            "pred_travel_time": np.round(tt_test_preds, 2),
            "actual_congestion": [idx_to_cong[i] for i in y_te_cong],
            "pred_congestion": [idx_to_cong[i] for i in cong_test_preds],
            "congestion_confidence": np.round(cong_test_conf, 4),
            "actual_accident": y_te_acc,
            "pred_accident_prob": np.round(acc_test_probs, 4),
        }
    )
    test_preds_df.to_parquet(models_dir / "classical_test_predictions.parquet", index=False)

    error_summary = compute_error_analysis(test_preds_df)
    with open(models_dir / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_summary, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 98)
    print(" FLOWCAST M5 CLASSICAL ML BENCHMARK SCOREBOARD (HELD-OUT TEST SET)")
    print("=" * 98)
    print(scoreboard_df.astype(object).fillna("-").to_string(index=False))
    print("=" * 98)
    print(f"Winning Volume Model      : {best_vol_name} -> RMSE={best_vol_test_m['RMSE']} | MAPE={best_vol_test_m['MAPE']}% (PRD <= 12%)")
    print(f"Winning Travel Time Model : {best_tt_name} -> RMSE={best_tt_test_m['RMSE']} | R2={best_tt_test_m['R2']}")
    print(f"Winning Congestion Model  : {best_cong_name} -> Macro-F1={best_cong_test_m['Macro_F1']} (PRD >= 0.80)")
    print(f"Winning Accident Model    : {best_acc_name} -> ROC-AUC={best_acc_test_m['ROC_AUC']} (PRD >= 0.75)")
    print(f"Total M5 Training Time    : {elapsed:.1f}s")
    print("=" * 98 + "\n")

    logger.info("M5 Classical ML Engine completed successfully.")
    return scoreboard_df


if __name__ == "__main__":
    train_classical_models()