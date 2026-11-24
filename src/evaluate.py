"""Shared evaluation metrics, error analysis, and model-card governance utilities."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import ensure_parent_dir


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute RMSE, MAE, safe MAPE (%), and R2 for regression targets."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    nonzero = y_true > 0
    if np.any(nonzero):
        mape = float(
            np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]))
            * 100.0
        )
    else:
        mape = 0.0

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 4),
        "R2": round(r2, 4),
    }


def compute_multiclass_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute Macro-F1, Accuracy, Precision, Recall, and Confusion Matrix."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    metrics = {
        "Macro_F1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Macro_Precision": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "Macro_Recall": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    return metrics, cm


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Select operating probability threshold on validation set maximizing F1."""
    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.linspace(0.05, 0.90, 86):
        preds = (y_prob >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)
    return round(best_thresh, 3)


def compute_binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute ROC-AUC, F1, Precision, Recall, and ROC curve dataframe."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)

    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    metrics = {
        "ROC_AUC": round(auc, 4),
        "F1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "Threshold": round(float(threshold), 3),
    }
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    return metrics, roc_df


def compute_error_analysis(pred_df: pd.DataFrame) -> Dict[str, Any]:
    """Break down volume forecast errors by hour, day of week, and weather (PRD 13.3)."""
    df = pred_df.copy()
    df["abs_error"] = np.abs(df["actual_volume"] - df["pred_volume"])
    df["sq_error"] = (df["actual_volume"] - df["pred_volume"]) ** 2
    df["residual"] = df["actual_volume"] - df["pred_volume"]

    def _agg(group_col: str) -> List[Dict[str, Any]]:
        grouped = (
            df.groupby(group_col, observed=False)
            .agg(
                count=("abs_error", "count"),
                mae=("abs_error", "mean"),
                rmse=("sq_error", lambda s: float(np.sqrt(np.mean(s)))),
                mean_residual=("residual", "mean"),
            )
            .reset_index()
        )
        for col in ["mae", "rmse", "mean_residual"]:
            grouped[col] = grouped[col].round(3)
        return grouped.to_dict(orient="records")

    return {
        "by_hour": _agg("hour"),
        "by_day_of_week": _agg("day_of_week"),
        "by_weather": _agg("weather_condition"),
        "residual_summary": {
            "mean": round(float(df["residual"].mean()), 4),
            "std": round(float(df["residual"].std()), 4),
            "p05": round(float(np.percentile(df["residual"], 5)), 4),
            "p95": round(float(np.percentile(df["residual"], 95)), 4),
        },
    }


def save_model_card(
    model_name: str,
    task_name: str,
    algorithm: str,
    train_window: List[str],
    test_window: List[str],
    features: List[str],
    hyperparameters: Dict[str, Any],
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    selection_rationale: str,
    models_dir: str = "models",
) -> str:
    """Persist structured JSON model card for governance & traceability (PRD 13.4)."""
    card = {
        "model_id": model_name,
        "version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": task_name,
        "algorithm": algorithm,
        "training_window": train_window,
        "test_window": test_window,
        "num_features": len(features),
        "features": features,
        "hyperparameters": hyperparameters,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "selection_rationale": selection_rationale,
    }
    card_path = str(Path(models_dir) / "model_cards" / f"{model_name}_card.json")
    ensure_parent_dir(card_path)
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    return card_path