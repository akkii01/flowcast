"""Automated Pytest Suite for FlowCast Data Quality, Leakage Guard, Math & Artifacts."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml_models import NumPyLinearRegressionGD
from src.utils import load_config


@pytest.fixture(scope="module")
def config():
    return load_config(str(ROOT / "config.yaml"))


@pytest.fixture(scope="module")
def clean_df(config):
    path = ROOT / config["paths"]["cleaned_merged"]
    if not path.exists():
        pytest.skip("Cleaned dataset not generated yet.")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def feat_df(config):
    path = ROOT / config["paths"]["processed_features"]
    if not path.exists():
        pytest.skip("Feature store not generated yet.")
    return pd.read_parquet(path)


def test_raw_source_files_present(config):
    """Verify all three Northline Corridor source CSV files exist."""
    for key in ["raw_calendar", "raw_weather", "raw_traffic"]:
        assert (ROOT / config["paths"][key]).exists(), f"Missing {key}"


def test_cleaned_dataset_integrity(clean_df, config):
    """Verify zero nulls, 25 segments, valid sensor bounds, and harmonised weather."""
    assert clean_df.isna().sum().sum() == 0, "Cleaned dataset contains nulls!"
    assert clean_df["road_id"].nunique() == 25, "Expected 25 road segments!"
    assert (clean_df["traffic_volume"] >= 0).all(), "Negative volume detected!"
    assert (clean_df["avg_speed"] <= config["cleaning"]["max_speed"]).all()
    assert (clean_df["occupancy"] <= config["cleaning"]["max_occupancy"]).all()
    assert set(clean_df["weather_condition"].unique()).issubset(
        {"Clear", "Cloudy", "Fog", "Rain"}
    )


def test_feature_store_no_nulls_and_chronological_splits(feat_df, config):
    """Verify zero feature NaNs and strictly non-overlapping chronological splits."""
    meta_path = (ROOT / config["paths"]["processed_features"]).parent / "feature_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cols = meta["feature_columns"] + list(meta["target_columns"].values())
    assert feat_df[cols].isna().sum().sum() == 0, "NaNs found in feature/target matrix!"

    ts = pd.to_datetime(feat_df["timestamp"], utc=True)
    max_train = ts[feat_df["split"] == "train"].max()
    min_val = ts[feat_df["split"] == "val"].min()
    max_val = ts[feat_df["split"] == "val"].max()
    min_test = ts[feat_df["split"] == "test"].min()

    assert max_train < min_val, "Temporal leakage between train and val!"
    assert max_val < min_test, "Temporal leakage between val and test!"


def test_from_scratch_numpy_gradient_descent():
    """PRD Section 10: Verify from-scratch NumPy GD converges and recovers weights."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, size=(500, 3))
    true_w = np.array([3.5, -2.0, 1.25])
    true_b = 4.0
    y = X @ true_w + true_b + rng.normal(0, 0.05, size=500)

    model = NumPyLinearRegressionGD(learning_rate=0.05, epochs=300, l2_reg=0.0)
    model.fit(X, y)

    assert model.loss_history[-1] < model.loss_history[0] * 0.01
    assert np.allclose(model.weights, true_w, atol=0.1)
    assert abs(model.bias - true_b) < 0.1


def test_persisted_models_cards_and_reports(config):
    """Verify trained models, model cards, benchmark CSV, notebooks, and reports exist."""
    models_dir = ROOT / config["paths"]["models_dir"]
    required_files = [
        models_dir / "volume_best_model.joblib",
        models_dir / "travel_time_best_model.joblib",
        models_dir / "congestion_best_model.joblib",
        models_dir / "accident_best_model.joblib",
        models_dir / "lstm_sequence_model.pt",
        models_dir / "bilstm_sequence_model.pt",
        models_dir / "classical_scoreboard.csv",
        models_dir / "classical_vs_dl_benchmark.csv",
        models_dir / "model_cards" / "volume_classical_winner_card.json",
        models_dir / "model_cards" / "lstm_sequence_model_card.json",
        ROOT / config["paths"]["data_quality_report"],
        ROOT / "reports" / "final_technical_report.md",
        ROOT / "notebooks" / "01_eda.ipynb",
        ROOT / "notebooks" / "02_cleaning.ipynb",
        ROOT / "notebooks" / "03_modelling.ipynb",
    ]
    for fpath in required_files:
        assert fpath.exists(), f"Missing required artefact: {fpath}"