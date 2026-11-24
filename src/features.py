"""M3: Leakage-Free Feature Engineering, Weekly Lags, Train-Only Baselines & Splits."""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import ensure_parent_dir, load_config, set_global_seed, setup_logger

logger = setup_logger("M3_Features")

CONGESTION_ORDER: Dict[str, int] = {
    "Free-flow": 0,
    "Moderate": 1,
    "Heavy": 2,
    "Severe": 3,
}

MODEL_FEATURES: List[str] = [
    # Segment static & planned attributes
    "road_idx",
    "latitude",
    "longitude",
    "road_capacity",
    "signal_timing",
    # Temporal cyclical & peak indicators
    "hour",
    "minute",
    "time_slot",
    "hour_sin",
    "hour_cos",
    "dow",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_morning_peak",
    "is_evening_peak",
    "is_peak",
    # Calendar & scheduled interactions
    "public_holiday",
    "event_flag",
    "roadwork_flag",
    "holiday_x_peak",
    "event_x_peak",
    "roadwork_x_peak",
    # Weather features & interactions
    "temperature",
    "rainfall",
    "visibility",
    "is_rain",
    "is_fog",
    "is_low_visibility",
    "temp_band_id",
    "weather_clear",
    "weather_cloudy",
    "weather_fog",
    "weather_rain",
    "rain_x_peak",
    "low_vis_x_peak",
    # Strictly lagged sensor telemetry (<= t-1, including 7-day lag_336)
    "vol_lag_1",
    "vol_lag_2",
    "vol_lag_3",
    "vol_lag_4",
    "vol_lag_48",
    "vol_lag_336",
    "speed_lag_1",
    "speed_lag_2",
    "speed_lag_48",
    "speed_lag_336",
    "occ_lag_1",
    "occ_lag_2",
    "occ_lag_48",
    "occ_lag_336",
    "tt_lag_1",
    "tt_lag_2",
    "tt_lag_48",
    "tt_lag_336",
    "share_2w_lag_1",
    "share_car_lag_1",
    "share_lcv_lag_1",
    "share_hcv_lag_1",
    # Strictly shifted rolling statistics & momentum (<= t-1)
    "vol_roll_mean_4",
    "vol_roll_std_4",
    "vol_roll_mean_8",
    "vol_roll_std_8",
    "vol_roll_mean_48",
    "speed_roll_mean_4",
    "speed_roll_std_4",
    "speed_roll_mean_8",
    "speed_roll_std_8",
    "occ_roll_mean_4",
    "occ_roll_mean_8",
    "vol_diff_1",
    "vol_diff_48",
    "vol_diff_336",
    "speed_diff_1",
    # Past incident history (<= t-1)
    "acc_lag_1",
    "acc_roll_sum_8",
    "acc_roll_sum_48",
    # Lagged capacity utilization & headroom (<= t-1)
    "vc_ratio_lag_1",
    "headroom_lag_1",
    "vc_ratio_roll_mean_4",
    # Train-only historical segment x time_slot x weekend baselines (leakage-free)
    "base_vol_mean",
    "base_vol_std",
    "base_tt_mean",
    "base_vc_mean",
    "road_hist_acc_rate",
    "vol_ratio_to_base",
    "rain_x_base_vc",
    "fog_x_base_vc",
    "inv_vis_x_base_vc",
]

TARGET_COLUMNS: Dict[str, str] = {
    "volume": "target_volume",
    "travel_time": "target_travel_time",
    "congestion": "target_congestion_id",
    "accident_risk": "target_accident_risk",
}


def engineer_features(
    clean_df: pd.DataFrame, config: dict
) -> Tuple[pd.DataFrame, Dict, Dict[str, pd.DataFrame]]:
    """Generate temporal, lag (incl. weekly), rolling, capacity, and train-only baseline features."""
    feat_cfg = config["features"]
    split_cfg = config["splitting"]

    df = clean_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

    # 1. Road segment deterministic index (NL-001 -> 0, ..., NL-025 -> 24)
    road_ids_sorted = sorted(df["road_id"].unique().tolist())
    road_to_idx = {rid: idx for idx, rid in enumerate(road_ids_sorted)}
    df["road_idx"] = df["road_id"].map(road_to_idx).astype(int)

    # 2. Temporal & cyclical encodings (PRD Section 9.2)
    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["minute"] = df["timestamp"].dt.minute.astype(int)
    df["time_slot"] = (df["hour"] * 2 + (df["minute"] // 30)).astype(int)

    hour_frac = df["hour"] + df["minute"] / 60.0
    df["hour_sin"] = np.sin(2.0 * np.pi * hour_frac / 24.0).round(6)
    df["hour_cos"] = np.cos(2.0 * np.pi * hour_frac / 24.0).round(6)

    df["dow"] = df["timestamp"].dt.day_of_week.astype(int)
    df["dow_sin"] = np.sin(2.0 * np.pi * df["dow"] / 7.0).round(6)
    df["dow_cos"] = np.cos(2.0 * np.pi * df["dow"] / 7.0).round(6)

    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_morning_peak"] = df["hour"].isin(feat_cfg["peak_morning_hours"]).astype(int)
    df["is_evening_peak"] = df["hour"].isin(feat_cfg["peak_evening_hours"]).astype(int)
    df["is_peak"] = ((df["is_morning_peak"] == 1) | (df["is_evening_peak"] == 1)).astype(int)

    # 3. Weather flags, temperature bands & controlled one-hot encodings
    df["is_rain"] = ((df["weather_condition"] == "Rain") | (df["rainfall"] > 0.0)).astype(int)
    df["is_fog"] = (df["weather_condition"] == "Fog").astype(int)
    df["is_low_visibility"] = (
        df["visibility"] < feat_cfg["low_visibility_threshold"]
    ).astype(int)

    df["temp_band_id"] = np.select(
        [df["temperature"] < 15.0, df["temperature"] < 25.0, df["temperature"] < 32.0],
        [0, 1, 2],
        default=3,
    ).astype(int)

    for cond in ["Clear", "Cloudy", "Fog", "Rain"]:
        df[f"weather_{cond.lower()}"] = (df["weather_condition"] == cond).astype(int)

    # 4. Calendar & weather-peak interaction terms
    df["public_holiday"] = df["public_holiday"].astype(int)
    df["event_flag"] = df["event_flag"].astype(int)
    df["roadwork_flag"] = df["roadwork_flag"].astype(int)

    df["holiday_x_peak"] = (df["public_holiday"] * df["is_peak"]).astype(int)
    df["event_x_peak"] = (df["event_flag"] * df["is_peak"]).astype(int)
    df["roadwork_x_peak"] = (df["roadwork_flag"] * df["is_peak"]).astype(int)
    df["rain_x_peak"] = (df["is_rain"] * df["is_peak"]).astype(int)
    df["low_vis_x_peak"] = (df["is_low_visibility"] * df["is_peak"]).astype(int)

    # 5. Strictly past per-segment lag features (incl. 7-day weekly seasonal lag 336)
    logger.info("Computing per-segment lag features (t-1, t-2, t-3, t-4, t-48, t-336)...")
    grp = df.groupby("road_id", sort=False)

    for lag in [1, 2, 3, 4, 48, 336]:
        df[f"vol_lag_{lag}"] = grp["traffic_volume"].shift(lag)

    for lag in [1, 2, 48, 336]:
        df[f"speed_lag_{lag}"] = grp["avg_speed"].shift(lag)
        df[f"occ_lag_{lag}"] = grp["occupancy"].shift(lag)
        df[f"tt_lag_{lag}"] = grp["travel_time"].shift(lag)

    for share_col in ["share_2w", "share_car", "share_lcv", "share_hcv"]:
        df[f"{share_col}_lag_1"] = grp[share_col].shift(1)

    df["acc_lag_1"] = grp["accident_count"].shift(1)

    # 6. Strictly past rolling statistics (shifted by 1 BEFORE rolling)
    logger.info("Computing strictly past rolling statistics (windows 4, 8, 48)...")
    for w in feat_cfg["rolling_windows"]:
        df[f"vol_roll_mean_{w}"] = grp["traffic_volume"].transform(
            lambda s, win=w: s.shift(1).rolling(win, min_periods=win).mean()
        )
        df[f"vol_roll_std_{w}"] = grp["traffic_volume"].transform(
            lambda s, win=w: s.shift(1).rolling(win, min_periods=win).std()
        )
        df[f"speed_roll_mean_{w}"] = grp["avg_speed"].transform(
            lambda s, win=w: s.shift(1).rolling(win, min_periods=win).mean()
        )
        df[f"speed_roll_std_{w}"] = grp["avg_speed"].transform(
            lambda s, win=w: s.shift(1).rolling(win, min_periods=win).std()
        )
        df[f"occ_roll_mean_{w}"] = grp["occupancy"].transform(
            lambda s, win=w: s.shift(1).rolling(win, min_periods=win).mean()
        )

    df["vol_roll_mean_48"] = grp["traffic_volume"].transform(
        lambda s: s.shift(1).rolling(48, min_periods=48).mean()
    )
    df["acc_roll_sum_8"] = grp["accident_count"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=1).sum()
    )
    df["acc_roll_sum_48"] = grp["accident_count"].transform(
        lambda s: s.shift(1).rolling(48, min_periods=1).sum()
    )

    # Momentum / short-term & seasonal change features (strictly <= t-1)
    df["vol_diff_1"] = df["vol_lag_1"] - df["vol_lag_2"]
    df["vol_diff_48"] = df["vol_lag_1"] - df["vol_lag_48"]
    df["vol_diff_336"] = df["vol_lag_1"] - df["vol_lag_336"]
    df["speed_diff_1"] = df["speed_lag_1"] - df["speed_lag_2"]

    # 7. Capacity utilization & headroom (using lagged volume)
    half_hour_cap = df["road_capacity"] / 2.0
    df["vc_ratio_current"] = (df["traffic_volume"] / half_hour_cap).round(4)
    df["vc_ratio_lag_1"] = (df["vol_lag_1"] / half_hour_cap).round(4)
    df["headroom_lag_1"] = (half_hour_cap - df["vol_lag_1"]).round(2)
    df["vc_ratio_roll_mean_4"] = (df["vol_roll_mean_4"] / half_hour_cap).round(4)

    # 8. Target framing (PRD Section 11.1)
    df["target_volume"] = df["traffic_volume"].astype(float)
    df["target_travel_time"] = df["travel_time"].astype(float)
    df["target_congestion_id"] = df["congestion_level"].map(CONGESTION_ORDER).astype(int)
    df["target_accident_risk"] = (df["accident_count"] > 0).astype(int)

    # Drop warm-up rows where 7-day lag (336) is unavailable
    pre_drop_cols = [c for c in MODEL_FEATURES if not c.startswith(("base_", "road_hist_", "vol_ratio_", "rain_x_base", "fog_x_base", "inv_vis_x_base"))]
    before_drop = len(df)
    df = df.dropna(subset=pre_drop_cols).reset_index(drop=True)
    logger.info(
        f"Dropped {before_drop - len(df):,} warm-up rows (first 7 days/segment). Remaining: {len(df):,}"
    )

    # 9. Chronological time-based split BEFORE computing baseline aggregations (Leakage Guard)
    unique_ts = np.sort(df["timestamp"].unique())
    n_ts = len(unique_ts)
    train_cutoff = unique_ts[int(n_ts * split_cfg["train_ratio"])]
    val_cutoff = unique_ts[int(n_ts * (split_cfg["train_ratio"] + split_cfg["val_ratio"]))]

    df["split"] = np.where(
        df["timestamp"] < train_cutoff,
        "train",
        np.where(df["timestamp"] < val_cutoff, "val", "test"),
    )

    # 10. Train-only segment x time_slot x weekend historical baselines (ZERO leakage)
    logger.info("Computing train-only segment x time_slot x weekend historical baselines...")
    train_only = df[df["split"] == "train"]

    slot_baseline = (
        train_only.groupby(["road_id", "time_slot", "is_weekend"])
        .agg(
            base_vol_mean=("target_volume", "mean"),
            base_vol_std=("target_volume", "std"),
            base_tt_mean=("target_travel_time", "mean"),
            base_vc_mean=("vc_ratio_current", "mean"),
        )
        .reset_index()
    )
    slot_baseline["base_vol_std"] = slot_baseline["base_vol_std"].fillna(0.0)

    # Empirical-Bayes smoothed historical accident rate per road (fit on train only)
    global_acc_rate = float(train_only["target_accident_risk"].mean())
    road_acc_stats = (
        train_only.groupby("road_id")["target_accident_risk"]
        .agg(["count", "sum"])
        .reset_index()
    )
    prior_weight = 500.0
    road_acc_stats["road_hist_acc_rate"] = (
        road_acc_stats["sum"] + prior_weight * global_acc_rate
    ) / (road_acc_stats["count"] + prior_weight)
    road_acc_baseline = road_acc_stats[["road_id", "road_hist_acc_rate"]]

    df = df.merge(slot_baseline, on=["road_id", "time_slot", "is_weekend"], how="left")
    df = df.merge(road_acc_baseline, on="road_id", how="left")

    df["base_vol_mean"] = df["base_vol_mean"].fillna(train_only["target_volume"].mean())
    df["base_vol_std"] = df["base_vol_std"].fillna(train_only["target_volume"].std())
    df["base_tt_mean"] = df["base_tt_mean"].fillna(train_only["target_travel_time"].mean())
    df["base_vc_mean"] = df["base_vc_mean"].fillna(train_only["vc_ratio_current"].mean())
    df["road_hist_acc_rate"] = df["road_hist_acc_rate"].fillna(global_acc_rate)

    # Contextual ratio & weather-congestion risk interactions
    df["vol_ratio_to_base"] = (df["vol_lag_1"] / (df["base_vol_mean"] + 1.0)).round(4)
    df["rain_x_base_vc"] = (df["is_rain"] * df["base_vc_mean"]).round(4)
    df["fog_x_base_vc"] = (df["is_fog"] * df["base_vc_mean"]).round(4)
    df["inv_vis_x_base_vc"] = ((1000.0 / (df["visibility"] + 100.0)) * df["base_vc_mean"]).round(4)

    # Final strict verification assertions
    null_feat_count = int(df[MODEL_FEATURES + list(TARGET_COLUMNS.values())].isna().sum().sum())
    assert null_feat_count == 0, f"Feature matrix contains {null_feat_count} NaNs!"

    max_train_ts = df.loc[df["split"] == "train", "timestamp"].max()
    min_val_ts = df.loc[df["split"] == "val", "timestamp"].min()
    max_val_ts = df.loc[df["split"] == "val", "timestamp"].max()
    min_test_ts = df.loc[df["split"] == "test", "timestamp"].min()
    assert max_train_ts < min_val_ts, "Temporal overlap between train and val!"
    assert max_val_ts < min_test_ts, "Temporal overlap between val and test!"

    split_counts = df["split"].value_counts().to_dict()
    metadata = {
        "total_rows": int(len(df)),
        "num_features": len(MODEL_FEATURES),
        "feature_columns": MODEL_FEATURES,
        "target_columns": TARGET_COLUMNS,
        "congestion_order": CONGESTION_ORDER,
        "road_to_idx": road_to_idx,
        "splits": {
            "train": {
                "rows": int(split_counts.get("train", 0)),
                "start": str(df.loc[df["split"] == "train", "timestamp"].min()),
                "end": str(max_train_ts),
            },
            "val": {
                "rows": int(split_counts.get("val", 0)),
                "start": str(min_val_ts),
                "end": str(max_val_ts),
            },
            "test": {
                "rows": int(split_counts.get("test", 0)),
                "start": str(min_test_ts),
                "end": str(df.loc[df["split"] == "test", "timestamp"].max()),
            },
        },
        "test_accident_positive_rate": round(
            float(df.loc[df["split"] == "test", "target_accident_risk"].mean()), 5
        ),
    }

    baselines = {
        "slot_baseline": slot_baseline,
        "road_acc_baseline": road_acc_baseline,
    }
    logger.info(
        f"Chronological splits -> Train: {metadata['splits']['train']['rows']:,} | "
        f"Val: {metadata['splits']['val']['rows']:,} | Test: {metadata['splits']['test']['rows']:,}"
    )
    return df, metadata, baselines


def run_feature_engineering(config_path: str = "config.yaml") -> pd.DataFrame:
    """Execute full M3 feature engineering pipeline and persist artifacts."""
    config = load_config(config_path)
    set_global_seed(config["project"]["random_seed"])

    in_path = Path(config["paths"]["cleaned_merged"])
    if not in_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset missing at {in_path}. Run python src/clean.py first."
        )

    logger.info(f"Loading cleaned dataset from {in_path}...")
    clean_df = pd.read_parquet(in_path)

    features_df, metadata, baselines = engineer_features(clean_df, config)

    out_path = config["paths"]["processed_features"]
    ensure_parent_dir(out_path)
    features_df.to_parquet(out_path, index=False)
    logger.info(f"Saved processed feature store -> {out_path}")

    meta_path = str(Path(out_path).parent / "feature_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save train-only baseline lookup tables for dashboard & upload inference (FR-17)
    models_dir = Path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    baselines["slot_baseline"].to_parquet(models_dir / "slot_baseline.parquet", index=False)
    baselines["road_acc_baseline"].to_parquet(models_dir / "road_acc_baseline.parquet", index=False)
    logger.info("Saved train-only baseline tables -> models/slot_baseline.parquet & road_acc_baseline.parquet")

    logger.info("M3 Feature Engineering completed successfully.")
    return features_df


if __name__ == "__main__":
    run_feature_engineering()