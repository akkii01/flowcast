"""M2: Deterministic Data Cleaning, Temporal Alignment, Merge & Quality Report."""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ingest import load_raw_tables, run_ingestion
from src.utils import ensure_parent_dir, load_config, set_global_seed, setup_logger

logger = setup_logger("M2_Clean")


def clean_weather(weather_df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, Dict]:
    """Harmonise weather labels, parse DD/MM/YYYY dates, and impute gaps."""
    rules = config["cleaning"]
    df = weather_df.copy()

    # Harmonise weather_condition labels
    df["weather_condition"] = (
        df["weather_condition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(rules["weather_mapping"])
    )
    if df["weather_condition"].isna().any():
        mode_val = df["weather_condition"].mode().iloc[0]
        df["weather_condition"] = df["weather_condition"].fillna(mode_val)

    # Parse DD/MM/YYYY + HH:MM into UTC timestamp
    df["timestamp_hour"] = pd.to_datetime(
        df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
        format="%d/%m/%Y %H:%M",
        utc=True,
    )
    df = df.sort_values(["station_id", "timestamp_hour"]).reset_index(drop=True)

    # Impute missing temperature/visibility/rainfall per station
    imputed_counts = {}
    for col in ["temperature", "visibility", "rainfall"]:
        before_nulls = int(df[col].isna().sum())
        df[col] = df.groupby("station_id")[col].transform(
            lambda s: s.interpolate(
                method="linear", limit=rules["weather_interpolation_limit"]
            )
        )
        station_hour_med = df.groupby(["station_id", "time"])[col].transform("median")
        df[col] = df[col].fillna(station_hour_med).fillna(df[col].median())
        imputed_counts[col] = before_nulls

    df["temperature"] = df["temperature"].round(2)
    df["rainfall"] = df["rainfall"].clip(lower=0.0).round(2)
    df["visibility"] = df["visibility"].clip(lower=0.0).round(1)

    stats = {
        "weather_clean_rows": int(len(df)),
        "weather_standardized_labels": sorted(df["weather_condition"].unique().tolist()),
        "weather_imputed": imputed_counts,
    }
    logger.info(
        f"Weather cleaned: {len(df):,} hourly rows | Labels: {stats['weather_standardized_labels']}"
    )
    return df, stats


def parse_vehicle_dist(series: pd.Series) -> pd.DataFrame:
    """Safely parse JSON vehicle_type_dist column into four numeric columns."""
    def _safe_parse(val):
        if pd.isna(val):
            return {"2W": np.nan, "Car": np.nan, "LCV": np.nan, "HCV": np.nan}
        if isinstance(val, dict):
            return val
        try:
            return json.loads(str(val))
        except Exception:
            return {"2W": np.nan, "Car": np.nan, "LCV": np.nan, "HCV": np.nan}

    parsed = pd.json_normalize(series.apply(_safe_parse))
    return pd.DataFrame(
        {
            "share_2w": pd.to_numeric(parsed.get("2W"), errors="coerce").values,
            "share_car": pd.to_numeric(parsed.get("Car"), errors="coerce").values,
            "share_lcv": pd.to_numeric(parsed.get("LCV"), errors="coerce").values,
            "share_hcv": pd.to_numeric(parsed.get("HCV"), errors="coerce").values,
        },
        index=series.index,
    )


def clean_and_merge_all(
    calendar_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    traffic_df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, Dict]:
    """Run full PRD Section 9.1 ordered cleaning and merging pipeline."""
    rules = config["cleaning"]
    stats: Dict = {}

    # Clean weather first
    weather_clean, weather_stats = clean_weather(weather_df, config)
    stats.update(weather_stats)

    # Clean calendar
    cal = calendar_df.copy()
    cal["holiday_name"] = cal["holiday_name"].fillna("None").astype(str)
    cal["event_name"] = cal["event_name"].fillna("None").astype(str)

    # 1. Traffic timestamps
    trf = traffic_df.copy()
    trf["timestamp"] = pd.to_datetime(
        trf["date"].astype(str).str.strip() + " " + trf["time"].astype(str).str.strip(),
        format="%Y-%m-%d %H:%M",
        utc=True,
    )

    # 2. Deduplicate on (road_id, timestamp), keeping most complete record
    before_dedup = len(trf)
    trf["_null_count"] = trf.isna().sum(axis=1)
    trf = (
        trf.sort_values(["road_id", "timestamp", "_null_count"])
        .drop_duplicates(subset=["road_id", "timestamp"], keep="first")
        .drop(columns=["_null_count"])
        .reset_index(drop=True)
    )
    stats["duplicates_removed"] = int(before_dedup - len(trf))
    logger.info(f"Removed {stats['duplicates_removed']:,} duplicate traffic rows.")

    # 3. Rule-based impossible outlier detection -> convert to NaN for imputation
    neg_vol = trf["traffic_volume"] < rules["min_volume"]
    bad_speed = (trf["avg_speed"] > rules["max_speed"]) | (trf["avg_speed"] <= 0)
    bad_occ = (trf["occupancy"] > rules["max_occupancy"]) | (
        trf["occupancy"] < rules["min_occupancy"]
    )
    stats["rule_outliers_repaired"] = {
        "negative_volume": int(neg_vol.sum()),
        "impossible_speed": int(bad_speed.sum()),
        "impossible_occupancy": int(bad_occ.sum()),
    }
    trf.loc[neg_vol, "traffic_volume"] = np.nan
    trf.loc[bad_speed, "avg_speed"] = np.nan
    trf.loc[bad_occ, "occupancy"] = np.nan

    # Statistical z-score extreme audit (logged for data-quality transparency)
    z_thresh = rules["zscore_threshold"]
    z_flags = {}
    for col in ["traffic_volume", "avg_speed", "occupancy"]:
        col_mean = trf[col].mean()
        col_std = trf[col].std()
        z_scores = (trf[col] - col_mean).abs() / (col_std + 1e-8)
        z_flags[col] = int((z_scores > z_thresh).sum())
    stats["statistical_zscore_extremes_gt4"] = z_flags

    # Parse nested JSON vehicle_type_dist
    vdist_df = parse_vehicle_dist(trf["vehicle_type_dist"])
    trf = pd.concat([trf.drop(columns=["vehicle_type_dist"]), vdist_df], axis=1)

    # 4. Reindex onto complete 25 roads x 30-min grid to expose sensor dropouts
    road_meta = (
        trf.groupby("road_id")[
            ["road_name", "latitude", "longitude", "weather_station_id", "road_capacity"]
        ]
        .first()
        .reset_index()
    )
    full_timestamps = pd.date_range(
        trf["timestamp"].min(), trf["timestamp"].max(), freq="30min", tz="UTC"
    )
    all_roads = sorted(trf["road_id"].unique().tolist())
    full_grid = pd.MultiIndex.from_product(
        [all_roads, full_timestamps], names=["road_id", "timestamp"]
    ).to_frame(index=False)

    stats["missing_sensor_windows_reconstructed"] = int(len(full_grid) - len(trf))
    logger.info(
        f"Reconstructed {stats['missing_sensor_windows_reconstructed']:,} dropped 30-min sensor windows."
    )

    trf_dynamic = trf.drop(
        columns=["road_name", "latitude", "longitude", "weather_station_id", "road_capacity"]
    )
    merged = full_grid.merge(trf_dynamic, on=["road_id", "timestamp"], how="left")
    merged = merged.merge(road_meta, on="road_id", how="left")
    merged["date"] = merged["timestamp"].dt.strftime("%Y-%m-%d")
    merged["time"] = merged["timestamp"].dt.strftime("%H:%M")
    merged = merged.sort_values(["road_id", "timestamp"]).reset_index(drop=True)

    # 5. Missing-value handling: short gaps via time interpolation, longer gaps via (road, time) median
    interp_cols = [
        "traffic_volume",
        "avg_speed",
        "occupancy",
        "travel_time",
        "signal_timing",
        "share_2w",
        "share_car",
        "share_lcv",
        "share_hcv",
    ]
    imputed_traffic = {}
    for col in interp_cols:
        null_before = int(merged[col].isna().sum())
        merged[col] = merged.groupby("road_id")[col].transform(
            lambda s: s.interpolate(method="linear", limit=rules["interpolation_limit"])
        )
        road_time_med = merged.groupby(["road_id", "time"])[col].transform("median")
        road_med = merged.groupby("road_id")[col].transform("median")
        merged[col] = merged[col].fillna(road_time_med).fillna(road_med)
        imputed_traffic[col] = null_before
    stats["traffic_values_imputed"] = imputed_traffic

    # Clean numeric bounds & dtypes
    merged["traffic_volume"] = merged["traffic_volume"].clip(lower=0).round().astype(int)
    merged["vehicle_count"] = (
        merged["vehicle_count"].fillna(merged["traffic_volume"]).clip(lower=0).round().astype(int)
    )
    merged["avg_speed"] = merged["avg_speed"].clip(
        lower=rules["min_speed"], upper=rules["max_speed"]
    ).round(2)
    merged["occupancy"] = merged["occupancy"].clip(
        lower=rules["min_occupancy"], upper=rules["max_occupancy"]
    ).round(2)
    merged["travel_time"] = merged["travel_time"].clip(lower=0.1).round(2)
    merged["signal_timing"] = merged["signal_timing"].clip(lower=1).round().astype(int)
    merged["accident_count"] = merged["accident_count"].fillna(0).clip(lower=0).astype(int)

    # Normalize vehicle distribution shares to sum cleanly to 1.0
    share_cols = ["share_2w", "share_car", "share_lcv", "share_hcv"]
    share_sum = merged[share_cols].sum(axis=1).replace(0, 1.0)
    for col in share_cols:
        merged[col] = (merged[col] / share_sum).round(4)

    # Derive missing congestion_level using V/C = traffic_volume / (road_capacity / 2)
    blank_cong = int(merged["congestion_level"].isna().sum())
    stats["congestion_derived_rows"] = blank_cong
    vc_ratio = merged["traffic_volume"] / (merged["road_capacity"] / 2.0)
    c_rules = rules["congestion_thresholds"]
    derived_cong = np.select(
        [
            vc_ratio < c_rules["free_flow_max"],
            vc_ratio < c_rules["moderate_max"],
            vc_ratio < c_rules["heavy_max"],
        ],
        ["Free-flow", "Moderate", "Heavy"],
        default="Severe",
    )
    merged["congestion_level"] = merged["congestion_level"].fillna(
        pd.Series(derived_cong, index=merged.index)
    )

    # 6. Temporal alignment & merge (hourly weather broadcast to both :00 and :30)
    merged["timestamp_hour"] = merged["timestamp"].dt.floor("h")
    merged = merged.merge(
        weather_clean[
            ["station_id", "timestamp_hour", "weather_condition", "temperature", "rainfall", "visibility"]
        ],
        left_on=["weather_station_id", "timestamp_hour"],
        right_on=["station_id", "timestamp_hour"],
        how="left",
    ).drop(columns=["station_id", "timestamp_hour", "weather_station_id"])

    # Merge calendar events on date
    merged = merged.merge(cal, on="date", how="left")
    merged["day_of_week"] = merged["timestamp"].dt.day_name()

    # 7. Final strict quality assertions
    total_nulls = int(merged.isna().sum().sum())
    assert total_nulls == 0, f"Cleaning failed: {total_nulls} nulls remain!"
    assert (merged["traffic_volume"] >= 0).all(), "Negative traffic_volume remains!"
    assert (merged["avg_speed"] <= rules["max_speed"]).all(), "Speed outlier remains!"
    assert (merged["occupancy"] <= rules["max_occupancy"]).all(), "Occupancy outlier remains!"

    stats["final_clean_rows"] = int(len(merged))
    stats["final_columns"] = int(len(merged.columns))
    stats["congestion_distribution"] = (
        merged["congestion_level"].value_counts().to_dict()
    )
    stats["accident_positive_windows"] = int((merged["accident_count"] > 0).sum())

    logger.info(
        f"Clean & merge complete: {len(merged):,} rows x {len(merged.columns)} cols | 0 nulls verified."
    )
    return merged, stats


def write_data_quality_report(audit: dict, clean_stats: dict, report_path: str) -> None:
    """Write PRD Section 7/9/18 data-quality report documenting all defects and resolutions."""
    ensure_parent_dir(report_path)
    cong = clean_stats["congestion_distribution"]
    lines = [
        "# FlowCast — Data Quality & Cleaning Audit Report",
        "",
        "**Corridor:** Northline Corridor (25 Arterial Road Segments)  ",
        "**Coverage:** 2025-01-01 to 2025-05-31 (30-minute aggregation)  ",
        "**Pipeline Version:** v1.0",
        "",
        "---",
        "",
        "## 1. Raw Source Inventory",
        "",
        "| Source Table | Raw Rows | Key Grain | Primary Defects Observed |",
        "| --- | ---: | --- | --- |",
        f"| `calendar_events.csv` | {audit['calendar_rows']:,} | `date` | Blank names on non-holiday/event days (expected) |",
        f"| `weather_observations.csv` | {audit['weather_rows']:,} | `station_id × hour` | Mixed `DD/MM/YYYY` dates, {len(audit['weather_raw_labels'])} unstandardised condition labels, {audit['weather_missing_temperature']:,} null temps, {audit['weather_missing_visibility']:,} null visibility |",
        f"| `traffic_sensor_log.csv` | {audit['traffic_raw_rows']:,} | `road_id × 30-min` | {audit['traffic_duplicates']:,} duplicates, {clean_stats['missing_sensor_windows_reconstructed']:,} missing windows, rule outliers, {audit['traffic_null_congestion']:,} blank congestion labels, nested JSON field |",
        "",
        "---",
        "",
        "## 2. Defects Detected, Quarantined & Resolved",
        "",
        f"- **Quarantined Defect Audit Log:** `{audit['quarantined_rows_logged']:,}` defective rows logged to `data/interim/quarantine_traffic_defects.csv` (zero silent drops).",
        f"- **Key-Level Deduplication:** Removed `{clean_stats['duplicates_removed']:,}` duplicate `(road_id, timestamp)` records, retaining the most complete reading.",
        f"- **Physically Impossible Outliers Repaired:**",
        f"  - Negative `traffic_volume` (`< 0`): `{clean_stats['rule_outliers_repaired']['negative_volume']:,}` flagged & imputed.",
        f"  - Impossible `avg_speed` (`> 200 km/h` or `<= 0`): `{clean_stats['rule_outliers_repaired']['impossible_speed']:,}` flagged & imputed.",
        f"  - Impossible `occupancy` (`> 100%` or `< 0%`): `{clean_stats['rule_outliers_repaired']['impossible_occupancy']:,}` flagged & imputed.",
        f"- **Statistical Z-Score Audit (`|z| > 4` after rule filter):** `{clean_stats['statistical_zscore_extremes_gt4']}` (retained as valid extreme congestion/peak tails).",
        f"- **Sensor Dropout Reconstruction:** Reindexed to full `25 segments × 7,248 windows = 181,200` grid, reconstructing `{clean_stats['missing_sensor_windows_reconstructed']:,}` dropped windows.",
        f"- **Imputation Strategy:** Linear time interpolation for short gaps (`<= 4` windows / 2 hours), followed by segment-and-time-of-day median fallback.",
        f"  - Imputed counts: `traffic_volume={clean_stats['traffic_values_imputed']['traffic_volume']:,}`, `avg_speed={clean_stats['traffic_values_imputed']['avg_speed']:,}`, `occupancy={clean_stats['traffic_values_imputed']['occupancy']:,}`.",
        f"- **Congestion Level Derivation:** Derived `{clean_stats['congestion_derived_rows']:,}` missing labels using `V/C = traffic_volume / (road_capacity / 2)` (`<0.50` Free-flow, `0.50–0.79` Moderate, `0.80–0.99` Heavy, `>=1.00` Severe).",
        f"- **Weather Harmonisation & Temporal Alignment:** Standardised labels to `{clean_stats['weather_standardized_labels']}`, parsed `DD/MM/YYYY` to UTC, and broadcast hourly weather to both `:00` and `:30` windows.",
        "- **Nested JSON Parsing:** Parsed `vehicle_type_dist` into `share_2w`, `share_car`, `share_lcv`, `share_hcv` (normalised to sum to `1.0`).",
        "",
        "---",
        "",
        "## 3. Final Analysis-Ready Dataset Verification",
        "",
        f"- **Final Row Count:** `{clean_stats['final_clean_rows']:,}` rows × `{clean_stats['final_columns']}` columns",
        "- **Missing Values Remaining:** `0` (verified via strict assertion)",
        f"- **Congestion Distribution:** Free-flow: `{cong.get('Free-flow', 0):,}` | Moderate: `{cong.get('Moderate', 0):,}` | Heavy: `{cong.get('Heavy', 0):,}` | Severe: `{cong.get('Severe', 0):,}`",
        f"- **Accident-Positive Windows (`accident_count > 0`):** `{clean_stats['accident_positive_windows']:,}` (`{clean_stats['accident_positive_windows'] / clean_stats['final_clean_rows'] * 100:.2f}%` — highly imbalanced binary target)",
        "",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Generated data-quality report -> {report_path}")


def run_cleaning(config_path: str = "config.yaml") -> pd.DataFrame:
    """Execute full M2 cleaning pipeline and generate quality report."""
    config = load_config(config_path)
    set_global_seed(config["project"]["random_seed"])

    audit_path = Path(config["paths"]["ingestion_audit"])
    if not audit_path.exists():
        audit = run_ingestion(config_path)
    else:
        with open(audit_path, "r", encoding="utf-8") as f:
            audit = json.load(f)

    calendar_df, weather_df, traffic_df = load_raw_tables(config)
    cleaned_df, clean_stats = clean_and_merge_all(
        calendar_df, weather_df, traffic_df, config
    )

    out_path = config["paths"]["cleaned_merged"]
    ensure_parent_dir(out_path)
    cleaned_df.to_parquet(out_path, index=False)
    logger.info(f"Saved cleaned merged dataset -> {out_path}")

    write_data_quality_report(
        audit, clean_stats, config["paths"]["data_quality_report"]
    )
    logger.info("M2 Cleaning & Wrangling completed successfully.")
    return cleaned_df


if __name__ == "__main__":
    run_cleaning()