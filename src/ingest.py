"""M1: Raw Data Ingestion, Schema Validation, and Defect Quarantine."""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import ensure_parent_dir, load_config, setup_logger

logger = setup_logger("M1_Ingest")

EXPECTED_SCHEMAS: Dict[str, List[str]] = {
    "calendar_events": [
        "date",
        "public_holiday",
        "holiday_name",
        "event_flag",
        "event_name",
        "roadwork_flag",
    ],
    "weather_observations": [
        "station_id",
        "date",
        "time",
        "weather_condition",
        "temperature",
        "rainfall",
        "visibility",
    ],
    "traffic_sensor_log": [
        "road_id",
        "road_name",
        "latitude",
        "longitude",
        "weather_station_id",
        "date",
        "time",
        "traffic_volume",
        "vehicle_count",
        "vehicle_type_dist",
        "avg_speed",
        "occupancy",
        "congestion_level",
        "travel_time",
        "accident_count",
        "signal_timing",
        "road_capacity",
    ],
}


def validate_schema(df: pd.DataFrame, table_name: str) -> None:
    """Verify all expected columns exist in the loaded dataframe."""
    expected = EXPECTED_SCHEMAS[table_name]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"[{table_name}] Missing required columns: {missing}")
    logger.info(f"[{table_name}] Schema validated: {len(df):,} rows, {len(df.columns)} cols")


def load_raw_tables(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three linked raw CSV files in recommended dictionary order."""
    paths = config["paths"]
    for key in ["raw_calendar", "raw_weather", "raw_traffic"]:
        if not Path(paths[key]).exists():
            raise FileNotFoundError(f"Missing raw input file: {paths[key]}")

    calendar_df = pd.read_csv(paths["raw_calendar"])
    validate_schema(calendar_df, "calendar_events")

    weather_df = pd.read_csv(paths["raw_weather"])
    validate_schema(weather_df, "weather_observations")

    traffic_df = pd.read_csv(paths["raw_traffic"])
    validate_schema(traffic_df, "traffic_sensor_log")

    return calendar_df, weather_df, traffic_df


def audit_and_quarantine(
    calendar_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    traffic_df: pd.DataFrame,
    config: dict,
) -> dict:
    """Detect schema/range/duplicate defects and write quarantine audit log."""
    rules = config["cleaning"]
    quarantine_records = []

    # 1. Traffic key duplicates
    dup_mask = traffic_df.duplicated(subset=["road_id", "date", "time"], keep="first")
    if dup_mask.any():
        q = traffic_df.loc[dup_mask].copy()
        q["defect_reason"] = "duplicate_key_road_date_time"
        quarantine_records.append(q)

    # 2. Impossible numeric outliers
    neg_vol_mask = traffic_df["traffic_volume"] < rules["min_volume"]
    if neg_vol_mask.any():
        q = traffic_df.loc[neg_vol_mask].copy()
        q["defect_reason"] = "negative_traffic_volume"
        quarantine_records.append(q)

    bad_speed_mask = (traffic_df["avg_speed"] > rules["max_speed"]) | (
        traffic_df["avg_speed"] <= 0
    )
    if bad_speed_mask.any():
        q = traffic_df.loc[bad_speed_mask].copy()
        q["defect_reason"] = "impossible_avg_speed"
        quarantine_records.append(q)

    bad_occ_mask = (traffic_df["occupancy"] > rules["max_occupancy"]) | (
        traffic_df["occupancy"] < rules["min_occupancy"]
    )
    if bad_occ_mask.any():
        q = traffic_df.loc[bad_occ_mask].copy()
        q["defect_reason"] = "impossible_occupancy"
        quarantine_records.append(q)

    # Persist quarantined defective rows (0 silent drops)
    quarantine_path = config["paths"]["quarantine_traffic"]
    ensure_parent_dir(quarantine_path)
    if quarantine_records:
        quarantine_df = pd.concat(quarantine_records, ignore_index=True)
        quarantine_df.to_csv(quarantine_path, index=False)
        logger.info(
            f"Quarantined {len(quarantine_df):,} defective traffic records -> {quarantine_path}"
        )
    else:
        pd.DataFrame(columns=list(traffic_df.columns) + ["defect_reason"]).to_csv(
            quarantine_path, index=False
        )

    # Weather inconsistent labels check
    raw_weather_labels = sorted(
        weather_df["weather_condition"].dropna().astype(str).unique().tolist()
    )

    audit = {
        "calendar_rows": int(len(calendar_df)),
        "calendar_date_range": [str(calendar_df["date"].min()), str(calendar_df["date"].max())],
        "weather_rows": int(len(weather_df)),
        "weather_stations": sorted(weather_df["station_id"].unique().tolist()),
        "weather_missing_temperature": int(weather_df["temperature"].isna().sum()),
        "weather_missing_visibility": int(weather_df["visibility"].isna().sum()),
        "weather_raw_labels": raw_weather_labels,
        "traffic_raw_rows": int(len(traffic_df)),
        "traffic_unique_roads": int(traffic_df["road_id"].nunique()),
        "traffic_duplicates": int(dup_mask.sum()),
        "traffic_null_volume": int(traffic_df["traffic_volume"].isna().sum()),
        "traffic_null_speed": int(traffic_df["avg_speed"].isna().sum()),
        "traffic_null_occupancy": int(traffic_df["occupancy"].isna().sum()),
        "traffic_null_congestion": int(traffic_df["congestion_level"].isna().sum()),
        "traffic_negative_volume": int(neg_vol_mask.sum()),
        "traffic_speed_outliers": int(bad_speed_mask.sum()),
        "traffic_occupancy_outliers": int(bad_occ_mask.sum()),
        "quarantined_rows_logged": int(
            sum(len(df) for df in quarantine_records)
        ),
    }

    audit_path = config["paths"]["ingestion_audit"]
    ensure_parent_dir(audit_path)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    logger.info(f"Saved ingestion audit summary -> {audit_path}")
    return audit


def run_ingestion(config_path: str = "config.yaml") -> dict:
    """Execute full M1 ingestion and validation stage."""
    config = load_config(config_path)
    calendar_df, weather_df, traffic_df = load_raw_tables(config)
    audit = audit_and_quarantine(calendar_df, weather_df, traffic_df, config)
    logger.info("M1 Ingestion & Validation completed successfully.")
    return audit


if __name__ == "__main__":
    run_ingestion()