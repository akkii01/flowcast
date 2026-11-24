# FlowCast — Data Quality & Cleaning Audit Report

**Corridor:** Northline Corridor (25 Arterial Road Segments)  
**Coverage:** 2025-01-01 to 2025-05-31 (30-minute aggregation)  
**Pipeline Version:** v1.0

---

## 1. Raw Source Inventory

| Source Table | Raw Rows | Key Grain | Primary Defects Observed |
| --- | ---: | --- | --- |
| `calendar_events.csv` | 151 | `date` | Blank names on non-holiday/event days (expected) |
| `weather_observations.csv` | 10,872 | `station_id × hour` | Mixed `DD/MM/YYYY` dates, 13 unstandardised condition labels, 167 null temps, 111 null visibility |
| `traffic_sensor_log.csv` | 178,468 | `road_id × 30-min` | 1,767 duplicates, 4,499 missing windows, rule outliers, 26,883 blank congestion labels, nested JSON field |

---

## 2. Defects Detected, Quarantined & Resolved

- **Quarantined Defect Audit Log:** `2,479` defective rows logged to `data/interim/quarantine_traffic_defects.csv` (zero silent drops).
- **Key-Level Deduplication:** Removed `1,767` duplicate `(road_id, timestamp)` records, retaining the most complete reading.
- **Physically Impossible Outliers Repaired:**
  - Negative `traffic_volume` (`< 0`): `239` flagged & imputed.
  - Impossible `avg_speed` (`> 200 km/h` or `<= 0`): `236` flagged & imputed.
  - Impossible `occupancy` (`> 100%` or `< 0%`): `229` flagged & imputed.
- **Statistical Z-Score Audit (`|z| > 4` after rule filter):** `{'traffic_volume': 121, 'avg_speed': 0, 'occupancy': 0}` (retained as valid extreme congestion/peak tails).
- **Sensor Dropout Reconstruction:** Reindexed to full `25 segments × 7,248 windows = 181,200` grid, reconstructing `4,499` dropped windows.
- **Imputation Strategy:** Linear time interpolation for short gaps (`<= 4` windows / 2 hours), followed by segment-and-time-of-day median fallback.
  - Imputed counts: `traffic_volume=9,086`, `avg_speed=9,078`, `occupancy=9,072`.
- **Congestion Level Derivation:** Derived `31,123` missing labels using `V/C = traffic_volume / (road_capacity / 2)` (`<0.50` Free-flow, `0.50–0.79` Moderate, `0.80–0.99` Heavy, `>=1.00` Severe).
- **Weather Harmonisation & Temporal Alignment:** Standardised labels to `['Clear', 'Cloudy', 'Fog', 'Rain']`, parsed `DD/MM/YYYY` to UTC, and broadcast hourly weather to both `:00` and `:30` windows.
- **Nested JSON Parsing:** Parsed `vehicle_type_dist` into `share_2w`, `share_car`, `share_lcv`, `share_hcv` (normalised to sum to `1.0`).

---

## 3. Final Analysis-Ready Dataset Verification

- **Final Row Count:** `181,200` rows × `30` columns
- **Missing Values Remaining:** `0` (verified via strict assertion)
- **Congestion Distribution:** Free-flow: `111,263` | Moderate: `43,234` | Heavy: `16,780` | Severe: `9,923`
- **Accident-Positive Windows (`accident_count > 0`):** `1,652` (`0.91%` — highly imbalanced binary target)
