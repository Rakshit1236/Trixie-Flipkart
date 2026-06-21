"""
Data pipeline for Trixie-flipkartgridlock.
Load, clean, preprocess with cyclical encoding and spatial grid features.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CSV_FILE, INDIA_LAT_MIN, INDIA_LAT_MAX, INDIA_LON_MIN, INDIA_LON_MAX,
    ROAD_PATTERNS, SEVERITY_WEIGHTS, RUSH_HOURS
)
from src.utils import (
    classify_road_type, extract_road_name, extract_area,
    severity_from_violation_types, is_rush_hour, get_time_bin,
    cyclical_encode, grid_cell_id
)


def load_raw() -> pd.DataFrame:
    """Load raw CSV data."""
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE, low_memory=False)
    print(f"  Raw records: {len(df):,}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and filter raw data."""
    print("Cleaning data...")
    
    # Rename columns to standard names
    col_mapping = {
        "created_datetime": "violation_start_datetime",
        "vehicle_number": "vehicle_number_masked",
        "vehicle_type": "type_of_vehicle",
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    # Filter approved only
    if "validation_status" in df.columns:
        df = df[df["validation_status"].str.lower() == "approved"]
        print(f"  After approved filter: {len(df):,}")
    
    # Drop nulls
    key_cols = ["latitude", "longitude", "violation_start_datetime"]
    existing_cols = [c for c in key_cols if c in df.columns]
    df = df.dropna(subset=existing_cols)
    print(f"  After null removal: {len(df):,}")
    
    # Parse datetime
    if "violation_start_datetime" in df.columns:
        df["violation_start_datetime"] = pd.to_datetime(
            df["violation_start_datetime"], errors="coerce"
        )
    if "closed_datetime" in df.columns:
        df["closed_datetime"] = pd.to_datetime(df["closed_datetime"], errors="coerce")
    
    # Compute duration
    df["violation_duration_minutes"] = df.apply(_safe_duration, axis=1)
    
    # Geographic bounds (India)
    df = df[
        (df["latitude"] >= INDIA_LAT_MIN) & (df["latitude"] <= INDIA_LAT_MAX) &
        (df["longitude"] >= INDIA_LON_MIN) & (df["longitude"] <= INDIA_LON_MAX)
    ]
    print(f"  After geo bounds: {len(df):,}")
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"  Final cleaned records: {len(df):,}")
    return df


def _safe_duration(row) -> float:
    """Compute violation duration in minutes."""
    try:
        if pd.notna(row.get("closed_datetime")) and pd.notna(row.get("violation_start_datetime")):
            delta = (row["closed_datetime"] - row["violation_start_datetime"])
            minutes = delta.total_seconds() / 60.0
            if minutes > 0:
                return max(minutes, 5.0)
    except Exception:
        pass
    return 30.0


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features including cyclical encoding and spatial grid."""
    print("Preprocessing with feature engineering...")
    
    df = df.copy()
    
    # ==================== TEMPORAL FEATURES ====================
    dt = df["violation_start_datetime"]
    
    df["hour"] = dt.dt.hour
    df["minute"] = dt.dt.minute
    df["day_of_week"] = dt.dt.dayofweek  # 0=Mon, 6=Sun
    df["day_name"] = dt.dt.day_name()
    df["month"] = dt.dt.month
    df["date"] = dt.dt.date
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].apply(lambda h: is_rush_hour(h, RUSH_HOURS)).astype(int)
    df["time_bin"] = df["hour"].apply(get_time_bin)
    
    # Cyclical encoding (NEW)
    df["hour_sin"], df["hour_cos"] = cyclical_encode(df["hour"].values, 24)
    df["dow_sin"], df["dow_cos"] = cyclical_encode(df["day_of_week"].values, 7)
    
    # ==================== SEVERITY FEATURES ====================
    if "violation_type" in df.columns:
        severity_result = df["violation_type"].apply(
            lambda x: severity_from_violation_types(x)
        )
        df["severity_weight"] = severity_result.apply(lambda x: x[0])
        df["dominant_violation"] = severity_result.apply(lambda x: x[1])
    else:
        df["severity_weight"] = 0.7
        df["dominant_violation"] = "default"
    
    # ==================== ROAD FEATURES ====================
    if "location" in df.columns:
        df["road_type"] = df["location"].apply(classify_road_type)
        df["road_name"] = df["location"].apply(extract_road_name)
        df["area"] = df["location"].apply(extract_area)
    else:
        df["road_type"] = "Other"
        df["road_name"] = "Unknown"
        df["area"] = "Unknown"
    
    # Number of lanes estimate
    lane_estimate = {"Ring": 4, "Main": 3, "Underpass": 2, "Cross": 2, "Other": 2}
    df["num_lanes"] = df["road_type"].map(lane_estimate).fillna(2).astype(int)
    
    # Junction detection
    if "location" in df.columns:
        junction_patterns = r"(?i)\bjunction\b|signal|crossroad|intersection"
        df["has_junction"] = df["location"].str.contains(
            junction_patterns, regex=True, na=False
        ).astype(int)
    else:
        df["has_junction"] = 0
    
    # Road importance
    road_importance = {"Ring": 1.0, "Main": 0.8, "Underpass": 0.7, "Cross": 0.5, "Other": 0.3}
    df["road_importance"] = df["road_type"].map(road_importance).fillna(0.3)
    
    # ==================== SPATIAL GRID (NEW) ====================
    df["grid_cell"] = df.apply(
        lambda row: grid_cell_id(row["latitude"], row["longitude"]), axis=1
    )
    
    # ==================== EVENT SCORE ====================
    daily_counts = df.groupby("date").size()
    median_daily = daily_counts.median()
    event_score_map = (daily_counts / median_daily).clip(0, 5).to_dict()
    df["event_score"] = df["date"].map(event_score_map).fillna(1.0)
    
    # Weekend boost
    df.loc[df["is_weekend"] == 1, "event_score"] = df.loc[df["is_weekend"] == 1, "event_score"].clip(lower=1.0)
    
    # ==================== UNIQUE IDENTIFIERS ====================
    if "vehicle_number_masked" in df.columns:
        df["vehicle_id"] = df["vehicle_number_masked"].astype(str).str[:6]
    else:
        df["vehicle_id"] = "unknown"
    
    if "type_of_vehicle" in df.columns:
        df["vehicle_type"] = df["type_of_vehicle"]
    else:
        df["vehicle_type"] = "unknown"
    
    print(f"  Total features: {len(df.columns)}")
    print(f"  Records: {len(df):,}")
    
    return df


def run_pipeline() -> pd.DataFrame:
    """Execute the full data pipeline."""
    df = load_raw()
    df = clean(df)
    df = preprocess(df)
    return df


if __name__ == "__main__":
    df = run_pipeline()
    print(f"\nPipeline complete: {len(df):,} records with {len(df.columns)} features")
