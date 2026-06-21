"""
Utility functions for Trixie-flipkartgridlock.
Haversine distance, road classification, severity mapping, normalization.
"""
import math
import re
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

# ==================== GEOGRAPHY ====================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometers."""
    R = 6371.0  # Earth radius in km
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_km_array(lat1: np.ndarray, lon1: np.ndarray,
                       lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ==================== ROAD CLASSIFICATION ====================

def classify_road_type(location: str) -> str:
    """Classify road type from location text."""
    if not isinstance(location, str):
        return "Other"
    for road_type, pattern in [
        ("Ring", r"(?i)\bring\b"),
        ("Main", r"(?i)\bmain\b"),
        ("Underpass", r"(?i)\bunderpass\b|flyover|elevated"),
        ("Cross", r"(?i)\bcross\b"),
    ]:
        if re.search(pattern, location):
            return road_type
    return "Other"


def extract_road_name(location: str) -> str:
    """Extract road name from location text."""
    if not isinstance(location, str):
        return "Unknown"
    parts = [p.strip() for p in re.split(r"[,/\-]", location) if p.strip()]
    return parts[0] if parts else "Unknown"


def extract_area(location: str) -> str:
    """Extract area/locality from location text."""
    if not isinstance(location, str):
        return "Unknown"
    parts = [p.strip() for p in re.split(r"[,/\-]", location) if p.strip()]
    return parts[-1] if len(parts) > 1 else "Unknown"

# ==================== TIME ====================

def is_rush_hour(hour: int, rush_hours: list = None) -> bool:
    """Check if hour falls within rush hour windows."""
    if rush_hours is None:
        rush_hours = [(8, 11), (17, 21)]
    return any(start <= hour < end for start, end in rush_hours)


def get_time_of_day(hour: int) -> str:
    """Classify time of day."""
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def get_time_bin(hour: int) -> str:
    """Bin hours into 3-hour periods."""
    bins = [
        (0, 3, "00-03"), (3, 6, "03-06"), (6, 9, "06-09"),
        (9, 12, "09-12"), (12, 15, "12-15"), (15, 18, "15-18"),
        (18, 21, "18-21"), (21, 24, "21-24"),
    ]
    for start, end, label in bins:
        if start <= hour < end:
            return label
    return "00-03"

# ==================== SEVERITY ====================

def severity_from_violation_types(violation_types: str) -> Tuple[float, str]:
    """Map violation type string to (weight, label)."""
    if not isinstance(violation_types, str):
        return (0.7, "default")
    
    vt = violation_types.lower().strip()
    
    mapping = [
        ("illegal_parking", 0.9, "Illegal Parking"),
        ("parking", 0.9, "Parking"),
        ("double_parking", 0.85, "Double Parking"),
        ("obstruction", 0.8, "Obstruction"),
        ("no_parking_zone", 0.75, "No Parking Zone"),
    ]
    
    for pattern, weight, label in mapping:
        if pattern in vt:
            return (weight, label)
    
    return (0.7, "default")


def classify_severity(blocked_fraction: float) -> str:
    """Classify severity based on blocked fraction."""
    if blocked_fraction >= 0.40:
        return "CRITICAL"
    elif blocked_fraction >= 0.25:
        return "HIGH"
    elif blocked_fraction >= 0.10:
        return "MEDIUM"
    else:
        return "LOW"

# ==================== NORMALIZATION ====================

def normalize_to_0_100(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to 0-100 range."""
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    if vmax == vmin:
        return np.full_like(values, 50.0, dtype=float)
    return ((values - vmin) / (vmax - vmin) * 100).astype(float)


def z_score_normalize(values: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    mean, std = np.nanmean(values), np.nanstd(values)
    if std == 0:
        return np.zeros_like(values, dtype=float)
    return ((values - mean) / std).astype(float)

# ==================== STATISTICS ====================

def coefficient_of_variation(values: np.ndarray) -> float:
    """Coefficient of variation (std/mean)."""
    mean = np.nanmean(values)
    if mean == 0:
        return 0.0
    return float(np.nanstd(values) / mean)


def rolling_trend(values: np.ndarray, window: int) -> float:
    """Compute trend as slope of linear fit over window."""
    if len(values) < window:
        return 0.0
    recent = values[-window:]
    x = np.arange(len(recent))
    coeffs = np.polyfit(x, recent, 1)
    return float(coeffs[0])

# ==================== CYCLICAL ENCODING ====================

def cyclical_encode(values: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """Encode cyclic values (hour, day) as sin/cos."""
    sin_vals = np.sin(2 * np.pi * values / period).astype(float)
    cos_vals = np.cos(2 * np.pi * values / period).astype(float)
    return sin_vals, cos_vals

# ==================== GRID ====================

def lat_to_grid_cell(lat: float, grid_size: float = 0.001) -> int:
    """Convert latitude to grid cell ID."""
    return int(lat / grid_size)


def lon_to_grid_cell(lon: float, grid_size: float = 0.001) -> int:
    """Convert longitude to grid cell ID."""
    return int(lon / grid_size)


def grid_cell_id(lat: float, lon: float, grid_size: float = 0.001) -> str:
    """Create a unique grid cell ID from lat/lon."""
    return f"{lat_to_grid_cell(lat, grid_size)}_{lon_to_grid_cell(lon, grid_size)}"

# ==================== PANDAS HELPERS ====================

def safe_duration_minutes(row) -> float:
    """Compute violation duration from start/end times."""
    try:
        if pd.notna(row.get("closed_datetime")) and pd.notna(row.get("violation_start_datetime")):
            delta = (pd.to_datetime(row["closed_datetime"]) -
                     pd.to_datetime(row["violation_start_datetime"]))
            return max(delta.total_seconds() / 60.0, 5.0)
    except Exception:
        pass
    return 30.0  # default
