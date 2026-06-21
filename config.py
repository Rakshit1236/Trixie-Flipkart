"""
Configuration for Trixie-Flipkart
All constants, weights, paths, and tunable parameters.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
HEATMAPS_DIR = OUTPUT_DIR / "heatmaps"
MODELS_DIR = OUTPUT_DIR / "models"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
REPORTS_DIR = OUTPUT_DIR / "reports"
VALIDATION_DIR = OUTPUT_DIR / "validation"

# Data file
CSV_FILE = DATA_DIR / "jan to may police violation_anonymized791b166.csv"

# Cache file
CACHE_FILE = OUTPUT_DIR / "pipeline_cache.pkl"

# ==================== TRAFFIC MODEL ====================
FREE_FLOW_SPEED_KMH = 40.0
JAM_DENSITY_VPKM = 200.0
IMPACT_DECAY_FACTOR = 0.85
PROPAGATION_RADIUS_KM = 2.0

DEFAULT_ROAD_CAPACITY_VPH = 1800
LANE_CAPACITY_VPH = 600

# ==================== CLUSTERING ====================
HDBSCAN_MIN_CLUSTER_SIZE = 20
HDBSCAN_MIN_SAMPLES = 5
HDBSCAN_METRIC = "haversine"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

# ==================== SCORING ====================
PRIORITY_WEIGHTS = {
    "frequency": 0.25,
    "impact": 0.35,
    "urgency": 0.20,
    "criticality": 0.20,
}

PRI_WEIGHTS = {
    "illegal": 0.4,
    "density": 0.3,
    "road": 0.2,
    "event": 0.1,
}

SEVERITY_WEIGHTS = {
    "parking": 0.9,
    "illegal_parking": 0.9,
    "double_parking": 0.85,
    "obstruction": 0.8,
    "no_parking_zone": 0.75,
    "default": 0.7,
}

LANE_ESTIMATE = {
    "Ring": 4,
    "Main": 3,
    "Underpass": 2,
    "Cross": 2,
    "Other": 2,
}

ROAD_IMPORTANCE = {
    "Ring": 1.0,
    "Main": 0.8,
    "Underpass": 0.7,
    "Cross": 0.5,
    "Other": 0.3,
}

CHRONIC_THRESHOLD_PCT = 0.30
RUSH_HOURS = [(8, 11), (17, 21)]

# ==================== CONFORMAL PREDICTION ====================
CONFORMAL_ALPHA = 0.10
CONFORMAL_N_BOOTSTRAP = 100
CONFORMAL_CV_FOLDS = 5

# ==================== MODEL MONITORING ====================
DRIFT_THRESHOLD = 0.20
RETRAIN_WINDOW_DAYS = 30
MIN_SAMPLES_FOR_METRICS = 5

# ==================== XAI ====================
XAI_BACKGROUND_SAMPLES = 100
XAI_MAX_DISPLAY_FEATURES = 10

# ==================== SCENARIO SIMULATOR ====================
WEATHER_MULTIPLIERS = {
    "Clear": 1.0,
    "Light Rain": 1.25,
    "Heavy Rain": 1.6,
    "Fog": 1.35,
}

DAY_TYPE_MULTIPLIERS = {
    "Weekday": 1.0,
    "Saturday": 1.15,
    "Sunday": 0.85,
    "Festival Day": 1.5,
    "Public Holiday": 0.7,
}

# ==================== SCENARIO PRESETS ====================
SCENARIO_PRESETS = {
    "Remove Illegal Parking": {
        "vehicle_reduction": 50,
        "weather": "Clear",
        "day_type": "Weekday",
        "description": "Simulate removing 50 illegally parked vehicles from all hotspots",
        "icon": "🚗",
    },
    "Festival Day": {
        "vehicle_reduction": 0,
        "weather": "Clear",
        "day_type": "Festival Day",
        "description": "Impact of a festival day on parking congestion",
        "icon": "🎉",
    },
    "Heavy Rain": {
        "vehicle_reduction": 0,
        "weather": "Heavy Rain",
        "day_type": "Weekday",
        "description": "Effect of heavy rainfall on traffic flow",
        "icon": "🌧️",
    },
    "Metro Delay": {
        "vehicle_reduction": 30,
        "weather": "Clear",
        "day_type": "Weekday",
        "description": "Metro service delay causing spillover parking",
        "icon": "🚇",
    },
    "Increase Capacity": {
        "vehicle_reduction": 100,
        "weather": "Clear",
        "day_type": "Weekday",
        "description": "Aggressive enforcement removing 100 vehicles",
        "icon": "📈",
    },
}

# ==================== EARLY WARNING ====================
WARNING_HORIZONS = [15, 30, 60]
THREAT_LEVELS = {
    "HIGH": 70,
    "MEDIUM": 40,
    "LOW": 0,
}

RUSH_HOUR_MULTIPLIER = 1.5
CHRONIC_MULTIPLIER = 1.3
TIME_OF_DAY_MULTIPLIERS = {
    "morning": 1.4,
    "afternoon": 1.0,
    "evening": 1.5,
    "night": 0.3,
}

# ==================== PROPAGATION ====================
PROPAGATION_SPEED_KMH = 15.0
PROPAGATION_MIN_DELAY_MIN = 8
PROPAGATION_SPEED_ESCALATION = 0.8

# ==================== DISPATCH ====================
OFFICER_DEPLOYMENT = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
}

DELAY_REDUCTION_PCT = {
    "CRITICAL": 0.60,
    "HIGH": 0.50,
    "MEDIUM": 0.35,
    "LOW": 0.20,
}

DELAY_REDUCTION_CAP = {
    "CRITICAL": 50,
    "HIGH": 40,
    "MEDIUM": 30,
    "LOW": 20,
}

RESOURCE_TYPE = {
    "CRITICAL": "Deploy Officers + Tow Truck",
    "HIGH": "Deploy Officers",
    "MEDIUM": "Patrol Unit",
    "LOW": "Monitor",
}

ETA_PER_OFFICER_MIN = 2

# ==================== GEOGRAPHY ====================
INDIA_LAT_MIN = 6.0
INDIA_LAT_MAX = 38.0
INDIA_LON_MIN = 68.0
INDIA_LON_MAX = 98.0

BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946

# ==================== ROAD CLASSIFICATION ====================
ROAD_PATTERNS = {
    "Ring": r"(?i)\bring\b",
    "Main": r"(?i)\bmain\b",
    "Underpass": r"(?i)\bunderpass\b|flyover|elevated",
    "Cross": r"(?i)\bcross\b",
}

# ==================== UI ====================
DARK_THEME_CONFIG = """
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
"""

# ==================== ML MODEL ====================
DEFAULT_LGB_PARAMS = {
    "objective": "regression",
    "metric": "r2",
    "boosting_type": "gbdt",
    "verbose": -1,
    "n_jobs": -1,
}

DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "verbosity": 0,
    "nthread": -1,
}

OPTUNA_N_TRIALS = 5
OPTUNA_CV_FOLDS = 2
OPTUNA_RANDOM_STATE = 42

ENSEMBLE_DEFAULT_WEIGHTS = {"lgb": 0.5, "xgb": 0.5}
CONFIDENCE_CI_WIDTH = 0.15

# ==================== FEATURE ENGINEERING ====================
LAG_FEATURES = [1, 2, 3, 5, 7, 14]
ROLLING_WINDOWS = [3, 5, 7, 14]
SEVERITY_ROLLING_WINDOWS = [3, 7]
TREND_WINDOWS = [3, 7, 14]
