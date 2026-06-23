"""Recompute XAI factor breakdowns with correct feature-to-factor mapping."""
import pickle, json, numpy as np
from pathlib import Path

# Correct mapping: ML feature name -> factor group
FEATURE_TO_FACTOR = {
    # Volume / Activity
    "total_violations": "Violation Volume",
    "avg_daily_rate": "Violation Volume",
    "days_active": "Pattern Consistency",
    # Historical Trends
    "violations_lag_1": "Historical Trend",
    "violations_lag_2": "Historical Trend",
    "violations_lag_3": "Historical Trend",
    "violations_lag_5": "Historical Trend",
    "violations_lag_7": "Historical Trend",
    "violations_lag_14": "Historical Trend",
    # Rolling Patterns
    "rolling_3d_mean": "Recent Pattern",
    "rolling_3d_max": "Recent Pattern",
    "rolling_5d_mean": "Recent Pattern",
    "rolling_5d_max": "Recent Pattern",
    "rolling_7d_mean": "Recent Pattern",
    "rolling_7d_max": "Recent Pattern",
    "rolling_7d_std": "Recent Pattern",
    "rolling_14d_mean": "Recent Pattern",
    "rolling_14d_max": "Recent Pattern",
    "rolling_14d_std": "Recent Pattern",
    # Temporal / Seasonality
    "dow_sin": "Seasonality",
    "dow_cos": "Seasonality",
    "month_sin": "Seasonality",
    "month_cos": "Seasonality",
    "dom_sin": "Seasonality",
    "dom_cos": "Seasonality",
    "week_of_year": "Seasonality",
    "day_of_week": "Seasonality",
    "day_of_month": "Seasonality",
    "month": "Seasonality",
    "is_month_start": "Seasonality",
    "is_month_end": "Seasonality",
    # Time of Day
    "avg_hour": "Time of Day",
    "std_hour": "Time of Day",
    "rush_hour_ratio": "Time of Day",
    "weekend_ratio": "Time of Day",
    # Momentum / Acceleration
    "acceleration_3d": "Momentum",
    "acceleration_7d": "Momentum",
    "momentum_3d": "Momentum",
    "momentum_7d": "Momentum",
    "trend_3d": "Momentum",
    "trend_7d": "Momentum",
    "trend_14d": "Momentum",
    # Volatility
    "cv_7d": "Volatility",
    "cv_14d": "Volatility",
    # Severity / Road Profile
    "avg_severity": "Severity Profile",
    "is_chronic": "Severity Profile",
    "has_junction": "Road Profile",
    "num_lanes": "Road Profile",
    "road_type_encoded": "Road Profile",
    # Ratios
    "ratio_to_7d_mean": "Relative Intensity",
    "ratio_to_14d_mean": "Relative Intensity",
    "violations_14d_mean": "Relative Intensity",
    # Recency
    "recency_weighted_mean": "Recency",
    # Days active
    "days_active": "Pattern Consistency",
}

FACTOR_COLORS = {
    "Violation Volume": "#FF6B6B",
    "Historical Trend": "#4ECDC4",
    "Recent Pattern": "#45B7D1",
    "Seasonality": "#96CEB4",
    "Time of Day": "#DDA0DD",
    "Momentum": "#FF8C00",
    "Volatility": "#FFE066",
    "Severity Profile": "#FF6600",
    "Road Profile": "#4ECDC4",
    "Relative Intensity": "#96CEB4",
    "Recency": "#FF6B6B",
}


def recompute_factor_breakdowns(cache):
    xai = cache.get("xai_explanations", {})
    print(f"Recomputing {len(xai)} XAI explanations...")

    for cid, explanation in xai.items():
        fi = explanation.get("feature_importance", {})
        if not fi:
            continue

        # Group features by factor
        factor_abs_shap = {}
        for feat_name, feat_data in fi.items():
            factor = FEATURE_TO_FACTOR.get(feat_name, "Other")
            abs_shap = feat_data.get("mean_abs_shap", 0)
            factor_abs_shap[factor] = factor_abs_shap.get(factor, 0) + abs_shap

        # Normalize to percentages
        total = sum(factor_abs_shap.values())
        if total > 0:
            pct = {f: round(v / total * 100, 1) for f, v in factor_abs_shap.items()}
        else:
            pct = {f: 0 for f in factor_abs_shap}

        # Sort by contribution
        pct = dict(sorted(pct.items(), key=lambda x: x[1], reverse=True))

        # Dominant factor
        dominant = list(pct.keys())[0] if pct else "Unknown"

        explanation["percentage_contributions"] = pct
        explanation["dominant_factor"] = dominant
        explanation["factor_colors"] = {f: FACTOR_COLORS.get(f, "#888888") for f in pct.keys()}

    return cache


cache_path = Path(r"C:\Users\raksh\Desktop\PROJECT\Trixie-Flipkart\output\pipeline_cache.pkl")
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

cache = recompute_factor_breakdowns(cache)

# Show samples
xai = cache["xai_explanations"]
samples = [0, 50, 100, 200, 500]
for cid in samples:
    cid_str = str(cid)
    if cid_str in xai:
        d = xai[cid_str]
        print(f"\nCluster {cid}: {d['dominant_factor']}")
        print(f"  {d['percentage_contributions']}")

# Save updated cache
with open(cache_path, "wb") as f:
    pickle.dump(cache, f)
print(f"\nSaved updated cache")
