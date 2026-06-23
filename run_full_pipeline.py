"""Run the full pipeline with XAI + area name resolution."""
import sys
sys.path.insert(0, r"C:\Users\raksh\Desktop\PROJECT\Trixie-Flipkart")

from src.data_pipeline import run_pipeline
from src.clustering import cluster_hotspots
from src.traffic_impact import run_impact_analysis
from src.scoring import run_scoring
from src.predictive_model import run_ml_pipeline, build_ml_features, generate_30day_forecast, compute_daily_forecast_summary
from src.analytics import run_analytics
from src.visualization import generate_all_visualizations
from src.pincode_map import get_area_name
from config import CACHE_FILE
import pickle
import numpy as np

print("=== Step 1: Data Pipeline ===")
df = run_pipeline()
print(f"  Records: {len(df):,}")

print("=== Step 2: Clustering ===")
df, profiles = cluster_hotspots(df)
print(f"  Hotspots: {len(profiles)}")

print("=== Fixing Area Names ===")
for cid, profile in profiles.items():
    profile["area"] = get_area_name(profile.get("area", ""))
print("  Area names resolved")

print("=== Step 3: Traffic Impact ===")
impact, ripple = run_impact_analysis(df, profiles)
print("  Impact computed")

print("=== Step 4: Scoring ===")
scores, pri_scores, ranked = run_scoring(profiles, impact, df)
print(f"  Ranked: {len(ranked)}")

print("=== Step 5: ML Pipeline ===")
predictions, ml_metrics, feature_names, ml_df, lgb_model = run_ml_pipeline(df, profiles)
r2 = ml_metrics["ensemble_r2"]
mae = ml_metrics["ensemble_mae"]
print(f"  R2={r2:.4f}, MAE={mae:.2f}")

print("=== Step 5b: 30-Day Forecast ===")
forecast_df = generate_30day_forecast(ml_df, profiles, ml_metrics, feature_names)
forecast_summary = compute_daily_forecast_summary(forecast_df)
print(f"  Forecast: {len(forecast_df)} rows, {len(forecast_summary)} days")

print("=== Step 5c: XAI Explanations ===")
from src.xai_engine import XAIEngine

# Get feature columns (exclude non-feature cols)
exclude_cols = {"cluster_id", "date", "violations"}
feature_cols = [c for c in feature_names if c in ml_df.columns]
X_all = ml_df[feature_cols].fillna(0)

# Create XAI engine with the trained LightGBM model
xai_engine = XAIEngine(lgb_model, X_all.sample(min(200, len(X_all)), random_state=42), feature_cols)

# Generate per-cluster XAI explanations
xai_explanations = {}
cluster_ids = sorted(ml_df["cluster_id"].unique())
for i, cid in enumerate(cluster_ids):
    cluster_data = ml_df[ml_df["cluster_id"] == cid]
    if len(cluster_data) < 2:
        continue
    X_cluster = cluster_data[feature_cols].fillna(0)
    try:
        explanation = xai_engine.generate_root_breakdown(X_cluster)
        xai_explanations[str(cid)] = {
            "dominant_factor": explanation["dominant_factor"],
            "percentage_contributions": explanation["factor_percentages"],
            "feature_importance": {
                name: {
                    "mean_shap": data.get("mean_shap", 0),
                    "mean_abs_shap": data.get("mean_abs_shap", 0),
                    "direction": data.get("direction", "unknown"),
                    "consistency": data.get("consistency", 0),
                }
                for name, data in explanation["explanation"]["feature_importance"].items()
            },
        }
    except Exception as e:
        xai_explanations[str(cid)] = {"error": str(e)}

    if (i + 1) % 200 == 0:
        print(f"  XAI: {i+1}/{len(cluster_ids)} clusters")

print(f"  XAI: {len(xai_explanations)} explanations generated")

print("=== Step 6: Analytics ===")
analytics_results = run_analytics(profiles, impact, scores, ripple, xai=xai_explanations)

print("=== Step 7: Visualizations ===")
viz_paths = generate_all_visualizations(df, profiles, impact, predictions, ranked)
print(f"  Viz: {len(viz_paths)}")

print("=== Saving Cache ===")
cache = {
    "profiles": profiles,
    "impact": impact,
    "ripple": ripple,
    "scores": scores,
    "pri_scores": pri_scores,
    "ranked": ranked,
    "predictions": predictions,
    "ml_metrics": ml_metrics,
    "feature_names": feature_names,
    "analytics": analytics_results,
    "viz_paths": viz_paths,
    "xai_explanations": xai_explanations,
    "forecast": forecast_df.to_dict(orient="records"),
    "forecast_summary": forecast_summary,
}
with open(CACHE_FILE, "wb") as f:
    pickle.dump(cache, f)
print(f"  Saved to {CACHE_FILE}")
print("DONE")
