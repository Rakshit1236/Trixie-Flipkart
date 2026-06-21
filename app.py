"""
Trixie-Flipkart — Main Entry Point
AI-powered dashboard for parking congestion intelligence.
Full-stack: Streamlit frontend + HuggingFace Spaces backend.
"""
import streamlit as st
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CACHE_FILE, MODELS_DIR, PREDICTIONS_DIR, OUTPUT_DIR,
    HEATMAPS_DIR, REPORTS_DIR, VALIDATION_DIR
)
from src.data_pipeline import run_pipeline
from src.clustering import cluster_hotspots
from src.traffic_impact import run_impact_analysis
from src.scoring import run_scoring
from src.predictive_model import run_ml_pipeline, load_models
from src.analytics import run_analytics, CongestionPropagation
from src.visualization import generate_all_visualizations
from src.model_monitor import ModelMonitor

from dashboard.tab_main import render_tab_main
from dashboard.tab_scenario import render_tab_scenario
from dashboard.tab_propagation import render_tab_propagation
from dashboard.tab_insights import render_tab_insights
from dashboard.tab_actions import render_tab_actions
from dashboard.tab_validation import render_tab_validation

st.set_page_config(
    page_title="Trixie — Parking Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car.png", width=80)
    st.title("Trixie")
    st.caption("Parking Intelligence Platform")

    st.divider()

    if st.button("🔄 Run Full Pipeline", use_container_width=True):
        with st.spinner("Running full pipeline... ~40-60 seconds."):
            try:
                st.status("Loading data...", expanded=True)
                df = run_pipeline()
                st.success(f"Data loaded: {len(df):,} records")

                st.status("Clustering hotspots...", expanded=True)
                df, profiles = cluster_hotspots(df)
                st.success(f"Clusters: {len(profiles)}")

                st.status("Computing traffic impact...", expanded=True)
                impact, ripple = run_impact_analysis(df, profiles)
                st.success("Impact computed")

                st.status("Scoring hotspots...", expanded=True)
                scores, pri_scores, ranked = run_scoring(profiles, impact, df)
                st.success("Scores computed")

                st.status("Training ML models...", expanded=True)
                predictions, ml_metrics, feature_names = run_ml_pipeline(df, profiles)
                st.success(f"ML trained (R²={ml_metrics['ensemble_r2']:.4f})")

                st.status("Running analytics...", expanded=True)
                analytics_results = run_analytics(profiles, impact, scores, ripple)
                st.success("Analytics complete")

                st.status("Generating visualizations...", expanded=True)
                viz_paths = generate_all_visualizations(df, profiles, impact, predictions, ranked)
                st.success(f"Visualizations: {len(viz_paths)}")

                monitor = ModelMonitor()
                monitor.log_predictions_batch(
                    datetime.now().strftime("%Y-%m-%d"), predictions
                )

                cache = {
                    "df": df,
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
                }

                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                with open(CACHE_FILE, "wb") as f:
                    pickle.dump(cache, f)

                st.success("Pipeline complete! Results cached.")
                st.rerun()

            except Exception as e:
                st.error(f"Pipeline failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    if st.button("🗑️ Clear Cache", use_container_width=True):
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            st.success("Cache cleared")
            st.rerun()

    st.divider()

    if "cache" in st.session_state:
        cache = st.session_state.cache
        st.metric("Hotspots", len(cache.get("profiles", {})))
        critical = sum(1 for c in cache.get("impact", {}).values() if c.get("severity_class") == "CRITICAL")
        st.metric("Critical", critical)
        st.metric("Records", f"{len(cache.get('df', [])):,}")

# ==================== LOAD DATA ====================
@st.cache_data
def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return None

cache = load_cache()

if cache is None:
    st.title("🚗 Trixie — Parking Intelligence Platform")
    st.markdown("""
    Welcome to **Trixie**, an AI-powered dashboard for parking congestion intelligence.

    ### Getting Started
    1. Click **"Run Full Pipeline"** in the sidebar
    2. Wait ~40-60 seconds
    3. Explore the 6 dashboard tabs

    ### Features
    - **Executive Overview** — KPIs, heatmaps, forecast with confidence gauges
    - **What-If Simulator** — One-click presets, counterfactual mode
    - **Congestion Propagation** — Step-by-step cascade timeline
    - **Insights** — SHAP-based XAI + Root Cause Attribution cards
    - **Actions** — Early Warning Timeline + Enhanced Dispatch cards
    - **Validation** — Backtesting + drift detection
    """)
    st.stop()

# Unpack
profiles = cache["profiles"]
impact = cache["impact"]
ripple = cache["ripple"]
scores = cache["scores"]
pri_scores = cache["pri_scores"]
ranked = cache["ranked"]
predictions = cache["predictions"]
ml_metrics = cache["ml_metrics"]
feature_names = cache["feature_names"]
analytics_results = cache["analytics"]
viz_paths = cache["viz_paths"]

st.session_state.cache = cache
st.session_state.xai_explanations = cache.get("xai_explanations", {})

# ==================== MAIN CONTENT ====================
st.title("🚗 Trixie — Parking Intelligence Platform")

tab_main, tab_scenario, tab_propagation, tab_insights, tab_actions, tab_validation = st.tabs([
    "📊 Overview & Maps",
    "🧪 What-If Simulator",
    "🌊 Congestion Propagation",
    "💡 Insights (XAI + PRI)",
    "⚡ Actions",
    "📈 Validation",
])

with tab_main:
    render_tab_main(
        profiles, impact, scores, ripple,
        predictions,
        analytics_results.get("recommendations"),
        analytics_results.get("dispatch_report"),
    )

with tab_scenario:
    render_tab_scenario(profiles, impact, scores)

with tab_propagation:
    render_tab_propagation(profiles, impact, scores)

with tab_insights:
    render_tab_insights(profiles, impact, scores, pri_scores, ranked)

with tab_actions:
    render_tab_actions(
        profiles, impact, scores,
        analytics_results.get("warnings"),
        analytics_results.get("recommendations"),
    )

with tab_validation:
    render_tab_validation()

st.divider()
st.caption(f"Trixie-Flipkart | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"Ensemble R²: {ml_metrics.get('ensemble_r2', 0):.4f} | MAE: {ml_metrics.get('ensemble_mae', 0):.2f}")
