"""
Trixie-Flipkart — Main Entry Point
AI-powered dashboard for parking congestion intelligence.
Full-stack: Streamlit frontend + HuggingFace Spaces backend.
"""
import streamlit as st
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "https://rakshit1236-trixie-backend.hf.space")

st.set_page_config(
    page_title="Trixie — Parking Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_get(endpoint, params=None):
    try:
        r = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(endpoint, payload):
    try:
        r = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ==================== SIDEBAR ====================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/car.png", width=80)
    st.title("Trixie")
    st.caption("Parking Intelligence Platform")
    st.divider()

    health = api_get("/health")
    if "error" in health:
        st.error(f"Backend offline: {health['error']}")
        st.stop()

    st.metric("Status", health.get("status", "unknown"))
    st.metric("Hotspots", health.get("hotspots", 0))
    st.metric("Critical", health.get("critical", 0))
    st.metric("Predictions", health.get("predictions", 0))
    st.metric("Ensemble R²", f"{health.get('ensemble_r2', 0):.4f}")
    st.metric("MAE", f"{health.get('ensemble_mae', 0):.2f}")
    st.divider()
    st.caption(f"Backend: {BACKEND_URL}")
    st.caption(f"Last updated: {health.get('last_updated', 'never')}")

# ==================== LOAD DATA FROM API ====================
@st.cache_data(ttl=300)
def load_all_data():
    hotspots = api_get("/hotspots")
    impact = api_get("/impact")
    scores = api_get("/scores")
    predictions = api_get("/predictions")
    recommendations = api_get("/analytics/recommendations")
    warnings = api_get("/analytics/warnings")

    if any("error" in d for d in [hotspots, impact, scores, predictions, recommendations, warnings]):
        return None

    return {
        "hotspots": hotspots.get("hotspots", {}),
        "impact": impact.get("impact", {}),
        "scores": scores.get("priority", {}),
        "pri_scores": scores.get("pri", {}),
        "ranked": scores.get("ranked", []),
        "predictions": predictions.get("predictions", []),
        "recommendations": recommendations.get("recommendations", []),
        "dispatch_report": recommendations.get("dispatch_report", ""),
        "warnings": warnings,
    }


data = load_all_data()
if data is None:
    st.error("Failed to load data from backend. Check if backend is running.")
    st.stop()

profiles = data["hotspots"]
impact = data["impact"]
scores = data["scores"]
pri_scores = data["pri_scores"]
ranked = data["ranked"]
predictions = data["predictions"]
recommendations = data["recommendations"]
dispatch_report = data["dispatch_report"]
warnings = data["warnings"]

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
    from dashboard.tab_main import render_tab_main
    render_tab_main(profiles, impact, scores, {}, predictions, recommendations, dispatch_report)

with tab_scenario:
    from dashboard.tab_scenario import render_tab_scenario
    render_tab_scenario(profiles, impact, scores)

with tab_propagation:
    from dashboard.tab_propagation import render_tab_propagation
    render_tab_propagation(profiles, impact, scores)

with tab_insights:
    from dashboard.tab_insights import render_tab_insights
    render_tab_insights(profiles, impact, scores, pri_scores, ranked)

with tab_actions:
    from dashboard.tab_actions import render_tab_actions
    render_tab_actions(profiles, impact, scores, warnings, recommendations)

with tab_validation:
    from dashboard.tab_validation import render_tab_validation
    render_tab_validation()

st.divider()
ml_metrics = {"ensemble_r2": health.get("ensemble_r2", 0), "ensemble_mae": health.get("ensemble_mae", 0)}
st.caption(f"Trixie-Flipkart | Backend: {BACKEND_URL}")
st.caption(f"Ensemble R²: {ml_metrics.get('ensemble_r2', 0):.4f} | MAE: {ml_metrics.get('ensemble_mae', 0):.2f}")
