"""
Validation dashboard tab for Trixie-Flipkart.
Shows backend health and model metrics.
"""
import streamlit as st
import requests
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "https://rakshit1236-trixie-backend.hf.space")


def api_get(endpoint):
    try:
        r = requests.get(f"{BACKEND_URL}{endpoint}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def render_tab_validation():
    st.subheader("Model Validation & Backend Status")

    st.markdown("### Backend Health")

    health = api_get("/health")

    if "error" in health:
        st.error(f"Backend is offline: {health['error']}")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Status", health.get("status", "unknown"))
    with col2:
        st.metric("Pipeline Loaded", "Yes" if health.get("pipeline_loaded") else "No")
    with col3:
        st.metric("Ensemble R²", f"{health.get('ensemble_r2', 0):.4f}")
    with col4:
        st.metric("MAE", f"{health.get('ensemble_mae', 0):.2f}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Hotspots", health.get("hotspots", 0))
    with col2:
        st.metric("Critical Hotspots", health.get("critical", 0))
    with col3:
        st.metric("Predictions", health.get("predictions", 0))

    st.caption(f"Backend URL: {BACKEND_URL}")
    st.caption(f"Last Updated: {health.get('last_updated', 'never')}")

    st.divider()
    st.markdown("### Pipeline Status")
    st.info("""
    The ML pipeline runs on the HuggingFace Spaces backend. The backend trains
    LightGBM, XGBoost, and Ridge regression models with Optuna hyperparameter
    tuning, then serves predictions via the API.

    **Models:** LightGBM + XGBoost + Ridge (Ensemble)
    **Features:** Cyclical time encoding, spatial grid, severity weights
    **Validation:** Time-series split with MAE, RMSE, R² metrics
    """)

    st.markdown("### Data Summary")
    st.info(f"""
    **Data Source:** Bengaluru traffic police violations (Jan-May)
    **Records:** 105MB+ of anonymized violation data
    **Clusters:** {health.get('hotspots', 0)} hotspot profiles identified via HDBSCAN
    **Predictions:** {health.get('predictions', 0)} cluster-level forecasts
    """)
