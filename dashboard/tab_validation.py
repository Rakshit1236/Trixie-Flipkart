"""
Validation dashboard tab for Trixie-Flipkart.
Backtesting and model performance monitoring.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model_monitor import ModelMonitor


def render_tab_validation():
    st.subheader("📈 Model Validation & Backtesting")
    st.markdown("Track model performance over time and detect drift.")

    monitor = ModelMonitor()

    st.markdown("### Performance Metrics")

    col1, col2 = st.columns(2)
    with col1:
        window = st.selectbox("Evaluation Window", [7, 14, 30, 60],
                             format_func=lambda x: f"Last {x} days", index=0)
    with col2:
        cluster_filter = st.selectbox("Filter by Cluster",
                                      ["All Clusters"] + [f"Cluster {i}" for i in range(100)], index=0)

    cid = None if cluster_filter == "All Clusters" else int(cluster_filter.replace("Cluster ", ""))
    metrics = monitor.compute_metrics(window, cid)

    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("MAE", f"{metrics['mae']:.2f}")
        with m2:
            st.metric("RMSE", f"{metrics['rmse']:.2f}")
        with m3:
            st.metric("MAPE", f"{metrics['mape']:.1f}%")
        with m4:
            st.metric("R²", f"{metrics['r2']:.4f}")

        st.caption(f"Based on {metrics['n_samples']} samples over {metrics['window_days']} days")
    else:
        st.info("Insufficient data for metrics. Log predictions with actuals to enable metrics.")

    st.markdown("### Drift Detection")

    drift_result = monitor.detect_drift()

    if drift_result.get("reason"):
        st.info(drift_result["reason"])
    else:
        col1, col2 = st.columns(2)
        with col1:
            drift_status = "🔴 DRIFT DETECTED" if drift_result["drift_detected"] else "🟢 STABLE"
            st.metric("Model Status", drift_status)
        with col2:
            st.metric("Drift Ratio", f"{drift_result['drift_ratio']:.3f}")

        if drift_result["drift_detected"]:
            st.warning("⚠️ Model drift detected! Consider retraining the model.")

    st.markdown("### Backtesting")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())

    backtest_df = monitor.generate_backtest_report(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    if len(backtest_df) > 0:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("MAE Over Time", "R² Over Time", "Predicted vs Actual", "Error Distribution"),
        )

        fig.add_trace(
            go.Scatter(x=backtest_df["date"], y=backtest_df["mae"], mode="lines+markers", name="MAE", line=dict(color="#FF6B6B")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=backtest_df["date"], y=backtest_df["r2"], mode="lines+markers", name="R²", line=dict(color="#4ECDC4")),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(x=backtest_df["mean_actual"], y=backtest_df["mean_predicted"], mode="markers", name="Pred vs Actual", marker=dict(color="#45B7D1", size=8)),
            row=2, col=1,
        )
        errors = backtest_df["mean_actual"] - backtest_df["mean_predicted"]
        fig.add_trace(
            go.Histogram(x=errors, name="Error Distribution", marker_color="#96CEB4"),
            row=2, col=2,
        )

        fig.update_layout(height=600, title_text="Backtesting Results", template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            backtest_df.style.background_gradient(subset=["mae", "rmse"], cmap="RdYlGn_r"),
            use_container_width=True,
        )
    else:
        st.info("No backtest data available for the selected date range.")

    st.markdown("### Prediction History Summary")

    summary = monitor.get_history_summary()

    if summary["total_predictions"] > 0:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Predictions", summary["total_predictions"])
        with m2:
            st.metric("With Actuals", summary["predictions_with_actuals"])
        with m3:
            st.metric("Unique Dates", summary["unique_dates"])
        with m4:
            st.metric("Unique Clusters", summary["unique_clusters"])
    else:
        st.info("No prediction history. Use the pipeline to generate and log predictions.")
