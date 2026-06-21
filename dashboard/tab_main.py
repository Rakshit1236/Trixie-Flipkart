"""
Main dashboard tab for Trixie-Flipkart.
Overview, heatmaps, forecast with confidence gauges, and dispatch.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HEATMAPS_DIR, REPORTS_DIR, PREDICTIONS_DIR


def render_overview(profiles, impact, scores, ripple):
    st.subheader("📊 Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Hotspots", len(profiles))

    with col2:
        critical_count = sum(1 for c in impact.values() if c.get("severity_class") == "CRITICAL")
        st.metric("Critical Hotspots", critical_count, delta=None, delta_color="inverse")

    with col3:
        avg_speed_drop = sum(c.get("worst_speed_drop_pct", 0) for c in impact.values()) / max(len(impact), 1)
        st.metric("Avg Speed Drop", f"{avg_speed_drop:.1f}%")

    with col4:
        total_vhl = sum(c.get("total_vhl", 0) for c in impact.values())
        st.metric("Vehicle-Hours Lost", f"{total_vhl:.0f}")

    st.subheader("Road Type Distribution")

    road_counts = {}
    for p in profiles.values():
        rt = p.get("road_type", "Other")
        road_counts[rt] = road_counts.get(rt, 0) + 1

    fig = px.bar(
        x=list(road_counts.keys()),
        y=list(road_counts.values()),
        labels={"x": "Road Type", "y": "Count"},
        title="Hotspots by Road Type",
        color=list(road_counts.keys()),
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)


def render_confidence_gauges(predictions):
    """Render confidence gauges for top hotspots."""
    st.subheader("🎯 Forecast Confidence")

    if predictions is None or len(predictions) == 0:
        st.info("Run pipeline to see confidence gauges")
        return

    top5 = predictions.head(5)

    cols = st.columns(5)
    for i, (_, row) in enumerate(top5.iterrows()):
        with cols[i]:
            confidence = row.get("confidence_pct", 80)
            pred = row.get("predicted_violations", 0)
            area = row.get("area", "Unknown")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={"text": f"C{int(row['cluster_id'])}<br>{area}", "font": {"size": 12}},
                number={"suffix": "%", "font": {"size": 20}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#4ECDC4" if confidence >= 80 else "#FFCC00" if confidence >= 60 else "#FF6B6B"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "steps": [
                        {"range": [0, 60], "color": "#FFE0E0"},
                        {"range": [60, 80], "color": "#FFF3E0"},
                        {"range": [80, 100], "color": "#E0FFE0"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": confidence,
                    },
                },
            ))
            fig.update_layout(height=200, margin=dict(t=60, b=20, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

            st.caption(f"Pred: {pred:.0f} violations")

    # Confidence distribution
    st.subheader("Confidence Distribution")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Confidence by Hotspot", "Model Agreement (LGB vs XGB)"),
    )

    fig.add_trace(
        go.Bar(
            x=[f"C{int(r['cluster_id'])}" for _, r in predictions.head(15).iterrows()],
            y=predictions.head(15)["confidence_pct"],
            marker_color=[
                "#4ECDC4" if c >= 80 else "#FFCC00" if c >= 60 else "#FF6B6B"
                for c in predictions.head(15)["confidence_pct"]
            ],
            name="Confidence %",
        ),
        row=1, col=1,
    )

    if "lgb_prediction" in predictions.columns and "xgb_prediction" in predictions.columns:
        lgb = predictions.head(15)["lgb_prediction"]
        xgb = predictions.head(15)["xgb_prediction"]
        agreement = 100 - abs(lgb - xgb) / ((lgb + xgb) / 2 + 1e-6) * 100
        agreement = agreement.clip(0, 100)

        fig.add_trace(
            go.Bar(
                x=[f"C{int(r['cluster_id'])}" for _, r in predictions.head(15).iterrows()],
                y=agreement,
                marker_color="#45B7D1",
                name="Agreement %",
            ),
            row=1, col=2,
        )

    fig.update_layout(height=350, template="plotly_dark", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_heatmaps():
    st.subheader("🗺️ Interactive Heatmaps")

    tabs = st.tabs(["Violation Density", "Traffic Impact", "Temporal Patterns"])

    with tabs[0]:
        heatmap_path = HEATMAPS_DIR / "hotspot_heatmap.html"
        if heatmap_path.exists():
            with open(heatmap_path, "r") as f:
                st.components.v1.html(f.read(), height=500)
        else:
            st.info("Run pipeline to generate heatmap")

    with tabs[1]:
        impact_path = HEATMAPS_DIR / "impact_map.html"
        if impact_path.exists():
            with open(impact_path, "r") as f:
                st.components.v1.html(f.read(), height=500)
        else:
            st.info("Run pipeline to generate impact map")

    with tabs[2]:
        temporal_path = HEATMAPS_DIR / "temporal_heatmap.png"
        if temporal_path.exists():
            st.image(str(temporal_path), use_container_width=True)
        else:
            st.info("Run pipeline to generate temporal heatmap")


def render_forecast(predictions):
    st.subheader("🔮 Tomorrow's Forecast")

    if predictions is None or len(predictions) == 0:
        st.info("Run pipeline to generate predictions")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        total_predicted = predictions["predicted_violations"].sum()
        st.metric("Total Predicted Violations", f"{total_predicted:.0f}")

    with col2:
        high_risk = len(predictions[predictions["predicted_violations"] >= 30])
        st.metric("High-Risk Hotspots", high_risk)

    with col3:
        avg_confidence = predictions["confidence_pct"].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.0f}%")

    fig = px.bar(
        predictions.head(20),
        x="cluster_id",
        y="predicted_violations",
        error_y="upper_bound",
        error_y_minus="lower_bound",
        color="road_type",
        hover_data=["area", "confidence_pct"],
        title="Top 20 Predicted Hotspots",
        labels={"cluster_id": "Cluster", "predicted_violations": "Predicted Violations"},
    )
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

    pred_map_path = HEATMAPS_DIR / "predictions_heatmap.html"
    if pred_map_path.exists():
        st.subheader("Predictions Map")
        with open(pred_map_path, "r") as f:
            st.components.v1.html(f.read(), height=400)


def render_dispatch(recommendations, dispatch_report):
    st.subheader("📋 Dispatch Recommendations")

    if dispatch_report:
        st.code(dispatch_report, language=None)

    if recommendations:
        total_officers = sum(r["officers_needed"] for r in recommendations)
        critical = [r for r in recommendations if r["severity"] == "CRITICAL"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Officers Needed", total_officers)
        with col2:
            st.metric("Critical Hotspots", len(critical))

        df = pd.DataFrame(recommendations)
        st.dataframe(
            df[["cluster_id", "area", "severity", "priority_score", "officers_needed", "timing"]],
            use_container_width=True,
        )


def render_tab_main(profiles, impact, scores, ripple, predictions, recommendations, dispatch_report):
    render_overview(profiles, impact, scores, ripple)

    st.divider()

    subtabs = st.tabs(["Confidence Gauges", "Heatmaps", "Tomorrow's Forecast", "Dispatch Report"])

    with subtabs[0]:
        render_confidence_gauges(predictions)

    with subtabs[1]:
        render_heatmaps()

    with subtabs[2]:
        render_forecast(predictions)

    with subtabs[3]:
        render_dispatch(recommendations, dispatch_report)
