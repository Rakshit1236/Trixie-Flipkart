"""
Insights tab for Trixie-Flipkart.
XAI breakdown with root cause attribution cards + Parking Risk Index ranking.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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


def render_root_cause_cards(explanation, cluster_id):
    if not explanation:
        st.info("No XAI explanation available for this hotspot")
        return

    dominant = explanation.get("dominant_factor", "Unknown")
    pct_contributions = explanation.get("percentage_contributions", {})

    if not pct_contributions:
        st.info("No contribution data available")
        return

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border: 2px solid #FF6B6B;
                border-radius: 12px;
                padding: 20px;
                margin: 10px 0;
                text-align: center;">
        <div style="font-size: 14px; color: #888; text-transform: uppercase; letter-spacing: 2px;">
            Dominant Factor
        </div>
        <div style="font-size: 28px; font-weight: bold; color: #FF6B6B; margin: 8px 0;">
            {dominant}
        </div>
        <div style="font-size: 13px; color: #4ECDC4;">
            Cluster {cluster_id} — Primary contributor to congestion
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Contributors")

    factor_colors = {
        "Illegal Parking": "#FF6B6B",
        "Parking": "#FF6B6B",
        "Road Width": "#4ECDC4",
        "Road Importance": "#4ECDC4",
        "Density": "#45B7D1",
        "Event Score": "#96CEB4",
        "Weather": "#FFE066",
        "Time of Day": "#DDA0DD",
        "Junction Proximity": "#FF8C00",
    }

    for factor, pct in pct_contributions.items():
        color = factor_colors.get(factor, "#888888")
        bar_width = max(pct, 5)

        st.markdown(f"""
        <div style="margin: 8px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <span style="font-size: 14px; font-weight: 500; color: {color};">{factor}</span>
                <span style="font-size: 14px; font-weight: bold; color: #FAFAFA;">{pct:.0f}%</span>
            </div>
            <div style="background: #2a2a3e; border-radius: 6px; height: 12px; overflow: hidden;">
                <div style="background: {color}; width: {bar_width}%; height: 100%; border-radius: 6px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_tab_insights(profiles, impact, scores, pri_scores, ranked):
    st.subheader("Insights")

    tabs = st.tabs(["Explainable AI (XAI)", "Parking Risk Index (PRI)"])

    with tabs[0]:
        st.markdown("### Root Cause Analysis")
        st.markdown("Select a hotspot to see the root cause breakdown based on SHAP values.")

        hotspot_options = [
            f"Cluster {cid} ({profiles[cid].get('area', 'Unknown')})"
            for cid in sorted(int(k) for k in profiles.keys())
        ]
        hotspot_selection = st.selectbox("Select Hotspot", hotspot_options, index=0, key="xai_hotspot")
        cid = int(hotspot_selection.split("(")[0].replace("Cluster ", "").strip())

        result = api_get(f"/root_cause/{cid}")

        if "error" in result:
            st.warning(f"No XAI explanation available for Cluster {cid}. {result.get('error', '')}")
        else:
            render_root_cause_cards(result, cid)

            st.divider()
            col1, col2 = st.columns(2)

            with col1:
                pct_contributions = result.get("percentage_contributions", {})
                if pct_contributions:
                    fig = go.Figure(data=[go.Pie(
                        labels=list(pct_contributions.keys()),
                        values=list(pct_contributions.values()),
                        hole=0.4,
                        marker_colors=px.colors.qualitative.Set2,
                    )])
                    fig.update_layout(height=350, title="Root Cause Distribution", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                feature_importance = result.get("feature_importance", {})
                if feature_importance:
                    top_features = dict(list(feature_importance.items())[:8])
                    fig = go.Figure(data=[go.Bar(
                        x=[v["mean_abs_shap"] for v in top_features.values()],
                        y=list(top_features.keys()),
                        orientation="h",
                        marker_color="coral",
                    )])
                    fig.update_layout(height=350, title="Top Feature Contributions (|SHAP|)", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

            feature_importance = result.get("feature_importance", {})
            if feature_importance:
                st.markdown("#### Detailed Feature Contributions")
                detail_data = []
                for name, data in feature_importance.items():
                    detail_data.append({
                        "Feature": name,
                        "Mean SHAP": f"{data.get('mean_shap', 0):.4f}",
                        "|SHAP|": f"{data.get('mean_abs_shap', 0):.4f}",
                        "Direction": data.get("direction", "N/A"),
                        "Consistency": f"{data.get('consistency', 0):.2f}",
                    })
                st.dataframe(pd.DataFrame(detail_data), use_container_width=True)

    with tabs[1]:
        st.markdown("### Parking Risk Index (PRI)")
        st.markdown("**PRI Formula:** `0.4 × Illegal Parking + 0.3 × Density + 0.2 × Road Importance + 0.1 × Event Score`")

        if pri_scores:
            pri_data = []
            for cid, pri in pri_scores.items():
                profile = profiles.get(cid, {})
                pri_data.append({
                    "cluster_id": cid,
                    "area": profile.get("area", "Unknown"),
                    "road_type": profile.get("road_type", "Other"),
                    "pri_score": pri.get("pri_score", 0),
                    "illegal_parking": pri.get("illegal_component", 0),
                    "density": pri.get("density_component", 0),
                    "road_importance": pri.get("road_component", 0),
                    "event_score": pri.get("event_component", 0),
                    "dominant_factor": pri.get("dominant_factor", "Unknown"),
                })

            pri_df = pd.DataFrame(pri_data).sort_values("pri_score", ascending=False)

            fig = px.histogram(pri_df, x="pri_score", nbins=30, title="PRI Score Distribution", color_discrete_sequence=["#FF6B6B"])
            fig.update_layout(height=350, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Top 15 Hotspots by PRI")
            st.dataframe(pri_df.head(15), use_container_width=True)

            st.markdown("### PRI Component Breakdown (Top 10)")
            top10 = pri_df.head(10)

            fig = go.Figure()
            for component, color in [
                ("illegal_parking", "#FF6B6B"),
                ("density", "#4ECDC4"),
                ("road_importance", "#45B7D1"),
                ("event_score", "#96CEB4"),
            ]:
                fig.add_trace(go.Bar(
                    name=component.replace("_", " ").title(),
                    x=[f"C{cid}" for cid in top10["cluster_id"]],
                    y=top10[component],
                    marker_color=color,
                ))

            fig.update_layout(barmode="stack", height=400, title="PRI Component Breakdown", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("PRI scores not available.")
