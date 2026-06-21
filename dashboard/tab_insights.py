"""
Insights tab for Trixie-Flipkart.
XAI breakdown with root cause attribution cards + Parking Risk Index ranking.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PRI_WEIGHTS


def render_root_cause_cards(explanation, cluster_id):
    """Render human-readable root cause attribution cards with progress bars."""
    if not explanation:
        st.info("No XAI explanation available for this hotspot")
        return

    dominant = explanation.get("dominant_factor", "Unknown")
    pct_contributions = explanation.get("percentage_contributions", {})

    if not pct_contributions:
        st.info("No contribution data available")
        return

    # Dominant factor highlight
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

    # Factor breakdown with progress bars
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
                <div style="background: {color}; width: {bar_width}%; height: 100%; border-radius: 6px;
                            transition: width 0.5s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_tab_insights(profiles, impact, scores, pri_scores, ranked):
    st.subheader("💡 Insights")

    tabs = st.tabs(["Explainable AI (XAI)", "Parking Risk Index (PRI)"])

    # ==================== XAI TAB ====================
    with tabs[0]:
        st.markdown("### 🔍 Root Cause Analysis")
        st.markdown("""
        Select a hotspot to see the root cause breakdown.
        Contributions are based on true SHAP values from the trained model.
        """)

        hotspot_options = [
            f"Cluster {cid} ({profiles[cid].get('area', 'Unknown')})"
            for cid in sorted(profiles.keys())
        ]
        hotspot_selection = st.selectbox("Select Hotspot", hotspot_options, index=0, key="xai_hotspot")

        cid = int(hotspot_selection.split("(")[0].replace("Cluster ", "").strip())

        if "xai_explanations" in st.session_state:
            explanations = st.session_state.xai_explanations

            if cid in explanations:
                explanation = explanations[cid]

                # ==================== ROOT CAUSE CARDS ====================
                render_root_cause_cards(explanation, cid)

                st.divider()

                # ==================== VISUAL CHARTS ====================
                col1, col2 = st.columns(2)

                with col1:
                    # Pie chart
                    pct_contributions = explanation.get("percentage_contributions", {})
                    if pct_contributions:
                        fig = go.Figure(data=[go.Pie(
                            labels=list(pct_contributions.keys()),
                            values=list(pct_contributions.values()),
                            hole=0.4,
                            marker_colors=px.colors.qualitative.Set2,
                        )])
                        fig.update_layout(
                            height=350,
                            title="Root Cause Distribution",
                            template="plotly_dark",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Feature importance bar chart
                    feature_importance = explanation.get("feature_importance", {})
                    if feature_importance:
                        top_features = dict(list(feature_importance.items())[:8])

                        fig = go.Figure(data=[go.Bar(
                            x=[v["mean_abs_shap"] for v in top_features.values()],
                            y=list(top_features.keys()),
                            orientation="h",
                            marker_color="coral",
                        )])
                        fig.update_layout(
                            height=350,
                            title="Top Feature Contributions (|SHAP|)",
                            xaxis_title="Mean |SHAP Value|",
                            template="plotly_dark",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Detailed breakdown table
                st.markdown("#### Detailed Feature Contributions")

                feature_importance = explanation.get("feature_importance", {})
                detail_data = []
                for name, data in feature_importance.items():
                    detail_data.append({
                        "Feature": name,
                        "Mean SHAP": f"{data['mean_shap']:.4f}",
                        "|SHAP|": f"{data['mean_abs_shap']:.4f}",
                        "Direction": data["direction"],
                        "Consistency": f"{data['consistency']:.2f}",
                    })

                st.dataframe(pd.DataFrame(detail_data), use_container_width=True)

            else:
                st.warning("No XAI explanation available for this hotspot. Run the full pipeline with XAI enabled.")

        else:
            st.info("XAI explanations not loaded. Run the full pipeline to enable SHAP-based explanations.")

    # ==================== PRI TAB ====================
    with tabs[1]:
        st.markdown("### 📊 Parking Risk Index (PRI)")

        st.markdown("""
        **PRI Formula:** `0.4 × Illegal Parking + 0.3 × Density + 0.2 × Road Importance + 0.1 × Event Score`

        All components normalized to 0-100.
        """)

        if pri_scores:
            pri_data = []
            for cid, pri in pri_scores.items():
                profile = profiles.get(cid, {})
                pri_data.append({
                    "cluster_id": cid,
                    "area": profile.get("area", "Unknown"),
                    "road_type": profile.get("road_type", "Other"),
                    "pri_score": pri["pri_score"],
                    "illegal_parking": pri["illegal_component"],
                    "density": pri["density_component"],
                    "road_importance": pri["road_component"],
                    "event_score": pri["event_component"],
                    "dominant_factor": pri["dominant_factor"],
                })

            pri_df = pd.DataFrame(pri_data).sort_values("pri_score", ascending=False)

            fig = px.histogram(
                pri_df, x="pri_score", nbins=30,
                title="PRI Score Distribution",
                labels={"pri_score": "PRI Score", "count": "Count"},
                color_discrete_sequence=["#FF6B6B"],
            )
            fig.update_layout(height=350, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Top 15 Hotspots by PRI")
            top15 = pri_df.head(15)

            st.dataframe(
                top15.style.background_gradient(subset=["pri_score"], cmap="RdYlGn_r"),
                use_container_width=True,
            )

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

            fig.update_layout(
                barmode="stack", height=400,
                title="PRI Component Breakdown",
                xaxis_title="Cluster", yaxis_title="Component Score",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("PRI scores not available. Run the full pipeline to compute PRI.")
