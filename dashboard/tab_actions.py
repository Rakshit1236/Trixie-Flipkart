"""
Actions tab for Trixie-Flipkart.
Early Warning Timeline + Enhanced Action Recommendations with ETA and resource type.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BENGALURU_LAT, BENGALURU_LON, WARNING_HORIZONS, RESOURCE_TYPE, ETA_PER_OFFICER_MIN


def render_warning_timeline(warnings, profiles):
    """Render early warning timeline with visual progression."""
    if not warnings:
        st.info("No warnings available")
        return

    st.markdown("### 🚨 Warning Progression Timeline")

    # Get threat counts for each horizon
    horizon_data = {}
    for horizon in WARNING_HORIZONS:
        horizon_warnings = warnings.get(horizon, [])
        high = sum(1 for w in horizon_warnings if w["threat_level"] == "HIGH")
        medium = sum(1 for w in horizon_warnings if w["threat_level"] == "MEDIUM")
        low = sum(1 for w in horizon_warnings if w["threat_level"] == "LOW")
        horizon_data[horizon] = {"high": high, "medium": medium, "low": low}

    # Visual timeline
    st.markdown("""
    <div style="display: flex; justify-content: space-around; align-items: center;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px; padding: 20px; margin: 10px 0;">
    """, unsafe_allow_html=True)

    cols = st.columns(len(WARNING_HORIZONS))
    for i, horizon in enumerate(WARNING_HORIZONS):
        with cols[i]:
            data = horizon_data[horizon]
            total_threats = data["high"] + data["medium"]

            if data["high"] > 0:
                color = "#FF0000"
                status = "CRITICAL"
                icon = "🔴"
            elif data["medium"] > 0:
                color = "#FF6600"
                status = "WARNING"
                icon = "🟠"
            else:
                color = "#00CC00"
                status = "STABLE"
                icon = "🟢"

            st.markdown(f"""
            <div style="text-align: center; padding: 15px;">
                <div style="font-size: 32px;">{icon}</div>
                <div style="font-size: 20px; font-weight: bold; color: {color};">
                    +{horizon} min
                </div>
                <div style="font-size: 14px; color: #888; margin-top: 4px;">
                    {status}
                </div>
                <div style="font-size: 12px; color: #666; margin-top: 8px;">
                    🔴 {data['high']} HIGH<br>
                    🟡 {data['medium']} MED<br>
                    🟢 {data['low']} LOW
                </div>
            </div>
            """, unsafe_allow_html=True)

            if i < len(WARNING_HORIZONS) - 1:
                st.markdown("""<div style="text-align: center; font-size: 24px; color: #666;">→</div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_enhanced_dispatch_card(rec, index):
    """Render an enhanced dispatch card with ETA, resource type, and expected impact."""
    severity_config = {
        "CRITICAL": {"color": "#FF0000", "bg": "linear-gradient(135deg, #2d1b1b 0%, #1a1a2e 100%)", "border": "#FF0000"},
        "HIGH": {"color": "#FF6600", "bg": "linear-gradient(135deg, #2d241b 0%, #1a1a2e 100%)", "border": "#FF6600"},
        "MEDIUM": {"color": "#FFCC00", "bg": "linear-gradient(135deg, #2d2a1b 0%, #1a1a2e 100%)", "border": "#FFCC00"},
        "LOW": {"color": "#00CC00", "bg": "linear-gradient(135deg, #1b2d1b 0%, #1a1a2e 100%)", "border": "#00CC00"},
    }

    config = severity_config.get(rec["severity"], severity_config["LOW"])
    chronic_tag = " ⚠️ CHRONIC" if rec.get("is_chronic", False) else ""
    resource = rec.get("resource_type", "Monitor")
    eta = rec.get("eta_minutes", 8)
    reduction = rec.get("expected_delay_reduction_pct", 20)

    st.markdown(f"""
    <div style="background: {config['bg']};
                border: 1px solid {config['border']};
                border-radius: 12px;
                padding: 16px;
                margin: 8px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 16px; font-weight: bold; color: #FAFAFA;">
                    {index}. Cluster {rec['cluster_id']} — {rec['area']}
                </span>
                <span style="font-size: 12px; color: {config['color']}; margin-left: 8px;">
                    {rec['severity']}{chronic_tag}
                </span>
            </div>
            <div style="font-size: 14px; color: #888;">
                Score: {rec['priority_score']:.0f}
            </div>
        </div>
        <div style="display: flex; gap: 16px; margin-top: 12px;">
            <div style="background: #0E1117; border-radius: 8px; padding: 10px 16px; text-align: center; min-width: 100px;">
                <div style="font-size: 11px; color: #888; text-transform: uppercase;">Officers</div>
                <div style="font-size: 20px; font-weight: bold; color: #4ECDC4;">{rec['officers_needed']}</div>
            </div>
            <div style="background: #0E1117; border-radius: 8px; padding: 10px 16px; text-align: center; min-width: 100px;">
                <div style="font-size: 11px; color: #888; text-transform: uppercase;">ETA</div>
                <div style="font-size: 20px; font-weight: bold; color: #FF6B6B;">{eta} min</div>
            </div>
            <div style="background: #0E1117; border-radius: 8px; padding: 10px 16px; text-align: center; min-width: 100px;">
                <div style="font-size: 11px; color: #888; text-transform: uppercase;">Reduce</div>
                <div style="font-size: 20px; font-weight: bold; color: #45B7D1;">{reduction}%</div>
            </div>
            <div style="background: #0E1117; border-radius: 8px; padding: 10px 16px; text-align: center; min-width: 120px;">
                <div style="font-size: 11px; color: #888; text-transform: uppercase;">Action</div>
                <div style="font-size: 13px; font-weight: bold; color: #96CEB4;">{resource}</div>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 12px; color: #666;">
            📍 {rec.get('road_type', 'Unknown')} | ⏰ {rec.get('timing', 'ASAP')} | 🚗 {rec.get('daily_rate', 0):.1f} violations/day
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_tab_actions(profiles, impact, scores, warnings, recommendations):
    st.subheader("⚡ Actions")

    tabs = st.tabs(["Early Warnings", "Action Recommendations"])

    # ==================== EARLY WARNINGS ====================
    with tabs[0]:
        st.markdown("### 🚨 Early Warning System")
        st.markdown("Micro-forecasts for the next 15, 30, and 60 minutes with threat levels.")

        if warnings:
            # ==================== WARNING TIMELINE ====================
            render_warning_timeline(warnings, profiles)

            st.divider()

            # Horizon selector
            horizon = st.selectbox(
                "Select Time Horizon",
                options=WARNING_HORIZONS,
                format_func=lambda x: f"{x} minutes",
                index=1,
            )

            horizon_warnings = warnings.get(horizon, [])

            if horizon_warnings:
                # Warning map
                st.markdown(f"### Threat Map — Next {horizon} Minutes")

                m = folium.Map(
                    location=[BENGALURU_LAT, BENGALURU_LON],
                    zoom_start=13,
                    tiles="CartoDB dark_matter",
                )

                threat_colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}

                for warning in horizon_warnings[:50]:
                    cid = warning["cluster_id"]
                    profile = profiles.get(cid, {})

                    folium.CircleMarker(
                        location=[
                            profile.get("centroid_lat", BENGALURU_LAT),
                            profile.get("centroid_lon", BENGALURU_LON),
                        ],
                        radius=max(5, min(15, warning["threat_score"] / 10)),
                        color=threat_colors.get(warning["threat_level"], "blue"),
                        fill=True,
                        fill_color=threat_colors.get(warning["threat_level"], "blue"),
                        fill_opacity=0.6,
                        popup=folium.Popup(
                            f"<b>Cluster {cid}</b><br>"
                            f"Area: {warning['area']}<br>"
                            f"Threat: {warning['threat_level']}<br>"
                            f"Score: {warning['threat_score']:.1f}<br>"
                            f"Chronic: {'Yes' if warning['is_chronic'] else 'No'}",
                            max_width=200,
                        ),
                    ).add_to(m)

                st_folium(m, width=800, height=500)

                # Warning table
                st.markdown("### Warning Details")
                df_warnings = pd.DataFrame(horizon_warnings[:20])
                display_cols = ["cluster_id", "area", "road_type", "threat_level",
                               "threat_score", "is_chronic"]
                available_cols = [c for c in display_cols if c in df_warnings.columns]

                st.dataframe(
                    df_warnings[available_cols],
                    use_container_width=True,
                )
            else:
                st.info(f"No warnings available for {horizon} minute horizon")
        else:
            st.info("Early warnings not available. Run the full pipeline to generate warnings.")

    # ==================== ACTION RECOMMENDATIONS ====================
    with tabs[1]:
        st.markdown("### 📋 Dispatch Recommendations")
        st.markdown("Priority-ranked dispatch decisions with officer deployment and expected impact.")

        if recommendations:
            # Summary
            total_officers = sum(r["officers_needed"] for r in recommendations)
            critical = [r for r in recommendations if r["severity"] == "CRITICAL"]
            high = [r for r in recommendations if r["severity"] == "HIGH"]

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Officers", total_officers)
            with m2:
                st.metric("Critical Hotspots", len(critical))
            with m3:
                st.metric("High Hotspots", len(high))
            with m4:
                avg_reduction = sum(r["expected_delay_reduction_pct"] for r in recommendations) / len(recommendations)
                st.metric("Avg Delay Reduction", f"{avg_reduction:.1f}%")

            # ==================== ENHANCED DISPATCH CARDS ====================
            st.markdown("### Dispatch Cards")

            for i, rec in enumerate(recommendations[:10]):
                render_enhanced_dispatch_card(rec, i + 1)

            # Full recommendations table
            st.markdown("### All Recommendations")

            df_recs = pd.DataFrame(recommendations)
            display_cols = ["cluster_id", "area", "severity", "priority_score",
                           "officers_needed", "expected_delay_reduction_pct",
                           "resource_type", "eta_minutes", "timing"]
            available_cols = [c for c in display_cols if c in df_recs.columns]

            st.dataframe(
                df_recs[available_cols].style.background_gradient(
                    subset=["priority_score"],
                    cmap="RdYlGn_r",
                ),
                use_container_width=True,
            )
        else:
            st.info("Recommendations not available. Run the full pipeline to generate recommendations.")
