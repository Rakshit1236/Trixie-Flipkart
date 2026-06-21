"""
Actions tab for Trixie-Flipkart.
Early Warning Timeline + Enhanced Action Recommendations.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium

BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946
WARNING_HORIZONS = ["15", "30", "60"]


def render_warning_timeline(warnings, profiles):
    if not warnings:
        st.info("No warnings available")
        return

    st.markdown("### Warning Progression Timeline")

    cols = st.columns(len(WARNING_HORIZONS))
    for i, horizon in enumerate(WARNING_HORIZONS):
        with cols[i]:
            horizon_warnings = warnings.get(horizon, [])
            high = sum(1 for w in horizon_warnings if w.get("threat_level") == "HIGH")
            medium = sum(1 for w in horizon_warnings if w.get("threat_level") == "MEDIUM")
            low = sum(1 for w in horizon_warnings if w.get("threat_level") == "LOW")

            if high > 0:
                color = "#FF0000"
                status = "CRITICAL"
                icon = "🔴"
            elif medium > 0:
                color = "#FF6600"
                status = "WARNING"
                icon = "🟠"
            else:
                color = "#00CC00"
                status = "STABLE"
                icon = "🟢"

            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: #1a1a2e; border-radius: 8px;">
                <div style="font-size: 32px;">{icon}</div>
                <div style="font-size: 20px; font-weight: bold; color: {color};">
                    +{horizon} min
                </div>
                <div style="font-size: 14px; color: #888; margin-top: 4px;">
                    {status}
                </div>
                <div style="font-size: 12px; color: #666; margin-top: 8px;">
                    🔴 {high} HIGH<br>
                    🟡 {medium} MED<br>
                    🟢 {low} LOW
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_enhanced_dispatch_card(rec, index):
    severity_config = {
        "CRITICAL": {"color": "#FF0000", "bg": "linear-gradient(135deg, #2d1b1b 0%, #1a1a2e 100%)", "border": "#FF0000"},
        "HIGH": {"color": "#FF6600", "bg": "linear-gradient(135deg, #2d241b 0%, #1a1a2e 100%)", "border": "#FF6600"},
        "MEDIUM": {"color": "#FFCC00", "bg": "linear-gradient(135deg, #2d2a1b 0%, #1a1a2e 100%)", "border": "#FFCC00"},
        "LOW": {"color": "#00CC00", "bg": "linear-gradient(135deg, #1b2d1b 0%, #1a1a2e 100%)", "border": "#00CC00"},
    }

    cfg = severity_config.get(rec.get("severity", "LOW"), severity_config["LOW"])
    chronic_tag = " ⚠️ CHRONIC" if rec.get("is_chronic", False) else ""
    resource = rec.get("resource_type", "Monitor")
    eta = rec.get("eta_minutes", 8)
    reduction = rec.get("expected_delay_reduction_pct", 20)

    st.markdown(f"""
    <div style="background: {cfg['bg']};
                border: 1px solid {cfg['border']};
                border-radius: 12px;
                padding: 16px;
                margin: 8px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 16px; font-weight: bold; color: #FAFAFA;">
                    {index}. Cluster {rec.get('cluster_id', '?')} — {rec.get('area', 'Unknown')}
                </span>
                <span style="font-size: 12px; color: {cfg['color']}; margin-left: 8px;">
                    {rec.get('severity', 'N/A')}{chronic_tag}
                </span>
            </div>
            <div style="font-size: 14px; color: #888;">
                Score: {rec.get('priority_score', 0):.0f}
            </div>
        </div>
        <div style="display: flex; gap: 16px; margin-top: 12px;">
            <div style="background: #0E1117; border-radius: 8px; padding: 10px 16px; text-align: center; min-width: 100px;">
                <div style="font-size: 11px; color: #888; text-transform: uppercase;">Officers</div>
                <div style="font-size: 20px; font-weight: bold; color: #4ECDC4;">{rec.get('officers_needed', 0)}</div>
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
    st.subheader("Actions")

    tabs = st.tabs(["Early Warnings", "Action Recommendations"])

    with tabs[0]:
        st.markdown("### Early Warning System")
        st.markdown("Micro-forecasts for the next 15, 30, and 60 minutes with threat levels.")

        if warnings:
            render_warning_timeline(warnings, profiles)
            st.divider()

            horizon = st.selectbox("Select Time Horizon", options=WARNING_HORIZONS, format_func=lambda x: f"{x} minutes", index=1)
            horizon_warnings = warnings.get(horizon, [])

            if horizon_warnings:
                st.markdown(f"### Threat Map — Next {horizon} Minutes")

                m = folium.Map(location=[BENGALURU_LAT, BENGALURU_LON], zoom_start=13, tiles="CartoDB dark_matter")
                threat_colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}

                for warning in horizon_warnings[:50]:
                    cid = str(warning.get("cluster_id", ""))
                    profile = profiles.get(cid, {})

                    folium.CircleMarker(
                        location=[
                            profile.get("centroid_lat", BENGALURU_LAT),
                            profile.get("centroid_lon", BENGALURU_LON),
                        ],
                        radius=max(5, min(15, warning.get("threat_score", 0) / 10)),
                        color=threat_colors.get(warning.get("threat_level", "LOW"), "blue"),
                        fill=True,
                        fill_color=threat_colors.get(warning.get("threat_level", "LOW"), "blue"),
                        fill_opacity=0.6,
                        popup=f"Cluster {cid}<br>{warning.get('area', '')}<br>Threat: {warning.get('threat_level', '')}",
                    ).add_to(m)

                st_folium(m, width=800, height=500)

                st.markdown("### Warning Details")
                df_warnings = pd.DataFrame(horizon_warnings[:20])
                display_cols = ["cluster_id", "area", "road_type", "threat_level", "threat_score", "is_chronic"]
                available_cols = [c for c in display_cols if c in df_warnings.columns]
                st.dataframe(df_warnings[available_cols], use_container_width=True)
            else:
                st.info(f"No warnings for {horizon} minute horizon")
        else:
            st.info("Early warnings not available.")

    with tabs[1]:
        st.markdown("### Dispatch Recommendations")
        st.markdown("Priority-ranked dispatch decisions with officer deployment and expected impact.")

        if recommendations:
            total_officers = sum(r.get("officers_needed", 0) for r in recommendations)
            critical = [r for r in recommendations if r.get("severity") == "CRITICAL"]
            high = [r for r in recommendations if r.get("severity") == "HIGH"]

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Officers", total_officers)
            with m2:
                st.metric("Critical Hotspots", len(critical))
            with m3:
                st.metric("High Hotspots", len(high))
            with m4:
                avg_reduction = sum(r.get("expected_delay_reduction_pct", 0) for r in recommendations) / max(len(recommendations), 1)
                st.metric("Avg Delay Reduction", f"{avg_reduction:.1f}%")

            st.markdown("### Dispatch Cards")
            for i, rec in enumerate(recommendations[:10]):
                render_enhanced_dispatch_card(rec, i + 1)

            st.markdown("### All Recommendations")
            df_recs = pd.DataFrame(recommendations)
            display_cols = [c for c in ["cluster_id", "area", "severity", "priority_score", "officers_needed", "expected_delay_reduction_pct", "resource_type", "eta_minutes", "timing"] if c in df_recs.columns]
            st.dataframe(df_recs[display_cols], use_container_width=True)
        else:
            st.info("Recommendations not available.")
