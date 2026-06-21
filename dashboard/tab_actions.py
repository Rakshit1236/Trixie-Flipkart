"""
Actions tab for Trixie-Flipkart.
Early Warning Timeline + Enhanced Dispatch cards.
"""
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946
WARNING_HORIZONS = ["15", "30", "60"]


def render_warning_timeline(warnings):
    if not warnings:
        st.info("No warnings available")
        return

    st.markdown("### Warning Progression Timeline")

    cols = st.columns(len(WARNING_HORIZONS))
    for i, horizon in enumerate(WARNING_HORIZONS):
        with cols[i]:
            hw = warnings.get(horizon, [])
            high = sum(1 for w in hw if w.get("threat_level") == "HIGH")
            medium = sum(1 for w in hw if w.get("threat_level") == "MEDIUM")
            low = sum(1 for w in hw if w.get("threat_level") == "LOW")

            if high > 0:
                color, status, icon = "#FF0000", "CRITICAL", "🔴"
            elif medium > 0:
                color, status, icon = "#FF6600", "WARNING", "🟠"
            else:
                color, status, icon = "#00CC00", "STABLE", "🟢"

            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: #1a1a2e; border-radius: 8px;">
                <div style="font-size: 32px;">{icon}</div>
                <div style="font-size: 20px; font-weight: bold; color: {color};">+{horizon} min</div>
                <div style="font-size: 14px; color: #888; margin-top: 4px;">{status}</div>
                <div style="font-size: 12px; color: #666; margin-top: 8px;">
                    🔴 {high} HIGH<br>🟡 {medium} MED<br>🟢 {low} LOW
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_dispatch_card(rec, index):
    sev_cfg = {
        "CRITICAL": {"color": "#FF0000", "bg": "linear-gradient(135deg, #2d1b1b 0%, #1a1a2e 100%)", "border": "#FF0000"},
        "HIGH": {"color": "#FF6600", "bg": "linear-gradient(135deg, #2d241b 0%, #1a1a2e 100%)", "border": "#FF6600"},
        "MEDIUM": {"color": "#FFCC00", "bg": "linear-gradient(135deg, #2d2a1b 0%, #1a1a2e 100%)", "border": "#FFCC00"},
        "LOW": {"color": "#00CC00", "bg": "linear-gradient(135deg, #1b2d1b 0%, #1a1a2e 100%)", "border": "#00CC00"},
    }
    cfg = sev_cfg.get(rec.get("severity", "LOW"), sev_cfg["LOW"])
    chronic = " ⚠️ CHRONIC" if rec.get("is_chronic") else ""

    st.markdown(f"""
    <div style="background: {cfg['bg']}; border: 1px solid {cfg['border']}; border-radius: 12px; padding: 16px; margin: 8px 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-size: 16px; font-weight: bold; color: #FAFAFA;">
                    {index}. C{rec.get('cluster_id', '?')} — {rec.get('area', 'Unknown')}
                </span>
                <span style="font-size: 12px; color: {cfg['color']}; margin-left: 8px;">{rec.get('severity', '')}{chronic}</span>
            </div>
            <div style="font-size: 14px; color: #888;">Score: {rec.get('priority_score', 0):.0f}</div>
        </div>
        <div style="display: flex; gap: 16px; margin-top: 12px;">
            <div style="background: #0E1117; border-radius: 8px; padding: 10px 16px; text-align: center; min-width: 100px;">
                <div style="font-size: 11px; color: #888;">OFFICERS</div>
                <div style="font-size: 20px; font-weight: bold; color: #4ECDC4;">{rec.get('officers_needed', 0)}</div>
            </div>
            <div style="background: #0E1117; border-radius: 8px; padding: 10px 16px; text-align: center; min-width: 100px;">
                <div style="font-size: 11px; color: #888;">ETA</div>
                <div style="font-size: 20px; font-weight: bold; color: #FF6B6B;">{rec.get('eta_minutes', 8)} min</div>
            </div>
            <div style="background: #0E1117; border-radius: 8px; padding: 10px 16px; text-align: center; min-width: 100px;">
                <div style="font-size: 11px; color: #888;">REDUCE</div>
                <div style="font-size: 20px; font-weight: bold; color: #45B7D1;">{rec.get('expected_delay_reduction_pct', 0)}%</div>
            </div>
            <div style="background: #0E1117; border-radius: 8px; padding: 10px 16px; text-align: center; min-width: 120px;">
                <div style="font-size: 11px; color: #888;">ACTION</div>
                <div style="font-size: 13px; font-weight: bold; color: #96CEB4;">{rec.get('resource_type', 'Monitor')}</div>
            </div>
        </div>
        <div style="margin-top: 10px; font-size: 12px; color: #666;">
            📍 {rec.get('road_type', 'Unknown')} | ⏰ {rec.get('timing', 'ASAP')} | 🚗 {rec.get('daily_rate', 0):.1f} violations/day
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_tab_actions(profiles, impact, scores, warnings, recommendations):
    st.subheader("Actions")

    tab1, tab2 = st.tabs(["Early Warnings", "Action Recommendations"])

    with tab1:
        st.markdown("### Early Warning System")
        st.markdown("Micro-forecasts for the next 15, 30, and 60 minutes with threat levels.")

        if warnings:
            render_warning_timeline(warnings)
            st.divider()

            horizon = st.selectbox("Time Horizon", options=WARNING_HORIZONS, format_func=lambda x: f"+{x} min", index=1, key="warn_horizon")
            hw = warnings.get(horizon, [])

            if hw:
                m = folium.Map(location=[BENGALURU_LAT, BENGALURU_LON], zoom_start=13, tiles="CartoDB dark_matter")
                tc = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}

                for w in hw[:50]:
                    cid = str(w.get("cluster_id", ""))
                    p = profiles.get(cid, {})
                    lat = p.get("centroid_lat", BENGALURU_LAT)
                    lon = p.get("centroid_lon", BENGALURU_LON)

                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=max(5, min(15, w.get("threat_score", 0) / 10)),
                        color=tc.get(w.get("threat_level", "LOW"), "blue"),
                        fill=True, fill_color=tc.get(w.get("threat_level", "LOW"), "blue"), fill_opacity=0.6,
                        popup=f"C{cid} — {w.get('area', '')}<br>Threat: {w.get('threat_level', '')}",
                    ).add_to(m)

                st_folium(m, width=800, height=500)

                df_w = pd.DataFrame(hw[:20])
                avail = [c for c in ["cluster_id", "area", "road_type", "threat_level", "threat_score", "is_chronic"] if c in df_w.columns]
                st.dataframe(df_w[avail], use_container_width=True)
            else:
                st.info(f"No warnings for +{horizon} min")
        else:
            st.info("No warnings available.")

    with tab2:
        st.markdown("### Dispatch Recommendations")

        if recommendations:
            total_off = sum(r.get("officers_needed", 0) for r in recommendations)
            crit = [r for r in recommendations if r.get("severity") == "CRITICAL"]
            high = [r for r in recommendations if r.get("severity") == "HIGH"]
            avg_red = sum(r.get("expected_delay_reduction_pct", 0) for r in recommendations) / max(len(recommendations), 1)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Officers", total_off)
            with m2:
                st.metric("Critical", len(crit))
            with m3:
                st.metric("High", len(high))
            with m4:
                st.metric("Avg Delay Reduction", f"{avg_red:.1f}%")

            for i, rec in enumerate(recommendations[:10]):
                render_dispatch_card(rec, i + 1)

            df_r = pd.DataFrame(recommendations)
            avail = [c for c in ["cluster_id", "area", "severity", "priority_score", "officers_needed", "expected_delay_reduction_pct", "resource_type", "eta_minutes", "timing"] if c in df_r.columns]
            if avail:
                st.dataframe(df_r[avail], use_container_width=True)
        else:
            st.info("No recommendations available.")
