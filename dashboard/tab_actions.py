"""
Actions tab for Trixie-Flipkart.
Early Warning Timeline + 30-Day Forecast + Dispatch cards.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime

BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946
WARNING_HORIZONS = ["15", "30", "60"]


def _compute_dynamic_warnings(profiles, scores, hour=None):
    """Compute warnings at render time for a given hour."""
    from config import RUSH_HOUR_MULTIPLIER, CHRONIC_MULTIPLIER, TIME_OF_DAY_MULTIPLIERS, WARNING_HORIZONS, THREAT_LEVELS
    from src.utils import get_time_of_day

    if hour is None:
        hour = datetime.now().hour

    time_of_day = get_time_of_day(hour)
    is_rush = any(start <= hour < end for start, end in [(8, 11), (17, 21)])

    def _threat(priority, is_rush, is_chronic, tod, horizon):
        s = priority
        if is_rush:
            s *= RUSH_HOUR_MULTIPLIER
        if is_chronic:
            s *= CHRONIC_MULTIPLIER
        s *= TIME_OF_DAY_MULTIPLIERS.get(tod, 1.0)
        s *= 1 + (horizon / 60) * 0.003
        return min(s, 100)

    warnings = {}
    for horizon in WARNING_HORIZONS:
        hw = []
        for cid, profile in profiles.items():
            priority = scores.get(cid, {}).get("priority_score_normalized", 50)
            ts = _threat(priority, is_rush, profile.get("is_chronic", False), time_of_day, int(horizon))
            level = "HIGH" if ts >= THREAT_LEVELS["HIGH"] else ("MEDIUM" if ts >= THREAT_LEVELS["MEDIUM"] else "LOW")
            hw.append({
                "cluster_id": cid,
                "area": profile.get("area", "Unknown"),
                "road_type": profile.get("road_type", "Other"),
                "threat_score": round(ts, 1),
                "threat_level": level,
                "is_chronic": profile.get("is_chronic", False),
                "centroid_lat": profile.get("centroid_lat", 0),
                "centroid_lon": profile.get("centroid_lon", 0),
            })
        hw.sort(key=lambda x: x["threat_score"], reverse=True)
        warnings[str(horizon)] = hw

    return warnings, time_of_day, is_rush


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


def render_tab_actions(profiles, impact, scores, warnings, recommendations, forecast_summary=None, backend_url=None):
    st.subheader("Actions")

    tab_warn, tab_forecast, tab_dispatch = st.tabs(["Early Warnings", "30-Day Forecast", "Action Recommendations"])

    # ==================== EARLY WARNINGS (DYNAMIC) ====================
    with tab_warn:
        st.markdown("### Early Warning System")
        st.markdown("Live micro-forecasts for the next 15, 30, and 60 minutes. Recomputed on every page load.")

        hour_options = list(range(24))
        hour_labels = [f"{h:02d}:00" for h in range(24)]
        selected_hour = st.selectbox("Forecast Hour", options=hour_options, format_func=lambda x: hour_labels[x],
                                      index=datetime.now().hour, key="warn_hour")

        dynamic_warnings, time_of_day, is_rush = _compute_dynamic_warnings(profiles, scores, selected_hour)

        rush_label = "🔴 Rush Hour" if is_rush else "🟢 Off-Peak"
        st.info(f"**Time:** {selected_hour:02d}:00 | **Period:** {time_of_day.title()} | **Status:** {rush_label}")

        render_warning_timeline(dynamic_warnings)
        st.divider()

        horizon = st.selectbox("Time Horizon", options=WARNING_HORIZONS, format_func=lambda x: f"+{x} min", index=1, key="warn_horizon_action")
        hw = dynamic_warnings.get(horizon, [])

        if hw:
            high_count = sum(1 for w in hw if w["threat_level"] == "HIGH")
            med_count = sum(1 for w in hw if w["threat_level"] == "MEDIUM")
            low_count = len(hw) - high_count - med_count

            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 HIGH", high_count)
            c2.metric("🟡 MEDIUM", med_count)
            c3.metric("🟢 LOW", low_count)

            m = folium.Map(location=[BENGALURU_LAT, BENGALURU_LON], zoom_start=13, tiles="CartoDB dark_matter")
            tc = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}

            for w in hw[:50]:
                cid = str(w.get("cluster_id", ""))
                p = profiles.get(cid, profiles.get(int(cid), {}))
                lat = p.get("centroid_lat", BENGALURU_LAT)
                lon = p.get("centroid_lon", BENGALURU_LON)

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=max(5, min(15, w.get("threat_score", 0) / 10)),
                    color=tc.get(w.get("threat_level", "LOW"), "blue"),
                    fill=True, fill_color=tc.get(w.get("threat_level", "LOW"), "blue"), fill_opacity=0.6,
                    popup=f"C{cid} — {w.get('area', '')}<br>Threat: {w.get('threat_level', '')} ({w.get('threat_score', 0):.0f})",
                ).add_to(m)

            st_folium(m, width=800, height=500)

            df_w = pd.DataFrame(hw[:30])
            avail = [c for c in ["cluster_id", "area", "road_type", "threat_level", "threat_score", "is_chronic"] if c in df_w.columns]
            st.dataframe(df_w[avail], use_container_width=True)

            # Hourly progression chart
            st.markdown("#### Threat Score Progression (24h)")
            hourly_data = []
            for h in range(24):
                hw_h, _, _ = _compute_dynamic_warnings(profiles, scores, h)
                for w in hw_h.get("30", []):
                    hourly_data.append({"hour": h, "cluster_id": w["cluster_id"], "threat_score": w["threat_score"], "level": w["threat_level"]})
            df_hourly = pd.DataFrame(hourly_data)
            if not df_hourly.empty:
                agg = df_hourly.groupby("hour").agg(avg_threat=("threat_score", "mean"), high_count=("level", lambda x: (x == "HIGH").sum())).reset_index()
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=agg["hour"], y=agg["avg_threat"], name="Avg Threat", line=dict(color="#FF6B6B", width=2)), secondary_y=False)
                fig.add_trace(go.Bar(x=agg["hour"], y=agg["high_count"], name="HIGH Count", marker_color="#FF0000", opacity=0.4), secondary_y=True)
                fig.update_layout(title="24-Hour Threat Progression", template="plotly_dark", height=350, xaxis_title="Hour", hovermode="x unified")
                fig.update_yaxes(title_text="Avg Threat Score", secondary_y=False)
                fig.update_yaxes(title_text="# HIGH Threats", secondary_y=True)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No warnings for +{horizon} min at {selected_hour:02d}:00")

    # ==================== 30-DAY FORECAST ====================
    with tab_forecast:
        st.markdown("### 30-Day Violation Forecast")
        st.markdown("ML-powered daily violation predictions for the next 30 days per hotspot.")

        if forecast_summary:
            summary_df = pd.DataFrame(forecast_summary)

            # Citywide summary chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=summary_df["date"], y=summary_df["upper_total"], mode="lines",
                                     line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=summary_df["date"], y=summary_df["lower_total"], fill="tonexty",
                                     mode="lines", line=dict(width=0), fillcolor="rgba(78,205,196,0.15)",
                                     name="90% CI"))
            fig.add_trace(go.Scatter(x=summary_df["date"], y=summary_df["total_predicted"], mode="lines+markers",
                                     line=dict(color="#4ECDC4", width=2), name="Predicted Total", marker=dict(size=4)))
            fig.update_layout(title="Citywide 30-Day Forecast", template="plotly_dark", height=400,
                              xaxis_title="Date", yaxis_title="Total Violations", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Per-hotspot detail (lazy-loaded, only full 8MB forecast fetched on demand)
            @st.cache_data(ttl=600, show_spinner="Loading detailed forecast data...")
            def fetch_full_forecast(url):
                import requests as _req
                try:
                    r = _req.get(f"{url}/forecast", timeout=60)
                    r.raise_for_status()
                    return r.json().get("forecast", [])
                except Exception:
                    return []

            show_detail = st.checkbox("Show per-hotspot forecast details (downloads ~8MB)", value=False, key="fc_toggle")
            if show_detail and backend_url:
                forecast = fetch_full_forecast(backend_url)
                if forecast:
                    forecast_df = pd.DataFrame(forecast)
                    top_hotspots = forecast_df.groupby("cluster_id")["predicted_violations"].mean().nlargest(20).index.tolist()
                    selected_clusters = st.multiselect("Select Hotspots", options=top_hotspots, default=top_hotspots[:5], key="fc_clusters")

                    if selected_clusters:
                        fig2 = go.Figure()
                        colors = px.colors.qualitative.Set2
                        for i, cid in enumerate(selected_clusters):
                            subset = forecast_df[forecast_df["cluster_id"] == cid]
                            area = subset["area"].iloc[0] if len(subset) > 0 else f"Cluster {cid}"
                            fig2.add_trace(go.Scatter(x=subset["date"], y=subset["predicted_violations"],
                                                      mode="lines+markers", name=f"C{cid} — {area}",
                                                      line=dict(color=colors[i % len(colors)], width=2),
                                                      marker=dict(size=3)))
                        fig2.update_layout(title="Per-Hotspot 30-Day Forecast", template="plotly_dark", height=400,
                                           xaxis_title="Date", yaxis_title="Predicted Violations", hovermode="x unified")
                        st.plotly_chart(fig2, use_container_width=True)

                    st.markdown("#### Top 20 Hotspots — Average Daily Forecast")
                    agg_fc = forecast_df.groupby(["cluster_id", "area", "road_type"]).agg(
                        avg_predicted=("predicted_violations", "mean"),
                        max_predicted=("predicted_violations", "max"),
                        avg_lower=("lower_bound", "mean"),
                        avg_upper=("upper_bound", "mean"),
                    ).reset_index().sort_values("avg_predicted", ascending=False)
                    st.dataframe(agg_fc.head(20), use_container_width=True)
                else:
                    st.warning("Could not load detailed forecast data.")
        else:
            st.info("No 30-day forecast available. Re-run the pipeline to generate forecasts.")

    # ==================== DISPATCH RECOMMENDATIONS ====================
    with tab_dispatch:
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
