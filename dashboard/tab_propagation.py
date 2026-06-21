"""
Congestion propagation tab for Trixie-Flipkart.
Calls backend API for propagation simulation.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import requests
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "https://rakshit1236-trixie-backend.hf.space")
BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946


def api_post(endpoint, payload):
    try:
        r = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def render_cascade_timeline(timeline, source_area):
    if not timeline:
        st.info("No propagation data available")
        return

    st.markdown(f"### Cascade Timeline — From {source_area}")

    severity_colors = {
        "CRITICAL": "#FF0000",
        "HIGH": "#FF6600",
        "MEDIUM": "#FFCC00",
        "LOW": "#00CC00",
    }
    severity_icons = {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢",
    }

    for i, entry in enumerate(timeline):
        color = severity_colors.get(entry.get("severity", "LOW"), "#666666")
        icon = severity_icons.get(entry.get("severity", "LOW"), "⚪")
        time_str = entry.get("time_str", f"+{entry['minute']} min")

        with st.container():
            cols = st.columns([1, 4])
            with cols[0]:
                st.markdown(f"""
                <div style="text-align:center; padding:10px;">
                    <div style="font-size:24px;">{icon}</div>
                    <div style="font-size:14px; font-weight:bold; color:{color};">
                        ⏱ {time_str}
                    </div>
                    <div style="font-size:12px; color:#888;">+{entry['minute']} min</div>
                </div>
                """, unsafe_allow_html=True)

            with cols[1]:
                speed_drop = entry.get("speed_drop_pct", 0)
                queue = entry.get("queue_length", 0)
                severity = entry.get("severity", "LOW")
                st.markdown(f"""
                <div style="background:#1E1E1E; border-left:4px solid {color}; padding:12px; border-radius:4px; margin:4px 0;">
                    <div style="font-size:16px; font-weight:bold; color:{color};">
                        {entry.get('area', 'Unknown')}
                    </div>
                    <div style="font-size:13px; color:#CCC; margin-top:4px;">
                        Cluster {entry.get('cluster_id', '?')} — {entry.get('road_type', 'Unknown')}
                    </div>
                    <div style="display:flex; gap:20px; margin-top:8px;">
                        <span style="color:#FF6B6B;">Speed: -{speed_drop:.0f}%</span>
                        <span style="color:#4ECDC4;">Queue: {queue:.0f} vehicles</span>
                        <span style="color:{color};">Severity: {severity}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if i < len(timeline) - 1:
                travel_time = timeline[i + 1]["minute"] - entry["minute"]
                st.markdown(f"""
                <div style="text-align:center; padding:2px; color:#666;">
                    ↓ ({travel_time} min travel time) ↓
                </div>
                """, unsafe_allow_html=True)


def render_tab_propagation(profiles, impact, scores):
    st.subheader("Congestion Propagation")

    st.markdown("Select a source hotspot and see how congestion cascades to neighboring areas over time.")

    col1, col2, col3 = st.columns(3)

    with col1:
        hotspot_options = [
            f"Cluster {cid} ({profiles[cid].get('area', 'Unknown')})"
            for cid in sorted(int(k) for k in profiles.keys())
        ]
        hotspot_selection = st.selectbox("Source Hotspot", hotspot_options, index=0)
        source_cid = int(hotspot_selection.split("(")[0].replace("Cluster ", "").strip())

    with col2:
        horizon = st.slider("Forecast Horizon (minutes)", 15, 120, 60, 15)

    with col3:
        start_hour = st.slider("Start Hour", 0, 23, 8)

    if st.button("Simulate Propagation", type="primary", use_container_width=True):
        with st.spinner("Simulating congestion cascade..."):
            result = api_post("/propagation", {
                "source_cluster_id": source_cid,
                "start_hour": start_hour,
                "horizon_minutes": horizon,
            })

        if "error" in result:
            st.error(f"Propagation failed: {result['error']}")
            return

        st.markdown("### Propagation Summary")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Source Hotspot", result.get("source_area", "Unknown"))
        with m2:
            st.metric("Affected Hotspots", result.get("n_affected", 0))
        with m3:
            st.metric("Time Horizon", f"{horizon} min")
        with m4:
            st.metric("Start Hour", f"{start_hour}:00")

        st.divider()
        timeline = result.get("timeline", [])
        render_cascade_timeline(timeline, result.get("source_area", "Unknown"))

        if timeline:
            st.divider()
            st.markdown("### Congestion Cascade Over Time")

            df_timeline = pd.DataFrame(timeline)

            fig = px.scatter(
                df_timeline,
                x="minute",
                y="speed_drop_pct",
                size="queue_length",
                color="severity",
                hover_data=["cluster_id", "area"],
                color_discrete_map={
                    "CRITICAL": "#FF0000",
                    "HIGH": "#FF6600",
                    "MEDIUM": "#FFCC00",
                    "LOW": "#00CC00",
                },
                title="Speed Drop vs Time",
            )
            fig.update_layout(height=400, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Affected Hotspots")
            display_cols = ["minute", "time_str", "cluster_id", "area", "road_type", "severity", "speed_drop_pct", "queue_length"]
            available_cols = [c for c in display_cols if c in df_timeline.columns]
            st.dataframe(df_timeline[available_cols], use_container_width=True)

            st.divider()
            st.markdown("### Propagation Map")

            m = folium.Map(location=[BENGALURU_LAT, BENGALURU_LON], zoom_start=13, tiles="CartoDB dark_matter")

            severity_folium = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"}

            for entry in timeline:
                cid = str(entry.get("cluster_id", ""))
                profile = profiles.get(cid, {})
                lat = profile.get("centroid_lat", BENGALURU_LAT)
                lon = profile.get("centroid_lon", BENGALURU_LON)

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=max(5, min(15, entry.get("speed_drop_pct", 0) / 5)),
                    color=severity_folium.get(entry.get("severity", "LOW"), "blue"),
                    fill=True,
                    fill_color=severity_folium.get(entry.get("severity", "LOW"), "blue"),
                    fill_opacity=0.6,
                    popup=f"Cluster {cid}<br>Speed: -{entry.get('speed_drop_pct', 0):.1f}%<br>Queue: {entry.get('queue_length', 0):.0f}",
                ).add_to(m)

            st_folium(m, width=800, height=500)
    else:
        st.info("Click **Simulate Propagation** to see the congestion cascade")
