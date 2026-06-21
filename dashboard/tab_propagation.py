"""
Congestion propagation tab for Trixie-Flipkart.
Step-by-step cascade timeline + map visualization.
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
from config import BENGALURU_LAT, BENGALURU_LON
from src.analytics import CongestionPropagation


def render_cascade_timeline(timeline, source_area):
    """Render step-by-step cascade timeline with colored dots and arrows."""
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
        color = severity_colors.get(entry["severity"], "#666666")
        icon = severity_icons.get(entry["severity"], "⚪")

        # Time and area header
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
                st.markdown(f"""
                <div style="background:#1E1E1E; border-left:4px solid {color}; padding:12px; border-radius:4px; margin:4px 0;">
                    <div style="font-size:16px; font-weight:bold; color:{color};">
                        {entry['area']}
                    </div>
                    <div style="font-size:13px; color:#CCC; margin-top:4px;">
                        Cluster {entry['cluster_id']} — {entry.get('road_type', 'Unknown')}
                    </div>
                    <div style="display:flex; gap:20px; margin-top:8px;">
                        <span style="color:#FF6B6B;">Speed: -{entry['speed_drop_pct']:.0f}%</span>
                        <span style="color:#4ECDC4;">Queue: {entry['queue_length']:.0f} vehicles</span>
                        <span style="color:#45B7D1;">Delay: {entry['delay_seconds']:.0f}s</span>
                        <span style="color:#96CEB4;">Demand: {entry['demand_vph']:.0f} vph</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Arrow between steps
            if i < len(timeline) - 1:
                next_entry = timeline[i + 1]
                travel_time = next_entry["minute"] - entry["minute"]
                st.markdown(f"""
                <div style="text-align:center; padding:2px; color:#666;">
                    ↓ ({travel_time} min travel time) ↓
                </div>
                """, unsafe_allow_html=True)


def render_tab_propagation(profiles, impact, scores):
    st.subheader("🌊 Congestion Propagation")

    st.markdown("""
    Select a source hotspot and see how congestion cascades to neighboring areas over time.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        hotspot_options = [
            f"Cluster {cid} ({profiles[cid].get('area', 'Unknown')})"
            for cid in sorted(profiles.keys())
        ]
        hotspot_selection = st.selectbox("Source Hotspot", hotspot_options, index=0)
        source_cid = int(hotspot_selection.split("(")[0].replace("Cluster ", "").strip())

    with col2:
        horizon = st.slider("Forecast Horizon (minutes)", 15, 120, 60, 15)

    with col3:
        start_hour = st.slider("Start Hour", 0, 23, 8)

    adjacency = {cid: p.get("neighbors", []) for cid, p in profiles.items()}
    propagator = CongestionPropagation(profiles, adjacency)

    result = propagator.simulate_propagation(source_cid, start_hour, horizon)

    if "error" in result:
        st.error(result["error"])
        return

    # Summary metrics
    st.markdown("### Propagation Summary")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Source Hotspot", result["source_area"])
    with m2:
        st.metric("Affected Hotspots", result["n_affected"])
    with m3:
        st.metric("Time Horizon", f"{horizon} min")
    with m4:
        st.metric("Start Hour", f"{start_hour}:00")

    # ==================== CASCADE TIMELINE ====================
    st.divider()
    render_cascade_timeline(result["timeline"], result["source_area"])

    # ==================== SCATTER PLOT ====================
    st.divider()
    st.markdown("### Congestion Cascade Over Time")

    timeline = result["timeline"]
    if timeline:
        df_timeline = pd.DataFrame(timeline)

        fig = px.scatter(
            df_timeline,
            x="minute",
            y="speed_drop_pct",
            size="queue_length",
            color="severity",
            hover_data=["cluster_id", "area", "demand_vph", "delay_seconds"],
            color_discrete_map={
                "CRITICAL": "#FF0000",
                "HIGH": "#FF6600",
                "MEDIUM": "#FFCC00",
                "LOW": "#00CC00",
            },
            title="Speed Drop vs Time",
            labels={
                "minute": "Time (minutes)",
                "speed_drop_pct": "Speed Drop (%)",
                "queue_length": "Queue Length",
            },
        )
        fig.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Affected hotspots table
        st.markdown("### Affected Hotspots")
        display_cols = ["minute", "time_str", "cluster_id", "area", "road_type", "severity",
                       "speed_drop_pct", "queue_length", "delay_seconds"]
        available_cols = [c for c in display_cols if c in df_timeline.columns]

        st.dataframe(
            df_timeline[available_cols].style.background_gradient(
                subset=["speed_drop_pct"],
                cmap="RdYlGn_r",
            ),
            use_container_width=True,
        )

    # ==================== MAP ====================
    st.divider()
    st.markdown("### Propagation Map")

    m = folium.Map(
        location=[BENGALURU_LAT, BENGALURU_LON],
        zoom_start=13,
        tiles="CartoDB dark_matter",
    )

    source_profile = profiles.get(source_cid, {})
    folium.Marker(
        location=[
            source_profile.get("centroid_lat", BENGALURU_LAT),
            source_profile.get("centroid_lon", BENGALURU_LON),
        ],
        popup=f"<b>SOURCE</b><br>Cluster {source_cid}<br>{result['source_area']}",
        icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
    ).add_to(m)

    severity_colors = {
        "CRITICAL": "red",
        "HIGH": "orange",
        "MEDIUM": "yellow",
        "LOW": "green",
    }

    for entry in timeline:
        cid = entry["cluster_id"]
        profile = profiles.get(cid, {})

        folium.CircleMarker(
            location=[
                profile.get("centroid_lat", BENGALURU_LAT),
                profile.get("centroid_lon", BENGALURU_LON),
            ],
            radius=max(5, min(15, entry["speed_drop_pct"] / 5)),
            color=severity_colors.get(entry["severity"], "blue"),
            fill=True,
            fill_color=severity_colors.get(entry["severity"], "blue"),
            fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>Cluster {cid}</b> — {entry['area']}<br>"
                f"Time: +{entry['minute']} min<br>"
                f"Severity: {entry['severity']}<br>"
                f"Speed Drop: {entry['speed_drop_pct']:.1f}%<br>"
                f"Queue: {entry['queue_length']:.0f} vehicles<br>"
                f"Delay: {entry['delay_seconds']:.0f}s",
                max_width=250,
            ),
        ).add_to(m)

    st_folium(m, width=800, height=500)
