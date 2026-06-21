"""
Scenario simulator tab for Trixie-Flipkart.
Calls backend API for what-if simulations.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "https://rakshit1236-trixie-backend.hf.space")

SCENARIO_PRESETS = {
    "Remove Illegal Parking": {"icon": "🚫", "description": "Remove 50 illegally parked vehicles", "vehicle_reduction": 50, "weather": "Clear", "day_type": "Weekday"},
    "Festival Day": {"icon": "🎉", "description": "Simulate festival day traffic surge", "vehicle_reduction": 0, "weather": "Clear", "day_type": "Festival Day"},
    "Heavy Rain": {"icon": "🌧", "description": "Simulate heavy rain conditions", "vehicle_reduction": 0, "weather": "Heavy Rain", "day_type": "Weekday"},
    "Metro Delay": {"icon": "🚇", "description": "Metro service delayed, more vehicles", "vehicle_reduction": 30, "weather": "Clear", "day_type": "Weekday"},
    "Increase Capacity": {"icon": "🛣", "description": "Add 100 vehicles of road capacity", "vehicle_reduction": 100, "weather": "Clear", "day_type": "Weekday"},
}


def api_post(endpoint, payload):
    try:
        r = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def render_tab_scenario(profiles, impact, scores):
    st.subheader("What-If Simulator")

    st.markdown("""
    Simulate different scenarios and see instant impact on traffic.
    Use **Quick Scenarios** for one-click presets, or customize manually.
    """)

    # ==================== QUICK SCENARIOS ====================
    st.markdown("### Quick Scenarios")

    preset_cols = st.columns(5)
    selected_preset = None

    for i, (name, preset) in enumerate(SCENARIO_PRESETS.items()):
        with preset_cols[i]:
            if st.button(
                f"{preset['icon']}\n{name}",
                key=f"preset_{name}",
                use_container_width=True,
                help=preset["description"],
            ):
                selected_preset = name

    # ==================== COUNTERFACTUAL MODE ====================
    st.divider()
    st.markdown("### Counterfactual AI")

    counterfactual_mode = st.toggle(
        "Enable Counterfactual Mode",
        value=False,
        help="What if parking occupancy were X% lower?",
    )

    if counterfactual_mode:
        occupancy_reduction = st.slider(
            "Parking Occupancy Reduction (%)",
            min_value=5, max_value=50, value=15, step=5,
        )

    # ==================== MANUAL PARAMETERS ====================
    st.divider()
    st.markdown("### Manual Parameters")

    col1, col2 = st.columns([1, 2])

    with col1:
        if selected_preset:
            preset = SCENARIO_PRESETS[selected_preset]
            vehicle_reduction = st.slider("Vehicles to Remove", 0, 200, preset["vehicle_reduction"], step=5)
            weather = st.selectbox("Weather", ["Clear", "Light Rain", "Heavy Rain", "Fog"], index=["Clear", "Light Rain", "Heavy Rain", "Fog"].index(preset["weather"]))
            day_type = st.selectbox("Day Type", ["Weekday", "Saturday", "Sunday", "Festival Day", "Public Holiday"], index=["Weekday", "Saturday", "Sunday", "Festival Day", "Public Holiday"].index(preset["day_type"]))
        else:
            vehicle_reduction = st.slider("Vehicles to Remove", 0, 200, 50, step=5)
            weather = st.selectbox("Weather", ["Clear", "Light Rain", "Heavy Rain", "Fog"], index=0)
            day_type = st.selectbox("Day Type", ["Weekday", "Saturday", "Sunday", "Festival Day", "Public Holiday"], index=0)

        hotspot_options = ["All Hotspots"] + [
            f"Cluster {cid} ({profiles[str(cid)].get('area', 'Unknown') if str(cid) in profiles else profiles.get(cid, {}).get('area', 'Unknown')})"
            for cid in sorted(int(k) for k in profiles.keys())
        ]
        hotspot_selection = st.selectbox("Focus on Hotspot", hotspot_options, index=0)

    with col2:
        st.markdown("### Simulation Results")

        if st.button("Run Simulation", type="primary", use_container_width=True):
            if counterfactual_mode:
                payload = {"occupancy_reduction_pct": occupancy_reduction}
                if hotspot_selection != "All Hotspots":
                    cid = int(hotspot_selection.split("(")[0].replace("Cluster ", "").strip())
                    payload["cluster_id"] = cid
                result = api_post("/counterfactual", payload)
            else:
                payload = {
                    "vehicle_reduction": vehicle_reduction,
                    "weather": weather,
                    "day_type": day_type,
                }
                if hotspot_selection != "All Hotspots":
                    cid = int(hotspot_selection.split("(")[0].replace("Cluster ", "").strip())
                    payload["cluster_id"] = cid
                result = api_post("/scenario", payload)

            if "error" in result:
                st.error(f"Simulation failed: {result['error']}")
            else:
                if "citywide" in result:
                    citywide = result["citywide"]
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Violations Reduced", f"{citywide.get('violations_reduction_pct', 0):.1f}%")
                    with m2:
                        st.metric("Baseline", f"{citywide.get('baseline_violations', 0):.0f}")
                    with m3:
                        st.metric("Scenario", f"{citywide.get('scenario_violations', 0):.0f}")
                    with m4:
                        st.metric("Hotspots Affected", result.get("n_hotspots", 0))

                    per_hotspot = result.get("per_hotspot", {})
                    if per_hotspot:
                        first_key = list(per_hotspot.keys())[0]
                        first_result = per_hotspot[first_key]
                        if "improvement" in first_result:
                            st.json(first_result["improvement"])
                elif "improvement" in result:
                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("Violations Reduced", f"{result['improvement'].get('violations_reduced', 0):.0f}")
                    with m2:
                        st.metric("Speed Gained", f"+{result['improvement'].get('speed_gained_kmh', 0):.1f} km/h")
                    st.json(result)
                else:
                    st.json(result)
        else:
            st.info("Click **Run Simulation** to see results")
