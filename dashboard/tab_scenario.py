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


def api_post(endpoint, payload):
    try:
        r = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def render_tab_scenario(profiles, impact, scores):
    st.subheader("What-If Simulator")

    st.markdown("Simulate different scenarios and see instant impact on traffic.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Parameters")

        scenario_type = st.radio("Mode", ["Quick Scenario", "Counterfactual AI"], horizontal=True)

        if scenario_type == "Quick Scenario":
            preset_name = st.selectbox("Scenario", [
                "Remove Illegal Parking",
                "Festival Day",
                "Heavy Rain",
                "Metro Delay",
                "Increase Capacity",
            ])
            presets = {
                "Remove Illegal Parking": {"vehicle_reduction": 50, "weather": "Clear", "day_type": "Weekday"},
                "Festival Day": {"vehicle_reduction": 0, "weather": "Clear", "day_type": "Festival Day"},
                "Heavy Rain": {"vehicle_reduction": 0, "weather": "Heavy Rain", "day_type": "Weekday"},
                "Metro Delay": {"vehicle_reduction": 30, "weather": "Clear", "day_type": "Weekday"},
                "Increase Capacity": {"vehicle_reduction": 100, "weather": "Clear", "day_type": "Weekday"},
            }
            p = presets[preset_name]
            vehicle_reduction = st.slider("Vehicles to Remove", 0, 200, p["vehicle_reduction"], step=5)
            weather = st.selectbox("Weather", ["Clear", "Light Rain", "Heavy Rain", "Fog"], index=["Clear", "Light Rain", "Heavy Rain", "Fog"].index(p["weather"]))
            day_type = st.selectbox("Day Type", ["Weekday", "Saturday", "Sunday", "Festival Day", "Public Holiday"], index=["Weekday", "Saturday", "Sunday", "Festival Day", "Public Holiday"].index(p["day_type"]))
        else:
            vehicle_reduction = 0
            weather = "Clear"
            day_type = "Weekday"
            occupancy_reduction = st.slider("Occupancy Reduction (%)", 5, 50, 15, step=5)

        hotspot_options = ["All Hotspots"] + [
            f"C{cid} — {profiles[cid].get('area', '?')}"
            for cid in sorted(profiles.keys(), key=lambda x: int(x))
        ]
        hotspot_selection = st.selectbox("Focus Hotspot", hotspot_options, index=0)

        if st.button("Run Simulation", type="primary", use_container_width=True):
            if hotspot_selection != "All Hotspots":
                cid = int(hotspot_selection.split("—")[0].replace("C", "").strip())
            else:
                cid = None

            with st.spinner("Running simulation..."):
                if scenario_type == "Counterfactual AI":
                    payload = {"occupancy_reduction_pct": occupancy_reduction}
                    if cid is not None:
                        payload["cluster_id"] = cid
                    result = api_post("/counterfactual", payload)
                else:
                    payload = {"vehicle_reduction": vehicle_reduction, "weather": weather, "day_type": day_type}
                    if cid is not None:
                        payload["cluster_id"] = cid
                    result = api_post("/scenario", payload)

            if "error" in result:
                st.error(f"Failed: {result['error']}")
            else:
                st.session_state["scenario_result"] = result

    with col2:
        st.markdown("### Results")
        result = st.session_state.get("scenario_result")

        if result is None:
            st.info("Configure parameters and click **Run Simulation**")
        elif "error" in result:
            st.error(f"Simulation error: {result['error']}")
        elif "citywide" in result:
            citywide = result["citywide"]
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Violations Reduced", f"{citywide.get('violations_reduction_pct', 0):.1f}%")
            with m2:
                st.metric("Baseline", f"{citywide.get('baseline_violations', 0):.0f}")
            with m3:
                st.metric("Scenario", f"{citywide.get('scenario_violations', 0):.0f}")
            with m4:
                st.metric("Hotspots", result.get("n_hotspots", 0))

            per_hs = result.get("per_hotspot", {})
            if per_hs:
                first_key = list(per_hs.keys())[0]
                first_r = per_hs[first_key]
                if "baseline" in first_r and "scenario" in first_r:
                    labels = ["Violations", "Speed (km/h)"]
                    b_vals = [first_r["baseline"].get("violations", 0), first_r["baseline"].get("speed_kmh", 0)]
                    s_vals = [first_r["scenario"].get("violations", 0), first_r["scenario"].get("speed_kmh", 0)]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(name="Baseline", x=labels, y=b_vals, marker_color="#FF6B6B"))
                    fig.add_trace(go.Bar(name="Scenario", x=labels, y=s_vals, marker_color="#4ECDC4"))
                    fig.update_layout(barmode="group", height=300, template="plotly_dark", title="Sample Hotspot Comparison")
                    st.plotly_chart(fig, use_container_width=True)
        elif "improvement" in result:
            imp = result["improvement"]
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Violations Reduced", f"{imp.get('violations_reduced', 0):.0f}")
            with m2:
                st.metric("Speed Gained", f"+{imp.get('speed_gained_kmh', 0):.1f} km/h")

            if "baseline" in result and "scenario" in result:
                labels = list(result["baseline"].keys())
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Baseline", x=labels, y=list(result["baseline"].values()), marker_color="#FF6B6B"))
                fig.add_trace(go.Bar(name="Scenario", x=labels, y=list(result["scenario"].values()), marker_color="#4ECDC4"))
                fig.update_layout(barmode="group", height=350, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.json(result)
