"""
Scenario simulator tab for Trixie-Flipkart.
Preset buttons + counterfactual mode + interactive what-if engine.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import WEATHER_MULTIPLIERS, DAY_TYPE_MULTIPLIERS, SCENARIO_PRESETS
from src.scenario_simulator import PhysicsSimulator


def render_tab_scenario(profiles, impact, scores):
    st.subheader("🧪 What-If Simulator")

    st.markdown("""
    Simulate different scenarios and see instant impact on traffic.
    Use **Quick Scenarios** for one-click presets, or customize manually.
    """)

    simulator = PhysicsSimulator()

    # ==================== QUICK SCENARIOS ====================
    st.markdown("### ⚡ Quick Scenarios")

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
    st.markdown("### 🔮 Counterfactual AI")

    counterfactual_mode = st.toggle(
        "Enable Counterfactual Mode",
        value=False,
        help="What if parking occupancy were X% lower?",
    )

    if counterfactual_mode:
        occupancy_reduction = st.slider(
            "Parking Occupancy Reduction (%)",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
            help="Percentage reduction in parking occupancy to simulate",
        )

    # ==================== MANUAL PARAMETERS ====================
    st.divider()
    st.markdown("### 🎛️ Manual Parameters")

    col1, col2 = st.columns([1, 2])

    with col1:
        if selected_preset:
            preset = SCENARIO_PRESETS[selected_preset]
            vehicle_reduction = st.slider(
                "Vehicles to Remove", 0, 200, preset["vehicle_reduction"], step=5,
            )
            weather = st.selectbox(
                "Weather", list(WEATHER_MULTIPLIERS.keys()),
                index=list(WEATHER_MULTIPLIERS.keys()).index(preset["weather"]),
            )
            day_type = st.selectbox(
                "Day Type", list(DAY_TYPE_MULTIPLIERS.keys()),
                index=list(DAY_TYPE_MULTIPLIERS.keys()).index(preset["day_type"]),
            )
        else:
            vehicle_reduction = st.slider("Vehicles to Remove", 0, 200, 50, step=5)
            weather = st.selectbox("Weather", list(WEATHER_MULTIPLIERS.keys()), index=0)
            day_type = st.selectbox("Day Type", list(DAY_TYPE_MULTIPLIERS.keys()), index=0)

        hotspot_options = ["All Hotspots"] + [
            f"Cluster {cid} ({profiles[cid].get('area', 'Unknown')})"
            for cid in sorted(profiles.keys())
        ]
        hotspot_selection = st.selectbox("Focus on Hotspot", hotspot_options, index=0)

    with col2:
        st.markdown("### 📊 Simulation Results")

        baselines = {}
        for cid, profile in profiles.items():
            impact_data = impact.get(cid, {})
            baselines[cid] = {
                "total_violations": profile.get("total_violations", 100),
                "worst_speed_drop_pct": impact_data.get("worst_speed_drop_pct", 25),
                "total_vhl": impact_data.get("total_vhl", 10),
                "num_lanes": profile.get("num_lanes", 2),
                "avg_duration_minutes": profile.get("avg_duration_minutes", 30),
            }

        if counterfactual_mode:
            # ==================== COUNTERFACTUAL RESULTS ====================
            if hotspot_selection == "All Hotspots":
                total_baseline_violations = 0
                total_scenario_violations = 0
                total_baseline_speed = 0
                total_scenario_speed = 0
                total_baseline_queue = 0
                total_scenario_queue = 0

                for cid, baseline in baselines.items():
                    result = simulator.simulate_counterfactual(baseline, occupancy_reduction, cid)
                    total_baseline_violations += result["baseline"]["violations"]
                    total_scenario_violations += result["scenario"]["violations"]
                    total_baseline_speed += result["baseline"]["speed_kmh"]
                    total_scenario_speed += result["scenario"]["speed_kmh"]
                    total_baseline_queue += result["baseline"]["queue_length"]
                    total_scenario_queue += result["scenario"]["queue_length"]

                n = len(baselines)
                avg_speed_change = (total_scenario_speed - total_baseline_speed) / n
                avg_queue_change = total_scenario_queue - total_baseline_queue

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric(
                        "Avg Speed Change",
                        f"+{avg_speed_change:.1f} km/h",
                        delta=f"{total_baseline_speed/n:.1f} → {total_scenario_speed/n:.1f}",
                    )
                with m2:
                    violations_change = total_baseline_violations - total_scenario_violations
                    st.metric("Violations Reduced", f"{violations_change:.0f}")
                with m3:
                    st.metric("Queue Change", f"{avg_queue_change:.1f} vehicles")
                with m4:
                    st.metric("Occupancy Reduction", f"{occupancy_reduction}%")

                st.info(f"**Counterfactual:** If parking occupancy were **{occupancy_reduction}% lower**, "
                       f"average speed would improve by **{avg_speed_change:.1f} km/h** "
                       f"and **{violations_change:.0f} violations** would be prevented.")

            else:
                cid = int(hotspot_selection.split("(")[0].replace("Cluster ", "").strip())
                baseline = baselines.get(cid, {})
                result = simulator.simulate_counterfactual(baseline, occupancy_reduction, cid)

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    speed_change = result["summary"]["avg_speed_change_pct"]
                    st.metric(
                        "Speed Change",
                        f"+{speed_change:.1f}%",
                        delta=f"{result['baseline']['speed_kmh']:.1f} → {result['scenario']['speed_kmh']:.1f}",
                    )
                with m2:
                    st.metric("Violations Reduced", f"{result['improvement']['violations_reduced']:.0f}")
                with m3:
                    st.metric("Queue Length", f"{result['scenario']['queue_length']:.1f}")
                with m4:
                    st.metric("Occupancy Reduction", f"{occupancy_reduction}%")

        else:
            # ==================== STANDARD SIMULATION ====================
            if hotspot_selection == "All Hotspots":
                result = simulator.simulate_citywide(baselines, vehicle_reduction, weather, day_type)
                citywide = result["citywide"]

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric(
                        "Violations Reduced",
                        f"{citywide['violations_reduction_pct']:.1f}%",
                        delta=f"{citywide['scenario_violations']:.0f} → {citywide['baseline_violations']:.0f}",
                    )
                with m2:
                    st.metric("VHL Reduced", f"{citywide['vhl_reduction_pct']:.1f}%")
                with m3:
                    st.metric("Weather Impact", f"{WEATHER_MULTIPLIERS[weather]:.2f}x")
                with m4:
                    st.metric("Day Impact", f"{DAY_TYPE_MULTIPLIERS[day_type]:.2f}x")

                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=("Violations", "Vehicle-Hours Lost"),
                )
                fig.add_trace(
                    go.Bar(
                        x=["Baseline", "Scenario"],
                        y=[citywide["baseline_violations"], citywide["scenario_violations"]],
                        marker_color=["#FF6B6B", "#4ECDC4"],
                        name="Violations",
                    ),
                    row=1, col=1,
                )
                fig.add_trace(
                    go.Bar(
                        x=["Baseline", "Scenario"],
                        y=[citywide["baseline_vhl"], citywide["scenario_vhl"]],
                        marker_color=["#FF6B6B", "#4ECDC4"],
                        name="VHL",
                    ),
                    row=1, col=2,
                )
                fig.update_layout(height=350, title_text="City-Wide Comparison", showlegend=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

            else:
                cid = int(hotspot_selection.split("(")[0].replace("Cluster ", "").strip())
                baseline = baselines.get(cid, {})
                result = simulator.simulate_removal(baseline, vehicle_reduction, weather, day_type, cid)

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    speed_change = result["improvement"]["speed_gained_kmh"]
                    st.metric(
                        "Speed Change",
                        f"+{speed_change:.1f} km/h",
                        delta=f"{result['baseline']['speed_kmh']:.1f} → {result['scenario']['speed_kmh']:.1f}",
                    )
                with m2:
                    st.metric("VHL Reduced", f"{result['improvement']['vhl_reduced']:.1f} hours")
                with m3:
                    st.metric("Violations Reduced", f"{result['improvement']['violations_reduced']:.0f}")
                with m4:
                    st.metric("Delay Reduced", f"{result['improvement']['delay_reduced_seconds']:.0f}s")

                fig = go.Figure()
                categories = ["Violations", "Speed (km/h)", "VHL", "Delay (s)"]
                baseline_vals = [
                    result["baseline"]["violations"],
                    result["baseline"]["speed_kmh"],
                    result["baseline"]["vehicle_hours_lost"],
                    result["baseline"]["delay_seconds"],
                ]
                scenario_vals = [
                    result["scenario"]["violations"],
                    result["scenario"]["speed_kmh"],
                    result["scenario"]["vehicle_hours_lost"],
                    result["scenario"]["delay_seconds"],
                ]
                fig.add_trace(go.Bar(name="Baseline", x=categories, y=baseline_vals, marker_color="#FF6B6B"))
                fig.add_trace(go.Bar(name="Scenario", x=categories, y=scenario_vals, marker_color="#4ECDC4"))
                fig.update_layout(barmode="group", height=350, title_text=f"Cluster {cid} — Before vs After", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Detailed Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Baseline**")
                    st.json(result["baseline"])
                with col2:
                    st.markdown("**Scenario**")
                    st.json(result["scenario"])
