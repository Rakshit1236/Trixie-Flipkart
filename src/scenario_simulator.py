"""
Scenario simulator for Trixie-Flipkart.
Physics-based what-if engine with preset scenarios and counterfactual mode.
"""
import numpy as np
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FREE_FLOW_SPEED_KMH, JAM_DENSITY_VPKM, LANE_CAPACITY_VPH,
    WEATHER_MULTIPLIERS, DAY_TYPE_MULTIPLIERS, SCENARIO_PRESETS
)
from src.utils import normalize_to_0_100
from src.traffic_impact import greenshields_speed, queue_model


class PhysicsSimulator:
    """Physics-based traffic simulator using Greenshields + M/M/1 queue model."""

    def __init__(self, free_flow_speed: float = FREE_FLOW_SPEED_KMH,
                 jam_density: float = JAM_DENSITY_VPKM):
        self.v_free = free_flow_speed
        self.k_jam = jam_density

    def simulate_removal(self, baseline: Dict, vehicle_reduction: int,
                          weather: str = "Clear", day_type: str = "Weekday",
                          cluster_id: int = None) -> Dict:
        weather_mult = WEATHER_MULTIPLIERS.get(weather, 1.0)
        day_mult = DAY_TYPE_MULTIPLIERS.get(day_type, 1.0)
        combined_mult = weather_mult * day_mult

        base_violations = baseline.get("total_violations", baseline.get("violations", 100))
        base_speed_drop = baseline.get("worst_speed_drop_pct", 25)
        base_vhl = baseline.get("total_vhl", baseline.get("vhl", 10))
        num_lanes = baseline.get("num_lanes", 2)
        avg_duration = baseline.get("avg_duration_minutes", 30)

        actual_reduction = min(vehicle_reduction, base_violations)
        reduction_factor = 1 - (actual_reduction / base_violations) if base_violations > 0 else 1

        new_violations = base_violations * reduction_factor
        new_speed_drop_pct = base_speed_drop * reduction_factor * combined_mult
        new_speed_kmh = self.v_free * (1 - new_speed_drop_pct / 100)

        capacity_vph = num_lanes * LANE_CAPACITY_VPH
        demand_vph = new_violations * (60 / avg_duration)
        queue_result = queue_model(demand_vph, capacity_vph)

        new_vhl = base_vhl * reduction_factor * combined_mult
        new_delay = queue_result["delay_seconds"]

        base_speed_kmh = self.v_free * (1 - base_speed_drop / 100)
        speed_improvement = new_speed_kmh - base_speed_kmh

        return {
            "cluster_id": cluster_id,
            "weather": weather,
            "day_type": day_type,
            "vehicle_reduction": actual_reduction,
            "baseline": {
                "violations": base_violations,
                "speed_kmh": round(base_speed_kmh, 1),
                "speed_drop_pct": round(base_speed_drop, 1),
                "vehicle_hours_lost": round(base_vhl, 2),
                "delay_seconds": round(queue_result["delay_seconds"], 1),
                "queue_length": round(queue_result["queue_length_veh"], 1),
            },
            "scenario": {
                "violations": round(new_violations, 1),
                "speed_kmh": round(new_speed_kmh, 1),
                "speed_drop_pct": round(new_speed_drop_pct, 1),
                "vehicle_hours_lost": round(new_vhl, 2),
                "delay_seconds": round(new_delay, 1),
                "queue_length": round(queue_result["queue_length_veh"], 1),
            },
            "improvement": {
                "violations_reduced": round(base_violations - new_violations, 1),
                "speed_gained_kmh": round(speed_improvement, 1),
                "vhl_reduced": round(base_vhl - new_vhl, 2),
                "delay_reduced_seconds": round(queue_result["delay_seconds"] - new_delay, 1),
            },
            "multipliers": {
                "weather": weather_mult,
                "day_type": day_mult,
                "combined": combined_mult,
            },
        }

    def simulate_counterfactual(self, baseline: Dict, occupancy_reduction_pct: float,
                                 cluster_id: int = None) -> Dict:
        """
        Counterfactual: What if parking occupancy were X% lower?
        """
        base_violations = baseline.get("total_violations", 100)
        base_speed_drop = baseline.get("worst_speed_drop_pct", 25)
        base_vhl = baseline.get("total_vhl", 10)
        num_lanes = baseline.get("num_lanes", 2)

        reduction_factor = 1 - (occupancy_reduction_pct / 100)
        new_violations = base_violations * reduction_factor
        new_speed_drop = base_speed_drop * reduction_factor
        new_speed_kmh = self.v_free * (1 - new_speed_drop / 100)
        new_vhl = base_vhl * reduction_factor

        capacity_vph = num_lanes * LANE_CAPACITY_VPH
        demand_vph = new_violations * (60 / 30)
        queue_result = queue_model(demand_vph, capacity_vph)

        base_speed_kmh = self.v_free * (1 - base_speed_drop / 100)

        return {
            "cluster_id": cluster_id,
            "occupancy_reduction_pct": occupancy_reduction_pct,
            "baseline": {
                "violations": base_violations,
                "speed_kmh": round(base_speed_kmh, 1),
                "speed_drop_pct": round(base_speed_drop, 1),
                "vehicle_hours_lost": round(base_vhl, 2),
                "delay_seconds": round(queue_result["delay_seconds"], 1),
                "queue_length": round(queue_result["queue_length_veh"], 1),
            },
            "scenario": {
                "violations": round(new_violations, 1),
                "speed_kmh": round(new_speed_kmh, 1),
                "speed_drop_pct": round(new_speed_drop, 1),
                "vehicle_hours_lost": round(new_vhl, 2),
                "delay_seconds": round(queue_result["delay_seconds"], 1),
                "queue_length": round(queue_result["queue_length_veh"], 1),
            },
            "improvement": {
                "violations_reduced": round(base_violations - new_violations, 1),
                "speed_gained_kmh": round(new_speed_kmh - base_speed_kmh, 1),
                "vhl_reduced": round(base_vhl - new_vhl, 2),
                "delay_reduced_seconds": round(queue_result["delay_seconds"] - new_delay, 1) if False else 0,
            },
            "summary": {
                "avg_speed_change_pct": round((new_speed_kmh - base_speed_kmh) / base_speed_kmh * 100, 1) if base_speed_kmh > 0 else 0,
                "queue_reduction_pct": round((1 - queue_result["queue_length_veh"] / max(queue_result["queue_length_veh"], 1)) * 100, 1),
                "violations_reduction_pct": round(occupancy_reduction_pct * reduction_factor, 1),
            },
        }

    def simulate_citywide(self, all_baselines: Dict[int, Dict], vehicle_reduction: int,
                           weather: str = "Clear", day_type: str = "Weekday") -> Dict:
        results = {}
        total_baseline_violations = 0
        total_scenario_violations = 0
        total_baseline_vhl = 0
        total_scenario_vhl = 0

        for cid, baseline in all_baselines.items():
            result = self.simulate_removal(baseline, vehicle_reduction, weather, day_type, cid)
            results[cid] = result

            total_baseline_violations += result["baseline"]["violations"]
            total_scenario_violations += result["scenario"]["violations"]
            total_baseline_vhl += result["baseline"]["vehicle_hours_lost"]
            total_scenario_vhl += result["scenario"]["vehicle_hours_lost"]

        v_reduction_pct = (
            (total_baseline_violations - total_scenario_violations) / total_baseline_violations * 100
            if total_baseline_violations > 0 else 0
        )
        vhl_reduction_pct = (
            (total_baseline_vhl - total_scenario_vhl) / total_baseline_vhl * 100
            if total_baseline_vhl > 0 else 0
        )

        return {
            "n_hotspots": len(results),
            "vehicle_reduction_per_hotspot": vehicle_reduction,
            "weather": weather,
            "day_type": day_type,
            "citywide": {
                "baseline_violations": round(total_baseline_violations, 1),
                "scenario_violations": round(total_scenario_violations, 1),
                "violations_reduction_pct": round(v_reduction_pct, 1),
                "baseline_vhl": round(total_baseline_vhl, 2),
                "scenario_vhl": round(total_scenario_vhl, 2),
                "vhl_reduction_pct": round(vhl_reduction_pct, 1),
            },
            "per_hotspot": results,
        }

    def run_preset(self, preset_name: str, all_baselines: Dict[int, Dict],
                    cluster_id: int = None) -> Dict:
        """Run a predefined scenario preset."""
        preset = SCENARIO_PRESETS.get(preset_name)
        if not preset:
            return {"error": f"Unknown preset: {preset_name}"}

        if cluster_id is not None:
            baseline = all_baselines.get(cluster_id, {})
            return self.simulate_removal(
                baseline,
                preset["vehicle_reduction"],
                preset["weather"],
                preset["day_type"],
                cluster_id,
            )
        else:
            return self.simulate_citywide(
                all_baselines,
                preset["vehicle_reduction"],
                preset["weather"],
                preset["day_type"],
            )


def create_simulator() -> PhysicsSimulator:
    return PhysicsSimulator()
