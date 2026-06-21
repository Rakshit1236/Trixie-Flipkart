"""
Analytics module for Trixie-Flipkart.
Propagation, recommendations, early warnings, XAI integration.
Enhanced with timeline data structures for UI.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import deque
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FREE_FLOW_SPEED_KMH, JAM_DENSITY_VPKM, LANE_CAPACITY_VPH,
    PROPAGATION_RADIUS_KM, PROPAGATION_SPEED_KMH,
    PROPAGATION_MIN_DELAY_MIN, PROPAGATION_SPEED_ESCALATION,
    RUSH_HOUR_MULTIPLIER, CHRONIC_MULTIPLIER, TIME_OF_DAY_MULTIPLIERS,
    WARNING_HORIZONS, THREAT_LEVELS,
    OFFICER_DEPLOYMENT, DELAY_REDUCTION_PCT, DELAY_REDUCTION_CAP,
    RESOURCE_TYPE, ETA_PER_OFFICER_MIN
)
from src.utils import haversine_km, get_time_of_day


class CongestionPropagation:
    """Physics-based congestion propagation using queue dynamics."""

    def __init__(self, profiles: Dict[int, Dict], adjacency: Dict[int, List[int]]):
        self.profiles = profiles
        self.adjacency = adjacency

    def simulate_propagation(self, source_cid: int, start_hour: int,
                              horizon_minutes: int = 60,
                              initial_demand: float = None) -> Dict:
        if source_cid not in self.profiles:
            return {"error": f"Source cluster {source_cid} not found"}

        if initial_demand is None:
            initial_demand = self.profiles[source_cid]["daily_rate"] * 10

        timeline = []
        affected = {}

        queue = deque([(source_cid, 0, initial_demand)])
        visited = {source_cid}

        while queue:
            cid, minute, demand = queue.popleft()

            if minute > horizon_minutes:
                continue

            if cid not in self.profiles:
                continue

            profile = self.profiles[cid]

            num_lanes = profile.get("num_lanes", 2)
            capacity_vph = num_lanes * LANE_CAPACITY_VPH

            rho = min(demand / (capacity_vph + 1e-6), 0.99)
            queue_length = rho / (1 - rho) if rho < 1 else float("inf")
            delay = queue_length / capacity_vph * 3600 if rho < 1 else 300

            speed_drop_pct = rho * 100
            speed_kmh = FREE_FLOW_SPEED_KMH * (1 - rho)

            if speed_drop_pct >= 40:
                severity = "CRITICAL"
            elif speed_drop_pct >= 25:
                severity = "HIGH"
            elif speed_drop_pct >= 10:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            hour = start_hour + (minute // 60)
            minute_of_hour = minute % 60
            time_str = f"{hour % 24}:{minute_of_hour:02d}"

            entry = {
                "minute": minute,
                "time_str": time_str,
                "cluster_id": cid,
                "area": profile.get("area", "Unknown"),
                "road_type": profile.get("road_type", "Other"),
                "demand_vph": round(demand, 1),
                "capacity_vph": capacity_vph,
                "utilization": round(rho, 3),
                "queue_length": round(queue_length, 1),
                "delay_seconds": round(delay, 1),
                "speed_drop_pct": round(speed_drop_pct, 1),
                "speed_kmh": round(speed_kmh, 1),
                "severity": severity,
                "centroid_lat": profile.get("centroid_lat", 0),
                "centroid_lon": profile.get("centroid_lon", 0),
            }

            timeline.append(entry)
            affected[cid] = entry

            neighbors = self.adjacency.get(cid, [])
            for nid in neighbors:
                if nid in visited:
                    continue

                dist_km = haversine_km(
                    profile["centroid_lat"], profile["centroid_lon"],
                    self.profiles[nid]["centroid_lat"],
                    self.profiles[nid]["centroid_lon"]
                )

                travel_time_min = (dist_km / PROPAGATION_SPEED_KMH * 60) + PROPAGATION_MIN_DELAY_MIN
                next_minute = minute + int(travel_time_min)

                if next_minute <= horizon_minutes:
                    decay = 0.85 ** dist_km
                    propagated_demand = demand * decay * 0.7

                    if propagated_demand > 10:
                        queue.append((nid, next_minute, propagated_demand))
                        visited.add(nid)

        timeline.sort(key=lambda x: (x["minute"], x["cluster_id"]))

        return {
            "source_cid": source_cid,
            "source_area": self.profiles[source_cid].get("area", "Unknown"),
            "start_hour": start_hour,
            "horizon_minutes": horizon_minutes,
            "n_affected": len(affected),
            "timeline": timeline,
            "affected_hotspots": affected,
        }


def generate_recommendations(profiles: Dict[int, Dict], impact: Dict[int, Dict],
                              scores: Dict[int, Dict], top_n: int = 20) -> List[Dict]:
    print("Generating action recommendations...")

    sorted_cids = sorted(
        scores.keys(),
        key=lambda cid: scores[cid].get("priority_score_normalized", 0),
        reverse=True
    )[:top_n]

    recommendations = []

    for cid in sorted_cids:
        profile = profiles.get(cid, {})
        impact_data = impact.get(cid, {})
        score_data = scores.get(cid, {})

        severity = impact_data.get("severity_class", "LOW")
        officers = OFFICER_DEPLOYMENT.get(severity, 1)

        base_reduction = DELAY_REDUCTION_PCT.get(severity, 0.20)
        cap = DELAY_REDUCTION_CAP.get(severity, 20)
        expected_reduction_pct = min(base_reduction * 100, cap)

        peak_hours = profile.get("peak_hours", [8, 9, 17, 18])
        timing = f"Deploy before {min(peak_hours)}:00"

        priority_score = score_data.get("priority_score_normalized", 0)
        resource = RESOURCE_TYPE.get(severity, "Monitor")
        eta = officers * ETA_PER_OFFICER_MIN

        recommendations.append({
            "cluster_id": cid,
            "area": profile.get("area", "Unknown"),
            "road_type": profile.get("road_type", "Other"),
            "severity": severity,
            "priority_score": priority_score,
            "officers_needed": officers,
            "expected_delay_reduction_pct": round(expected_reduction_pct, 1),
            "timing": timing,
            "peak_hours": peak_hours,
            "total_violations": profile.get("total_violations", 0),
            "daily_rate": round(profile.get("daily_rate", 0), 1),
            "is_chronic": profile.get("is_chronic", False),
            "resource_type": resource,
            "eta_minutes": eta,
            "centroid_lat": profile.get("centroid_lat", 0),
            "centroid_lon": profile.get("centroid_lon", 0),
        })

    print(f"  Generated {len(recommendations)} recommendations")
    return recommendations


def generate_early_warnings(profiles: Dict[int, Dict], impact: Dict[int, Dict],
                             scores: Dict[int, Dict]) -> Dict[int, Dict]:
    print("Generating early warnings...")

    now = datetime.now()
    current_hour = now.hour
    time_of_day = get_time_of_day(current_hour)
    is_rush = any(start <= current_hour < end for start, end in [(8, 11), (17, 21)])

    warnings = {}

    for horizon in WARNING_HORIZONS:
        horizon_warnings = []

        for cid, profile in profiles.items():
            priority = scores.get(cid, {}).get("priority_score_normalized", 50)

            threat_score = _compute_threat_score(
                priority, is_rush, profile.get("is_chronic", False),
                time_of_day, horizon
            )

            if threat_score >= THREAT_LEVELS["HIGH"]:
                threat_level = "HIGH"
            elif threat_score >= THREAT_LEVELS["MEDIUM"]:
                threat_level = "MEDIUM"
            else:
                threat_level = "LOW"

            horizon_warnings.append({
                "cluster_id": cid,
                "area": profile.get("area", "Unknown"),
                "road_type": profile.get("road_type", "Other"),
                "threat_score": round(threat_score, 1),
                "threat_level": threat_level,
                "is_chronic": profile.get("is_chronic", False),
                "centroid_lat": profile.get("centroid_lat", 0),
                "centroid_lon": profile.get("centroid_lon", 0),
            })

        horizon_warnings.sort(key=lambda x: x["threat_score"], reverse=True)
        warnings[horizon] = horizon_warnings

    print(f"  Generated warnings for {len(WARNING_HORIZONS)} horizons")
    return warnings


def _compute_threat_score(priority: float, is_rush: bool, is_chronic: bool,
                           time_of_day: str, horizon: int) -> float:
    score = priority

    if is_rush:
        score *= RUSH_HOUR_MULTIPLIER

    if is_chronic:
        score *= CHRONIC_MULTIPLIER

    tod_mult = TIME_OF_DAY_MULTIPLIERS.get(time_of_day, 1.0)
    score *= tod_mult

    horizon_factor = 1 + (horizon / 60) * 0.003
    score *= horizon_factor

    return min(score, 100)


def generate_dispatch_report(recommendations: List[Dict], date_str: str = None) -> str:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    lines = []
    lines.append("=" * 70)
    lines.append(f"DISPATCH REPORT — {date_str}")
    lines.append("=" * 70)
    lines.append("")

    total_officers = sum(r["officers_needed"] for r in recommendations)
    critical = [r for r in recommendations if r["severity"] == "CRITICAL"]
    high = [r for r in recommendations if r["severity"] == "HIGH"]

    lines.append(f"Total Hotspots: {len(recommendations)}")
    lines.append(f"Critical: {len(critical)} | High: {len(high)}")
    lines.append(f"Total Officers Needed: {total_officers}")
    lines.append("")

    if critical:
        lines.append("CRITICAL HOTSPOTS:")
        lines.append("-" * 40)
        for r in critical:
            lines.append(f"  [{r['cluster_id']}] {r['area']} ({r['road_type']})")
            lines.append(f"    Officers: {r['officers_needed']} | "
                        f"Delay Reduction: {r['expected_delay_reduction_pct']}%")
            lines.append(f"    Action: {r.get('resource_type', 'Deploy Officers')} | "
                        f"ETA: {r.get('eta_minutes', 8)} min")
            lines.append(f"    Timing: {r['timing']}")
            lines.append("")

    if high:
        lines.append("HIGH PRIORITY HOTSPOTS:")
        lines.append("-" * 40)
        for r in high:
            lines.append(f"  [{r['cluster_id']}] {r['area']} ({r['road_type']})")
            lines.append(f"    Officers: {r['officers_needed']} | "
                        f"Delay Reduction: {r['expected_delay_reduction_pct']}%")
            lines.append("")

    lines.append("ALL RECOMMENDATIONS:")
    lines.append("-" * 40)
    for i, r in enumerate(recommendations, 1):
        chronic_tag = " [CHRONIC]" if r["is_chronic"] else ""
        lines.append(f"{i}. [{r['cluster_id']}] {r['area']} — "
                    f"Score: {r['priority_score']:.0f} | "
                    f"Severity: {r['severity']}{chronic_tag}")
        lines.append(f"   Officers: {r['officers_needed']} | "
                    f"Action: {r.get('resource_type', 'Monitor')} | "
                    f"ETA: {r.get('eta_minutes', 8)} min")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def run_analytics(profiles: Dict[int, Dict], impact: Dict[int, Dict],
                   scores: Dict[int, Dict], ripple: Dict[int, Dict]) -> Dict:
    print("Running analytics...")

    recommendations = generate_recommendations(profiles, impact, scores)
    warnings = generate_early_warnings(profiles, impact, scores)
    dispatch_report = generate_dispatch_report(recommendations)

    return {
        "recommendations": recommendations,
        "warnings": warnings,
        "dispatch_report": dispatch_report,
    }
