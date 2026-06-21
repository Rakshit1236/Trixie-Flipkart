"""
Traffic impact analysis for Trixie-flipkartgridlock.
Greenshields speed-density model + M/M/1 queue approximation.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FREE_FLOW_SPEED_KMH, JAM_DENSITY_VPKM, LANE_CAPACITY_VPH,
    DEFAULT_ROAD_CAPACITY_VPH, PROPAGATION_RADIUS_KM, IMPACT_DECAY_FACTOR
)
from src.utils import classify_severity, haversine_km


def compute_hourly_impact_vectorized(df: pd.DataFrame, profiles: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    Vectorized hourly impact computation for ALL clusters at once.
    Returns per-cluster impact metrics.
    """
    print("Computing hourly traffic impact (vectorized)...")
    
    clustered = df[df["cluster_id"] != -1].copy()
    
    # Aggregate hourly by cluster
    hourly = clustered.groupby(["cluster_id", "hour"]).agg(
        blocked_severity=("severity_weight", "sum"),
        violations=("latitude", "count"),
        avg_severity=("severity_weight", "mean"),
        avg_duration=("violation_duration_minutes", "mean") if "violation_duration_minutes" in clustered.columns else ("latitude", "count"),
    ).reset_index()
    
    # Compute blocked fraction per cluster-hour
    hourly["num_lanes"] = hourly["cluster_id"].map(
        {cid: profiles[cid]["num_lanes"] for cid in profiles}
    ).fillna(2)
    
    hourly["blocked_fraction"] = np.minimum(
        hourly["blocked_severity"] / hourly["num_lanes"], 0.95
    )
    
    # Greenshields speed model
    hourly["speed_kmh"] = FREE_FLOW_SPEED_KMH * (1 - hourly["blocked_fraction"])
    hourly["speed_drop_pct"] = hourly["blocked_fraction"] * 100
    
    # Queue model (M/M/1 approximation)
    hourly["capacity_vph"] = hourly["num_lanes"] * LANE_CAPACITY_VPH
    hourly["demand_vph"] = hourly["violations"] * (60 / hourly["avg_duration"].clip(lower=5))
    
    # Queue utilization
    hourly["rho"] = np.minimum(hourly["demand_vph"] / hourly["capacity_vph"], 0.99)
    
    # Vehicle-hours lost (VHL) per cluster-hour
    hourly["vehicle_hours_lost"] = hourly["violations"] * hourly["avg_duration"] / 60.0
    
    # Delay per vehicle (seconds)
    hourly["delay_seconds"] = np.where(
        hourly["rho"] < 1,
        hourly["rho"] / (1 - hourly["rho"]) / hourly["capacity_vph"] * 3600,
        300  # cap at 5 minutes for gridlock
    )
    
    # Severity classification
    hourly["severity_class"] = hourly["blocked_fraction"].apply(classify_severity)
    
    # Find worst hour per cluster
    worst_by_cluster = hourly.loc[hourly.groupby("cluster_id")["speed_drop_pct"].idxmax()]
    
    # Aggregate per-cluster impact
    impact = {}
    for cid in profiles:
        cluster_hours = hourly[hourly["cluster_id"] == cid]
        
        if len(cluster_hours) == 0:
            impact[cid] = {
                "worst_speed_drop_pct": 0,
                "worst_hour": 0,
                "total_vhl": 0,
                "avg_speed_kmh": FREE_FLOW_SPEED_KMH,
                "total_violations": 0,
                "severity_class": "LOW",
                "total_delay_seconds": 0,
            }
            continue
        
        worst = worst_by_cluster[worst_by_cluster["cluster_id"] == cid]
        
        impact[cid] = {
            "worst_speed_drop_pct": float(worst["speed_drop_pct"].iloc[0]) if len(worst) > 0 else 0,
            "worst_hour": int(worst["hour"].iloc[0]) if len(worst) > 0 else 0,
            "total_vhl": float(cluster_hours["vehicle_hours_lost"].sum()),
            "avg_speed_kmh": float(cluster_hours["speed_kmh"].mean()),
            "total_violations": int(cluster_hours["violations"].sum()),
            "severity_class": worst["severity_class"].iloc[0] if len(worst) > 0 else "LOW",
            "total_delay_seconds": float(cluster_hours["delay_seconds"].sum()),
        }
    
    print(f"  Computed impact for {len(impact)} clusters")
    return impact


def compute_ripple_effects(profiles: Dict[int, Dict], impact: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    Compute congestion ripple effects between neighboring hotspots.
    Uses KD-tree adjacency (already computed in clustering).
    """
    print("Computing ripple effects...")
    
    ripple = {}
    
    for cid, profile in profiles.items():
        neighbors = profile.get("neighbors", [])
        
        if not neighbors or cid not in impact:
            ripple[cid] = {
                "neighbor_count": 0,
                "max_neighbor_impact": 0,
                "total_ripple_vhl": 0,
            }
            continue
        
        neighbor_impacts = []
        total_ripple_vhl = 0
        
        for nid in neighbors:
            if nid not in impact or nid == cid:
                continue
            
            dist_km = haversine_km(
                profile["centroid_lat"], profile["centroid_lon"],
                profiles[nid]["centroid_lat"], profiles[nid]["centroid_lon"]
            )
            
            # Decay factor
            decay = IMPACT_DECAY_FACTOR ** dist_km
            
            # Ripple impact
            neighbor_impact = impact[nid]["worst_speed_drop_pct"] * decay
            neighbor_impacts.append(neighbor_impact)
            
            # Ripple VHL
            ripple_vhl = impact[nid]["total_vhl"] * decay
            total_ripple_vhl += ripple_vhl
        
        ripple[cid] = {
            "neighbor_count": len(neighbors),
            "max_neighbor_impact": max(neighbor_impacts) if neighbor_impacts else 0,
            "total_ripple_vhl": total_ripple_vhl,
        }
    
    print(f"  Ripple effects computed for {len(ripple)} clusters")
    return ripple


def greenshields_speed(density: float, v_free: float = FREE_FLOW_SPEED_KMH,
                       k_jam: float = JAM_DENSITY_VPKM) -> float:
    """Greenshields speed-density model: v = v_f * (1 - k/k_j)"""
    return v_free * max(0, 1 - density / k_jam)


def queue_model(demand_vph: float, capacity_vph: float) -> Dict[str, float]:
    """
    M/M/1 queue approximation.
    Returns queue length and delay.
    """
    rho = min(demand_vph / (capacity_vph + 1e-6), 0.99)
    
    if rho >= 1:
        return {
            "queue_length_veh": float("inf"),
            "delay_seconds": 300,  # cap at 5 min
            "utilization": rho,
        }
    
    queue_length = rho / (1 - rho)
    delay = queue_length / capacity_vph * 3600  # seconds
    
    return {
        "queue_length_veh": queue_length,
        "delay_seconds": delay,
        "utilization": rho,
    }


def run_impact_analysis(df: pd.DataFrame, profiles: Dict[int, Dict]) -> Tuple[Dict, Dict]:
    """Run full impact analysis."""
    impact = compute_hourly_impact_vectorized(df, profiles)
    ripple = compute_ripple_effects(profiles, impact)
    return impact, ripple


if __name__ == "__main__":
    from src.data_pipeline import run_pipeline
    from src.clustering import cluster_hotspots
    
    df = run_pipeline()
    df, profiles = cluster_hotspots(df)
    impact, ripple = run_impact_analysis(df, profiles)
    
    print(f"\nImpact analysis complete:")
    print(f"  Clusters with impact: {len(impact)}")
    print(f"  Ripple effects: {len(ripple)}")
