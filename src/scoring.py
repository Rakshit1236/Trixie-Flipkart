"""
Scoring module for Trixie-flipkartgridlock.
Priority scoring (0-100) and Parking Risk Index (PRI).
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PRIORITY_WEIGHTS, PRI_WEIGHTS, ROAD_IMPORTANCE, RUSH_HOURS
)
from src.utils import normalize_to_0_100, is_rush_hour, get_time_of_day


def compute_priority_scores(profiles: Dict[int, Dict], impact: Dict[int, Dict]) -> Dict[int, Dict]:
    """
    Compute composite priority scores (0-100) for each hotspot.
    
    priority = w_freq × frequency + w_impact × impact + w_urgency × urgency + w_criticality × criticality
    """
    print("Computing priority scores...")
    
    if not profiles:
        return {}
    
    # Get total unique dates across all profiles
    total_dates = max(p["unique_days"] for p in profiles.values()) if profiles else 1
    
    # Current time for urgency
    now = datetime.now()
    current_hour = now.hour
    
    scores = {}
    
    for cid, profile in profiles.items():
        impact_data = impact.get(cid, {})
        
        # 1. Frequency (0-100)
        frequency = (profile["unique_days"] / max(total_dates, 1)) * 100
        
        # 2. Impact (0-100)
        worst_speed_drop = impact_data.get("worst_speed_drop_pct", 0)
        impact_score = min(worst_speed_drop, 100)
        
        # 3. Urgency (time-relative)
        urgency = _compute_urgency(current_hour, profile.get("peak_hours", []))
        
        # 4. Criticality (road importance + junction bonus)
        criticality = _compute_criticality(profile)
        
        # Weighted composite
        priority = (
            PRIORITY_WEIGHTS["frequency"] * frequency +
            PRIORITY_WEIGHTS["impact"] * impact_score +
            PRIORITY_WEIGHTS["urgency"] * urgency +
            PRIORITY_WEIGHTS["criticality"] * criticality
        )
        
        scores[cid] = {
            "priority_score": round(priority, 2),
            "frequency": round(frequency, 2),
            "impact": round(impact_score, 2),
            "urgency": round(urgency, 2),
            "criticality": round(criticality, 2),
            "severity_class": impact_data.get("severity_class", "LOW"),
        }
    
    # Normalize to 0-100
    all_scores = [s["priority_score"] for s in scores.values()]
    normalized = normalize_to_0_100(np.array(all_scores))
    
    for i, cid in enumerate(scores.keys()):
        scores[cid]["priority_score_normalized"] = round(float(normalized[i]), 2)
    
    print(f"  Computed priority for {len(scores)} hotspots")
    return scores


def _compute_urgency(current_hour: int, peak_hours: List[int]) -> float:
    """Compute urgency based on current time and peak hours."""
    # Check if current hour is a peak hour
    if current_hour in peak_hours:
        return 100.0
    
    # Check if within 2 hours of peak
    for ph in peak_hours:
        if abs(current_hour - ph) <= 2:
            return 70.0
    
    # Check if within 4 hours of peak
    for ph in peak_hours:
        if abs(current_hour - ph) <= 4:
            return 40.0
    
    # Default
    return 10.0


def _compute_criticality(profile: Dict) -> float:
    """Compute criticality based on road importance and junction."""
    road_type = profile.get("road_type", "Other")
    base = ROAD_IMPORTANCE.get(road_type, 0.3) * 100
    
    # Junction bonus
    if profile.get("has_junction", 0):
        base = min(base + 15, 100)
    
    return base


def compute_parking_risk_index(profiles: Dict[int, Dict], impact: Dict[int, Dict],
                                df: pd.DataFrame = None) -> Dict[int, Dict]:
    """
    Compute proprietary Parking Risk Index (PRI).
    
    PRI = 0.4×illegal_parking + 0.3×density + 0.2×road_importance + 0.1×event_score
    All sub-components normalized to 0-100.
    """
    print("Computing Parking Risk Index...")
    
    if not profiles:
        return {}
    
    # Extract raw values for normalization
    cids = list(profiles.keys())
    
    # Illegal parking score (based on violations per day)
    illegal_scores = np.array([
        profiles[cid].get("daily_rate", 0) for cid in cids
    ])
    
    # Density score (based on total violations)
    density_scores = np.array([
        profiles[cid].get("total_violations", 0) for cid in cids
    ])
    
    # Road importance
    road_scores = np.array([
        ROAD_IMPORTANCE.get(profiles[cid].get("road_type", "Other"), 0.3) * 100
        for cid in cids
    ])
    
    # Event score (based on avg severity as proxy)
    event_scores = np.array([
        profiles[cid].get("avg_severity", 0.7) * 100 for cid in cids
    ])
    
    # Normalize each to 0-100
    illegal_norm = normalize_to_0_100(illegal_scores)
    density_norm = normalize_to_0_100(density_scores)
    road_norm = normalize_to_0_100(road_scores)
    event_norm = normalize_to_0_100(event_scores)
    
    # Compute PRI
    pri_scores = {}
    for i, cid in enumerate(cids):
        pri = (
            PRI_WEIGHTS["illegal"] * illegal_norm[i] +
            PRI_WEIGHTS["density"] * density_norm[i] +
            PRI_WEIGHTS["road"] * road_norm[i] +
            PRI_WEIGHTS["event"] * event_norm[i]
        )
        
        pri_scores[cid] = {
            "pri_score": round(float(pri), 2),
            "illegal_component": round(float(illegal_norm[i]), 2),
            "density_component": round(float(density_norm[i]), 2),
            "road_component": round(float(road_norm[i]), 2),
            "event_component": round(float(event_norm[i]), 2),
            "dominant_factor": _get_dominant_factor(
                illegal_norm[i], density_norm[i], road_norm[i], event_norm[i]
            ),
        }
    
    print(f"  Computed PRI for {len(pri_scores)} hotspots")
    return pri_scores


def _get_dominant_factor(illegal: float, density: float, road: float, event: float) -> str:
    """Identify dominant factor in PRI score."""
    factors = {
        "Illegal Parking": illegal,
        "Density": density,
        "Road Importance": road,
        "Event Score": event,
    }
    return max(factors, key=factors.get)


def rank_hotspots(scores: Dict[int, Dict], pri_scores: Dict[int, Dict],
                  profiles: Dict[int, Dict], top_n: int = 20) -> pd.DataFrame:
    """Create ranked DataFrame of hotspots by priority score."""
    print(f"Ranking top {top_n} hotspots...")
    
    rows = []
    for cid in scores:
        profile = profiles.get(cid, {})
        pri = pri_scores.get(cid, {})
        
        rows.append({
            "cluster_id": cid,
            "area": profile.get("area", "Unknown"),
            "road_type": profile.get("road_type", "Other"),
            "total_violations": profile.get("total_violations", 0),
            "daily_rate": round(profile.get("daily_rate", 0), 1),
            "is_chronic": profile.get("is_chronic", False),
            "priority_score": scores[cid]["priority_score_normalized"],
            "severity_class": scores[cid]["severity_class"],
            "pri_score": pri.get("pri_score", 0),
            "dominant_factor": pri.get("dominant_factor", "Unknown"),
            "centroid_lat": profile.get("centroid_lat", 0),
            "centroid_lon": profile.get("centroid_lon", 0),
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("priority_score", ascending=False).head(top_n)
    df = df.reset_index(drop=True)
    df.index = df.index + 1  # 1-indexed rank
    
    print(f"  Top hotspot: cluster {df.iloc[0]['cluster_id']} (score={df.iloc[0]['priority_score']})")
    return df


def run_scoring(profiles: Dict[int, Dict], impact: Dict[int, Dict],
                df: pd.DataFrame = None) -> Tuple[Dict, Dict, pd.DataFrame]:
    """Run full scoring pipeline."""
    scores = compute_priority_scores(profiles, impact)
    pri_scores = compute_parking_risk_index(profiles, impact, df)
    ranked = rank_hotspots(scores, pri_scores, profiles)
    return scores, pri_scores, ranked


if __name__ == "__main__":
    from src.data_pipeline import run_pipeline
    from src.clustering import cluster_hotspots
    from src.traffic_impact import run_impact_analysis
    
    df = run_pipeline()
    df, profiles = cluster_hotspots(df)
    impact, ripple = run_impact_analysis(df, profiles)
    scores, pri_scores, ranked = run_scoring(profiles, impact, df)
    
    print(f"\nScoring complete:")
    print(f"  Priority scores: {len(scores)}")
    print(f"  PRI scores: {len(pri_scores)}")
    print(f"  Ranked hotspots: {len(ranked)}")
    print(ranked.head())
