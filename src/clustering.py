"""
Clustering module for Trixie-flipkartgridlock.
HDBSCAN spatial clustering with KD-tree adjacency for O(n log n) neighbor lookup.
"""
import numpy as np
import pandas as pd
import hdbscan
from scipy.spatial import KDTree
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
    HDBSCAN_METRIC, HDBSCAN_CLUSTER_SELECTION_METHOD,
    PROPAGATION_RADIUS_KM, CHRONIC_THRESHOLD_PCT
)
from src.utils import is_rush_hour


def cluster_hotspots(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Run HDBSCAN clustering on geospatial data.
    Returns: (df_with_clusters, cluster_profiles)
    """
    print("Running HDBSCAN clustering...")
    
    # Prepare coordinates in radians for haversine
    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    
    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
        core_dist_n_jobs=-1,
        algorithm="best",
    )
    
    df = df.copy()
    df["cluster_id"] = clusterer.fit_predict(coords_rad)
    
    n_clusters = len(set(df["cluster_id"])) - (1 if -1 in df["cluster_id"].values else 0)
    n_noise = (df["cluster_id"] == -1).sum()
    
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise:,}")
    
    # Build cluster profiles
    profiles = build_profiles(df)
    
    # Build KD-tree adjacency
    adjacency = build_adjacency_kdtree(profiles)
    
    # Attach adjacency to profiles
    for cid, neighbors in adjacency.items():
        if cid in profiles:
            profiles[cid]["neighbors"] = neighbors
    
    print(f"  Profiles built: {len(profiles)}")
    
    return df, profiles


def build_profiles(df: pd.DataFrame) -> Dict[int, Dict]:
    """Build per-cluster aggregation profiles."""
    print("Building cluster profiles...")
    
    profiles = {}
    
    # Exclude noise
    clustered = df[df["cluster_id"] != -1].copy()
    
    # Group by cluster
    for cid, group in clustered.groupby("cluster_id"):
        # Centroid
        centroid_lat = group["latitude"].median()
        centroid_lon = group["longitude"].median()
        
        # Basic stats
        total_violations = len(group)
        unique_days = group["date"].nunique()
        
        # Peak hours (top 3)
        hour_counts = group["hour"].value_counts()
        peak_hours = hour_counts.head(3).index.tolist()
        
        # Peak day
        day_counts = group["day_of_week"].value_counts()
        peak_day = day_counts.index[0]
        
        # Dominant violation type
        if "dominant_violation" in group.columns:
            dominant_violation = group["dominant_violation"].mode().iloc[0] if len(group["dominant_violation"].mode()) > 0 else "default"
        else:
            dominant_violation = "default"
        
        # Chronic flag
        total_unique_dates = df["date"].nunique()
        days_appeared = group["date"].nunique()
        is_chronic = (days_appeared / total_unique_dates) >= CHRONIC_THRESHOLD_PCT if total_unique_dates > 0 else False
        
        # Average duration
        avg_duration = group["violation_duration_minutes"].mean() if "violation_duration_minutes" in group.columns else 30.0
        
        # Severity
        avg_severity = group["severity_weight"].mean() if "severity_weight" in group.columns else 0.7
        
        # Daily rate
        date_range = (group["date"].max() - group["date"].min()).days + 1
        daily_rate = total_violations / max(date_range, 1)
        
        # Road features
        road_type = group["road_type"].mode().iloc[0] if "road_type" in group.columns and len(group["road_type"].mode()) > 0 else "Other"
        area = group["area"].mode().iloc[0] if "area" in group.columns and len(group["area"].mode()) > 0 else "Unknown"
        has_junction = int(group["has_junction"].sum() > 0) if "has_junction" in group.columns else 0
        num_lanes = int(group["num_lanes"].mode().iloc[0]) if "num_lanes" in group.columns and len(group["num_lanes"].mode()) > 0 else 2
        
        # Unique vehicles
        unique_vehicles = group["vehicle_id"].nunique() if "vehicle_id" in group.columns else 0
        
        # Unique vehicle types
        unique_vehicle_types = group["vehicle_type"].nunique() if "vehicle_type" in group.columns else 0
        
        profiles[cid] = {
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "total_violations": total_violations,
            "unique_days": unique_days,
            "peak_hours": peak_hours,
            "peak_day": int(peak_day),
            "dominant_violation": dominant_violation,
            "is_chronic": is_chronic,
            "avg_duration_minutes": avg_duration,
            "avg_severity": avg_severity,
            "daily_rate": daily_rate,
            "road_type": road_type,
            "area": area,
            "has_junction": has_junction,
            "num_lanes": num_lanes,
            "unique_vehicles": unique_vehicles,
            "unique_vehicle_types": unique_vehicle_types,
            "neighbors": [],
        }
    
    print(f"  Built {len(profiles)} profiles")
    return profiles


def build_adjacency_kdtree(profiles: Dict[int, Dict], radius_km: float = None) -> Dict[int, List[int]]:
    """
    Build adjacency list using KD-tree for O(n log n) neighbor lookup.
    Replaces O(n²) pairwise computation.
    """
    if radius_km is None:
        radius_km = PROPAGATION_RADIUS_KM
    
    print(f"Building KD-tree adjacency (radius={radius_km}km)...")
    
    if len(profiles) == 0:
        return {}
    
    # Extract coordinates
    cids = list(profiles.keys())
    coords = np.array([
        [profiles[cid]["centroid_lat"], profiles[cid]["centroid_lon"]]
        for cid in cids
    ])
    
    # Convert to radians for haversine-like distance
    coords_rad = np.radians(coords)
    
    # Build KD-tree
    tree = KDTree(coords_rad)
    
    # Query within radius (convert km to radians: radius_km / 6371)
    radius_rad = radius_km / 6371.0
    
    # Query all points
    adjacency_pairs = tree.query_ball_tree(tree, r=radius_rad)
    
    # Build adjacency dict
    adjacency = {}
    for i, cid in enumerate(cids):
        neighbors = [cids[j] for j in adjacency_pairs[i] if j != i]  # exclude self
        adjacency[cid] = neighbors
    
    # Stats
    avg_neighbors = np.mean([len(v) for v in adjacency.values()]) if adjacency else 0
    print(f"  Avg neighbors per hotspot: {avg_neighbors:.1f}")
    
    return adjacency


def get_timeseries(df: pd.DataFrame, profiles: Dict[int, Dict]) -> pd.DataFrame:
    """Build cluster × hour timeseries matrix."""
    print("Building timeseries matrix...")
    
    clustered = df[df["cluster_id"] != -1].copy()
    
    # Create cluster × hour counts
    ts = clustered.groupby(["cluster_id", "hour"]).size().unstack(fill_value=0)
    
    # Ensure all 24 hours exist
    for h in range(24):
        if h not in ts.columns:
            ts[h] = 0
    
    ts = ts[sorted(ts.columns)]
    
    print(f"  Timeseries shape: {ts.shape}")
    return ts


if __name__ == "__main__":
    from src.data_pipeline import run_pipeline
    df = run_pipeline()
    df, profiles = cluster_hotspots(df)
    ts = get_timeseries(df, profiles)
    print(f"\nClustering complete: {len(profiles)} hotspots")
