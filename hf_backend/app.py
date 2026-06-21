"""
FastAPI backend for Trixie-Flipkart.
Serves pre-computed pipeline results + runs simulations on demand.
"""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import numpy as np

app = FastAPI(title="Trixie Parking Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PIPELINE_STATE = {}


def load_cache():
    global PIPELINE_STATE
    hf_repo = os.environ.get("HF_REPO_ID", "Rakshit1236/trixie-data")

    # Load JSON cache (cross-version compatible)
    cache_path = Path("output/pipeline_cache.json")
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=hf_repo, filename="pipeline_cache.json", repo_type="dataset", local_dir="output")
            cache_path = Path("output/pipeline_cache.json")
        except Exception as e:
            print(f"Could not download cache: {e}")
            return False

    if cache_path.exists():
        with open(cache_path, "r") as f:
            PIPELINE_STATE = json.load(f)
        print(f"Loaded cache: {len(PIPELINE_STATE)} keys")
        return True
    return False


@app.on_event("startup")
async def startup():
    if not load_cache():
        print("No cache found — run pipeline locally and upload output/pipeline_cache.json")


@app.get("/reload")
def reload_cache():
    # Delete local cache to force re-download from HuggingFace
    cache_path = Path("output/pipeline_cache.json")
    if cache_path.exists():
        cache_path.unlink()
    load_cache()
    return {"reloaded": len(PIPELINE_STATE) > 0, "keys": list(PIPELINE_STATE.keys())}


@app.get("/")
def root():
    return {"message": "Trixie Parking Intelligence API", "status": "running"}


@app.get("/health")
def health():
    return {
        "status": "healthy" if PIPELINE_STATE else "no_data",
        "pipeline_loaded": bool(PIPELINE_STATE),
        "hotspots": len(PIPELINE_STATE.get("profiles", {})),
        "critical": sum(
            1 for c in PIPELINE_STATE.get("impact", {}).values()
            if c.get("severity_class") == "CRITICAL"
        ),
        "ensemble_r2": PIPELINE_STATE.get("ml_metrics", {}).get("ensemble_r2", 0),
        "ensemble_mae": PIPELINE_STATE.get("ml_metrics", {}).get("ensemble_mae", 0),
        "predictions": len(PIPELINE_STATE.get("predictions", [])),
        "last_updated": PIPELINE_STATE.get("last_updated", "never"),
    }


@app.get("/pipeline/status")
def pipeline_status():
    if not PIPELINE_STATE:
        return {"loaded": False}
    return {
        "loaded": True,
        "hotspots": len(PIPELINE_STATE["profiles"]),
        "critical": sum(
            1 for c in PIPELINE_STATE["impact"].values()
            if c.get("severity_class") == "CRITICAL"
        ),
        "ensemble_r2": PIPELINE_STATE["ml_metrics"]["ensemble_r2"],
        "ensemble_mae": PIPELINE_STATE["ml_metrics"]["ensemble_mae"],
        "predictions": len(PIPELINE_STATE["predictions"]),
        "last_updated": PIPELINE_STATE.get("last_updated"),
    }


@app.get("/hotspots")
def get_hotspots():
    if not PIPELINE_STATE:
        raise HTTPException(status_code=404, detail="Pipeline not loaded")
    profiles = PIPELINE_STATE["profiles"]
    return {
        "count": len(profiles),
        "hotspots": {
            str(k): {
                "centroid_lat": v["centroid_lat"],
                "centroid_lon": v["centroid_lon"],
                "total_violations": v["total_violations"],
                "area": v.get("area", "Unknown"),
                "road_type": v.get("road_type", "Other"),
                "is_chronic": v.get("is_chronic", False),
                "peak_hours": v.get("peak_hours", []),
                "num_lanes": v.get("num_lanes", 2),
                "daily_rate": round(v.get("daily_rate", 0), 1),
            }
            for k, v in profiles.items()
        },
    }


@app.get("/impact")
def get_impact():
    if not PIPELINE_STATE:
        raise HTTPException(status_code=404, detail="Pipeline not loaded")
    return {
        "count": len(PIPELINE_STATE["impact"]),
        "impact": {str(k): v for k, v in PIPELINE_STATE["impact"].items()},
    }


@app.get("/scores")
def get_scores():
    if not PIPELINE_STATE:
        raise HTTPException(status_code=404, detail="Pipeline not loaded")
    ranked = PIPELINE_STATE.get("ranked")
    return {
        "priority": {str(k): v for k, v in PIPELINE_STATE["scores"].items()},
        "pri": {str(k): v for k, v in PIPELINE_STATE["pri_scores"].items()},
        "ranked": ranked if isinstance(ranked, list) else [],
    }


@app.get("/predictions")
def get_predictions():
    if not PIPELINE_STATE:
        raise HTTPException(status_code=404, detail="Pipeline not loaded")
    preds = PIPELINE_STATE["predictions"]
    if isinstance(preds, dict):
        preds = [preds]
    return {
        "count": len(preds),
        "predictions": preds,
    }


@app.get("/analytics/recommendations")
def get_recommendations():
    if not PIPELINE_STATE:
        raise HTTPException(status_code=404, detail="Pipeline not loaded")
    analytics = PIPELINE_STATE.get("analytics", {})
    return {
        "recommendations": analytics.get("recommendations", []),
        "dispatch_report": analytics.get("dispatch_report", ""),
    }


@app.get("/analytics/warnings")
def get_warnings():
    if not PIPELINE_STATE:
        raise HTTPException(status_code=404, detail="Pipeline not loaded")
    warnings = PIPELINE_STATE.get("analytics", {}).get("warnings", {})
    timestamps = PIPELINE_STATE.get("warning_timestamps", {})
    return {
        "warnings": {str(k): v for k, v in warnings.items()},
        "timestamps": timestamps,
    }


@app.get("/root_cause/{cluster_id}")
def get_root_cause(cluster_id: int):
    if not PIPELINE_STATE:
        raise HTTPException(status_code=404, detail="Pipeline not loaded")
    xai = PIPELINE_STATE.get("xai_explanations", {})
    cid_str = str(cluster_id)
    if cid_str not in xai and cluster_id not in xai:
        raise HTTPException(status_code=404, detail=f"No XAI explanation for cluster {cluster_id}")
    return xai.get(cid_str, xai.get(cluster_id, {}))


class ScenarioRequest(BaseModel):
    vehicle_reduction: int = 50
    weather: str = "Clear"
    day_type: str = "Weekday"
    cluster_id: Optional[int] = None
    preset: Optional[str] = None


@app.post("/scenario")
def run_scenario(req: ScenarioRequest):
    if not PIPELINE_STATE:
        raise HTTPException(status_code=404, detail="Pipeline not loaded")

    FREE_FLOW_SPEED = 40.0
    LANE_CAPACITY = 600
    WEATHER_MULT = {"Clear": 1.0, "Light Rain": 1.25, "Heavy Rain": 1.6, "Fog": 1.35}
    DAY_MULT = {"Weekday": 1.0, "Saturday": 1.15, "Sunday": 0.85, "Festival Day": 1.5, "Public Holiday": 0.7}

    PRESETS = {
        "Remove Illegal Parking": {"vehicle_reduction": 50, "weather": "Clear", "day_type": "Weekday"},
        "Festival Day": {"vehicle_reduction": 0, "weather": "Clear", "day_type": "Festival Day"},
        "Heavy Rain": {"vehicle_reduction": 0, "weather": "Heavy Rain", "day_type": "Weekday"},
        "Metro Delay": {"vehicle_reduction": 30, "weather": "Clear", "day_type": "Weekday"},
        "Increase Capacity": {"vehicle_reduction": 100, "weather": "Clear", "day_type": "Weekday"},
    }

    if req.preset and req.preset in PRESETS:
        preset = PRESETS[req.preset]
        vehicle_reduction = preset["vehicle_reduction"]
        weather = preset["weather"]
        day_type = preset["day_type"]
    else:
        vehicle_reduction = req.vehicle_reduction
        weather = req.weather
        day_type = req.day_type

    profiles = PIPELINE_STATE["profiles"]
    impact = PIPELINE_STATE["impact"]
    weather_m = WEATHER_MULT.get(weather, 1.0)
    day_m = DAY_MULT.get(day_type, 1.0)
    combined_m = weather_m * day_m

    def simulate_one(cid, baseline):
        reduction = min(vehicle_reduction, baseline["total_violations"])
        factor = 1 - (reduction / baseline["total_violations"]) if baseline["total_violations"] > 0 else 1
        new_violations = baseline["total_violations"] * factor
        new_speed_drop = baseline["worst_speed_drop_pct"] * factor * combined_m
        new_speed = FREE_FLOW_SPEED * (1 - new_speed_drop / 100)
        capacity = baseline["num_lanes"] * LANE_CAPACITY
        demand = new_violations * (60 / 30)
        rho = min(demand / (capacity + 1e-6), 0.99)
        new_vhl = baseline["total_vhl"] * factor * combined_m
        base_speed = FREE_FLOW_SPEED * (1 - baseline["worst_speed_drop_pct"] / 100)
        return {
            "cluster_id": cid,
            "baseline": {"violations": baseline["total_violations"], "speed_kmh": round(base_speed, 1),
                         "speed_drop_pct": baseline["worst_speed_drop_pct"], "vehicle_hours_lost": baseline["total_vhl"]},
            "scenario": {"violations": round(new_violations, 1), "speed_kmh": round(new_speed, 1),
                         "speed_drop_pct": round(new_speed_drop, 1), "vehicle_hours_lost": round(new_vhl, 2)},
            "improvement": {"violations_reduced": round(baseline["total_violations"] - new_violations, 1),
                            "speed_gained_kmh": round(new_speed - base_speed, 1),
                            "vhl_reduced": round(baseline["total_vhl"] - new_vhl, 2)},
        }

    if req.cluster_id is not None:
        cid = req.cluster_id
        if cid not in profiles and str(cid) not in profiles:
            raise HTTPException(status_code=404, detail=f"Cluster {cid} not found")
        profile = profiles.get(cid, profiles.get(str(cid), {}))
        impact_data = impact.get(cid, impact.get(str(cid), {}))
        baseline = {
            "total_violations": profile.get("total_violations", 100),
            "worst_speed_drop_pct": impact_data.get("worst_speed_drop_pct", 25),
            "total_vhl": impact_data.get("total_vhl", 10),
            "num_lanes": profile.get("num_lanes", 2),
        }
        return simulate_one(cid, baseline)
    else:
        baselines = {}
        for cid, profile in profiles.items():
            impact_data = impact.get(cid, impact.get(str(cid), {}))
            baselines[str(cid)] = {
                "total_violations": profile.get("total_violations", 100),
                "worst_speed_drop_pct": impact_data.get("worst_speed_drop_pct", 25),
                "total_vhl": impact_data.get("total_vhl", 10),
                "num_lanes": profile.get("num_lanes", 2),
            }
        results = {cid: simulate_one(cid, b) for cid, b in baselines.items()}
        total_base = sum(r["baseline"]["violations"] for r in results.values())
        total_scenario = sum(r["scenario"]["violations"] for r in results.values())
        return {
            "n_hotspots": len(results),
            "citywide": {
                "baseline_violations": round(total_base, 1),
                "scenario_violations": round(total_scenario, 1),
                "violations_reduction_pct": round((total_base - total_scenario) / max(total_base, 1) * 100, 1),
            },
            "per_hotspot": results,
        }


class CounterfactualRequest(BaseModel):
    occupancy_reduction_pct: float = 15.0
    cluster_id: Optional[int] = None


@app.post("/counterfactual")
def run_counterfactual(req: CounterfactualRequest):
    if not PIPELINE_STATE:
        raise HTTPException(status_code=404, detail="Pipeline not loaded")

    FREE_FLOW_SPEED = 40.0
    profiles = PIPELINE_STATE["profiles"]
    impact = PIPELINE_STATE["impact"]

    def simulate_cf(cid, baseline):
        reduction_factor = 1 - (req.occupancy_reduction_pct / 100)
        new_violations = baseline["total_violations"] * reduction_factor
        new_speed_drop = baseline["worst_speed_drop_pct"] * reduction_factor
        new_speed = FREE_FLOW_SPEED * (1 - new_speed_drop / 100)
        base_speed = FREE_FLOW_SPEED * (1 - baseline["worst_speed_drop_pct"] / 100)
        return {
            "cluster_id": cid,
            "occupancy_reduction_pct": req.occupancy_reduction_pct,
            "baseline": {"violations": baseline["total_violations"], "speed_kmh": round(base_speed, 1)},
            "scenario": {"violations": round(new_violations, 1), "speed_kmh": round(new_speed, 1)},
            "improvement": {
                "violations_reduced": round(baseline["total_violations"] - new_violations, 1),
                "speed_gained_kmh": round(new_speed - base_speed, 1),
            },
        }

    if req.cluster_id is not None:
        cid = req.cluster_id
        if cid not in profiles and str(cid) not in profiles:
            raise HTTPException(status_code=404, detail=f"Cluster {cid} not found")
        profile = profiles.get(cid, profiles.get(str(cid), {}))
        impact_data = impact.get(cid, impact.get(str(cid), {}))
        baseline = {
            "total_violations": profile.get("total_violations", 100),
            "worst_speed_drop_pct": impact_data.get("worst_speed_drop_pct", 25),
            "total_vhl": impact_data.get("total_vhl", 10),
            "num_lanes": profile.get("num_lanes", 2),
        }
        return simulate_cf(cid, baseline)
    else:
        results = {}
        for cid, profile in profiles.items():
            impact_data = impact.get(cid, impact.get(str(cid), {}))
            baseline = {
                "total_violations": profile.get("total_violations", 100),
                "worst_speed_drop_pct": impact_data.get("worst_speed_drop_pct", 25),
                "total_vhl": impact_data.get("total_vhl", 10),
                "num_lanes": profile.get("num_lanes", 2),
            }
            results[str(cid)] = simulate_cf(cid, baseline)
        total_base = sum(r["baseline"]["violations"] for r in results.values())
        total_scenario = sum(r["scenario"]["violations"] for r in results.values())
        return {
            "n_hotspots": len(results),
            "occupancy_reduction_pct": req.occupancy_reduction_pct,
            "citywide": {
                "baseline_violations": round(total_base, 1),
                "scenario_violations": round(total_scenario, 1),
                "violations_reduction_pct": round((total_base - total_scenario) / max(total_base, 1) * 100, 1),
            },
            "per_hotspot": results,
        }


class PropagationRequest(BaseModel):
    source_cluster_id: int
    start_hour: int = 8
    horizon_minutes: int = 60


@app.post("/propagation")
def run_propagation(req: PropagationRequest):
    if not PIPELINE_STATE:
        raise HTTPException(status_code=404, detail="Pipeline not loaded")

    from collections import deque

    profiles = PIPELINE_STATE["profiles"]
    adjacency = {cid: p.get("neighbors", []) for cid, p in profiles.items()}

    FREE_FLOW_SPEED = 40.0
    LANE_CAPACITY = 600
    PROPAGATION_SPEED = 15.0
    PROPAGATION_MIN_DELAY = 8

    source_cid = req.source_cluster_id
    source_cid_str = str(source_cid)
    if source_cid not in profiles and source_cid_str not in profiles:
        raise HTTPException(status_code=404, detail=f"Source cluster {source_cid} not found")

    source_profile = profiles.get(source_cid, profiles.get(source_cid_str, {}))
    initial_demand = source_profile["daily_rate"] * 10

    timeline = []
    queue = deque([(source_cid_str, 0, initial_demand)])
    visited = {source_cid_str}

    while queue:
        cid_str, minute, demand = queue.popleft()
        if minute > req.horizon_minutes:
            continue

        profile = profiles.get(cid_str, {})
        if not profile:
            continue

        num_lanes = profile.get("num_lanes", 2)
        capacity_vph = num_lanes * LANE_CAPACITY
        rho = min(demand / (capacity_vph + 1e-6), 0.99)
        speed_drop_pct = rho * 100

        if speed_drop_pct >= 40:
            severity = "CRITICAL"
        elif speed_drop_pct >= 25:
            severity = "HIGH"
        elif speed_drop_pct >= 10:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        hour = req.start_hour + (minute // 60)
        minute_of_hour = minute % 60

        timeline.append({
            "minute": minute,
            "time_str": f"{hour % 24}:{minute_of_hour:02d}",
            "cluster_id": cid_str,
            "area": profile.get("area", "Unknown"),
            "road_type": profile.get("road_type", "Other"),
            "speed_drop_pct": round(speed_drop_pct, 1),
            "speed_kmh": round(FREE_FLOW_SPEED * (1 - rho), 1),
            "queue_length": round(rho / (1 - rho) if rho < 1 else 999, 1),
            "severity": severity,
        })

        for nid_str in adjacency.get(cid_str, []):
            if nid_str in visited:
                continue
            neighbor = profiles.get(nid_str, {})
            if not neighbor:
                continue

            from math import radians, sin, cos, sqrt, atan2
            R = 6371.0
            lat1, lon1 = radians(profile["centroid_lat"]), radians(profile["centroid_lon"])
            lat2, lon2 = radians(neighbor["centroid_lat"]), radians(neighbor["centroid_lon"])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
            dist_km = R * 2 * atan2(sqrt(a), sqrt(1-a))

            travel_time = (dist_km / PROPAGATION_SPEED * 60) + PROPAGATION_MIN_DELAY
            next_minute = minute + int(travel_time)
            if next_minute <= req.horizon_minutes:
                decay = 0.85 ** dist_km
                propagated_demand = demand * decay * 0.7
                if propagated_demand > 10:
                    queue.append((nid_str, next_minute, propagated_demand))
                    visited.add(nid_str)

    timeline.sort(key=lambda x: x["minute"])
    return {
        "source_cid": source_cid_str,
        "source_area": source_profile.get("area", "Unknown"),
        "start_hour": req.start_hour,
        "horizon_minutes": req.horizon_minutes,
        "n_affected": len(timeline),
        "timeline": timeline,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
