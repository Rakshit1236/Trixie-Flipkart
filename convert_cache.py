"""
Convert pipeline cache from pickle to JSON.
Only serializes what the backend needs (not the raw DataFrame).
"""
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

INPUT = Path("output/pipeline_cache.pkl")
OUTPUT_JSON = Path("output/pipeline_cache.json")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        if isinstance(obj, (pd.NaT.__class__,)):
            return None
        return super().default(obj)

def sanitize(obj):
    """Deep-sanitize any object for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp) or isinstance(obj, type(pd.NaT)):
        return str(obj)
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

def convert():
    print("Loading pickle cache...")
    with open(INPUT, "rb") as f:
        cache = pickle.load(f)

    # Only keep what the backend API needs
    backend_keys = ["profiles", "impact", "scores", "pri_scores", "predictions", "analytics", "ml_metrics"]
    
    minimal = {}
    for key in backend_keys:
        if key in cache:
            print(f"  Sanitizing {key}...")
            minimal[key] = sanitize(cache[key])

    # Handle ranked (might be DataFrame)
    if "ranked" in cache:
        ranked = cache["ranked"]
        if isinstance(ranked, pd.DataFrame):
            minimal["ranked"] = ranked.to_dict(orient="records")
        else:
            minimal["ranked"] = sanitize(ranked)

    # Convert string keys in dicts
    for key in minimal:
        if isinstance(minimal[key], dict):
            minimal[key] = {str(k): v for k, v in minimal[key].items()}

    print(f"\nSaving JSON cache ({len(minimal)} keys)...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(minimal, f, cls=NumpyEncoder)

    size_mb = OUTPUT_JSON.stat().st_size / 1024 / 1024
    print(f"Saved: {OUTPUT_JSON} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    convert()
