import pickle, json, numpy as np, pandas as pd
from pathlib import Path

cache_path = r"C:\Users\raksh\Desktop\PROJECT\Trixie-Flipkart\output\pipeline_cache.pkl"
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

if "df" in cache:
    del cache["df"]

for key in list(cache.keys()):
    if isinstance(cache[key], pd.DataFrame):
        cache[key] = cache[key].to_dict(orient="records")

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return super().default(obj)

json_path = r"C:\Users\raksh\Desktop\PROJECT\Trixie-Flipkart\output\pipeline_cache.json"
with open(json_path, "w") as f:
    json.dump(cache, f, cls=Encoder, default=str)

size = Path(json_path).stat().st_size / 1024 / 1024
print(f"JSON cache: {size:.1f} MB")
print(f"Keys: {list(cache.keys())}")
print(f"XAI explanations: {len(cache.get('xai_explanations', {}))}")
print(f"Sample area: {list(cache['profiles'].values())[0].get('area', 'N/A')}")
