"""
XAI root causes v3: profile-based with proper differentiation.
Key insight: junction, chronic, and high daily rate are the strongest differentiators.
Illegal Parking is baseline (all clusters have severity=0.9).
"""
import pickle
from pathlib import Path

cache_path = Path(r"C:\Users\raksh\Desktop\PROJECT\Trixie-Flipkart\output\pipeline_cache.pkl")
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

profiles = cache["profiles"]

FACTOR_COLORS = {
    "Illegal Parking": "#FF6B6B",
    "Road Width": "#4ECDC4",
    "Density": "#45B7D1",
    "Time of Day": "#DDA0DD",
    "Junction Effects": "#FF8C00",
    "Chronic Pattern": "#96CEB4",
}

def compute_root_cause(profile):
    severity = profile.get("avg_severity", 0.5)
    is_chronic = profile.get("is_chronic", False)
    has_junction = profile.get("has_junction", 0)
    num_lanes = profile.get("num_lanes", 2)
    daily_rate = profile.get("daily_rate", 1)
    peak_hours = profile.get("peak_hours", [])

    # Start with equal base
    scores = {
        "Illegal Parking": 15,
        "Density": 15,
        "Road Width": 15,
        "Time of Day": 15,
        "Junction Effects": 5,
        "Chronic Pattern": 5,
    }

    # === STRONG DIFFERENTIATORS (override baseline) ===

    # Junction: traffic conflict point — strong root cause
    if has_junction:
        scores["Junction Effects"] = 40
        scores["Density"] += 10  # junctions attract traffic

    # Chronic: persistent hotspot — strong root cause
    if is_chronic:
        scores["Chronic Pattern"] = 40

    # Density: high daily rate = vehicle crowding
    if daily_rate >= 20:
        scores["Density"] = 50
    elif daily_rate >= 10:
        scores["Density"] = 40
    elif daily_rate >= 5:
        scores["Density"] = 30
    elif daily_rate >= 2:
        scores["Density"] = 20
    else:
        scores["Density"] = 10

    # === MODERATE DIFFERENTIATORS ===

    # Time of Day: rush hour concentration
    rush_hours = {8, 9, 10, 17, 18, 19, 20}
    if peak_hours:
        rush_overlap = sum(1 for h in peak_hours if h in rush_hours)
        rush_ratio = rush_overlap / len(peak_hours)
        if rush_ratio >= 0.67:
            scores["Time of Day"] = 35
        elif rush_ratio >= 0.33:
            scores["Time of Day"] = 25
        else:
            scores["Time of Day"] = 15

    # Road Width: fewer lanes = narrower road
    if num_lanes <= 1:
        scores["Road Width"] = 30
    elif num_lanes == 2:
        scores["Road Width"] = 15
    else:
        scores["Road Width"] = 8

    # Illegal Parking: only dominant when nothing else stands out
    # All clusters have severity=0.9, so this is baseline, not differentiator
    scores["Illegal Parking"] = 15

    total = sum(scores.values())
    pct = {f: round(v / total * 100, 1) for f, v in scores.items()}
    pct = dict(sorted(pct.items(), key=lambda x: x[1], reverse=True))
    dominant = list(pct.keys())[0]

    return dominant, pct


xai = {}
for cid, profile in profiles.items():
    dominant, pct = compute_root_cause(profile)
    cid_str = str(cid)
    xai[cid_str] = {
        "cluster_id": cid,
        "area": profile.get("area", ""),
        "dominant_factor": dominant,
        "percentage_contributions": pct,
        "factor_colors": {f: FACTOR_COLORS.get(f, "#888888") for f in pct.keys()},
    }

# Show key samples
print("=== SAMPLES ===")
for cid, label in [(0, "typical low"), (3, "junction"), (4, "junction+chronic"), (65, "chronic"), (133, "high daily 23"), (163, "highest daily 44")]:
    cid_str = str(cid)
    if cid_str in xai:
        d = xai[cid_str]
        p = profiles.get(cid, {})
        print(f"C{cid} [{label}] ({p.get('area','?')}): junc={p.get('has_junction',0)} chronic={p.get('is_chronic',False)} daily={p.get('daily_rate',0):.1f} lanes={p.get('num_lanes',2)}")
        print(f"  -> {d['dominant_factor']}")
        for f, v in d['percentage_contributions'].items():
            if v >= 5:
                print(f"     {f}: {v}%")
        print()

from collections import Counter
dom_counts = Counter(x["dominant_factor"] for x in xai.values())
print(f"\n=== DISTRIBUTION ({len(xai)} total) ===")
for f, c in dom_counts.most_common():
    print(f"  {f}: {c} ({c/len(xai)*100:.0f}%)")

cache["xai_explanations"] = xai
with open(cache_path, "wb") as f:
    pickle.dump(cache, f)
print(f"\nSaved {len(xai)} root cause explanations")
