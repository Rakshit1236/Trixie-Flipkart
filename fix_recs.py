import pickle, json

cache_path = r'C:\Users\raksh\Desktop\PROJECT\Trixie-Flipkart\output\pipeline_cache.pkl'
with open(cache_path, "rb") as f:
    cache = pickle.load(f)

xai = cache.get("xai_explanations", {})
recs = cache.get("analytics", {}).get("recommendations", [])

ACTION_MAP = {
    "Illegal Parking": {
        "action": "Deploy Enforcement + Boot/Tow",
        "detail": "High illegal parking concentration - deploy officers to boot or tow repeat offenders during peak hours",
        "resource": "Enforcement Team + Tow Truck",
    },
    "Density": {
        "action": "Reroute + Deploy Traffic Officers",
        "detail": "Vehicle density exceeds capacity - redirect traffic to alternate routes and deploy officers at entry points",
        "resource": "Traffic Police + Reroute Signs",
    },
    "Junction Effects": {
        "action": "Signal Timing + Officer at Junction",
        "detail": "Junction conflict causing cascade - optimize signal timing and station officer to manage flow",
        "resource": "Traffic Police + Signal Tech",
    },
    "Time of Day": {
        "action": "Rush Hour Deployment Surge",
        "detail": "Peak-hour violation concentration - surge deployment 30 min before rush hour",
        "resource": "Mobile Patrol Units",
    },
    "Chronic Pattern": {
        "action": "Persistent Hotspot Crackdown",
        "detail": "Chronic recurring hotspot - sustained weekly enforcement blitz with camera monitoring",
        "resource": "Enforcement Team + CCTV",
    },
    "Road Width": {
        "action": "No-Parking Zone + Signage",
        "detail": "Narrow road with limited capacity - enforce strict no-parking zone and install clear signage",
        "resource": "Signage Team + Enforcement",
    },
}
DEFAULT_ACTION = {"action": "Deploy Officers", "detail": "Standard deployment to monitor and enforce parking regulations", "resource": "Patrol Unit"}

for rec in recs:
    cid_str = str(rec.get("cluster_id", ""))
    dominant = xai.get(cid_str, {}).get("dominant_factor", "Unknown")
    cfg = ACTION_MAP.get(dominant, DEFAULT_ACTION)
    rec["dominant_factor"] = dominant
    rec["action"] = cfg["action"]
    rec["action_detail"] = cfg["detail"]
    rec["resource_type"] = cfg["resource"]

# Show first 3
for r in recs[:3]:
    print(f"C{r['cluster_id']} ({r['area']}): {r['dominant_factor']} -> {r['action']}")

from collections import Counter
actions = Counter(r["action"] for r in recs)
print(f"\nAction distribution:")
for a, c in actions.most_common():
    print(f"  {a}: {c}")

with open(cache_path, "wb") as f:
    pickle.dump(cache, f)
print(f"\nUpdated {len(recs)} recommendations")
