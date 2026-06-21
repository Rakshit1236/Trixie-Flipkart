import requests, json

BASE = 'https://rakshit1236-trixie-backend.hf.space'

for ep in ['/hotspots', '/impact', '/scores', '/predictions', '/analytics/recommendations', '/analytics/warnings']:
    r = requests.get(BASE + ep, timeout=30)
    d = r.json()
    print(f'\n=== {ep} ===')
    print(f'Status: {r.status_code}')
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, dict):
                fk = list(v.keys())[:1]
                if fk:
                    sample = v[fk[0]]
                    print(f'  {k}: dict({len(v)} items), sample_key={fk[0]}, sample_keys={list(sample.keys()) if isinstance(sample, dict) else type(sample).__name__}')
                else:
                    print(f'  {k}: dict(0 items)')
            elif isinstance(v, list):
                print(f'  {k}: list({len(v)} items)')
                if v:
                    s = v[0]
                    if isinstance(s, dict):
                        print(f'    first_keys: {list(s.keys())}')
                    else:
                        print(f'    first: {s}')
            else:
                print(f'  {k}: {v}')
