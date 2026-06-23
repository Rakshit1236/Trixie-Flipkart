import pickle
f=open(r'C:\Users\raksh\Desktop\PROJECT\Trixie-Flipkart\output\pipeline_cache.pkl','rb')
c=pickle.load(f)
f.close()
p=c['profiles']
xai=c['xai_explanations']
junc=[k for k,v in p.items() if v.get('has_junction',0)==1]
print(f'Junction clusters ({len(junc)}): first 5 = {junc[:5]}')
for k in junc[:3]:
    prof=p[k]
    ex=xai.get(str(k),{})
    print(f'  C{k} ({prof.get("area","?")}): junction={prof.get("has_junction")} lanes={prof.get("num_lanes")} daily={prof.get("daily_rate",0):.1f} -> {ex.get("dominant_factor","?")}')

chro=[k for k,v in p.items() if v.get('is_chronic')]
print(f'\nChronic clusters ({len(chro)}): first 5 = {chro[:5]}')
for k in chro[:3]:
    prof=p[k]
    ex=xai.get(str(k),{})
    print(f'  C{k} ({prof.get("area","?")}): chronic={prof.get("is_chronic")} severity={prof.get("avg_severity",0):.2f} -> {ex.get("dominant_factor","?")}')

high=[k for k,v in p.items() if v.get('daily_rate',0)>=10]
print(f'\nHigh daily rate ({len(high)}): first 5 = {high[:5]}')
for k in high[:3]:
    prof=p[k]
    ex=xai.get(str(k),{})
    print(f'  C{k} ({prof.get("area","?")}): daily={prof.get("daily_rate",0):.1f} severity={prof.get("avg_severity",0):.2f} -> {ex.get("dominant_factor","?")}')
