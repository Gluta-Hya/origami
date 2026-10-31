"""Consolidated check of the closed form for bisecting stubs (Appendix A, Table 3).
The sampled meshes are the Miura patch and three flat-foldable meshes."""
import numpy as np, json, time, sys
from collections import Counter
from origami_cone import (miura, flat_foldable_quad, raise_degree_at, fast_component_count,
                          predict_pattern, narrow_side)
def samp(s,nr,nc):
    rng=np.random.default_rng(10_000+s)
    if s==0: return miura(nr,nc)
    for _ in range(60):
        t=rng.uniform(0.25,1.6,nc)*rng.choice([-1,1],nc); h=rng.uniform(0.6,1.6,nr+1)
        try: return flat_foldable_quad(nr,nc,t,h,spacing=1.0)
        except ValueError: pass
PLACE={"single":[[(k,.5)] for k in range(4)],
       "pair":[[(a,.5),(b,.5)] for a in range(4) for b in range(a+1,4)],
       "same":[[(k,1/3),(k,2/3)] for k in range(4)],
       "deg8":[[(0,.5),(1,.5),(2,.5),(3,.5)]]}
res=Counter(); fails=[]; t0=time.time(); lean=Counter()
sizes=[(3,2),(3,3),(3,4),(3,5),(3,6),(4,3),(4,4),(4,5),(5,3),(5,4)]
for nr,nc in sizes:
    for s in (0,1,2,3):
        P0=samp(s,nr,nc)
        for slot,(i,j) in enumerate(P0.interior):
            lean[narrow_side(P0,slot)]+=1
            for cls,pls in PLACE.items():
                for pl in pls:
                    a,_=fast_component_count(raise_degree_at(P0,slot,pl))
                    pr=predict_pattern(P0,nr,nc,slot,pl)
                    key=cls+("-outer" if j in (1,nr) else "")
                    res[(key,a==pr)]+=1
                    if a!=pr: fails.append((nr,nc,s,i,j,pl,a,pr))
    print(f"{nr}x{nc} done {time.time()-t0:.0f}s", flush=True)
tot=sum(res.values()); ok=sum(v for (k,b),v in res.items() if b)
summary={"total":tot,"ok":ok,"by_class":{f"{k}|{b}":v for (k,b),v in res.items()},
         "lean":dict(lean),"fails":[str(f) for f in fails[:20]],"sizes":sizes,"positions":4}
json.dump(summary,open("paper_verify.json","w"),indent=1)
print(json.dumps(summary,indent=1))
