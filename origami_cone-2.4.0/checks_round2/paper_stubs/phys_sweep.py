"""Theorem 9.3, earlier check (superseded by
checks_round3/paper_stubs_sample.py): N when every stub leaves through the top
or bottom edge."""
import numpy as np, sys
from extend import *
def rows_crossed(info,P0pos,nr,vstar):
    pass
def is_long(P,info,nr,nc,vstar):
    # every ray crosses every horizontal line on its side
    j0=vstar[1]; ok=True
    for inf in info:
        up=np.sin(inf["phi"])>0
        rows=set()
        for (a,b) in inf["crossed"]:
            if a[1]==b[1] and not (isinstance(a[0],str)): rows.add(a[1])
        need=set(range(j0+1,nr+1)) if up else set(range(1,j0))
        if not need<=rows: ok=False
    return ok
def pred(places,P0,vstar):
    A=sorted(t[0] for t in P0.vdata[P0.interior.index(vstar)])
    nu=sum(1 for s,f in places if s in (0,1)); nd=len(places)-nu
    B=2**P0.meta.get("nc",0) if False else None
    return nu,nd
if __name__=="__main__":
    rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 0)
    tot=bad=nlong=0; nonlong=[]
    for it in range(int(sys.argv[2]) if len(sys.argv)>2 else 60):
        nr=int(rng.integers(3,6)); nc=int(rng.integers(2,6))
        i0=int(rng.integers(1,nc+1)); j0=int(rng.integers(2,nr))
        t=int(rng.integers(1,5)); pl=[(int(rng.integers(0,4)),float(rng.uniform(.05,.95))) for _ in range(t)]
        try: P,info=build_physical(nr,nc,(i0,j0),pl)
        except ValueError: continue
        nu=sum(1 for inf in info if np.sin(inf["phi"])>0); nd=t-nu
        N,L=fast_component_count(P); B=2**nc
        Nbr=B+(nu>=1)+(nd>=1)+(nu>=1)*(nd>=1); Np=Nbr+(nu>=2)+(nd>=2)
        lg=is_long(P,info,nr,nc,(i0,j0))
        if lg:
            tot+=1; nlong+=1; bad+=(N!=Np)
            if N!=Np: print("MISMATCH",nr,nc,(i0,j0),pl,N,L,Np)
        else: nonlong.append((nr,nc,(i0,j0),pl,N,L,Np))
    print("long rays: agree",tot-bad,"/",tot)
    print("non-long cases:",len(nonlong))
    for r in nonlong[:12]: print("  ",r)
