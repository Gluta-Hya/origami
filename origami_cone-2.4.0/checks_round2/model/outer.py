"""Appendix A, 'Outer rows': |L| = N = N_br = B at every outer-row vertex
(Theorem 7.5, Corollary 8.6)."""
import os as _os, sys as _sys; _sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')); _sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, sys
from branches_all import *
rng=np.random.default_rng(4); tot=bad=0
for nr,nc in ((2,3),(3,3),(3,4),(4,4),(3,5),(2,4)):
    P0=miura(nr,nc)
    for (i0,j0) in P0.interior:
        if 1<j0<nr: continue
        for _ in range(3):
            t=int(rng.integers(1,5)); pl=[(int(rng.integers(0,4)),float(rng.uniform(.05,.95))) for _ in range(t)]
            alive,own,fp,ft=run(nr,nc,i0,j0,pl,rng)
            P=raise_degree_at(P0,[k for k,v in enumerate(P0.interior) if v==(i0,j0)][0],pl)
            L=len(incremental_linear_cells(P)[0]); B=2**nc
            tot+=1; ok=(L==B and alive==B and own==B); bad+=not ok
            if not ok: print(nr,nc,(i0,j0),pl,"L",L,"N",alive,"own",own,"B",B)
print("outer rows: |L|=N=N_br=B in",tot-bad,"/",tot)
