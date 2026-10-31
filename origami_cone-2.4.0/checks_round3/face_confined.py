"""Section 9.1, Proposition 9.1 (face-confined stubs): with v* in the first
column of an interior row of M(4,4) and one stub in each of the two left
sectors, sweep the stub angle beta and record N over the face-confined
positions (stubs that cross no crease), for horizontal side a = 1, 1/2, 3/5
and zigzag advance d = 3/5. N = 2B = 32 occurs iff a < d."""
import os as _os, sys as _sys
_h = _os.path.dirname(_os.path.abspath(__file__))
for _p in ("..", _os.path.join("..", "checks_round2", "model"), _os.path.join("..", "checks_round2", "paper_stubs")):
    _sys.path.insert(0, _os.path.join(_h, _p))
import numpy as np
from fractions import Fraction as F
from extend import build_physical
from origami_cone import miura
from origami_cone.cone import fast_component_count
d=F(3,5)
for a in (F(1),F(1,2),F(3,5)):
    for j0 in (2,3):
        P0=miura(4,4,d=d,a=a); k=P0.interior.index((1,j0)); it=sorted(t[0] for t in P0.vdata[k])
        zp=it[1]; lam=1 if zp<np.pi/2 else -1
        A=it+[2*np.pi]; w=A[2]-A[1]
        vals={}; conf_ok=[]
        for beta in np.linspace(0.01,w-0.01,80):
            pl=[(1,1-beta/w),(2,beta/(A[3]-A[2]))]
            try: P,info=build_physical(4,4,(1,j0),pl,a=a)
            except ValueError: continue
            if all(len(i["crossed"])==0 for i in info):
                N=fast_component_count(P)[0]; vals.setdefault(N,[]).append(round(np.degrees(beta),1))
        print(f"a={a} d={d} v*=(1,{j0}) lean={'R' if lam==1 else 'L'} (left sectors {'wide' if lam==1 else 'narrow'}): face-confined N values:",{k:(min(v),max(v)) for k,v in vals.items()}, "B=16, 2B=32")
