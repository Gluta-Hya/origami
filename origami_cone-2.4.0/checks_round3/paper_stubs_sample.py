"""Appendix A, 'Stubs in paper': one sample, N and N_br together.
Sample: seed 2026; n_r in {3,4,5}, n_c in {3,...,6}, Miura offset d uniform in [0.3,1.2],
v* uniform among vertices of interior rows, t uniform in {1,...,4}, each stub in a uniform
random sector at a uniform fraction in [0.05,0.95]. Configurations in which a stub passes
through a vertex, or some stub crosses no horizontal fold line, are skipped."""
import os as _os, sys as _sys
_h = _os.path.dirname(_os.path.abspath(__file__))
for _p in ("..", _os.path.join("..", "checks_round2", "model"), _os.path.join("..", "checks_round2", "paper_stubs")):
    _sys.path.insert(0, _os.path.join(_h, _p))

import numpy as np, sys
from fractions import Fraction as F
from extend import build_physical
from phys_general import rowsets, predict
from phys_branches import own_count
from origami_cone.cone import fast_component_count
rng=np.random.default_rng(2026); want=int(sys.argv[1]) if len(sys.argv)>1 else 100
acc=okN=okB=drawn=skipv=skips=0
while acc<want:
    drawn+=1
    nr=int(rng.integers(3,6)); nc=int(rng.integers(3,7)); d=F(round(float(rng.uniform(.3,1.2)),3)).limit_denominator(1000)
    j0=int(rng.integers(2,nr)); i0=int(rng.integers(1,nc+1)); t=int(rng.integers(1,5))
    pl=[(int(rng.integers(0,4)),float(rng.uniform(.05,.95))) for _ in range(t)]
    try: P,info=build_physical(nr,nc,(i0,j0),pl,d=d)
    except ValueError: skipv+=1; continue
    if any(len(R)==0 for u,R in rowsets(info)): skips+=1; continue
    Np,Nbr=predict(info,nc); N,L=fast_component_count(P); alive,own,_=own_count(P)
    acc+=1; okN+=(N==Np); okB+=(own==Nbr)
    if N!=Np or own!=Nbr: print("MISMATCH",nr,nc,float(d),(i0,j0),pl,N,Np,own,Nbr)
print(f"drawn {drawn}, skipped {skipv} (through a vertex) and {skips} (a stub crossing no horizontal line); accepted {acc}: N {okN}/{acc}, N_br {okB}/{acc}")
