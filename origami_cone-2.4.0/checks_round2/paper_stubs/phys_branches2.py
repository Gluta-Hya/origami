"""Theorem 9.3, earlier check (superseded by
checks_round3/paper_stubs_sample.py): N_br on random configurations."""
import numpy as np, sys
from phys_branches import own_count
from phys_general import rowsets, predict
from extend import build_physical
rng=np.random.default_rng(int(sys.argv[1])); tot=bad=0
while tot<int(sys.argv[2]):
    nr=int(rng.integers(3,6)); nc=int(rng.integers(2,6)); i0=int(rng.integers(1,nc+1)); j0=int(rng.integers(2,nr))
    t=int(rng.integers(1,5)); pl=[(int(rng.integers(0,4)),float(rng.uniform(.03,.97))) for _ in range(t)]
    try: P,info=build_physical(nr,nc,(i0,j0),pl)
    except ValueError: continue
    rs=rowsets(info)
    if any(len(R)==0 for u,R in rs): continue
    long_=all((R==frozenset(range(j0+1,nr+1)) if u else R==frozenset(range(1,j0))) for u,R in rs)
    if long_ and rng.random()<0.7: continue      # favour non-long cases
    Np,Nbr=predict(info,nc); alive,own,L=own_count(P); tot+=1
    ok=(alive==Np and own==Nbr); bad+=not ok
    if not ok: print("BAD",nr,nc,(i0,j0),pl,alive,Np,own,Nbr)
print("general (every ray crosses a horizontal line): agree",tot-bad,"/",tot)
