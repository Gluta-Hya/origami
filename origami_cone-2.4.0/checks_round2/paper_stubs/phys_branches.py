"""Theorem 9.3, earlier check (superseded by
checks_round3/paper_stubs_sample.py): N and N_br when every stub leaves
through the top or bottom edge."""
import sys
import numpy as np, sys
import os as _os; sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')); sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','model'))
from extend import *
from phys_sweep import is_long
from origami_cone.cone import kernel, incremental_linear_cells
from branches_all import sample_iso, lifts
def own_count(P):
    B,_=kernel(P); cells,blocks=incremental_linear_cells(P); Qv=[p for k,p in blocks if k=="quadric"][0]
    rng=np.random.default_rng(0); alive=own=0
    for idx,(d,N) in enumerate(cells):
        M=N.T@Qv@N; M=(M+M.T)/2; x,st=sample_iso(M,rng)
        if st=="definite": continue
        alive+=1
        for trial in range(8):
            x,st=sample_iso(M,rng); X=B@N@x; X/=np.linalg.norm(X)
            only=all(np.linalg.norm(X-B@N2@(N2.T@(B.T@X)))>1e-6 for j,(d2,N2) in enumerate(cells) if j!=idx)
            if only and lifts(P,X): own+=1; break
    return alive,own,len(cells)
if __name__=="__main__":
  rng=np.random.default_rng(int(sys.argv[1]) if len(sys.argv)>1 else 1)
  tot=bad=0
  while tot<int(sys.argv[2]) if len(sys.argv)>2 else 30:
      nr=int(rng.integers(3,5)); nc=int(rng.integers(4,7)); i0=int(rng.integers(2,nc)); j0=int(rng.integers(2,nr))
      t=int(rng.integers(1,5)); pl=[(int(rng.integers(0,4)),float(rng.uniform(.2,.95))) for _ in range(t)]
      try: P,info=build_physical(nr,nc,(i0,j0),pl)
      except ValueError: continue
      if not is_long(P,info,nr,nc,(i0,j0)): continue
      nu=sum(1 for inf in info if np.sin(inf["phi"])>0); nd=t-nu; B=2**nc
      Nbr=B+(nu>=1)+(nd>=1)+(nu>=1)*(nd>=1); Np=Nbr+(nu>=2)+(nd>=2)
      alive,own,L=own_count(P); tot+=1
      ok=(alive==Np and own==Nbr); bad+=not ok
      print(nr,nc,(i0,j0),"nu,nd",nu,nd,"N",alive,"pred",Np,"own",own,"pred",Nbr,"|L|",L,"OK" if ok else "BAD")
  print("agree",tot-bad,"/",tot)
