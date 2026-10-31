"""Appendix A, 'Branches': as branches_all.py, on 3x3, 3x4, 4x4, 3x5, 5x3 with
another seed."""
import sys
import numpy as np, sys
import os as _os; sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')); sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','model'))
from origami_cone import miura, raise_degree_at
from origami_cone.cone import kernel, incremental_linear_cells
from origami_cone.kinematics import solve_slice, residual
from implant_coords import labels
from branches import classify
def sample_iso(M,rng,tol=1e-9):
    ev,V=np.linalg.eigh(M); pos=ev>tol; neg=ev<-tol; zer=~(pos|neg); d=len(ev)
    if pos.any() and neg.any():
        wp=V[:,pos]@rng.normal(size=pos.sum()); wn=V[:,neg]@rng.normal(size=neg.sum())
        wz=V[:,zer]@rng.normal(size=zer.sum()) if zer.any() else 0
        return np.sqrt(-(wn@M@wn)/(wp@M@wp))*wp+wn+wz, "indef"
    if pos.any() and neg.any()==False and zer.any() or (neg.any() and not pos.any() and zer.any()):
        return V[:,zer]@rng.normal(size=zer.sum()), "semidef"
    if not pos.any() and not neg.any(): return rng.normal(size=d), "zero"
    return None, "definite"
def lifts(P,X,eps=(0.02,0.01,0.005)):
    ds=[];rs=[]
    for e in eps:
        r=solve_slice(P,X,e,e*X); rs.append(np.linalg.norm(residual(P,r))); ds.append(np.linalg.norm(r/e-X))
    return max(rs)<1e-10 and ds[-1]<0.6*ds[0]+1e-12
def run(nr,nc,i0,j0,pl,rng):
    P0=miura(nr,nc); slot=[k for k,v in enumerate(P0.interior) if v==(i0,j0)][0]
    P=raise_degree_at(P0,slot,pl); B,_=kernel(P); H,Z,ST=labels(P)
    cells,blocks=incremental_linear_cells(P); Qv=[p for k,p in blocks if k=="quadric"][0]
    secs=set(s for s,_ in pl); fakes_pred=int(not({0,1}&secs))+int(not({2,3}&secs))
    alive=own=0; fake_types=[]
    for idx,(d,N) in enumerate(cells):
        M=N.T@Qv@N; M=(M+M.T)/2; x,st=sample_iso(M,rng)
        if st=="definite": continue
        alive+=1; got=None
        for trial in range(6):
            x,st=sample_iso(M,rng)
            X=B@N@x; X/=np.linalg.norm(X)
            only=all(np.linalg.norm(X-B@N2@(N2.T@(B.T@X)))>1e-6 for j,(d2,N2) in enumerate(cells) if j!=idx)
            if only:
                if lifts(P,X): got="own"; break
                got="LIFTFAIL"
        if got=="own": own+=1
        else: fake_types.append(got or classify(B@N@x,Z,i0,nc,nr))
    return alive,own,fakes_pred,fake_types
rng=np.random.default_rng(47); tot=bad=0
for nr,nc in ((3,3),(3,4),(4,4),(3,5),(5,3)):
    P0=miura(nr,nc)
    for (i0,j0) in P0.interior:
        if not 1<j0<nr: continue
        for _ in range(5):
            t=int(rng.integers(1,5)); pl=[(int(rng.integers(0,4)),float(rng.uniform(.05,.95))) for _ in range(t)]
            alive,own,fp,ft=run(nr,nc,i0,j0,pl,rng); tot+=1
            ok=(alive-own==fp)
            bad+=not ok
            if not ok or False: print(nr,nc,(i0,j0),pl,"alive",alive,"own",own,"pred fakes",fp,ft)
print("agree",tot-bad,"/",tot)
