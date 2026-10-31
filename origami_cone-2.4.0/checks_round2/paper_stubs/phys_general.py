"""Theorem 9.3: the predicted N from the crossed rows R_k (rowsets, predict), and
an earlier random check of N."""
import numpy as np, sys, itertools
from extend import *
def rowsets(info):
    out=[]
    for inf in info:
        R=frozenset(a[1] for (a,b) in inf["crossed"] if a[1]==b[1])
        out.append((np.sin(inf["phi"])>0,R))
    return out
def closed_sets(Rs):
    n=len(Rs); res=[]
    for m in range(1,n+1):
        for S in itertools.combinations(range(n),m):
            RS=frozenset().union(*[Rs[k] for k in S])
            if set(k for k in range(n) if Rs[k]<=RS)==set(S): res.append(S)
    return res
def predict(info,nc):
    rs=rowsets(info); B=2**nc
    up=[R for u,R in rs if u]; dn=[R for u,R in rs if not u]
    cu=closed_sets(up); cd=closed_sets(dn)
    Nbr=B+len(cu)+len(cd)+len(cu)*len(cd)
    N=Nbr+sum(1 for S in cu if len(S)>=2)+sum(1 for S in cd if len(S)>=2)
    return N,Nbr
if __name__=="__main__":
    rng=np.random.default_rng(int(sys.argv[1])); tot=bad=0; skipped=0
    for it in range(int(sys.argv[2])):
        nr=int(rng.integers(3,6)); nc=int(rng.integers(2,6)); i0=int(rng.integers(1,nc+1)); j0=int(rng.integers(2,nr))
        t=int(rng.integers(1,5)); pl=[(int(rng.integers(0,4)),float(rng.uniform(.03,.97))) for _ in range(t)]
        try: P,info=build_physical(nr,nc,(i0,j0),pl)
        except ValueError: continue
        if any(len(R)==0 for u,R in rowsets(info)): skipped+=1; continue
        N,L=fast_component_count(P); Np,Nbr=predict(info,nc); tot+=1; bad+=(N!=Np)
        if N!=Np: print("MISMATCH",nr,nc,(i0,j0),pl,N,L,Np,rowsets(info))
    print("every ray crosses >=1 horizontal line: agree",tot-bad,"/",tot," (skipped short rays:",skipped,")")
