"""Proposition 11.1: incremental intersection with and without pruning to maximal
subspaces; state counts and times. The vertices are processed in the order of
P.interior, which is row by row (row j = 1, 2, ..., left to right in each row);
with order="col" they are processed column by column instead. The maximal
number of stored subspaces depends on the order (e.g. 64 row by row and 52
column by column for the 5x5 patch with a degree-8 implant), the final flats
do not."""
import sys
import numpy as np, time, sys
import os as _os; sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')); sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','model'))
from origami_cone import miura, raise_degree_at
from origami_cone.cone import kernel, hessian_blocks, classify_block, incremental_linear_cells
def inc_cells(P, prune=True, tol_nest=1e-7, order="row"):
    B,_=kernel(P); k=B.shape[1]
    blocks=[classify_block(B.T@Q@B,1e-9) for Q in hessian_blocks(P)]
    idx=range(len(blocks))
    if order=="col": idx=sorted(idx,key=lambda k:(P.interior[k][0],P.interior[k][1]))
    lin=[blocks[k][1] for k in idx if blocks[k][0]=="linear"]
    states=[np.eye(k)]; smax=1
    for factors in lin:
        nxt=[]
        for N in states:
            for f in factors:
                w=N.T@np.asarray(f,float); nw=np.linalg.norm(w)
                if nw<1e-10: M=N
                else:
                    if N.shape[1]==1: continue
                    _,_,Vt=np.linalg.svd(np.atleast_2d(w/nw)); M=N@Vt[1:].T
                if not any(M.shape[1]==O.shape[1] and np.linalg.norm(M-O@(O.T@M))<tol_nest for O in nxt): nxt.append(M)
        if prune:
            nxt.sort(key=lambda X:-X.shape[1]); keep=[]
            for N in nxt:
                if not any(N.shape[1]<=O.shape[1] and np.linalg.norm(N-O@(O.T@N))<tol_nest for O in keep): keep.append(N)
            nxt=keep
        states=nxt; smax=max(smax,len(states))
    keep=[]
    for N in sorted(states,key=lambda X:-X.shape[1]):
        if not any(N.shape[1]<=O.shape[1] and np.linalg.norm(N-O@(O.T@N))<tol_nest for O in keep): keep.append(N)
    return keep,smax
for nr,nc,v,pl in [(3,4,(2,2),[(1,.5),(2,.5)]),(4,5,(3,2),[(0,.5),(2,.5)]),(5,5,(3,3),[(0,.5),(1,.5),(2,.5),(3,.5)]),(5,6,(3,3),[(1,.5)])]:
    P0=miura(nr,nc); P=raise_degree_at(P0,P0.interior.index(v),pl)
    t=time.time(); a,s1=inc_cells(P,False); t1=time.time()-t
    t=time.time(); b,s2=inc_cells(P,True); t2=time.time()-t
    same=len(a)==len(b) and all(any(X.shape==Y.shape and np.linalg.norm(X-Y@(Y.T@X))<1e-7 for Y in b) for X in a)
    c,s3=inc_cells(P,True,order="col")
    same_c=len(c)==len(b) and all(any(X.shape==Y.shape and np.linalg.norm(X-Y@(Y.T@X))<1e-7 for Y in b) for X in c)
    print(nr,nc,v,"cells",len(a),len(b),"same",same,"max states: unpruned",s1,"pruned row order",s2,
          "pruned column order",s3,"(same flats %s)"%same_c,"time %.2fs %.2fs"%(t1,t2))
