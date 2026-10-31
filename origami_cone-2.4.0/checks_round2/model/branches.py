"""Theorem 8.4: classifier of flat types (M00, M-0, M0+, M-+, Y_U, Y_W) and the
branch test on one configuration."""
import sys
import numpy as np, sys
import os as _os; sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')); sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','model'))
from origami_cone import miura, raise_degree_at
from origami_cone.cone import kernel, incremental_linear_cells
from origami_cone.kinematics import solve_slice, residual
from implant_coords import labels
def classify(R,Z,i0,nc,nr):
    nz=lambda v: np.abs(v).max()>1e-8
    wl=any(nz(R[Z[(i,0)]]) for i in range(1,i0)); wr=any(nz(R[Z[(i,0)]]) for i in range(i0+1,nc+1))
    zm,zp=Z[(i0,0)],Z[(i0,nr)]
    m0,p0=not nz(R[zm]),not nz(R[zp]); eq=not nz(R[zm]-R[zp])
    if not(wl or wr): return {(True,True):"M00",(False,True):"M-0",(True,False):"M0+",(False,False):"M-+"}[(m0,p0)]
    if eq: return "YU"
    return "F(%s,%s)"%("L" if wl else "R","-" if p0 else "+")
def study(nr,nc,i0,j0,pl,rng,eps=(0.02,0.01,0.005),verbose=True):
    P0=miura(nr,nc); slot=[k for k,v in enumerate(P0.interior) if v==(i0,j0)][0]
    P=raise_degree_at(P0,slot,pl); B,_=kernel(P); H,Z,ST=labels(P)
    cells,blocks=incremental_linear_cells(P); Qv=[p for k,p in blocks if k=="quadric"][0]
    out=[]
    for d,N in cells:
        typ=classify(B@N,Z,i0,nc,nr)
        M=N.T@Qv@N; M=(M+M.T)/2; ev,V=np.linalg.eigh(M); tol=1e-9
        pos=ev>tol; neg=ev<-tol; zer=~(pos|neg)
        if pos.any() and neg.any():
            # generic regular isotropic direction
            c=rng.normal(size=d); 
            wpos=V[:,pos]@rng.normal(size=pos.sum()); wneg=V[:,neg]@rng.normal(size=neg.sum()); wz=V[:,zer]@rng.normal(size=zer.sum()) if zer.any() else 0
            a=np.sqrt(-(wneg@M@wneg)/(wpos@M@wpos)); x=a*wpos+wneg+wz
            X=B@N@x; X/=np.linalg.norm(X)
            ds=[];rs=[]
            for e in eps:
                r=solve_slice(P,X,e,e*X); rs.append(np.linalg.norm(residual(P,r))); ds.append(np.linalg.norm(r/e-X))
            status="indef"; info=(max(rs),ds)
        elif pos.any() or neg.any():
            if zer.any():
                # semidefinite degenerate: kernel vectors -> crease vectors, which type?
                K=B@N@V[:,zer]
                kinds=set(classify(K@rng.normal(size=zer.sum()),Z,i0,nc,nr) for _ in range(3))
                status="semidef"; info=("kernel in",kinds)
            else: status="definite(dies)"; info=None
        else: status="zero"; info=None
        out.append((typ,d,status,info))
    return out
if __name__=="__main__":
    rng=np.random.default_rng(5)
    for nr,nc,i0,j0,pl,name in [(3,3,1,2,[(0,.5),(1,.5)],"upper pair col1 (N=13)"),
                                (3,4,2,2,[(2,.4)],"one stub sector2"),
                                (3,4,2,2,[(1,.5),(2,.5)],"left pair bisect"),
                                (3,4,2,2,[(1,.03),(2,.97)],"left pair near zigzags"),
                                (3,4,2,2,[(0,.5),(2,.5)],"diagonal"),
                                (3,4,3,2,[(0,.5),(1,.5),(2,.5),(3,.5)],"degree 8"),
                                (3,4,2,2,[(0,.2),(0,.8),(3,.5)],"deg7 wall survive")]:
        print("==",name)
        for typ,d,st,info in study(nr,nc,i0,j0,pl,rng):
            if st=="indef": print(f"  {typ:8s} d={d} indef  res={info[0]:.1e} |rho/e - x| = "+", ".join(f"{v:.2e}" for v in info[1]))
            else: print(f"  {typ:8s} d={d} {st} {info if info else ''}")
