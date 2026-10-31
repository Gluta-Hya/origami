"""Census of maximal flats by the types in the proof of Theorem 5.2."""
import os as _os, sys as _sys; _sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')); _sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, sys
from collections import Counter
from origami_cone import miura, flat_foldable_quad, raise_degree_at
from origami_cone.cone import kernel, incremental_linear_cells
from implant_coords import labels

def samp(s,nr,nc):
    rng=np.random.default_rng(10_000+s)
    if s==0: return miura(nr,nc)
    for _ in range(60):
        t=rng.uniform(0.25,1.6,nc)*rng.choice([-1,1],nc); h=rng.uniform(0.6,1.6,nr+1)
        try: return flat_foldable_quad(nr,nc,t,h,spacing=1.0)
        except ValueError: pass

def census(P0,nr,nc,i0,j0,places):
    slot=[k for k,v in enumerate(P0.interior) if v==(i0,j0)][0]
    P=raise_degree_at(P0,slot,places); B,_=kernel(P); H,Z,ST=labels(P)
    zm,zp=Z[(i0,0)],Z[(i0,nr)]
    cells,_=incremental_linear_cells(P)
    c=Counter(); dimsA={}
    for d,N in cells:
        R=B@N; nz=lambda v: np.abs(v).max()>1e-9
        w_left=any(nz(R[Z[(i,0)]]) for i in range(1,i0))
        w_right=any(nz(R[Z[(i,0)]]) for i in range(i0+1,nc+1))
        m0,p0=not nz(R[zm]),not nz(R[zp]); eq=not nz(R[zm]-R[zp])
        if not(w_left or w_right):
            key={(True,True):"M00",(False,True):"M-0",(True,False):"M0+",(False,False):"M-+"}[(m0,p0)]
            c[key]+=1; dimsA[key]=d
        elif eq: c["b1"]+=1
        elif (w_left!=w_right) and (m0!=p0): c["b2"]+=1
        else: c["OTHER"]+=1
    return c,dimsA,len(cells)

if __name__=="__main__":
    bad=tot=0
    PL=[[(0,.5)],[(1,.5),(2,.5)],[(0,.5),(1,.5)],[(0,.5),(2,.5)],[(0,1/3),(0,2/3)],[(0,.5),(1,.5),(2,.5),(3,.5)]]
    for nr,nc in ((3,3),(3,4),(4,4),(5,3),(4,5),(5,4)):
        for s in (0,1,2):
            P0=samp(s,nr,nc)
            for (i0,j0) in P0.interior:
                if not 1<j0<nr: continue
                A,C=2**(i0-1),2**(nc-i0); a,c=A-1,C-1; Bn=2**nc
                for pl in PL:
                    cn,dA,n=census(P0,nr,nc,i0,j0,pl); tot+=1
                    want={"M00":1,"M-0":1,"M0+":1,"M-+":1,"b1":Bn-2,"b2":2*a+2*c}
                    ok = all(cn.get(k,0)==v for k,v in want.items()) and cn.get("OTHER",0)==0
                    if not ok:
                        bad+=1
                        if bad<6: print("BEDA",nr,nc,s,(i0,j0),pl,dict(cn),want)
    print(f"census of flat types agrees with the proof: {tot-bad}/{tot}")
