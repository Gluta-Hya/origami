"""Coordinates (h, z, z', stubs) on ker J0 for a patch with one implanted vertex,
and the quadric form of that vertex in these coordinates (Section 4)."""
import numpy as np
from origami_cone import miura, raise_degree_at
from origami_cone.cone import kernel, hessian_blocks

def labels(P):
    H={};Z={};ST=[]
    for key,e in P.cidx.items():
        a,b=tuple(key)
        if (isinstance(a,tuple) and a and a[0]=="stub") or (isinstance(b,tuple) and b and b[0]=="stub"):
            ST.append(e); continue
        (i1,j1),(i2,j2)=a,b
        if j1==j2: H[(j1,min(i1,i2))]=e
        else: Z[(i1,min(j1,j2))]=e
    return H,Z,sorted(ST)

def coords(nr,nc,i0,j0,places):
    P0=miura(nr,nc)
    slot=[k for k,v in enumerate(P0.interior) if v==(i0,j0)][0]
    P=raise_degree_at(P0,slot,places)
    B,_=kernel(P); d=B.shape[1]
    H,Z,ST=labels(P)
    # coordinate creases: h_j = H(j,0), z_i = Z(i,0), z' = Z(i0, nr), then the stubs
    pick=[H[(j,0)] for j in range(1,nr+1)]+[Z[(i,0)] for i in range(1,nc+1)]+[Z[(i0,nr)]]
    names=[f"h{j}" for j in range(1,nr+1)]+[f"z{i}" for i in range(1,nc+1)]+["z'"]
    for k,s in enumerate(ST):
        if len(pick)<d: pick.append(s); names.append(f"s{k+1}")
    Bs=B[pick]
    if abs(np.linalg.det(Bs))<1e-10: raise RuntimeError("invalid coordinates")
    T=B@np.linalg.inv(Bs)            # crease = T @ koordinat
    return P,slot,T,names,H,Z,ST

if __name__=="__main__":
    np.set_printoptions(precision=5,suppress=True,linewidth=190)
    for places in ([(0,.5)],[(0,.5),(1,.5)]):
        P,slot,T,names,H,Z,ST=coords(3,3,2,2,places)
        print(f"=== 3x3, implant at (2,2), placement {places}, dim ker {T.shape[1]} ===")
        Qi=hessian_blocks(P)[slot]; M=T.T@Qi@T; M=(M+M.T)/2
        ev=np.linalg.eigvalsh(M)
        print("quadric form of the implanted vertex (coordinates:", names,")")
        print("   " + " ".join(f"{n:>9s}" for n in names))
        for r,n in enumerate(names):
            print(f"{n:>4s} " + " ".join(f"{x:9.5f}" for x in M[r]))
        print("  eigen:", np.round(ev,5), " rank", int((np.abs(ev)>1e-9).sum()))
        # row j0: does H(j0, i) still follow h + kappa S ?
        print("  horizontal relations in row j0=2 (column -> row of T):")
        for i in range(0,4):
            print(f"    H(2,{i})  ", np.round(T[H[(2,i)]],5))
        print()
