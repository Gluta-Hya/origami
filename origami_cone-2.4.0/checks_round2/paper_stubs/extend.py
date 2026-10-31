import sys
"""Physical stubs: each stub is a straight crease from v* to the edge of the sheet.
Every crease it crosses gets a degree-4 crossing vertex."""
import numpy as np, sys
import os as _os; sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')); sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','model'))
from origami_cone import miura
from origami_cone.patterns import from_vertices
from origami_cone.cone import fast_component_count

def sheet_boundary(P0, nr, nc):
    ring=[(i,0) for i in range(nc+2)]+[(nc+1,j) for j in range(1,nr+2)]+[(i,nr+1) for i in range(nc,-1,-1)]+[(0,j) for j in range(nr,0,-1)]
    return [(ring[k],ring[(k+1)%len(ring)]) for k in range(len(ring))]

def seg_hit(c,u,a,b):
    # ray c+r u (r>0) vs segment a+s(b-a), s in [0,1]
    d=b-a; M=np.array([[u[0],-d[0]],[u[1],-d[1]]]); det=np.linalg.det(M)
    if abs(det)<1e-14: return None
    r,s=np.linalg.solve(M,a-c)
    if r>1e-12 and -1e-12<=s<=1+1e-12: return r,s
    return None

def sector_angles(P0,v):
    k=P0.interior.index(v); return sorted(t[0] for t in P0.vdata[k])

def build_physical(nr,nc,vstar,places,d=None,stop_at_edge=True,a=1):
    from fractions import Fraction as _F
    P0=miura(nr,nc,d=(_F(3,5) if d is None else d),a=a)
    pos={k:np.array(v,float) for k,v in P0.pos.items()}
    interior=list(P0.interior)
    edges=[tuple(e) for e in (tuple(k) for k in P0.cidx)]
    origin={e:e for e in edges}
    bnd=sheet_boundary(P0,nr,nc)
    A=sector_angles(P0,vstar)+[2*np.pi]
    c=pos[vstar]; info=[]
    for n,(sec,frac) in enumerate(places):
        phi=A[sec]+frac*(A[sec+1]-A[sec]); u=np.array([np.cos(phi),np.sin(phi)])
        # first boundary hit
        rb=min(h[0] for a,b in bnd if (h:=seg_hit(c,u,pos[a],pos[b])) is not None)
        hits=[]
        for (a,b) in edges:
            if vstar in (a,b): continue
            h=seg_hit(c,u,pos[a],pos[b])
            if h and h[0]<rb-1e-12:
                if h[1]<1e-9 or h[1]>1-1e-9: raise ValueError("ray through a vertex")
                hits.append((h[0],(a,b)))
        hits.sort()
        prev=vstar; crossed=[]
        for r,(a,b) in hits:
            X=("x",n,len(crossed)); pos[X]=c+r*u; interior.append(X)
            edges.remove((a,b)); edges+= [(a,X),(X,b)]
            o=origin.pop((a,b)); origin[(a,X)]=o; origin[(X,b)]=o
            edges.append((prev,X)); origin[(prev,X)]=("ray",n); prev=X; crossed.append(o)
        E=("e",n); pos[E]=c+rb*u; edges.append((prev,E)); origin[(prev,E)]=("ray",n)
        info.append(dict(phi=phi,crossed=crossed))
    P=from_vertices(pos,interior,edges,name="physical")
    return P,info

if __name__=="__main__":
    for pl in ([(1,.5)],[(1,.5),(2,.5)],[(0,.5),(3,.5)],[(0,.5),(1,.5)],[(0,.5),(2,.5)],[(1,.03),(2,.97)],[(0,.5),(1,.5),(2,.5),(3,.5)]):
        P,info=build_physical(3,4,(2,2),pl)
        print(pl,[len(i["crossed"]) for i in info],fast_component_count(P))
