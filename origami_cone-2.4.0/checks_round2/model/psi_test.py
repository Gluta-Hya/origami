"""Theorem 8.2: the map Xi sends every flat to exact solutions at every vertex
other than v*."""
import sys
import numpy as np, sys
import os as _os; sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')); sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','model'))
from origami_cone import miura, raise_degree_at
from origami_cone.cone import kernel, incremental_linear_cells
from origami_cone.kinematics import vertex_closure
from implant_coords import labels
from branches import classify
def theta_of(P0):
    it=sorted(t[0] for t in P0.vdata[0]); zp=[a for a in it if 0<a<np.pi][0]
    return zp if zp<np.pi/2 else np.pi-zp
def Psi(x,Zidx,th):
    y=x.copy(); c=np.cos(th)
    for e in Zidx: y[e]=2*np.arctan(np.tan(x[e]*c/2)/c)
    return y
rng=np.random.default_rng(2)
for nr,nc,i0,j0,pl in [(3,3,2,2,[]),(3,4,2,2,[(0,.5),(1,.5)]),(4,4,3,2,[(1,.3),(2,.6),(3,.5)]),(3,4,2,2,[(2,.4)])]:
    P0=miura(nr,nc); th=theta_of(P0)
    if pl:
        slot=[k for k,v in enumerate(P0.interior) if v==(i0,j0)][0]; P=raise_degree_at(P0,slot,pl)
    else: P=P0; slot=None
    B,_=kernel(P); H,Z,ST=labels(P); Zidx=list(Z.values())
    cells,_=incremental_linear_cells(P)
    worst_other=0; worst_lin=0
    for d,N in cells:
        for _ in range(3):
            x=B@N@rng.normal(size=d); x*=0.6/np.abs(x).max()
            y=Psi(x,Zidx,th)
            for k in range(P.V):
                if k==slot: continue
                worst_other=max(worst_other,np.abs(vertex_closure(P,k,y)-np.eye(3)).max())
                worst_lin=max(worst_lin,np.abs(vertex_closure(P,k,x)-np.eye(3)).max())
    print(nr,nc,pl,"cells",len(cells),"max closure error at v!=v*: Psi(x) %.1e   (plain x %.1e)"%(worst_other,worst_lin))
