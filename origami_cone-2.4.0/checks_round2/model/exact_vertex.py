"""Lemma 8.1 numerically: at an exact Miura vertex H_l = -H_r = lambda G(zeta)."""
import sys
import numpy as np, sys
import os as _os; sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')); sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','model'))
from origami_cone.kinematics import rodrigues
from scipy.optimize import least_squares
def Phi(angles,vals):
    M=np.eye(3)
    for a,v in sorted(zip(angles,vals)):
        M=M@rodrigues(np.array([np.cos(a),np.sin(a)]),v)
    return M
for th in (0.4,1.03,1.4):
  for lam in (1,-1):
    au=th if lam==1 else np.pi-th; ad=2*np.pi-th if lam==1 else np.pi+th
    ang=[0,au,np.pi,ad]
    for zeta in (0.3,1.0,2.0):
        # unknowns Hr,Hl,zd given zu=zeta
        f=lambda p: (Phi(ang,[p[0],zeta,p[1],p[2]])-np.eye(3)).ravel()
        g0=lam*np.cos(th)*zeta
        s=least_squares(f,[-g0,g0,zeta],xtol=1e-15,ftol=1e-15)
        Hr,Hl,zd=s.x
        pred=2*np.arctan(lam*np.cos(th)*np.tan(zeta/2))
        print(th,lam,zeta,"res %.1e"%np.abs(f(s.x)).max(),"zd-zu %.1e"%(zd-zeta),"Hl+Hr %.1e"%(Hl+Hr),"Hl-pred %.1e"%(Hl-pred))
