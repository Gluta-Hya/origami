"""Appendix A, 'The death criterion': Theorem 6.7 evaluated at v* alone against
the full count N."""
import os as _os, sys as _sys; _sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')); _sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, sys
from crit import gram_dies
from origami_cone import miura, raise_degree_at, flat_foldable_quad
from origami_cone.cone import fast_component_count
def vinfo(P0,slot):
    it=sorted(t[0] for t in P0.vdata[slot])   # 4 angles
    # Hr at 0 ; zp is between 0 and pi
    zp=[a for a in it if 0<a<np.pi][0]
    lam=1 if zp<np.pi/2 else -1; theta=zp if lam==1 else np.pi-zp
    return theta,lam,it
def run(P0,nr,nc,rng,trials,label):
    bad=tot=0; nondef=0
    for (i0,j0) in P0.interior:
        if not 1<j0<nr: continue
        slot=[k for k,v in enumerate(P0.interior) if v==(i0,j0)][0]
        theta,lam,it=vinfo(P0,slot)
        if abs(it[0])>1e-9: print("warn Hr not at 0",it); continue
        A,C=2**(i0-1),2**(nc-i0); a,c=A-1,C-1; L=2**nc+2*A+2*C-2
        A4=list(it)+[2*np.pi]
        for _ in range(trials):
            t=rng.integers(1,5); pl=[(int(rng.integers(0,4)),float(rng.uniform(0.03,0.97))) for _ in range(t)]
            st=[A4[s]+f*(A4[s+1]-A4[s]) for s,f in pl]
            if len(set(np.round(st,9)))<t: continue
            dead={(sd,e):int(gram_dies(theta,lam,st,sd,e)[0]) for sd in 'LR' for e in '-+'}
            pred=L-a*(dead[('L','-')]+dead[('L','+')])-c*(dead[('R','-')]+dead[('R','+')])
            got,nL=fast_component_count(raise_degree_at(P0,slot,pl))
            tot+=1
            if got!=pred or nL!=L:
                bad+=1
                if bad<5: print("MISMATCH",label,(i0,j0),pl,got,nL,pred,L)
    print(label,"agree",tot-bad,"/",tot)
rng=np.random.default_rng(7)
for nr,nc in ((3,3),(3,4),(4,4),(3,5)):
    run(miura(nr,nc),nr,nc,rng,25,f"miura {nr}x{nc}")
def samp(s,nr,nc):
    r=np.random.default_rng(10_000+s)
    for _ in range(60):
        t=r.uniform(0.25,1.6,nc)*r.choice([-1,1],nc); h=r.uniform(0.6,1.6,nr+1)
        try: return flat_foldable_quad(nr,nc,t,h,spacing=1.0)
        except ValueError: pass
for s in (1,2):
    run(samp(s,3,4),3,4,rng,25,f"ffquad{s} 3x4")
