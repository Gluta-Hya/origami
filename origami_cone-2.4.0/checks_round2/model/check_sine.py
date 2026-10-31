"""Appendix A, 'The death criterion': the sine forms of Corollary 7.3(a),(b)
against the Gram criterion of Theorem 6.7."""
import os as _os, sys as _sys; _sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')); _sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
from crit import gram_dies
rng=np.random.default_rng(11); bad=n=0
S=np.sin
for _ in range(4000):
    th=rng.uniform(.08,1.5); lam=rng.choice([1,-1])
    # sector boundaries for lam
    au=th if lam==1 else np.pi-th; ad=2*np.pi-th if lam==1 else np.pi+th
    secs=[(0,au),(au,np.pi),(np.pi,ad),(ad,2*np.pi)]   # (start,end)
    # horizontal end of each sector: sec0 start(Hr),1 end(Hl),2 start(Hl),3 end(Hr)
    def place(k,u):
        a,b=secs[k]; ang=a+u*(b-a); w=b-a
        beta = ang-a if k in (0,2) else b-ang; gam=w-beta
        return ang,beta,gam,w
    narrow=[0,3] if lam==1 else [1,2]
    side={0:'R',3:'R',1:'L',2:'L'}
    assoc={('R','-'):0,('L','-'):1,('L','+'):2,('R','+'):3}
    for pl in ([0,3],[1,2],[0,2],[1,3]):
        P=[place(k,rng.uniform(.002,.998)) for k in pl]; st=[p[0] for p in P]
        for fam,a in assoc.items():
            if a in pl: continue
            dead=gram_dies(th,lam,st,*fam)[0]
            if side[pl[0]]==side[pl[1]]:   # pair on one side
                surv=np.cos(th)**2*S(P[0][1])*S(P[1][1])>=S(P[0][2])*S(P[1][2])
            else:
                k=0 if pl[0] in narrow else 1; l=1-k
                bk,gk,bl,gl=P[k][1],P[k][2],P[l][1],P[l][2]
                if side[a]==side[pl[k]]:   # family on narrow side
                    surv=S(th)**2*np.cos(bk)**2*S(bl)>=S(bk)*S(gk)*S(gl)
                else:
                    surv=S(th)**2*np.cos(bl)**2*S(bk)>=S(gk)*S(bl)*S(gl)
            n+=1; bad+= (not dead)!=surv
print("sine-form mismatches",bad,"/",n)
