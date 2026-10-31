"""Appendix A, 'The death criterion': the explicit Q1, Q2 of Corollary 7.3
against the Gram criterion, and the bisector values of Corollary 7.3(c)."""
import os as _os, sys as _sys; _sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')); _sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
from crit import vertex, kv, area_form, fam_functionals, gram_dies, direct_dies
cot=lambda x:1/np.tan(x)
rng=np.random.default_rng(3)
# 1) explicit 2Q formulas for R- and L- (lambda=+1) vs direct
def twoQ(theta,stubs,fam):
    # stubs: list of (phi,s); sector by angle
    ang=lambda p: p if p<2*np.pi-theta else p-2*np.pi   # order from -theta
    st=sorted(stubs,key=lambda x:ang(x[0]))
    Pi=sum(st[p][1]*st[q][1]*np.sin(st[q][0]-st[p][0]) for p in range(len(st)) for q in range(p+1,len(st)))
    sec=lambda p: 0 if p<theta else 1 if p<np.pi else 2 if p<2*np.pi-theta else 3
    Ys=[sum(s*np.sin(p) for p,s in stubs if sec(p)==k) for k in range(4)]
    Y=sum(Ys); C=sum(s*np.cos(p) for p,s in stubs)
    A=Ys[3]-Ys[0]-Ys[1]-Ys[2]; Bv=Ys[0]+Ys[1]+Ys[3]-Ys[2]
    return Pi+cot(theta)*Y*A+C*Bv if fam=='R-' else Pi+C*A+cot(theta)*Y*Bv
def direct_twoQ(theta,stubs,fam):
    # build crease values from stubs and compute 2q
    C=sum(s*np.cos(p) for p,s in stubs); Y=sum(s*np.sin(p) for p,s in stubs); z=Y/np.sin(theta)
    vals={"Hr":(-np.cos(theta)*z if fam=='R-' else -C),"Hl":(C if fam=='R-' else np.cos(theta)*z),"zp":0,"zm":z}
    cr=[("Hr",0),("zp",theta),("Hl",np.pi),("zm",2*np.pi-theta)]+[(f"s{k}",p) for k,(p,s) in enumerate(stubs)]
    x={**vals,**{f"s{k}":s for k,(p,s) in enumerate(stubs)}}
    cr.sort(key=lambda c:c[1]); tot=0
    for i in range(len(cr)):
        for j in range(i+1,len(cr)): tot+=x[cr[i][0]]*x[cr[j][0]]*np.sin(cr[j][1]-cr[i][1])
    # closure check
    cx=sum(x[n]*np.cos(a) for n,a in cr); cy=sum(x[n]*np.sin(a) for n,a in cr)
    assert abs(cx)<1e-9 and abs(cy)<1e-9, (cx,cy)
    return tot
mx=0
for _ in range(2000):
    th=rng.uniform(.1,1.5); t=rng.integers(1,5); A4=[0,th,np.pi,2*np.pi-th,2*np.pi]
    st=[]
    for _ in range(t):
        k=rng.integers(0,4); st.append((A4[k]+rng.uniform(.01,.99)*(A4[k+1]-A4[k]),rng.normal()))
    for fam in ('R-','L-'): mx=max(mx,abs(twoQ(th,st,fam)-direct_twoQ(th,st,fam)))
print("2Q formula max err",mx)
# 2) explicit pair formulas vs gram, lambda=+1
f=lambda g,th: np.sin(th+g)/np.sin(g); ft=lambda h,th: np.sin(th-h)/np.sin(h)
bad=0;n=0
for _ in range(3000):
    th=rng.uniform(.1,1.5)
    # wide pair {1,2}
    g1=rng.uniform(.001,1)*(np.pi-th); g2=rng.uniform(.001,1)*(np.pi-th)
    st=[th+g1, 2*np.pi-th-g2]
    pred_surv=np.cos(th)**2*f(g1,th)*f(g2,th)>=1
    for fam in [('R','-'),('R','+')]:
        n+=1; bad+= (not gram_dies(th,1,st,*fam)[0])!=pred_surv
    # narrow pair {0,3}
    h0=rng.uniform(.001,1)*th; h3=rng.uniform(.001,1)*th
    st=[th-h0, 2*np.pi-th+h3]
    pred_surv=np.cos(th)**2*ft(h0,th)*ft(h3,th)>=1
    for fam in [('L','-'),('L','+')]:
        n+=1; bad+= (not gram_dies(th,1,st,*fam)[0])!=pred_surv
    # diagonal {0,2}
    p=rng.uniform(.001,.999)*th; r=rng.uniform(.001,.999)*(np.pi-th); pp=th-p; rr=np.pi-th-r
    st=[p, np.pi+r]
    c2=np.cos(th)**2
    Rp=(cot(rr)+cot(th))*(cot(p)+cot(th)+c2*(cot(pp)-cot(th)))>=1/np.sin(th)**2
    Lm=(cot(pp)-cot(th))*(cot(r)-cot(th)+c2*(cot(rr)+cot(th)))>=1/np.sin(th)**2
    n+=2; bad+=(not gram_dies(th,1,st,'R','+')[0])!=Rp; bad+=(not gram_dies(th,1,st,'L','-')[0])!=Lm
print("pair formulas mismatches",bad,"/",n)
# 3) bisector values
for th in np.linspace(.1,1.5,8):
    out=[]
    for st,fams in (([th+(np.pi-th)/2, np.pi+(np.pi-th)/2],[('R','-'),('R','+')]),([th/2,2*np.pi-th/2],[('L','-'),('L','+')]),([th/2,np.pi+(np.pi-th)/2],[('R','+'),('L','-')])):
        out.append([gram_dies(th,1,st,*fm)[0] for fm in fams])
    print(round(th,2),out)
