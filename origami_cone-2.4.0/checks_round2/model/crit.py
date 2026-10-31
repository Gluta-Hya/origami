"""Appendix A, 'The death criterion': Theorem 6.7 (Gram determinant) against
direct definiteness of q_* on E_F, and the signature of Lemma 6.4."""
import sys
import numpy as np, sys
import os as _os; sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')); sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','model'))
def vertex(theta,lam,stub_angles):
    au=theta if lam==1 else np.pi-theta; ad=(2*np.pi-theta) if lam==1 else np.pi+theta
    cr=[("Hr",0.0),("zp",au),("Hl",np.pi),("zm",ad)]+[(f"s{k}",a%(2*np.pi)) for k,a in enumerate(stub_angles)]
    cr.sort(key=lambda c:c[1]); return cr
def area_form(cr):
    n=len(cr); a=np.array([c[1] for c in cr]); Q=np.zeros((n,n))
    for p in range(n):
        for q in range(p+1,n):
            Q[p,q]=Q[q,p]=0.5*np.sin(a[q]-a[p])/2   # q=1/2 sum x_p x_q sin ; bilinear sym -> /2 each
    return Q
def kv(cr):
    a=np.array([c[1] for c in cr]); J=np.vstack([np.cos(a),np.sin(a)])
    u,s,vt=np.linalg.svd(J); return vt[2:].T
def fam_functionals(cr,theta,lam,side,eta):
    idx={c[0]:i for i,c in enumerate(cr)}; n=len(cr)
    f1=np.zeros(n); f1[idx["zp" if eta=='-' else "zm"]]=1
    f2=np.zeros(n); zeta=idx["zm" if eta=='-' else "zp"]
    if side=='L': f2[idx["Hl"]]=1; f2[zeta]=-lam*np.cos(theta)
    else: f2[idx["Hr"]]=1; f2[zeta]=lam*np.cos(theta)
    return f1,f2
def direct_dies(theta,lam,stubs,side,eta,tol=1e-10):
    cr=vertex(theta,lam,stubs); N=kv(cr); f1,f2=fam_functionals(cr,theta,lam,side,eta)
    F=np.vstack([f1@N,f2@N]); u,s,vt=np.linalg.svd(F); E=N@vt[2:].T
    M=E.T@area_form(cr)@E; ev=np.linalg.eigvalsh(M)
    return bool(np.all(ev>tol) or np.all(ev<-tol)), ev
def D(cr,j):
    n=len(cr); a=np.array([c[1] for c in cr]); out=np.zeros(n)
    gp=(a[j]-a[j-1])%(2*np.pi); gn=(a[(j+1)%n]-a[j])%(2*np.pi)
    out[j]=-1/np.tan(gp)-1/np.tan(gn); out[j-1]+=1/np.sin(gp); out[(j+1)%n]+=1/np.sin(gn); return out
def gram_dies(theta,lam,stubs,side,eta):
    cr=vertex(theta,lam,stubs); f1,f2=fam_functionals(cr,theta,lam,side,eta)
    Y=[2*sum(c*D(cr,k) for k,c in enumerate(f) if c!=0) for f in (f1,f2)]
    G=np.array([[f@y for y in Y] for f in (f1,f2)]); d=np.linalg.det(G)
    t=len(stubs); return (d<0) if t>=2 else (abs(d)>1e-12), G
if __name__=="__main__":
    rng=np.random.default_rng(1); bad=0; tot=0
    for it in range(4000):
        theta=rng.uniform(0.15,1.45); lam=rng.choice([1,-1]); t=rng.integers(1,5)
        cr=vertex(theta,lam,[]); A=[c[1] for c in cr]+[2*np.pi]
        st=[]
        for _ in range(t):
            s=rng.integers(0,4); st.append(A[s]+rng.uniform(0.02,0.98)*(A[s+1]-A[s]))
        for side in 'LR':
            for eta in '-+':
                d1,_=direct_dies(theta,lam,st,side,eta); d2,_=gram_dies(theta,lam,st,side,eta)
                tot+=1; bad+=(d1!=d2)
    print("gram vs direct mismatches",bad,"/",tot)
    # Lorentzian check
    sig=set()
    for it in range(500):
        theta=rng.uniform(0.15,1.45); lam=rng.choice([1,-1]); t=rng.integers(1,5)
        cr=vertex(theta,lam,list(rng.uniform(0,2*np.pi,t))); N=kv(cr); ev=np.linalg.eigvalsh(N.T@area_form(cr)@N)
        sig.add(((ev>1e-9).sum(),(ev<-1e-9).sum()-len(ev)))
    print("signatures (pos, neg-n):",sig)
