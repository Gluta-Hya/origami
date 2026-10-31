"""Appendix A, 'Maximal flats': type census together with the dimensions n_r+t-1,
n^+ +t+1, n^- +t+1, t+2 of the four flats of case (a)."""
import os as _os, sys as _sys; _sys.path.insert(0,_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),'..','..')); _sys.path.insert(0,_os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
from census import samp, census
PL=[[(0,.5)],[(1,.5),(2,.5)],[(0,.5),(1,.5)],[(0,.5),(2,.5)],[(0,1/3),(0,2/3)],[(0,.5),(1,.5),(2,.5),(3,.5)]]
tot=okc=okd=0
for nr,nc in ((3,3),(3,4),(4,4),(5,3),(4,5),(5,4)):
    for s in (0,1,2):
        P0=samp(s,nr,nc)
        for (i0,j0) in P0.interior:
            if not 1<j0<nr: continue
            A,C=2**(i0-1),2**(nc-i0); a,c=A-1,C-1; Bn=2**nc; nm,npl=j0-1,nr-j0
            for pl in PL:
                t=len(pl); cn,dA,n=census(P0,nr,nc,i0,j0,pl); tot+=1
                want={"M00":1,"M-0":1,"M0+":1,"M-+":1,"b1":Bn-2,"b2":2*a+2*c}
                okc+= all(cn.get(k,0)==v for k,v in want.items()) and cn.get("OTHER",0)==0
                wd={"M00":nr+t-1,"M-0":npl+t+1,"M0+":nm+t+1,"M-+":t+2}
                okd+= dA==wd
print(tot,okc,okd)
