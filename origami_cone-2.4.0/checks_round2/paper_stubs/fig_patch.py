"""Figure 1: the hinged-stub model and the stubs continued to the edge of the
sheet."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from extend import build_physical, sheet_boundary
from origami_cone import miura, raise_degree_at
nr,nc,v=4,5,(3,2); pl=[(0,.5),(2,.5)]
fig,axs=plt.subplots(1,2,figsize=(11,4.6))
for ax,mode in zip(axs,("model","paper")):
    P0=miura(nr,nc)
    if mode=="model": P=raise_degree_at(P0,P0.interior.index(v),pl)
    else: P,info=build_physical(nr,nc,v,pl)
    bnd=sheet_boundary(P0,nr,nc)
    poly=[P0.pos[a] for a,b in bnd]
    ax.fill(*zip(*poly),color="#f4efe4",zorder=0)
    for a,b in bnd: ax.plot(*zip(P0.pos[a],P0.pos[b]),color="#555",lw=1.2,zorder=1)
    for key,e in P.cidx.items():
        a,b=tuple(key); pa,pb=P.pos[a],P.pos[b]
        def rid(x):
            if x==v: return "*"
            if isinstance(x,tuple) and isinstance(x[0],str): return x[1] if x[0] in ("x","e") else "stub"
            return None
        ra,rb=rid(a),rid(b)
        isray=(ra is not None and rb is not None and (ra==rb or "*" in (ra,rb))) or "stub" in (ra,rb)
        if isray: ax.plot(*zip(pa,pb),color="#d9730d",lw=2.0,ls="--" if mode=="model" else "-",zorder=3)
        else:
            horiz=abs(pa[1]-pb[1])<1e-9
            ax.plot(*zip(pa,pb),color="#1f5aa6" if not horiz else "#222",lw=1.4,zorder=2)
    for w in P.interior:
        p=P.pos[w]
        if isinstance(w,tuple) and isinstance(w[0],str): ax.plot(*p,"o",ms=6,mfc="white",mec="#d9730d",mew=1.6,zorder=5)
        else: ax.plot(*p,"o",ms=3.5,color="#222",zorder=4)
    if mode=="model":
        for key in P.cidx:
            for x in key:
                if isinstance(x,tuple) and x[0]=="stub": ax.plot(*P.pos[x],"s",ms=5,color="#d9730d",zorder=5)
    ax.plot(*P.pos[v],"o",ms=9,mfc="#d9730d",mec="k",zorder=6)
    ax.annotate(r"$v^*$",P.pos[v],xytext=(6,-14),textcoords="offset points",fontsize=12)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) hinged-stub model" if mode=="model" else "(b) stubs continued to the edge of the sheet",fontsize=11)
plt.tight_layout(); plt.savefig("fig_patch.pdf"); plt.savefig("fig_patch.png",dpi=130)
