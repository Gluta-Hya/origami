"""Figure 3 of the paper (fig_counts.pdf): the bisector closed form across the
columns of a 3 x 6 patch, and the lattice of death coefficients."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from origami_cone import predicted_at
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":9.5,"mathtext.fontset":"dejavuserif"})
BLUE,ORANGE,PURPLE,GREEN,GREY="#2a78d6","#eb6834","#8d6fc4","#3f9e72","#5b6470"
fig,ax=plt.subplots(1,2,figsize=(10.4,4.1),gridspec_kw={"width_ratios":[1.35,1]})
nc=6; cols=np.arange(1,nc+1)
cls=[("degree 8, one stub per sector",[(0,.5),(1,.5),(2,.5),(3,.5)],ORANGE,"o-"),
     ("right pair $\\{0,3\\}$",[(0,.5),(3,.5)],PURPLE,"^-"),
     ("upper pair $\\{0,1\\}$",[(0,.5),(1,.5)],BLUE,"s-"),
     ("left pair $\\{1,2\\}$",[(1,.5),(2,.5)],GREEN,"v-"),
     ("single stub",[(0,.5)],GREY,"D-")]
for lab,pl,c,st in cls:
    y=[predicted_at(3,nc,int(i),2,pl) for i in cols]
    ax[0].plot(cols,y,st,color=c,lw=1.7,ms=5.5,mec="white",mew=0.9,label=lab)
ax[0].axhline(2**nc,color=GREY,lw=0.9,ls=":")
ax[0].text(6.15,2**nc*0.992,"$2^{n_c}=64$: pure Miura, or any outer-row implant",
           ha="right",va="top",fontsize=8.0,color=GREY)
ax[0].set_yscale("log",base=2); ax[0].set_xticks(cols); ax[0].set_ylim(2**5.93,2**7.35)
ax[0].set_xlabel("column $i_0$ of the implanted vertex"); ax[0].set_ylabel("surviving flats $N$")
ax[0].set_title("(a)  $n_c=6$, bisecting stubs: counts by placement",loc="left",fontsize=10)
ax[0].legend(frameon=False,fontsize=8,loc="upper center",ncol=2)
for sp in ("top","right"): ax[0].spines[sp].set_visible(False)
ax[0].grid(axis="y",lw=0.4,color="#e4e7ec")

# (b) death lattice
lab={(0,0):"degree 8",(1,0):"diagonal,\nnarrow side R",(0,1):"diagonal,\nnarrow side L",
     (2,0):"right pair",(0,2):"left pair",(1,1):"upper or\nlower pair",
     (2,1):"two stubs in\none right sector",(1,2):"two stubs in\none left sector",(2,2):"single stub"}
a=ax[1]
for (x,y),t in lab.items():
    a.add_patch(plt.Rectangle((x-.46,y-.46),.92,.92,fc="#f3f6fb",ec=BLUE,lw=1.0))
    a.text(x,y,t,ha="center",va="center",fontsize=7.6)
a.set_xlim(-.6,2.6); a.set_ylim(-.6,2.6); a.set_aspect("equal")
a.set_xticks([0,1,2]); a.set_yticks([0,1,2])
a.set_xlabel("$\\alpha$ = dead left families"); a.set_ylabel("$\\beta$ = dead right families")
a.set_title("(b)  deaths $=\\alpha a+\\beta c$",loc="left",fontsize=10)
for sp in ("top","right"): a.spines[sp].set_visible(False)
plt.tight_layout()
out="/tmp/claude-0/-home-claude/09266b35-5419-5c2a-abf3-dbd233144881/scratchpad/paper/fig_counts.pdf"
plt.savefig(out,bbox_inches="tight"); plt.savefig(out.replace(".pdf",".png"),dpi=170,bbox_inches="tight")
print("ok")
