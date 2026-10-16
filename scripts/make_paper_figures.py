"""Regenerate the data figures of the paper from results/ (matplotlib only)."""
import pandas as pd, numpy as np, json, matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=str(ROOT/"results")+"/"
FIG=ROOT/"paper"/"figures"
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"mathtext.fontset":"dejavuserif",
  "font.size":7.5,"axes.linewidth":0.5,"xtick.major.width":0.5,"ytick.major.width":0.5,
  "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
INK2="#52514e"; GRID="#e4e3df"
BLUE,ORANGE,AQUA="#2a78d6","#eb6834","#1baf7a"

# ---------------- Fig: loss cascade -----------------
s=json.load(open(R+"pipeline/pipeline_summary.json"))["loss_cascade"]
parts=[("Lost at Stage 1 (gate)",s["lost_stage1"],"#b8461b"),("Lost at Stage 2 (router)",s["lost_stage2"],"#eb6834"),
       ("Lost at Stage 3",s["lost_stage3"],"#f2a27f"),("Detected",s["found"],BLUE)]
T=s["total"]
fig,ax=plt.subplots(figsize=(3.45,1.15))
left=0
for name,v,c in parts:
    ax.barh(0,v,left=left,color=c,height=0.6,edgecolor="white",linewidth=1.2,label=name)
    if v/T>0.1:
        ax.text(left+v/2,0,f"{v}\n{v/T:.1%}",ha="center",va="center",fontsize=6.4,linespacing=1.05,color="white")
    else:
        ax.annotate(f"{v} ({v/T:.1%})",(left+v/2,-0.3),xytext=(left+v/2,-0.62),ha="center",va="top",
                    fontsize=6.4,color="#0b0b0b",arrowprops=dict(arrowstyle="-",lw=0.5,color=INK2))
    left+=v
ax.set_xlim(0,T); ax.set_ylim(-0.95,0.4); ax.set_yticks([]); ax.set_xticks([])
for sp in ("left","bottom"): ax.spines[sp].set_visible(False)
ax.legend(ncol=2,frameon=False,fontsize=6.4,loc="lower center",bbox_to_anchor=(0.5,0.98),
          handlelength=1.0,columnspacing=1.0,handletextpad=0.4)
fig.tight_layout(pad=0.15); fig.savefig(FIG/"loss_cascade.pdf")

# ---------------- Fig: per-disease F1 -----------------
pipe=pd.read_csv(R+"pipeline/pipeline_per_disease.csv").set_index("disease")
flat=pd.read_csv(R+"flat_baseline/flat_per_disease.csv").set_index("disease")
ens=pd.read_csv(R+"final_ensemble/test_per_disease_ensemble_flat_lc.csv").set_index("disease")
top=ens.sort_values("n_true",ascending=False).index[:16]
x=np.arange(len(top)); w=0.27
fig,ax=plt.subplots(figsize=(7.1,1.8))
for k,(lab,df,c) in enumerate([("Three-stage pipeline",pipe,ORANGE),("Flat",flat,AQUA),("Flat + label-corr. ensemble + TTA",ens,BLUE)]):
    ax.bar(x+(k-1)*w,df.loc[top,"f1"],w*0.92,color=c,label=lab,linewidth=0)
ax.set_xticks(x); ax.set_xticklabels([f"{d}\n{int(ens.loc[d,'n_true'])}" for d in top],fontsize=6.5)
ax.set_ylabel("Test F1",color=INK2); ax.set_ylim(0,1)
ax.yaxis.grid(True,color=GRID,linewidth=0.5); ax.set_axisbelow(True)
ax.tick_params(colors=INK2,length=2)
ax.legend(ncol=3,frameon=False,loc="upper right",fontsize=6.8,bbox_to_anchor=(1,1.12))
fig.tight_layout(pad=0.2); fig.savefig(FIG/"per_disease_f1.pdf")
print("ok")

# ---------------- Fig: ablation dot plot -----------------
abl=pd.read_csv(R+"ablation/ablation_table.csv")
names=["Flat","Flat + TTA","Flat + TTA + re-tuned thr.","Three-stage cascade","Cascade + flat",
       "Label-corr.","Label-corr. + TTA","Three-way ensemble","Flat + label-corr. + TTA"]
y=np.arange(len(names))[::-1]
fig,ax=plt.subplots(figsize=(3.45,2.2))
for col,lab,c,mk in [("micro_f1","micro-F1",BLUE,"o"),("macro_f1","macro-F1",ORANGE,"s"),("detection","Detection",AQUA,"D")]:
    ax.scatter(abl[col],y,s=16,color=c,marker=mk,label=lab,zorder=3,edgecolor="white",linewidth=0.6)
for yy in y: ax.axhline(yy,color=GRID,lw=0.5,zorder=1)
ax.axhspan(-0.5,0.5,color="#ddeaf9",zorder=0)
ax.set_yticks(y); ax.set_yticklabels(names,fontsize=6.6)
ax.set_xlim(0.37,0.71); ax.set_xlabel("Test score",color=INK2)
ax.tick_params(colors=INK2,length=2); ax.spines["left"].set_visible(False)
ax.legend(ncol=3,frameon=False,fontsize=6.6,loc="lower center",bbox_to_anchor=(0.35,1.0),handletextpad=0.2,columnspacing=0.8)
fig.tight_layout(pad=0.2); fig.savefig(FIG/"ablation.pdf")
print("abl ok")
