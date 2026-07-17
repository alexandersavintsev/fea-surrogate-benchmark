# -*- coding: utf-8 -*-
"""Построение рисунков ВКR по реальным данным (разделы 3,4,5)."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle, FancyBboxPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.abspath(os.path.join(ROOT, ".."))
FIG  = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": .3})

MODELS = ["Ridge", "GPR_sklearn", "GradientBoosting", "RandomForest"]
LBL = {"Ridge":"Ridge","GPR_sklearn":"GPR","GradientBoosting":"Grad.Boosting","RandomForest":"Rand.Forest"}

# ---------- Раздел 3: сравнение по R2 и MAPE (GPR_100, test, mean) ----------
m = pd.read_csv(os.path.join(SRC, "metrics_recalculated.csv"))
m.columns=[c.strip().lstrip("﻿") for c in m.columns]
sub = m[(m.dataset=="GPR_100")&(m.subset=="test")&(m.target=="mean_over_outputs")]
r2  = [float(sub[sub.model==k].R2.values[0]) for k in MODELS]
mape= [float(sub[sub.model==k].MAPE_percent.values[0]) for k in MODELS]

plt.figure(figsize=(7,4))
bars=plt.bar([LBL[k] for k in MODELS], r2, color=["#4C72B0","#55A868","#C44E52","#8172B2"])
for b,v in zip(bars,r2): plt.text(b.get_x()+b.get_width()/2, v+0.0005, f"{v:.4f}", ha="center", fontsize=9)
plt.ylim(0.96,1.001); plt.ylabel("Средний $R^2$ на тесте")
plt.title("Задача Кирша (100 точек): сравнение моделей по $R^2$")
plt.tight_layout(); plt.savefig(f"{FIG}/fig3_r2_comparison.png", dpi=150); plt.close()

plt.figure(figsize=(7,4))
bars=plt.bar([LBL[k] for k in MODELS], mape, color=["#4C72B0","#55A868","#C44E52","#8172B2"])
for b,v in zip(bars,mape): plt.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.3f}", ha="center", fontsize=9)
plt.ylabel("Средний MAPE, %"); plt.title("Задача Кирша (100 точек): сравнение моделей по MAPE")
plt.tight_layout(); plt.savefig(f"{FIG}/fig3_mape_comparison.png", dpi=150); plt.close()

# ---------- Раздел 3: прогноз против расчёта (GPR_sklearn, test) ----------
pred = pd.read_csv(os.path.join(SRC, "GPR_100_GPR_sklearn_test_predictions.csv"))
pred.columns=[c.strip().lstrip("﻿") for c in pred.columns]
for tgt,unit in [("smax","усл. ед."),("area","усл. ед.")]:
    yt,yp = pred[f"{tgt}_true"].values, pred[f"{tgt}_pred"].values
    plt.figure(figsize=(5,5))
    plt.scatter(yt,yp,s=30,color="#55A868",edgecolor="k",linewidth=.4)
    lo,hi=min(yt.min(),yp.min()),max(yt.max(),yp.max()); pad=(hi-lo)*0.05
    plt.plot([lo-pad,hi+pad],[lo-pad,hi+pad],"k--",lw=1)
    plt.xlabel(f"ANSYS, {tgt}"); plt.ylabel(f"Прогноз GPR, {tgt}")
    plt.title(f"GPR (100 точек): {tgt}, тест")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig3_pred_vs_true_{tgt}.png", dpi=150); plt.close()

# ---------- Раздел 3: SHAP importance ----------
sh = json.load(open(os.path.join(SRC,"shap_values_GPR100.json")))
names, vals = sh["feature_names"], sh["mean_abs_shap"]
order=np.argsort(vals)
plt.figure(figsize=(6,3.6))
plt.barh([names[i] for i in order],[vals[i] for i in order],color="#4C72B0")
for i,idx in enumerate(order): plt.text(vals[idx], i, f" {vals[idx]:.1f}", va="center", fontsize=9)
plt.xlabel("mean(|SHAP|)"); plt.title("Задача Кирша: важность параметров (GPR, 100 точек)")
plt.tight_layout(); plt.savefig(f"{FIG}/fig3_shap_importance.png", dpi=150); plt.close()

# ---------- Раздел 4: схема геометрии и ГУ консоли ----------
def cantilever(ax, bc=False):
    L,H=6,1.2
    ax.add_patch(Rectangle((0,0),L,H,fill=True,facecolor="#d9e6f2",edgecolor="k",lw=1.5))
    # защемление слева (штриховка)
    for y in np.linspace(0,H,9):
        ax.plot([-0.35,0],[y-0.05,y+0.02],color="k",lw=1)
    ax.plot([0,0],[0,H],color="k",lw=3)
    # распределённое давление сверху
    for x in np.linspace(0.3,L-0.1,12):
        ax.add_patch(FancyArrow(x,H+0.6,0,-0.5,width=0.005,head_width=0.12,head_length=0.12,color="#c0392b"))
    ax.plot([0.2,L],[H+0.62,H+0.62],color="#c0392b",lw=1)
    ax.text(L/2,H+0.85,"p (давление)",ha="center",color="#c0392b")
    ax.annotate("",xy=(L,-0.35),xytext=(0,-0.35),arrowprops=dict(arrowstyle="<->"))
    ax.text(L/2,-0.6,"lx",ha="center")
    ax.annotate("",xy=(L+0.45,H),xytext=(L+0.45,0),arrowprops=dict(arrowstyle="<->"))
    ax.text(L+0.65,H/2,"ly",va="center")
    if bc:
        ax.text(-0.95,H/2,"ux=0\nuy=0",va="center",ha="center",fontsize=9)
        ax.text(L-1.4,-0.95,"PLANE182, плоское\nнапряжённое состояние",fontsize=8,color="#555")
    ax.set_xlim(-1.4,L+1.2); ax.set_ylim(-1.2,H+1.2); ax.axis("off"); ax.set_aspect("equal")

fig,ax=plt.subplots(figsize=(7,3.4)); cantilever(ax,bc=False)
ax.set_title("Консольная пластина: расчётная схема")
plt.tight_layout(); plt.savefig(f"{FIG}/cantilever_geometry_scheme.png", dpi=150); plt.close()
fig,ax=plt.subplots(figsize=(7,3.4)); cantilever(ax,bc=True)
ax.set_title("Консольная пластина: граничные условия и нагрузка")
plt.tight_layout(); plt.savefig(f"{FIG}/cantilever_boundary_conditions.png", dpi=150); plt.close()

# ---------- Раздел 5: время ANSYS против суррогата ----------
per=1289.55/100; pred_t=4.86e-3
plt.figure(figsize=(6,4))
b=plt.bar(["1 расчёт ANSYS\n(среднее)","1 прогноз GPR"],[per,pred_t],color=["#C44E52","#55A868"])
plt.yscale("log"); plt.ylabel("время, с (лог. шкала)")
for r,v in zip(b,[per,pred_t]): plt.text(r.get_x()+r.get_width()/2, v, f" {v:.4g} с", ha="center", va="bottom", fontsize=10)
plt.title(f"Одна оценка отклика: ускорение ~{per/pred_t:,.0f}x".replace(",", " "))
plt.tight_layout(); plt.savefig(f"{FIG}/ansys_vs_surrogate_time.png", dpi=150); plt.close()

# ---------- Раздел 5: схема рабочего процесса оптимизации ----------
fig,ax=plt.subplots(figsize=(9,2.6)); ax.axis("off")
steps=["Диапазоны\nпараметров","ANSYS-расчёты\n(DOE, 100 точек)","Обучение\nGPR-суррогата",
       "Оптимизация\n(DE / поиск)","Найденные\nпараметры","Проверочный\nANSYS-расчёт"]
x=0.02
for i,s in enumerate(steps):
    ax.add_patch(FancyBboxPatch((x,0.3),0.13,0.4,boxstyle="round,pad=0.01",
                 facecolor="#d9e6f2" if i not in(4,5) else "#fde9d9",edgecolor="k"))
    ax.text(x+0.065,0.5,s,ha="center",va="center",fontsize=8.5)
    if i<len(steps)-1: ax.annotate("",xy=(x+0.17,0.5),xytext=(x+0.15,0.5),arrowprops=dict(arrowstyle="->",lw=1.4))
    x+=0.163
plt.title("Параметрическая оптимизация на ML-суррогате",fontsize=11)
plt.tight_layout(); plt.savefig(f"{FIG}/optimization_workflow.png", dpi=150); plt.close()

print("figures:", sorted(os.listdir(FIG)))
