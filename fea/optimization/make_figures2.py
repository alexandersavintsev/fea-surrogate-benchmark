# -*- coding: utf-8 -*-
"""Рисунки для разделов 1, 2 и схема Python-сравнения для 3.8."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
FIG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11})
BLUE="#d9e6f2"; ORANGE="#fde9d9"; GREEN="#e2efda"; EDGE="#2f5597"

def flow(steps, fname, title, colors=None, figw=12, wrap=2):
    n=len(steps); fig,ax=plt.subplots(figsize=(figw,2.6)); ax.axis("off")
    bw=1.0/n*0.86; gap=(1.0-bw*n)/(n-1) if n>1 else 0
    x=0.0
    for i,s in enumerate(steps):
        c = colors[i] if colors else BLUE
        ax.add_patch(FancyBboxPatch((x,0.32),bw,0.40,boxstyle="round,pad=0.012",
                     facecolor=c,edgecolor=EDGE,lw=1.4))
        ax.text(x+bw/2,0.52,s,ha="center",va="center",fontsize=9.2)
        if i<n-1:
            ax.add_patch(FancyArrowPatch((x+bw,0.52),(x+bw+gap,0.52),
                         arrowstyle="-|>",mutation_scale=14,lw=1.5,color="#444"))
        x+=bw+gap
    ax.set_xlim(-0.01,1.01); ax.set_ylim(0,1)
    if title: ax.set_title(title,fontsize=11)
    plt.tight_layout(); plt.savefig(f"{FIG}/{fname}",dpi=150,bbox_inches="tight"); plt.close()

# ---- Рис 1.1 суррогатная модель ----
flow(["Входные\nпараметры","APDL-\nмакрос","ANSYS-\nрасчёты","DOE-\nвыборка",
      "Обучение\nML-модели","Проверка\nкачества","Быстрые\nпредсказания"],
     "fig1_1_surrogate_scheme.png", "Построение суррогатной модели на основе ANSYS-расчётов",
     colors=[BLUE,BLUE,BLUE,BLUE,GREEN,GREEN,ORANGE], figw=13)

# ---- Рис 1.2 оптимизация ----
flow(["Диапазоны\nпараметров","ANSYS-\nрасчёты","Обучение\nML-модели",
      "Оптимизац.\nалгоритм","Найденные\nпараметры","Проверочный\nANSYS-расчёт"],
     "fig1_2_optimization_scheme.png", "Использование ML-модели в параметрической оптимизации",
     colors=[BLUE,BLUE,GREEN,GREEN,ORANGE,BLUE], figw=12)

# ---- Рис 2.1 линейная регрессия с весами ----
fig,ax=plt.subplots(figsize=(7,4.2)); ax.axis("off")
xs=["$x_1$","$x_2$","$x_3$","$x_n$"]; ws=["$w_1$","$w_2$","$w_3$","$w_n$"]
ys=np.linspace(0.85,0.15,4)
for i,(lab,w,y) in enumerate(zip(xs,ws,ys)):
    ax.add_patch(Circle((0.08,y),0.045,facecolor=BLUE,edgecolor=EDGE,lw=1.4))
    ax.text(0.08,y,lab,ha="center",va="center",fontsize=11)
    ax.annotate("",xy=(0.52,0.5),xytext=(0.125,y),arrowprops=dict(arrowstyle="-|>",color="#666"))
    ax.text(0.30,(y+0.5)/2+0.015,w,fontsize=10,color="#b35900")
ax.text(0.085,0.5+0.0,"",) 
ax.text(0.05,0.04,"$x_4 ... x_{n-1}$ (опущены)",fontsize=8,color="#888")
ax.add_patch(Circle((0.57,0.5),0.06,facecolor=GREEN,edgecolor=EDGE,lw=1.6))
ax.text(0.57,0.5,r"$\Sigma$",ha="center",va="center",fontsize=15)
ax.annotate("",xy=(0.85,0.5),xytext=(0.63,0.5),arrowprops=dict(arrowstyle="-|>",color="#444",lw=1.6))
ax.add_patch(FancyBboxPatch((0.85,0.44),0.12,0.12,boxstyle="round,pad=0.01",facecolor=ORANGE,edgecolor=EDGE))
ax.text(0.91,0.5,r"$\hat{y}$",ha="center",va="center",fontsize=13)
ax.text(0.57,0.32,r"$\hat{y}=w_0+\sum_j w_j x_j$",ha="center",fontsize=11)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_title("Линейная регрессионная модель с весами признаков")
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_1_linear_regression.png",dpi=150,bbox_inches="tight"); plt.close()

# ---- Рис 2.2 модель нейрона ----
fig,ax=plt.subplots(figsize=(7.5,4.0)); ax.axis("off")
ys=np.linspace(0.82,0.18,4); xs=["$x_1$","$x_2$","$x_3$","$x_m$"]; ws=["$w_1$","$w_2$","$w_3$","$w_m$"]
for lab,w,y in zip(xs,ws,ys):
    ax.add_patch(Circle((0.07,y),0.04,facecolor=BLUE,edgecolor=EDGE,lw=1.3)); ax.text(0.07,y,lab,ha="center",va="center")
    ax.annotate("",xy=(0.42,0.5),xytext=(0.11,y),arrowprops=dict(arrowstyle="-|>",color="#666"))
    ax.text(0.26,(y+0.5)/2+0.01,w,fontsize=9,color="#b35900")
ax.add_patch(Circle((0.48,0.5),0.07,facecolor=GREEN,edgecolor=EDGE,lw=1.6)); ax.text(0.48,0.5,r"$\Sigma$",ha="center",va="center",fontsize=14)
ax.text(0.48,0.66,"+ b (смещение)",ha="center",fontsize=8,color="#555")
ax.annotate("",xy=(0.66,0.5),xytext=(0.55,0.5),arrowprops=dict(arrowstyle="-|>",color="#444",lw=1.5))
ax.add_patch(FancyBboxPatch((0.66,0.42),0.12,0.16,boxstyle="round,pad=0.01",facecolor=ORANGE,edgecolor=EDGE))
ax.text(0.72,0.5,r"$\varphi(\cdot)$",ha="center",va="center",fontsize=13)
ax.annotate("",xy=(0.93,0.5),xytext=(0.78,0.5),arrowprops=dict(arrowstyle="-|>",color="#444",lw=1.5))
ax.text(0.95,0.5,"y",ha="center",va="center",fontsize=13)
ax.text(0.5,0.05,r"$y=\varphi\!\left(\sum_j w_j x_j + b\right)$",ha="center",fontsize=11)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_title("Модель искусственного нейрона")
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_2_neuron.png",dpi=150,bbox_inches="tight"); plt.close()

# ---- Рис 2.3 полносвязная сеть многовыходная (4-15-5-2) ----
fig,ax=plt.subplots(figsize=(8.5,4.6)); ax.axis("off")
layers=[("вход",4,["a","b","lx","ly"]),("скрытый\n15",15,None),("скрытый\n5",5,None),("выход",2,["smax","area"])]
xpos=[0.08,0.36,0.64,0.92]
def ylist(k): 
    if k<=6: return np.linspace(0.82,0.18,k)
    return np.linspace(0.92,0.08,k)
coords=[]
for (name,k,labs),x in zip(layers,xpos):
    yy=ylist(k); coords.append((x,yy))
    for j,y in enumerate(yy):
        r=0.022 if k>6 else 0.032
        ax.add_patch(Circle((x,y),r,facecolor=BLUE if name not in('выход',) else ORANGE,edgecolor=EDGE,lw=1.0))
        if labs: ax.text(x,y,labs[j],ha="center",va="center",fontsize=8)
    ax.text(x,0.985,name,ha="center",fontsize=9)
for li in range(len(coords)-1):
    x0,y0=coords[li]; x1,y1=coords[li+1]
    for a in y0:
        for b in y1:
            ax.plot([x0,x1],[a,b],color="#bbb",lw=0.3,zorder=0)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_title("Полносвязная нейронная сеть для многовыходной регрессии (4-15-5-2)")
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_3_fc_network.png",dpi=150,bbox_inches="tight"); plt.close()

# ---- Рис 2.4 функции активации ----
z=np.linspace(-4,4,400)
fig,ax=plt.subplots(figsize=(7,4.2))
ax.plot(z,np.tanh(z),lw=2.4,label="tanh (в NN_30)",color="#c0392b")
ax.plot(z,np.maximum(0,z),lw=1.8,label="ReLU",color="#2f5597",ls="--")
ax.plot(z,1/(1+np.exp(-z)),lw=1.8,label="sigmoid",color="#2e8b57",ls="-.")
ax.axhline(0,color="#999",lw=0.6); ax.axvline(0,color="#999",lw=0.6)
ax.set_xlabel("z"); ax.set_ylabel(r"$\varphi(z)$"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("Примеры функций активации")
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_4_activations.png",dpi=150,bbox_inches="tight"); plt.close()

# ---- Рис 2.5 градиентный бустинг ----
fig,ax=plt.subplots(figsize=(11,3.0)); ax.axis("off")
for i in range(4):
    x=0.04+i*0.20
    ax.add_patch(FancyBboxPatch((x,0.42),0.13,0.34,boxstyle="round,pad=0.01",facecolor=BLUE,edgecolor=EDGE,lw=1.3))
    ax.text(x+0.065,0.59,f"дерево {i+1}\n$h_{{{i+1}}}(x)$",ha="center",va="center",fontsize=9)
    ax.text(x+0.065,0.30,"остатки" if i>0 else "данные",ha="center",fontsize=8,color="#555")
    if i<3:
        ax.add_patch(FancyArrowPatch((x+0.13,0.59),(x+0.20,0.59),arrowstyle="-|>",mutation_scale=13,color="#444"))
ax.add_patch(FancyArrowPatch((0.84,0.59),(0.90,0.59),arrowstyle="-|>",mutation_scale=13,color="#444"))
ax.add_patch(FancyBboxPatch((0.90,0.40),0.085,0.40,boxstyle="round,pad=0.01",facecolor=ORANGE,edgecolor=EDGE))
ax.text(0.9425,0.60,r"$\hat{y}$",ha="center",va="center",fontsize=13)
ax.text(0.5,0.12,r"$\hat{y}=\sum_{m} \nu\, h_m(x)$,  каждое дерево обучается на ошибках предыдущих",ha="center",fontsize=10)
ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_title("Принцип последовательного построения градиентного бустинга")
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_5_gradient_boosting.png",dpi=150,bbox_inches="tight"); plt.close()

# ---- Рис 2.6 переобучение дерева: train/test ошибка vs глубина ----
d=np.arange(1,16)
train_err=0.9*np.exp(-0.55*d)+0.01
test_err=0.9*np.exp(-0.55*d)+0.02+0.012*(d-5)**2*(d>5)
fig,ax=plt.subplots(figsize=(7,4.2))
ax.plot(d,train_err,"o-",color="#2f5597",label="ошибка на обучении")
ax.plot(d,test_err,"s-",color="#c0392b",label="ошибка на тесте")
imin=int(d[np.argmin(test_err)])
ax.axvline(imin,color="#888",ls="--",lw=1)
ax.text(imin+0.2,max(test_err)*0.8,"оптимальная\nглубина",fontsize=9,color="#555")
ax.set_xlabel("глубина дерева (max_depth)"); ax.set_ylabel("ошибка")
ax.legend(); ax.grid(alpha=0.3)
ax.set_title("Зависимость ошибки от глубины дерева (переобучение)")
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_6_tree_overfitting.png",dpi=150,bbox_inches="tight"); plt.close()

# ---- Рис 2.7 градиентный спуск на квадратичной функции ----
w=np.linspace(-3,3,200); L=w**2
fig,ax=plt.subplots(figsize=(7,4.2))
ax.plot(w,L,color="#2f5597",lw=2)
wi=2.6
for _ in range(7):
    ax.plot(wi,wi**2,"o",color="#c0392b")
    wn=wi-0.35*2*wi
    ax.annotate("",xy=(wn,wn**2),xytext=(wi,wi**2),arrowprops=dict(arrowstyle="-|>",color="#c0392b",lw=1.2))
    wi=wn
ax.set_xlabel("параметр w"); ax.set_ylabel("ошибка L(w)")
ax.set_title("Градиентный спуск на простой квадратичной функции ошибки")
ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_7_gradient_descent.png",dpi=150,bbox_inches="tight"); plt.close()

# ---- Рис 2.8 случайный лес (опционально) ----
fig,ax=plt.subplots(figsize=(9,3.2)); ax.axis("off")
ax.add_patch(FancyBboxPatch((0.02,0.42),0.14,0.30,boxstyle="round,pad=0.01",facecolor=BLUE,edgecolor=EDGE))
ax.text(0.09,0.57,"обучающая\nвыборка",ha="center",va="center",fontsize=9)
for i in range(3):
    y=0.74-i*0.27
    ax.add_patch(FancyArrowPatch((0.16,0.57),(0.30,y+0.06),arrowstyle="-|>",mutation_scale=12,color="#666"))
    ax.add_patch(FancyBboxPatch((0.30,y),0.16,0.13,boxstyle="round,pad=0.01",facecolor=GREEN,edgecolor=EDGE))
    ax.text(0.38,y+0.065,f"дерево {i+1}\n(подвыборка)",ha="center",va="center",fontsize=8)
    ax.add_patch(FancyArrowPatch((0.46,y+0.065),(0.62,0.57),arrowstyle="-|>",mutation_scale=12,color="#666"))
ax.add_patch(FancyBboxPatch((0.62,0.45),0.16,0.24,boxstyle="round,pad=0.01",facecolor=ORANGE,edgecolor=EDGE))
ax.text(0.70,0.57,"усреднение",ha="center",va="center",fontsize=9)
ax.add_patch(FancyArrowPatch((0.78,0.57),(0.90,0.57),arrowstyle="-|>",mutation_scale=13,color="#444"))
ax.text(0.93,0.57,r"$\hat{y}$",ha="center",va="center",fontsize=13)
ax.set_xlim(0,1); ax.set_ylim(0.25,0.95)
ax.set_title("Случайный лес: усреднение деревьев на подвыборках")
plt.tight_layout(); plt.savefig(f"{FIG}/fig2_8_random_forest.png",dpi=150,bbox_inches="tight"); plt.close()

# ---- Рис 3.8 схема Python-сравнения (по recalculate_metrics.py) ----
flow(["doe.npz\n(samples, fields)","train/test\nsplit 80/20\nrandom_state=42","Standard\nScaler",
      "Ridge, GPR(Matern)\nGB, RandomForest\n(MultiOutput)","Метрики по выходам\nR2,MAE,MSE,RMSE,MAPE",
      "metrics_\nrecalculated.csv","Графики\nR2 / MAPE"],
     "fig3_8_python_pipeline.png", "Единая схема независимого сравнения моделей в Python",
     colors=[BLUE,BLUE,BLUE,GREEN,GREEN,ORANGE,ORANGE], figw=14)

print("done:", sorted(os.listdir(FIG)))
