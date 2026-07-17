# -*- coding: utf-8 -*-
"""
Построение рисунков для раздела 4 (matplotlib, без seaborn).
Всегда строит: расчётную схему, схему сетки, заглушки полей UY/SEQV, DOE-pairplot.
При наличии метрик/предсказаний строит сравнение моделей (R2, MAPE, pred-vs-true)
и важность признаков.
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data"); FIG = os.path.join(ROOT, "figures")
MODELS = os.path.join(ROOT, "models"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11})


def fig_scheme():
    fig, ax = plt.subplots(figsize=(8, 3.2)); ax.axis("off")
    L, H = 8.0, 1.3
    ax.add_patch(Rectangle((0, 0), L, H, facecolor="#d9e6f2", edgecolor="k", lw=1.6))
    for y in np.linspace(0, H, 9):                     # заделка слева
        ax.plot([-0.45, 0], [y-0.07, y+0.03], color="k", lw=1)
    ax.plot([0, 0], [0, H], color="k", lw=3)
    for y in np.linspace(0.1, H-0.1, 6):               # сила на правом торце
        ax.add_patch(FancyArrow(L, y, 0, -0.85, width=0.006, head_width=0.14,
                                head_length=0.16, color="#c0392b", length_includes_head=True))
    ax.text(L+0.15, H/2-0.55, "force", color="#c0392b", fontsize=11, va="center")
    ax.annotate("", xy=(L, -0.45), xytext=(0, -0.45), arrowprops=dict(arrowstyle="<->"))
    ax.text(L/2, -0.72, "len", ha="center")
    ax.annotate("", xy=(-0.85, H), xytext=(-0.85, 0), arrowprops=dict(arrowstyle="<->"))
    ax.text(-1.05, H/2, "height", va="center", ha="center", rotation=90)
    ax.text(L/2, H/2, "PLANE182\nплоское напряжённое\nсостояние, толщина thk",
            ha="center", va="center", fontsize=9, color="#333")
    # координатный индикатор в свободном нижнем-левом углу
    ox, oy = -1.25, -0.85
    ax.annotate("", xy=(ox+0.7, oy), xytext=(ox, oy), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("", xy=(ox, oy+0.6), xytext=(ox, oy), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.text(ox+0.78, oy, "x", va="center", fontsize=10); ax.text(ox, oy+0.68, "y", ha="center", fontsize=10)
    ax.set_xlim(-1.7, L+1.0); ax.set_ylim(-1.1, H+0.4); ax.set_aspect("equal")
    ax.set_title("Расчётная схема консольно защемлённого элемента")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_4_1_scheme.png", dpi=150, bbox_inches="tight"); plt.close()


def fig_mesh():
    fig, ax = plt.subplots(figsize=(8, 1.9)); ax.axis("off")
    L, H, nx, ny = 8.0, 1.0, 24, 4
    for i in range(nx+1): ax.plot([i*L/nx]*2, [0, H], color="#4C72B0", lw=0.6)
    for j in range(ny+1): ax.plot([0, L], [j*H/ny]*2, color="#4C72B0", lw=0.6)
    ax.add_patch(Rectangle((0, 0), L, H, fill=False, edgecolor="k", lw=1.4))
    ax.set_xlim(-0.3, L+0.3); ax.set_ylim(-0.3, H+0.3); ax.set_aspect("equal")
    ax.set_title("Структура конечно-элементной сетки (иллюстрация; реальную сетку экспортировать из ANSYS)")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_4_2_mesh_placeholder.png", dpi=150, bbox_inches="tight"); plt.close()


def fig_field_placeholder(fname, title):
    fig, ax = plt.subplots(figsize=(8, 2.0)); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 8, 1, facecolor="#f2f2f2", edgecolor="k", lw=1.2))
    ax.text(4, 0.5, "Место для экспорта поля из ANSYS\n(" + title + ")",
            ha="center", va="center", fontsize=11, color="#777")
    ax.set_xlim(-0.2, 8.2); ax.set_ylim(-0.2, 1.2); ax.set_aspect("equal")
    plt.tight_layout(); plt.savefig(f"{FIG}/{fname}", dpi=150, bbox_inches="tight"); plt.close()


def fig_pairplot():
    p = os.path.join(DATA, "cantilever_doe_inputs.csv")
    if not os.path.exists(p): return
    import pandas as pd
    d = pd.read_csv(p); cols = ["len", "height", "thk", "force"]
    n = len(cols); fig, ax = plt.subplots(n, n, figsize=(8, 8))
    for i in range(n):
        for j in range(n):
            a = ax[i, j]
            if i == j: a.hist(d[cols[j]], bins=12, color="#55A868")
            else: a.scatter(d[cols[j]], d[cols[i]], s=8, color="#4C72B0")
            if i == n-1: a.set_xlabel(cols[j], fontsize=9)
            if j == 0: a.set_ylabel(cols[i], fontsize=9)
            a.tick_params(labelsize=7)
    fig.suptitle("Распределение DOE-точек по парам параметров (LHS)")
    plt.tight_layout(); plt.savefig(f"{FIG}/fig_4_doe_pairplot.png", dpi=150); plt.close()


def fig_comparisons():
    import pandas as pd
    mm = os.path.join(DATA, "cantilever_metrics_mean.csv")
    if os.path.exists(mm):
        m = pd.read_csv(mm)
        plt.figure(figsize=(7, 4)); plt.bar(m.model, m.R2_mean, color="#4C72B0")
        plt.ylabel("Средний R2 (test)"); plt.ylim(min(0.9, m.R2_mean.min()-0.02), 1.005)
        plt.title("Консольная пластина: сравнение моделей по R2"); plt.xticks(rotation=15)
        plt.tight_layout(); plt.savefig(f"{FIG}/fig_4_5_r2_comparison.png", dpi=150); plt.close()
        plt.figure(figsize=(7, 4)); plt.bar(m.model, m.MAPE_mean_percent, color="#c0504d")
        plt.ylabel("Средний MAPE, %"); plt.title("Консольная пластина: сравнение моделей по MAPE")
        plt.xticks(rotation=15); plt.tight_layout()
        plt.savefig(f"{FIG}/fig_4_6_mape_comparison.png", dpi=150); plt.close()
        best = m.sort_values("R2_mean").iloc[-1]["model"]
    else:
        best = None; print("[skip] нет метрик -> рис. 4.5/4.6 не построены")
    pp = os.path.join(DATA, "cantilever_predictions.csv")
    if os.path.exists(pp) and best:
        pr = pd.read_csv(pp); col = f"{best}_pred"
        plot_targets = [
            {
                "target": "uy_tip",
                "filename": "fig_4_7_pred_true_uy_tip.png",
                "scale": 1000.0,
                "xlabel": r"ANSYS $u_{tip}$, мм",
                "ylabel": rf"Прогноз {best} $u_{{tip}}$, мм",
                "title": r"$u_{tip}$: прогноз против расчёта"
            },
            {
                "target": "sig_max",
                "filename": "fig_4_8_pred_true_sig_max.png",
                "scale": 1e-6,
                "xlabel": r"ANSYS $\sigma_{max}$, МПа",
                "ylabel": rf"Прогноз {best} $\sigma_{{max}}$, МПа",
                "title": r"$\sigma_{max}$: прогноз против расчёта"
            }
        ]

        for cfg in plot_targets:
            tg = cfg["target"]
            sub = pr[pr.target == tg]
            if sub.empty or col not in sub:
                continue

            yt = sub.y_true.values * cfg["scale"]
            yp = sub[col].values * cfg["scale"]

            lo = min(yt.min(), yp.min())
            hi = max(yt.max(), yp.max())

            pad = 0.05 * (hi - lo)
            lo -= pad
            hi += pad

            plt.figure(figsize=(5, 5))
            plt.scatter(yt, yp, s=28)
            plt.plot([lo, hi], [lo, hi], "k--", lw=1)

            plt.xlabel(cfg["xlabel"])
            plt.ylabel(cfg["ylabel"])
            plt.title(cfg["title"])
            plt.grid(True, alpha=0.3)

            plt.xlim(lo, hi)
            plt.ylim(lo, hi)

            plt.tight_layout()
            plt.savefig(os.path.join(FIG, cfg["filename"]), dpi=300)
            plt.close()
    # важность признаков
    imp_model = None
    for nm in ["RandomForest", "GradientBoosting"]:
        fp = os.path.join(MODELS, f"{nm}.joblib")
        if os.path.exists(fp):
            import joblib; mdl = joblib.load(fp); imp_model = (nm, mdl); break
    if imp_model:
        nm, mdl = imp_model
        try:
            est = getattr(mdl, "estimators_", [mdl])
            imp = np.mean([e.feature_importances_ for e in (est if hasattr(est, "__len__") else [mdl])], axis=0) \
                  if hasattr(mdl, "estimators_") else mdl.feature_importances_
        except Exception:
            imp = getattr(mdl, "feature_importances_", None)
        if imp is not None:
            feats = ["len", "height", "thk", "force"]; order = np.argsort(imp)
            plt.figure(figsize=(6, 3.6)); plt.barh([feats[i] for i in order], [imp[i] for i in order], color="#4C72B0")
            plt.xlabel("важность признака"); plt.title(f"Важность параметров ({nm})")
            plt.tight_layout(); plt.savefig(f"{FIG}/fig_4_9_feature_importance.png", dpi=150); plt.close()


def main():
    fig_scheme(); fig_mesh()
    fig_field_placeholder("fig_4_3_uy_field_placeholder.png", "распределение UY")
    fig_field_placeholder("fig_4_4_seqv_field_placeholder.png", "эквивалентные напряжения по Мизесу")
    fig_pairplot(); fig_comparisons()
    print("figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
