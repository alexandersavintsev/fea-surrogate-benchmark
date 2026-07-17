"""Итоговый график кроссовера: GPR против лучшего дерева по всем задачам.

Читает results/crossover_all.csv (создаёт run_all.py) и рисает столбчатую
диаграмму: для каждой задачи - R2 у GPR и у лучшего ансамбля деревьев,
задачи сгруппированы по режиму (гладкие / негладкие).

Запуск: python -m src.plot_crossover
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TREE_MODELS = {"RandomForest", "GradientBoosting(sk)", "XGBoost", "LightGBM", "CatBoost"}


def main(src="results/crossover_all.csv", out="results/crossover.png"):
    df = pd.read_csv(src).dropna(subset=["R2"])
    order = (df[["dataset", "regime"]].drop_duplicates()
             .sort_values(["regime", "dataset"]))
    names, gpr_vals, tree_vals, regimes = [], [], [], []
    for _, r in order.iterrows():
        sub = df[df.dataset == r.dataset]
        gpr = sub[sub.model == "GPR"].R2.max()
        trees = sub[sub.model.isin(TREE_MODELS)].R2
        names.append(r.dataset)
        regimes.append(r.regime)
        gpr_vals.append(gpr)
        tree_vals.append(trees.max() if not trees.empty else np.nan)

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.1), 4.5))
    ax.bar(x - w/2, gpr_vals, w, label="GPR", color="#0E7C7B")
    ax.bar(x + w/2, tree_vals, w, label="лучший ансамбль деревьев", color="#B85042")
    ax.set_ylabel("R² (k-fold CV)")
    ax.set_title("Кроссовер: где выигрывает GPR, а где деревья")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(min(0.0, np.nanmin(gpr_vals + tree_vals)), 1.02)
    ax.legend()
    # разделитель гладкие/негладкие
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1]:
            ax.axvline(i - 0.5, color="#888", ls="--", lw=0.8)
    ax.grid(axis="y", color="#E2E8F0", lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
