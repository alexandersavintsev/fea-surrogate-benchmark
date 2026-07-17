"""Единый прогон всех задач одной командой -> сводная таблица кроссовера.

Идея: на гладких задачах (Кирш, упругая консоль) лучшая - GPR; на негладких
(пластическая консоль, синтетические разрывные/пороговые) - ансамбли деревьев.
Скрипт прогоняет 8 моделей по k-fold CV на каждом доступном датасете, помечает
режим (smooth / non-smooth) и собирает результаты в один CSV + печатает
кроссовер (GPR против лучшего дерева).

Запуск:
    python -m src.run_all
Датасеты, файлов которых ещё нет (например пластика до прогона ANSYS),
пропускаются автоматически.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .synthetic import make_dataset
from .models import get_models
from .benchmark import cv_eval

TREE_MODELS = {"RandomForest", "GradientBoosting(sk)", "XGBoost", "LightGBM", "CatBoost"}

# kind: "csv" (реальные данные) или "synth" (синтетика)
CONFIG = [
    # --- гладкие упругие задачи -> ожидаем победу GPR ---
    dict(name="kirsch_100",     kind="csv", regime="smooth",
         path="data/kirsch/kirsch_romai_doe_100.csv", feats=["a","b","lx","ly"], target="smax"),
    dict(name="kirsch_30_A",    kind="csv", regime="smooth",
         path="data/kirsch/kirsch_30_seedA.csv", feats=["a","b","lx","ly"], target="smax"),
    dict(name="kirsch_30_B",    kind="csv", regime="smooth",
         path="data/kirsch/kirsch_30_seedB.csv", feats=["a","b","lx","ly"], target="smax"),
    dict(name="cantilever_80",  kind="csv", regime="smooth",
         path="data/cantilever/cantilever_dataset.csv",
         feats=["len","height","thk","force"], target="sig_max"),
    dict(name="cantilever_120", kind="csv", regime="smooth",
         path="data/cantilever/ext/cantilever_dataset_ext.csv",
         feats=["len","height","thk","force"], target="sig_max"),
    dict(name="cantilever_240", kind="csv", regime="smooth",
         path="data/cantilever/ext_h020_N240/cantilever_dataset_ext_h020_N240.csv",
         feats=["len","height","thk","force"], target="sig_max"),
    # --- негладкая пластическая консоль (появится после прогона ANSYS) ---
    dict(name="plastic_cantilever", kind="csv", regime="non-smooth",
         path="data/plastic_results.csv",
         feats=["len","height","thk","force"], target="epl_max"),
    # --- резонансная задача (появится после гармонического прогона ANSYS) ---
    dict(name="resonance_cantilever", kind="csv", regime="non-smooth",
         path="data/harmonic_results.csv",
         feats=["len","height","thk","force"], target="uy_amp"),
    # --- синтетика: контроль механизма ---
    dict(name="synth_smooth_powerlaw",   kind="synth", regime="smooth"),
    dict(name="synth_threshold_relu",    kind="synth", regime="non-smooth"),
    dict(name="synth_step_discontinuous",kind="synth", regime="non-smooth"),
    dict(name="synth_high_frequency",    kind="synth", regime="non-smooth"),
    dict(name="synth_resonance_peak",    kind="synth", regime="non-smooth"),
]


def load_csv(spec):
    df = pd.read_csv(spec["path"], encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    X = df[spec["feats"]].to_numpy(float)
    y = df[spec["target"]].to_numpy(float)
    return X, y


def main():
    models = get_models()
    print("Модели:", ", ".join(models))
    rows = []
    for spec in CONFIG:
        if spec["kind"] == "csv":
            if not os.path.exists(spec["path"]):
                print(f"[пропуск] {spec['name']}: нет файла {spec['path']}")
                continue
            X, y = load_csv(spec)
        else:
            ds = spec["name"].replace("synth_", "")
            X, y = make_dataset(ds, n=400, d=4, noise=0.02, seed=0)
        print(f"\n=== {spec['name']} ({spec['regime']}, N={len(y)}) ===")
        for mname, model in models.items():
            try:
                r2, r2s, mape = cv_eval(X, y, model, k=5, seeds=(0, 1, 2))
            except Exception as exc:
                print(f"  {mname:20s} ошибка: {exc}")
                r2, r2s, mape = float("nan"), float("nan"), float("nan")
            rows.append(dict(dataset=spec["name"], regime=spec["regime"], model=mname,
                             R2=round(r2, 4), R2_std=round(r2s, 4), MAPE=round(mape, 3)))
            print(f"  {mname:20s} R2={r2:.4f}  MAPE={mape:.2f}%")

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/crossover_all.csv", index=False)

    # --- сводка кроссовера: GPR против лучшего дерева ---
    print("\n" + "=" * 64)
    print("КРОССОВЕР: лучшая модель и GPR против лучшего дерева")
    print("=" * 64)
    print(f"{'dataset':22s} {'режим':11s} {'лучшая':16s} {'GPR':>7s} {'best_tree':>10s}  вывод")
    for name in df.dataset.unique():
        sub = df[df.dataset == name].dropna(subset=["R2"])
        if sub.empty:
            continue
        regime = sub.regime.iloc[0]
        best = sub.loc[sub.R2.idxmax()]
        gpr = sub[sub.model == "GPR"].R2.max()
        trees = sub[sub.model.isin(TREE_MODELS)].R2
        best_tree = trees.max() if not trees.empty else float("nan")
        verdict = "GPR" if (np.isnan(best_tree) or gpr >= best_tree) else "дерево"
        print(f"{name:22s} {regime:11s} {best['model']:16s} {gpr:7.3f} {best_tree:10.3f}  -> {verdict}")
    print(f"\nСохранено: results/crossover_all.csv")


if __name__ == "__main__":
    main()
