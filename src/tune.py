"""Optuna-тюнинг гиперпараметров моделей по каждому датасету.

Настраиваются ансамбли деревьев (RF, GB, XGBoost, LightGBM, CatBoost); GPR, Ridge и
MLP берутся с дефолтами (GPR самонастраивается внутри по marginal likelihood - это
честное сравнение). Для каждого датасета Optuna максимизирует R2 по быстрой 4-fold CV,
затем лучший конфиг оценивается полной CV (5-fold x 3 сида).

Запуск:
    python -m src.tune --datasets resonance_cantilever --trials 30   # быстрый прогон одной задачи
    python -m src.tune                                               # все датасеты (долго)
Результат: results/crossover_tuned.csv + сводка кроссовера.
"""
from __future__ import annotations
import argparse, os, warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.base import clone
from sklearn.model_selection import KFold

from .run_all import CONFIG, load_csv
from .synthetic import make_dataset
from .models import get_models
from .benchmark import cv_eval, metrics

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNABLE = ["RandomForest", "GradientBoosting(sk)", "XGBoost", "LightGBM", "CatBoost"]
TREE_MODELS = set(TUNABLE)


def build(name, p):
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=0, n_jobs=-1, **p)
    if name == "GradientBoosting(sk)":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=0, **p)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=0, n_jobs=-1, **p)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=0, n_jobs=-1, verbose=-1, **p)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=0, verbose=False, **p)
    raise ValueError(name)


def suggest(name, t):
    if name == "RandomForest":
        return dict(n_estimators=t.suggest_int("n_estimators", 200, 800, step=100),
                    max_depth=t.suggest_int("max_depth", 3, 20),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 6),
                    max_features=t.suggest_float("max_features", 0.4, 1.0))
    if name == "GradientBoosting(sk)":
        return dict(n_estimators=t.suggest_int("n_estimators", 100, 500, step=50),
                    max_depth=t.suggest_int("max_depth", 2, 5),
                    learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    subsample=t.suggest_float("subsample", 0.7, 1.0))
    if name == "XGBoost":
        return dict(n_estimators=t.suggest_int("n_estimators", 200, 800, step=100),
                    max_depth=t.suggest_int("max_depth", 2, 8),
                    learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    subsample=t.suggest_float("subsample", 0.7, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 5.0, log=True))
    if name == "LightGBM":
        return dict(n_estimators=t.suggest_int("n_estimators", 200, 900, step=100),
                    num_leaves=t.suggest_int("num_leaves", 15, 127),
                    learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    subsample=t.suggest_float("subsample", 0.7, 1.0),
                    min_child_samples=t.suggest_int("min_child_samples", 5, 40))
    if name == "CatBoost":
        return dict(iterations=t.suggest_int("iterations", 200, 900, step=100),
                    depth=t.suggest_int("depth", 3, 8),
                    learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                    l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1.0, 9.0))
    raise ValueError(name)


def quick_r2(X, y, est, k=4):
    kf = KFold(n_splits=k, shuffle=True, random_state=0)
    r2s = []
    for tr, te in kf.split(X):
        m = clone(est); m.fit(X[tr], y[tr])
        r2s.append(metrics(y[te], m.predict(X[te]))["R2"])
    return float(np.mean(r2s))


def tune_model(name, X, y, trials):
    def obj(t):
        try:
            return quick_r2(X, y, build(name, suggest(name, t)))
        except Exception:
            return -1e9
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(obj, n_trials=trials, show_progress_bar=False)
    return build(name, study.best_params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="all", help="имена через запятую или 'all'")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--out", default="results/crossover_tuned.csv")
    a = ap.parse_args()

    want = None if a.datasets == "all" else set(s.strip() for s in a.datasets.split(","))
    avail = set(get_models().keys())
    default_models = get_models()   # для нетюнящихся (Ridge, GPR, MLP)

    rows = []
    for spec in CONFIG:
        if want and spec["name"] not in want:
            continue
        if spec["kind"] == "csv":
            if not os.path.exists(spec["path"]):
                print(f"[пропуск] {spec['name']}: нет {spec['path']}"); continue
            X, y = load_csv(spec)
        else:
            X, y = make_dataset(spec["name"].replace("synth_", ""), n=400, d=4, noise=0.02, seed=0)

        print(f"\n=== {spec['name']} ({spec['regime']}, N={len(y)}) ===")
        for mname in default_models:
            if mname in TREE_MODELS and mname in avail:
                est = tune_model(mname, X, y, a.trials)
                tag = "tuned"
            else:
                est = default_models[mname]
                tag = "default"
            r2, r2s, mape = cv_eval(X, y, est, k=5, seeds=(0, 1, 2))
            rows.append(dict(dataset=spec["name"], regime=spec["regime"], model=mname,
                             tuned=(tag == "tuned"), R2=round(r2, 4), MAPE=round(mape, 3)))
            print(f"  {mname:20s} [{tag:7s}] R2={r2:.4f}  MAPE={mape:.2f}%")

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv(a.out, index=False)

    print("\n" + "=" * 64)
    print("КРОССОВЕР (после тюнинга): GPR против лучшего дерева")
    print("=" * 64)
    print(f"{'dataset':22s} {'режим':11s} {'лучшая':16s} {'GPR':>7s} {'best_tree':>10s}  вывод")
    for name in df.dataset.unique():
        sub = df[df.dataset == name].dropna(subset=["R2"])
        if sub.empty:
            continue
        best = sub.loc[sub.R2.idxmax()]
        gpr = sub[sub.model == "GPR"].R2.max()
        trees = sub[sub.model.isin(TREE_MODELS)].R2
        bt = trees.max() if not trees.empty else float("nan")
        verdict = "GPR" if (np.isnan(bt) or gpr >= bt) else "дерево"
        print(f"{name:22s} {sub.regime.iloc[0]:11s} {best['model']:16s} {gpr:7.3f} {bt:10.3f}  -> {verdict}")
    print(f"\nСохранено: {a.out}")


if __name__ == "__main__":
    main()
