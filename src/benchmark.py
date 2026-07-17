"""K-fold кросс-валидация всех моделей на всех датасетах.

Запуск:
    python -m src.benchmark                       # синтетические датасеты
    python -m src.benchmark --csv data/plastic.csv --target epl_max
                                                  # реальные данные ANSYS из CSV
Результат: results/benchmark.csv + сводная таблица в консоли.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from .synthetic import make_dataset, DATASETS
from .models import get_models


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) or 1e-12
    r2 = 1.0 - ss_res / ss_tot
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    mape = float(np.nanmean(np.abs(err / denom)) * 100.0)
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}


def cv_eval(X, y, model, k=5, seeds=(0, 1, 2)):
    r2s, mapes = [], []
    for seed in seeds:
        kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        for tr, te in kf.split(X):
            m = clone(model)
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            mt = metrics(y[te], p)
            r2s.append(mt["R2"])
            mapes.append(mt["MAPE"])
    return float(np.mean(r2s)), float(np.std(r2s)), float(np.nanmean(mapes))


def run_synthetic(n=400, d=4, noise=0.02, k=5):
    rows = []
    for ds in DATASETS:
        X, y = make_dataset(ds, n=n, d=d, noise=noise, seed=0)
        for name, model in get_models().items():
            r2_mean, r2_std, mape = cv_eval(X, y, model, k=k)
            rows.append(dict(dataset=ds, model=name,
                             R2_mean=round(r2_mean, 4), R2_std=round(r2_std, 4),
                             MAPE=round(mape, 3)))
            print(f"{ds:20s} {name:20s} R2={r2_mean:.4f}±{r2_std:.4f}  MAPE={mape:.2f}%")
    return pd.DataFrame(rows)


def run_csv(path, target, drop=None):
    df = pd.read_csv(path)
    y = df[target].to_numpy(float)
    drop_cols = {target}
    drop_cols.update(c for c in (drop or []) if c in df.columns)
    drop_cols.update(c for c in ("case_id", "id") if c in df.columns)
    feats = [c for c in df.columns if c not in drop_cols]
    X = df[feats].to_numpy(float)
    print(f"признаки: {feats} | цель: {target}")
    rows = []
    for name, model in get_models().items():
        r2_mean, r2_std, mape = cv_eval(X, y, model)
        rows.append(dict(dataset=os.path.basename(path), model=name,
                         R2_mean=round(r2_mean, 4), R2_std=round(r2_std, 4),
                         MAPE=round(mape, 3)))
        print(f"{name:20s} R2={r2_mean:.4f}±{r2_std:.4f}  MAPE={mape:.2f}%")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="CSV с реальными данными (ANSYS)")
    ap.add_argument("--target", default="epl_max", help="имя целевого столбца в CSV")
    ap.add_argument("--drop", default="", help="доп. столбцы-выходы, исключить из признаков (через запятую)")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default="results/benchmark.csv")
    args = ap.parse_args()

    drop = [c.strip() for c in args.drop.split(",") if c.strip()]
    df = run_csv(args.csv, args.target, drop=drop) if args.csv else run_synthetic(n=args.n)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nСохранено: {args.out}")
    print("\n=== Лучшая модель по R2 на каждом датасете ===")
    print(df.loc[df.groupby('dataset')['R2_mean'].idxmax(), ['dataset', 'model', 'R2_mean']]
          .to_string(index=False))


if __name__ == "__main__":
    main()
