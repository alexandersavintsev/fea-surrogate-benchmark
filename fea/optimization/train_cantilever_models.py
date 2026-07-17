# -*- coding: utf-8 -*-
"""
Обучение и сравнение регрессионных моделей для задачи об изгибе
консольной пластины. Читает cantilever_dataset.csv (lx, ly, p, umax, smax, area),
делит на train/test, считает метрики по каждому выходу и усреднённо,
сохраняет таблицы и графики.

Зависимости: numpy, pandas, scikit-learn, matplotlib.
Запуск: python train_cantilever_models.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
FIG  = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)

INPUTS = ["lx", "ly", "p"]
OUTPUTS = ["umax", "smax", "area"]


def mape(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = np.abs(y) > 1e-12
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100.0)


def make_models():
    k = ConstantKernel(1.0) * RBF(length_scale=[1.0]*len(INPUTS)) + WhiteKernel(1e-3)
    return {
        "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "GPR_sklearn": Pipeline([("sc", StandardScaler()),
            ("m", GaussianProcessRegressor(kernel=k, normalize_y=True,
                                           n_restarts_optimizer=5, random_state=0))]),
        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                      learning_rate=0.05, random_state=0)),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=0),
        "MLP": Pipeline([("sc", StandardScaler()),
            ("m", MLPRegressor(hidden_layer_sizes=(32, 16), activation="tanh",
                               solver="adam", max_iter=5000, random_state=0))]),
    }


def main():
    ds = os.path.join(DATA, "cantilever_dataset.csv")
    df = pd.read_csv(ds)
    if df.dropna().shape[0] < 10:
        sys.exit(f"[stop] В {ds} недостаточно строк с данными. "
                 f"Сначала заполните датасет: generate_cantilever_doe.py --run-ansys")
    df = df.dropna().reset_index(drop=True)
    print(f"[ok] датасет: {df.shape[0]} строк")

    X = df[INPUTS].values
    Y = df[OUTPUTS].values
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)

    rows_by_target, rows_mean = [], []
    preds = {}
    for name, model in make_models().items():
        model.fit(Xtr, Ytr)
        Yp = model.predict(Xte)
        if Yp.ndim == 1:
            Yp = Yp.reshape(-1, 1)
        preds[name] = Yp
        r2s = []
        for j, tgt in enumerate(OUTPUTS):
            r2 = r2_score(Yte[:, j], Yp[:, j]); r2s.append(r2)
            rows_by_target.append(dict(model=name, target=tgt,
                R2=r2, MAE=mean_absolute_error(Yte[:, j], Yp[:, j]),
                MSE=mean_squared_error(Yte[:, j], Yp[:, j]),
                RMSE=mean_squared_error(Yte[:, j], Yp[:, j])**0.5,
                MAPE_percent=mape(Yte[:, j], Yp[:, j])))
        rows_mean.append(dict(model=name, R2_mean=float(np.mean(r2s)),
            MAPE_percent_mean=float(np.mean(
                [mape(Yte[:, j], Yp[:, j]) for j in range(len(OUTPUTS))]))))
        print(f"  {name:18s} mean R2 = {np.mean(r2s):.4f}")

    by_t = pd.DataFrame(rows_by_target)
    mn = pd.DataFrame(rows_mean)
    by_t.to_csv(os.path.join(DATA, "cantilever_metrics_by_target.csv"), index=False)
    mn.to_csv(os.path.join(DATA, "cantilever_metrics_mean.csv"), index=False)
    print("[ok] метрики сохранены")

    # графики сравнения
    plt.figure(figsize=(7, 4))
    plt.bar(mn.model, mn.R2_mean)
    plt.ylabel("Средний R2 (test)"); plt.ylim(0, 1.02); plt.xticks(rotation=20)
    plt.title("Консольная пластина: сравнение моделей по R2")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "cantilever_r2_comparison.png"), dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(mn.model, mn.MAPE_percent_mean, color="#c0504d")
    plt.ylabel("Средний MAPE, %"); plt.xticks(rotation=20)
    plt.title("Консольная пластина: сравнение моделей по MAPE")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "cantilever_mape_comparison.png"), dpi=150); plt.close()

    best = mn.sort_values("R2_mean").iloc[-1]["model"]
    for j, tgt in [(0, "umax"), (1, "smax")]:
        plt.figure(figsize=(5, 5))
        yt, yp = Yte[:, j], preds[best][:, j]
        plt.scatter(yt, yp, s=24)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        plt.plot([lo, hi], [lo, hi], "k--", lw=1)
        plt.xlabel(f"ANSYS {tgt}"); plt.ylabel(f"Прогноз {tgt} ({best})")
        plt.title(f"{tgt}: прогноз против расчёта")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"cantilever_pred_vs_true_{tgt}.png"), dpi=150); plt.close()

    # ошибки
    plt.figure(figsize=(7, 4))
    for name in preds:
        err = np.abs(preds[name][:, 0] - Yte[:, 0])
        plt.plot(np.sort(err), label=name)
    plt.xlabel("точка (сорт.)"); plt.ylabel("|ошибка| umax"); plt.legend(fontsize=8)
    plt.title("Распределение абсолютной ошибки по umax")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "cantilever_errors.png"), dpi=150); plt.close()
    print(f"[ok] графики в {FIG}")


if __name__ == "__main__":
    main()
