# -*- coding: utf-8 -*-
"""
Обучение и сравнение регрессионных моделей для задачи об изгибе консольной
пластины. Читает ../data/cantilever_dataset.csv.

Признаки : len, height, thk, force
Выходы   : uy_tip, uy_max, sig_max, area  (area = len*height - геометрический
           контрольный выход; основной физический смысл у перемещений и напряжения)
Разбиение: train/test = 80/20, random_state = 42

Зависимости: numpy, pandas, scikit-learn, matplotlib, joblib.
Выходы:
  ../data/cantilever_metrics_by_target.csv
  ../data/cantilever_metrics_mean.csv
  ../data/cantilever_predictions.csv
  ../models/<model>.joblib
"""
import os, sys
import numpy as np, pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data"); MODELS = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)
FEATURES = ["len", "height", "thk", "force"]
TARGETS = ["uy_tip", "uy_max", "sig_max", "area"]
RS = 42


def mape(y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    m = np.abs(y) > 1e-12
    return float(np.mean(np.abs((y[m]-yh[m])/y[m]))*100.0)


def make_models(n_train):
    k = ConstantKernel(1.0)*RBF([1.0]*len(FEATURES)) + WhiteKernel(1e-3)
    models = {
        "Ridge": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "GPR": Pipeline([("sc", StandardScaler()),
            ("m", GaussianProcessRegressor(kernel=k, normalize_y=True,
                                           n_restarts_optimizer=5, random_state=0))]),
        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=400, max_depth=3,
                                      learning_rate=0.05, random_state=0)),
        "RandomForest": RandomForestRegressor(n_estimators=500, random_state=0),
    }
    '''if n_train >= 40:   # MLP только при достаточной выборке
        models["MLP"] = Pipeline([("sc", StandardScaler()),
            ("m", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                               solver="adam", max_iter=8000, random_state=0))])'''
    return models


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(DATA, "cantilever_dataset.csv"))
    ap.add_argument("--outdir", default=DATA)
    ap.add_argument("--models", default=MODELS)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True); os.makedirs(a.models, exist_ok=True)
    ds = a.dataset
    if not os.path.exists(ds):
        sys.exit("[stop] Нет data/cantilever_dataset.csv. Сначала выполните ANSYS-расчёты "
                 "(scripts/02_run_ansys_batch.py).")
    df = pd.read_csv(ds)
    if df[TARGETS].dropna().shape[0] < 10:
        sys.exit("[stop] В датасете нет выходных величин. Выполните ANSYS-расчёты.")
    df = df.dropna(subset=FEATURES+TARGETS).reset_index(drop=True)
    print(f"[ok] датасет: {len(df)} строк")

    X = df[FEATURES].values; Y = df[TARGETS].values
    cid = df["case_id"].values if "case_id" in df else np.arange(len(df))
    Xtr, Xte, Ytr, Yte, _, cte = train_test_split(X, Y, cid, test_size=0.2, random_state=RS)

    rows_t, rows_m, preds = [], [], {}
    for name, model in make_models(len(Xtr)).items():
        try:
            model.fit(Xtr, Ytr); Yp = model.predict(Xte)
        except Exception as e:
            print(f"[skip] {name}: {e}"); continue
        if Yp.ndim == 1: Yp = Yp.reshape(-1, 1)
        preds[name] = Yp
        r2s, maes, mses, rmses, mapes = [], [], [], [], []
        for j, tg in enumerate(TARGETS):
            r2 = r2_score(Yte[:, j], Yp[:, j]); mae = mean_absolute_error(Yte[:, j], Yp[:, j])
            mse = mean_squared_error(Yte[:, j], Yp[:, j]); rmse = mse**0.5; mp = mape(Yte[:, j], Yp[:, j])
            rows_t.append(dict(model=name, target=tg, R2=r2, MAE=mae, MSE=mse, RMSE=rmse, MAPE_percent=mp))
            r2s.append(r2); maes.append(mae); mses.append(mse); rmses.append(rmse); mapes.append(mp)
        rows_m.append(dict(model=name, R2_mean=np.mean(r2s), MAE_mean=np.mean(maes),
                           MSE_mean=np.mean(mses), RMSE_mean=np.mean(rmses), MAPE_mean_percent=np.mean(mapes)))
        joblib.dump(model, os.path.join(a.models, f"{name}.joblib"))
        print(f"  {name:16s} mean R2={np.mean(r2s):.4f}  mean MAPE={np.mean(mapes):.3f}%")

    pd.DataFrame(rows_t).to_csv(os.path.join(a.outdir, "cantilever_metrics_by_target.csv"), index=False)
    pd.DataFrame(rows_m).to_csv(os.path.join(a.outdir, "cantilever_metrics_mean.csv"), index=False)

    # предсказания (длинный формат)
    prows = []
    for j, tg in enumerate(TARGETS):
        for i in range(len(Xte)):
            row = {"case_id": cte[i], "target": tg, "y_true": Yte[i, j]}
            for name in preds: row[f"{name}_pred"] = preds[name][i, j]
            prows.append(row)
    pd.DataFrame(prows).to_csv(os.path.join(a.outdir, "cantilever_predictions.csv"), index=False)
    print("[ok] метрики и предсказания сохранены в data/, модели в models/")


if __name__ == "__main__":
    main()
