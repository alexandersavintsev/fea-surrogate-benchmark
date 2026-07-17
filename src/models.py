"""Зоопарк регрессионных моделей.

Базовые - всегда (scikit-learn). Современный бустинг (XGBoost / LightGBM / CatBoost)
подключается, если установлен; иначе мягко пропускается с предупреждением.
Так закрывается замечание «почему нет CatBoost/LightGBM».
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _gpr():
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3)
    return make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                 n_restarts_optimizer=2, random_state=0),
    )


def _optional(name: str):
    try:
        if name == "XGBoost":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                subsample=0.9, colsample_bytree=0.9,
                                random_state=0, n_jobs=-1)
        if name == "LightGBM":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=600, num_leaves=31, learning_rate=0.05,
                                 subsample=0.9, random_state=0, n_jobs=-1, verbose=-1)
        if name == "CatBoost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(iterations=600, depth=6, learning_rate=0.05,
                                     random_seed=0, verbose=False)
    except Exception as exc:  # библиотека не установлена
        print(f"[skip] {name} недоступен: {exc}")
        return None


def get_models() -> dict:
    models = {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "GPR": _gpr(),
        "MLP": make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(128, 128),
                                          max_iter=2000, random_state=0)),
        "RandomForest": RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1),
        "GradientBoosting(sk)": GradientBoostingRegressor(random_state=0),
    }
    for name in ("XGBoost", "LightGBM", "CatBoost"):
        est = _optional(name)
        if est is not None:
            models[name] = est
    return models


if __name__ == "__main__":
    print("Доступные модели:", list(get_models().keys()))
