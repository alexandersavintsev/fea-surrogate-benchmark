"""Smoke-тесты: данные генерируются, базовые модели обучаются, метрики считаются."""
import numpy as np
from src.synthetic import make_dataset, DATASETS
from src.benchmark import metrics, cv_eval


def test_datasets_shapes():
    for ds in DATASETS:
        X, y = make_dataset(ds, n=60, d=4, seed=0)
        assert X.shape == (60, 4)
        assert y.shape == (60,)
        assert np.isfinite(y).all()


def test_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = metrics(y, y)
    assert m["R2"] > 0.999
    assert m["MAE"] < 1e-9


def test_cv_runs_on_basic_models():
    from sklearn.linear_model import Ridge
    X, y = make_dataset("smooth_powerlaw", n=80, d=4, seed=0)
    r2_mean, r2_std, mape = cv_eval(X, y, Ridge(alpha=1.0), k=4, seeds=(0,))
    assert np.isfinite(r2_mean)
