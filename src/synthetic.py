"""Синтетические датасеты с контролируемой гладкостью отклика.

Цель - воспроизводимо показать границу: на гладких функциях лучше GPR,
на негладких (порог, разрыв, высокая частота, резонансный пик) - деревья.
Все входы генерируются методом латинского гиперкуба (LHS).
"""
from __future__ import annotations
import numpy as np

DATASETS = [
    "smooth_powerlaw",      # гладкая мультипликативная (как упругая МКЭ) -> GPR
    "threshold_relu",       # порог по гиперплоскости (как пластичность)  -> деревья
    "step_discontinuous",   # разрыв                                       -> деревья
    "high_frequency",       # высокочастотная осцилляция                   -> деревья/MLP
    "resonance_peak",       # острый пик (резонанс)                        -> деревья
]


def lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Латинский гиперкуб в [0, 1]^d."""
    cut = np.linspace(0.0, 1.0, n + 1)
    lo, hi = cut[:n][:, None], cut[1:][:, None]
    pts = lo + rng.uniform(size=(n, d)) * (hi - lo)
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def make_dataset(name: str, n: int = 400, d: int = 4, noise: float = 0.0, seed: int = 0):
    """Вернуть (X, y). X в [0,1]^d, y - скалярный отклик."""
    rng = np.random.default_rng(seed)
    X = lhs(n, d, rng)
    x = X * 2.0 - 1.0  # [-1, 1]^d

    if name == "smooth_powerlaw":
        h = 0.2 + 0.8 * X[:, 0]
        L = 0.5 + 1.5 * X[:, 1]
        F = 0.5 + 1.5 * X[:, 2]
        y = F * L ** 3 / h ** 3
    elif name == "threshold_relu":
        w = rng.normal(size=d)
        s = x @ w
        y = np.maximum(0.0, s) ** 1.5          # ровно 0 ниже порога, затем рост
    elif name == "step_discontinuous":
        s = x[:, 0] + 0.5 * x[:, 1]
        y = (s > 0).astype(float) + 0.1 * s    # скачок
    elif name == "high_frequency":
        y = np.sin(6.0 * np.pi * X[:, 0]) * np.cos(4.0 * np.pi * X[:, 1])
    elif name == "resonance_peak":
        f0 = 0.3 + 0.4 * X[:, 1]               # положение пика зависит от X1
        gamma = 0.02
        y = 1.0 / ((X[:, 0] - f0) ** 2 + gamma ** 2)
    else:
        raise ValueError(f"unknown dataset: {name}")

    y = np.asarray(y, dtype=float)
    if noise:
        y = y + noise * float(np.std(y)) * rng.normal(size=y.shape)
    return X, y


if __name__ == "__main__":
    for ds in DATASETS:
        X, y = make_dataset(ds, n=50, d=4, seed=0)
        print(f"{ds:20s} X{X.shape} y[min={y.min():.3g}, max={y.max():.3g}]")
