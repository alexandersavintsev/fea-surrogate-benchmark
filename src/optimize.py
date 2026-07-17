"""Оптимизация конструкции на обученном суррогате.

Идея: обучаем быстрый суррогат на данных (КЭ или синтетика), затем глобально
оптимизируем целевую функцию НА СУРРОГАТЕ (тысячи вызовов за секунды), а найденный
оптимум проверяем «дорогой» функцией (в реальном проекте - полным расчётом ANSYS).

Демонстрация ниже использует синтетическую «дорогую» функцию вместо ANSYS, чтобы
скрипт был самодостаточным. В реальном проекте замените true_response() вызовом
расчёта или предобученной модели на КЭ-данных.

Запуск: python -m src.optimize
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import differential_evolution

from .synthetic import lhs
from .models import get_models


def true_response(X: np.ndarray) -> np.ndarray:
    """Имитация дорогого отклика (заменить на ANSYS). Минимум напряжения при
    геометрических ограничениях; гладкая мультипликативная зависимость."""
    h = 0.2 + 0.8 * X[:, 0]
    L = 0.5 + 1.5 * X[:, 1]
    b = 0.2 + 0.8 * X[:, 2]
    F = 0.5 + 1.5 * X[:, 3]
    sigma = 6.0 * F * L / (b * h ** 2)   # балочная оценка напряжения
    return sigma


def main(d: int = 4, n_train: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = lhs(n_train, d, rng)
    y = true_response(X)

    # лучший гладкий суррогат - GPR
    model = get_models()["GPR"]
    model.fit(X, y)

    bounds = [(0.0, 1.0)] * d

    def surrogate_obj(x):
        return float(model.predict(x.reshape(1, -1))[0])

    # ограничение: масса (~ b*h*L) не больше порога -> штраф
    def mass(x):
        h = 0.2 + 0.8 * x[0]; L = 0.5 + 1.5 * x[1]; b = 0.2 + 0.8 * x[2]
        return b * h * L

    mass_cap = 0.6

    def penalized(x):
        pen = 1e3 * max(0.0, mass(x) - mass_cap)
        return surrogate_obj(x) + pen

    t0 = time.time()
    res = differential_evolution(penalized, bounds, seed=seed, maxiter=200, tol=1e-7)
    t_surr = time.time() - t0

    x_opt = res.x
    sigma_surrogate = surrogate_obj(x_opt)
    sigma_true = float(true_response(x_opt.reshape(1, -1))[0])  # «проверка ANSYS»

    print("=== Оптимизация на суррогате ===")
    print(f"x* = {np.round(x_opt, 3)}")
    print(f"масса(x*) = {mass(x_opt):.3f} (ограничение <= {mass_cap})")
    print(f"напряжение: суррогат = {sigma_surrogate:.3f}, проверка = {sigma_true:.3f}")
    print(f"расхождение суррогат/проверка = {abs(sigma_surrogate - sigma_true) / sigma_true * 100:.2f}%")
    print(f"время оптимизации на суррогате = {t_surr:.3f} c "
          f"(~{res.nfev} вызовов суррогата вместо стольких же расчётов ANSYS)")


if __name__ == "__main__":
    main()
