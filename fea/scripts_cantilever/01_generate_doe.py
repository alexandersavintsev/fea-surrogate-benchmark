# -*- coding: utf-8 -*-
"""
Генерация DOE-выборки (Latin Hypercube Sampling) для задачи об изгибе
консольно защемлённой пластины (PLANE182) и проверка базовой точки по
балочной формуле Эйлера-Бернулли.

Зависимости: numpy, pandas. scipy - опционально (scipy.stats.qmc).
Запуск:  python 01_generate_doe.py
Выходы:
  ../data/cantilever_doe_inputs.csv     case_id, len, height, thk, force
  ../data/cantilever_theory_check.csv   проверка прогиба базовой точки
"""
import os, csv
import numpy as np

N = 80          # число расчётных точек DOE (можно изменить)
SEED = 20240608

BOUNDS = {              # диапазоны входных параметров
    "len":    (6.0, 14.0),    # длина консоли, м
    "height": (0.6, 1.4),     # высота расчётной области, м
    "thk":    (0.06, 0.14),   # толщина, м
    "force":  (500.0, 2000.0) # полная вертикальная сила на торце, Н
}
E = 2.1e11    # модуль Юнга, Па

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
os.makedirs(DATA, exist_ok=True)


def lhs_unit(n, d, seed):
    """LHS в [0,1]^d. Использует scipy.stats.qmc при наличии, иначе numpy."""
    try:
        from scipy.stats import qmc
        return qmc.LatinHypercube(d=d, seed=seed).random(n)
    except Exception:
        rng = np.random.default_rng(seed)
        cut = np.linspace(0, 1, n + 1)
        u = rng.random((n, d))
        pts = cut[:n, None] + u * (1.0 / n)
        for j in range(d):
            rng.shuffle(pts[:, j])
        return pts


def main():
    names = list(BOUNDS.keys())
    s01 = lhs_unit(N, len(names), SEED)
    X = np.empty_like(s01)
    for j, nm in enumerate(names):
        lo, hi = BOUNDS[nm]
        X[:, j] = lo + s01[:, j] * (hi - lo)

    # контроль диапазонов
    for j, nm in enumerate(names):
        lo, hi = BOUNDS[nm]
        assert X[:, j].min() >= lo - 1e-9 and X[:, j].max() <= hi + 1e-9, nm

    inp = os.path.join(DATA, "cantilever_doe_inputs.csv")
    with open(inp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "len", "height", "thk", "force"])
        for i in range(N):
            w.writerow([f"{i+1:04d}"] + [f"{X[i,j]:.6f}" for j in range(len(names))])
    print(f"[ok] {inp}  ({N} точек)")
    print(f"     len {X[:,0].min():.2f}..{X[:,0].max():.2f} | "
          f"height {X[:,1].min():.2f}..{X[:,1].max():.2f} | "
          f"thk {X[:,2].min():.3f}..{X[:,2].max():.3f} | "
          f"force {X[:,3].min():.0f}..{X[:,3].max():.0f}")

    # ---- проверка базовой точки по балочной формуле ----
    L, h, t, F = 10.0, 1.0, 0.1, 1000.0
    I = t * h**3 / 12.0
    u_theory = F * L**3 / (3.0 * E * I)
    chk = os.path.join(DATA, "cantilever_theory_check.csv")
    with open(chk, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["len", "height", "thk", "force", "I", "u_theory",
                    "uy_tip_ansys", "rel_error_percent"])
        w.writerow([L, h, t, F, f"{I:.6e}", f"{u_theory:.6e}", "", ""])
    print(f"[ok] {chk}")
    print(f"     базовая точка: I={I:.6e} м^4, u_theory={u_theory:.6e} м "
          f"(uy_tip_ansys и ошибка - после расчёта ANSYS)")


if __name__ == "__main__":
    main()
