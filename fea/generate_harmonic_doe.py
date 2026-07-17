"""Генерация DOE для резонансной (гармонической) задачи + проверка, сколько
точек попадает в зону резонанса (где первая собственная частота ~ f0).

Первая собственная частота консоли (оценка): f1 ~ 835 * height / len^2  (Гц)
при стали (E=2.1e11, rho=7850). Возбуждаем на f0 = 9 Гц. Где f1 ~ f0 - резонанс.

Запуск:
    python fea/generate_harmonic_doe.py
Выход: data/harmonic_doe_inputs.csv  (case_id, len, height, thk, force)
"""
from __future__ import annotations
import argparse, os
import numpy as np

F0 = 9.0  # частота возбуждения, Гц (должна совпадать с f0 в макросе)


def lhs(n, d, rng):
    cut = np.linspace(0.0, 1.0, n + 1)
    lo, hi = cut[:n][:, None], cut[1:][:, None]
    pts = lo + rng.uniform(size=(n, d)) * (hi - lo)
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--lmin", type=float, default=8.0);  ap.add_argument("--lmax", type=float, default=12.0)
    ap.add_argument("--hmin", type=float, default=0.8);  ap.add_argument("--hmax", type=float, default=1.2)
    ap.add_argument("--bmin", type=float, default=0.08); ap.add_argument("--bmax", type=float, default=0.12)
    ap.add_argument("--fmin", type=float, default=500.0); ap.add_argument("--fmax", type=float, default=2000.0)
    ap.add_argument("--out", default="data/harmonic_doe_inputs.csv")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    U = lhs(a.n, 4, rng)
    L = a.lmin + U[:, 0] * (a.lmax - a.lmin)
    h = a.hmin + U[:, 1] * (a.hmax - a.hmin)
    b = a.bmin + U[:, 2] * (a.bmax - a.bmin)
    F = a.fmin + U[:, 3] * (a.fmax - a.fmin)

    f1 = 835.0 * h / L**2               # оценка первой собственной частоты
    near = np.mean(np.abs(f1 - F0) / F0 < 0.3)   # в пределах +-30% от f0

    print(f"f0 (возбуждение) = {F0} Гц")
    print(f"оценка f1: min={f1.min():.1f}  med={np.median(f1):.1f}  max={f1.max():.1f} Гц")
    print(f"доля точек около резонанса (|f1-f0|/f0 < 0.3): {near*100:.0f} %")
    if not (0.15 <= near <= 0.6):
        print("  [!] мало/много резонансных точек - можно сдвинуть f0 в макросе "
              "или диапазоны геометрии.")
    else:
        print("  OK: резонансная зона покрыта.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("case_id,len,height,thk,force\n")
        for i in range(a.n):
            f.write(f"{i+1:04d},{L[i]:.6f},{h[i]:.6f},{b[i]:.6f},{F[i]:.3f}\n")
    print(f"Сохранено: {a.out}")


if __name__ == "__main__":
    main()
