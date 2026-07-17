# -*- coding: utf-8 -*-
"""
ДОПОЛНИТЕЛЬНЫЙ эксперимент (раздел 4): расширенный физически корректный диапазон,
где зависимость резче за счёт малых height и thk. Основной эксперимент НЕ трогается -
всё пишется в data/ext/.

Диапазоны (проверено по балочной оценке: u/L <= ~2.4 %, sigma <= ~135 МПа < предел текучести):
  len    6.0 ... 14.0 м
  height 0.25 ... 1.4 м   (меньше -> резче)
  thk    0.02 ... 0.14 м  (меньше -> резче)
  force  500 ... 2000 Н
N = 120 точек.

Запуск:  python scripts/08_generate_doe_ext.py
Выход :  data/ext/cantilever_doe_inputs_ext.csv,  data/ext/cantilever_theory_check_ext.csv
"""
import os, csv
import numpy as np

N = 120
SEED = 20240609
BOUNDS = {"len": (6.0, 14.0), "height": (0.25, 1.4), "thk": (0.02, 0.14), "force": (500.0, 2000.0)}
E = 2.1e11

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data", "ext")); os.makedirs(DATA, exist_ok=True)


def lhs(n, d, seed):
    try:
        from scipy.stats import qmc
        return qmc.LatinHypercube(d=d, seed=seed).random(n)
    except Exception:
        rng = np.random.default_rng(seed)
        cut = np.linspace(0, 1, n + 1); u = rng.random((n, d))
        pts = cut[:n, None] + u / n
        for j in range(d):
            rng.shuffle(pts[:, j])
        return pts


def main():
    names = list(BOUNDS); s = lhs(N, len(names), SEED)
    X = np.empty_like(s)
    for j, nm in enumerate(names):
        lo, hi = BOUNDS[nm]; X[:, j] = lo + s[:, j] * (hi - lo)
    inp = os.path.join(DATA, "cantilever_doe_inputs_ext.csv")
    with open(inp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case_id"] + names)
        for i in range(N):
            w.writerow([f"{i+1:04d}"] + [f"{X[i,j]:.6f}" for j in range(len(names))])
    # контроль физкорректности по углам
    def beam(L, h, t, Fo):
        I = t*h**3/12.0; u = Fo*L**3/(3*E*I); return u, u/L*100, 6*Fo*L/(t*h**2)/1e6
    worst_u = max(beam(*c)[1] for c in [(14, 0.25, 0.02, 2000)])
    worst_s = max(beam(*c)[2] for c in [(14, 0.25, 0.02, 2000)])
    chk = os.path.join(DATA, "cantilever_theory_check_ext.csv")
    with open(chk, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["worst_u_over_L_percent", "worst_sigma_MPa", "yield_MPa", "physically_valid"])
        w.writerow([f"{worst_u:.2f}", f"{worst_s:.1f}", 250, worst_u < 10 and worst_s < 250])
    print(f"[ok] {inp} ({N} точек)")
    print(f"     физпроверка: макс u/L={worst_u:.2f}% (<10), макс sigma={worst_s:.0f} МПа (<250) -> корректно")


if __name__ == "__main__":
    main()
