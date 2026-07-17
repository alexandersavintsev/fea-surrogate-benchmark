"""Генерация DOE для упругопластической консоли + авто-проверка доли пластики.

Запуск:
    python fea/generate_plastic_doe.py                 # 200 точек, дефолтные диапазоны
    python fea/generate_plastic_doe.py --n 200 --fmax 6e5
    python fea/generate_plastic_doe.py --vary-sy       # ещё и предел текучести варьируем

Идея: текучесть наступает, когда балочное напряжение sigma = 6*F*L/(b*h^2)
превышает предел текучести sy. Скрипт считает sigma для всех точек и говорит,
какая доля уйдёт в пластику. Цель - 40-60% (часть точек упругие -> epl=0,
часть пластические -> epl>0; это и есть пороговый отклик для ML).

Выход: data/plastic_doe_inputs.csv  (колонки: case_id, len, height, thk, force, sy)
Дальше эти строки подставляются в apdl/plastic_cantilever.mac тем же батч-скриптом,
что и для упругой задачи.
"""
from __future__ import annotations
import argparse
import os
import numpy as np

# --- константы материала (в макросе те же) ---
SY_FIXED = 2.5e8     # предел текучести, Па (250 МПа)
SY_RANGE = (2.0e8, 3.0e8)


def lhs(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    cut = np.linspace(0.0, 1.0, n + 1)
    lo, hi = cut[:n][:, None], cut[1:][:, None]
    pts = lo + rng.uniform(size=(n, d)) * (hi - lo)
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lmin", type=float, default=8.0)
    ap.add_argument("--lmax", type=float, default=12.0)
    ap.add_argument("--hmin", type=float, default=0.8)
    ap.add_argument("--hmax", type=float, default=1.2)
    ap.add_argument("--bmin", type=float, default=0.08)
    ap.add_argument("--bmax", type=float, default=0.12)
    ap.add_argument("--fmin", type=float, default=1.5e5)
    ap.add_argument("--fmax", type=float, default=6.0e5)
    ap.add_argument("--vary-sy", action="store_true")
    ap.add_argument("--out", default="data/plastic_doe_inputs.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = 5 if args.vary_sy else 4
    U = lhs(args.n, d, rng)

    L = args.lmin + U[:, 0] * (args.lmax - args.lmin)
    h = args.hmin + U[:, 1] * (args.hmax - args.hmin)
    b = args.bmin + U[:, 2] * (args.bmax - args.bmin)
    F = args.fmin + U[:, 3] * (args.fmax - args.fmin)
    sy = SY_RANGE[0] + U[:, 4] * (SY_RANGE[1] - SY_RANGE[0]) if args.vary_sy \
        else np.full(args.n, SY_FIXED)

    # балочная оценка макс. напряжения (на стенке защемления)
    sigma = 6.0 * F * L / (b * h ** 2)
    ratio = sigma / sy                      # >1 => пластика
    frac = float(np.mean(ratio > 1.0))

    print(f"Точек: {args.n} | диапазон force: {args.fmin:.2e}..{args.fmax:.2e} Н")
    print(f"sigma/sy: min={ratio.min():.2f}  med={np.median(ratio):.2f}  max={ratio.max():.2f}")
    print(f"Доля точек в пластике (sigma>sy): {frac*100:.1f} %")
    if frac < 0.4:
        sug = args.fmax * 0.5 / max(frac, 0.05)
        print(f"  МАЛО пластики. Подними --fmax примерно до {sug:.2e} Н и перегенери.")
    elif frac > 0.6:
        sug = args.fmax * 0.5 / frac
        print(f"  МНОГО пластики. Опусти --fmax примерно до {sug:.2e} Н и перегенери.")
    else:
        print("  OK: баланс упругих/пластических точек хороший.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("case_id,len,height,thk,force,sy\n")
        for i in range(args.n):
            f.write(f"{i+1:04d},{L[i]:.6f},{h[i]:.6f},{b[i]:.6f},{F[i]:.3f},{sy[i]:.1f}\n")
    print(f"Сохранено: {args.out}")
    print("Дальше: прогон-пилот на ~20 строках через apdl/plastic_cantilever.mac, "
          "проверь сколько с epl_max>0, потом гони все.")


if __name__ == "__main__":
    main()
