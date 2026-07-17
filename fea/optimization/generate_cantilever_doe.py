# -*- coding: utf-8 -*-
"""
Генерация DOE-выборки методом латинского гиперкуба для задачи об изгибе
консольно защемлённой пластины и (при наличии ANSYS) автоматический сбор
датасета через макрос cantilever_plate_romai.mac + run_cantilever_ansys.bat.

Запуск:
    python generate_cantilever_doe.py            # только входная DOE-таблица
    python generate_cantilever_doe.py --run-ansys  # + запуск ANSYS по точкам

Выходы:
    cantilever_doe_inputs.csv   входы DOE (lx, ly, p)
    cantilever_dataset.csv      датасет (lx, ly, p, umax, smax, area) или шаблон
"""
import os, csv, argparse, shutil, subprocess, re
import numpy as np

# ----- диапазоны входных параметров (плоское напряжённое состояние) -----
BOUNDS = {
    "lx": (100.0, 300.0),   # длина (вылет) консоли
    "ly": (10.0, 40.0),     # высота сечения (управляет жёсткостью ~ ly^3)
    "p":  (1.0, 10.0),      # интенсивность давления на верхней грани
}
N_SNAPSHOTS = 100
SEED = 20240601

HERE = os.path.dirname(os.path.abspath(__file__))
MACRO = os.path.join(HERE, "cantilever_plate_romai.mac")
BAT   = os.path.join(HERE, "run_cantilever_ansys.bat")


def lhs(n, d, seed=SEED):
    """Латинский гиперкуб в [0,1]^d без внешних зависимостей."""
    rng = np.random.default_rng(seed)
    cut = np.linspace(0, 1, n + 1)
    u = rng.random((n, d))
    pts = cut[:n, None] + u * (1.0 / n)
    for j in range(d):
        rng.shuffle(pts[:, j])
    return pts


def scale(sample01):
    names = list(BOUNDS.keys())
    out = np.empty_like(sample01)
    for j, nm in enumerate(names):
        lo, hi = BOUNDS[nm]
        out[:, j] = lo + sample01[:, j] * (hi - lo)
    return names, out


def write_inputs(names, X, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for row in X:
            w.writerow([f"{v:.6f}" for v in row])


def patch_macro(names, values, src, dst):
    """Подставляет значения параметров в копию макроса (строки lx=, ly=, p=)."""
    with open(src, "r") as f:
        text = f.read()
    for nm, val in zip(names, values):
        text = re.sub(rf"(?m)^{nm}=.*$", f"{nm}={val:.6f}", text, count=1)
    with open(dst, "w") as f:
        f.write(text)


def parse_obj(path):
    res = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                try:
                    res[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ansys", action="store_true")
    ap.add_argument("--n", type=int, default=N_SNAPSHOTS)
    args = ap.parse_args()

    names, X = scale(lhs(args.n, len(BOUNDS)))
    inp_path = os.path.join(HERE, "..", "data", "cantilever_doe_inputs.csv")
    inp_path = os.path.abspath(inp_path)
    write_inputs(names, X, inp_path)
    print(f"[ok] входная DOE-таблица: {inp_path} ({args.n} точек)")

    ds_path = os.path.abspath(os.path.join(HERE, "..", "data", "cantilever_dataset.csv"))
    cols = names + ["umax", "smax", "area"]

    if not args.run_ansys or not os.path.exists(BAT):
        # ANSYS не запускается: пишем шаблон датасета с заголовком
        with open(ds_path, "w", newline="") as f:
            csv.writer(f).writerow(cols)
        print(f"[template] ANSYS не запускался. Создан шаблон: {ds_path}")
        print("           Запустите с --run-ansys на машине с ANSYS, чтобы заполнить.")
        return

    # ---- цикл расчётов ANSYS ----
    workdir = os.path.join(HERE, "_cant_run")
    os.makedirs(workdir, exist_ok=True)
    rows = []
    for i, vals in enumerate(X):
        patch_macro(names, vals, MACRO, os.path.join(workdir, "cantilever_plate_romai.mac"))
        shutil.copy(BAT, os.path.join(workdir, "run_cantilever_ansys.bat"))
        subprocess.run(["cmd", "/c", "run_cantilever_ansys.bat"], cwd=workdir, check=False)
        obj = parse_obj(os.path.join(workdir, "obj.txt"))
        rows.append(list(vals) + [obj.get("umax"), obj.get("smax"), obj.get("area")])
        print(f"  [{i+1}/{len(X)}] umax={obj.get('umax')} smax={obj.get('smax')} area={obj.get('area')}")
    with open(ds_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)
    print(f"[ok] датасет: {ds_path}")


if __name__ == "__main__":
    main()
