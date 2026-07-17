# -*- coding: utf-8 -*-
"""
Проверка найденной оптимизатором точки полным расчётом в ANSYS.
Берёт лучшие точки из optimization_results.csv, подставляет (a,b,lx,ly) в
макрос Кирша plate_with_hole_ROMAI.txt, запускает ANSYS в batch-режиме,
читает obj.txt и сравнивает прогноз суррогата с расчётом ANSYS.

Требует наличия ANSYS и рабочего run_ansys.bat (как для задачи Кирша).
Без ANSYS скрипт только сообщит, что запуск невозможен (результат не выдумывается).

Выход: optimum_ansys_check.csv  (task, a, b, lx, ly, smax_pred, smax_ansys,
        area_pred, area_ansys, err_smax_%, err_area_%)
"""
import os, re, csv, shutil, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
KIRSCH_MACRO = os.path.abspath(os.path.join(ROOT, "..", "plate_with_hole_ROMAI.txt"))
KIRSCH_BAT   = os.path.abspath(os.path.join(ROOT, "..", "run_ansys.bat"))
PARAMS = ["a", "b", "lx", "ly"]


def patch(macro_src, dst, values):
    with open(macro_src) as f:
        text = f.read()
    for nm, val in values.items():
        text = re.sub(rf"(?m)^{nm}=.*$", f"{nm}={val}", text, count=1)
    with open(dst, "w") as f:
        f.write(text)


def parse_obj(path):
    res = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                try: res[p[0]] = float(p[1])
                except ValueError: pass
    return res


def main():
    res = pd.read_csv(os.path.join(DATA, "optimization_results.csv"))
    if not (os.path.exists(KIRSCH_MACRO) and os.path.exists(KIRSCH_BAT)):
        print("[stop] Не найден макрос/бат Кирша или ANSYS недоступен.")
        print("       Запустите этот скрипт на машине с ANSYS. Результат проверки")
        print("       не выдумывается; ожидаемый выход: optimum_ansys_check.csv")
        return
    wd = os.path.join(HERE, "_opt_check"); os.makedirs(wd, exist_ok=True)
    rows = []
    for _, r in res.iterrows():
        vals = {k: float(r[k]) for k in PARAMS}
        patch(KIRSCH_MACRO, os.path.join(wd, "plate_with_hole_ROMAI.txt"), vals)
        shutil.copy(KIRSCH_BAT, os.path.join(wd, "run_ansys.bat"))
        subprocess.run(["cmd", "/c", "run_ansys.bat"], cwd=wd, check=False)
        obj = parse_obj(os.path.join(wd, "obj.txt"))
        sa, aa = obj.get("smax"), obj.get("area")
        sp, ap = float(r["smax_pred"]), float(r["area_pred"])
        rows.append(dict(task=r["task"], **vals,
            smax_pred=sp, smax_ansys=sa, area_pred=ap, area_ansys=aa,
            err_smax_pct=None if sa in (None,0) else round(abs(sp-sa)/sa*100, 3),
            err_area_pct=None if aa in (None,0) else round(abs(ap-aa)/aa*100, 3)))
    out = os.path.join(DATA, "optimum_ansys_check.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[ok] {out}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
