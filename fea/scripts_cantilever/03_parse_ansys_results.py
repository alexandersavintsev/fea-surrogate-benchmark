# -*- coding: utf-8 -*-
"""
Сбор результатов ANSYS из папок runs/case_XXXX/cantilever_result.txt в один
датасет ../data/cantilever_dataset.csv. Используйте, если расчёты запускались
вручную (без 02_run_ansys_batch.py).
"""
import os, csv, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RUNS = os.path.join(ROOT, "runs")
DATA = os.path.join(ROOT, "data")
COLS = ["case_id", "len", "height", "thk", "force",
        "uy_tip", "uy_max", "u_sum_max", "sig_max", "area", "volume"]


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(RUNS, "case_*"))):
        cid = os.path.basename(d).replace("case_", "")
        rp = os.path.join(d, "cantilever_result.txt")
        if not os.path.exists(rp):
            continue
        v = [float(x) for x in open(rp).read().split()]
        if len(v) < 10:
            print(f"[warn] {cid}: неполная строка результата"); continue
        rows.append([cid] + v[:10])
    if not rows:
        print("[stop] не найдено ни одного cantilever_result.txt в runs/. "
              "Сначала выполните расчёты ANSYS (02_run_ansys_batch.py).")
        return
    out = os.path.join(DATA, "cantilever_dataset.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(rows)
    print(f"[ok] {out}  ({len(rows)} строк)")


if __name__ == "__main__":
    main()
