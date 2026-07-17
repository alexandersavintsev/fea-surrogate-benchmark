# -*- coding: utf-8 -*-
"""
Серия ANSYS-расчётов по DOE-выборке для консольной пластины (PLANE182).
Для каждой точки подставляет len/height/thk/force в макрос, запускает ANSYS
в batch-режиме в отдельной папке runs/case_XXXX/ и собирает результаты в
../data/cantilever_dataset.csv.

Запуск:  python 02_run_ansys_batch.py
Требуется ANSYS Mechanical APDL. Путь задаётся в apdl/run_cantilever_ansys.bat
(переменная ANSYS_EXE) или переменной окружения ANSYS_EXE.
"""
import os, re, csv, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
APDL = os.path.join(ROOT, "apdl")
RUNS = os.path.join(ROOT, "runs")
MACRO = os.path.join(APDL, "cantilever_plane182.mac")
BAT = os.path.join(APDL, "run_cantilever_ansys.bat")
RESULT = "cantilever_result.txt"
COLS = ["case_id", "len", "height", "thk", "force",
        "uy_tip", "uy_max", "u_sum_max", "sig_max", "area", "volume"]


def find_ansys():
    exe = os.environ.get("ANSYS_EXE")
    if exe and os.path.exists(exe.strip('"')):
        return exe.strip('"')
    # попытка вытащить путь из bat
    try:
        for line in open(BAT, encoding="utf-8", errors="ignore"):
            m = re.search(r'set\s+ANSYS_EXE\s*=\s*"?([^"\r\n]+)"?', line, re.I)
            if m and os.path.exists(m.group(1)):
                return m.group(1)
    except OSError:
        pass
    return None


def patch_macro(values, dst):
    text = open(MACRO, encoding="utf-8").read()
    for nm, val in values.items():
        text = re.sub(rf"(?m)^{nm}\s*=.*$", f"{nm}    = {val}", text, count=1)
    open(dst, "w", encoding="utf-8").write(text)


def parse_result(path):
    vals = open(path).read().split()
    vals = [float(v) for v in vals]
    # порядок: len,height,thk,force,uy_tip,uy_max,u_sum_max,sig_max,area,volume
    keys = ["len", "height", "thk", "force", "uy_tip", "uy_max",
            "u_sum_max", "sig_max", "area", "volume"]
    return dict(zip(keys, vals))


def write_instructions(reason):
    os.makedirs(os.path.join(ROOT, "text"), exist_ok=True)
    p = os.path.join(ROOT, "text", "run_instructions.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("ЗАПУСК РАСЧЁТОВ РАЗДЕЛА 4 (консольная пластина, PLANE182)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Причина остановки: {reason}\n\n")
        f.write("1. Откройте apdl/run_cantilever_ansys.bat и впишите путь к ANSYS:\n")
        f.write('   set ANSYS_EXE=\"C:\\Program Files\\ANSYS Inc\\vXXX\\ansys\\bin\\winx64\\ANSYSXXX.exe\"\n')
        f.write("   (или задайте переменную окружения ANSYS_EXE)\n\n")
        f.write("2. Сгенерируйте DOE (если ещё нет):\n   python scripts/01_generate_doe.py\n\n")
        f.write("3. Запустите серию расчётов:\n   python scripts/02_run_ansys_batch.py\n\n")
        f.write("   Появятся папки runs/case_0001 ... и файл data/cantilever_dataset.csv\n\n")
        f.write("4. Обучите модели и постройте графики:\n")
        f.write("   python scripts/04_train_models.py\n   python scripts/05_make_plots.py\n")
        f.write("   python scripts/06_make_section4_tables.py\n\n")
        f.write("После этого обновите text/section_4_full.txt значениями из CSV по маркерам.\n")
    print(f"[instructions] {p}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=os.path.join(DATA, "cantilever_doe_inputs.csv"))
    ap.add_argument("--dataset", default=os.path.join(DATA, "cantilever_dataset.csv"))
    ap.add_argument("--runs", default=RUNS)
    a = ap.parse_args()
    if not os.path.exists(a.inputs):
        sys.exit(f"[stop] Нет {a.inputs}. Сначала: python 01_generate_doe.py")
    exe = find_ansys()
    if not exe:
        print("[ERROR] ANSYS не найден. Расчёты не выполнены, результаты не выдумываются.")
        write_instructions("ANSYS_EXE не найден или путь неверен")
        sys.exit(1)

    os.makedirs(a.runs, exist_ok=True)
    rows = []
    with open(a.inputs) as f:
        reader = list(csv.DictReader(f))
    for r in reader:
        cid = r["case_id"]
        wd = os.path.join(a.runs, f"case_{cid}")
        os.makedirs(wd, exist_ok=True)
        patch_macro({k: r[k] for k in ["len", "height", "thk", "force"]},
                    os.path.join(wd, "cantilever_plane182.mac"))
        subprocess.run([exe, "-b", "-np", "1",
                        "-i", "cantilever_plane182.mac",
                        "-o", "output_cantilever.txt"], cwd=wd, check=False)
        rp = os.path.join(wd, RESULT)
        if not os.path.exists(rp):
            print(f"[warn] case {cid}: нет {RESULT} (расчёт не сошёлся?)")
            continue
        res = parse_result(rp)
        rows.append([cid] + [res[k] for k in COLS[1:]])
        print(f"  case {cid}: uy_tip={res['uy_tip']:.4e} sig_max={res['sig_max']:.4e}")

    out = a.dataset
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(rows)
    print(f"[ok] {out}  ({len(rows)} строк)")


if __name__ == "__main__":
    main()
