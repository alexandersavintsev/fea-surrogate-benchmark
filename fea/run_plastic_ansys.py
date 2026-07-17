# -*- coding: utf-8 -*-
"""Серия ANSYS-расчётов упругопластической консоли по DOE-выборке.

Копия рабочей машинки из section4_cantilever/scripts/02_run_ansys_batch.py,
переделанная под пластический макрос. Для каждой точки подставляет
len/height/thk/force/sy в apdl/plastic_cantilever.mac, запускает ANSYS в batch
в отдельной папке runs_plastic/case_XXXX/ и собирает результаты в
data/plastic_results.csv.

Запуск:
    python fea/run_plastic_ansys.py --pilot 20     # сначала пилот на 20 точках
    python fea/run_plastic_ansys.py                # затем все 200

Путь к ANSYS берётся из переменной окружения ANSYS_EXE или из уже настроенного
bat твоей упругой задачи (section4_cantilever/apdl/run_cantilever_ansys.bat).
"""
import os, re, csv, subprocess, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))          # itmo_surrogate
WORK = os.path.abspath(os.path.join(REPO, ".."))          # C:\ANSYS_WORK_DIPLOM
MACRO = os.path.join(REPO, "apdl", "plastic_cantilever.mac")
RESULT = "plastic_result.txt"
# порядок значений в plastic_result.txt (см. *VWRITE в макросе)
KEYS = ["len", "height", "thk", "force", "sy", "epl_max", "uy_tip", "sig_max", "area"]
COLS = ["case_id"] + KEYS

BAT_CANDIDATES = [
    os.path.join(REPO, "fea", "run_plastic_ansys.bat"),
    os.path.join(WORK, "section4_cantilever", "apdl", "run_cantilever_ansys.bat"),
    os.path.join(WORK, "Сonsole_Optimization", "code", "run_cantilever_ansys.bat"),
]


def find_ansys():
    exe = os.environ.get("ANSYS_EXE")
    if exe and os.path.exists(exe.strip('"')):
        return exe.strip('"')
    for bat in BAT_CANDIDATES:
        try:
            for line in open(bat, encoding="utf-8", errors="ignore"):
                m = re.search(r'set\s+"?ANSYS_EXE\s*=\s*"?([^"\r\n]+)"?', line, re.I)
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
    txt = open(path).read().replace("D", "E").replace("d", "e")
    vals = [float(v) for v in txt.split()]
    return dict(zip(KEYS, vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=os.path.join(REPO, "data", "plastic_doe_inputs.csv"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "plastic_results.csv"))
    ap.add_argument("--runs", default=os.path.join(REPO, "runs_plastic"))
    ap.add_argument("--pilot", type=int, default=0, help="прогнать только первые N точек")
    a = ap.parse_args()

    if not os.path.exists(a.inputs):
        sys.exit(f"[stop] нет файла {a.inputs} (сначала: python fea/generate_plastic_doe.py --fmax 8e5)")
    exe = find_ansys()
    if not exe:
        sys.exit("[ERROR] ANSYS не найден. Задай путь: set ANSYS_EXE=\"...ANSYS241.exe\" "
                 "или впиши его в fea/run_plastic_ansys.bat")
    print(f"[ansys] {exe}")

    os.makedirs(a.runs, exist_ok=True)
    reader = list(csv.DictReader(open(a.inputs, encoding="utf-8-sig")))
    if a.pilot:
        reader = reader[:a.pilot]
        print(f"[pilot] только первые {len(reader)} точек")

    rows, n_plastic = [], 0
    for r in reader:
        cid = r["case_id"]
        wd = os.path.join(a.runs, f"case_{cid}")
        os.makedirs(wd, exist_ok=True)
        patch_macro({k: r[k] for k in ["len", "height", "thk", "force", "sy"]},
                    os.path.join(wd, "plastic_cantilever.mac"))
        subprocess.run([exe, "-b", "-np", "1", "-i", "plastic_cantilever.mac",
                        "-o", "output_plastic.txt"], cwd=wd, check=False)
        rp = os.path.join(wd, RESULT)
        if not os.path.exists(rp):
            print(f"[warn] case {cid}: нет {RESULT} (расчёт не сошёлся?)")
            continue
        try:
            res = parse_result(rp)
        except Exception as e:
            print(f"[warn] case {cid}: не разобрал результат ({e})")
            continue
        if res["epl_max"] > 0:
            n_plastic += 1
        rows.append([cid] + [res[k] for k in KEYS])
        print(f"  case {cid}: epl_max={res['epl_max']:.3e}  sig_max={res['sig_max']:.3e}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(rows)
    frac = (n_plastic / len(rows) * 100) if rows else 0
    print(f"\n[ok] {a.out}  ({len(rows)} строк, в пластике {n_plastic} = {frac:.0f} %)")
    if rows and not 20 <= frac <= 80:
        print("  [!] доля пластики далека от ~50%. Подвинь force в generate_plastic_doe.py и перегенери DOE.")


if __name__ == "__main__":
    main()
