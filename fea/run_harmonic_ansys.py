# -*- coding: utf-8 -*-
"""Серия гармонических ANSYS-расчётов (резонансная задача) по DOE-выборке.

Та же машинка, что и для пластики, но с гармоническим макросом. Для каждой точки
подставляет len/height/thk/force в apdl/harmonic_cantilever.mac, запускает ANSYS
в batch и собирает амплитуды колебаний торца в data/harmonic_results.csv.

Запуск:
    python fea/run_harmonic_ansys.py --pilot 20     # пилот
    python fea/run_harmonic_ansys.py                # все 200
"""
import os, re, csv, subprocess, sys, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.path.abspath(os.path.join(REPO, ".."))
MACRO = os.path.join(REPO, "apdl", "harmonic_cantilever.mac")
RESULT = "harmonic_result.txt"
KEYS = ["len", "height", "thk", "force", "f0", "uy_amp", "area"]
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
    return dict(zip(KEYS, [float(v) for v in txt.split()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=os.path.join(REPO, "data", "harmonic_doe_inputs.csv"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "harmonic_results.csv"))
    ap.add_argument("--runs", default=os.path.join(REPO, "runs_harmonic"))
    ap.add_argument("--pilot", type=int, default=0)
    a = ap.parse_args()

    if not os.path.exists(a.inputs):
        sys.exit(f"[stop] нет {a.inputs} (сначала: python fea/generate_harmonic_doe.py)")
    exe = find_ansys()
    if not exe:
        sys.exit("[ERROR] ANSYS не найден (задай ANSYS_EXE или впиши в run_plastic_ansys.bat)")
    print(f"[ansys] {exe}")

    os.makedirs(a.runs, exist_ok=True)
    reader = list(csv.DictReader(open(a.inputs, encoding="utf-8-sig")))
    if a.pilot:
        reader = reader[:a.pilot]
        print(f"[pilot] только первые {len(reader)} точек")

    rows, amps = [], []
    for r in reader:
        cid = r["case_id"]
        wd = os.path.join(a.runs, f"case_{cid}")
        os.makedirs(wd, exist_ok=True)
        patch_macro({k: r[k] for k in ["len", "height", "thk", "force"]},
                    os.path.join(wd, "harmonic_cantilever.mac"))
        subprocess.run([exe, "-b", "-np", "1", "-i", "harmonic_cantilever.mac",
                        "-o", "output_harmonic.txt"], cwd=wd, check=False)
        rp = os.path.join(wd, RESULT)
        if not os.path.exists(rp):
            print(f"[warn] case {cid}: нет {RESULT}")
            continue
        try:
            res = parse_result(rp)
        except Exception as e:
            print(f"[warn] case {cid}: не разобрал ({e})")
            continue
        amps.append(res["uy_amp"])
        rows.append([cid] + [res[k] for k in KEYS])
        print(f"  case {cid}: uy_amp={res['uy_amp']:.4e}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(rows)
    if amps:
        import statistics
        a_ = [abs(x) for x in amps]
        print(f"\n[ok] {a.out}  ({len(rows)} строк)")
        print(f"  амплитуда |uy|: min={min(a_):.3e}  медиана={statistics.median(a_):.3e}  max={max(a_):.3e}")
        print(f"  разброс max/min = {max(a_)/min(a_):.1f}x  (большой разброс = резонанс проявился)")


if __name__ == "__main__":
    main()
