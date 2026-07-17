# -*- coding: utf-8 -*-
"""
SHAP-анализ влияния признаков для раздела 4 (методически согласован с разделом 3,
где интерпретация выполнялась через SHAP средствами RomAI).

GPR остаётся лучшей моделью по точности. Здесь RandomForest используется ТОЛЬКО как
вспомогательная интерпретируемая модель: для деревьев SHAP считается точно и быстро
(shap.TreeExplainer), тогда как для GPR потребовался бы медленный KernelExplainer.

Признаки : len, height, thk, force
Выходы   : uy_tip, uy_max, sig_max  (физические; area исключён - это геометрия len*height)

Зависимости: numpy, pandas, matplotlib, scikit-learn; shap - желательно (pip install shap).
Если shap не установлен, используется permutation_importance из sklearn (с пометкой).

Запуск:  python scripts/07_make_shap.py
         python scripts/07_make_shap.py --dataset data/ext/cantilever_dataset_ext.csv --outdir data/ext --figdir figures/ext
Выходы:
  data/cantilever_shap_importance.csv
  figures/fig_4_10_shap_importance.png
"""
import os, sys, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FEATURES = ["len", "height", "thk", "force"]
TARGETS = ["uy_tip", "uy_max", "sig_max"]


def importance_for_target(X, y, feat):
    """Возвращает (mean_abs_shap по признакам, метка_метода)."""
    rf = RandomForestRegressor(n_estimators=500, random_state=0).fit(X, y)
    try:
        import shap
        sv = shap.TreeExplainer(rf).shap_values(X)
        return np.abs(np.asarray(sv)).mean(axis=0), "SHAP (TreeExplainer, RandomForest)"
    except Exception:
        try:
            from sklearn.inspection import permutation_importance
            r = permutation_importance(rf, X, y, n_repeats=20, random_state=0)
            return np.clip(r.importances_mean, 0, None), "permutation_importance (shap не установлен)"
        except Exception:
            return rf.feature_importances_, "feature_importances_ (запасной вариант)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=os.path.join(ROOT, "data", "cantilever_dataset.csv"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--figdir", default=os.path.join(ROOT, "figures"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True); os.makedirs(a.figdir, exist_ok=True)

    if not os.path.exists(a.dataset):
        sys.exit(f"[stop] Нет {a.dataset}. Сначала выполните ANSYS-расчёты (02) и проверьте датасет.")
    df = pd.read_csv(a.dataset).dropna(subset=FEATURES + TARGETS).reset_index(drop=True)
    if len(df) < 20:
        sys.exit("[stop] Слишком мало строк в датасете для устойчивого SHAP-анализа.")
    X = df[FEATURES].values
    print(f"[ok] датасет: {len(df)} строк")

    rows, rel_by_target, method = [], {}, None
    for tg in TARGETS:
        imp, method = importance_for_target(X, df[tg].values, FEATURES)
        rel = imp / (imp.sum() + 1e-12) * 100.0   # относительная важность, %
        rel_by_target[tg] = rel
        for f, mv, rv in zip(FEATURES, imp, rel):
            rows.append(dict(target=tg, feature=f, mean_abs_shap=mv, rel_percent=rv))
        print(f"  {tg:8s}: " + ", ".join(f"{f}={r:.1f}%" for f, r in zip(FEATURES, rel)))

    # усреднённая по трём выходам относительная важность (масштаб-инвариантна)
    overall = np.mean([rel_by_target[tg] for tg in TARGETS], axis=0)
    for f, ov in zip(FEATURES, overall):
        rows.append(dict(target="среднее по uy_tip,uy_max,sig_max", feature=f, mean_abs_shap=np.nan, rel_percent=ov))
    out_csv = os.path.join(a.outdir, "cantilever_shap_importance.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[ok] {out_csv}  (метод: {method})")

    order = np.argsort(overall)
    plt.figure(figsize=(6.5, 4))
    plt.barh([FEATURES[i] for i in order], [overall[i] for i in order], color="#4C72B0")
    for i, idx in enumerate(order):
        plt.text(overall[idx], i, f" {overall[idx]:.1f}%", va="center", fontsize=10)
    plt.xlabel("Средняя относительная важность признака, %")
    plt.title("Важность признаков по " + ("SHAP" if method.startswith("SHAP") else method.split()[0]) +
              " (RandomForest; uy_tip, uy_max, sig_max)")
    plt.tight_layout()
    out_png = os.path.join(a.figdir, "fig_4_10_shap_importance.png")
    plt.savefig(out_png, dpi=150); plt.close()
    print(f"[ok] {out_png}")


if __name__ == "__main__":
    main()
