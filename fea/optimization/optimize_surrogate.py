# -*- coding: utf-8 -*-
"""
Параметрическая оптимизация на основе ML-суррогата (раздел 5).

По умолчанию работает с задачей Кирша по реальному 100-точечному датасету
(kirsch_romai_doe_100.csv): входы (a, b, lx, ly), выходы (smax, area).
Суррогат GPR обучается на этих данных и используется в двух задачах:

  Задача A: minimize smax  при ограничении area <= AREA_MAX
  Задача B: minimize area  при ограничении smax <= SMAX_ALLOW

Предпочтительно: scikit-learn (GPR) + scipy.optimize.differential_evolution.
Если их нет, используется встроенный GPR на numpy и случайный поиск
(результат воспроизводим: фиксированный SEED).

Выходы:
  optimization_results.csv           лучшие точки по обеим задачам
  optimization_history.csv           кандидаты случайного/эволюционного поиска
  optimization_objective.png         сходимость / распределение целевой функции
  optimization_candidate_check.txt   сводка с прогнозом суррогата в оптимуме
"""
import os, csv
import numpy as np
import pandas as pd

SEED = 7
AREA_MAX   = 13000.0   # ограничение площади для задачи A
SMAX_ALLOW = 220.0     # допускаемое напряжение для задачи B

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
FIG  = os.path.join(ROOT, "figures")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIG, exist_ok=True)

INPUTS  = ["a", "b", "lx", "ly"]
OUTPUTS = ["smax", "area"]
BOUNDS  = [(8.0, 16.0), (15.0, 25.0), (100.0, 140.0), (100.0, 140.0)]

# поиск датасета Кирша
CANDIDATES = [
    os.path.join(ROOT, "..", "kirsch_romai_doe_100.csv"),
    os.path.join(ROOT, "..", "GPR_100_doe_export.csv"),
    os.path.join(DATA, "kirsch_romai_doe_100.csv"),
]


def load_dataset():
    for p in CANDIDATES:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = [c.strip().lstrip("﻿") for c in df.columns]
            return df[INPUTS].values.astype(float), df[OUTPUTS].values.astype(float), os.path.abspath(p)
    raise FileNotFoundError(
        "Не найден датасет Кирша (kirsch_romai_doe_100.csv / GPR_100_doe_export.csv). "
        "Положите его рядом со скриптом или в ../data.")


# ----------------------- встроенный GPR на numpy -----------------------
class NumpyGPR:
    """Анизотропный RBF-GPR. Гиперпараметры подбираются максимизацией
    логарифма маргинального правдоподобия (случайный поиск + покоординатное
    уточнение в логарифмическом масштабе). Один экземпляр на один выход."""
    def __init__(self, seed=SEED):
        self.rng = np.random.default_rng(seed)

    def _nll(self, theta, X, y):
        d = X.shape[1]
        ls = np.exp(theta[:d]); sf2 = np.exp(theta[d]); sn2 = np.exp(theta[d+1])
        diff = X[:, None, :] - X[None, :, :]
        K = sf2 * np.exp(-0.5 * np.sum((diff/ls)**2, axis=2)) + sn2*np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e25
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        nll = 0.5*y@alpha + np.sum(np.log(np.diag(L))) + 0.5*len(X)*np.log(2*np.pi)
        return float(nll)

    def fit(self, X, y):
        self.Xm, self.Xs = X.mean(0), X.std(0)+1e-12
        self.ym, self.ys = y.mean(), y.std()+1e-12
        Xn = (X-self.Xm)/self.Xs; yn = (y-self.ym)/self.ys
        d = X.shape[1]
        best, best_nll = None, np.inf
        for _ in range(600):
            theta = np.concatenate([self.rng.uniform(-1.5, 2.0, d),
                                    [self.rng.uniform(-1, 1)],
                                    [self.rng.uniform(-7, -2)]])
            v = self._nll(theta, Xn, yn)
            if v < best_nll: best_nll, best = v, theta
        # покоординатное уточнение
        for _ in range(4):
            for k in range(len(best)):
                for step in (0.4, -0.4, 0.15, -0.15):
                    cand = best.copy(); cand[k] += step
                    v = self._nll(cand, Xn, yn)
                    if v < best_nll: best_nll, best = v, cand
        self.theta = best
        ls = np.exp(best[:d]); sf2 = np.exp(best[d]); sn2 = np.exp(best[d+1])
        diff = Xn[:, None, :]-Xn[None, :, :]
        K = sf2*np.exp(-0.5*np.sum((diff/ls)**2, axis=2)) + sn2*np.eye(len(Xn))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, yn))
        self.Xn, self.ls, self.sf2 = Xn, ls, sf2
        return self

    def predict(self, Xq):
        Xq = np.atleast_2d(Xq); Xn = (Xq-self.Xm)/self.Xs
        out = np.empty(len(Xn))
        for i in range(0, len(Xn), 4000):
            chunk = Xn[i:i+4000]
            diff = chunk[:, None, :]-self.Xn[None, :, :]
            Ks = self.sf2*np.exp(-0.5*np.sum((diff/self.ls)**2, axis=2))
            out[i:i+4000] = Ks@self.alpha
        return out*self.ys + self.ym


class Surrogate:
    """Двувыходный суррогат (smax, area). Пытается использовать sklearn,
    иначе встроенный NumpyGPR."""
    def __init__(self):
        self.backend = None
    def fit(self, X, Y):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            self.models = []
            for j in range(Y.shape[1]):
                k = ConstantKernel(1.0)*RBF([1.0]*X.shape[1]) + WhiteKernel(1e-3)
                m = Pipeline([("sc", StandardScaler()),
                              ("g", GaussianProcessRegressor(kernel=k, normalize_y=True,
                                    n_restarts_optimizer=6, random_state=0))])
                m.fit(X, Y[:, j]); self.models.append(m)
            self.backend = "sklearn"
        except Exception:
            self.models = [NumpyGPR(seed=SEED+j).fit(X, Y[:, j]) for j in range(Y.shape[1])]
            self.backend = "numpy"
        return self
    def predict(self, Xq):
        return np.column_stack([m.predict(Xq) for m in self.models])


def r2(y, yh):
    y = np.asarray(y); yh = np.asarray(yh)
    return 1 - np.sum((y-yh)**2)/np.sum((y-y.mean())**2)


def optimize(objective_idx, constr_idx, constr_max, surr, sense_min=True,
             penalty=1e6, n_random=300000):
    """Случайный поиск по боксу с штрафом за нарушение ограничения.
    Возвращает (best_x, best_pred, history_df)."""
    rng = np.random.default_rng(SEED)
    lo = np.array([b[0] for b in BOUNDS]); hi = np.array([b[1] for b in BOUNDS])
    X = lo + rng.random((n_random, len(BOUNDS)))*(hi-lo)
    P = surr.predict(X)
    obj = P[:, objective_idx].copy()
    viol = np.maximum(0.0, P[:, constr_idx]-constr_max)
    score = obj + penalty*viol
    order = np.argsort(score)
    best = order[0]
    hist = pd.DataFrame(np.column_stack([X[order[:2000]], P[order[:2000]], score[order[:2000]]]),
                        columns=INPUTS+OUTPUTS+["score"])
    # попытка уточнить через scipy DE, если доступно
    try:
        from scipy.optimize import differential_evolution
        def f(x):
            p = surr.predict(x.reshape(1, -1))[0]
            return p[objective_idx] + penalty*max(0.0, p[constr_idx]-constr_max)
        res = differential_evolution(f, BOUNDS, seed=SEED, maxiter=60,
                                     popsize=20, tol=1e-7, polish=True)
        if f(res.x) < score[best]:
            xb = res.x
        else:
            xb = X[best]
    except Exception:
        xb = X[best]
    pb = surr.predict(xb.reshape(1, -1))[0]
    return xb, pb, hist


def main():
    X, Y, src = load_dataset()
    print(f"[ok] датасет Кирша: {src}  ({len(X)} точек)")

    # валидация суррогата на holdout
    rng = np.random.default_rng(0); idx = rng.permutation(len(X))
    ntr = int(0.8*len(X)); tr, te = idx[:ntr], idx[ntr:]
    s_val = Surrogate().fit(X[tr], Y[tr]); Pv = s_val.predict(X[te])
    print(f"[surrogate backend = {s_val.backend}]")
    for j, t in enumerate(OUTPUTS):
        print(f"  holdout R2[{t}] = {r2(Y[te, j], Pv[:, j]):.5f}")

    # финальный суррогат на всех точках
    surr = Surrogate().fit(X, Y)

    results, hists = [], []
    # Задача A: min smax, area <= AREA_MAX
    xa, pa, ha = optimize(0, 1, AREA_MAX, surr)
    ha["task"] = "A_min_smax"; hists.append(ha)
    results.append(dict(task="A_min_smax", constraint=f"area<={AREA_MAX:.0f}",
        **{k: float(round(v, 4)) for k, v in zip(INPUTS, xa)},
        smax_pred=float(round(pa[0], 4)), area_pred=float(round(pa[1], 4))))
    # Задача B: min area, smax <= SMAX_ALLOW
    xb, pb, hb = optimize(1, 0, SMAX_ALLOW, surr)
    hb["task"] = "B_min_area"; hists.append(hb)
    results.append(dict(task="B_min_area", constraint=f"smax<={SMAX_ALLOW:.0f}",
        **{k: float(round(v, 4)) for k, v in zip(INPUTS, xb)},
        smax_pred=float(round(pb[0], 4)), area_pred=float(round(pb[1], 4))))

    pd.DataFrame(results).to_csv(os.path.join(DATA, "optimization_results.csv"), index=False)
    pd.concat(hists, ignore_index=True).to_csv(os.path.join(DATA, "optimization_history.csv"), index=False)

    # график целевой функции (сортированные лучшие кандидаты)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(np.sort(ha["smax"].values)); ax[0].axhline(pa[0], color="r", ls="--",
        label=f"оптимум smax={pa[0]:.1f}")
    ax[0].set_title(f"Задача A: min smax, area<= {AREA_MAX:.0f}")
    ax[0].set_xlabel("кандидаты (сорт.)"); ax[0].set_ylabel("smax"); ax[0].legend(fontsize=8)
    ax[1].plot(np.sort(hb["area"].values)); ax[1].axhline(pb[1], color="r", ls="--",
        label=f"оптимум area={pb[1]:.0f}")
    ax[1].set_title(f"Задача B: min area, smax<= {SMAX_ALLOW:.0f}")
    ax[1].set_xlabel("кандидаты (сорт.)"); ax[1].set_ylabel("area"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "optimization_objective.png"), dpi=150); plt.close()

    with open(os.path.join(DATA, "optimization_candidate_check.txt"), "w") as f:
        f.write(f"surrogate backend: {s_val.backend}\n")
        f.write(f"dataset: {src}\n\n")
        for rr in results:
            f.write(str(rr) + "\n")
    print("[ok] результаты оптимизации сохранены в data/ и figures/")
    for rr in results:
        print("  ", rr)


if __name__ == "__main__":
    main()
