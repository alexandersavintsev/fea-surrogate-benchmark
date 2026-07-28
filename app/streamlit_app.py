"""Streamlit-демо бенчмарка суррогатных моделей.

Запуск:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import cv_eval
from src.models import get_models
from src.synthetic import DATASETS, make_dataset


REAL_DATASETS = {
    "Кирш, 100 точек": {
        "name": "kirsch_100",
        "path": "data/kirsch/kirsch_doe_100.csv",
        "features": ["a", "b", "lx", "ly"],
        "target": "smax",
    },
    "Консоль, 80 точек": {
        "name": "cantilever_80",
        "path": "data/cantilever/cantilever_dataset.csv",
        "features": ["len", "height", "thk", "force"],
        "target": "sig_max",
    },
    "Консоль, расширенный диапазон, 120 точек": {
        "name": "cantilever_120",
        "path": "data/cantilever/ext/cantilever_dataset_ext.csv",
        "features": ["len", "height", "thk", "force"],
        "target": "sig_max",
    },
    "Консоль, 240 точек": {
        "name": "cantilever_240",
        "path": "data/cantilever/ext_h020_N240/cantilever_dataset_ext_h020_N240.csv",
        "features": ["len", "height", "thk", "force"],
        "target": "sig_max",
    },
    "Пластичность, 200 точек": {
        "name": "plastic_cantilever",
        "path": "data/plastic_results.csv",
        "features": ["len", "height", "thk", "force"],
        "target": "epl_max",
    },
    "Резонанс, 200 точек": {
        "name": "resonance_cantilever",
        "path": "data/harmonic_results.csv",
        "features": ["len", "height", "thk", "force"],
        "target": "uy_amp",
    },
}

DATASET_LABELS = {
    spec["name"]: label for label, spec in REAL_DATASETS.items()
}


@st.cache_data
def load_real_dataset(label: str) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    spec = REAL_DATASETS[label]
    df = pd.read_csv(ROOT / spec["path"], encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("?") for c in df.columns]
    X = df[spec["features"]].to_numpy(float)
    y = df[spec["target"]].to_numpy(float)
    return X, y, spec["features"], spec["target"]


@st.cache_data
def load_precomputed() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results/crossover_tuned.csv")


@st.cache_data(show_spinner=False)
def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_names: tuple[str, ...],
    k: int,
    seeds: tuple[int, ...],
) -> tuple[pd.DataFrame, float]:
    started = time.perf_counter()
    models = get_models()
    rows = []
    for name in model_names:
        r2_mean, r2_std, mape = cv_eval(X, y, models[name], k=k, seeds=seeds)
        rows.append(
            {
                "Модель": name,
                "R²": round(r2_mean, 4),
                "Стандартное отклонение R²": round(r2_std, 4),
                "MAPE, %": round(mape, 2),
            }
        )
    result = pd.DataFrame(rows).sort_values("R²", ascending=False)
    return result, time.perf_counter() - started


st.set_page_config(page_title="FEA Surrogate Benchmark", layout="wide")
st.title("Суррогаты для МКЭ: где GPR, а где бустинг")
st.caption(
    "1000 расчётов - это сумма пяти физических постановок. "
    "Для каждой постановки модель обучается отдельно: у задач разные входы и отклики."
)

c1, c2, c3 = st.columns(3)
c1.metric("Расчёты ANSYS", "1000")
c2.metric("Регрессионные модели", "8")
c3.metric("Финальная проверка", "5-fold × 3 random seed")

mode = st.sidebar.radio(
    "Режим",
    ["Готовые результаты полного эксперимента", "Быстрый пересчёт"],
)

if mode == "Готовые результаты полного эксперимента":
    st.subheader("Полный эксперимент")
    st.write(
        "Здесь показаны сохранённые результаты честного сравнения: "
        "5-fold cross-validation на трёх значениях random seed, "
        "ансамбли деревьев настроены через Optuna."
    )

    full = load_precomputed()
    available_names = [
        name for name in DATASET_LABELS if name in set(full["dataset"])
    ]
    selected_name = st.selectbox(
        "Физическая постановка",
        available_names,
        format_func=lambda name: DATASET_LABELS[name],
        index=max(0, len(available_names) - 1),
    )
    result = (
        full.loc[full["dataset"] == selected_name, ["model", "R2", "MAPE", "tuned"]]
        .rename(
            columns={
                "model": "Модель",
                "R2": "R²",
                "MAPE": "MAPE, %",
                "tuned": "Настроена Optuna",
            }
        )
        .sort_values("R²", ascending=False)
    )

    best = result.iloc[0]
    st.success(f"Лучшая модель: {best['Модель']} (R² = {best['R²']:.4f})")
    left, right = st.columns([1.15, 1])
    with left:
        st.dataframe(result, use_container_width=True, hide_index=True)
    with right:
        st.bar_chart(result.set_index("Модель")["R²"])

else:
    st.subheader("Быстрый пересчёт")
    st.info(
        "Этот режим нужен для живого демо: 5-fold cross-validation на одном "
        "random seed. Полный эксперимент использует три seed, а пересчёт "
        "восьми моделей на 1000 точках занимает несколько минут из-за GPR и MLP."
    )

    source = st.sidebar.radio(
        "Данные",
        ["Расчёты ANSYS из проекта", "Синтетика", "Загрузить CSV"],
    )

    if source == "Расчёты ANSYS из проекта":
        label = st.sidebar.selectbox("Постановка", list(REAL_DATASETS))
        X, y, feature_names, target_name = load_real_dataset(label)
        st.caption(
            f"{label}: {len(y)} точек, признаки {', '.join(feature_names)}, "
            f"целевой отклик {target_name}."
        )
    elif source == "Синтетика":
        dataset = st.sidebar.selectbox("Функция", DATASETS)
        n_points = st.sidebar.slider("Число точек", 100, 1000, 400, 50)
        X, y = make_dataset(dataset, n=n_points, d=4, noise=0.02, seed=0)
        st.caption(f"{dataset}: {n_points} точек, 4 признака.")
    else:
        uploaded = st.sidebar.file_uploader("CSV", type="csv")
        if uploaded is None:
            st.info("Загрузите CSV с числовыми данными.")
            st.stop()
        frame = pd.read_csv(uploaded)
        target_name = st.sidebar.selectbox("Целевой столбец", frame.columns)
        feature_names = [c for c in frame.columns if c != target_name]
        y = frame[target_name].to_numpy(float)
        X = frame[feature_names].to_numpy(float)
        st.caption(
            f"Загружено {len(y)} строк, признаки {', '.join(feature_names)}, "
            f"цель {target_name}."
        )

    available_models = list(get_models())
    default_models = [name for name in ("GPR", "CatBoost") if name in available_models]
    selected_models = st.multiselect(
        "Модели для быстрого пересчёта",
        available_models,
        default=default_models,
    )

    if st.button("Запустить 5-fold cross-validation"):
        if not selected_models:
            st.warning("Выберите хотя бы одну модель.")
            st.stop()
        with st.spinner("Обучаю модели и считаю метрики..."):
            result, elapsed = evaluate(
                X,
                y,
                tuple(selected_models),
                k=5,
                seeds=(0,),
            )
        st.success(f"Готово за {elapsed:.1f} с")
        left, right = st.columns([1.15, 1])
        with left:
            st.dataframe(result, use_container_width=True, hide_index=True)
        with right:
            st.bar_chart(result.set_index("Модель")["R²"])
