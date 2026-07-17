"""Streamlit-демо: обучи суррогат на данных и сравни модели + быстрый прогноз.

Запуск: streamlit run app/streamlit_app.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st

from src.synthetic import make_dataset, DATASETS
from src.models import get_models
from src.benchmark import cv_eval

st.set_page_config(page_title="FEA Surrogate Benchmark", layout="wide")
st.title("Суррогаты для МКЭ: где GPR, а где бустинг")

st.sidebar.header("Данные")
src = st.sidebar.radio("Источник", ["Синтетика", "CSV (ANSYS)"])

if src == "Синтетика":
    ds = st.sidebar.selectbox("Датасет", DATASETS)
    n = st.sidebar.slider("Число точек", 100, 1000, 400, 50)
    X, y = make_dataset(ds, n=n, d=4, noise=0.02, seed=0)
    st.caption(f"Датасет: {ds}, точек: {n}, признаков: 4")
else:
    up = st.sidebar.file_uploader("CSV", type="csv")
    if up is None:
        st.info("Загрузите CSV с данными ANSYS.")
        st.stop()
    df = pd.read_csv(up)
    target = st.sidebar.selectbox("Целевой столбец", df.columns)
    y = df[target].to_numpy(float)
    X = df.drop(columns=[target]).to_numpy(float)

if st.button("Сравнить модели (k-fold CV)"):
    rows = []
    prog = st.progress(0.0)
    models = get_models()
    for i, (name, model) in enumerate(models.items()):
        r2_mean, r2_std, mape = cv_eval(X, y, model, k=5, seeds=(0, 1))
        rows.append(dict(model=name, R2=round(r2_mean, 4),
                         R2_std=round(r2_std, 4), MAPE=round(mape, 2)))
        prog.progress((i + 1) / len(models))
    res = pd.DataFrame(rows).sort_values("R2", ascending=False)
    st.subheader("Результаты сравнения")
    st.dataframe(res, use_container_width=True)
    st.bar_chart(res.set_index("model")["R2"])
    best = res.iloc[0]
    st.success(f"Лучшая модель: {best['model']} (R² = {best['R2']})")
