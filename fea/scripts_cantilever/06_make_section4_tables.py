# -*- coding: utf-8 -*-
"""Сборка text/section_4_tables.md из CSV. Статические таблицы строятся всегда;
табл. 4.4-4.6 заполняются при наличии данных, иначе - маркеры."""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data"); TEXT = os.path.join(ROOT, "text")
os.makedirs(TEXT, exist_ok=True)


def md_table(df, floatfmt=None):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |",
           "|" + "|".join(["---"]*len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if floatfmt and isinstance(v, float):
                cells.append(floatfmt(c, v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    L = []
    L.append("# Таблицы раздела 4\n")

    L.append("## Таблица 4.1 - Диапазоны изменения входных параметров\n")
    L.append(md_table(pd.DataFrame([
        ["Длина", "len", 6.0, 14.0, "м"],
        ["Высота", "height", 0.6, 1.4, "м"],
        ["Толщина", "thk", 0.06, 0.14, "м"],
        ["Сила", "force", 500, 2000, "Н"]],
        columns=["Параметр", "Обозначение", "Минимум", "Максимум", "Единица"])))

    L.append("\n## Таблица 4.2 - Выходные величины конечно-элементной модели\n")
    L.append(md_table(pd.DataFrame([
        ["Перемещение контрольной точки", "uy_tip", "Перемещение узла правого торца у середины высоты", "м"],
        ["Макс. вертикальное перемещение", "uy_max", "Максимум перемещения по Y по модулю", "м"],
        ["Макс. суммарное перемещение", "u_sum_max", "Максимум полного перемещения", "м"],
        ["Макс. эквивалентное напряжение", "sig_max", "Максимум по Мизесу", "Па"],
        ["Площадь", "area", "Площадь расчётной области len*height", "м^2"]],
        columns=["Величина", "Обозначение", "Описание", "Единица"])))

    L.append("\n## Таблица 4.3 - Проверка базовой точки по балочной формуле\n")
    tc = os.path.join(DATA, "cantilever_theory_check.csv")
    if os.path.exists(tc):
        d = pd.read_csv(tc)
        d2 = d[["len", "height", "thk", "force", "u_theory", "uy_tip_ansys", "rel_error_percent"]].copy()
        d2 = d2.rename(columns={"u_theory": "u_theory, м", "uy_tip_ansys": "uy_tip ANSYS, м",
                                "rel_error_percent": "Относ. отклонение, %"})
        d2 = d2.fillna("[заполнить после запуска ANSYS]")
        L.append(md_table(d2))
    else:
        L.append("[таблица появится после 01_generate_doe.py]")

    L.append("\n## Таблица 4.4 - Пример строк расчётной выборки\n")
    ds = os.path.join(DATA, "cantilever_dataset.csv")
    if os.path.exists(ds) and pd.read_csv(ds).shape[0] > 0:
        d = pd.read_csv(ds).head(8)[["case_id", "len", "height", "thk", "force", "uy_tip", "uy_max", "sig_max", "area"]]
        L.append(md_table(d, floatfmt=lambda c, v: f"{v:.4g}"))
    else:
        L.append("[заполнить после запуска ANSYS: первые 8 строк data/cantilever_dataset.csv "
                 "со столбцами case_id, len, height, thk, force, uy_tip, uy_max, sig_max, area]")

    L.append("\n## Таблица 4.5 - Метрики качества моделей по целевым величинам\n")
    mt = os.path.join(DATA, "cantilever_metrics_by_target.csv")
    if os.path.exists(mt):
        L.append(md_table(pd.read_csv(mt), floatfmt=lambda c, v: f"{v:.4g}"))
    else:
        L.append("[заполнить из data/cantilever_metrics_by_target.csv после 04_train_models.py]")

    L.append("\n## Таблица 4.6 - Средние метрики качества моделей\n")
    mm = os.path.join(DATA, "cantilever_metrics_mean.csv")
    if os.path.exists(mm):
        L.append(md_table(pd.read_csv(mm), floatfmt=lambda c, v: f"{v:.4g}"))
    else:
        L.append("[заполнить из data/cantilever_metrics_mean.csv после 04_train_models.py]")

    L.append("\n## Таблица 4.7 - Сравнение моделей по инженерной интерпретации\n")
    L.append(md_table(pd.DataFrame([
        ["Ridge", "Линейная база", "Прост, быстр, не переобучается", "Не ловит нелинейность len^3 и взаимодействия"],
        ["GPR", "Гладкий интерполятор", "Точен на малых выборках, даёт неопределённость", "Дорог при больших N, чувствителен к ядру"],
        ["Gradient Boosting", "Ансамбль деревьев", "Хорош на нелинейностях и взаимодействиях", "Риск переобучения при плохих гиперпараметрах"],
        ["Random Forest", "Ансамбль деревьев", "Устойчив к шуму", "Слабая экстраполяция за диапазон"],
        ["MLPRegressor", "Нейросеть", "Гибкая аппроксимация", "Требует больше данных, нестабильна на малых N"]],
        columns=["Модель", "Тип модели", "Сильные стороны", "Ограничения"])))

    open(os.path.join(TEXT, "section_4_tables.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"[ok] {os.path.join(TEXT, 'section_4_tables.md')}")


if __name__ == "__main__":
    main()
