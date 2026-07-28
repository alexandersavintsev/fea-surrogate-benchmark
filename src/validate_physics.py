"""Воспроизводимые физические проверки расчётных выборок.

Запуск:
    python -m src.validate_physics

Проверки не заменяют верификацию КЭ-модели, но ловят ошибки масштаба,
единиц измерения, граничных условий и неверный режим расчёта.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def check_kirsch() -> dict:
    """Сравнить ANSYS с формулой Инглиса для бесконечной пластины."""
    frame = pd.read_csv(
        DATA / "kirsch/kirsch_doe_100.csv",
        encoding="utf-8-sig",
    )
    sigma_inf = 100.0
    sigma_theory = sigma_inf * (1.0 + 2.0 * frame["a"] / frame["b"])
    relative_error = np.abs(frame["smax"] - sigma_theory) / sigma_theory * 100.0
    maximum = float(relative_error.max())
    return {
        "Проверка": "Кирш / Инглис",
        "Результат": f"медиана {relative_error.median():.2f}%, максимум {maximum:.2f}%",
        "Критерий": "максимум <= 10%",
        "Пройдена": maximum <= 10.0,
    }


def check_elastic_cantilever() -> dict:
    """Сравнить прогиб ANSYS с формулой Эйлера-Бернулли."""
    frame = pd.read_csv(DATA / "cantilever/cantilever_theory_check.csv")
    error = float(frame.loc[0, "rel_error_percent"])
    return {
        "Проверка": "Упругая консоль",
        "Результат": f"ошибка прогиба {error:.2f}%",
        "Критерий": "ошибка <= 5%",
        "Пройдена": error <= 5.0,
    }


def check_plasticity() -> dict:
    """Проверить момент появления пластики балочной оценкой напряжения."""
    frame = pd.read_csv(DATA / "plastic_results.csv")
    sigma_beam = (
        6.0
        * frame["force"]
        * frame["len"]
        / (frame["thk"] * frame["height"] ** 2)
    )
    predicted_plastic = sigma_beam > frame["sy"]
    ansys_plastic = frame["epl_max"] > 1e-12
    agreement = float(np.mean(predicted_plastic == ansys_plastic) * 100.0)
    return {
        "Проверка": "Порог текучести",
        "Результат": (
            f"совпадение {agreement:.1f}%, "
            f"пластика {int(ansys_plastic.sum())}/200"
        ),
        "Критерий": "совпадение >= 95%",
        "Пройдена": agreement >= 95.0,
    }


def check_resonance() -> dict:
    """Проверить положение пика оценкой первой частоты консоли."""
    frame = pd.read_csv(DATA / "harmonic_results.csv")
    young = 2.1e11
    density = 7850.0
    beta_1 = 1.875104068711961
    inertia = frame["thk"] * frame["height"] ** 3 / 12.0
    area = frame["thk"] * frame["height"]
    frequency_1 = (
        beta_1**2
        / (2.0 * np.pi * frame["len"] ** 2)
        * np.sqrt(young * inertia / (density * area))
    )

    peak_index = frame["uy_amp"].idxmax()
    peak_frequency = float(frequency_1.loc[peak_index])
    excitation = float(frame.loc[peak_index, "f0"])
    frequency_error = abs(peak_frequency - excitation) / excitation * 100.0
    amplitude_ratio = float(frame["uy_amp"].max() / frame["uy_amp"].min())
    passed = frequency_error <= 5.0 and amplitude_ratio >= 10.0
    return {
        "Проверка": "Гармонический резонанс",
        "Результат": (
            f"пик при f1={peak_frequency:.3f} Гц, "
            f"ошибка {frequency_error:.2f}%, разброс {amplitude_ratio:.1f}x"
        ),
        "Критерий": "ошибка <= 5%, разброс >= 10x",
        "Пройдена": passed,
    }


def main() -> None:
    checks = pd.DataFrame(
        [
            check_kirsch(),
            check_elastic_cantilever(),
            check_plasticity(),
            check_resonance(),
        ]
    )
    print(checks.to_string(index=False))
    if not bool(checks["Пройдена"].all()):
        raise SystemExit("Одна или несколько физических проверок не пройдены.")


if __name__ == "__main__":
    main()
