from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


WORKDIR = Path(r"C:\ANSYS_WORK_DIPLOM")
CSV_PATH = WORKDIR / "kirsch_romai_doe_100.csv"
OUT_DIR = WORKDIR / "PYTHON_MODELS_100"
OUT_DIR.mkdir(exist_ok=True)


def calc_metrics(y_true, y_pred, model_name):
    rows = []

    target_names = ["smax", "area"]

    for i, target in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mae = mean_absolute_error(yt, yp)
        rmse = mean_squared_error(yt, yp) ** 0.5
        r2 = r2_score(yt, yp)
        mape = np.mean(np.abs((yt - yp) / yt)) * 100

        rows.append({
            "model": model_name,
            "target": target,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE_percent": mape
        })

    # Общая метрика сразу по двум выходам
    rows.append({
        "model": model_name,
        "target": "mean_over_outputs",
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred, multioutput="uniform_average"),
        "MAPE_percent": np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    })

    return rows


def main():
    df = pd.read_csv(CSV_PATH)

    X = df[["a", "b", "lx", "ly"]].values
    y = df[["smax", "area"]].values

    # Важно: датасет маленький, поэтому test_size умеренный
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    models = {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),

        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                random_state=42
            )
        ),

        "RandomForest": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                random_state=42
            )
        )
    }

    all_metrics = []

    for name, model in models.items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_metrics = calc_metrics(y_train, y_train_pred, name + "_train")
        test_metrics = calc_metrics(y_test, y_test_pred, name + "_test")

        all_metrics.extend(train_metrics)
        all_metrics.extend(test_metrics)

        pred_df = pd.DataFrame({
            "a": X_test[:, 0],
            "b": X_test[:, 1],
            "lx": X_test[:, 2],
            "ly": X_test[:, 3],
            "smax_true": y_test[:, 0],
            "smax_pred": y_test_pred[:, 0],
            "area_true": y_test[:, 1],
            "area_pred": y_test_pred[:, 1],
        })

        pred_df.to_csv(OUT_DIR / f"{name}_test_predictions.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)

    print()
    print("Metrics saved to:", OUT_DIR / "metrics.csv")
    print(metrics_df)


if __name__ == "__main__":
    main()
