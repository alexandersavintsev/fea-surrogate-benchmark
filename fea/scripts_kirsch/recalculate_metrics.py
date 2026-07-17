from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


ROOT = Path(r"C:\ANSYS_WORK_DIPLOM")

DATASETS = {
    "GPR_30": ROOT / "GPR_30" / "doe.npz",
    "GPR_100": ROOT / "GPR_100" / "doe.npz",
    "NN_30": ROOT / "NN_30" / "doe.npz",
    # "CURRENT_ROOT": ROOT / "doe.npz",  # только для проверки, не для диплома
}

OUT_DIR = ROOT / "FINAL_METRICS"
OUT_DIR.mkdir(exist_ok=True)

INPUT_COLUMNS = ["a", "b", "lx", "ly"]
OUTPUT_COLUMNS = ["smax", "area"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def calc_metrics(y_true, y_pred, dataset_name, model_name, subset_name):
    rows = []

    for i, target in enumerate(OUTPUT_COLUMNS):
        true = y_true[:, i]
        pred = y_pred[:, i]

        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        mape = np.mean(np.abs((true - pred) / true)) * 100

        rows.append({
            "dataset": dataset_name,
            "model": model_name,
            "subset": subset_name,
            "target": target,
            "R2": r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE_percent": mape,
        })

    rows.append({
        "dataset": dataset_name,
        "model": model_name,
        "subset": subset_name,
        "target": "mean_over_outputs",
        "R2": np.mean([r["R2"] for r in rows]),
        "MAE": np.mean([r["MAE"] for r in rows]),
        "MSE": np.mean([r["MSE"] for r in rows]),
        "RMSE": np.mean([r["RMSE"] for r in rows]),
        "MAPE_percent": np.mean([r["MAPE_percent"] for r in rows]),
    })

    return rows


def export_predictions(out_dir, dataset_name, model_name, X_test, y_test, y_pred):
    df = pd.DataFrame(X_test, columns=INPUT_COLUMNS)

    for i, target in enumerate(OUTPUT_COLUMNS):
        df[f"{target}_true"] = y_test[:, i]
        df[f"{target}_pred"] = y_pred[:, i]
        df[f"{target}_abs_error"] = np.abs(y_test[:, i] - y_pred[:, i])
        df[f"{target}_rel_error_percent"] = (
            np.abs((y_test[:, i] - y_pred[:, i]) / y_test[:, i]) * 100
        )

    df.to_csv(
        out_dir / f"{dataset_name}_{model_name}_test_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )


def process_dataset(dataset_name, doe_path):
    if not doe_path.exists():
        print(f"Skip missing: {doe_path}")
        return []

    data = np.load(doe_path, allow_pickle=True)
    X = data["samples"]
    y = data["fields"]

    print(f"\nLoaded {dataset_name}:")
    print(f"X shape = {X.shape}")
    print(f"y shape = {y.shape}")

    df = pd.DataFrame(
        np.hstack([X, y]),
        columns=INPUT_COLUMNS + OUTPUT_COLUMNS
    )
    df.to_csv(OUT_DIR / f"{dataset_name}_doe_export.csv", index=False, encoding="utf-8-sig")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    models = {
        "Ridge": Pipeline([
            ("x_scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "GPR_sklearn": Pipeline([
            ("x_scaler", StandardScaler()),
            ("model", MultiOutputRegressor(
                GaussianProcessRegressor(
                    kernel=Matern(nu=1.5),
                    alpha=1e-10,
                    normalize_y=True,
                    random_state=RANDOM_STATE
                )
            ))
        ]),
        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE
            )
        ),
        "RandomForest": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE
            )
        ),
    }

    all_rows = []

    for model_name, model in models.items():
        print(f"Training {dataset_name} / {model_name}...")

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        all_rows.extend(calc_metrics(y_train, y_train_pred, dataset_name, model_name, "train"))
        all_rows.extend(calc_metrics(y_test, y_test_pred, dataset_name, model_name, "test"))

        export_predictions(OUT_DIR, dataset_name, model_name, X_test, y_test, y_test_pred)

    return all_rows


def main():
    all_rows = []

    for dataset_name, doe_path in DATASETS.items():
        all_rows.extend(process_dataset(dataset_name, doe_path))

    metrics = pd.DataFrame(all_rows)
    metrics_path = OUT_DIR / "metrics_recalculated.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("\nSaved:")
    print(metrics_path)
    print(metrics)


if __name__ == "__main__":
    main()
