from pathlib import Path
import numpy as np
import pandas as pd

WORKDIR = Path(r"C:\ANSYS_WORK_DIPLOM")
DOE_PATH = WORKDIR / "doe.npz"
CSV_PATH = WORKDIR / "kirsch_romai_doe_100.csv"

data = np.load(DOE_PATH, allow_pickle=True)

X = data["samples"]
y = data["fields"]

df = pd.DataFrame(
    data=np.column_stack([X, y]),
    columns=["a", "b", "lx", "ly", "smax", "area"]
)

df.to_csv(CSV_PATH, index=False)

print(f"Saved CSV: {CSV_PATH}")
print(df.head())
print()
print(df.describe())
