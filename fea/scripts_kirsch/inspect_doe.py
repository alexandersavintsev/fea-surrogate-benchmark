from pathlib import Path
import numpy as np

WORKDIR = Path(r"C:\ANSYS_WORK_DIPLOM")
DOE_PATH = WORKDIR / "doe.npz"

def main():
    if not DOE_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {DOE_PATH}")

    data = np.load(DOE_PATH, allow_pickle=True)

    print("Файл:", DOE_PATH)
    print("Ключи внутри doe.npz:")
    print(data.files)
    print()

    for key in data.files:
        arr = data[key]
        print("=" * 80)
        print(f"KEY: {key}")
        print(f"type: {type(arr)}")
        print(f"shape: {getattr(arr, 'shape', None)}")
        print(f"dtype: {getattr(arr, 'dtype', None)}")
        print("Первые значения:")

        try:
            if arr.ndim == 0:
                print(arr)
            elif arr.ndim == 1:
                print(arr[:10])
            elif arr.ndim == 2:
                print(arr[:10, :])
            else:
                print(arr)
        except Exception as e:
            print(f"Не удалось вывести массив: {e}")

    data.close()

if __name__ == "__main__":
    main()