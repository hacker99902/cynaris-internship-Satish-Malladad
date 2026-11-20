import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

IRIS_URL = (
    "https://archive.ics.uci.edu/"
    "ml/machine-learning-databases/iris/iris.data"
)

COLUMN_NAMES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

def load_iris(csv_path):
    """Load the Iris dataset into a DataFrame."""
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading the Iris dataset...")
        urllib.request.urlretrieve(IRIS_URL, csv_path)

    df = pd.read_csv(csv_path)

    # Remove the unnecessary ID column if present.
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # Standardize column names.
    df.columns = COLUMN_NAMES

    return df   


def iris_statistics(df):
    """Calculate mean, standard deviation and correlation."""
    data = df[FEATURES].to_numpy()

    return {
        "mean": np.mean(data, axis=0),
        "std": np.std(data, axis=0),
        "corr": np.corrcoef(data, rowvar=False),
    }


def main():
    # Task 1: Create NumPy arrays
    print("\n========== TASK 1: ARRAY CREATION ==========")

    arr1 = np.array([10, 20, 30, 40, 50])

    arr2 = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])

    arr3 = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ])

    print("1D Array:\n", arr1)
    print("\n2D Array:\n", arr2)
    print("\n3D Array:\n", arr3)

    # Task 2: Check shapes and dimensions
    print("\n========== TASK 2: ARRAY SHAPES ==========")

    for name, array in [("1D", arr1), ("2D", arr2), ("3D", arr3)]:
        print(f"{name} shape: {array.shape}")
        print(f"{name} dimensions: {array.ndim}")

    # Task 3: Broadcasting and vectorized operations
    print("\n========== TASK 3: BROADCASTING ==========")

    print("Array + 10:\n", arr1 + 10)

    matrix = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
    ])

    vector = np.array([1, 2, 3])

    print("\nMatrix + Vector:\n", matrix + vector)
    print("\nMatrix * 2:\n", matrix * 2)
    print("\nSquare root:\n", np.sqrt(matrix))

    # Task 4: Matrix multiplication
    print("\n========== TASK 4: MATRIX MULTIPLICATION ==========")

    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])

    B = np.array([
        [7, 8],
        [9, 10],
        [11, 12],
    ])

    result = A @ B

    print("Matrix A:\n", A)
    print("\nMatrix B:\n", B)
    print("\nA @ B:\n", result)

    # Task 5: Statistics using the Iris dataset
    print("\n========== TASK 5: IRIS STATISTICS ==========")

    # Works when running as a Python file or from a notebook.
    if "__file__" in globals():
        base_path = Path(__file__).resolve().parent
    else:
        base_path = Path.cwd()

    csv_path = base_path / "data" / "iris.csv"
    df = load_iris(csv_path)

    print("First five rows:")
    print(df.head())

    print("\nDataset shape:", df.shape)

    stats = iris_statistics(df)

    print("\nMean:")
    print(np.round(stats["mean"], 3))

    print("\nStandard deviation:")
    print(np.round(stats["std"], 3))

    print("\nCorrelation matrix:")
    print(np.round(stats["corr"], 3))

    # Final validation
    print("\n========== FINAL VALIDATION ==========")

    assert arr1.shape == (5,)
    assert arr2.shape == (3, 3)
    assert arr3.shape == (2, 2, 2)
    assert (matrix + vector).shape == (3, 3)
    assert result.shape == (2, 2)
    assert df.shape == (150, 5)
    assert stats["mean"].shape == (4,)
    assert stats["std"].shape == (4,)
    assert stats["corr"].shape == (4, 4)
    assert np.all(np.isfinite(stats["mean"]))
    assert np.all(np.isfinite(stats["std"]))
    assert np.all(np.isfinite(stats["corr"]))

    print("All 5 tasks completed successfully!")


if __name__ == "__main__":
    main()
