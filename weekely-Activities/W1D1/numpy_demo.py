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

FEATURES = COLUMN_NAMES[:4]


def load_iris(csv_path: Path) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        print("Downloading Iris dataset from UCI...")

        try:
            with urllib.request.urlopen(IRIS_URL, timeout=30) as response:
                content = response.read()

            csv_path.write_bytes(content)

        except (OSError, urllib.error.URLError) as error:
            raise RuntimeError(
                f"Failed to download Iris dataset: {error}"
            ) from error

    # Read the first line to identify the dataset format.
    with csv_path.open("r", encoding="utf-8-sig") as file:
        first_line = file.readline().strip().lower()

    if first_line.startswith("id,") or "sepallengthcm" in first_line:
        # Kaggle CSV format.
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        # Select columns explicitly rather than relying on their order.
        expected_columns = [
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm",
            "Species",
        ]

        df = df[expected_columns]
        df.columns = COLUMN_NAMES

    else:
        # Original UCI format has no header.
        df = pd.read_csv(
            csv_path,
            header=None,
            names=COLUMN_NAMES,
            encoding="utf-8-sig",
        )

    if df.shape[1] != 5:
        raise ValueError("Iris dataset must contain exactly 5 columns.")

    if df.empty:
        raise ValueError("Iris dataset is empty.")

    return df



# --------------------------------------------------
# 3. Iris Statistics
# --------------------------------------------------

def iris_statistics(
    df: pd.DataFrame
) -> dict[str, np.ndarray]:
    """
    Calculate mean, standard deviation and correlation.

    Args:
        df: Iris dataset DataFrame.

    Returns:
        Dictionary containing statistical results.
    """

    data = df[FEATURES].to_numpy(dtype=float)

    return {
        "mean": np.mean(data, axis=0),
        "std": np.std(data, axis=0),
        "corr": np.corrcoef(data, rowvar=False),
    }


# --------------------------------------------------
# 4. Main Function
# --------------------------------------------------

def main() -> None:

    np.set_printoptions(
        precision=3,
        suppress=True
    )

    # ==============================================
    # TASK 1: ARRAY CREATION
    # ==============================================

    print("\n========== TASK 1: ARRAY CREATION ==========")

    arr1 = np.array([10, 20, 30, 40, 50])

    arr2 = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    arr3 = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])

    print("1D Array:\n", arr1)
    print("\n2D Array:\n", arr2)
    print("\n3D Array:\n", arr3)

    # ==============================================
    # TASK 2: SHAPES AND DIMENSIONS
    # ==============================================

    print("\n========== TASK 2: ARRAY SHAPES ==========")

    for name, array in [
        ("1D", arr1),
        ("2D", arr2),
        ("3D", arr3)
    ]:
        print(f"{name} Shape: {array.shape}")
        print(f"{name} Dimensions: {array.ndim}")

    # ==============================================
    # TASK 3: BROADCASTING
    # ==============================================

    print("\n========== TASK 3: BROADCASTING ==========")

    print("Array + 10:\n", arr1 + 10)

    matrix = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ])

    vector = np.array([1, 2, 3])

    print("\nMatrix + Vector:\n", matrix + vector)

    print("\nMatrix * 2:\n", matrix * 2)

    print("\nSquare Root:\n", np.sqrt(matrix))

    # ==============================================
    # TASK 4: MATRIX MULTIPLICATION
    # ==============================================

    print("\n========== TASK 4: MATRIX MULTIPLICATION ==========")

    A = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    B = np.array([
        [7, 8],
        [9, 10],
        [11, 12]
    ])

    result = A @ B

    print("Matrix A:\n", A)
    print("\nMatrix B:\n", B)
    print("\nA @ B:\n", result)

    # ==============================================
    # TASK 5: IRIS DATASET STATISTICS
    # ==============================================

    print("\n========== TASK 5: IRIS STATISTICS ==========")

    if "__file__" in globals():
        base_path = Path(__file__).resolve().parent
    else:
        base_path = Path.cwd()

    csv_path = base_path / "data" / "iris.csv"

    df = load_iris(csv_path)

    print("\nFirst Five Rows:")
    print(df.head())

    print("\nDataset Shape:", df.shape)

    stats = iris_statistics(df)

    print("\nMean:")
    print(stats["mean"])

    print("\nStandard Deviation:")
    print(stats["std"])

    print("\nCorrelation Matrix:")
    print(stats["corr"])

    # ==============================================
    # FINAL VALIDATION
    # ==============================================

    print("\n========== FINAL VALIDATION ==========")

    assert arr1.shape == (5,)
    assert arr2.shape == (3, 3)
    assert arr3.shape == (2, 2, 2)

    assert (matrix + vector).shape == (3, 3)
    assert result.shape == (2, 2)

    assert df.shape == (150, 5)
    assert list(df.columns) == COLUMN_NAMES

    assert stats["mean"].shape == (4,)
    assert stats["std"].shape == (4,)
    assert stats["corr"].shape == (4, 4)

    assert np.all(np.isfinite(stats["mean"]))
    assert np.all(np.isfinite(stats["std"]))
    assert np.all(np.isfinite(stats["corr"]))

    print("All 5 tasks completed successfully!")


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    main()
