import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request


# Task 1: Creating NumPy arrays

print("\n========== TASK 1: ARRAY CREATION ==========")

# Creating a 1D array
arr1 = np.array([10, 20, 30, 40, 50])

# Creating a 2D array
arr2 = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Creating a 3D array
arr3 = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print("1D Array:")
print(arr1)

print("\n2D Array:")
print(arr2)

print("\n3D Array:")
print(arr3)


# Task 2: Checking array shapes and dimensions

print("\n========== TASK 2: ARRAY SHAPES ==========")

print("Shape of 1D array:", arr1.shape)
print("Shape of 2D array:", arr2.shape)
print("Shape of 3D array:", arr3.shape)

print("Dimensions of 1D array:", arr1.ndim)
print("Dimensions of 2D array:", arr2.ndim)
print("Dimensions of 3D array:", arr3.ndim)


# Task 3: Broadcasting and vectorized operations

print("\n========== TASK 3: BROADCASTING ==========")

# Adding 10 to every element
result1 = arr1 + 10

print("Adding 10 to the array:")
print(result1)

# Adding a vector to a matrix using broadcasting
matrix = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])

vector = np.array([1, 2, 3])

result2 = matrix + vector

print("\nMatrix + Vector:")
print(result2)

# Multiplying every element by 2
result3 = matrix * 2

print("\nMatrix multiplied by 2:")
print(result3)

# Finding square root of each element
result4 = np.sqrt(matrix)

print("\nSquare root of matrix:")
print(result4)


# Task 4: Matrix multiplication

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

# Using @ operator for matrix multiplication
result5 = A @ B

print("Matrix A:")
print(A)

print("\nMatrix B:")
print(B)

print("\nResult of A @ B:")
print(result5)


# Task 5: Calculating statistics using Iris dataset

print("\n========== TASK 5: CSV STATISTICS ==========")

# Creating a data folder if it doesn't exist
data_folder = Path(__file__).parent / "data"
data_folder.mkdir(exist_ok=True)

csv_file = data_folder / "iris.csv"

# Downloading the dataset only if it is not already available
if not csv_file.exists():

    url = (
        "https://archive.ics.uci.edu/"
        "ml/machine-learning-databases/iris/iris.data"
    )

    column_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ]

    urllib.request.urlretrieve(url, csv_file)

    iris = pd.read_csv(
        csv_file,
        header=None,
        names=column_names
    )

    iris.to_csv(csv_file, index=False)

# Reading the dataset
df = pd.read_csv(csv_file)

print("First five rows:")
print(df.head())

# Taking only the four flower measurement columns
features = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

data = df[features].to_numpy()

print("\nDataset shape:", data.shape)

# Finding the mean of each column
mean_values = np.mean(data, axis=0)

print("\nMean:")
print(mean_values)

# Finding standard deviation
std_values = np.std(data, axis=0)

print("\nStandard Deviation:")
print(std_values)

# Finding correlation between the features
correlation = np.corrcoef(data, rowvar=False)

print("\nCorrelation Matrix:")
print(np.round(correlation, 3))


# Checking whether all operations worked correctly

print("\n========== FINAL VALIDATION ==========")

assert arr1.shape == (5,)
assert arr2.shape == (3, 3)
assert arr3.shape == (2, 2, 2)

assert result2.shape == (3, 3)
assert result5.shape == (2, 2)

assert data.shape == (150, 4)
assert mean_values.shape == (4,)
assert std_values.shape == (4,)
assert correlation.shape == (4, 4)

assert np.all(np.isfinite(mean_values))
assert np.all(np.isfinite(std_values))
assert np.all(np.isfinite(correlation))

print("All 5 tasks completed successfully!")
