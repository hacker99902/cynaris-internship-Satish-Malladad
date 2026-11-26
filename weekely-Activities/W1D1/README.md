# NumPy Fundamentals 

## Overview
This project demonstrates the fundamental concepts of NumPy required for Machine Learning.

It covers array creation, array dimensions, broadcasting, vectorized operations, matrix multiplication, and statistical calculations using a real-world Iris dataset.

## Learning Objectives
- Understand NumPy arrays (1D, 2D and 3D).
- Verify array shapes and dimensions.
- Perform broadcasting operations.
- Understand vectorized operations without Python loops.
- Perform matrix multiplication.
- Calculate statistical measures using NumPy.

## Project Structure

```text
numpy-fundamentals/
│
├── data/
│   └── iris.csv
│
├── numpy_demo.py
├── requirements.txt
└── README.md
```

## Technologies Used
- Python 3.11
- NumPy
- Pandas
- Iris Dataset

## Tasks Implemented

### Task 1: Array Creation
Created 1D, 2D and 3D NumPy arrays using `np.array()`.

### Task 2: Array Shapes
Used:
- `shape` to identify array dimensions.
- `ndim` to identify the number of dimensions.

### Task 3: Broadcasting and Vectorization
Implemented:
- Scalar broadcasting.
- Vector broadcasting.
- Element-wise multiplication.
- Vectorized square root operations.

### Task 4: Matrix Multiplication
Performed matrix multiplication using NumPy's `@` operator.

### Task 5: Statistical Analysis
Used the Iris dataset to calculate:
- Mean using `np.mean()`.
- Standard deviation using `np.std()`.
- Correlation matrix using `np.corrcoef()`.

Only the four numerical measurement features are used for statistical analysis.

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python numpy_demo.py
```

## Validation

The program includes assertions to verify:
- Array shapes.
- Broadcasting output dimensions.
- Matrix multiplication dimensions.
- Statistical output dimensions.
- Numerical validity of statistical results.

Successful execution displays:

```text
All 5 tasks completed successfully!
```

## Dataset
The project uses the Iris dataset, containing flower measurements such as sepal length, sepal width, petal length and petal width.

## Learning Outcome
Developed practical understanding of NumPy arrays, broadcasting, vectorization, matrix operations and statistical analysis for Machine Learning.
