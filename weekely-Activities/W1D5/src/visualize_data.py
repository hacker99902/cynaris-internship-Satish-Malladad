import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Locate the W1D5 project directory
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Dataset location
DATA_PATH = PROJECT_DIR / "data" / "states.csv"

# Output directory
os.makedirs("result", exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

# Identify numerical and categorical columns
numeric_columns = df.select_dtypes(include="number").columns.tolist()
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

sns.set_theme(style="whitegrid")

# 1. Numerical distributions
if numeric_columns:
    df[numeric_columns].hist(
        figsize=(12, 8),
        bins=20,
        edgecolor="black"
    )
    plt.suptitle("Numerical Feature Distributions")
    plt.tight_layout()
    plt.savefig("docs/numerical_distributions.png", dpi=150)
    plt.show()
    plt.close()

# 2. Correlation heatmap
if len(numeric_columns) >= 2:
    plt.figure(figsize=(10, 7))
    correlation = df[numeric_columns].corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("docs/correlation_heatmap.png", dpi=150)
    plt.show()
    plt.close()

# 3. Categorical count plots
for column in categorical_columns[:3]:
    plt.figure(figsize=(10, 5))

    order = df[column].value_counts().head(10).index

    sns.countplot(
        data=df,
        y=column,
        order=order
    )

    plt.title(f"Top Categories in {column}")
    plt.tight_layout()

    filename = column.replace(" ", "_")
    plt.savefig(f"docs/{filename}_countplot.png", dpi=150)
    plt.show()
    plt.close()

# 4. Box plots for numerical columns
if numeric_columns:
    plt.figure(figsize=(12, 6))

    sns.boxplot(data=df[numeric_columns])

    plt.title("Numerical Features - Outlier Analysis")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("docs/boxplot_outliers.png", dpi=150)
    plt.show()
    plt.close()

print("\nVisualization completed!")
print("Charts saved inside the docs/ folder.")
