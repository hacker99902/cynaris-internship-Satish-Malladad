import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR / "plots"
OUTPUT_DIR = BASE_DIR / "outputs"


def get_data_path():
    parser = argparse.ArgumentParser(description="Explore a CSV dataset")
    parser.add_argument(
        "--data",
        type=Path,
        default=BASE_DIR / "data" / "sample_dataset.csv",
        help="Path to the CSV file",
    )
    return parser.parse_args().data


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    data_file = get_data_path()
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # Load and inspect the data
    df = pd.read_csv(data_file)
    if df.empty:
        raise ValueError("The dataset is empty.")

    logging.info("Loaded %s rows and %s columns", df.shape[0], df.shape[1])
    print("\nFirst five rows:")
    print(df.head())

    print("\nDataFrame information:")
    df.info()

    numeric_cols = df.select_dtypes(include="number").columns
    category_cols = df.select_dtypes(include=["object", "category"]).columns

    print("\nNumeric summary:")
    print(df[numeric_cols].describe())

    print("\nCategorical summary:")
    if len(category_cols):
        print(df[category_cols].describe())
    else:
        print("No categorical columns found.")

    missing = df.isnull().sum()
    print("\nMissing values:")
    print(missing)

    # Save summaries
    df[numeric_cols].describe().to_csv(OUTPUT_DIR / "numeric_summary.csv")
    if len(category_cols):
        df[category_cols].describe().to_csv(
            OUTPUT_DIR / "categorical_summary.csv"
        )
    missing.to_csv(OUTPUT_DIR / "missing_values.csv")

    # Plot each numeric column
    for col in numeric_cols:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        name = re.sub(r"[^A-Za-z0-9_-]", "_", str(col))
        plt.savefig(PLOT_DIR / f"{name}_distribution.png")
        plt.close()

    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(9, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "correlation_heatmap.png")
        plt.close()

    # Top 10 values for categorical columns
    for col in category_cols:
        counts = df[col].fillna("Missing").astype(str).value_counts().head(10)
        plt.figure(figsize=(8, 4))
        counts.plot(kind="bar")
        plt.title(f"Top 10 values in {col}")
        plt.ylabel("Count")
        plt.tight_layout()
        name = re.sub(r"[^A-Za-z0-9_-]", "_", str(col))
        plt.savefig(PLOT_DIR / f"{name}_top10.png")
        plt.close()

    # Basic data-quality checks
    duplicates = int(df.duplicated().sum())
    negative_values = (df[numeric_cols] < 0).sum()

    print("\nDuplicate rows:", duplicates)
    print("\nNegative values:")
    print(negative_values)

    (OUTPUT_DIR / "data_quality.txt").write_text(
        f"Duplicate rows: {duplicates}\n\n"
        f"Negative values by column:\n{negative_values.to_string()}\n",
        encoding="utf-8",
    )

    # Five observations based on the actual dataset
    observations = [
        f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.",
        f"It has {len(numeric_cols)} numeric columns.",
        f"It has {len(category_cols)} categorical columns.",
        f"There are {int(missing.sum())} missing values in total.",
        f"There are {duplicates} duplicate rows.",
    ]

    print("\nFive observations:")
    for number, observation in enumerate(observations, start=1):
        print(f"{number}. {observation}")

    (OUTPUT_DIR / "five_observations.txt").write_text(
        "\n".join(observations), encoding="utf-8"
    )

    # EDA narrative
    narrative = """
The dataset contains information arranged in rows and columns, with both
numeric and categorical data. I started by checking the dataset structure,
data types and summary statistics to understand what kind of information
it contains.

During the initial inspection, I checked for missing values and duplicate
records. Missing values need attention because they can affect the results
of further analysis. Duplicate records should also be reviewed to make sure
the same information is not being counted more than once. I also checked
for negative values to identify anything that might need further
investigation.

The histograms helped me understand how the numeric values are distributed.
The category charts showed which values appear most frequently, while the
correlation heatmap helped identify relationships between numeric columns.
These patterns give a better understanding of the dataset, although
correlation does not necessarily mean that one value causes another.

Before using this data for machine learning, I would investigate unusual
values, decide how to handle missing data and duplicates, and check whether
any transformations are needed. I would also keep the original dataset
unchanged so that the analysis can be repeated whenever required.
""".strip()

    (OUTPUT_DIR / "EDA_NARRATIVE.md").write_text(
        "# EDA Narrative\n\n" + narrative + "\n",
        encoding="utf-8",
    )

    logging.info("EDA finished. Check the plots and outputs folders.")


if __name__ == "__main__":
    main()
