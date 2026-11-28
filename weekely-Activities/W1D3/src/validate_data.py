
import pandas as pd


REQUIRED_COLUMNS = [
    "Date",
    "State",
    "Confirmed",
    "Recovered",
    "Deceased",
    "Other",
    "Tested"
]


def validate_data(df):
    """Validate dataset structure and identify data quality issues."""

    print("\n========== DATA VALIDATION ==========")

    # Check required columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Required columns: OK")

    # Check empty dataset
    if df.empty:
        raise ValueError("Dataset is empty.")

    print(f"Dataset rows: {len(df)}")

    # Check duplicate rows
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}")

    # Check invalid dates
    invalid_dates = pd.to_datetime(
        df["Date"], errors="coerce"
    ).isna().sum()

    print(f"Invalid dates: {invalid_dates}")

    # Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Check negative numeric values
    numeric_columns = [
        "Confirmed", "Recovered", "Deceased", "Other", "Tested"
    ]

    for column in numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        negative_count = (values < 0).sum()

        if negative_count:
            print(
                f"Warning: {column} contains "
                f"{negative_count} negative values."
            )

    print("\nData validation completed.")

    return True
