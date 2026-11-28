import pandas as pd

def clean_data(df):
    """
    Clean the dataset by handling missing values,
    removing empty rows and columns, and formatting data.
    """

    df = df.copy()

    # Remove rows and columns that are completely empty
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # Remove extra spaces from text columns
    for column in df.select_dtypes(include=["object", "str"]).columns:
        df[column] = df[column].str.strip()

    # Convert Date column into datetime format
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Fill missing numeric values with the column mean
    for column in df.select_dtypes(include="number").columns:
        if df[column].isna().any():
            mean_value = df[column].mean()

            if pd.notna(mean_value):
                df[column] = df[column].fillna(mean_value)

    # Fill missing text values with the most frequent value
    for column in df.select_dtypes(include=["object", "str"]).columns:
        if df[column].isna().any():
            mode_values = df[column].mode()

            if not mode_values.empty:
                df[column] = df[column].fillna(mode_values[0])

    print("Data cleaning completed.")

    return df
