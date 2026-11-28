def inspect_data(df):
    """
    Display important information about the dataset.
    """

    print("\n========== DATASET OVERVIEW ==========")

    # Dataset dimensions
    print("\nDataset shape:")
    print(df.shape)

    # First five rows
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    # Column data types
    print("\nColumn data types:")
    print(df.dtypes)

    # Statistical summary
    print("\nNumerical summary:")
    print(df.describe().to_string())

    # Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    print("\n========== INSPECTION COMPLETED ==========")
