
from src.load_data import load_data
from src.clean_data import clean_data
from src.inspect_data import inspect_data
from src.validate_data import validate_data

def main():
    # Load the original dataset
    df = load_data("data/raw/states.csv")

    # Inspect the original dataset
    print("\nINSPECTING RAW DATA")
    inspect_data(df)

    # Clean the dataset
    cleaned_df = clean_data(df)
    # Validate the cleaned dataset
    validate_data(cleaned_df)
    # Inspect the cleaned dataset
    print("\nINSPECTING CLEANED DATA")
    inspect_data(cleaned_df)

    # Save the cleaned dataset
    cleaned_df.to_csv(
        "data/cleaned/states_cleaned.csv",
        index=False
    )

    print("\nCleaned dataset saved successfully.")


if __name__ == "__main__":
    main()
