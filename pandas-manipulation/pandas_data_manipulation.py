import pandas as pd
import os

# Load the dataset
df = pd.read_csv("data/states.csv")

# Convert date into datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Remove India because we only need state-wise data
df = df[df["State"] != "India"]

# Display basic information about the dataset
print("\nDataset Information")
print("Shape:", df.shape)

print("\nColumn Data Types:")
print(df.dtypes)

print("\nFirst 10 Rows:")
print(df.head(10))


# Filter rows where confirmed cases are above 10,000
filtered_data = df[df["Confirmed"] > 10000]

print("\nFiltered Data:")
print(filtered_data.head())


# Group the data state-wise
# Find the highest confirmed, recovered and deceased cases
state_summary = filtered_data.groupby("State").agg({
    "Confirmed": "max",
    "Recovered": "max",
    "Deceased": "max"
}).reset_index()

print("\nState-wise Summary:")
print(state_summary.head())


# Get the latest record for each state
latest_data = (
    df.sort_values("Date")
    .groupby("State")
    .tail(1)
)

latest_data = latest_data[["State", "Date", "Tested"]]

# Merge both datasets using State
merged_data = pd.merge(
    state_summary,
    latest_data,
    on="State",
    how="left"
)

print("\nMerged Data:")
print(merged_data.head())


# Create a pivot table showing confirmed cases by state and date
pivot_data = pd.pivot_table(
    df,
    index="State",
    columns="Date",
    values="Confirmed",
    aggfunc="max",
    fill_value=0
)

print("\nPivot Table Shape:", pivot_data.shape)


# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)

# Save the result in CSV format
merged_data.to_csv("output/merged_data.csv", index=False)

# Save the same result in Parquet format
merged_data.to_parquet("output/merged_data.parquet", index=False)


# Compare the sizes of CSV and Parquet files
csv_size = os.path.getsize("output/merged_data.csv") / 1024
parquet_size = os.path.getsize("output/merged_data.parquet") / 1024

print("\nFile Size Comparison:")
print(f"CSV Size: {csv_size:.2f} KB")
print(f"Parquet Size: {parquet_size:.2f} KB")

if csv_size > parquet_size:
    print("Parquet file takes less storage space.")
elif parquet_size > csv_size:
    print("CSV file takes less storage space.")
else:
    print("Both files have the same size.")

print("\nPandas data manipulation completed successfully!")
