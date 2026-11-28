# Pandas Data Manipulation

## Overview

This project demonstrates basic data manipulation techniques using Python and Pandas. It uses an Indian COVID-19 state-wise dataset to explore, filter, group, merge and summarize data.

## Dataset

- File: `data/states.csv`
- Contains historical COVID-19 records for Indian states and union territories.
- The national-level India records are excluded from state-wise analysis.

## Operations Performed

1. Loaded the CSV dataset into a Pandas DataFrame.
2. Checked the dataset shape, column data types and first 10 records.
3. Filtered records with more than 10,000 confirmed cases.
4. Grouped records by state to calculate maximum confirmed, recovered and deceased cases.
5. Merged the state-wise summary with the latest testing information.
6. Created a pivot table showing confirmed cases by state and date.
7. Exported the processed data into CSV and Parquet formats.
8. Compared the sizes of both output files.

## Technologies Used

- Python
- Pandas
- NumPy
- PyArrow

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the script:

```bash
python pandas_data_manipulation.py
```

## Output

Generated files are saved in the `output/` directory:

- `merged_data.csv`
- `merged_data.parquet`

## Note

The dataset contains historical COVID-19 information and should not be interpreted as current statistics.
