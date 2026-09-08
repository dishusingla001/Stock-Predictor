import glob
import os

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def check_file(file_path):
    data = pd.read_csv(file_path)
    file_name = os.path.basename(file_path)

    print("\n" + "=" * 70)
    print(f"FILE: {file_name}")
    print("=" * 70)
    print(f"Rows: {len(data)} | Columns: {len(data.columns)}")

    if "Date" in data.columns:
        dates = pd.to_datetime(data["Date"], errors="coerce")
        print(f"Date range: {dates.min().date()} to {dates.max().date()}")
        print(f"Invalid dates: {dates.isna().sum()}")

    missing_values = data.isna().sum()
    missing_values = missing_values[missing_values > 0]
    print("\nMissing values:")
    print(missing_values.to_string() if not missing_values.empty else "None")

    if "Target" in data.columns:
        target_counts = data["Target"].value_counts().sort_index()
        print("\nTarget distribution:")
        print(target_counts.rename(index={0: "DOWN (0)", 1: "UP (1)"}).to_string())

    print("\nColumns:")
    print(", ".join(data.columns))
    print("\nFirst 3 rows:")
    print(data.head(3).to_string(index=False))


def main():
    file_paths = sorted(
        glob.glob(os.path.join(PROCESSED_DIR, "RELIANCE_features.csv"))
        + glob.glob(os.path.join(PROCESSED_DIR, "RELIANCE_ML.csv"))
    )

    if not file_paths:
        print(f"No processed CSV files found in: {PROCESSED_DIR}")
        return

    print(f"Checking {len(file_paths)} processed dataset(s)...")
    for file_path in file_paths:
        check_file(file_path)

    print("\nProcessed data check complete.")


if __name__ == "__main__":
    main()