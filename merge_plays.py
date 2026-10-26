import os
import argparse
import pandas as pd

# Search this project folder by default, on any operating system.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Save the merged file in the archive folder
OUTPUT_DIR = os.path.join(BASE_DIR, "archive")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "2010_2026_plays.csv")

DATA_PLAYS = [
    f"{year}_plays.csv"
    for year in range(2010, 2027)
]


def find_csv_file(filename):
    """Search ML and all its subfolders for a CSV file."""

    for root, dirs, files in os.walk(BASE_DIR):
        # Don't search generated output folders if desired
        if filename in files:
            path = os.path.join(root, filename)

            # Don't use the merged output as an input
            if os.path.abspath(path) == os.path.abspath(OUTPUT_FILE):
                continue

            return path

    return None


def merge_plays():
    """Merge yearly play CSV files into one CSV."""

    all_plays = []

    for filename in DATA_PLAYS:
        path = find_csv_file(filename)

        if path is None:
            print(f"WARNING: {filename} not found.")
            continue

        try:
            df = pd.read_csv(path)
            df["source_year"] = int(filename[:4])
            all_plays.append(df)

            print(f"Loaded {filename}: {len(df)} rows")
            print(f"  Location: {path}")

        except Exception as e:
            print(f"ERROR reading {filename}: {e}")

    if not all_plays:
        print("\nNo yearly play CSV files found.")
        print(f"Searched: {BASE_DIR}")
        print("Check the dataset's location and actual filenames.")
        return

    merged_df = pd.concat(
        all_plays,
        ignore_index=True,
        sort=False
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_df.to_csv(OUTPUT_FILE, index=False)

    print("\nMerge completed!")
    print(f"Files merged: {len(all_plays)}")
    print(f"Total rows: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge yearly NFL play CSV files.")
    parser.add_argument("--data-dir", default=BASE_DIR,
                        help="Folder containing YEAR_plays.csv files (searched recursively).")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Folder for the merged CSV.")
    args = parser.parse_args()
    BASE_DIR = os.path.abspath(args.data_dir)
    OUTPUT_DIR = os.path.abspath(args.output_dir)
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "2010_2026_plays.csv")
    merge_plays()
