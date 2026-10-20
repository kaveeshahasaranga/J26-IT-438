
from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datasets"

files = [
    "member04_ui_quality_dataset.csv",
    "member04_issue_severity_dataset.csv",
    "member04_recommendation_dataset.csv"
]

print("MEMBER 04 DATASET CHECK")
print("-" * 40)

for filename in files:
    path = DATA_DIR / filename

    if not path.exists():
        print(f"MISSING: {filename}")
        continue

    try:
        df = pd.read_csv(path)

        print(f"PASS: {filename}")
        print(f"  Records: {len(df)}")
        print(f"  Columns: {len(df.columns)}")

    except EmptyDataError:
        print(f"EMPTY FILE: {filename}")
        print("  No CSV column headers found.")

print("-" * 40)
print("DATASET CHECK COMPLETED")
