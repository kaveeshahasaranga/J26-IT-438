
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "datasets" / "member04_ui_quality_dataset.csv"
SCREENSHOT_DIR = BASE_DIR / "screenshots"

df = pd.read_csv(CSV_PATH, dtype=str).fillna("")

print("\nMEMBER 04 - DATASET VALIDATION")
print("-" * 40)
print("Total CSV records:", len(df))

# Check duplicate website URLs
duplicates = df[
    df.duplicated(subset=["page_url", "device_type"], keep=False)
]

if duplicates.empty:
    print("PASS: No duplicate URL/device combinations")
else:
    print("\nWARNING: Possible duplicate interfaces")
    print(
        duplicates[
            ["interface_id", "page_url", "device_type"]
        ].to_string(index=False)
    )

# Check matching screenshot files
missing = []

for _, row in df.iterrows():
    filename = row["screenshot_filename"]
    if not filename or not (SCREENSHOT_DIR / filename).is_file():
        missing.append(row["interface_id"])

if missing:
    print("\nWARNING: Missing screenshots:", missing)
else:
    print("PASS: Every CSV record has a matching screenshot")

# Check expert ratings
pending = df[
    df["expert_quality_rating"].str.strip() == ""
]

print("Pending expert ratings:", len(pending))
print("-" * 40)
print("VALIDATION CHECK COMPLETED")
