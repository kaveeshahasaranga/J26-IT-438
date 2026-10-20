
from pathlib import Path
from datetime import date
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "datasets" / "member04_ui_quality_dataset.csv"

df = pd.read_csv(CSV_PATH, dtype=str)

print("MEMBER 04 - UI/UX DATA COLLECTION")
print("-" * 40)

# Automatically generate next interface ID
existing_ids = pd.to_numeric(
    df["interface_id"].str.extract(r"^UI(\d+)$")[0],
    errors="coerce"
).dropna()

next_id = int(existing_ids.max()) + 1 if not existing_ids.empty else 1
interface_id = f"UI{next_id:03d}"

print("Interface ID:", interface_id)

page_url = input("Website page URL: ").strip()
page_type = input("Page type (Homepage/Login/Product/etc): ").strip()
device_type = input("Device type (Desktop/Mobile): ").strip()

if not page_url.startswith(("http://", "https://")):
    raise ValueError("Please enter a full http:// or https:// URL")

if not page_type or not device_type:
    raise ValueError("Page type and device type are required")

new_record = {
    "interface_id": interface_id,
    "page_url": page_url,
    "page_type": page_type,
    "device_type": device_type,
    "screenshot_filename": f"{interface_id}.png",
    "collection_date": date.today().isoformat(),
    "reviewer_id": "",
    "expert_quality_rating": "",
    "rating_scale": "0-100",
    "reviewer_comments": "",
    "label_status": "pending"
}

df = pd.concat(
    [df, pd.DataFrame([new_record])],
    ignore_index=True
)

df.to_csv(CSV_PATH, index=False)

print("\nDATA SAVED SUCCESSFULLY!")
print("Interface ID:", interface_id)
print("Total records:", len(df))
print("Expert rating status: pending")
print("Remember to capture the actual screenshot.")
