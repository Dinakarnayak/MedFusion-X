"""Validate a local NIH ChestX-ray14 installation for MedFusion-X."""
from pathlib import Path
import argparse
import pandas as pd

EXPECTED_CSV = "Data_Entry_2017.csv"
EXPECTED_IMAGE_DIR = "images"

def validate(root: Path) -> bool:
    csv_path = root / EXPECTED_CSV
    image_dir = root / EXPECTED_IMAGE_DIR
    print(f"Dataset root: {root}")
    if not csv_path.exists():
        print("ERROR: Data_Entry_2017.csv was not found.")
        return False
    if not image_dir.exists():
        print("ERROR: images/ directory was not found.")
        return False
    df = pd.read_csv(csv_path)
    required = {"Image Index", "Finding Labels"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: missing CSV columns: {sorted(missing)}")
        return False
    print(f"Metadata rows: {len(df):,}")
    print(f"Image files present: {len(list(image_dir.glob("*.png"))):,}")
    print("Dataset structure is valid for MedFusion-X.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/nih_chest_xray14")
    args = parser.parse_args()
    raise SystemExit(0 if validate(Path(args.root)) else 1)