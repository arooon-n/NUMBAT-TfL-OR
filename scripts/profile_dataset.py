from pathlib import Path
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion.loader import find_workbook, profile_workbook, write_json

if __name__ == "__main__":
    path = find_workbook(os.getenv("DATA_FILE"))
    profile = profile_workbook(path)
    write_json(profile, Path("data/processed/dataset_profile.json"))
    print(f"Profile written for {path}: selected_sheet={profile['selected_sheet']}, valid={profile['valid']}")
    if not profile["valid"]:
        raise SystemExit("No valid unique link sheet was identified. Review dataset_profile.json and config/schema_mapping.yaml.")
