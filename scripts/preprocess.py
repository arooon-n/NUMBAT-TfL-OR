from pathlib import Path
import json, os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion.loader import find_workbook, profile_workbook, load_selected
from src.preprocessing.clean import preprocess

if __name__ == "__main__":
    path=find_workbook(os.getenv("DATA_FILE")); profile=profile_workbook(path); frame,schema=load_selected(path,profile); preprocess(frame,schema); Path("data/processed/dataset_profile.json").write_text(json.dumps(profile,indent=2,default=str),encoding="utf-8"); print("Cleaned data written to data/processed/cleaned_data.parquet")
