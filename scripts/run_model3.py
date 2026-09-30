from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.models.goal_programming import model3
from src.utils.config import read_config_csv, read_yaml

if __name__ == "__main__":
    frame=pd.read_parquet("data/processed/cleaned_data.parquet"); result,status,termination=model3(frame,read_config_csv("config/capacity_by_line.csv"),read_config_csv("config/service_constraints.csv"),read_yaml("config/goal_weights.yaml")); result.to_csv("outputs/model3_results.csv",index=False); print(status,termination)
