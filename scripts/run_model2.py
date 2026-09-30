from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.models.optimization import model2
from src.utils.config import read_config_csv

if __name__ == "__main__":
    result=model2(pd.read_parquet("data/processed/cleaned_data.parquet"),read_config_csv("config/capacity_by_line.csv"),read_config_csv("config/service_constraints.csv"),read_config_csv("config/fleet_budget.csv")); Path("outputs").mkdir(exist_ok=True); result.frame.to_csv("outputs/model2_results.csv",index=False); print(result.status,result.termination)
