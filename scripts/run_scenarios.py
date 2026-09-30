from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.scenarios.runner import run_scenarios
from src.utils.config import read_config_csv

if __name__ == "__main__":
    result=run_scenarios(pd.read_parquet("data/processed/cleaned_data.parquet"),read_config_csv("config/capacity_by_line.csv"),read_config_csv("config/service_constraints.csv"),read_config_csv("config/fleet_budget.csv")); result.to_csv("outputs/scenario_results.csv",index=False); print("Scenario results written")
