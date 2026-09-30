from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.validation.checks import validate_result, save_report

if __name__ == "__main__":
    reports=[]
    for name,path in (("Model 1","outputs/model1_results.csv"),("Model 2","outputs/model2_results.csv"),("Model 3","outputs/model3_results.csv")):
        try: reports.append(validate_result(pd.read_csv(path),name))
        except FileNotFoundError: reports.append({"model":name,"passed":False,"checks":{"result_file_present":False}})
    save_report({"models":reports,"passed":all(r["passed"] for r in reports)})
    print("Validation report written")
