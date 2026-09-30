from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from src.validation.checks import save_report

if __name__ == "__main__":
    report={"status":"Review model output files and run model-specific validation before presentation.","outputs":sorted(str(p) for p in Path("outputs").glob("*.csv"))}; save_report(report); print("Validation report written")
