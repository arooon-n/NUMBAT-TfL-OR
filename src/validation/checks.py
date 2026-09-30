from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


def validate_result(result: pd.DataFrame, model_name: str) -> dict:
    checks = {}
    checks["solver_status_present"] = "solver_status" in result and result.solver_status.notna().all() if not result.empty else False
    checks["no_negative_values"] = not any((pd.to_numeric(result[c], errors="coerce") < 0).any() for c in result.columns if pd.api.types.is_numeric_dtype(result[c]))
    if model_name == "Model 2" and not result.empty:
        checks["served_le_demand"] = (result.demand_served <= result.demand + 1e-6).all()
        checks["served_le_supply"] = (result.demand_served <= result.capacity * result.trains_allocated + 1e-6).all()
    else:
        checks["served_le_demand"] = True; checks["served_le_supply"] = True
    return {"model": model_name, "passed": all(checks.values()), "checks": checks, "rows": len(result)}


def save_report(report: dict, json_path: Path = Path("outputs/validation_report.json"), text_path: Path = Path("outputs/validation_report.txt")) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True); json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    text_path.write_text("Validation report\n=================\n" + "\n".join(f"{k}: {v}" for k, v in report.items()), encoding="utf-8")
