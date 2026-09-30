from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.optimization import model1, model2
from src.models.goal_programming import model3


DEMO_NOTE = "DEMONSTRATION ONLY: illustrative capacities, bounds, targets, weights, and no fleet budget. Not TfL operational data."


def main() -> None:
    frame = pd.read_parquet("data/processed/cleaned_data.parquet")
    frame = frame.dropna(subset=["line", "time_period", "link_load", "line_boarders"]).copy()
    lines = sorted(frame["line"].dropna().unique())
    capacity = pd.DataFrame({
        "mode": "TEST DEMO",
        "line": lines,
        "service_group": lines,
        "capacity_per_train": 10000.0,
        "capacity_definition": "Illustrative value only",
        "source": "DEMO ONLY",
        "source_url": "",
        "notes": DEMO_NOTE,
    })
    constraints = pd.DataFrame({
        "mode": "TEST DEMO",
        "line": lines,
        "service_group": lines,
        "min_frequency_per_15min": 0,
        "max_frequency_per_15min": 20,
        "target_frequency_per_15min": 8,
        "notes": DEMO_NOTE,
    })
    weights = {"unmet_demand": 1.0, "frequency_deviation": 0.5, "crowding": 2.0, "service_usage": 0.01}
    output = Path("outputs/demo_run")
    output.mkdir(parents=True, exist_ok=True)

    result1 = model1(frame, capacity, constraints)
    result1.frame.to_csv(output / "model1_demo_results.csv", index=False)
    result2 = model2(frame, capacity, constraints, pd.DataFrame())
    result2.frame.to_csv(output / "model2_demo_results.csv", index=False)
    result3, status3, termination3 = model3(frame, capacity, constraints, weights, target_utilization=0.80)
    result3.to_csv(output / "model3_demo_results.csv", index=False)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "data/processed/cleaned_data.parquet",
        "records": len(frame),
        "lines": lines,
        "note": DEMO_NOTE,
        "parameters": {
            "capacity_per_train": 10000,
            "min_frequency_per_15min": 0,
            "max_frequency_per_15min": 20,
            "target_frequency_per_15min": 8,
            "target_utilization": 0.80,
            "goal_weights": weights,
            "fleet_budget": None,
        },
        "models": {
            "model1": {"status": result1.status, "termination": result1.termination, "rows": len(result1.frame)},
            "model2": {"status": result2.status, "termination": result2.termination, "rows": len(result2.frame)},
            "model3": {"status": status3, "termination": termination3, "rows": len(result3)},
        },
    }
    (output / "demo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
