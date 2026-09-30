from __future__ import annotations
import pandas as pd
from src.models.optimization import model1


def run_scenarios(frame, capacity, constraints, fleet, multipliers=(1.0, 1.10, 1.20)) -> pd.DataFrame:
    outputs=[]
    for multiplier in multipliers:
        result = model1(frame, capacity, constraints, multiplier, fleet).frame.copy()
        result["scenario"] = f"demand_x{multiplier:g}"
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
