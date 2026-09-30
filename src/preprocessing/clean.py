from __future__ import annotations
from pathlib import Path
import pandas as pd


def canonicalize(frame: pd.DataFrame, schema: dict[str, str | None]) -> pd.DataFrame:
    rename = {source: canonical for canonical, source in schema.items() if source and source in frame.columns}
    result = frame.rename(columns=rename).copy()
    for field in ["line", "from_station", "to_station", "direction", "mode", "day_type"]:
        if field in result:
            result[field] = result[field].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    for field in ["link_load", "link_frequency", "line_boarders", "station_boarders", "alighters", "station_entry", "station_exit", "journey_time"]:
        if field in result:
            result[field] = pd.to_numeric(result[field], errors="coerce")
    if "time_period" in result:
        result["time_period"] = result["time_period"].astype("string").str.strip()
    result = result.drop_duplicates().reset_index(drop=True)
    if "link_load" in result and (result["link_load"] < 0).any():
        raise ValueError("Negative link loads detected; data was not silently altered.")
    if "link_frequency" in result and (result["link_frequency"] < 0).any():
        raise ValueError("Negative link frequencies detected; data was not silently altered.")
    return result


def preprocess(frame: pd.DataFrame, schema: dict[str, str | None], output: Path = Path("data/processed/cleaned_data.parquet")) -> pd.DataFrame:
    result = canonicalize(frame, schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result
