from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re
import pandas as pd
import yaml

ALIASES: dict[str, list[str]] = {
    "time_period": ["time", "timeband", "time period", "15 minute", "quarter hour", "period"],
    "day_type": ["day type", "day_type", "day", "profile"],
    "mode": ["mode", "transport mode"],
    "line": ["line", "line name", "route", "service"],
    "from_station": ["from station", "origin station", "station from", "from"],
    "to_station": ["to station", "destination station", "station to", "to"],
    "direction": ["direction", "dir"],
    "link_load": ["link load", "load", "link passengers", "passengers link"],
    "link_frequency": ["link frequency", "frequency", "trains", "train frequency"],
    "line_boarders": ["line boarders", "line boarding", "boarders", "line entries"],
    "station_boarders": ["station boarders", "station boarding"],
    "alighters": ["alighters", "station alighters"],
    "station_entry": ["station entry", "entries", "entry"],
    "station_exit": ["station exit", "exits", "exit"],
    "journey_time": ["journey time", "time journey"],
}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def find_workbook(explicit: str | None = None, root: Path = Path(".")) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Configured DATA_FILE does not exist: {path}")
        return path
    candidates = sorted((root / "data" / "raw").glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError("No NUMBAT workbook found. Download the official workbook from https://crowding.data.tfl.gov.uk/ and place it in data/raw/.")
    if len(candidates) > 1:
        raise ValueError(f"Multiple workbooks found in data/raw; set DATA_FILE explicitly: {[p.name for p in candidates]}")
    return candidates[0]


def detect_schema(columns: list[str], mapping: dict[str, str] | None = None) -> dict[str, str | None]:
    mapping = mapping or {}
    normalized = {_norm(c): c for c in columns}
    result: dict[str, str | None] = {}
    for canonical, aliases in ALIASES.items():
        if canonical in mapping:
            result[canonical] = mapping[canonical]
            continue
        result[canonical] = next((normalized[_norm(alias)] for alias in aliases if _norm(alias) in normalized), None)
    return result


def load_mapping(path: Path = Path("config/schema_mapping.yaml")) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def profile_workbook(path: Path, mapping_path: Path = Path("config/schema_mapping.yaml")) -> dict[str, Any]:
    workbook = pd.ExcelFile(path, engine="openpyxl")
    mapping = load_mapping(mapping_path)
    sheets: list[dict[str, Any]] = []
    for sheet in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet)
        schema = detect_schema([str(c) for c in frame.columns], mapping.get("columns"))
        duplicates = int(frame.duplicated().sum())
        numeric = frame.select_dtypes(include="number")
        negative = {str(c): int((numeric[c] < 0).sum()) for c in numeric if (numeric[c] < 0).any()}
        sheets.append({"sheet": sheet, "rows": len(frame), "columns": len(frame.columns), "column_names": [str(c) for c in frame.columns], "schema": schema, "duplicate_rows": duplicates, "null_counts": frame.isna().sum().astype(int).to_dict(), "negative_counts": negative})
    likely = [s for s in sheets if s["schema"].get("link_load") and s["schema"].get("link_frequency")]
    wide_numbat = {"Link_Loads", "Link_Frequencies"}.issubset(set(workbook.sheet_names))
    required = ["line", "from_station", "to_station", "link_load", "link_frequency"]
    selected = mapping.get("sheet") or ("Link_Loads" if wide_numbat else (likely[0]["sheet"] if len(likely) == 1 else None))
    valid = wide_numbat or bool(selected and all(next(s for s in sheets if s["sheet"] == selected)["schema"].get(f) for f in required))
    return {"workbook": str(path), "sheets": sheets, "selected_sheet": selected, "required_fields": required, "candidate_sheets": [s["sheet"] for s in likely], "wide_numbat_layout": wide_numbat, "valid": valid}


def load_selected(path: Path, profile: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str | None]]:
    if profile.get("wide_numbat_layout"):
        return load_wide_numbat(path), {"layout": "NUMBAT wide sheets", "link_load": "Link_Loads", "link_frequency": "Link_Frequencies", "line_boarders": "Line_Boarders"}
    if not profile.get("selected_sheet"):
        raise ValueError("Could not identify a unique link sheet. Inspect data/processed/dataset_profile.json and set sheet/columns in config/schema_mapping.yaml.")
    frame = pd.read_excel(path, sheet_name=profile["selected_sheet"], engine="openpyxl")
    schema = next(s["schema"] for s in profile["sheets"] if s["sheet"] == profile["selected_sheet"])
    missing = [field for field in profile["required_fields"] if not schema.get(field)]
    if missing:
        raise ValueError(f"Required NUMBAT fields could not be mapped: {missing}. Add explicit mappings in config/schema_mapping.yaml.")
    return frame, schema


def _wide_sheet(path: Path, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet, header=2, engine="openpyxl")


def load_wide_numbat(path: Path) -> pd.DataFrame:
    loads = _wide_sheet(path, "Link_Loads")
    frequencies = _wide_sheet(path, "Link_Frequencies")
    link_ids = ["Link", "Line", "Dir", "Order", "From NLC", "From ASC", "From Station", "To NLC", "To ASC", "To Station"]
    load_id = [c for c in link_ids if c in loads.columns]
    time_columns = [c for c in loads.columns if isinstance(c, str) and "-" in c]
    def melt(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
        ids = [c for c in load_id if c in frame.columns]
        return frame.melt(id_vars=ids, value_vars=[c for c in time_columns if c in frame.columns], var_name="time_period", value_name=value_name)
    result = melt(loads, "link_load").merge(melt(frequencies, "link_frequency"), on=load_id + ["time_period"], how="left")
    result = result.rename(columns={"Line": "line", "Dir": "direction", "From Station": "from_station", "To Station": "to_station"})
    boarders = _wide_sheet(path, "Line_Boarders")
    boarder_line = "Line Name" if "Line Name" in boarders.columns else "Line"
    boarders = boarders.melt(id_vars=[boarder_line], value_vars=[c for c in time_columns if c in boarders.columns], var_name="time_period", value_name="line_boarders")
    boarders = boarders.rename(columns={boarder_line: "line"})
    result = result.merge(boarders, on=["line", "time_period"], how="left")
    return result[["line", "direction", "from_station", "to_station", "time_period", "link_load", "link_frequency", "line_boarders"]]


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
