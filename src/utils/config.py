from __future__ import annotations
from pathlib import Path
import pandas as pd
import yaml


def read_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_config_csv(path: str | Path) -> pd.DataFrame:
    file = Path(path)
    if not file.exists():
        return pd.DataFrame()
    return pd.read_csv(file, comment="#").dropna(how="all")
