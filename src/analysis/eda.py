from __future__ import annotations
import pandas as pd


def demand_by_time(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("time_period", as_index=False)["link_load"].sum()


def frequency_by_time(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("time_period", as_index=False)["link_frequency"].mean()


def top_links(frame: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    return frame.sort_values("link_load", ascending=False).head(n)


def baseline_utilization(frame: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    work = frame.merge(capacity[["line", "capacity_per_train"]], on="line", how="left")
    work["utilization"] = work["link_load"] / (work["capacity_per_train"] * work["link_frequency"])
    return work
