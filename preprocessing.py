"""Validation, cleaning and reshaping of NUMBAT link data.

Policy (documented in the README and shown in the app):
* Missing demand is never replaced by zero. Genuine numeric zeros are kept.
* Non-numeric and negative values are flagged as invalid and excluded.
* Exact duplicate rows are dropped once; records whose link key is still
  duplicated are ambiguous and excluded (never aggregated or fuzzy-matched).
* Demand and frequency are joined on a validated link key plus the interval.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_loader import LinkSheet, Workbook, interval_order, interval_start_text

TEXT_FIELDS = ["link_id", "line", "direction", "from_station", "to_station", "from_asc", "to_asc"]
ATTRIBUTE_FIELDS = ["link_id", "line", "direction", "order", "from_nlc", "from_station", "to_nlc", "to_station"]
# Candidate join keys, most specific first. The first one present in both sheets
# and unique in both is used.
KEY_CANDIDATES = [
    ["link_id"],
    ["line", "direction", "from_nlc", "to_nlc"],
    ["line", "direction", "from_station", "to_station"],
    ["line", "from_station", "to_station"],
]


@dataclass
class Processed:
    data: pd.DataFrame                 # long link-interval table (all records, with status)
    link_summary: pd.DataFrame         # one row per link: daily total and time-of-day columns
    key_fields: list[str]
    intervals: list[str]
    metrics: dict = field(default_factory=dict)
    sheet_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    exclusions: pd.DataFrame = field(default_factory=pd.DataFrame)


def _base_name(name) -> str:
    """Station name without a trailing bracketed qualifier, e.g. 'Kennington (Bank)' -> 'kennington'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(name)).strip().lower()


def _clean_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return text.mask(text.isin(["", "nan", "None"]))


def _prepare_sheet(sheet: LinkSheet, value_name: str, actions: list[str]) -> tuple[pd.DataFrame, dict]:
    """Rename fields, clean identifiers and convert interval values to numbers."""
    raw = sheet.data
    keep = list(sheet.fields.values()) + list(sheet.interval_columns) + sheet.period_columns
    frame = raw[keep].rename(columns={v: k for k, v in sheet.fields.items()})
    frame = frame.rename(columns=sheet.interval_columns)
    intervals = list(sheet.interval_columns.values())
    stats = {"sheet": sheet.sheet_name, "records": len(frame), "interval_columns": len(intervals)}

    for col in TEXT_FIELDS:
        if col in frame:
            frame[col] = _clean_text(frame[col])
    for col in ["order", "from_nlc", "to_nlc"]:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    dup_mask = frame.astype(str).duplicated()
    stats["exact_duplicate_rows"] = int(dup_mask.sum())
    if dup_mask.any():
        actions.append(f"{sheet.sheet_name}: dropped {int(dup_mask.sum())} exact duplicate row(s).")
    frame = frame[~dup_mask]

    no_id = frame[[c for c in ["line", "from_station", "to_station"] if c in frame]].isna().any(axis=1)
    stats["rows_missing_identifiers"] = int(no_id.sum())
    if no_id.any():
        actions.append(f"{sheet.sheet_name}: excluded {int(no_id.sum())} row(s) without line/station identifiers.")
    frame = frame[~no_id].reset_index(drop=True)

    values = frame[intervals]
    numeric = values.apply(pd.to_numeric, errors="coerce")
    missing = values.isna() | values.map(lambda v: isinstance(v, str) and not v.strip())
    non_numeric = numeric.isna() & ~missing
    negative = numeric < 0
    stats.update({
        "cells": int(values.size),
        "missing_values": int(missing.to_numpy().sum()),
        "non_numeric_values": int(non_numeric.to_numpy().sum()),
        "negative_values": int(negative.to_numpy().sum()),
    })
    status = np.select([missing, non_numeric, negative], ["missing", "non_numeric", "negative"], "valid")
    frame[intervals] = numeric
    status_frame = pd.DataFrame(status, columns=intervals, index=frame.index)

    # Daily 'total' and time-of-day columns stay out of the interval table (they are not per-interval values).
    id_cols = [c for c in frame.columns if c not in intervals and c not in sheet.period_columns and c != "total"]
    long = frame[id_cols + intervals].melt(id_vars=id_cols, var_name="interval", value_name=value_name)
    long[f"{value_name}_status"] = status_frame.melt(value_name="s")["s"].to_numpy()
    return long, {"frame": frame, "stats": stats}


def _choose_key(loads: pd.DataFrame, freqs: pd.DataFrame) -> tuple[list[str], list[str]]:
    notes = []
    for key in KEY_CANDIDATES:
        if not all(k in loads.columns and k in freqs.columns for k in key):
            continue
        if loads[key].isna().any().any() or freqs[key].isna().any().any():
            notes.append(f"Key {key} has missing values; trying a more general key.")
            continue
        if not loads.duplicated(key).any() and not freqs.duplicated(key).any():
            return key, notes
        notes.append(f"Key {key} is not unique in both sheets; trying another key.")
    fallback = next(k for k in KEY_CANDIDATES if all(c in loads.columns and c in freqs.columns for c in k))
    notes.append(f"No fully unique key found; using {fallback} and excluding duplicated links as ambiguous.")
    return fallback, notes


def preprocess(wb: Workbook) -> Processed:
    if wb.loads is None:
        raise ValueError("Link_Loads could not be parsed, so no link demand is available.")
    actions: list[str] = []
    warnings: list[str] = []
    loads_long, l_info = _prepare_sheet(wb.loads, "demand", actions)
    loads_wide = l_info["frame"]
    sheet_stats = [l_info["stats"]]

    if wb.frequencies is not None:
        freq_long, f_info = _prepare_sheet(wb.frequencies, "frequency", actions)
        freqs_wide = f_info["frame"]
        sheet_stats.append(f_info["stats"])
    else:
        freqs_wide = pd.DataFrame(columns=loads_wide.columns)
        freq_long = pd.DataFrame(columns=list(loads_long.columns[:-2]) + ["frequency", "frequency_status"])
        warnings.append("No Link_Frequencies sheet: observed service is unavailable, so Models 2 and 3 are disabled.")

    key, key_notes = _choose_key(loads_wide, freqs_wide if len(freqs_wide) else loads_wide)
    actions.extend(key_notes)
    actions.append(f"Demand and frequency joined on link key {key} + 15-minute interval.")

    # Links whose key is still duplicated are ambiguous: excluded, never aggregated.
    amb_l = set(map(tuple, loads_wide.loc[loads_wide.duplicated(key, keep=False), key].to_numpy()))
    amb_f = set(map(tuple, freqs_wide.loc[freqs_wide.duplicated(key, keep=False), key].to_numpy())) if len(freqs_wide) else set()
    ambiguous = amb_l | amb_f
    if ambiguous:
        actions.append(f"Excluded {len(ambiguous)} link key(s) that are duplicated with different values (ambiguous).")

    attr_cols = [c for c in ATTRIBUTE_FIELDS if c in loads_long.columns]
    f_cols = key + ["interval", "frequency", "frequency_status"]
    f_attr = [c for c in attr_cols if c in freq_long.columns and c not in key]
    merged = loads_long.merge(
        freq_long[f_cols + f_attr].rename(columns={c: f"{c}__freq" for c in f_attr}),
        on=key + ["interval"], how="outer", indicator=True,
    )
    for c in f_attr:  # fill attributes of frequency-only rows; detect conflicting attributes
        other = merged.pop(f"{c}__freq")
        both = merged[c].notna() & other.notna()
        if pd.api.types.is_numeric_dtype(merged[c]) or c in ("order", "from_nlc", "to_nlc"):
            differs = both & (pd.to_numeric(merged[c], errors="coerce") != pd.to_numeric(other, errors="coerce"))
        else:
            differs = both & (merged[c].astype(str) != other.astype(str))
        if differs.any():
            if c == "order":
                # Sequence numbers are descriptive only; they do not identify the link.
                n = merged.loc[differs, key].drop_duplicates().shape[0]
                warnings.append(f"{n} link(s) have a different 'Order' in the two sheets; the Link_Loads order is kept for display.")
            else:
                # Branch qualifiers such as 'Kennington (Bank)' vs 'Kennington' name the same place.
                if c in ("from_station", "to_station"):
                    differs &= merged[c].map(_base_name) != other.map(_base_name)
                    labels = both & ~differs & (merged[c].astype(str) != other.astype(str))
                    if labels.any():
                        n = merged.loc[labels, key].drop_duplicates().shape[0]
                        warnings.append(f"{n} link(s) label '{c}' with/without a branch qualifier in brackets in the two sheets; "
                                        "treated as the same station (Link_Loads label shown).")
                if differs.any():
                    bad = merged.loc[differs, key].drop_duplicates()
                    ambiguous |= set(map(tuple, bad.to_numpy()))
                    examples = merged.loc[differs, [c]].assign(other=other[differs]).drop_duplicates().head(3)
                    pairs = "; ".join(f"'{a}' vs '{b}'" for a, b in examples.to_numpy())
                    warnings.append(f"{len(bad)} link(s) have conflicting '{c}' values between the sheets ({pairs}); "
                                    "they are excluded as ambiguous.")
        merged[c] = merged[c].where(merged[c].notna(), other)

    merged["demand_status"] = merged["demand_status"].where(merged["_merge"] != "right_only", "no_demand_record")
    merged["frequency_status"] = merged["frequency_status"].where(merged["_merge"] != "left_only", "no_frequency_record")
    is_amb = pd.Series([tuple(r) in ambiguous for r in merged[key].to_numpy()], index=merged.index) if ambiguous else pd.Series(False, index=merged.index)
    merged.loc[is_amb, ["demand_status", "frequency_status"]] = "ambiguous_link"
    merged = merged.drop(columns="_merge")

    merged["eligible_demand"] = merged["demand_status"].eq("valid")
    merged["eligible_service"] = merged["eligible_demand"] & merged["frequency_status"].eq("valid")
    merged.loc[merged["demand_status"] != "valid", "demand"] = np.nan
    merged.loc[merged["frequency_status"] != "valid", "frequency"] = np.nan

    merged["interval_order"] = merged["interval"].map(interval_order)
    merged["interval_start"] = merged["interval"].map(interval_start_text)
    merged["link_key"] = merged[key].astype(str).agg(" | ".join, axis=1)
    if "link_id" not in merged:
        merged["link_id"] = merged["link_key"]
    merged["link_label"] = merged["from_station"].astype(str) + " → " + merged["to_station"].astype(str)
    if "direction" in merged:
        merged["link_label"] += " (" + merged["direction"].fillna("").astype(str) + ")"
    sort_cols = [c for c in ["line", "direction", "order", "link_key", "interval_order"] if c in merged]
    merged = merged.sort_values(sort_cols).reset_index(drop=True)

    non_int = merged["frequency"].notna() & (merged["frequency"] % 1 != 0)
    if non_int.any():
        warnings.append(f"{int(non_int.sum())} frequency value(s) are not whole trains; they are kept as observed averages.")

    # Passengers on a link in an interval with no scheduled train: internally inconsistent observations.
    # Flag them (the optimisation policy is the user's choice) and record whether the same link has
    # trains in the previous or next interval, which points to a period-boundary effect.
    merged["zero_train_demand"] = merged["eligible_service"] & (merged["frequency"] == 0) & (merged["demand"] > 0)
    grp = merged.groupby("link_key")
    adjacent = pd.Series(False, index=merged.index)
    for step in (1, -1):
        same_link_next = grp["interval_order"].shift(-step) == merged["interval_order"] + step
        adjacent |= same_link_next & (grp["frequency"].shift(-step) > 0)
    merged["zero_train_adjacent_service"] = merged["zero_train_demand"] & adjacent
    zt = merged[merged["zero_train_demand"]]
    total_valid = merged.loc[merged["eligible_service"], "demand"].sum()
    if len(zt):
        warnings.append(
            f"{len(zt):,} link-interval(s) report passengers but 0 scheduled trains "
            f"({zt['demand'].sum():,.0f} passengers, {100 * zt['demand'].sum() / max(total_valid, 1):.2f}% of demand). "
            f"For {100 * zt.loc[zt['zero_train_adjacent_service'], 'demand'].sum() / zt['demand'].sum():.0f}% of that demand "
            "the same link has trains in an adjacent interval. Choose how the models treat them in the sidebar.")

    # Interval coverage across the two sheets.
    l_int = set(wb.loads.interval_columns.values())
    f_int = set(wb.frequencies.interval_columns.values()) if wb.frequencies else set()
    intervals = sorted(l_int | f_int, key=interval_order)
    if wb.frequencies is not None and l_int != f_int:
        warnings.append(f"Intervals only in Link_Loads: {sorted(l_int - f_int, key=interval_order)}; "
                        f"only in Link_Frequencies: {sorted(f_int - l_int, key=interval_order)}.")
    if len(intervals) != 96:
        warnings.append(f"{len(intervals)} 15-minute intervals detected (a full NUMBAT day has 96).")

    # Daily totals and time-of-day columns are kept separately for description only.
    summary_cols = key + [c for c in attr_cols if c not in key] + wb.loads.period_columns
    total_col = wb.loads.fields.get("total")
    if total_col:
        loads_wide = loads_wide.rename(columns={total_col: "total"}) if total_col in loads_wide else loads_wide
    summary = loads_wide[[c for c in dict.fromkeys(summary_cols + (["total"] if "total" in loads_wide else [])) if c in loads_wide]].copy()
    for c in wb.loads.period_columns + (["total"] if "total" in summary else []):
        summary[c] = pd.to_numeric(summary[c], errors="coerce")
    if "total" in summary:
        interval_sum = loads_wide[list(l_int)].sum(axis=1, min_count=1)
        gap = (interval_sum - summary["total"]).abs()
        tol = 1e-6 * summary["total"].abs().clip(lower=1)
        n_bad = int((gap > tol).sum())
        actions.append("Checked daily 'Total' against the sum of 15-minute values: "
                       + ("consistent for every link." if n_bad == 0 else f"{n_bad} link(s) differ (interval values are used for optimisation)."))

    excluded = merged[~merged["eligible_demand"]]
    exclusions = (excluded.groupby(["demand_status"], dropna=False).size().rename("observations").reset_index()
                  .rename(columns={"demand_status": "reason"}))
    svc_excl = merged[merged["eligible_demand"] & ~merged["eligible_service"]]
    if len(svc_excl):
        extra = svc_excl.groupby("frequency_status").size().rename("observations").reset_index()
        extra["reason"] = "frequency " + extra.pop("frequency_status") + " (excluded from Models 2-3 only)"
        exclusions = pd.concat([exclusions, extra], ignore_index=True)
    if len(exclusions):
        actions.append(f"{len(excluded):,} link-interval observation(s) excluded from all optimisation; "
                       f"{len(svc_excl):,} more excluded from the models that need observed frequency.")
    else:
        actions.append("No link-interval observations had to be excluded.")

    stations = pd.unique(pd.concat([merged["from_station"], merged["to_station"]]).dropna())
    n = len(merged)
    metrics = {
        "workbook": wb.name,
        "sheets": wb.sheet_names,
        "links": int(merged["link_key"].nunique()),
        "lines": int(merged["line"].nunique()),
        "stations": int(len(stations)),
        "intervals": len(intervals),
        "observations": n,
        "eligible_demand": int(merged["eligible_demand"].sum()),
        "eligible_service": int(merged["eligible_service"].sum()),
        "links_only_in_loads": int(merged.loc[merged["frequency_status"] == "no_frequency_record", "link_key"].nunique()),
        "links_only_in_frequencies": int(merged.loc[merged["demand_status"] == "no_demand_record", "link_key"].nunique()),
        "ambiguous_links": len(ambiguous),
        "missing_demand": int((merged["demand_status"] == "missing").sum()),
        "missing_frequency": int((merged["frequency_status"] == "missing").sum()),
        "pct_eligible_service": 100.0 * merged["eligible_service"].sum() / n if n else 0.0,
        "zero_train_obs": int(len(zt)),
        "zero_train_demand": float(zt["demand"].sum()),
        "zero_train_demand_pct": 100.0 * zt["demand"].sum() / total_valid if total_valid else 0.0,
        "zero_train_adjacent_pct": 100.0 * zt.loc[zt["zero_train_adjacent_service"], "demand"].sum() / zt["demand"].sum() if len(zt) else 0.0,
    }
    return Processed(merged, summary, key, intervals, metrics, pd.DataFrame(sheet_stats), actions, warnings, exclusions)


def readiness(proc: Processed | None) -> dict[str, tuple[bool, str]]:
    """Which analyses can run with the processed data (capacity is checked in the UI)."""
    if proc is None:
        msg = "Upload a NUMBAT workbook with a Link_Loads sheet."
        return {k: (False, msg) for k in ["Exploratory analysis", "Model 1", "Model 2", "Model 3"]}
    d, s = proc.metrics["eligible_demand"], proc.metrics["eligible_service"]
    out = {
        "Exploratory analysis": (d > 0, f"{d:,} valid demand observations." if d else "No valid demand observations."),
        "Model 1": (d > 0, f"{d:,} valid demand observations; needs a passenger capacity per train." if d
                    else "No valid demand observations to optimise."),
    }
    msg = (f"{s:,} observations with matched demand and observed frequency (used for the allocation budget and frequency targets)."
           if s else "Needs Link_Frequencies matched to Link_Loads: no observation has both valid demand and frequency.")
    out["Model 2"] = (s > 0, msg)
    out["Model 3"] = (s > 0, msg)
    return out
