"""Descriptive analysis, result interpretation and scenario analysis."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from optimization import Result, Settings, run_model


def filter_scope(data: pd.DataFrame, lines: list[str], directions: list[str] | None,
                 first: int, last: int) -> pd.DataFrame:
    """Selected lines/directions and an inclusive range of interval positions (0 = 05:00)."""
    mask = data["line"].isin(lines) & data["interval_order"].between(first, last)
    if directions:
        mask &= data["direction"].isin(directions)
    return data[mask]


def time_profile(scope: pd.DataFrame) -> pd.DataFrame:
    valid = scope[scope["eligible_demand"]]
    return (valid.groupby(["interval_order", "interval", "interval_start"], as_index=False)
            .agg(demand=("demand", "sum"), frequency=("frequency", "sum"))
            .sort_values("interval_order"))


def link_totals(scope: pd.DataFrame) -> pd.DataFrame:
    valid = scope[scope["eligible_demand"]]
    out = (valid.groupby(["link_id", "link_label", "line", "from_station", "to_station"], as_index=False)
           .agg(demand=("demand", "sum"), peak_demand=("demand", "max"), frequency=("frequency", "sum")))
    out["display"] = out["link_label"] + " [" + out["line"] + "]"
    return out


def with_observed_capacity(scope: pd.DataFrame, capacity: dict, rho: float) -> pd.DataFrame:
    """Observed effective capacity per link-interval = observed trains x C x rho (needs a capacity assumption)."""
    obs = scope[scope["eligible_service"]].copy()
    obs["C"] = obs["line"].map(capacity)
    obs = obs[obs["C"] > 0]
    obs["offered"] = obs["frequency"] * np.floor(obs["C"] * rho + 1e-9)  # whole passengers, as in the models
    with np.errstate(divide="ignore", invalid="ignore"):
        obs["load_factor"] = np.where(obs["frequency"] > 0, obs["demand"] / (obs["frequency"] * obs["C"]), np.nan)
    return obs


def mismatch_tables(obs: pd.DataFrame, rho: float, low: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Over-capacity link-intervals (demand > observed effective capacity) and under-used ones (load < `low`).
    Link-intervals with passengers but no scheduled train are not capacity mismatches but data inconsistencies;
    they are reported separately in the Data Quality tab."""
    cols = ["line", "link_label", "interval", "demand", "frequency", "offered", "load_factor"]
    served = obs[obs["frequency"] > 0]
    over = served[served["demand"] > served["offered"]].assign(excess=lambda d: d["demand"] - d["offered"])
    over = over.sort_values("excess", ascending=False)[cols + ["excess"]]
    under = obs[(obs["frequency"] > 0) & (obs["load_factor"] < low)].sort_values("load_factor")[cols]
    return over, under


def interpret(res: Result) -> list[str]:
    """Plain-language findings generated only from the computed result."""
    if not res.ok:
        return []
    t, k = res.table, res.kpis
    lines = []
    if k.get("total_existing_allocations"):
        change = k["allocation_change"]
        pct = 100 * change / k["total_existing_allocations"]
        lines.append(f"The plan uses **{k['total_optimized_allocations']:,}** link-train allocations versus "
                     f"**{k['total_existing_allocations']:,.0f}** observed ({pct:+.1f}%).")
    lines.append(f"It serves **{k['satisfaction_pct']:.2f}%** of modelled demand "
                 f"({k['unmet_demand']:,.0f} passengers unmet across {k['observations']:,} link-intervals).")
    if k["existing_observations_with_shortfall"]:
        lines.append(f"Under the same capacity assumptions the observed timetable leaves "
                     f"{k['existing_observations_with_shortfall']:,} link-interval(s) above effective capacity; "
                     f"the plan leaves {k['observations_with_shortfall']:,}.")
    unmet = t[t["unmet_demand"] > 0.5]
    if len(unmet):
        worst = unmet.groupby("link_label")["unmet_demand"].sum().nlargest(3)
        lines.append("Largest unmet demand: " + "; ".join(f"{name} ({v:,.0f})" for name, v in worst.items()) + ".")
    by_link = t.groupby("link_label")["frequency_change"].sum()
    if by_link.notna().any():
        cut, add = by_link.nsmallest(1), by_link.nlargest(1)
        if cut.iloc[0] < 0:
            lines.append(f"Largest reduction: {cut.index[0]} ({cut.iloc[0]:+,.0f} allocations over the scope).")
        if add.iloc[0] > 0:
            lines.append(f"Largest increase: {add.index[0]} ({add.iloc[0]:+,.0f} allocations over the scope).")
    if "binding_intervals" in k:
        lines.append(f"The allocation budget limits service (exhausted with demand left unmet) in "
                     f"{k['binding_intervals']} of {len(res.budget)} intervals.")
    if res.settings and res.settings.min_frequency == 0 and k.get("allocation_change", 0) < 0:
        lines.append("Note: with a minimum frequency of 0, the models size service to passenger volume only; "
                     "they do not include waiting-time or headway standards, which is why observed service is higher.")
    return lines


def scenario_definitions(budget_cut_pct: float, alt_utilizations: list[float], base: Settings) -> list[dict]:
    scen = [
        {"scenario": "A. Baseline", "demand_multiplier": 1.0},
        {"scenario": "B. Demand +10%", "demand_multiplier": 1.10},
        {"scenario": "C. Demand +20%", "demand_multiplier": 1.20},
        {"scenario": f"D. Budget -{budget_cut_pct:g}%", "budget_factor": base.budget_factor * (1 - budget_cut_pct / 100),
         "use_budget": True},
    ]
    for rho in alt_utilizations:
        scen.append({"scenario": f"E. Utilisation {rho:.0%}", "target_utilization": rho})
    return scen


def run_scenarios(model: str, obs: pd.DataFrame, capacity: dict, base: Settings, scenarios: list[dict],
                  progress=None) -> pd.DataFrame:
    rows = []
    for n, sc in enumerate(scenarios):
        params = {k: v for k, v in sc.items() if k != "scenario"}
        s = replace(base, **params)
        res = run_model(model, obs, capacity, s)
        k = res.kpis
        rows.append({
            "scenario": sc["scenario"], "feasible": res.ok, "status": res.status,
            "demand_multiplier": s.demand_multiplier, "target_utilization": s.target_utilization,
            "budget": f"{s.budget_factor:.0%} of observed" if (s.use_budget or model == "Model 2") else "not applied",
            "total_allocations": k.get("total_optimized_allocations") if res.ok else None,
            "observed_allocations": k.get("total_existing_allocations") if res.ok else None,
            "total_demand": k.get("total_demand") if res.ok else None,
            "demand_served": k.get("demand_served") if res.ok else None,
            "unmet_demand": k.get("unmet_demand") if res.ok else None,
            "satisfaction_pct": k.get("satisfaction_pct") if res.ok else None,
            "avg_utilization_pct": k.get("avg_utilization_pct") if res.ok else None,
            "under_capacity_links": k.get("links_with_shortfall") if res.ok else None,
            "notes": " ".join(res.diagnostics)[:300],
        })
        if progress:
            progress((n + 1) / len(scenarios), sc["scenario"])
    return pd.DataFrame(rows)
