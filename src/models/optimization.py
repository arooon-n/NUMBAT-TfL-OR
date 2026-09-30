from __future__ import annotations
from dataclasses import dataclass
from math import ceil
from typing import Any
import pandas as pd
from pyomo.environ import (ConcreteModel, Var, NonNegativeReals, NonNegativeIntegers, Objective, Constraint, SolverFactory, maximize, minimize, value)

@dataclass
class SolveResult:
    frame: pd.DataFrame
    status: str
    termination: str


def _solver():
    for name in ("highs", "appsi_highs"):
        solver = SolverFactory(name)
        if solver is not None and solver.available(exception_flag=False):
            return solver
    raise RuntimeError("HiGHS is unavailable. Install requirements.txt and confirm highspy is installed.")


def _inputs(frame: pd.DataFrame, capacity: pd.DataFrame, constraints: pd.DataFrame, multiplier: float = 1.0):
    required = {"line", "time_period", "link_load", "link_frequency"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Optimisation requires canonical columns: {sorted(missing)}")
    work = frame.copy()
    work["service_group"] = work.get("service_group", work["line"])
    cap = capacity.copy()
    if cap.empty or "capacity_per_train" not in cap:
        raise ValueError("No verified capacity_by_line.csv values are configured.")
    work = work.merge(cap[[c for c in ["line", "service_group", "capacity_per_train"] if c in cap]], on=[c for c in ["line", "service_group"] if c in cap.columns], how="left")
    if work["capacity_per_train"].isna().any():
        raise ValueError("Capacity is missing for one or more lines/service groups.")
    con = constraints.copy()
    if con.empty:
        raise ValueError("No service constraints are configured.")
    for col, default in [("min_frequency_per_15min", 0), ("max_frequency_per_15min", None), ("target_frequency_per_15min", 0)]:
        if col not in con: con[col] = default
    work = work.merge(con[[c for c in ["line", "service_group", "min_frequency_per_15min", "max_frequency_per_15min", "target_frequency_per_15min"] if c in con]], on=[c for c in ["line", "service_group"] if c in con.columns], how="left")
    work["max_frequency_per_15min"] = work["max_frequency_per_15min"].fillna(999999)
    work["demand"] = work["link_load"].astype(float) * multiplier
    return work


def _solve(model: ConcreteModel) -> tuple[str, str]:
    result = _solver().solve(model, tee=False)
    return str(result.solver.status), str(result.solver.termination_condition)


def model1(frame: pd.DataFrame, capacity: pd.DataFrame, constraints: pd.DataFrame, multiplier: float = 1.0, fleet: pd.DataFrame | None = None) -> SolveResult:
    work = _inputs(frame, capacity, constraints, multiplier)
    groups = list(work[["line", "service_group", "time_period"]].drop_duplicates().itertuples(index=False, name=None))
    model = ConcreteModel(); model.K = range(len(groups))
    model.x = Var(model.K, domain=NonNegativeIntegers)
    for k, (_, _, t) in enumerate(groups):
        rows = work[(work.line == groups[k][0]) & (work.service_group == groups[k][1]) & (work.time_period == t)]
        model.add_component(f"lo_{k}", Constraint(expr=model.x[k] >= float(rows.min_frequency_per_15min.iloc[0])))
        model.add_component(f"hi_{k}", Constraint(expr=model.x[k] <= float(rows.max_frequency_per_15min.iloc[0])))
        model.add_component(f"cap_{k}", Constraint(expr=model.x[k] * float(rows.capacity_per_train.max()) >= float(rows.demand.max())))
    if fleet is not None and not fleet.empty:
        budgets = dict(zip(fleet.time_period.astype(str), fleet.fleet_budget))
        for t, budget in budgets.items():
            ids = [k for k, g in enumerate(groups) if str(g[2]) == t]
            if ids: model.add_component(f"budget_{t}", Constraint(expr=sum(model.x[k] for k in ids) <= float(budget)))
    model.obj = Objective(expr=sum(model.x[k] for k in model.K), sense=minimize)
    status, termination = _solve(model)
    rows = []
    for k, (line, group, t) in enumerate(groups):
        subset = work[(work.line == line) & (work.service_group == group) & (work.time_period == t)]
        observed = float(subset.link_frequency.mean())
        optimized = int(round(value(model.x[k]))) if termination in ("optimal", "feasible") else None
        cap = float(subset.capacity_per_train.max()); load = float(subset.demand.max())
        rows.append({"time_period": t, "line": line, "service_group": group, "observed_frequency": observed, "optimized_frequency": optimized, "frequency_change": None if optimized is None else optimized-observed, "max_link_load": load, "capacity_per_train": cap, "baseline_utilization": load/(cap*observed) if observed else None, "optimized_utilization": load/(cap*optimized) if optimized else None, "demand_satisfaction": None if optimized is None else min(1.0, cap*optimized/load) if load else 1.0, "solver_status": status, "termination": termination})
    return SolveResult(pd.DataFrame(rows), status, termination)


def model2(frame: pd.DataFrame, capacity: pd.DataFrame, constraints: pd.DataFrame, fleet: pd.DataFrame, multiplier: float = 1.0) -> SolveResult:
    work = _inputs(frame, capacity, constraints, multiplier)
    if "line_boarders" not in work: raise ValueError("Model 2 requires NUMBAT line_boarders.")
    work["boarders"] = work["line_boarders"].fillna(0).astype(float)
    grouped = work.groupby(["line", "service_group", "time_period"], as_index=False).agg(demand=("boarders", "max"), capacity=("capacity_per_train", "max"), min_frequency=("min_frequency_per_15min", "max"), max_frequency=("max_frequency_per_15min", "min"))
    budgets = {} if fleet is None or fleet.empty else dict(zip(fleet.time_period.astype(str), fleet.fleet_budget))
    keys = list(grouped.itertuples(index=False, name=None)); model = ConcreteModel(); model.K = range(len(keys)); model.x = Var(model.K, domain=NonNegativeIntegers); model.y = Var(model.K, domain=NonNegativeReals)
    for k, row in enumerate(keys):
        model.add_component(f"lo_{k}", Constraint(expr=model.x[k] >= row[5])); model.add_component(f"hi_{k}", Constraint(expr=model.x[k] <= row[6])); model.add_component(f"demand_{k}", Constraint(expr=model.y[k] <= row[3])); model.add_component(f"supply_{k}", Constraint(expr=model.y[k] <= row[4]*model.x[k]))
    for t, budget in budgets.items():
        ids = [k for k, row in enumerate(keys) if str(row[5]) == t]
        if ids: model.add_component(f"budget_{t}", Constraint(expr=sum(model.x[k] for k in ids) <= float(budget)))
    model.obj = Objective(expr=sum(model.y[k] for k in model.K), sense=maximize); status, termination = _solve(model)
    out=[]
    for k, row in enumerate(keys):
        x = int(round(value(model.x[k]))) if termination in ("optimal", "feasible") else None; served = float(value(model.y[k])) if x is not None else None; demand=float(row[3]); cap=float(row[4])
        out.append({"time_period": row[5], "line": row[0], "service_group": row[1], "demand": demand, "trains_allocated": x, "capacity": cap, "demand_served": served, "unmet_demand": None if served is None else demand-served, "coverage_ratio": None if served is None or not demand else served/demand, "utilization": None if served is None or not x else served/(cap*x), "solver_status": status, "termination": termination})
    return SolveResult(pd.DataFrame(out), status, termination)


def analytical_required_frequency(frame: pd.DataFrame, capacity: pd.DataFrame, multiplier: float = 1.0) -> pd.DataFrame:
    work = _inputs(frame, capacity, pd.DataFrame({"line": frame.line.unique(), "service_group": frame.get("service_group", frame.line), "min_frequency_per_15min": 0, "max_frequency_per_15min": 999999}), multiplier)
    return work.groupby(["line", "service_group", "time_period"], as_index=False).agg(required_frequency=("demand", lambda x: ceil(x.max() / work.loc[x.index, "capacity_per_train"].max())))
