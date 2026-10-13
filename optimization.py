"""The three optimisation models (Pyomo + HiGHS).

Index i = one (link, 15-minute interval) observation; t = an interval.
D_i  passenger demand on the link in the interval (Link_Loads)
f_i  observed trains on the link in the interval (Link_Frequencies)
C_i  rated passenger capacity per train (user assumption, per line)
rho  target utilisation, so C_i^eff = rho * C_i
x_i  train allocations (integer decision), y_i passengers served (continuous)
F_t  link-service allocation budget for interval t (beta * sum_{i in t} f_i)
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import SolutionStatus, TerminationCondition

MODEL_NAMES = {
    "Model 1": "Minimum train frequency (integer programming)",
    "Model 2": "Capacity-constrained service allocation (MILP)",
    "Model 3": "Multi-objective service plan (weighted goal programming)",
}
GOALS = {
    "efficiency": "Operational efficiency (fewer train allocations)",
    "demand": "Demand satisfaction (less unmet demand)",
    "frequency": "Service pattern (stay close to observed frequency)",
    "utilization": "Target utilisation (load close to rho x capacity)",
}
SOLVER_HELP = ("The HiGHS solver is not available to Pyomo. Install it with "
               "`python -m pip install highspy pyomo` (both are in requirements.txt) and restart the app.")
TOL = 1e-6
# Pyomo's HiGHS interface redirects the process-wide stdout/stderr during every solve. Streamlit runs each
# rerun in its own thread, so overlapping solves (e.g. a widget changed mid-sensitivity run) deadlock Pyomo's
# capture_output. Solves are therefore serialised: a new run waits for the previous solve to finish.
_SOLVE_LOCK = threading.Lock()


class SolverUnavailable(RuntimeError):
    pass


@dataclass
class Settings:
    target_utilization: float = 0.85          # rho, 0 < rho <= 1
    min_frequency: int = 0                    # trains per link per 15 min
    max_frequency: int | None = None          # optional hard upper bound
    min_only_in_service: bool = True          # apply the minimum only where trains are observed
    use_budget: bool = False                  # Model 1 / Model 3: optional; Model 2: always on
    budget_factor: float = 1.0                # F_t = floor(beta * observed allocations in t)
    demand_multiplier: float = 1.0            # scenario multiplier on D_i
    zero_train_policy: str = "exclude"        # demand where 0 trains are observed: "exclude" or "include"
    weights: dict = field(default_factory=lambda: {g: 1.0 for g in GOALS})
    time_limit: float = 120.0
    mip_gap: float = 1e-4

    def validate(self) -> list[str]:
        errors = []
        if not 0 < self.target_utilization <= 1:
            errors.append("Target utilisation must be greater than 0% and at most 100%.")
        if self.min_frequency < 0:
            errors.append("Minimum frequency cannot be negative.")
        if self.max_frequency is not None and self.max_frequency < self.min_frequency:
            errors.append(f"Minimum frequency ({self.min_frequency}) exceeds maximum frequency ({self.max_frequency}).")
        if self.budget_factor <= 0:
            errors.append("The allocation budget must be a positive percentage of the observed baseline.")
        if self.demand_multiplier < 0:
            errors.append("Demand multiplier cannot be negative.")
        if self.zero_train_policy not in ("exclude", "include"):
            errors.append("Unknown policy for demand on link-intervals with zero observed trains.")
        if any(w < 0 for w in self.weights.values()):
            errors.append("Goal weights cannot be negative.")
        return errors


@dataclass
class Result:
    model: str
    ok: bool                      # a validated, usable solution exists
    status: str                   # human-readable solver outcome
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    kpis: dict = field(default_factory=dict)
    budget: pd.DataFrame = field(default_factory=pd.DataFrame)
    goals: pd.DataFrame = field(default_factory=pd.DataFrame)
    checks: list = field(default_factory=list)
    diagnostics: list = field(default_factory=list)
    settings: Settings | None = None
    proven_optimal: bool = False
    gap: float | None = None


# ---------------------------------------------------------------- inputs

def prepare(obs: pd.DataFrame, capacity: dict[str, float], s: Settings, need_frequency: bool) -> pd.DataFrame:
    """Attach demand, capacity and bounds to each eligible link-interval observation."""
    mask = obs["eligible_service"] if need_frequency else obs["eligible_demand"]
    # Passengers reported where the timetable has 0 trains contradict the frequency sheet (first-review concern).
    # Default: exclude them from optimisation and disclose how much demand that removes.
    zero = mask & (obs["frequency"] == 0) & (obs["demand"] > 0)
    if s.zero_train_policy == "exclude":
        mask = mask & ~zero
    w = obs.loc[mask].copy()
    w.attrs["zero_train"] = {"policy": s.zero_train_policy, "observations": int(zero.sum()),
                             "demand": int(whole_passengers(obs.loc[zero, "demand"], s.demand_multiplier).sum())}
    missing = sorted(set(w["line"]) - {k for k, v in capacity.items() if v and v > 0})
    if missing:
        raise ValueError(f"No positive passenger capacity configured for line(s): {missing}.")
    cap = w["line"].map(capacity).astype(float)
    fractional = sorted(set(w.loc[cap % 1 != 0, "line"]))
    if fractional:
        raise ValueError(f"Passenger capacity per train must be a whole number of passengers for line(s): {fractional}.")
    # Passengers and trains are whole units. NUMBAT link loads are modelled estimates with fractional values,
    # so demand is rounded to the nearest whole passenger after the scenario multiplier.
    w["D"] = whole_passengers(w["demand"], s.demand_multiplier)
    w["C"] = cap.astype(np.int64)
    # Rounded down: a train cannot carry part of a passenger, and rounding up would exceed the target utilisation.
    w["C_eff"] = np.floor(w["C"] * s.target_utilization + 1e-9).astype(np.int64)
    too_small = sorted(set(w.loc[w["C_eff"] < 1, "line"]))
    if too_small:
        raise ValueError(f"Effective capacity rho x C is below one passenger per train for line(s): {too_small}.")
    w["f_obs"] = w["frequency"].astype(float)
    in_service = w["f_obs"].fillna(1) > 0 if s.min_only_in_service else True
    w["lb"] = np.where(in_service, s.min_frequency, 0).astype(np.int64)
    w["ub_user"] = np.inf if s.max_frequency is None else float(s.max_frequency)
    w["need"] = np.maximum(w["lb"], -(-w["D"] // w["C_eff"])).astype(np.int64)  # exact integer ceil(D / C_eff)
    return w.reset_index(drop=True)


def whole_passengers(demand: pd.Series, multiplier: float = 1.0) -> pd.Series:
    """Demand in whole passengers: nearest integer (halves round up) after the scenario multiplier."""
    return np.floor(demand.astype(float) * multiplier + 0.5 + 1e-9).astype(np.int64)


def interval_budgets(w: pd.DataFrame, factor: float) -> pd.Series:
    """F_t = floor(beta * sum of observed link frequencies in interval t)."""
    return np.floor(w.groupby("interval")["f_obs"].sum() * factor + 1e-9).astype(np.int64)


def _budget_diagnostics(w: pd.DataFrame, budgets: pd.Series, column: str, what: str) -> list[str]:
    need = w.groupby("interval")[column].sum()
    short = (need - budgets).loc[lambda d: d > 0]
    if short.empty:
        return []
    worst = ", ".join(f"{t} (needs {need[t]}, budget {budgets[t]})" for t in short.sort_values(ascending=False).index[:5])
    return [f"The allocation budget is below the {what} in {len(short)} interval(s), e.g. {worst}. "
            "Increase the budget percentage, lower the minimum frequency, or relax the capacity assumptions."]


# ---------------------------------------------------------------- solver

def solver_available() -> bool:
    try:
        return bool(SolverFactory("highs").available())
    except Exception:
        return False


def _solve(model: pyo.ConcreteModel, s: Settings) -> dict:
    try:
        solver = SolverFactory("highs")
        if not solver.available():
            raise SolverUnavailable(SOLVER_HELP)
    except SolverUnavailable:
        raise
    except Exception as exc:
        raise SolverUnavailable(f"{SOLVER_HELP} ({exc})") from exc
    with _SOLVE_LOCK:
        res = solver.solve(model, time_limit=s.time_limit, load_solutions=False,
                           raise_exception_on_nonoptimal_result=False,
                           solver_options={"mip_rel_gap": s.mip_gap})
        has_solution = res.solution_status in (SolutionStatus.optimal, SolutionStatus.feasible)
        if has_solution:
            res.solution_loader.load_vars()
    optimal = res.termination_condition == TerminationCondition.convergenceCriteriaSatisfied
    gap = None
    if res.incumbent_objective is not None and res.objective_bound is not None:
        gap = abs(res.incumbent_objective - res.objective_bound) / max(1e-9, abs(res.incumbent_objective))
    tc = res.termination_condition
    if optimal and gap is not None and gap > max(10 * s.mip_gap, 1e-3) and abs(res.incumbent_objective or 0) > 1e-6:
        optimal = False  # never claim optimality when the primal and dual bounds disagree
        text = f"Solver reported optimal but its bounds disagree (gap {gap:.1%}); treat this solution with caution"
    elif optimal:
        text = "Optimal"
    elif tc in (TerminationCondition.provenInfeasible, TerminationCondition.locallyInfeasible,
                TerminationCondition.infeasibleOrUnbounded):
        text = "Infeasible"
    elif tc == TerminationCondition.maxTimeLimit:
        text = "Time limit reached" + (" with a feasible (not proven optimal) solution" if has_solution else " without any feasible solution")
    else:
        text = f"Solver stopped: {tc.name}"
    return {"has_solution": has_solution, "optimal": optimal, "text": text, "gap": gap,
            "objective": res.incumbent_objective}


def _x_bounds(w: pd.DataFrame, tech_ub: np.ndarray | None):
    ub = w["ub_user"].to_numpy(dtype=float)
    if tech_ub is not None:
        ub = np.minimum(ub, np.maximum(tech_ub, w["lb"].to_numpy()))
    lb = w["lb"].to_numpy()
    return {i: (int(lb[i]), None if math.isinf(ub[i]) else int(ub[i])) for i in range(len(w))}


def _base_model(w: pd.DataFrame, x_bounds: dict, budgets: pd.Series | None) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()
    m.I = pyo.RangeSet(0, len(w) - 1)
    m.x = pyo.Var(m.I, within=pyo.NonNegativeIntegers, bounds=lambda _, i: x_bounds[i])
    if budgets is not None:
        groups = w.groupby("interval").indices
        m.T = pyo.Set(initialize=list(groups), ordered=True)
        m.budget = pyo.Constraint(m.T, rule=lambda m, t: sum(m.x[i] for i in groups[t]) <= int(budgets[t]))
    return m


def _values(var, n) -> np.ndarray:
    return np.array([pyo.value(var[i]) for i in range(n)], dtype=float)


# ---------------------------------------------------------------- models

def model1(obs: pd.DataFrame, capacity: dict, s: Settings) -> Result:
    """min sum x_i  s.t.  C_eff_i x_i >= D_i,  lb_i <= x_i <= ub_i,  [sum_{i in t} x_i <= F_t]."""
    w = prepare(obs, capacity, s, need_frequency=s.use_budget)
    if w.empty:
        return Result("Model 1", False, "No eligible observations in the selected scope.", settings=s)
    budgets = interval_budgets(w, s.budget_factor) if s.use_budget else None
    diag = []
    over = w[w["need"] > w["ub_user"]]
    if len(over):
        ex = over.nlargest(3, "need")
        diag.append(f"{len(over)} observation(s) need more trains than the maximum frequency ({s.max_frequency}), e.g. "
                    + "; ".join(f"{r.link_label} {r.interval}: needs {r.need}" for r in ex.itertuples())
                    + ". Raise the maximum frequency, the capacity per train, or the target utilisation.")
    if budgets is not None:
        diag += _budget_diagnostics(w, budgets, "need", "minimum trains required to carry demand")

    m = _base_model(w, _x_bounds(w, None), budgets)
    D, Ce = w["D"].to_numpy(), w["C_eff"].to_numpy()
    m.capacity = pyo.Constraint(m.I, rule=lambda m, i: Ce[i] * m.x[i] >= D[i])
    m.obj = pyo.Objective(expr=sum(m.x[i] for i in m.I), sense=pyo.minimize)
    sol = _solve(m, s)
    if not sol["has_solution"]:
        return Result("Model 1", False, sol["text"], diagnostics=diag or ["The solver found no solution satisfying every constraint."], settings=s)
    x = np.round(_values(m.x, len(w)))
    return _finish("Model 1", w, x, np.minimum(w["D"], Ce * x), budgets, sol, s, diag)


def model2(obs: pd.DataFrame, capacity: dict, s: Settings) -> Result:
    """Stage 1: max sum y_i  s.t.  y_i <= D_i,  y_i <= C_eff_i x_i,  sum_{i in t} x_i <= F_t,  bounds on x.
    Stage 2: min sum x_i with sum y_i fixed at the stage-1 optimum (removes useless allocations)."""
    w = prepare(obs, capacity, s, need_frequency=True)
    if w.empty:
        return Result("Model 2", False, "No observations with both valid demand and observed frequency in the scope.", settings=s)
    budgets = interval_budgets(w, s.budget_factor)
    diag = _budget_diagnostics(w, budgets, "lb", "sum of minimum frequencies")
    D, Ce = w["D"].to_numpy(), w["C_eff"].to_numpy()
    tech_ub = -(-D // Ce)  # more trains than ceil(D / C_eff) cannot carry more passengers
    m = _base_model(w, _x_bounds(w, tech_ub), budgets)
    m.y = pyo.Var(m.I, within=pyo.NonNegativeIntegers, bounds=lambda _, i: (0, int(D[i])))
    m.served_cap = pyo.Constraint(m.I, rule=lambda m, i: m.y[i] <= Ce[i] * m.x[i])
    m.obj = pyo.Objective(expr=sum(m.y[i] for i in m.I), sense=pyo.maximize)
    sol1 = _solve(m, s)
    if not sol1["has_solution"]:
        return Result("Model 2", False, sol1["text"], diagnostics=diag or ["The solver found no solution satisfying every constraint."], settings=s)
    best = int(round(sum(_values(m.y, len(w)))))
    # Stage 2: keep the same served demand (whole passengers, so exactly), use as few allocations as possible.
    m.obj.deactivate()
    m.keep_served = pyo.Constraint(expr=sum(m.y[i] for i in m.I) >= best)
    m.obj2 = pyo.Objective(expr=sum(m.x[i] for i in m.I), sense=pyo.minimize)
    sol2 = _solve(m, s)
    if not sol2["has_solution"]:  # fall back to stage-1 solution is impossible after re-solve; report honestly
        return Result("Model 2", False, "Stage 2 (removing unused allocations) failed: " + sol2["text"], diagnostics=diag, settings=s)
    x = np.round(_values(m.x, len(w)))
    y = np.minimum(_values(m.y, len(w)), np.minimum(D, Ce * x))
    sol = {**sol2, "optimal": sol1["optimal"] and sol2["optimal"],
           "text": "Optimal" if sol1["optimal"] and sol2["optimal"] else f"Stage 1: {sol1['text']}; stage 2: {sol2['text']}",
           "gap": sol1["gap"]}
    return _finish("Model 2", w, x, y, budgets, sol, s, diag)


def model3(obs: pd.DataFrame, capacity: dict, s: Settings) -> Result:
    """Weighted goal programming.

    Hard: y_i <= D_i,  y_i <= C_i x_i (rated capacity),  bounds on x,  [budget].
    Goals (deviation variables >= 0), each normalised by its own target (percentage normalisation):
      G1 efficiency:  sum x - E = p1 - n1        E  = sum_i max(lb_i, ceil(D_i / C_eff_i)), least effort carrying all demand
      G2 demand:      unmet = sum (D_i - y_i)    D0 = sum D_i
      G3 frequency:   x_i - f_i = p3_i - n3_i    F0 = sum f_i (observed allocations)
      G4 utilisation: D_i - C_eff_i x_i = p4_i - n4_i  (p4: load above target, n4: unused target capacity)
    min  w1 p1/E + w2 unmet/D0 + w3 sum(p3 + n3)/F0 + w4 sum(p4 + n4)/D0"""
    w = prepare(obs, capacity, s, need_frequency=True)
    if w.empty:
        return Result("Model 3", False, "No observations with both valid demand and observed frequency in the scope.", settings=s)
    budgets = interval_budgets(w, s.budget_factor) if s.use_budget else None
    diag = _budget_diagnostics(w, budgets, "lb", "sum of minimum frequencies") if budgets is not None else []
    D, C, Ce, f = (w[c].to_numpy() for c in ["D", "C", "C_eff", "f_obs"])
    tech_ub = np.maximum.reduce([-(-D // Ce), f, w["lb"].to_numpy()])  # beyond this every goal worsens
    m = _base_model(w, _x_bounds(w, tech_ub), budgets)
    # y and the deviations are implied integers, so they are not declared integer (declaring ~5 x |I| extra integer
    # variables makes the whole network intractable for no gain): x, D, f, C and C_eff are whole numbers and every
    # one of these variables has a positive cost, so for any integer x the best y_i = min(D_i, C_i x_i) and each
    # deviation is a difference of whole trains or passengers. y is set to that value after solving.
    m.y = pyo.Var(m.I, within=pyo.NonNegativeReals, bounds=lambda _, i: (0, int(D[i])))
    m.p3 = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.n3 = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.p4 = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.n4 = pyo.Var(m.I, within=pyo.NonNegativeReals)
    m.served_cap = pyo.Constraint(m.I, rule=lambda m, i: m.y[i] <= C[i] * m.x[i])
    m.freq_goal = pyo.Constraint(m.I, rule=lambda m, i: m.x[i] - f[i] == m.p3[i] - m.n3[i])
    m.util_goal = pyo.Constraint(m.I, rule=lambda m, i: D[i] - Ce[i] * m.x[i] == m.p4[i] - m.n4[i])
    E = max(int(w["need"].sum()), 1)
    F0 = max(f.sum(), 1.0)
    D0 = max(int(D.sum()), 1)
    wt = s.weights
    m.p1 = pyo.Var(within=pyo.NonNegativeReals)
    m.n1 = pyo.Var(within=pyo.NonNegativeReals)
    m.effort_goal = pyo.Constraint(expr=sum(m.x[i] for i in m.I) - E == m.p1 - m.n1)
    m.z_eff = pyo.Expression(expr=m.p1 / E)
    m.z_dem = pyo.Expression(expr=(D.sum() - sum(m.y[i] for i in m.I)) / D0)
    m.z_freq = pyo.Expression(expr=sum(m.p3[i] + m.n3[i] for i in m.I) / F0)
    m.z_util = pyo.Expression(expr=sum(m.p4[i] + m.n4[i] for i in m.I) / D0)
    # A tiny equal weight on every goal breaks ties when a weight is zero, so the plan stays Pareto-efficient.
    eps = 1e-3 * max(max(wt.values()), 1e-9)
    weighted = sum((wt[g] + eps) * z for g, z in
                   [("efficiency", m.z_eff), ("demand", m.z_dem), ("frequency", m.z_freq), ("utilization", m.z_util)])
    # Multiplying by a constant does not change the optimum but keeps cost coefficients well scaled for HiGHS
    # (dividing by millions of passengers otherwise produces costs the presolve treats as zero).
    m.obj = pyo.Objective(expr=max(E, F0, D0) * weighted, sense=pyo.minimize)
    sol = _solve(m, s)
    if not sol["has_solution"]:
        return Result("Model 3", False, sol["text"], diagnostics=diag or ["The solver found no solution satisfying every constraint."], settings=s)
    x = np.round(_values(m.x, len(w)))
    y = np.minimum(D, C * x)  # best whole-passenger service for x (also exact if the solver stopped early)
    res = _finish("Model 3", w, x, y, budgets, sol, s, diag)
    def goal_values(alloc, served):  # the normalised deviations z_k for any allocation
        return {"efficiency": max(alloc.sum() - E, 0) / E, "demand": (D - served).sum() / D0,
                "frequency": np.abs(alloc - f).sum() / F0, "utilization": np.abs(D - Ce * alloc).sum() / D0}
    obs_goal = goal_values(f, np.minimum(D, C * f))  # observed timetable, for comparison
    opt_goal = goal_values(x, y)
    res.kpis["min_effort_E"] = int(E)
    res.goals = pd.DataFrame({
        "goal": [GOALS[g] for g in GOALS],
        "weight": [wt[g] for g in GOALS],
        "normalised deviation - observed service": [obs_goal[g] for g in GOALS],
        "normalised deviation - optimised plan": [opt_goal[g] for g in GOALS],
        "weighted contribution (optimised)": [wt[g] * opt_goal[g] for g in GOALS],
    })
    return res


MODELS = {"Model 1": model1, "Model 2": model2, "Model 3": model3}


def run_model(name: str, obs: pd.DataFrame, capacity: dict, s: Settings) -> Result:
    errors = s.validate()
    if errors:
        return Result(name, False, "Invalid parameters", diagnostics=errors, settings=s)
    try:
        return MODELS[name](obs, capacity, s)
    except ValueError as exc:
        return Result(name, False, "Invalid input", diagnostics=[str(exc)], settings=s)


# ---------------------------------------------------------------- results

def _counts(a: np.ndarray) -> np.ndarray:
    """Whole-number column (trains or passengers) as int64. Columns with missing values (observed frequency
    unknown, Model 1 only) or fractional observed data stay float, so NaN never becomes pandas' pd.NA."""
    a = np.asarray(a, dtype=float)
    return a.astype(np.int64) if not np.isnan(a).any() and np.all(a % 1 == 0) else a


def _finish(name, w, x, y, budgets, sol, s: Settings, diag) -> Result:
    D, C, Ce = (w[c].to_numpy(dtype=np.int64) for c in ["D", "C", "C_eff"])
    f = w["f_obs"].to_numpy(dtype=float)
    checks = validate(name, w, x, y, budgets)  # on the raw solver values, before rounding
    ok = all(c[1] for c in checks)
    x, y = np.round(x).astype(np.int64), np.round(y).astype(np.int64)
    offered = Ce * x
    with np.errstate(divide="ignore", invalid="ignore"):
        util = np.where(C * x > 0, y / (C * x), np.nan)
        util_obs = np.where(C * f > 0, D / (C * f), np.nan)
    table = pd.DataFrame({
        "link_id": w["link_id"], "line": w["line"], "direction": w.get("direction"),
        "from_station": w["from_station"], "to_station": w["to_station"],
        "interval": w["interval"], "interval_order": w["interval_order"],
        "demand": D, "existing_frequency": _counts(f), "optimized_frequency": x,
        "frequency_change": _counts(x - f),
        "capacity_per_train": C, "effective_capacity_per_train": Ce,
        "offered_capacity": offered, "demand_served": y, "unmet_demand": D - y,
        "capacity_shortfall": np.maximum(D - offered, 0),
        "utilization": util,
        "existing_offered_capacity": _counts(Ce * f),
        "existing_shortfall": _counts(np.where(np.isnan(f), np.nan, np.maximum(D - Ce * f, 0))),
        "existing_utilization": util_obs,
    })
    table["link_label"] = w["link_label"]
    budget_table = pd.DataFrame()
    if budgets is not None:
        used = table.groupby("interval")["optimized_frequency"].sum()
        budget_table = pd.DataFrame({"budget": budgets, "allocated": used.reindex(budgets.index).fillna(0).astype(np.int64)})
        budget_table["observed"] = table.groupby("interval")["existing_frequency"].sum().reindex(budgets.index)
        budget_table["unmet_demand"] = table.groupby("interval")["unmet_demand"].sum().reindex(budgets.index)
        # Binding = the budget is exhausted AND demand is left unmet, i.e. the constraint actually limits service
        # (a zero budget in an interval with no demand is at equality but restricts nothing).
        budget_table["binding"] = (budget_table["allocated"] >= budget_table["budget"]) & (budget_table["unmet_demand"] > 1e-6)
        budget_table = budget_table.reset_index().rename(columns={"index": "interval"})
    status = sol["text"] if ok else "Solution failed validation"
    k = kpis(table, budget_table)
    zt = w.attrs.get("zero_train", {})
    k["zero_train_policy"] = zt.get("policy")
    k["zero_train_observations"] = zt.get("observations", 0)
    k["zero_train_demand"] = zt.get("demand", 0.0)
    return Result(name, ok, status, table, k, budget_table, checks=checks,
                  diagnostics=diag, settings=s, proven_optimal=sol["optimal"], gap=sol["gap"])


def validate(name, w, x, y, budgets) -> list[tuple[str, bool, str]]:
    """Independent post-solve checks of the decision variables and key constraints."""
    D, C, Ce, lb, ub = (w[c].to_numpy(dtype=float) for c in ["D", "C", "C_eff", "lb", "ub_user"])
    out = []

    def add(label, mask):
        n = int(np.sum(~mask))
        out.append((label, n == 0, "all observations" if n == 0 else f"{n} violation(s)"))

    add("Train allocations are non-negative integers", (x >= 0) & (np.abs(x - np.round(x)) < TOL))
    add("Minimum frequency respected", x >= lb - TOL)
    add("Maximum frequency respected", x <= ub + TOL)
    add("Served demand is in whole passengers", (y >= -TOL) & (np.abs(y - np.round(y)) < TOL))
    add("Served demand does not exceed demand", y <= D + 1e-6 * np.maximum(D, 1))
    cap = Ce if name != "Model 3" else C
    add("Served demand does not exceed offered capacity", y <= cap * x + 1e-6 * np.maximum(D, 1))
    if name == "Model 1":
        add("Effective capacity covers demand (C_eff x >= D)", Ce * x >= D - 1e-6 * np.maximum(D, 1))
    if budgets is not None:
        used = pd.Series(x).groupby(w["interval"].to_numpy()).sum()
        add("Allocation budget respected in every interval", (used <= budgets.reindex(used.index) + TOL).to_numpy())
    return out


def kpis(t: pd.DataFrame, budget: pd.DataFrame | None = None) -> dict:
    demand = t["demand"].sum()
    served = t["demand_served"].sum()
    rated_offered = (t["capacity_per_train"] * t["optimized_frequency"]).sum()
    existing = t["existing_frequency"].sum(min_count=1)
    count = lambda v: int(v) if float(v) % 1 == 0 else float(v)  # whole trains / passengers as int
    out = {
        "observations": len(t),
        "links": t["link_id"].nunique(),
        "total_demand": int(demand),
        "total_optimized_allocations": int(t["optimized_frequency"].sum()),
        "total_existing_allocations": None if pd.isna(existing) else count(existing),
        "allocation_change": None if pd.isna(existing) else count(t["optimized_frequency"].sum() - existing),
        "demand_served": int(served),
        "satisfaction_pct": 100 * served / demand if demand else 100.0,
        "unmet_demand": int(t["unmet_demand"].sum()),
        "offered_effective_capacity": int(t["offered_capacity"].sum()),
        "avg_utilization_pct": 100 * served / rated_offered if rated_offered else None,
        "observations_with_shortfall": int((t["capacity_shortfall"] > 1e-6).sum()),
        "links_with_shortfall": int(t.loc[t["capacity_shortfall"] > 1e-6, "link_id"].nunique()),
        "existing_observations_with_shortfall": int((t["existing_shortfall"] > 1e-6).sum()),
        "links_more_service": int(t.loc[t["frequency_change"] > 0, "link_id"].nunique()),
        "links_less_service": int(t.loc[t["frequency_change"] < 0, "link_id"].nunique()),
    }
    if budget is not None and len(budget):
        out["binding_intervals"] = int(budget["binding"].sum())
        out["budget_total"] = int(budget["budget"].sum())
    return out
