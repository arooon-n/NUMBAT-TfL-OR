import math

import numpy as np
import pandas as pd
import pytest

from analysis import run_scenarios, scenario_definitions
from optimization import Settings, run_model

CAP = {"L1": 100.0}


def obs(demand, freq, intervals=None):
    """Synthetic eligible link-interval observations (unit tests of the mathematics only)."""
    n = len(demand)
    intervals = intervals or ["0800-0815"] * n
    return pd.DataFrame({
        "link_id": [f"link{i}" for i in range(n)], "line": "L1", "direction": "NB",
        "from_station": [f"S{i}" for i in range(n)], "to_station": [f"S{i + 1}" for i in range(n)],
        "link_label": [f"S{i} -> S{i + 1}" for i in range(n)],
        "interval": intervals, "interval_order": [0] * n,
        "demand": demand, "frequency": freq,
        "eligible_demand": True, "eligible_service": True,
    })


def test_model1_matches_closed_form():
    d = [0, 1, 79, 80, 81, 799.5, 1600]
    res = run_model("Model 1", obs(d, [1] * len(d)), CAP, Settings(target_utilization=0.8))
    assert res.ok and res.proven_optimal
    expected = [math.ceil(v / 80 - 1e-9) for v in d]
    assert list(res.table.optimized_frequency) == expected
    assert res.kpis["unmet_demand"] == pytest.approx(0)


def test_model1_minimum_frequency_and_service_rule():
    res = run_model("Model 1", obs([10, 10], [3, 0]), CAP,
                    Settings(target_utilization=1, min_frequency=2, min_only_in_service=True))
    assert list(res.table.optimized_frequency) == [2, 1]   # minimum only where trains are observed


def test_model1_infeasible_with_max_frequency():
    res = run_model("Model 1", obs([1000], [5]), CAP, Settings(target_utilization=1, max_frequency=5))
    assert not res.ok and res.status == "Infeasible"
    assert any("maximum frequency" in d for d in res.diagnostics)


def test_model1_infeasible_with_budget():
    res = run_model("Model 1", obs([500, 500], [2, 2]), CAP,
                    Settings(target_utilization=1, use_budget=True, budget_factor=1.0))
    assert not res.ok and "budget" in " ".join(res.diagnostics)


def test_model2_respects_budget_demand_and_capacity():
    d = [300, 250, 40, 10, 0]
    f = [2, 2, 2, 1, 1]  # budget F_t = 8 trains
    res = run_model("Model 2", obs(d, f), CAP, Settings(target_utilization=1.0))
    t = res.table
    assert res.ok
    assert t.optimized_frequency.sum() <= 8
    assert (t.demand_served <= t.demand + 1e-6).all()
    assert (t.demand_served <= 100 * t.optimized_frequency + 1e-6).all()
    # brute-force optimum over all integer allocations within the budget
    best = 0
    for a in range(5):
        for b in range(5):
            for c in range(2):
                for e in range(2):
                    if a + b + c + e <= 8:
                        best = max(best, min(300, 100 * a) + min(250, 100 * b) + min(40, 100 * c) + min(10, 100 * e))
    assert t.demand_served.sum() == pytest.approx(best)
    # stage 2: no allocation that serves nobody
    assert not ((t.optimized_frequency > 0) & (t.demand_served == 0)).any()


def test_model2_budget_reduction_lowers_service():
    o = obs([300, 300, 300], [3, 3, 3])
    full = run_model("Model 2", o, CAP, Settings(target_utilization=1.0, budget_factor=1.0))
    cut = run_model("Model 2", o, CAP, Settings(target_utilization=1.0, budget_factor=0.5))
    assert full.kpis["satisfaction_pct"] == pytest.approx(100)
    assert cut.kpis["total_optimized_allocations"] <= 4 and cut.kpis["unmet_demand"] > 0


def test_model3_weights_change_the_plan():
    o = obs([100, 400, 50, 0], [6, 6, 6, 6])
    base = dict(target_utilization=0.8)
    pattern = run_model("Model 3", o, CAP, Settings(**base, weights={"efficiency": 0, "demand": 0, "frequency": 10, "utilization": 0}))
    effort = run_model("Model 3", o, CAP, Settings(**base, weights={"efficiency": 10, "demand": 0, "frequency": 0, "utilization": 0}))
    demand = run_model("Model 3", o, CAP, Settings(**base, weights={"efficiency": 0, "demand": 10, "frequency": 0, "utilization": 0}))
    assert all(r.ok for r in (pattern, effort, demand))
    assert list(pattern.table.optimized_frequency) == [6, 6, 6, 6]           # follows the observed pattern
    assert demand.kpis["satisfaction_pct"] == pytest.approx(100)              # rated capacity covers all demand
    assert effort.kpis["total_optimized_allocations"] < pattern.kpis["total_optimized_allocations"]
    assert set(pattern.goals.columns) >= {"goal", "weight", "normalised deviation - optimised plan"}


def test_model3_budget_enforced():
    o = obs([400, 400], [4, 4])
    res = run_model("Model 3", o, CAP, Settings(target_utilization=1.0, use_budget=True, budget_factor=0.5,
                                                 weights={"efficiency": 0, "demand": 1, "frequency": 0, "utilization": 0}))
    assert res.ok and res.table.optimized_frequency.sum() <= 4


def test_invalid_parameters_are_rejected():
    res = run_model("Model 1", obs([10], [1]), CAP, Settings(min_frequency=5, max_frequency=2))
    assert not res.ok and "exceeds" in res.diagnostics[0]
    res = run_model("Model 1", obs([10], [1]), {"L1": None}, Settings())
    assert not res.ok and "capacity" in res.diagnostics[0]


def test_scenario_multipliers():
    o = obs([100.0, 200.0], [1, 1])
    base = Settings(target_utilization=1.0)
    scen = scenario_definitions(10, [0.5], base)
    multipliers = {s["scenario"]: s.get("demand_multiplier", 1.0) for s in scen}
    assert multipliers["B. Demand +10%"] == pytest.approx(1.10)
    assert multipliers["C. Demand +20%"] == pytest.approx(1.20)
    table = run_scenarios("Model 1", o, CAP, base, scen).set_index("scenario")
    assert table.loc["A. Baseline", "total_demand"] == pytest.approx(300)
    assert table.loc["B. Demand +10%", "total_demand"] == pytest.approx(330)
    assert table.loc["C. Demand +20%", "total_demand"] == pytest.approx(360)
    assert table.loc["E. Utilisation 50%", "total_allocations"] == 6     # ceil(100/50) + ceil(200/50)
    assert not table.loc["D. Budget -10%", "feasible"]                    # budget 1.8 -> 1 train < 3 needed


def test_result_table_uses_solution_values():
    res = run_model("Model 1", obs([150, 50], [4, 1]), CAP, Settings(target_utilization=1.0))
    t = res.table
    assert list(t.frequency_change) == [2 - 4, 1 - 1]
    assert np.allclose(t.offered_capacity, 100 * t.optimized_frequency)
    assert all(passed for _, passed, _ in res.checks)
