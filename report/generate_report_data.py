"""Regenerate every number, table and chart dataset used in the IEEE report.

Run from the project root:   python report/generate_report_data.py
Outputs: report/data/*.csv (pgfplots data) and report/data/results.tex (LaTeX macros for numbers in the text).
All values come from the same modules the Streamlit app uses.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analysis as an  # noqa: E402
from data_loader import load_workbook_bytes  # noqa: E402
from optimization import Settings, run_model  # noqa: E402
from preprocessing import preprocess  # noqa: E402

WORKBOOK = ROOT / "data" / "raw" / "NBT25TWT_Outputs.xlsx"
OUT = ROOT / "report" / "data"
CASE_LINE = "Victoria"
RHO = 0.80
macros: dict[str, str] = {}


def num(v, digits=0) -> str:
    """LaTeX-friendly number with thousands separators."""
    return f"{v:,.{digits}f}".replace(",", "{,}")


def tex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%").replace("→", r"$\rightarrow$")


def put(name: str, value) -> None:
    macros[name] = tex_escape(value) if isinstance(value, str) else num(value)


def write_csv(df: pd.DataFrame, name: str) -> None:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].astype(str).map(tex_escape)
    df.to_csv(OUT / name, index=False)


def kpi_row(label: str, res, extra=None) -> dict:
    k = res.kpis
    row = {"case": label, "status": res.status if res.ok else res.status,
           "allocations": k.get("total_optimized_allocations"), "observed": k.get("total_existing_allocations"),
           "satisfaction": k.get("satisfaction_pct"), "unmet": k.get("unmet_demand"),
           "utilisation": k.get("avg_utilization_pct"), "shortfall_links": k.get("links_with_shortfall")}
    row.update(extra or {})
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    proc = preprocess(load_workbook_bytes(WORKBOOK.read_bytes(), WORKBOOK.name))
    data, m = proc.data, proc.metrics
    caps_df = pd.read_csv(ROOT / "config" / "capacity_by_line.csv")
    cap = dict(zip(caps_df["line"], caps_df["capacity_per_train"].astype(float)))
    put("ParseSeconds", num(time.time() - t0, 1))

    # ---------------- dataset and data quality
    for key, name in [("links", "NLinks"), ("lines", "NLines"), ("stations", "NStations"), ("intervals", "NIntervals"),
                      ("observations", "NObs"), ("eligible_demand", "NEligDemand"), ("eligible_service", "NEligService"),
                      ("ambiguous_links", "NAmbiguousLinks"), ("zero_train_obs", "ZeroObs"),
                      ("zero_train_demand", "ZeroDemand")]:
        put(name, m[key])
    put("NExcludedObs", m["observations"] - m["eligible_service"])
    put("PctEligService", num(m["pct_eligible_service"], 1))
    put("ZeroObsPct", num(100 * m["zero_train_obs"] / m["eligible_service"], 2))
    put("ZeroDemandPct", num(m["zero_train_demand_pct"], 2))
    put("ZeroAdjacentPct", num(m["zero_train_adjacent_pct"], 0))
    label_warn = [w for w in proc.warnings if "branch qualifier" in w]
    put("NLabelVariants", sum(int(w.split()[0]) for w in label_warn))
    order_warn = [w for w in proc.warnings if "'Order'" in w]
    put("NOrderDiffs", int(order_warn[0].split()[0]) if order_warn else 0)

    zt = data[data["zero_train_demand"]]
    zl = (zt.groupby("line").agg(obs=("demand", "size"), passengers=("demand", "sum"), largest=("demand", "max"))
          .sort_values("passengers", ascending=False).reset_index())
    write_csv(zl.round(1), "zero_train_by_line.csv")
    zi = zt.groupby(["interval_order", "interval"], as_index=False)["demand"].sum().sort_values("interval_order")
    zi = data.drop_duplicates("interval_order")[["interval_order"]].merge(zi, how="left").fillna({"demand": 0})
    write_csv(zi[["interval_order", "demand"]], "zero_train_by_interval.csv")
    put("ZeroFirstHourPct", num(100 * zt[zt["interval_order"] < 4]["demand"].sum() / zt["demand"].sum(), 0))
    late = zt["interval_order"] >= 76  # 00:00 onwards
    put("ZeroLatePct", num(100 * zt[late]["demand"].sum() / zt["demand"].sum(), 0))
    put("ZeroTopLine", zl.iloc[0]["line"])
    put("ZeroTopLinePass", zl.iloc[0]["passengers"])

    # ---------------- exploratory analysis (network-wide)
    valid = data[data["eligible_service"]]
    prof = an.time_profile(valid)
    prof["hour"] = prof["interval_order"] / 4 + 5
    write_csv(prof[["interval_order", "hour", "demand", "frequency"]].round(1), "network_profile.csv")
    peak = prof.loc[prof["demand"].idxmax()]
    put("NetPeakInterval", f"{peak['interval'][:2]}:{peak['interval'][2:4]}--{peak['interval'][5:7]}:{peak['interval'][7:]}")
    put("NetPeakDemand", peak["demand"])
    put("NetDailyDemand", valid["demand"].sum())
    put("NetDailyTrains", valid["frequency"].sum())
    pm = prof[(prof["hour"] >= 16) & (prof["hour"] < 20)]
    pmpk = pm.loc[pm["demand"].idxmax()]
    put("NetPMPeakInterval", f"{pmpk['interval'][:2]}:{pmpk['interval'][2:4]}")
    put("NetPMPeakDemand", pmpk["demand"])

    periods = [p for p in ["Early", "AM Peak", "Midday", "PM Peak", "Evening", "Late"] if p in proc.link_summary]
    per = proc.link_summary[periods].sum()
    write_csv(pd.DataFrame({"period": periods, "demand": per.values.round(0)}), "period_demand.csv")
    put("PctPeak", num(100 * (per.get("AM Peak", 0) + per.get("PM Peak", 0)) / per.sum(), 1))

    by_line = (valid.groupby("line").agg(demand=("demand", "sum"), trains=("frequency", "sum"), links=("link_id", "nunique"))
               .sort_values("demand", ascending=False).reset_index())
    by_line["pass_per_train"] = by_line["demand"] / by_line["trains"].where(by_line["trains"] > 0)
    write_csv(by_line.round(1), "line_demand.csv")

    lt = an.link_totals(valid)
    top = lt.nlargest(10, "demand")[["line", "from_station", "to_station", "demand", "peak_demand"]]
    write_csv(top.round(0), "top_links.csv")
    put("TopLink", f"{top.iloc[0]['from_station']} $\\rightarrow$ {top.iloc[0]['to_station']} ({top.iloc[0]['line']})")
    put("TopLinkDemand", top.iloc[0]["demand"])

    obs_cap = an.with_observed_capacity(valid, cap, RHO)
    over, under = an.mismatch_tables(obs_cap, RHO)
    put("NOverCap", len(over))
    put("NOverCapLinks", over["link_label"].nunique())
    put("NUnderUsed", len(under))
    put("PctUnderUsed", num(100 * len(under) / (obs_cap["frequency"] > 0).sum(), 1))
    lf = obs_cap[obs_cap["frequency"] > 0]
    lf_line = (lf.assign(over=lf["demand"] > lf["offered"])
               .groupby("line").agg(mean_lf=("load_factor", "mean"), p95=("load_factor", lambda s: s.quantile(0.95)),
                                    max_lf=("load_factor", "max"), over=("over", "sum"))
               .sort_values("max_lf", ascending=False).reset_index())
    lf_line[["mean_lf", "p95", "max_lf"]] *= 100
    write_csv(lf_line.round(1), "load_factor_by_line.csv")
    put("MaxLFLine", lf_line.iloc[0]["line"])
    put("MaxLF", num(lf_line.iloc[0]["max_lf"], 0))
    put("MaxLFLineOver", lf_line.iloc[0]["over"])
    second = lf_line.sort_values("over", ascending=False).iloc[1]
    put("SecondOverLine", second["line"])
    put("SecondOverCount", second["over"])
    put("NLinesWithOver", int((lf_line["over"] > 0).sum()))
    put("CaseMaxLF", num(lf_line.set_index("line").loc[CASE_LINE, "max_lf"], 0))
    worst = over.iloc[0]
    put("WorstOverLink", f"{worst['link_label'].split(' (')[0]} ({worst['line']}, {worst['interval'][:2]}:{worst['interval'][2:4]})")
    put("WorstOverLF", num(100 * worst["load_factor"], 0))

    # ---------------- network-wide Model 1 (both zero-train policies)
    base = Settings(target_utilization=RHO)
    t = time.time(); m1_net = run_model("Model 1", data, cap, base); t_net = time.time() - t
    m1_net_inc = run_model("Model 1", data, cap, replace(base, zero_train_policy="include"))
    put("NetMOneAlloc", m1_net.kpis["total_optimized_allocations"])
    put("NetMOneObserved", m1_net.kpis["total_existing_allocations"])
    put("NetMOneReductionPct", num(-100 * m1_net.kpis["allocation_change"] / m1_net.kpis["total_existing_allocations"], 1))
    put("NetMOneAllocInclude", m1_net_inc.kpis["total_optimized_allocations"])
    put("NetZeroExtraAlloc", m1_net_inc.kpis["total_optimized_allocations"] - m1_net.kpis["total_optimized_allocations"])
    put("NetMOneSeconds", num(t_net, 1))
    put("NetMOneObs", m1_net.kpis["observations"])
    req = (m1_net.table.groupby("line").agg(observed=("existing_frequency", "sum"), required=("optimized_frequency", "sum"),
                                            peak_req=("optimized_frequency", "max"), peak_obs=("existing_frequency", "max"))
           .reset_index())
    req["ratio"] = req["required"] / req["observed"]
    req = req.sort_values("ratio", ascending=False)
    write_csv(req.round(3), "model1_by_line.csv")
    served_lines = req[req["observed"] > 0]  # lines with no scheduled service (London Trams here) have no ratio
    put("HighestRatioLine", served_lines.iloc[0]["line"])
    put("HighestRatio", num(100 * served_lines.iloc[0]["ratio"], 0))
    put("LowestRatioLine", served_lines.iloc[-1]["line"])
    put("LowestRatio", num(100 * served_lines.iloc[-1]["ratio"], 0))
    no_service = by_line[by_line["trains"] == 0]
    put("NoServiceLines", ", ".join(no_service["line"]) if len(no_service) else "none")

    # ---------------- Victoria line case study
    vic = data[data["line"] == CASE_LINE]
    put("CaseLine", CASE_LINE)
    put("CaseCap", cap[CASE_LINE])
    put("CaseLinks", vic["link_id"].nunique())
    put("CaseObs", int(vic["eligible_service"].sum()))
    put("CaseDemand", vic.loc[vic["eligible_service"], "demand"].sum())
    put("CaseObserved", vic.loc[vic["eligible_service"], "frequency"].sum())
    put("CasePeakFreq", vic["frequency"].max())
    vp = an.time_profile(vic[vic["eligible_service"]])
    vpk = vp.loc[vp["demand"].idxmax()]
    put("CasePeakInterval", f"{vpk['interval'][:2]}:{vpk['interval'][2:4]}")
    vlt = an.link_totals(vic[vic["eligible_service"]]).nlargest(1, "peak_demand").iloc[0]
    put("CasePeakLink", f"{vlt['from_station']} $\\rightarrow$ {vlt['to_station']}")
    put("CasePeakLinkDemand", vlt["peak_demand"])

    rows, timing = [], []
    results = {}
    for label, model, s in [
        ("M1, min 0", "Model 1", base),
        ("M1, min 2", "Model 1", replace(base, min_frequency=2)),
        ("M1, min 4", "Model 1", replace(base, min_frequency=4)),
        ("M1, zero-train incl.", "Model 1", replace(base, zero_train_policy="include")),
        ("M2, budget 100%", "Model 2", base),
        ("M2, budget 25%", "Model 2", replace(base, budget_factor=0.25)),
        ("M3, equal weights", "Model 3", base),
    ]:
        t = time.time(); res = run_model(model, vic, cap, s); dt = time.time() - t
        results[label] = res
        rows.append(kpi_row(label, res, {"seconds": round(dt, 2)}))
        timing.append({"case": label, "seconds": dt})
    case = pd.DataFrame(rows)
    write_csv(case.round(2), "case_models.csv")
    r = results
    put("CaseMOneAlloc", r["M1, min 0"].kpis["total_optimized_allocations"])
    put("CaseMOneRed", num(-100 * r["M1, min 0"].kpis["allocation_change"] / r["M1, min 0"].kpis["total_existing_allocations"], 1))
    put("CaseMOneUtil", num(r["M1, min 0"].kpis["avg_utilization_pct"], 1))
    put("CaseMOneMinTwo", r["M1, min 2"].kpis["total_optimized_allocations"])
    put("CaseMOneMinFour", r["M1, min 4"].kpis["total_optimized_allocations"])
    put("CaseMOneMinFourUtil", num(r["M1, min 4"].kpis["avg_utilization_pct"], 1))
    put("CaseMOneExistingShort", r["M1, min 0"].kpis["existing_observations_with_shortfall"])
    put("CaseMTwoSat", num(r["M2, budget 100%"].kpis["satisfaction_pct"], 3))
    put("CaseMTwoUnmet", num(r["M2, budget 100%"].kpis["unmet_demand"], 0))
    put("CaseMTwoAlloc", r["M2, budget 100%"].kpis["total_optimized_allocations"])
    put("CaseMTwoBinding", r["M2, budget 100%"].kpis.get("binding_intervals", 0))
    put("CaseMTwoQSat", num(r["M2, budget 25%"].kpis["satisfaction_pct"], 2))
    put("CaseMTwoQBinding", r["M2, budget 25%"].kpis.get("binding_intervals", 0))
    put("CaseMThreeAlloc", r["M3, equal weights"].kpis["total_optimized_allocations"])
    put("CaseMThreeSat", num(r["M3, equal weights"].kpis["satisfaction_pct"], 2))
    put("CaseMThreeE", r["M3, equal weights"].kpis["min_effort_E"])
    put("CaseMTwoInclUnmet", num(run_model("Model 2", vic, cap, replace(base, zero_train_policy="include")).kpis["unmet_demand"], 0))

    # before/after per interval (observed vs Model 1 min 0 vs min 2 vs Model 3)
    def per_t(res, col="optimized_frequency"):
        return res.table.groupby("interval_order")[col].sum()
    ba = pd.DataFrame({"observed": per_t(r["M1, min 0"], "existing_frequency"), "m1": per_t(r["M1, min 0"]),
                       "m1min2": per_t(r["M1, min 2"]), "m3": per_t(r["M3, equal weights"]),
                       "demand": per_t(r["M1, min 0"], "demand") / 1000}).reset_index()
    ba["hour"] = ba["interval_order"] / 4 + 5
    write_csv(ba.round(2), "case_before_after.csv")

    # Model 2 budget sweep
    sweep = []
    for b in [1.0, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]:
        res = run_model("Model 2", vic, cap, replace(base, budget_factor=b))
        sweep.append({"budget_pct": int(round(100 * b)), "satisfaction": res.kpis["satisfaction_pct"],
                      "allocations": res.kpis["total_optimized_allocations"], "unmet": res.kpis["unmet_demand"],
                      "binding": res.kpis.get("binding_intervals", 0), "ok": res.ok})
    write_csv(pd.DataFrame(sweep).round(3), "model2_budget_sweep.csv")
    sw = pd.DataFrame(sweep)
    full = sw[sw["satisfaction"] >= 99.99]
    put("MTwoFullAtBudget", int(full["budget_pct"].min()) if len(full) else 100)

    # Model 3 weight sets
    wsets = {
        "Equal (1,1,1,1)": {"efficiency": 1, "demand": 1, "frequency": 1, "utilization": 1},
        "Efficiency-led (5,1,1,1)": {"efficiency": 5, "demand": 1, "frequency": 1, "utilization": 1},
        "Demand-led (1,5,1,1)": {"efficiency": 1, "demand": 5, "frequency": 1, "utilization": 1},
        "Pattern-led (1,1,5,1)": {"efficiency": 1, "demand": 1, "frequency": 5, "utilization": 1},
        "Pattern-led (1,1,20,1)": {"efficiency": 1, "demand": 1, "frequency": 20, "utilization": 1},
        "Utilisation-led (1,1,1,5)": {"efficiency": 1, "demand": 1, "frequency": 1, "utilization": 5},
    }
    g_rows = []
    for name, w in wsets.items():
        res = run_model("Model 3", vic, cap, replace(base, weights=w))
        z = res.goals.set_index("goal")["normalised deviation - optimised plan"].to_numpy()
        g_rows.append({"weights": name, "allocations": res.kpis["total_optimized_allocations"],
                       "satisfaction": res.kpis["satisfaction_pct"], "z1": z[0], "z2": z[1], "z3": z[2], "z4": z[3],
                       "status": res.status})
        if name.startswith("Equal"):
            write_csv(res.goals.round(4), "model3_goals_equal.csv")
    write_csv(pd.DataFrame(g_rows).round(4), "model3_weights.csv")
    gw = pd.DataFrame(g_rows).set_index("weights")
    for key, name in [("Demand-led (1,5,1,1)", "Demand"), ("Pattern-led (1,1,5,1)", "PatternFive"),
                      ("Pattern-led (1,1,20,1)", "PatternTwenty"), ("Utilisation-led (1,1,1,5)", "Util"),
                      ("Efficiency-led (5,1,1,1)", "Eff")]:
        put(f"MThree{name}Alloc", gw.loc[key, "allocations"])
        put(f"MThree{name}Sat", num(gw.loc[key, "satisfaction"], 2))

    # Sensitivity analysis (app scenario runner) for Models 1 and 2
    for model in ["Model 1", "Model 2"]:
        scen = an.scenario_definitions(25 if model == "Model 2" else 10, [0.7, 0.9, 1.0], base)
        tab = an.run_scenarios(model, vic, cap, base, scen)
        cols = ["scenario", "feasible", "total_allocations", "satisfaction_pct", "unmet_demand",
                "avg_utilization_pct", "under_capacity_links"]
        write_csv(tab[cols].round(2), f"sensitivity_{model[-1]}.csv")
        if model == "Model 1":
            t1 = tab.set_index("scenario")
            put("SensMOneBase", t1.loc["A. Baseline", "total_allocations"])
            put("SensMOneTwenty", t1.loc["C. Demand +20%", "total_allocations"])
            put("SensMOneTwentyPct", num(100 * (t1.loc["C. Demand +20%", "total_allocations"] / t1.loc["A. Baseline", "total_allocations"] - 1), 1))
            put("SensMOneTen", t1.loc["B. Demand +10%", "total_allocations"])
            put("SensMOneTenPct", num(100 * (t1.loc["B. Demand +10%", "total_allocations"] / t1.loc["A. Baseline", "total_allocations"] - 1), 1))
            put("SensMOneRhoSeventy", t1.loc["E. Utilisation 70%", "total_allocations"])
            put("SensMOneRhoHundred", t1.loc["E. Utilisation 100%", "total_allocations"])
            put("SensMOneBudgetFeasible", "feasible" if t1.loc["D. Budget -10%", "feasible"] else "infeasible")

    # Peak-hour (07:00-10:00) Model 1 with max frequency equal to observed peak, to show an infeasible case
    am = vic[vic["interval_order"].between(8, 19)]
    inf = run_model("Model 1", am, cap, replace(base, target_utilization=0.5, max_frequency=int(vic["frequency"].max())))
    put("InfeasStatus", inf.status)
    put("InfeasDiag", inf.diagnostics[0].split(", e.g.")[0] if inf.diagnostics else "")

    # Runtime table across scopes
    rt = []
    for scope_name, scope in [("Victoria (30 links)", vic),
                              ("Central + Northern (212 links)", data[data["line"].isin(["Central", "Northern"])]),
                              ("Whole network (1,214 links)", data)]:
        for model in ["Model 1", "Model 2", "Model 3"]:
            if scope_name.startswith("Whole") and model != "Model 1":
                continue
            t = time.time(); res = run_model(model, scope, cap, base); dt = time.time() - t
            rt.append({"scope": scope_name, "model": model, "observations": res.kpis.get("observations"),
                       "seconds": round(dt, 1), "status": res.status})
    write_csv(pd.DataFrame(rt), "runtimes.csv")

    put("Rho", num(100 * RHO, 0))
    tex = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items())
    (OUT / "results.tex").write_text("% Generated by report/generate_report_data.py -- do not edit by hand\n" + tex + "\n",
                                     encoding="utf-8")
    print(f"Wrote {len(macros)} macros and {len(list(OUT.glob('*.csv')))} CSV files to {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
