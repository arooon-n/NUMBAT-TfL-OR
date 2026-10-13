"""Turn the CSVs written by generate_report_data.py into LaTeX table bodies (report/data/tab_*.tex).

Run from the project root after generate_report_data.py:   python report/make_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"


def n(v, d=0):
    if pd.isna(v):
        return "--"
    return f"{v:,.{d}f}".replace(",", "{,}")


def write(name: str, rows: list[str]) -> None:
    (DATA / name).write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    lines = pd.read_csv(DATA / "line_demand.csv")
    lf = pd.read_csv(DATA / "load_factor_by_line.csv").set_index("line")
    rows = []
    for r in lines.itertuples():
        f = lf.loc[r.line] if r.line in lf.index else None
        rows.append(f"{r.line} & {n(r.links)} & {n(r.demand / 1e3, 0)} & {n(r.trains)} & {n(r.pass_per_train, 0)} & "
                    f"{n(f.mean_lf, 1) if f is not None else '--'} & {n(f.max_lf, 0) if f is not None else '--'} \\\\")
    write("tab_lines.tex", rows)

    top = pd.read_csv(DATA / "top_links.csv")
    write("tab_top_links.tex", [f"{i + 1} & {r.from_station} $\\rightarrow$ {r.to_station} & {r.line} & {n(r.demand)} & {n(r.peak_demand)} \\\\"
                                for i, r in enumerate(top.itertuples())])

    zl = pd.read_csv(DATA / "zero_train_by_line.csv").head(8)
    write("tab_zero_lines.tex", [f"{r.line} & {n(r.obs)} & {n(r.passengers)} & {n(r.largest, 0)} \\\\" for r in zl.itertuples()])

    m1 = pd.read_csv(DATA / "model1_by_line.csv")
    write("tab_model1_lines.tex", [f"{r.line} & {n(r.observed)} & {n(r.required)} & "
                                   f"{'--' if pd.isna(r.ratio) else n(100 * r.ratio, 0) + chr(92) + '%'} & "
                                   f"{n(r.peak_obs)} & {n(r.peak_req)} \\\\" for r in m1.itertuples()])

    case = pd.read_csv(DATA / "case_models.csv")
    write("tab_case.tex", [f"{r.case} & {n(r.allocations)} & {n(r.satisfaction, 2)} & {n(r.unmet)} & "
                           f"{n(r.utilisation, 1)} & {n(r.shortfall_links)} & {n(r.seconds, 1)} \\\\" for r in case.itertuples()])

    w = pd.read_csv(DATA / "model3_weights.csv")
    write("tab_weights.tex", [f"{r.weights} & {n(r.allocations)} & {n(r.satisfaction, 2)} & {r.z1:.3f} & {r.z2:.4f} & "
                              f"{r.z3:.3f} & {r.z4:.3f} \\\\" for r in w.itertuples()])

    for k in ("1", "2"):
        s = pd.read_csv(DATA / f"sensitivity_{k}.csv")
        rows = []
        for r in s.itertuples():
            if r.feasible:
                rows.append(f"{r.scenario} & {n(r.total_allocations)} & {n(r.satisfaction_pct, 2)} & {n(r.unmet_demand)} & "
                            f"{n(r.avg_utilization_pct, 1)} & {n(r.under_capacity_links)} \\\\")
            else:
                rows.append(f"{r.scenario} & \\multicolumn{{5}}{{c}}{{\\emph{{infeasible (reported, not relaxed)}}}} \\\\")
        write(f"tab_sens_{k}.tex", rows)

    sw = pd.read_csv(DATA / "model2_budget_sweep.csv")
    write("tab_sweep.tex", [f"{r.budget_pct}\\% & {n(r.allocations)} & {n(r.satisfaction, 2)} & {n(r.unmet)} & {r.binding} \\\\"
                            for r in sw.itertuples()])

    rt = pd.read_csv(DATA / "runtimes.csv")
    write("tab_runtimes.tex", [f"{r.scope} & {r.model} & {n(r.observations)} & {n(r.seconds, 1)} & {r.status} \\\\"
                               for r in rt.itertuples()])

    g = pd.read_csv(DATA / "model3_goals_equal.csv")
    names = ["G1 efficiency", "G2 demand", "G3 service pattern", "G4 utilisation"]
    write("tab_goals.tex", [f"{names[i]} & {r[1]:g} & {r[2]:.3f} & {r[3]:.3f} \\\\" for i, r in enumerate(g.itertuples(index=False))])
    print("tables written to", DATA)


if __name__ == "__main__":
    main()
