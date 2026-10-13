"""Streamlit dashboard: Data-Driven Urban Metro Train Frequency and Capacity Optimization."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

import analysis as an
import visualization as viz
from data_loader import DATA_SOURCE_URL, load_workbook_bytes
from optimization import GOALS, MODEL_NAMES, SOLVER_HELP, Settings, run_model, solver_available
from preprocessing import preprocess, readiness

CAPACITY_FILE = Path(__file__).parent / "config" / "capacity_by_line.csv"
LARGE_SCOPE = 30_000

st.set_page_config(page_title="Metro Frequency Optimizer", page_icon="\U0001F687", layout="wide")
st.markdown("""<style>
div[data-testid="stMetricValue"] {font-size: 1.45rem;}
.block-container {padding-top: 2rem;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data

# Bump when preprocessing output changes. st.cache_resource otherwise keeps objects built by older code
# alive in a running server (this caused a KeyError after the zero-train metrics were added).
PREPROCESS_VERSION = "2026-10-zero-train"


@st.cache_resource(max_entries=2, show_spinner=False)
def load_and_process(content: bytes, name: str, _version: str = PREPROCESS_VERSION):
    wb = load_workbook_bytes(content, name)
    proc, error = None, None
    if not wb.errors or wb.loads is not None:
        try:
            proc = preprocess(wb)
        except Exception as exc:  # reported to the user, never a raw traceback
            error = str(exc)
    meta = {"errors": wb.errors, "warnings": wb.warnings, "cover": wb.cover_notes,
            "sheets": wb.sheet_names, "sizes": wb.sheet_sizes,
            "loads_fields": wb.loads.fields if wb.loads else {},
            "freq_fields": wb.frequencies.fields if wb.frequencies else {},
            "periods": wb.loads.period_columns if wb.loads else []}
    return meta, proc, error


@st.cache_data(show_spinner=False)
def default_capacities() -> pd.DataFrame:
    if CAPACITY_FILE.exists():
        return pd.read_csv(CAPACITY_FILE)
    return pd.DataFrame(columns=["line", "capacity_per_train", "rolling_stock", "source", "source_url"])


def fmt(v, digits=0, suffix=""):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{v:,.{digits}f}{suffix}"


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ------------------------------------------------------------------ sidebar: upload

st.sidebar.title("\U0001F687 Metro Frequency Optimizer")
upload = st.sidebar.file_uploader("NUMBAT workbook (.xlsx)", type=["xlsx"],
                                  help=f"Download from {DATA_SOURCE_URL} and upload it unchanged.")

if upload is None:
    st.title("Data-Driven Urban Metro Train Frequency and Capacity Optimization")
    st.markdown(f"""
Upload an original **TfL NUMBAT** workbook (from [{DATA_SOURCE_URL}]({DATA_SOURCE_URL})) in the sidebar.

The app will
1. detect the `Link_Loads` and `Link_Frequencies` sheets, their header rows and 15-minute interval columns;
2. clean and join demand with observed train frequency, and report data quality;
3. let you explore demand and service, then run one of three optimisation models
   (integer programming, capacity-constrained MILP, weighted goal programming) with HiGHS;
4. run demand and budget scenarios and download every result as CSV.
""")
    if not solver_available():
        st.error(SOLVER_HELP)
    st.stop()

content = upload.getvalue()
file_id = hashlib.sha256(content).hexdigest()[:16]
if st.session_state.get("file_id") != file_id:  # new workbook: drop every stale result
    st.session_state.clear()
    st.session_state["file_id"] = file_id

with st.spinner("Reading and validating the workbook..."):
    meta, proc, prep_error = load_and_process(content, upload.name)

if meta["errors"] and proc is None:
    st.error("The workbook cannot be used for optimisation:")
    for e in meta["errors"]:
        st.markdown(f"- {e}")
    st.info("Make sure you uploaded an unmodified NUMBAT workbook that contains link sheets.")
    st.stop()
if prep_error:
    st.error(f"Preprocessing failed: {prep_error}")
    st.stop()

data = proc.data
ready = readiness(proc)

# ------------------------------------------------------------------ sidebar: scope

st.sidebar.header("1. Scope")
line_demand = data[data["eligible_demand"]].groupby("line")["demand"].sum().sort_values(ascending=False)
all_lines = list(line_demand.index) + sorted(set(data["line"].dropna()) - set(line_demand.index))
default_line = ["Victoria"] if "Victoria" in all_lines else all_lines[:1]
all_toggle = st.sidebar.checkbox("All lines", value=False)
lines = all_lines if all_toggle else st.sidebar.multiselect("Lines", all_lines, default=default_line)
dirs_available = sorted(data.loc[data["line"].isin(lines), "direction"].dropna().unique()) if "direction" in data else []
directions = st.sidebar.multiselect("Directions (empty = all)", dirs_available, default=[])
interval_labels = (data.drop_duplicates("interval_order").sort_values("interval_order")
                   .set_index("interval_order")["interval"])
orders = list(interval_labels.index)
first, last = st.sidebar.select_slider(
    "Time window (15-min intervals)", options=orders, value=(orders[0], orders[-1]),
    format_func=lambda o: interval_labels[o])
scope = an.filter_scope(data, lines, directions, first, last) if lines else data.iloc[0:0]
scope_text = (f"{', '.join(lines) if len(lines) <= 4 else f'{len(lines)} lines'}"
              f"{' / ' + ', '.join(directions) if directions else ''}; "
              f"{interval_labels[first][:4]}–{interval_labels[last][5:]}")

# ------------------------------------------------------------------ sidebar: assumptions

st.sidebar.header("2. Capacity assumptions")
st.sidebar.caption("Not in the NUMBAT data: you must supply capacity per train.")
cap_mode = st.sidebar.radio("Passenger capacity per train",
                            ["Per-line values (published defaults, editable)", "One uniform value for all lines"])
defaults = default_capacities()
if cap_mode.startswith("Per-line"):
    base = pd.DataFrame({"line": lines}).merge(defaults[["line", "capacity_per_train", "rolling_stock"]], on="line", how="left")
    edited = st.sidebar.data_editor(
        base, hide_index=True, disabled=["line", "rolling_stock"], key=f"cap_editor_{','.join(lines)}",
        column_config={"capacity_per_train": st.column_config.NumberColumn("Capacity", min_value=1, step=1)})
    capacity = {r.line: (float(r.capacity_per_train) if pd.notna(r.capacity_per_train) else None) for r in edited.itertuples()}
    cap_note = "Per-line capacities (defaults from config/capacity_by_line.csv, sourced from TfL rolling-stock data sheets; see README)"
else:
    uniform = st.sidebar.number_input("Capacity per train (passengers)", min_value=1, value=None, step=10,
                                      placeholder="Enter a value")
    capacity = {line: (float(uniform) if uniform else None) for line in lines}
    cap_note = f"Uniform capacity of {uniform} passengers per train applied to every selected line (user assumption)"
missing_cap = [k for k, v in capacity.items() if not v or v <= 0]
if missing_cap:
    st.sidebar.warning(f"Enter a positive capacity for: {', '.join(missing_cap)}")

rho = st.sidebar.slider("Target utilisation ρ (usable share of capacity)", 5, 100, 80, step=5,
                        help="Effective capacity C_eff = C x ρ. A modelling choice, not a TfL value.") / 100

st.sidebar.header("3. Service constraints")
min_freq = st.sidebar.number_input("Minimum trains per link per 15 min", 0, 30, 0)
min_only = st.sidebar.checkbox("Apply the minimum only where trains are observed", value=True,
                               help="Avoids forcing service in intervals when the link is closed (e.g. at night).")
use_max = st.sidebar.checkbox("Set a maximum trains per link per 15 min", value=False)
max_freq = st.sidebar.number_input("Maximum trains per link per 15 min", 1, 60, 10) if use_max else None
if max_freq is not None and max_freq < min_freq:
    st.sidebar.error("Minimum frequency exceeds the maximum.")

st.sidebar.header("4. Data policy")
zt_choice = st.sidebar.radio(
    "Passengers reported where 0 trains are scheduled",
    ["Exclude from optimisation (disclosed)", "Include as demand"],
    help="Link_Loads sometimes shows passengers on a link in an interval where Link_Frequencies shows no train. "
         "See the Data Quality tab.")
zero_policy = "exclude" if zt_choice.startswith("Exclude") else "include"

# ------------------------------------------------------------------ layout

st.title("Metro Train Frequency and Capacity Optimization")
st.caption(f"Workbook **{upload.name}** · scope: {scope_text} · source: TfL NUMBAT ({DATA_SOURCE_URL})")
tabs = st.tabs(["Overview", "Data Quality", "Explore Data", "Optimization", "Sensitivity Analysis"])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    st.subheader("Readiness")
    cols = st.columns(4)
    for col, (name, (ok, msg)) in zip(cols, ready.items()):
        if name.startswith("Model") and ok and missing_cap:
            ok, msg = False, "Needs a positive capacity for every selected line (sidebar)."
        col.markdown(f"**{name}**  \n{'✅ Ready' if ok else '⛔ Not ready'}  \n<small>{msg}</small>",
                     unsafe_allow_html=True)
    if not solver_available():
        st.error(SOLVER_HELP)

    st.subheader("Selected scope")
    if scope.empty:
        st.warning("The selected scope is empty: choose at least one line and a time window.")
    else:
        prof = an.time_profile(scope)
        peak = prof.loc[prof["demand"].idxmax()] if len(prof) else None
        m = st.columns(5)
        m[0].metric("Links", f"{scope['link_id'].nunique():,}")
        m[1].metric("Valid link-intervals", f"{int(scope['eligible_demand'].sum()):,}")
        m[2].metric("Passenger-link demand", fmt(scope["demand"].sum()))
        m[3].metric("Observed train allocations", fmt(scope["frequency"].sum()))
        m[4].metric("Peak interval", peak["interval"] if peak is not None else "n/a",
                    help="Interval with the highest total link demand in the scope")
        st.plotly_chart(viz.time_profile(prof), width="stretch")
        st.caption("Demand is the sum of link loads (a passenger crossing five links counts five times). "
                   "Observed allocations are the sum of link frequencies, not a count of physical trains.")
    if meta["cover"]:
        with st.expander("Dataset notes from the workbook cover sheet"):
            for note in meta["cover"]:
                st.markdown(note.replace("\n", "  \n"))

# ------------------------------------------------------------------ Data Quality
with tabs[1]:
    mt = proc.metrics
    st.subheader("Data quality and readiness")
    c = st.columns(6)
    c[0].metric("Worksheets", len(meta["sheets"]))
    c[1].metric("Links", f"{mt['links']:,}")
    c[2].metric("Lines", mt["lines"])
    c[3].metric("Stations", f"{mt['stations']:,}")
    c[4].metric("15-min intervals", mt["intervals"])
    c[5].metric("Eligible for all models", f"{mt['pct_eligible_service']:.1f}%")
    st.markdown(f"**Detected worksheets:** {', '.join(meta['sheets'])}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Link sheet profile**")
        sm = proc.sheet_metrics.set_index("sheet").T
        st.dataframe(sm, width="stretch")
        st.markdown("**Join and coverage**")
        n = mt["observations"]
        cov = pd.DataFrame({
            "check": ["Link-interval observations", "Valid demand (Models 1, exploration)",
                      "Valid demand + frequency (Models 2-3)", "Links only in Link_Loads",
                      "Links only in Link_Frequencies", "Ambiguous links excluded",
                      "Missing demand values", "Missing frequency values"],
            "count": [n, mt["eligible_demand"], mt["eligible_service"], mt["links_only_in_loads"],
                      mt["links_only_in_frequencies"], mt["ambiguous_links"], mt["missing_demand"], mt["missing_frequency"]],
        })
        cov["% of observations"] = [100.0, 100 * mt["eligible_demand"] / n, 100 * mt["eligible_service"] / n,
                                    None, None, None, 100 * mt["missing_demand"] / n, 100 * mt["missing_frequency"] / n]
        st.dataframe(cov, hide_index=True, width="stretch",
                     column_config={"% of observations": st.column_config.NumberColumn(format="%.2f")})
    with right:
        st.markdown("**Preprocessing actions**")
        for a in proc.actions:
            st.markdown(f"- {a}")
        st.markdown("**Warnings**")
        for w in meta["warnings"] + proc.warnings or ["None."]:
            st.markdown(f"- {w}")
        st.markdown("**Excluded observations**")
        if proc.exclusions.empty:
            st.success("No observations excluded.")
        else:
            st.dataframe(proc.exclusions, hide_index=True, width="stretch")
        st.caption("Policy: missing demand is never set to zero; invalid, negative and ambiguous records are "
                   "excluded and reported, not imputed or merged.")

    st.markdown("**Passengers with zero scheduled trains** (consistency check between the two sheets)")
    if mt.get("zero_train_obs"):
        z1, z2, z3 = st.columns(3)
        z1.metric("Link-intervals affected", f"{mt['zero_train_obs']:,}",
                  delta=f"{100 * mt['zero_train_obs'] / max(mt['eligible_service'], 1):.2f}% of matched", delta_color="off")
        z2.metric("Passengers involved", fmt(mt["zero_train_demand"]),
                  delta=f"{mt['zero_train_demand_pct']:.2f}% of demand", delta_color="off")
        z3.metric("Demand with trains in an adjacent interval", f"{mt['zero_train_adjacent_pct']:.0f}%")
        zt = data[data["zero_train_demand"]]
        c_a, c_b = st.columns(2)
        with c_a:
            st.dataframe(zt.groupby("line").agg(link_intervals=("demand", "size"), passengers=("demand", "sum"),
                                                largest=("demand", "max")).sort_values("passengers", ascending=False),
                         width="stretch")
        with c_b:
            st.dataframe(zt.groupby("interval").agg(link_intervals=("demand", "size"), passengers=("demand", "sum"))
                         .sort_values("passengers", ascending=False).head(12), width="stretch")
        st.caption("A train cannot carry passengers in an interval where none is scheduled, so these records are "
                   "inconsistent across the two sheets. They concentrate at the start and end of the traffic day and "
                   "usually sit next to an interval with trains, which suggests period-boundary assignment in NUMBAT. "
                   f"Current policy (sidebar): **{zt_choice.lower()}**.")
    else:
        st.success("Every link-interval with passengers has at least one scheduled train.")

    with st.expander("Detected fields"):
        st.write({"Link_Loads": meta["loads_fields"], "Link_Frequencies": meta["freq_fields"],
                  "time-of-day columns": meta["periods"], "join key": proc.key_fields})
    st.markdown("**Processed link-interval data (preview)**")
    preview_cols = ["link_id", "line", "direction", "from_station", "to_station", "interval", "demand",
                    "frequency", "demand_status", "frequency_status", "eligible_service"]
    st.dataframe(data[[c for c in preview_cols if c in data]].head(200), hide_index=True, width="stretch")
    st.download_button("Download processed link-interval data (CSV)",
                       csv_bytes(data.drop(columns=["link_key"], errors="ignore")),
                       file_name="numbat_link_interval_processed.csv", mime="text/csv")

# ------------------------------------------------------------------ Explore
with tabs[2]:
    if scope.empty:
        st.warning("The selected scope is empty.")
    else:
        lt = an.link_totals(scope)
        a, b = st.columns(2)
        with a:
            st.plotly_chart(viz.top_links(lt, "display", "demand", "Highest-demand links (sum over the time window)"),
                            width="stretch")
        with b:
            line_tot = (an.link_totals(data[data["interval_order"].between(first, last)])
                        .groupby("line", as_index=False)["demand"].sum())
            st.plotly_chart(viz.top_links(line_tot, "line", "demand", "Demand by line (all lines, same time window)", n=25),
                            width="stretch")
        periods = [p for p in meta["periods"] if p in proc.link_summary]
        if periods:
            summ = proc.link_summary[proc.link_summary["line"].isin(lines)]
            if directions and "direction" in summ:
                summ = summ[summ["direction"].isin(directions)]
            st.plotly_chart(viz.period_bars(summ, periods), width="stretch")
            st.caption("Peak versus off-peak uses the workbook's own time-of-day columns for the selected links "
                       "(whole day, independent of the time window).")

        n_heat = st.slider("Links in the heatmap", 10, 80, 30, step=5)
        top = lt.nlargest(n_heat, "demand")["link_id"]
        hm = scope[scope["link_id"].isin(top)]
        pivot = hm.pivot_table(index="link_label", columns="interval_order", values="demand", aggfunc="sum")
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
        pivot.columns = [interval_labels[c][:2] + ":" + interval_labels[c][2:4] for c in pivot.columns]
        st.plotly_chart(viz.heatmap(pivot, f"Demand heatmap — top {n_heat} links by demand"), width="stretch")

        st.subheader("Demand versus observed service")
        if missing_cap:
            st.info("Enter a capacity for every selected line to compare demand with service capacity.")
        elif not scope["eligible_service"].any():
            st.info("No observed frequency is available for this scope.")
        else:
            obs = an.with_observed_capacity(scope, capacity, rho)
            st.plotly_chart(viz.demand_vs_service(obs), width="stretch")
            over, under = an.mismatch_tables(obs, rho)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Potential mismatches:** {len(over):,} link-intervals where demand exceeds "
                            f"observed effective capacity (C × ρ = {rho:.0%} of rated)")
                st.dataframe(over.head(50), hide_index=True, width="stretch")
            with c2:
                st.markdown(f"**Potentially under-used service:** {len(under):,} link-intervals with trains "
                            "running below 15% of rated capacity")
                st.dataframe(under.head(50), hide_index=True, width="stretch")
            st.caption("Derived metrics: they depend on your capacity and utilisation assumptions, not on TfL data. "
                       "Link-intervals with passengers but no scheduled train are data inconsistencies, not capacity "
                       "mismatches; they appear at zero capacity in the chart and are analysed in the Data Quality tab.")

        st.subheader("Network topology (schematic)")
        n_edges = st.slider("Links shown (highest metric first)", 20, 400, 120, step=20)
        metric_opts = {"Passenger demand": "demand"}
        last_res = st.session_state.get("last_result")
        if last_res is not None and last_res.ok:
            metric_opts["Unmet demand in last optimisation"] = "unmet"
        metric_name = st.radio("Shade links by", list(metric_opts), horizontal=True)
        if metric_opts[metric_name] == "demand":
            edges = lt.rename(columns={"demand": "value"})
        else:
            edges = (last_res.table.groupby(["link_id", "link_label", "line", "from_station", "to_station"], as_index=False)
                     ["unmet_demand"].sum().rename(columns={"unmet_demand": "value"}))
        edges = edges.nlargest(n_edges, "value")
        st.plotly_chart(viz.network(edges, "value", metric_name), width="stretch")
        st.caption("Positions come from a force-directed layout of the link graph. This is a schematic "
                   "topology view, not a geographic TfL map; NUMBAT contains no station coordinates.")

# ------------------------------------------------------------------ Optimization
FORMULATIONS = {
    "Model 1": r"""
**Decision variable:** $x_i \in \mathbb{Z}_{\ge 0}$ = trains allocated to link-interval $i$.

$$\min \sum_i x_i \quad \text{s.t.} \quad C_i^{\text{eff}} x_i \ge D_i,\;\; l_i \le x_i \le u_i,\;\; \Big[\textstyle\sum_{i\in t} x_i \le F_t\Big]$$

$D_i$ is demand rounded to whole passengers and $C_i^{\text{eff}} = \lfloor \rho\,C_i \rfloor$ (whole passengers,
rounded down so load never exceeds the target). Without a budget the problem separates and the optimum is
$x_i = \max(l_i, \lceil D_i / C_i^{\text{eff}} \rceil)$; the solver result is checked against this.
""",
    "Model 2": r"""
**Decision variables:** $x_i \in \mathbb{Z}_{\ge 0}$ (train allocations), $y_i \in \mathbb{Z}_{\ge 0}$ (passengers served).

$$\max \sum_i y_i \quad \text{s.t.}\quad 0 \le y_i \le D_i,\;\; y_i \le C_i^{\text{eff}} x_i,\;\; \sum_{i\in t} x_i \le F_t \;\forall t,\;\; l_i \le x_i \le u_i$$

$F_t = \lfloor \beta \sum_{i \in t} f_i \rfloor$: the **link-service allocation budget**, i.e. $\beta$ times the observed
link frequencies in interval $t$. A second stage minimises $\sum_i x_i$ while keeping $\sum_i y_i$ at its
optimum, so no useless allocations remain. $u_i \le \lceil D_i / C_i^{\text{eff}} \rceil$ is a technical bound
(more trains cannot serve more passengers).
""",
    "Model 3": r"""
**Decision variables:** $x_i \in \mathbb{Z}_{\ge 0}$, $y_i \in \{0,\dots,D_i\}$ with $y_i \le C_i x_i$ (rated capacity is the hard limit),
and non-negative integer deviation variables (differences of whole trains or whole passengers).

| Goal | Goal constraint | Normalised deviation $z_k$ |
|---|---|---|
| G1 efficiency | $\sum_i x_i - E = p_1 - n_1$, $E=\sum_i \max(l_i,\lceil D_i/C^{\text{eff}}_i\rceil)$ | $p_1 / E$ |
| G2 demand | unmet $=\sum_i (D_i - y_i)$ | unmet $/ \sum_i D_i$ |
| G3 service pattern | $x_i - f_i = p_{3i} - n_{3i}$ | $\sum_i (p_{3i}+n_{3i}) / \sum_i f_i$ |
| G4 utilisation | $D_i - \rho C_i x_i = p_{4i} - n_{4i}$ | $\sum_i (p_{4i}+n_{4i}) / \sum_i D_i$ |

$$\min \sum_k w_k z_k$$

Each deviation is divided by its goal's target (percentage normalisation), so passengers and trains are comparable.
G4's over-target part $p_4$ (load above $\rho C$) is related to G2: unmet demand arises only once load exceeds the
*rated* capacity $C$, while $p_4$ starts at $\rho C$. NUMBAT does not observe crowding directly; $p_4$ is derived
from link loads and the capacity assumption. A $10^{-3}$ tie-break weight on every goal keeps solutions Pareto-efficient.
""",
}

with tabs[3]:
    model = st.radio("Model", list(MODEL_NAMES), horizontal=True, format_func=lambda m: f"{m}: {MODEL_NAMES[m]}")
    questions = {
        "Model 1": "What is the minimum train frequency required to satisfy observed demand under the selected capacity assumptions?",
        "Model 2": "Given a limited service-allocation budget, how should train allocations be distributed to maximise passenger demand served?",
        "Model 3": "Which service plan gives the best compromise between efficiency, demand satisfaction and service quality under the selected priorities?",
    }
    st.markdown(f"*{questions[model]}*")
    with st.expander("Mathematical formulation"):
        st.markdown(FORMULATIONS[model])

    c1, c2 = st.columns(2)
    with c1:
        if model == "Model 2":
            use_budget = True
            st.markdown("**Link-service allocation budget** (required)")
        else:
            use_budget = st.checkbox("Enforce a link-service allocation budget per interval", value=False)
        budget_pct = st.slider("Budget as % of observed link allocations in each interval", 10, 200, 100, step=5,
                               disabled=not use_budget)
        st.caption("Budget Fₜ = β × (sum of observed link frequencies in interval t). A modelling "
                   "assumption: link allocations are not physical trains (one train runs over many links).")
    with c2:
        weights = {g: 1.0 for g in GOALS}
        if model == "Model 3":
            st.markdown("**Goal priorities** (editable modelling settings, not TfL values)")
            for g, label in GOALS.items():
                weights[g] = st.slider(label, 0.0, 5.0, 1.0, 0.25, key=f"w_{g}")
        time_limit = st.number_input("Solver time limit (seconds)", 10, 1800, 120, step=10)

    settings = Settings(target_utilization=rho, min_frequency=int(min_freq), max_frequency=max_freq,
                        min_only_in_service=min_only, use_budget=use_budget, budget_factor=budget_pct / 100,
                        weights=weights, time_limit=float(time_limit), zero_train_policy=zero_policy)
    problems = settings.validate()
    model_ready, ready_msg = ready[model]
    if not model_ready:
        problems.append(ready_msg)
    if missing_cap:
        problems.append(f"Missing capacity for: {', '.join(missing_cap)}.")
    if scope.empty:
        problems.append("The selected scope is empty.")
    if not solver_available():
        problems.append(SOLVER_HELP)
    n_obs = int(scope["eligible_service" if model != "Model 1" else "eligible_demand"].sum())
    if n_obs > LARGE_SCOPE and model != "Model 1":
        st.warning(f"{n_obs:,} link-intervals selected: {model} may take minutes. Narrow the lines or time window "
                   "for faster runs; the time limit is enforced.")
    for p in problems:
        st.error(p)

    run_sig = (model, file_id, tuple(lines), tuple(directions), first, last, str(asdict(settings)), str(sorted(capacity.items())))
    if st.button("Optimize", type="primary", disabled=bool(problems)):
        with st.spinner(f"Solving {model} for {n_obs:,} link-intervals with HiGHS..."):
            res = run_model(model, scope, capacity, settings)
        res.scope_text = scope_text
        res.cap_note = cap_note
        res.signature = run_sig
        st.session_state.setdefault("results", {})[model] = res
        st.session_state["last_result"] = res

    res = st.session_state.get("results", {}).get(model)
    if res is not None:
        st.divider()
        if res.signature != run_sig:
            st.info("Inputs changed since this result was produced. It is shown with the parameters it used; "
                    "press Optimize to update.")
        s = res.settings
        with st.container(border=True):
            st.markdown(f"**{res.model}: {MODEL_NAMES[res.model]}** · dataset `{upload.name}` · scope: {res.scope_text}")
            st.markdown(
                f"Assumptions: {res.cap_note}; target utilisation ρ = {s.target_utilization:.0%}; "
                f"min {s.min_frequency} trains/15 min{' where observed in service' if s.min_only_in_service else ''}; "
                f"max {s.max_frequency if s.max_frequency is not None else 'none (technical bound only)'}; "
                f"budget {f'{s.budget_factor:.0%} of observed allocations per interval' if (s.use_budget or res.model == 'Model 2') else 'not enforced'}"
                + (f"; weights {', '.join(f'{k} {v:g}' for k, v in s.weights.items())}" if res.model == "Model 3" else "")
                + f"; demand multiplier {s.demand_multiplier:g}; zero-train demand: {s.zero_train_policy}d.")
        if not res.ok:
            st.error(f"**{res.status}.** No result is shown because there is no valid solution.")
            for d in res.diagnostics:
                st.markdown(f"- {d}")
            if res.checks:
                st.dataframe(pd.DataFrame(res.checks, columns=["check", "passed", "detail"]), hide_index=True)
        else:
            (st.success if res.proven_optimal else st.warning)(
                f"Solver status: **{res.status}**" + ("" if res.proven_optimal else
                f" (relative gap {res.gap:.2%})" if res.gap is not None else ""))
            for d in res.diagnostics:
                st.warning(d)
            k = res.kpis
            r1 = st.columns(4)
            r1[0].metric("Passenger demand (scope)", fmt(k["total_demand"]))
            r1[1].metric("Optimised train allocations", f"{k['total_optimized_allocations']:,}",
                         delta=None if k["allocation_change"] is None else f"{k['allocation_change']:+,.0f} vs observed",
                         delta_color="off")
            r1[2].metric("Demand served", fmt(k["demand_served"]), delta=f"{k['satisfaction_pct']:.2f}% satisfied", delta_color="off")
            r1[3].metric("Unmet demand", fmt(k["unmet_demand"]))
            r2 = st.columns(4)
            r2[0].metric("Effective capacity offered", fmt(k["offered_effective_capacity"]))
            r2[1].metric("Utilisation of rated capacity", fmt(k["avg_utilization_pct"], 1, "%"))
            r2[2].metric("Links with a capacity shortfall", f"{k['links_with_shortfall']:,}",
                         help="Links where demand exceeds C x ρ x trains in at least one interval")
            r2[3].metric("Links with more / less service", f"{k['links_more_service']} / {k['links_less_service']}")

            if k.get("zero_train_observations"):
                if k["zero_train_policy"] == "exclude":
                    st.info(f"{k['zero_train_observations']:,} link-intervals with passengers but 0 scheduled trains "
                            f"({k['zero_train_demand']:,.0f} passengers) were excluded from this run.")
                else:
                    st.info(f"{k['zero_train_observations']:,} link-intervals with passengers but 0 scheduled trains "
                            "are included, so the plan may add service where the timetable has none.")
            st.markdown("**Interpretation** (generated from the computed results)")
            for line in an.interpret(res):
                st.markdown(f"- {line}")

            st.plotly_chart(viz.before_after(res.table), width="stretch")
            st.plotly_chart(viz.change_heatmap(res.table), width="stretch")

            if not res.budget.empty:
                with st.expander(f"Allocation budget by interval ({k.get('binding_intervals', 0)} binding)"):
                    st.dataframe(res.budget, hide_index=True, width="stretch")
            if not res.goals.empty:
                st.markdown("**Goal deviations** (0 = goal fully met; lower is better)")
                st.dataframe(res.goals, hide_index=True, width="stretch",
                             column_config={c: st.column_config.NumberColumn(format="%.4f") for c in res.goals.columns[1:]})
                st.caption("G1 compares allocations with E, the fewest allocations that carry all demand at the target "
                           "utilisation. Raising a goal's weight trades the other goals for it.")

            others = {m: r for m, r in st.session_state.get("results", {}).items()
                      if m != res.model and r.ok and r.signature[1:6] == res.signature[1:6]}
            if others:
                st.markdown("**Comparison with other models run on the same data scope**")
                comp = pd.DataFrame([{"model": m, **{kk: r.kpis.get(kk) for kk in
                                     ["total_optimized_allocations", "total_existing_allocations", "demand_served",
                                      "satisfaction_pct", "unmet_demand", "avg_utilization_pct", "links_with_shortfall"]}}
                                     for m, r in sorted({**others, res.model: res}.items())])
                st.dataframe(comp, hide_index=True, width="stretch")
                st.caption("Comparable only where the assumptions shown above match; each model answers a different question.")

            with st.expander("Solution validation checks"):
                st.dataframe(pd.DataFrame(res.checks, columns=["check", "passed", "detail"]), hide_index=True,
                             width="stretch")

            st.markdown("**Detailed results**")
            show = res.table.drop(columns=["interval_order", "link_label"])
            st.dataframe(show.head(1000), hide_index=True, width="stretch")
            st.caption("Observed: demand, existing_frequency. User assumptions: capacity, ρ. Derived: offered capacity, "
                       "utilisation, shortfalls. Recommendations: optimized_frequency, demand_served. "
                       "existing_* columns evaluate the observed timetable under the same assumptions.")
            st.download_button("Download detailed results (CSV)", csv_bytes(show),
                               file_name=f"{res.model.replace(' ', '_').lower()}_results.csv", mime="text/csv")

# ------------------------------------------------------------------ Sensitivity
with tabs[4]:
    st.markdown("Re-solve one model under demand, budget and utilisation scenarios. Other inputs are taken "
                "from the sidebar and the Optimization tab.")
    c1, c2, c3 = st.columns(3)
    s_model = c1.selectbox("Model", list(MODEL_NAMES), index=list(MODEL_NAMES).index(model))
    cut = c2.slider("D. Budget reduction (%)", 5, 50, 10, step=5)
    alt = c3.multiselect("E. Alternative target utilisations", [x / 100 for x in range(50, 101, 5)],
                         default=[0.7, 0.9], format_func=lambda v: f"{v:.0%}")
    names = [d["scenario"] for d in an.scenario_definitions(cut, alt, settings)]
    chosen = st.multiselect("Scenarios", names, default=names)
    st.caption("Scenario D applies to the modelled link-service allocation budget, not to TfL's physical fleet. "
               "Infeasible scenarios are reported, never relaxed.")
    s_ok, s_msg = ready[s_model]
    blockers = settings.validate() + ([] if s_ok else [s_msg]) + (["Missing capacities."] if missing_cap else []) \
        + (["Empty scope."] if scope.empty else []) + ([SOLVER_HELP] if not solver_available() else [])
    for b in blockers:
        st.error(b)
    if st.button("Run Sensitivity Analysis", type="primary", disabled=bool(blockers) or not chosen):
        scen = [d for d in an.scenario_definitions(cut, alt, settings) if d["scenario"] in chosen]
        bar = st.progress(0.0, text="Starting...")
        table = an.run_scenarios(s_model, scope, capacity, settings, scen,
                                 progress=lambda p, name: bar.progress(p, text=f"Solved {name}"))
        bar.empty()
        st.session_state["scenarios"] = {"model": s_model, "table": table, "scope": scope_text}
    sc = st.session_state.get("scenarios")
    if sc is not None:
        st.markdown(f"**{sc['model']}** · scope: {sc['scope']}")
        st.dataframe(sc["table"], hide_index=True, width="stretch",
                     column_config={"satisfaction_pct": st.column_config.NumberColumn(format="%.2f"),
                                    "avg_utilization_pct": st.column_config.NumberColumn(format="%.1f")})
        metric_labels = {"total_allocations": "Total train allocations", "satisfaction_pct": "Demand satisfaction (%)",
                         "unmet_demand": "Unmet demand", "avg_utilization_pct": "Utilisation of rated capacity (%)",
                         "under_capacity_links": "Links with a capacity shortfall"}
        mkey = st.selectbox("Chart metric", list(metric_labels), format_func=metric_labels.get)
        st.plotly_chart(viz.scenario_bars(sc["table"], mkey, metric_labels[mkey]), width="stretch")
        st.download_button("Download scenario comparison (CSV)", csv_bytes(sc["table"]),
                           file_name="scenario_comparison.csv", mime="text/csv")
