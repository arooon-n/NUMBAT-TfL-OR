# Data-Driven Urban Metro Train Frequency and Capacity Optimization

An Operations Research course project. It turns an original **Transport for London NUMBAT** workbook
(<https://crowding.data.tfl.gov.uk/>) into link-level demand and service data, profiles its quality, and
solves three optimisation models with **Pyomo + HiGHS** in a Streamlit dashboard.

## Features

- Upload an unmodified NUMBAT `.xlsx`. Sheets, header row, link fields and 15-minute interval columns are detected from
  cell contents (no fixed rows or columns).
- **Data Quality** report: record counts, missing, non-numeric and negative values, duplicates, ambiguous links,
  unmatched links and intervals, exclusions, preprocessing log, preview, and processed-data CSV download.
- **Explore Data**: demand and frequency over the day, highest-demand links and lines, peak vs off-peak, a demand
  heatmap, demand vs service capacity, mismatch tables, and a schematic network topology (NetworkX).
- **Optimization**: three models (below), KPIs, auto-generated interpretation, before/after charts, solution-validation
  checks, and a detailed results CSV.
- **Sensitivity Analysis**: baseline, demand +10% and +20%, reduced allocation budget, and alternative utilisation
  targets, shown as a comparison table, chart and CSV.

## Install and run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1            # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints, then upload the workbook in the sidebar. HiGHS is installed with `highspy`. If the
solver is missing, the app says so and refuses to optimise; it never shows invented results.

Run the tests with `python -m pytest -q`. The tests that use the supplied workbook run when it is at
`data/raw/NBT25TWT_Outputs.xlsx`.

## Project structure

```text
app.py               Streamlit UI (tabs: Overview, Data Quality, Explore Data, Optimization, Sensitivity Analysis)
data_loader.py       Workbook parsing: sheet, header-row, field-alias and time-interval detection; cover notes
preprocessing.py     Cleaning, wide->long reshaping, validated demand/frequency join, profiling, readiness
optimization.py      Models 1-3 (Pyomo/HiGHS), result tables, KPIs, post-solve validation
analysis.py          Descriptive summaries, interpretation text, scenario runner
visualization.py     Plotly charts and the schematic network
config/capacity_by_line.csv   Default capacity per train for each NUMBAT line, with sources (editable in the app)
tests/               pytest: loader, preprocessing, optimisation (synthetic fixtures + the supplied workbook)
```

## Data handling

Only **`Link_Loads`** (passengers on a link in a 15-minute interval) and **`Link_Frequencies`** (trains on that link
in that interval) feed the models. Station sheets are not substitutes for link loads.

- **Intervals**: headers like `0800-0815` are recognised by pattern and must span 15 minutes. They are ordered from the
  05:00 start of the NUMBAT day across midnight (`0445-0500` is last). Daily `Total` and the time-of-day columns
  (`Early`, `AM Peak`, and so on) are kept separately, for description only, and are never mixed with interval values.
  The app checks that `Total` equals the sum of the intervals.
- **Join**: the key is the first candidate that exists and is unique in both sheets: `Link` id; or line + direction +
  NLCs; or line + direction + station names. The data is joined on key + interval, never on row position.
- **Missing values are not zero.** Missing, non-numeric and negative values are flagged and excluded, and the count of
  each is reported. Genuine zeros are kept.
- **Duplicates**: exact duplicate rows are dropped once. Records sharing a key with different values are *ambiguous* and
  excluded, never aggregated. When the two sheets disagree on a link's identity (line, direction, NLC, or a different
  station), the link is excluded. Station labels that differ only by a bracketed branch qualifier
  (`Kennington (Bank)` vs `Kennington`) and different `Order` numbers are reported but kept.
- **Supplied workbook (NBT25TWT)**: 1,214 links, 19 lines, 482 stations, 96 intervals, with no missing or invalid values.
  2 links are excluded because the sheets disagree (`Liverpool Street` vs `Hackney Downs` as one link's destination, and
  a different `To NLC`). That leaves 99.8% of the 116,544 link-intervals eligible.
- **Passengers with zero scheduled trains** (first-review concern): 5,770 link-intervals (0.41% of demand) report
  passengers where `Link_Frequencies` shows no train. They cluster at the start and end of the traffic day, and for 85%
  of that demand the link has trains in an adjacent interval, which points to a period-boundary effect. They are
  flagged in the Data Quality tab. A sidebar policy either **excludes** them from optimisation (the default; the removed
  demand is shown with every result) or **includes** them. Including them makes Model 1 add about 5,760 trains in
  quarter-hours where the timetable runs none.

Readiness is shown per model. Model 1 needs only demand. Models 2 and 3 need matched observed frequency for the budget
and frequency targets.

## User-supplied parameters (not in NUMBAT)

| Parameter | Meaning |
|---|---|
| Capacity per train $C_i$ | Per line (defaults below, editable) or one uniform value. A uniform value is disclosed with every result. |
| Target utilisation $\rho$ | $C_i^{\text{eff}} = \rho C_i$, the usable share of rated capacity (0 < $\rho$ ≤ 1); 80% by default, a modelling choice. |
| Min / max trains per link per 15 min | Optional bounds. The minimum can be limited to intervals where the link is observed in service. Min ≤ max is enforced. |
| Link-service allocation budget $F_t$ | $\lfloor\beta \sum_{i\in t} f_i\rfloor$: $\beta$ × observed link frequencies in interval $t$. |

**The allocation budget is not a fleet size.** One physical train runs over many consecutive links, so summed link
frequencies are service allocations, not distinct trains.

### Default capacities (`config/capacity_by_line.csv`)

These values were checked against public sources in October 2026. The previous file had several values that did not
match TfL's own data sheets, and it covered only the Underground lines.

| Line(s) | Stock | Capacity | Basis / source |
|---|---|---|---|
| Bakerloo | 1972 | 851 | 268 seats + 583 standing @5/m² — TfL RS Info Sheets 4th ed. |
| Central | 1992 | 1,047 | 272 + 775 — same |
| Waterloo & City | 1992 (4-car) | 506 | 136 + 370 — same |
| Piccadilly | 1973 | 798 | 228 + 570 — same |
| Northern | 1995 | 752 | 200 + 552 (48 tip-up seats excluded) — same |
| Jubilee | 1996 (7-car) | 964 | 234 + 730 (perch seats excluded) — same |
| Victoria | 2009 | 986 | 252 + 734 — same |
| District, H&C and Circle | S7 | 1,045 | 256 + 789 — same |
| Metropolitan | S8 | 1,176 | 306 + 870 — same |
| Elizabeth line | Class 345 (9-car) | 1,497 | TfL "maximum full load" — Class 345 data sheet |
| LO Mildmay, LO Windrush | Class 378 (5-car) | 840 | TfL "maximum full load (seated + 70% crush)" — Class 378 data sheet |
| LO Liberty, Weaver, Suffragette, Lioness | Class 710 (4-car) | 678 | 189 + 489, Wikipedia (no TfL sheet found) |
| DLR | B07/B2K 3-car | 688 | Wikipedia (secondary; 2-car trains are smaller, B23 units entered service in Sept 2025) |
| London Trams | Variobahn | 206 | Wikipedia (the workbook shows 0 tram frequency) |

The bases are not identical across modes: LU uses 5 standing passengers/m², while Elizabeth line and Overground use
TfL's "full load". Every row has a source URL. Secondary sources are labelled. Edit the values in the sidebar.

Sources:
[TfL Rolling Stock Information Sheets, 4th ed.](https://foi.tfl.gov.uk/FOI-0237-2223/RS%20Info%20Sheets%204%20Edition.pdf) ·
[Class 345 data sheet](https://foi.tfl.gov.uk/FOI-0416-2223/Class%20345%20Data%20Sheet%20ver%203.pdf) ·
[Class 378 data sheet](https://foi.tfl.gov.uk/FOI-0117-2425/Class%20378%20Data%20Sheet.pdf) ·
[Class 710](https://en.wikipedia.org/wiki/British_Rail_Class_710) ·
[DLR B07](https://en.wikipedia.org/wiki/Docklands_Light_Railway_B07_Stock) ·
[Tramlink](https://en.wikipedia.org/wiki/Tramlink)

## The three models

Index $i$ = one (link, 15-minute interval) observation; $t$ = interval. $D_i$ is observed demand, $f_i$ is observed
trains, $x_i \in \mathbb{Z}_{\ge0}$ is train allocations, and $y_i \in \mathbb{Z}_{\ge0}$ is passengers served. Demand $D_i$ is rounded to whole passengers (NUMBAT loads are fractional modelled estimates) and $C_i^{\text{eff}}=\lfloor\rho C_i\rfloor$.

### Model 1 — Minimum train frequency (integer programming)

$$\min \sum_i x_i \quad\text{s.t.}\quad C_i^{\text{eff}}x_i \ge D_i,\quad l_i \le x_i \le u_i,\quad \Big[\textstyle\sum_{i\in t}x_i \le F_t\Big]$$

Without a budget the problem separates, and $x_i^* = \max(l_i,\lceil D_i/C_i^{\text{eff}}\rceil)$. The tests check the
solver against this closed form. If the maximum frequency or budget cannot cover demand, the model is reported as
infeasible, with the conflicting links or intervals listed. It is never relaxed.

*Question answered:* what is the minimum frequency that carries observed demand under the capacity assumptions?

### Model 2 — Capacity-constrained service allocation (MILP)

$$\max\sum_i y_i\quad\text{s.t.}\quad 0\le y_i\le D_i,\quad y_i\le C_i^{\text{eff}}x_i,\quad \sum_{i\in t}x_i\le F_t\;\forall t,\quad l_i\le x_i\le u_i$$

Stage 2 minimises $\sum_i x_i$ while holding $\sum_i y_i$ at its stage-1 optimum, which removes allocations that serve
no one. The technical bound $u_i \le \lceil D_i/C_i^{\text{eff}}\rceil$ is not restrictive, because extra trains cannot
serve more passengers. The output lists binding budget intervals.

*Question answered:* with a limited allocation budget, where should trains go to serve the most passengers?

### Model 3 — Weighted goal programming

Hard constraints: $y_i \le D_i$, $y_i \le C_i x_i$ (rated capacity), bounds, and an optional budget. Goals use
non-negative deviation variables and **percentage normalisation** (each deviation divided by its goal's target):

| Goal | Constraint | $z_k$ |
|---|---|---|
| G1 efficiency | $\sum x_i - E = p_1 - n_1$, $E=\sum\max(l_i,\lceil D_i/C^{\text{eff}}_i\rceil)$ | $p_1/E$ |
| G2 demand | unmet $=\sum(D_i-y_i)$ | unmet$/\sum D_i$ |
| G3 service pattern | $x_i - f_i = p_{3i}-n_{3i}$ | $\sum(p_{3i}+n_{3i})/\sum f_i$ |
| G4 utilisation | $D_i-\rho C_i x_i = p_{4i}-n_{4i}$ | $\sum(p_{4i}+n_{4i})/\sum D_i$ |

$$\min\sum_k w_k z_k$$

- **Weights**: start equal. They are user settings, not TfL values.
- **G3 target**: the observed frequency, which is always available because Model 3 only uses matched observations.
- **G2 vs G4**: G2 counts passengers above *rated* capacity, while G4 measures distance from the *target* load
  $\rho C$. NUMBAT does not observe crowding directly, so G4 is derived from link loads and the capacity assumption.
- **Ties**: a $10^{-3}$ tie-break on every goal keeps solutions Pareto-efficient when a weight is 0.
- **Scaling**: the objective is multiplied by a constant for numerical conditioning, which does not change the optimum.
  This matters because, unscaled, HiGHS presolve treated the tiny coefficients as zero.

The app shows each goal's normalised deviation for both the observed timetable and the plan.

*Question answered:* which plan gives the best compromise between efficiency, demand satisfaction and service quality
under the chosen priorities?

### Solver handling and validation

HiGHS runs through Pyomo with a time limit and a MIP gap of 0.01%. Results are shown only when a solution exists, and
"Optimal" only when HiGHS proves it and the primal and dual bounds agree. After every solve, independent checks verify
integrality, bounds, $y\le D$, $y\le$ capacity × $x$, Model 1 demand coverage, and the budget. A failed check means no
result is shown.

## Sensitivity analysis

On the **Sensitivity Analysis** tab, pick a model and scenarios, then press **Run Sensitivity Analysis**:

- **A** baseline
- **B** $D_i' = 1.10D_i$
- **C** $D_i' = 1.20D_i$
- **D** the budget reduced by the chosen % (a cut to the modelled allocation budget, not to TfL's fleet)
- **E** user-chosen alternative $\rho$ values

The comparison covers allocations, demand served or unmet, satisfaction, utilisation, under-capacity links and
feasibility. Infeasible scenarios are reported as infeasible.

## Interpreting results

- **Observed** values: demand and existing frequency.
- **Your assumptions**: capacity and ρ.
- **Derived** values: offered capacity, utilisation and shortfalls.
- **Recommendations**: optimised frequency and demand served.
- `existing_*` columns evaluate the observed timetable under the same assumptions.

With a minimum frequency of 0, the models size service to passenger volume alone. They usually recommend far fewer
trains than TfL runs, because real timetables also follow headway and waiting-time standards, which you can represent
with the minimum-frequency setting.

## Assumptions and limitations

- Train capacity is **not** in NUMBAT. It must be configured, and the defaults are documented above. One uniform value
  ignores differences between lines unless per-line values are used.
- The frequency-derived allocation budget is not a physical fleet inventory.
- Link-level allocation is not a timetable. It does not cover train circulation, rolling-stock scheduling or
  route/branch consistency: neighbouring links can receive different frequencies.
- Link loads are modelled NUMBAT estimates for a typical autumn day. Demand is counted per link, so one passenger
  appears on every link they ride.
- The network chart is a schematic topology (force-directed layout), not a geographic TfL map.
- Missing and ambiguous observations are excluded and disclosed, which can understate demand on affected links.
