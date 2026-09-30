# Instructions: TfL NUMBAT Rail Optimization

## 1. Project overview

This project is an explainable academic decision-support model for comparing NUMBAT scheduled service with service recommendations generated from configured demand, capacity, frequency bounds, and an optional train-equivalent budget. It must never be presented as an actual TfL timetable.

NUMBAT variables are interpreted as follows:

- **Link Load:** demand/load between consecutive stations. It is used for link capacity and crowding.
- **Link Frequency:** scheduled trains per 15-minute period. It is planned supply, not real-time operation.
- **Line Boarders:** line-level boardings. Model 2 and Model 3 use this for line demand coverage.
- **Station Boarders/Alighters and Entry/Exit:** exploratory station analysis.
- **Journey Time:** descriptive analysis only.
- **OD Flow:** validation/descriptive analysis; link loads are never summed and called unique journeys.

## 2. Official data acquisition

Download the official workbook from <https://crowding.data.tfl.gov.uk/>. Prefer the NUMBAT 2025 Tuesday-Wednesday-Thursday weekday profile. Verify the file is an XLSX from the official TfL domain, record its release/year/day type, and retain the original file unchanged.

Put the workbook in `data/raw/`. The application does not assume a filename. With another location, set `DATA_FILE`:

```powershell
$env:DATA_FILE = 'D:\path\to\official_numbat.xlsx'
python scripts/profile_dataset.py
```

Generic command shells may use `DATA_FILE=/path/to/workbook.xlsx python scripts/profile_dataset.py`.

## 3. Installation

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If script activation is restricted, use `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and prefix commands with `\.venv\Scripts\python.exe`.

Generic Python:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## 4. Exact first run

1. Download the official NUMBAT workbook.
2. Put it in `data/raw/`, or set `DATA_FILE`.
3. Run `python scripts/profile_dataset.py`.
4. Inspect `data/processed/dataset_profile.json`; confirm the selected sheet and aliases.
5. If necessary, set `sheet` and explicit `columns` in `config/schema_mapping.yaml`.
6. Enter verified train capacities in `config/capacity_by_line.csv`.
7. Enter frequency bounds and targets in `config/service_constraints.csv`.
8. Optionally enter a train-equivalent budget in `config/fleet_budget.csv`.
9. Run `python scripts/preprocess.py`.
10. Run `python scripts/run_model1.py`.
11. Run `python scripts/run_model2.py`.
12. Run `python scripts/run_model3.py`.
13. Run `python scripts/run_scenarios.py`.
14. Launch `streamlit run app.py`.

## 5. Manual inputs and their meaning

| File                              | Field                        | Meaning and unit                        | Source/status                                                 | If omitted                                           |
| --------------------------------- | ---------------------------- | --------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------- |
| `config/capacity_by_line.csv`     | `capacity_per_train`         | passengers per train                    | mandatory external operational input; document source and URL | Models requiring capacity stop; no invented capacity |
| same                              | `service_group`              | frequency decision entity               | verify from route/branch evidence                             | defaults are not safe for branches; add mapping      |
| `config/service_constraints.csv`  | `min_frequency_per_15min`    | integer trains per 15 minutes           | user-verified assumption or source                            | model cannot run without constraints                 |
| same                              | `max_frequency_per_15min`    | integer trains per 15 minutes           | user-verified bound                                           | model cannot run safely                              |
| same                              | `target_frequency_per_15min` | goal-programming target                 | configured target                                             | Model 3 cannot run meaningfully                      |
| `config/fleet_budget.csv`         | `fleet_budget`               | train-equivalent services per period    | optional explicit budget, not physical TfL fleet              | budget constraint is skipped                         |
| `config/goal_weights.yaml`        | four weights                 | relative penalty weights, dimensionless | mandatory for declared Model 3 policy                         | defaults are templates; verify/change them           |
| `config/scenario_parameters.yaml` | multipliers/target           | sensitivity inputs                      | editable assumptions                                          | scenario command uses listed values                  |
| `config/schema_mapping.yaml`      | `sheet`, `columns`           | workbook mapping overrides              | required only when auto-detection is ambiguous                | profiler stops rather than guessing                  |
| `config/service_groups.csv`       | branch fields                | verified link-to-service grouping       | required where branches cannot be inferred                    | keep branch mapping explicit                         |

The capacity source, source URL, definition, constraints, and assumptions must be recorded before presentation. A fleet budget here is a simplified train-equivalent service budget, not TfL physical fleet size.

## 6. Profiling and preprocessing

`profile_dataset.py` lists sheets, dimensions, columns, aliases, duplicates, nulls, negatives, and candidate link sheets. It writes `data/processed/dataset_profile.json`. It fails if a unique valid link sheet cannot be identified. This is intentional: the software will not guess a schema.

`preprocess.py` standardizes canonical names, trims text, normalizes numeric measures, removes exact accidental duplicate rows, rejects negative loads/frequencies, and writes `data/processed/cleaned_data.parquet`. Raw files are never modified. Missing demand is not silently filled.

Canonical required link fields are `line`, `from_station`, `to_station`, `link_load`, and `link_frequency`. `time_period` and `line_boarders` are required by relevant analyses/models.

## 7. Network and analysis

`src/network/graph.py` creates a directed MultiDiGraph from consecutive station links, preserving line, direction, period, load, and frequency edge attributes. It reports station count, link count, weak connectivity, and branch nodes. The graph does not force a simple corridor.

Reusable EDA functions cover demand by time, scheduled frequency, top links, and capacity-based baseline utilization. Baseline utilization is only valid when `capacity_per_train` is supplied:

`link_load / (capacity_per_train * scheduled_frequency)`.

## 8. Optimization formulations

Model 1 uses integer `x[r,t]`, minimizes total train-equivalent service, and constrains every link by `capacity[r] * x[r,t] >= link_load * demand_multiplier`. The analytical check is `ceil(max link load / capacity)` when bounds and budgets do not alter the result.

Model 2 uses integer `x[r,t]` and continuous `y[r,t]` for served **line-boarder demand**. It enforces `y <= demand`, `y <= capacity*x`, frequency bounds, and an optional budget. It does not claim unique passenger journeys from summed link loads.

Model 3 uses explicit goal/deviation variables for unmet boarder demand, target-frequency deviations, crowding excess over the configured utilization target, and service usage. Weights come from `config/goal_weights.yaml`, not hidden constants.

HiGHS/highspy is used through Pyomo. Every result carries solver status and termination. Infeasible results must be investigated through constraints and configuration; they are not replaced with zeros.

## 9. Scenarios and outputs

`run_scenarios.py` reuses Model 1 for baseline, +10%, and +20% demand. Extend the runner for reduced budgets or alternate utilization targets after entering those explicit scenario values. Generated outputs include:

- `outputs/model1_results.csv`
- `outputs/model2_results.csv`
- `outputs/model3_results.csv`
- `outputs/scenario_results.csv`
- `outputs/validation_report.json` and `.txt`
- `data/processed/dataset_profile.json`

Each optimization run should be archived with its configuration, dataset hash, model parameters, solver, status, and figures before results are used in a presentation. Do not overwrite a result that needs to be reproduced.

## 10. Dashboard

Run:

```powershell
streamlit run app.py
```

The current dashboard shows the verified dataset overview, demand by time, demand by line, and high-load links. It refuses to show optimization results until the workbook is profiled and cleaned, and it clearly reports missing data/configuration. Run the model scripts first to populate result files.

## 11. Validation checklist

Check solver status, termination, feasibility, integer frequencies, frequency bounds, capacity inequalities, optional budgets, served demand limits, nonnegative values, and unit consistency. For Model 1 compare solver frequency with the analytical ceiling under unconstrained settings. Keep validation reports with the run.

## 12. Troubleshooting

- **No workbook:** place an official `.xlsx` in `data/raw/` or set `DATA_FILE`.
- **Multiple workbooks:** set `DATA_FILE`; do not let the application choose silently.
- **Sheet mismatch/missing columns:** inspect `dataset_profile.json`, then add exact `sheet`/`columns` mappings.
- **Missing capacity:** populate `capacity_by_line.csv` with a source; never guess.
- **Missing 15-minute periods:** investigate release/day-type coverage; do not invent rows.
- **Duplicate records:** review profiler counts and determine whether they are accidental or meaningful dimensions.
- **Invalid line/branch mapping:** use verified rows in `service_groups.csv`.
- **Infeasible model:** relax or correct contradictory min/max/budget/capacity inputs and inspect solver termination.
- **HiGHS unavailable:** activate the project environment and run `python -m pip install -r requirements.txt`.
- **Wrong paths:** run commands from the project root or use absolute `DATA_FILE`.
- **Parquet error:** install `pyarrow` in the active environment.

## 13. Interpretation and viva preparation

Report baseline scheduled frequency separately from optimized integer recommendations. Key metrics are frequency change, served line-boarder demand, unmet demand, coverage ratio, train-equivalent usage, and link utilization. A reduction means only that the configured mathematical model finds less service sufficient under its assumptions; it is not an operational deployment recommendation.

For a viva, explain that NUMBAT is a TfL crowding/demand data source selected for official, time-banded rail demand. Link Load is between consecutive stations; Link Frequency is scheduled supply. Line Boarders are used in allocation so demand is not double-counted across links. Capacities and fleet budgets are external because NUMBAT does not supply all operational parameters. Model 1 is an integer service minimization problem, Model 2 is a capacity-constrained allocation problem, and Model 3 is weighted goal programming. “Optimal” means optimal for the stated objective, data, constraints, and assumptions, not globally optimal TfL operations.

Limitations include typical-day demand, scheduled rather than observed service, externally configured capacity, simplified service groups, no train circulation/crew/depot/disruption/stochastic model, and no claim of actual timetable feasibility.

## 14. Final presentation checklist

- [ ] Correct dataset year and day type
- [ ] Official source and download date documented
- [ ] Workbook profile inspected
- [ ] Capacity sources and definitions documented
- [ ] Bounds, targets, weights, and budget documented
- [ ] Branch/service mappings verified
- [ ] Solver status successful
- [ ] Validation passed
- [ ] Baseline versus optimized comparison generated
- [ ] Scenario results generated
- [ ] Figures and raw/processed distinction documented
- [ ] Dashboard working
- [ ] Run configuration and dataset hash archived
- [ ] Limitations stated clearly
