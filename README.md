# Data-Driven Urban Rail Train Frequency and Capacity Optimization

An academic decision-support system using official TfL NUMBAT data. It profiles and cleans the workbook, reconstructs a link-level network, analyzes demand and scheduled frequency, and solves configurable Pyomo/HiGHS service models.

## Models

- **Model 1:** integer minimum service frequency by service group and 15-minute period, subject to configured capacity, bounds, and optional train-equivalent budget.
- **Model 2:** integer service allocation maximizing served line-boarder demand under a budget.
- **Model 3:** weighted goal programming for unmet demand, target-frequency deviation, crowding, and service usage.

These are academic recommendations under explicit assumptions, not TfL timetables.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place the official NUMBAT XLSX in `data/raw/`. The preferred release is the 2025 Tuesday-Wednesday-Thursday profile. Then run:

```powershell
python scripts/profile_dataset.py
python scripts/preprocess.py
# complete and verify config/capacity_by_line.csv and config/service_constraints.csv
python scripts/run_model1.py
python scripts/run_model2.py
python scripts/run_model3.py
python scripts/run_scenarios.py
streamlit run app.py
```

See [instructions.md](instructions.md) for the full workflow, configuration meanings, data dictionary, validation, troubleshooting, and viva guidance.

## Structure

`src/ingestion` profiles workbooks; `src/preprocessing` creates canonical Parquet; `src/network` builds directed station-link graphs; `src/analysis` provides EDA; `src/models` contains the optimization formulations; `src/validation` checks outputs; `src/scenarios` reruns models with multipliers. Generated files go to `data/processed/` and `outputs/`.

## Technology

Python 3.11+, pandas, NumPy, OpenPyXL, PyArrow, Pyomo, HiGHS/highspy, NetworkX, Plotly, Streamlit, PyYAML, and pytest.

## Limitations

NUMBAT is a typical demand profile. Link Frequency is scheduled supply, not observed operation. Capacity, service bounds, and fleet budgets must be sourced and entered by the user. The model does not simulate train circulation, crew, depots, disruptions, or stochastic demand.
