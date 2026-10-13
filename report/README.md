# IEEE report

`main.tex` is the IEEE conference-format report (IEEEtran). `main.pdf` is a compiled copy, with screenshot
placeholders until the screenshots are added.

## Compile on Overleaf

1. Zip the whole `report/` folder (including `data/` and `figures/`) and upload it as a new Overleaf project.
2. Compiler: **pdfLaTeX**. Main document: `main.tex`.

## Add the UI screenshots

Save these five PNGs in `report/figures/` with exactly these names. Each one replaces its placeholder automatically.

| File | What to capture |
|---|---|
| `ui_overview.png` | **Overview** tab after uploading the workbook, with the sidebar visible (readiness row, KPIs, demand/service chart) |
| `ui_data_quality.png` | **Data Quality** tab, scrolled to show the "Passengers with zero scheduled trains" section (metrics + tables) |
| `ui_explore.png` | **Explore Data** tab: the demand heatmap or the schematic network |
| `ui_optimization.png` | **Optimization** tab after running **Model 3**: assumptions box, solver status, KPI rows and interpretation |
| `ui_sensitivity.png` | **Sensitivity Analysis** tab after **Run Sensitivity Analysis**: the table and chart |

Use a wide browser window (about 1600 px) so the text stays legible at page width.

## Regenerating the numbers

Every number, table and chart in the report comes from the app's own code:

```powershell
.venv\Scripts\python.exe report\generate_report_data.py   # runs all experiments (~3 min)
.venv\Scripts\python.exe report\make_tables.py            # writes data\tab_*.tex
```
