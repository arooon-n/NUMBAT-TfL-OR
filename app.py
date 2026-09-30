from pathlib import Path
import os
import streamlit as st
import pandas as pd
import plotly.express as px
from src.ingestion.loader import find_workbook, profile_workbook, write_json
from src.utils.config import read_config_csv, read_yaml

st.set_page_config(page_title="NUMBAT Rail Service Optimizer", layout="wide")
st.title("TfL NUMBAT Rail Frequency and Capacity Optimization")
st.caption("Academic decision-support model. Results are recommendations under configured assumptions, not TfL operational timetables.")
try:
    workbook=find_workbook(os.getenv("DATA_FILE")); profile=profile_workbook(workbook); write_json(profile,Path("data/processed/dataset_profile.json"))
except Exception as exc:
    st.error(str(exc)); st.info("Download an official NUMBAT workbook from https://crowding.data.tfl.gov.uk/, place it in data/raw/, then rerun the profiler."); st.stop()

st.sidebar.header("Data overview")
st.sidebar.write({"workbook": workbook.name, "selected sheet": profile.get("selected_sheet"), "valid": profile.get("valid")})
if not Path("data/processed/cleaned_data.parquet").exists():
    st.warning("Run `python scripts/preprocess.py` before exploring or optimizing."); st.stop()
frame=pd.read_parquet("data/processed/cleaned_data.parquet")
tabs = st.tabs(["Overview", "Exploration", "Optimization", "Scenarios", "Validation", "Export"])
with tabs[0]:
    col1,col2,col3,col4=st.columns(4)
    col1.metric("Rows",len(frame)); col2.metric("Lines",frame.line.nunique()); col3.metric("Stations",len(set(frame.from_station.dropna())|set(frame.to_station.dropna()))); col4.metric("Periods",frame.time_period.nunique())
    st.json({"workbook": workbook.name, "sheet": profile.get("selected_sheet"), "wide_layout": profile.get("wide_numbat_layout"), "missing_frequency_values": int(frame.link_frequency.isna().sum())})
with tabs[1]:
    left,right=st.columns(2)
    with left: st.plotly_chart(px.line(frame.groupby("time_period",as_index=False).link_load.sum(),x="time_period",y="link_load",title="Link load by time"),use_container_width=True)
    with right: st.plotly_chart(px.line(frame.groupby("time_period",as_index=False).link_frequency.mean(),x="time_period",y="link_frequency",title="Scheduled frequency by time"),use_container_width=True)
    st.plotly_chart(px.bar(frame.groupby("line",as_index=False).link_load.sum().sort_values("link_load",ascending=False),x="line",y="link_load",title="Demand by line"),use_container_width=True)
    st.dataframe(frame.sort_values("link_load",ascending=False).head(25),use_container_width=True)
with tabs[2]:
    selected=st.selectbox("Model",["Model 1","Model 2","Model 3"])
    st.write({"demand multiplier": 1.0, "target utilization": read_yaml("config/scenario_parameters.yaml").get("target_utilization", "not configured")})
    result_path={"Model 1":"outputs/model1_results.csv","Model 2":"outputs/model2_results.csv","Model 3":"outputs/model3_results.csv"}[selected]
    if Path(result_path).exists(): st.dataframe(pd.read_csv(result_path),use_container_width=True)
    else: st.warning(f"No {selected} result yet. Verify configuration, then run the matching script in instructions.md.")
    st.info("Capacity, bounds, targets, weights, and optional budgets are read from config files. No operational values are invented by this dashboard.")
with tabs[3]:
    path=Path("outputs/scenario_results.csv")
    if path.exists(): st.dataframe(pd.read_csv(path),use_container_width=True)
    else: st.warning("Run scripts/run_scenarios.py after configuring capacities and constraints.")
with tabs[4]:
    path=Path("outputs/validation_report.json")
    if path.exists(): st.json(__import__("json").loads(path.read_text(encoding="utf-8")))
    else: st.warning("Run scripts/run_validation.py after producing model results.")
with tabs[5]:
    for path in sorted(Path("outputs").glob("*.csv")):
        st.download_button(f"Download {path.name}",path.read_bytes(),file_name=path.name,mime="text/csv")
    for path in [Path("outputs/validation_report.json"),Path("outputs/validation_report.txt")]:
        if path.exists(): st.download_button(f"Download {path.name}",path.read_bytes(),file_name=path.name)
