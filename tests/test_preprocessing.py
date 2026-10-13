import numpy as np
import pytest

from conftest import REAL_WORKBOOK, link_row, make_workbook
from data_loader import load_workbook_bytes
from preprocessing import preprocess, readiness


def process(loads, freqs, intervals):
    return preprocess(load_workbook_bytes(make_workbook(loads, freqs, intervals)))


def test_join_by_key_not_row_position(intervals):
    loads = [link_row("A>B", "L1", "NB", 1, "A", "B", [10, 20, 30, 40]),
             link_row("B>C", "L1", "NB", 2, "B", "C", [1, 2, 3, 4])]
    freqs = [link_row("B>C", "L1", "NB", 2, "B", "C", [5, 6, 7, 8]),   # reversed row order
             link_row("A>B", "L1", "NB", 1, "A", "B", [1, 1, 2, 2])]
    p = process(loads, freqs, intervals)
    row = p.data[(p.data.link_id == "B>C") & (p.data.interval == "0515-0530")].iloc[0]
    assert row.demand == 2 and row.frequency == 6
    assert p.key_fields == ["link_id"]
    assert "total" not in p.data.columns            # daily totals never enter the interval table
    assert list(p.data[p.data.link_id == "A>B"].interval) == intervals  # operating-day order across midnight


def test_missing_invalid_negative_and_zero(intervals):
    loads = [link_row("A>B", "L1", "NB", 1, "A", "B", [None, "n/a", -5, 0])]
    freqs = [link_row("A>B", "L1", "NB", 1, "A", "B", [1, 1, 1, 1])]
    p = process(loads, freqs, intervals)
    d = p.data.set_index("interval")
    assert d.loc["0500-0515", "demand_status"] == "missing" and np.isnan(d.loc["0500-0515", "demand"])
    assert d.loc["0515-0530", "demand_status"] == "non_numeric"
    assert d.loc["2345-0000", "demand_status"] == "negative"
    assert d.loc["0000-0015", "demand_status"] == "valid" and d.loc["0000-0015", "demand"] == 0  # genuine zero kept
    assert p.metrics["eligible_demand"] == 1
    assert p.sheet_metrics.set_index("sheet").loc["Link_Loads", "non_numeric_values"] == 1


def test_duplicates_and_unmatched_links(intervals):
    row = link_row("A>B", "L1", "NB", 1, "A", "B", [1, 2, 3, 4])
    loads = [row, row,                                                     # exact duplicate: dropped once
             link_row("C>D", "L1", "NB", 2, "C", "D", [1, 1, 1, 1]),
             link_row("C>D", "L1", "NB", 2, "C", "D", [9, 9, 9, 9]),        # same key, different values: ambiguous
             link_row("E>F", "L1", "NB", 3, "E", "F", [1, 1, 1, 1])]        # no frequency record
    freqs = [row, link_row("C>D", "L1", "NB", 2, "C", "D", [1, 1, 1, 1])]
    p = process(loads, freqs, intervals)
    assert p.sheet_metrics.set_index("sheet").loc["Link_Loads", "exact_duplicate_rows"] == 1
    assert (p.data[p.data.link_id == "C>D"].demand_status == "ambiguous_link").all()
    ef = p.data[p.data.link_id == "E>F"]
    assert (ef.frequency_status == "no_frequency_record").all()
    assert ef.eligible_demand.all() and not ef.eligible_service.any()
    assert p.metrics["links_only_in_loads"] == 1
    assert p.data[p.data.link_id == "A>B"].eligible_service.all()


def test_zero_train_demand_is_flagged(intervals):
    loads = [link_row("A>B", "L1", "NB", 1, "A", "B", [5, 20, 0, 3])]
    freqs = [link_row("A>B", "L1", "NB", 1, "A", "B", [0, 2, 0, 0])]
    p = process(loads, freqs, intervals)
    d = p.data.set_index("interval")
    assert d.loc["0500-0515", "zero_train_demand"] and d.loc["0500-0515", "zero_train_adjacent_service"]
    assert not d.loc["2345-0000", "zero_train_demand"]          # no passengers, no trains: consistent
    assert d.loc["0000-0015", "zero_train_demand"] and not d.loc["0000-0015", "zero_train_adjacent_service"]
    assert p.metrics["zero_train_obs"] == 2 and p.metrics["zero_train_demand"] == 8
    assert d.loc["0500-0515", "demand"] == 5                    # observation kept, only flagged


def test_readiness_without_frequencies(intervals):
    loads = [link_row("A>B", "L1", "NB", 1, "A", "B", [1, 2, 3, 4])]
    p = preprocess(load_workbook_bytes(make_workbook(loads, [], intervals, include_freq=False)))
    r = readiness(p)
    assert r["Model 1"][0] and not r["Model 2"][0] and not r["Model 3"][0]


@pytest.mark.skipif(not REAL_WORKBOOK.exists(), reason="supplied NUMBAT workbook not present")
def test_supplied_workbook_preprocesses():
    p = preprocess(load_workbook_bytes(REAL_WORKBOOK.read_bytes(), REAL_WORKBOOK.name))
    assert p.metrics["intervals"] == 96
    assert p.metrics["links"] == 1214
    assert p.metrics["links_only_in_loads"] == 0 and p.metrics["links_only_in_frequencies"] == 0
    assert p.metrics["eligible_service"] > 0.95 * p.metrics["observations"]
    # one row per link and interval
    assert not p.data.duplicated(["link_key", "interval"]).any()
