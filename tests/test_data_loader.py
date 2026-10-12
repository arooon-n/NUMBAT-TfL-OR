import pytest

from conftest import REAL_WORKBOOK, link_row, make_workbook
from data_loader import interval_order, load_workbook_bytes, parse_interval


def test_parse_interval_variants():
    assert parse_interval("0800-0815") == "0800-0815"
    assert parse_interval("08:00 - 08:15") == "0800-0815"
    assert parse_interval("2345-0000") == "2345-0000"      # crosses midnight
    assert parse_interval("0800-0830") is None              # not 15 minutes
    assert parse_interval("Total") is None
    assert parse_interval(None) is None


def test_interval_order_starts_at_0500_and_wraps_midnight():
    labels = ["0000-0015", "0445-0500", "0500-0515", "2345-0000", "1200-1215"]
    ordered = sorted(labels, key=interval_order)
    assert ordered == ["0500-0515", "1200-1215", "2345-0000", "0000-0015", "0445-0500"]
    assert interval_order("0500-0515") == 0 and interval_order("0445-0500") == 95


def test_header_and_intervals_detected_without_fixed_positions(intervals):
    rows = [link_row("A>B", "L1", "NB", 1, "A", "B", [1, 2, 3, 4])]
    for title_rows in (1, 3):  # header row position varies
        wb = load_workbook_bytes(make_workbook(rows, rows, intervals, title_rows=title_rows))
        assert not wb.errors
        assert wb.loads.header_row == title_rows
        assert wb.loads.fields["from_station"] == "From Station"
        assert wb.loads.fields["link_id"] == "Link"
        assert list(wb.loads.interval_columns.values()) == intervals
        assert wb.loads.period_columns == ["Early", "AM Peak"]
        assert wb.cover_notes


def test_missing_frequency_sheet_is_reported_not_fatal(intervals):
    rows = [link_row("A>B", "L1", "NB", 1, "A", "B", [1, 2, 3, 4])]
    wb = load_workbook_bytes(make_workbook(rows, [], intervals, include_freq=False))
    assert wb.loads is not None and wb.frequencies is None
    assert any("Link_Frequencies" in e for e in wb.errors)


def test_non_excel_file_reports_error():
    wb = load_workbook_bytes(b"not an excel file", "bad.xlsx")
    assert wb.errors and wb.loads is None


@pytest.mark.skipif(not REAL_WORKBOOK.exists(), reason="supplied NUMBAT workbook not present")
def test_supplied_workbook_parses():
    wb = load_workbook_bytes(REAL_WORKBOOK.read_bytes(), REAL_WORKBOOK.name)
    assert not wb.errors
    assert {"Link_Loads", "Link_Frequencies"} <= set(wb.sheet_names)
    for sheet in (wb.loads, wb.frequencies):
        labels = list(sheet.interval_columns.values())
        assert len(labels) == 96
        assert labels[0] == "0500-0515" and labels[-1] == "0445-0500"
        assert sorted(labels, key=interval_order) == labels
        for f in ["link_id", "line", "direction", "from_station", "to_station", "total"]:
            assert f in sheet.fields
