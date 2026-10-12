"""Shared helpers: build small synthetic NUMBAT-style workbooks in memory (unit tests only)."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REAL_WORKBOOK = ROOT / "data" / "raw" / "NBT25TWT_Outputs.xlsx"
FIELDS = ["Link", "Line", "Dir", "Order", "From NLC", "From ASC", "From Station",
          "To NLC", "To ASC", "To Station", "Total", "Early", "AM Peak"]


def link_row(link, line, direction, order, frm, to, values):
    nums = [v for v in values if isinstance(v, (int, float))]
    return [link, line, direction, order, order * 10, f"{frm[:3]}u", frm, order * 10 + 10, f"{to[:3]}u", to,
            sum(nums), 0, 0] + list(values)


def make_workbook(loads_rows, freq_rows, intervals, title_rows=2, include_freq=True) -> bytes:
    """Wide layout like NUMBAT: title rows, a header row, then one row per link."""
    wb = Workbook()
    wb.active.title = "_Cover"
    wb.active["B6"] = "Link_Loads: Demand on trains between platforms. " * 3
    sheets = [("Link_Loads", "Link Loads", loads_rows)]
    if include_freq:
        sheets.append(("Link_Frequencies", "Link Frequency", freq_rows))
    for name, title, rows in sheets:
        ws = wb.create_sheet(name)
        ws.append([title])
        for _ in range(title_rows - 1):
            ws.append([None] * len(FIELDS) + ["quarter hour (qhr)"])
        ws.append(FIELDS + list(intervals))
        for r in rows:
            ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@pytest.fixture
def intervals():
    return ["0500-0515", "0515-0530", "2345-0000", "0000-0015"]
