"""Excel parsing and schema detection for TfL NUMBAT workbooks.

NUMBAT link sheets are wide: a few title rows, one header row with the field
names, then one row per link with one column per 15-minute interval.  Nothing
here assumes fixed row or column positions: the header row, the link fields
and the interval columns are all discovered from the cell contents.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import pandas as pd
from openpyxl import load_workbook

DATA_SOURCE_URL = "https://crowding.data.tfl.gov.uk/"

# Canonical field -> accepted header spellings (compared after normalisation).
FIELD_ALIASES: dict[str, list[str]] = {
    "link_id": ["link", "link id", "link code", "link name"],
    "line": ["line", "line name", "service line"],
    "direction": ["dir", "direction"],
    "order": ["order", "sequence", "seq", "link order"],
    "from_nlc": ["from nlc", "origin nlc"],
    "from_asc": ["from asc", "origin asc"],
    "from_station": ["from station", "origin station", "from stn", "from"],
    "to_nlc": ["to nlc", "destination nlc"],
    "to_asc": ["to asc", "destination asc"],
    "to_station": ["to station", "destination station", "to stn", "to"],
    "total": ["total", "daily total", "total day"],
}
PERIOD_COLUMNS = ["early", "am peak", "midday", "pm peak", "evening", "late"]
REQUIRED_LINK_FIELDS = ["line", "from_station", "to_station"]

LINK_SHEETS = {
    "loads": ["link loads", "link load", "linkloads"],
    "frequencies": ["link frequencies", "link frequency", "linkfrequencies", "link freq"],
}

INTERVAL_RE = re.compile(r"^\s*(\d{1,2}):?(\d{2})\s*[-–—to]+\s*(\d{1,2}):?(\d{2})\s*$")
DAY_START_MINUTE = 5 * 60  # NUMBAT operating day runs 05:00 -> 04:59


def normalise(text) -> str:
    """Lower-case, collapse punctuation/underscores to single spaces."""
    return re.sub(r"[^a-z0-9&]+", " ", str(text).lower()).strip()


def parse_interval(label) -> str | None:
    """Return a canonical 'HHMM-HHMM' label for a 15-minute interval, else None."""
    if label is None:
        return None
    match = INTERVAL_RE.match(str(label))
    if not match:
        return None
    h1, m1, h2, m2 = (int(g) for g in match.groups())
    if not (0 <= h1 <= 24 and 0 <= h2 <= 24 and m1 < 60 and m2 < 60):
        return None
    start, end = (h1 * 60 + m1) % 1440, (h2 * 60 + m2) % 1440
    if (end - start) % 1440 != 15:  # crossing midnight handled by the modulo
        return None
    return f"{h1 % 24:02d}{m1:02d}-{h2 % 24:02d}{m2:02d}"


def interval_order(label: str) -> int:
    """Position of an interval in the operating day (0 = 0500-0515, 95 = 0445-0500)."""
    start = int(label[:2]) * 60 + int(label[2:4])
    return ((start - DAY_START_MINUTE) % 1440) // 15


def interval_start_text(label: str) -> str:
    return f"{label[:2]}:{label[2:4]}"


@dataclass
class LinkSheet:
    """One parsed wide-format link sheet."""
    sheet_name: str
    header_row: int                       # 0-based row index in the sheet
    fields: dict[str, str]                # canonical field -> original header
    interval_columns: dict[str, str]      # original header -> canonical interval
    period_columns: list[str]
    data: pd.DataFrame                    # raw rows below the header (object dtype)
    issues: list[str] = field(default_factory=list)


@dataclass
class Workbook:
    name: str
    sheet_names: list[str]
    sheet_sizes: dict[str, tuple[int, int]]
    cover_notes: list[str]
    loads: LinkSheet | None
    frequencies: LinkSheet | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _find_sheet(sheet_names: list[str], aliases: list[str], frames: dict[str, pd.DataFrame]) -> str | None:
    """Find a sheet by name; fall back to a sheet whose title cell matches."""
    wanted = {normalise(a).replace(" ", "") for a in aliases}
    for name in sheet_names:
        if normalise(name).replace(" ", "") in wanted:
            return name
    for name, frame in frames.items():
        top_left = frame.iloc[:3, :3].astype(str).map(lambda v: normalise(v).replace(" ", ""))
        if top_left.isin(wanted).any().any():
            return name
    return None


def _detect_header_row(frame: pd.DataFrame, max_scan: int = 20) -> int | None:
    """Return the row whose cells best match known link fields and interval labels."""
    best_row, best_score = None, 0
    known = {normalise(a) for aliases in FIELD_ALIASES.values() for a in aliases}
    for row in range(min(max_scan, len(frame))):
        cells = frame.iloc[row].tolist()
        fields = sum(normalise(c) in known for c in cells if c is not None)
        intervals = sum(parse_interval(c) is not None for c in cells)
        score = fields * 10 + intervals
        if fields >= 2 and score > best_score:
            best_row, best_score = row, score
    return best_row


def _map_fields(headers: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map canonical fields to headers. Exact alias order decides; ties are reported."""
    issues: list[str] = []
    mapping: dict[str, str] = {}
    by_norm: dict[str, list[str]] = {}
    for h in headers:
        by_norm.setdefault(normalise(h), []).append(h)
    used: set[str] = set()
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            candidates = [h for h in by_norm.get(normalise(alias), []) if h not in used]
            if len(candidates) > 1:
                issues.append(f"Field '{canonical}' matches several columns {candidates}; it was not mapped.")
                break
            if candidates:
                mapping[canonical] = candidates[0]
                used.add(candidates[0])
                break
    return mapping, issues


def parse_link_sheet(frame: pd.DataFrame, sheet_name: str) -> LinkSheet | None:
    frame = frame.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    header_row = _detect_header_row(frame)
    if header_row is None:
        return None
    raw_headers = frame.iloc[header_row].tolist()
    headers, seen = [], {}
    for i, h in enumerate(raw_headers):  # make every header a unique string
        text = str(h).strip() if h is not None and str(h).strip() else f"column_{i}"
        seen[text] = seen.get(text, 0) + 1
        headers.append(text if seen[text] == 1 else f"{text}__{seen[text]}")
    data = frame.iloc[header_row + 1:].copy()
    data.columns = headers
    data = data.dropna(how="all").reset_index(drop=True)

    fields, issues = _map_fields(headers)
    intervals: dict[str, str] = {}
    for h in headers:
        label = parse_interval(h)
        if label is None:
            continue
        if label in intervals.values():
            issues.append(f"Interval {label} appears in more than one column; only the first is used.")
            continue
        intervals[h] = label
    periods = [h for h in headers if normalise(h) in PERIOD_COLUMNS]
    missing = [f for f in REQUIRED_LINK_FIELDS if f not in fields]
    if missing:
        issues.append(f"Required link fields not found: {missing}.")
    if not intervals:
        issues.append("No 15-minute interval columns (e.g. '0800-0815') were detected.")
    return LinkSheet(sheet_name, header_row, fields, intervals, periods, data, issues)


def _cover_notes(frame: pd.DataFrame) -> list[str]:
    notes = []
    for value in frame.to_numpy().ravel():
        if isinstance(value, str) and len(value.strip()) > 60:
            notes.append(value.strip())
    return notes


def load_workbook_bytes(content: bytes, name: str = "workbook.xlsx") -> Workbook:
    """Parse an uploaded NUMBAT workbook. Never raises for content problems:
    errors and warnings are returned on the Workbook object."""
    try:
        book = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    except Exception as exc:  # corrupt or not an xlsx file
        return Workbook(name, [], {}, [], None, None, errors=[f"The file could not be opened as an Excel workbook: {exc}"])
    sheet_names = book.sheetnames
    sizes = {ws.title: (ws.max_row or 0, ws.max_column or 0) for ws in book.worksheets}
    book.close()

    # Only the cover and candidate link sheets are read fully (station sheets are large).
    candidate_names = [n for n in sheet_names if "link" in normalise(n) or "cover" in normalise(n)]
    if not any("link" in normalise(n) for n in candidate_names):
        candidate_names = sheet_names  # unusual naming: inspect every sheet's title cells
    # keep_default_na=False: text such as 'n/a' must be reported as non-numeric, not silently read as missing.
    frames = pd.read_excel(io.BytesIO(content), sheet_name=candidate_names, header=None, dtype=object,
                           engine="openpyxl", keep_default_na=False, na_values=[""])
    frames = {k: v.astype(object).where(v.notna(), None) for k, v in frames.items()}

    wb = Workbook(name, sheet_names, sizes, [], None, None)
    cover = next((n for n in frames if "cover" in normalise(n)), None)
    if cover:
        wb.cover_notes = _cover_notes(frames[cover])

    link_frames = {k: v for k, v in frames.items() if k != cover}
    for role, aliases in LINK_SHEETS.items():
        sheet = _find_sheet(sheet_names, aliases, link_frames)
        label = "Link_Loads (passenger demand)" if role == "loads" else "Link_Frequencies (trains per 15 min)"
        if sheet is None:
            wb.errors.append(f"Worksheet {label} was not found. Sheets present: {sheet_names}.")
            continue
        if link_frames[sheet].dropna(how="all").empty:
            wb.errors.append(f"Worksheet '{sheet}' is empty.")
            continue
        parsed = parse_link_sheet(link_frames[sheet], sheet)
        if parsed is None:
            wb.errors.append(f"No header row with link fields (Line, From Station, To Station...) was found in '{sheet}'.")
            continue
        for issue in parsed.issues:
            (wb.errors if issue.startswith(("Required", "No 15")) else wb.warnings).append(f"{sheet}: {issue}")
        setattr(wb, role, parsed)
    return wb
