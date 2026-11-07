"""Shared constants: reporting calendar, lines of business, expected loss ratios.

Dates are stored as SAS date integers (days since 1960-01-01), which is how the
legacy extracts deliver them. All money is stored in integer cents.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

SAS_EPOCH = dt.date(1960, 1, 1)
SEED = 20261107


def to_sas(d: dt.date) -> int:
    """Convert a calendar date to a SAS date integer."""
    return (d - SAS_EPOCH).days


def from_sas(n: int) -> dt.date:
    """Convert a SAS date integer to a calendar date."""
    return SAS_EPOCH + dt.timedelta(days=int(n))


@dataclass(frozen=True)
class Quarter:
    idx: int  # 1..8
    label: str  # e.g. 2024Q1
    start: int  # SAS date, inclusive
    end: int  # SAS date, inclusive (quarter end = close date)


def _quarters() -> list[Quarter]:
    out = []
    i = 1
    for year in (2024, 2025):
        for q in range(4):
            start = dt.date(year, 3 * q + 1, 1)
            nxt = dt.date(year + (q == 3), (3 * q + 3) % 12 + 1, 1)
            out.append(Quarter(i, f"{year}Q{q + 1}", to_sas(start), to_sas(nxt) - 1))
            i += 1
    return out


QUARTERS: list[Quarter] = _quarters()
QUARTER_BY_LABEL = {q.label: q for q in QUARTERS}

# Canonical lines of business. Raw extracts also contain mis-cased / padded codes
# ("Motor", "home ") and a small run-off book ("PET") that has no mapping.
LOBS = ["COMMERCIAL", "HOME", "LIABILITY", "MOTOR", "TRAVEL"]
REGIONS = ["EAST", "NORTH", "SOUTH", "WEST"]

# Expected loss ratio table used for the IBNR estimate, in basis points.
# The legacy Excel sheet holds the first five rows only (sorted A-Z) and looks it
# up with VLOOKUP(..., TRUE). The modern model adds an explicit OTHER row.
ELR_BP = {"COMMERCIAL": 6200, "HOME": 5800, "LIABILITY": 7400, "MOTOR": 7100, "TRAVEL": 4500}
ELR_OTHER_BP = 6500

# IBNR became a required column of the regulatory return from 2025Q1, which is
# when the Iceberg report table's schema is evolved to add it.
IBNR_FROM_QUARTER = 5
