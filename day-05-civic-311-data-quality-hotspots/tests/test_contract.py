"""Every rule fires on a hand-made row with exactly that defect, and only there; GE agrees."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from civic311 import contract, geo

GOOD = {
    "sr_number": "SR25-00000001", "sr_type_raw": "Tree Trim", "owner_department": "Streets & Sanitation",
    "status": "Completed", "origin": "Phone Call", "created_raw": "2025-03-01 10:00:00",
    "closed_raw": "2025-03-02 10:00:00", "ward_raw": "12", "ward": 12, "latitude": 41.88, "longitude": -87.70,
    "category": "Tree Trim",
}  # fmt: skip

CASES = {
    "SR_NUMBER_MISSING": {"sr_number": None},
    "SR_NUMBER_MALFORMED": {"sr_number": "sr25 0000123"},
    "CREATED_MISSING_OR_UNPARSEABLE": {"created_raw": "N/A"},
    "CREATED_IN_FUTURE": {"created_raw": "2026-05-01 00:00:00", "status": "Open", "closed_raw": None},
    "CREATED_BEFORE_WINDOW": {"created_raw": "2024-12-31 23:00:00"},
    "CLOSED_UNPARSEABLE": {"closed_raw": "yesterday"},
    "CLOSED_PLACEHOLDER_DATE": {"closed_raw": "1900-01-01 00:00:00"},
    "CLOSED_IN_FUTURE": {"closed_raw": "2026-02-01 00:00:00"},
    "CLOSED_BEFORE_CREATED": {"closed_raw": "2025-02-28 10:00:00"},
    "STATUS_INVALID": {"status": "DONE"},
    "COMPLETED_WITHOUT_CLOSE_DATE": {"closed_raw": None},
    "OPEN_WITH_CLOSE_DATE": {"status": "Open"},
    "COORDS_MISSING": {"latitude": np.nan},
    "COORDS_NULL_ISLAND": {"latitude": 0.0, "longitude": 0.0},
    "COORDS_OUT_OF_BBOX": {"latitude": 40.71, "longitude": -74.0},
    "COORDS_SWAPPED_LATLON": {"latitude": -87.70, "longitude": 41.88},
    "COORDS_OUTSIDE_CITY": {"latitude": 41.70, "longitude": -87.90},  # in the bbox, south-west suburbs
    "WARD_INVALID": {"ward_raw": "99", "ward": 99},
    "CATEGORY_UNMAPPED": {"sr_type_raw": "TEST ENTRY", "category": None},
}
# a few defects legitimately trip a second rule as well
ALSO = {
    "CLOSED_PLACEHOLDER_DATE": {"CLOSED_BEFORE_CREATED"},
    "COORDS_SWAPPED_LATLON": {"COORDS_OUT_OF_BBOX"},
    "CLOSED_UNPARSEABLE": set(),
}


def frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.insert(0, "source_line", np.arange(1, len(df) + 1, dtype="int64"))
    df["created_ts"] = pd.to_datetime(df.created_raw, format="%Y-%m-%d %H:%M:%S", errors="coerce").astype(
        "datetime64[ns]"
    )
    df["closed_ts"] = pd.to_datetime(df.closed_raw, format="%Y-%m-%d %H:%M:%S", errors="coerce").astype(
        "datetime64[ns]"
    )
    df["ward"] = df.ward.astype("Int64")
    df["latitude"] = df.latitude.astype(float)
    df["longitude"] = df.longitude.astype(float)
    lon, lat = df.longitude.to_numpy(), df.latitude.to_numpy()
    df["in_bbox"] = geo.in_bbox(lon, lat)
    df["in_city_polygon"] = geo.in_city(lon, lat)
    df["swapped_in_bbox"] = geo.in_bbox(lat, lon)
    return df


@pytest.fixture(scope="module")
def cases_df():
    rows = [GOOD] + [{**GOOD, **chg, "sr_number": chg.get("sr_number", f"SR25-{i + 2:08d}")}
                     for i, chg in enumerate(CASES.values())]  # fmt: skip
    rows.append({**GOOD})  # exact duplicate of row 1 -> DUPLICATE_RECORD on the copy only
    return frame(rows)


def test_rule_registry_is_consistent():
    codes = [r.code for r in contract.RULES]
    assert len(codes) == len(set(codes)) == 20
    assert set(CASES) | {"DUPLICATE_RECORD"} == set(codes)


def test_each_defect_hits_its_rule(cases_df):
    rej = contract.run_pandera(cases_df)
    got = rej.groupby("source_line").reason_code.agg(set).to_dict()
    assert 1 not in got, "the good row must pass"
    for i, code in enumerate(CASES, start=2):
        assert got.get(i, set()) == {code} | ALSO.get(code, set()), code
    assert got[len(cases_df)] == {"DUPLICATE_RECORD"}
    assert set(rej.rule_id) <= set(contract.BY_ID)
    assert (rej.contract_version == contract.CONTRACT_VERSION).all()


def test_independent_masks_match_pandera(cases_df):
    rej = contract.run_pandera(cases_df)
    pandera_pairs = set(zip(rej.source_line, rej.reason_code, strict=True))
    masks = contract.rule_masks(cases_df)
    mask_pairs = {(int(s), c) for c, m in masks.items() for s in cases_df.source_line[m]}
    assert pandera_pairs == mask_pairs


def test_great_expectations_suite(cases_df):
    good = frame([GOOD, {**GOOD, "sr_number": "SR25-00000002"}])
    ok = contract.run_ge(good, "good")
    assert ok["success"], [r for r in ok["results"] if not r["success"]]
    bad = contract.run_ge(cases_df, "cases")
    assert not bad["success"]
    failed = {r["code"] for r in bad["results"] if not r["success"]}
    assert failed == {r.code for r in contract.RULES}
    assert not any(r["exception"] for r in bad["results"])
