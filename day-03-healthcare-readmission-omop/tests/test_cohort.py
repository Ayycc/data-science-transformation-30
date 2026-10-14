"""Cohort SQL: exclusions and outcome definition on hand-built patients."""

from __future__ import annotations

import pandas as pd

from conftest import bundle, stay
from readmit.config import SAMPLE_DIR
from readmit.db import build_cohort, connect, load_omop
from readmit.normalise import normalise
from readmit.pipeline import build, cohort_fingerprint
from readmit.schemas import validate_bundle


def _cohort(resources: list, **kw) -> pd.DataFrame:
    con = connect()
    load_omop(con, normalise([validate_bundle(bundle(resources, **kw))]))
    build_cohort(con)
    return con.execute("SELECT * FROM index_candidates ORDER BY admit_date").df()


def test_unplanned_readmission_counts() -> None:
    c = _cohort(stay("e1", "2023-03-01", "2023-03-05") + stay("e2", "2023-03-15", "2023-03-18"))
    first = c.iloc[0]
    assert first.exclusion_reason is None or pd.isna(first.exclusion_reason)
    assert bool(first.readmit_30d) and first.days_to_unplanned == 10


def test_planned_readmission_not_an_outcome() -> None:
    c = _cohort(stay("e1", "2023-03-01", "2023-03-05") + stay("e2", "2023-03-15", "2023-03-16", priority="EL"))
    first = c.iloc[0]
    assert not bool(first.readmit_30d)
    assert first.n_planned_within_horizon == 1


def test_readmission_after_30_days_not_counted() -> None:
    c = _cohort(stay("e1", "2023-03-01", "2023-03-05") + stay("e2", "2023-04-05", "2023-04-08"))
    assert not bool(c.iloc[0].readmit_30d)
    assert bool(
        _cohort(stay("e1", "2023-03-01", "2023-03-05") + stay("e2", "2023-04-04", "2023-04-08")).iloc[0].readmit_30d
    )


def test_exclusions() -> None:
    assert (
        _cohort(stay("e1", "2023-03-01", "2023-03-05", "exp"), deceased="2023-03-05T12:00:00Z")
        .iloc[0]
        .exclusion_reason.startswith("04")
    )
    assert _cohort(stay("e1", "2023-03-01", "2023-03-05", "other-hcf")).iloc[0].exclusion_reason.startswith("05")
    assert _cohort(stay("e1", "2023-03-01", "2023-03-05", "aadvice")).iloc[0].exclusion_reason.startswith("06")
    chemo = _cohort(stay("e1", "2023-03-01", "2023-03-02", priority="EL", dx="Z51.11"))
    assert chemo.iloc[0].exclusion_reason.startswith("07")
    died = _cohort(stay("e1", "2023-03-01", "2023-03-05"), deceased="2023-03-20T00:00:00Z")
    assert died.iloc[0].exclusion_reason.startswith("08")
    assert _cohort(stay("e1", "2021-06-01", "2021-06-05")).iloc[0].exclusion_reason.startswith("01")
    assert _cohort(stay("e1", "2024-12-10", "2024-12-12")).iloc[0].exclusion_reason.startswith("02")
    assert _cohort(stay("e1", "2023-03-01", "2023-03-05"), birth="2010-01-01").iloc[0].exclusion_reason.startswith("03")


def test_death_after_readmission_keeps_index() -> None:
    c = _cohort(
        stay("e1", "2023-03-01", "2023-03-05") + stay("e2", "2023-03-10", "2023-03-12", "exp"),
        deceased="2023-03-12T15:00:00Z",
    )
    first = c.iloc[0]
    assert pd.isna(first.exclusion_reason) and bool(first.readmit_30d)


def test_cohort_reproducible_from_single_command(sample_build: dict) -> None:
    again = build(SAMPLE_DIR)
    pd.testing.assert_frame_equal(sample_build["attrition"], again["attrition"])
    assert cohort_fingerprint(sample_build["features"]) == cohort_fingerprint(again["features"])
    assert (
        sample_build["attrition"]["n"].sum()
        == (sample_build["frames"]["visit_occurrence"]["visit_concept_id"] == 9201).sum()
    )


def test_features_have_no_future_information(sample_build: dict) -> None:
    f = sample_build["features"]
    assert (f["discharge_date"] >= f["admit_date"]).all()
    assert f["lace"].between(0, 19).all()
    assert f["readmit_30d"].sum() == f["days_to_unplanned"].notna().sum()
