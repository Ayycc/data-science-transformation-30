"""Normalisation into OMOP-like tables: units, vocabularies, keys."""

from __future__ import annotations

from conftest import bundle, stay
from readmit.normalise import COLUMNS, TABLES, normalise
from readmit.schemas import validate_bundle
from readmit.vocab import CONDITIONS


def test_units_and_vocabularies_harmonised() -> None:
    a = bundle(stay("e1", "2023-03-01", "2023-03-05"), bid="a")
    b = bundle(stay("e1", "2023-03-01", "2023-03-05"), bid="b")
    b["meta"]["source"] = "ehr_c"
    b["entry"][3]["resource"]["valueQuantity"] = {"value": 121.0, "unit": "g/L"}
    b["entry"][2]["resource"]["code"]["coding"][0] = {"system": "http://snomed.info/sct", "code": "233604007"}
    t = normalise([validate_bundle(a), validate_bundle(b)])
    m = t["measurement"]
    assert m["value_as_number"].tolist() == [12.1, 12.1]
    assert set(m["unit_source_value"]) == {"g/dL"}
    c = t["condition_occurrence"]
    assert c["condition_concept_id"].nunique() == 1
    assert c["condition_concept_id"].iloc[0] == CONDITIONS["pneumonia"][0]
    assert set(c["condition_source_value"]) == {"ICD10CM:J18.9", "SNOMED:233604007"}
    assert t["person"]["person_id"].tolist() == [1, 2]
    v = t["visit_occurrence"]
    assert v["visit_concept_id"].tolist() == [9201, 9201]
    assert v["visit_occurrence_id"].is_unique
    assert m["visit_occurrence_id"].tolist() == v["visit_occurrence_id"].tolist()


def test_all_tables_have_omop_columns(sample_build: dict) -> None:
    for name in TABLES:
        assert list(sample_build["frames"][name].columns) == COLUMNS[name]


def test_sample_row_counts_match_resources(sample_build: dict) -> None:
    rep = sample_build["report"]
    f = sample_build["frames"]
    assert len(f["person"]) == rep.n_valid
    assert f["measurement"]["value_as_number"].notna().all()
    # every lab is in its canonical unit
    assert f["measurement"].groupby("measurement_key")["unit_source_value"].nunique().max() == 1
