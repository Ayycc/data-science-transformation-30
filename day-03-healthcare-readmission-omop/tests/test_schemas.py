"""Invalid bundles must be rejected with clear, located error messages."""

from __future__ import annotations

import copy
import json

import pytest

from conftest import bundle, stay
from readmit.schemas import Bundle, BundleRejected, validate_bundle


def _good() -> dict:
    return bundle(stay("e1", "2023-03-01", "2023-03-05"))


def test_valid_bundle_passes() -> None:
    b = validate_bundle(_good())
    assert isinstance(b, Bundle)
    assert len(b.entry) == 4


def test_json_text_input_passes() -> None:
    assert validate_bundle(json.dumps(_good())).id == "b1"


def _reject(b: dict) -> list[str]:
    with pytest.raises(BundleRejected) as ei:
        validate_bundle(b)
    assert ei.value.bundle_id == b.get("id", "?")
    return ei.value.errors


def test_bad_gender() -> None:
    b = _good()
    b["entry"][0]["resource"]["gender"] = "M"
    errs = _reject(b)
    assert any("Patient.gender" in e and "'male', 'female'" in e for e in errs)


def test_future_birthdate() -> None:
    errs = _reject(bundle([], birth="2090-01-01"))
    assert any("birthDate 2090-01-01 is after the data extract date" in e for e in errs)


def test_period_end_before_start() -> None:
    b = bundle(stay("e1", "2023-03-05", "2023-03-01"))
    errs = _reject(b)
    assert any("Encounter.period" in e and "is before period.start" in e for e in errs)


def test_missing_subject() -> None:
    b = _good()
    del b["entry"][1]["resource"]["subject"]
    errs = _reject(b)
    assert any(e.endswith("Encounter.subject: Field required") for e in errs)


def test_wrong_unit_for_loinc() -> None:
    b = _good()
    b["entry"][3]["resource"]["valueQuantity"]["unit"] = "mg"
    errs = _reject(b)
    assert any("unit 'mg' is not allowed for LOINC 718-7" in e for e in errs)


def test_implausible_value_after_unit_conversion() -> None:
    b = _good()
    b["entry"][3]["resource"]["valueQuantity"] = {"value": 121.0, "unit": "g/dL"}  # g/L value mislabelled
    errs = _reject(b)
    assert any("outside the plausible range" in e for e in errs)


def test_si_units_accepted() -> None:
    b = _good()
    b["entry"][3]["resource"]["valueQuantity"] = {"value": 121.0, "unit": "g/L"}
    validate_bundle(b)


def test_dangling_encounter_reference() -> None:
    b = _good()
    b["entry"][2]["resource"]["encounter"] = {"reference": "Encounter/nope"}
    errs = _reject(b)
    assert any("references Encounter/nope, not in bundle" in e for e in errs)


def test_subject_must_be_bundle_patient() -> None:
    b = _good()
    b["entry"][2]["resource"]["subject"] = {"reference": "Patient/someone-else"}
    errs = _reject(b)
    assert any("not Patient/p1" in e for e in errs)


def test_exactly_one_patient() -> None:
    b = _good()
    p2 = copy.deepcopy(b["entry"][0])
    p2["resource"]["id"] = "p2"
    b["entry"].append(p2)
    assert any("exactly one Patient, found 2" in e for e in _reject(b))


def test_unknown_resource_type() -> None:
    b = _good()
    b["entry"].append({"resource": {"resourceType": "Procedur", "id": "x"}})
    assert any("Input tag 'Procedur'" in e for e in _reject(b))


def test_encounter_after_death() -> None:
    b = bundle(stay("e1", "2023-03-01", "2023-03-05"), deceased="2023-02-01T00:00:00Z")
    assert any("after the patient's death 2023-02-01" in e for e in _reject(b))


def test_inpatient_requires_disposition() -> None:
    b = _good()
    del b["entry"][1]["resource"]["hospitalization"]["dischargeDisposition"]
    assert any("has no dischargeDisposition" in e for e in _reject(b))


def test_unknown_encounter_class_and_priority() -> None:
    b = _good()
    b["entry"][1]["resource"]["class"]["code"] = "XYZ"
    b["entry"][1]["resource"]["priority"]["coding"][0]["code"] = "ZZ"
    errs = _reject(b)
    assert any("encounter class 'XYZ'" in e for e in errs)
    assert any("priority 'ZZ'" in e for e in errs)


def test_duplicate_resource_ids() -> None:
    b = _good()
    b["entry"].append(copy.deepcopy(b["entry"][3]))
    assert any("duplicate resource id Observation/o-e1" in e for e in _reject(b))


def test_malformed_json() -> None:
    with pytest.raises(BundleRejected) as ei:
        validate_bundle(b'{"resourceType": "Bundle", ')
    assert ei.value.errors


def test_sample_rejections_match_injected_defects(sample_build: dict) -> None:
    from readmit.config import SAMPLE_DIR

    injected = json.loads((SAMPLE_DIR / "manifest.json").read_text())["corrupted_bundles"]
    rejected = {r["bundle_id"] for r in sample_build["report"].rejected}
    assert injected, "sample should contain injected defects"
    assert rejected == set(injected)
