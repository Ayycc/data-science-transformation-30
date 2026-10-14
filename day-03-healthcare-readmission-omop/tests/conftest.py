"""Shared fixtures: hand-built FHIR-like bundles and the committed SYNTHETIC sample (data/sample)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from readmit.config import SAMPLE_DIR  # noqa: E402
from readmit.pipeline import build  # noqa: E402

LOINC = "http://loinc.org"
ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"


def stay(eid: str, start: str, end: str, disposition: str = "home", priority: str = "EM", dx: str = "J18.9") -> list:
    """An inpatient encounter with a primary diagnosis and a discharge hemoglobin."""
    enc = {
        "resourceType": "Encounter",
        "id": eid,
        "status": "finished",
        "class": {"code": "IMP"},
        "priority": {"coding": [{"code": priority}]},
        "subject": {"reference": "Patient/p1"},
        "period": {"start": f"{start}T08:00:00Z", "end": f"{end}T12:00:00Z"},
        "hospitalization": {
            "admitSource": {"coding": [{"code": "emd" if priority == "EM" else "gp"}]},
            "dischargeDisposition": {"coding": [{"code": disposition}]},
        },
    }
    cond = {
        "resourceType": "Condition",
        "id": f"c-{eid}",
        "subject": {"reference": "Patient/p1"},
        "encounter": {"reference": f"Encounter/{eid}"},
        "code": {"coding": [{"system": ICD10, "code": dx}]},
        "onsetDateTime": f"{start}T08:00:00Z",
        "extension": [{"url": "diagnosis-rank", "valueString": "primary"}],
    }
    obs = {
        "resourceType": "Observation",
        "id": f"o-{eid}",
        "status": "final",
        "code": {"coding": [{"system": LOINC, "code": "718-7"}]},
        "subject": {"reference": "Patient/p1"},
        "encounter": {"reference": f"Encounter/{eid}"},
        "effectiveDateTime": f"{end}T07:00:00Z",
        "valueQuantity": {"value": 12.1, "unit": "g/dL"},
    }
    return [enc, cond, obs]


def bundle(resources: list, bid: str = "b1", birth: str = "1950-03-01", deceased: str | None = None) -> dict:
    patient = {"resourceType": "Patient", "id": "p1", "gender": "female", "birthDate": birth}
    if deceased:
        patient["deceasedDateTime"] = deceased
    return {
        "resourceType": "Bundle",
        "id": bid,
        "type": "collection",
        "meta": {"source": "ehr_a"},
        "entry": [{"resource": r} for r in [patient, *resources]],
    }


@pytest.fixture(scope="session")
def sample_build() -> dict:
    return build(SAMPLE_DIR)
