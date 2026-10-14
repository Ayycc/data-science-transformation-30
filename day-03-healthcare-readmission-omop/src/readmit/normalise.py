"""Validate FHIR-like bundles and normalise them into OMOP-like tables.

Tables follow OMOP CDM v5.4 column names (person, death, visit_occurrence, condition_occurrence,
drug_exposure, measurement). One non-standard column is added, ``visit_occurrence.priority_source_value``,
because the CDM has no slot for admission priority and the cohort needs it to tell planned from
unplanned readmissions. Lab values are converted to one canonical unit per measurement.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from .schemas import Bundle, BundleRejected, Condition, Encounter, MedicationRequest, Observation, validate_bundle
from .vocab import (
    ADMIT_SOURCE,
    CONDITIONS,
    DISCHARGE,
    DRUGS,
    EHR_TYPE_CONCEPT,
    GENDER_CONCEPT,
    ICD10,
    ICD10_TO_KEY,
    LABS,
    LOINC_TO_KEY,
    RXNORM_TO_KEY,
    SNOMED_TO_KEY,
    VISIT_CONCEPT,
)

TABLES = ("person", "death", "visit_occurrence", "condition_occurrence", "drug_exposure", "measurement")
COLUMNS = {
    "person": [
        "person_id",
        "gender_concept_id",
        "year_of_birth",
        "month_of_birth",
        "day_of_birth",
        "birth_datetime",
        "person_source_value",
        "gender_source_value",
        "care_site_source_value",
    ],
    "death": ["person_id", "death_date", "death_type_concept_id"],
    "visit_occurrence": [
        "visit_occurrence_id",
        "person_id",
        "visit_concept_id",
        "visit_start_date",
        "visit_start_datetime",
        "visit_end_date",
        "visit_end_datetime",
        "visit_type_concept_id",
        "visit_source_value",
        "admitted_from_concept_id",
        "admitted_from_source_value",
        "discharged_to_concept_id",
        "discharged_to_source_value",
        "priority_source_value",
    ],
    "condition_occurrence": [
        "condition_occurrence_id",
        "person_id",
        "condition_concept_id",
        "condition_start_date",
        "condition_type_concept_id",
        "visit_occurrence_id",
        "condition_source_value",
        "condition_status_source_value",
        "condition_key",
    ],
    "drug_exposure": [
        "drug_exposure_id",
        "person_id",
        "drug_concept_id",
        "drug_exposure_start_date",
        "drug_type_concept_id",
        "visit_occurrence_id",
        "drug_source_value",
    ],
    "measurement": [
        "measurement_id",
        "person_id",
        "measurement_concept_id",
        "measurement_date",
        "measurement_datetime",
        "measurement_type_concept_id",
        "value_as_number",
        "unit_source_value",
        "visit_occurrence_id",
        "measurement_source_value",
        "value_source_value",
        "measurement_key",
    ],
}


@dataclass
class ValidationReport:
    n_bundles: int = 0
    n_valid: int = 0
    rejected: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "bundles_read": self.n_bundles,
            "bundles_valid": self.n_valid,
            "bundles_rejected": len(self.rejected),
            "rejections_by_rule": dict(sorted(Counter(rule_of(r["first_error"]) for r in self.rejected).items())),
        }


def rule_of(message: str) -> str:
    """Collapse an error message to its rule (drop entry indexes, ids and dates)."""
    msg = re.sub(r"entry\.\d+\.resource\.?", "", message)
    loc, sep, text = msg.partition(": ")
    if sep and " " not in loc:
        return loc or "resource"
    return re.sub(r"[\w.-]*\d[\w.:-]*", "#", msg)


def validate_all(lines: Iterable[tuple[str, bytes | str | dict]]) -> tuple[list[Bundle], ValidationReport]:
    """Validate every (source, raw bundle) pair; return valid bundles and a rejection report."""
    rep = ValidationReport()
    valid: list[Bundle] = []
    for source, raw in lines:
        rep.n_bundles += 1
        try:
            valid.append(validate_bundle(raw))
            rep.n_valid += 1
        except BundleRejected as exc:
            rep.rejected.append(
                {
                    "source_file": source,
                    "bundle_id": exc.bundle_id,
                    "n_errors": len(exc.errors),
                    "first_error": exc.errors[0],
                    "errors": " | ".join(exc.errors),
                }
            )
    return valid, rep


def _condition_key(c: Condition) -> tuple[str | None, str]:
    cd = c.code.first()
    if cd.system == ICD10:
        return ICD10_TO_KEY.get(cd.code), f"ICD10CM:{cd.code}"
    return SNOMED_TO_KEY.get(cd.code), f"SNOMED:{cd.code}"


def normalise(bundles: list[Bundle]) -> dict[str, pd.DataFrame]:
    """Map validated bundles to OMOP-like DataFrames with integer surrogate keys."""
    person, death, visit, cond, drug, meas = [], [], [], [], [], []
    visit_id = cond_id = drug_id = meas_id = 0
    for person_id, b in enumerate(bundles, start=1):
        src = b.meta.source
        res = [e.resource for e in b.entry]
        pat = next(r for r in res if r.resourceType == "Patient")
        person.append(
            {
                "person_id": person_id,
                "gender_concept_id": GENDER_CONCEPT.get(pat.gender, 0),
                "year_of_birth": pat.birthDate.year,
                "month_of_birth": pat.birthDate.month,
                "day_of_birth": pat.birthDate.day,
                "birth_datetime": pd.Timestamp(pat.birthDate),
                "person_source_value": f"{src}:{pat.id}",
                "gender_source_value": pat.gender,
                "care_site_source_value": src,
            }
        )
        if pat.deceasedDateTime is not None:
            death.append(
                {
                    "person_id": person_id,
                    "death_date": pat.deceasedDateTime.date(),
                    "death_type_concept_id": EHR_TYPE_CONCEPT,
                }
            )
        enc_map: dict[str, int] = {}
        for r in res:
            if isinstance(r, Encounter):
                visit_id += 1
                enc_map[r.id] = visit_id
                h = r.hospitalization
                adm = h.admitSource.first().code if h and h.admitSource else None
                dis = h.dischargeDisposition.first().code if h and h.dischargeDisposition else None
                start = r.period.start.replace(tzinfo=None)
                end = (r.period.end or r.period.start).replace(tzinfo=None)
                visit.append(
                    {
                        "visit_occurrence_id": visit_id,
                        "person_id": person_id,
                        "visit_concept_id": VISIT_CONCEPT[r.class_.code],
                        "visit_start_date": start.date(),
                        "visit_start_datetime": start,
                        "visit_end_date": end.date(),
                        "visit_end_datetime": end,
                        "visit_type_concept_id": EHR_TYPE_CONCEPT,
                        "visit_source_value": f"{src}:{r.id}",
                        "admitted_from_concept_id": ADMIT_SOURCE.get(adm, 0) if adm else 0,
                        "admitted_from_source_value": adm,
                        "discharged_to_concept_id": DISCHARGE.get(dis, 0) if dis else 0,
                        "discharged_to_source_value": dis,
                        "priority_source_value": r.priority.first().code if r.priority else None,
                    }
                )
        for r in res:
            v_id = enc_map.get(r.encounter.id) if getattr(r, "encounter", None) is not None else None
            if isinstance(r, Condition):
                cond_id += 1
                key, source_value = _condition_key(r)
                cond.append(
                    {
                        "condition_occurrence_id": cond_id,
                        "person_id": person_id,
                        "condition_concept_id": CONDITIONS[key][0] if key else 0,
                        "condition_start_date": r.onsetDateTime.replace(tzinfo=None).date(),
                        "condition_type_concept_id": EHR_TYPE_CONCEPT,
                        "visit_occurrence_id": v_id,
                        "condition_source_value": source_value,
                        "condition_status_source_value": r.rank,
                        "condition_key": key,  # convenience label for SQL readability
                    }
                )
            elif isinstance(r, MedicationRequest):
                drug_id += 1
                rx = r.medicationCodeableConcept.first().code
                key = RXNORM_TO_KEY.get(rx)
                drug.append(
                    {
                        "drug_exposure_id": drug_id,
                        "person_id": person_id,
                        "drug_concept_id": DRUGS[key][0] if key else 0,
                        "drug_exposure_start_date": r.authoredOn.replace(tzinfo=None).date(),
                        "drug_type_concept_id": EHR_TYPE_CONCEPT,
                        "visit_occurrence_id": v_id,
                        "drug_source_value": f"RxNorm:{rx}",
                    }
                )
            elif isinstance(r, Observation):
                meas_id += 1
                loinc = r.code.first().code
                key = LOINC_TO_KEY.get(loinc)
                val, unit = r.valueQuantity.value, r.valueQuantity.unit
                if key:
                    val = val * LABS[key][4][unit]
                    unit_out = LABS[key][3]
                else:
                    unit_out = unit
                eff = r.effectiveDateTime.replace(tzinfo=None)
                meas.append(
                    {
                        "measurement_id": meas_id,
                        "person_id": person_id,
                        "measurement_concept_id": LABS[key][0] if key else 0,
                        "measurement_date": eff.date(),
                        "measurement_datetime": eff,
                        "measurement_type_concept_id": EHR_TYPE_CONCEPT,
                        "value_as_number": round(val, 4),
                        "unit_source_value": unit_out,
                        "visit_occurrence_id": v_id,
                        "measurement_source_value": f"LOINC:{loinc}",
                        "value_source_value": f"{r.valueQuantity.value} {unit}",
                        "measurement_key": key,
                    }
                )
    rows = {
        "person": person,
        "death": death,
        "visit_occurrence": visit,
        "condition_occurrence": cond,
        "drug_exposure": drug,
        "measurement": meas,
    }
    frames = {name: pd.DataFrame(rows[name], columns=COLUMNS[name]) for name in TABLES}
    for name, cols in {
        "death": ["death_date"],
        "visit_occurrence": ["visit_start_date", "visit_end_date"],
        "condition_occurrence": ["condition_start_date"],
        "drug_exposure": ["drug_exposure_start_date"],
        "measurement": ["measurement_date"],
    }.items():
        for c in cols:
            frames[name][c] = pd.to_datetime(frames[name][c])
    for name in ("condition_occurrence", "drug_exposure", "measurement"):
        frames[name]["visit_occurrence_id"] = frames[name]["visit_occurrence_id"].astype("Int64")
    return frames
