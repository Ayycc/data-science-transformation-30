"""Pydantic v2 schemas for incoming FHIR-like bundles.

These are deliberately a strict *subset* of FHIR R4: the fields this pipeline needs are typed and
checked, unknown fields are ignored. A bundle is rejected as a whole if any resource fails, or if
the bundle breaks a cross-resource rule (one Patient, references resolve, no care after death).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .config import WINDOW_END
from .vocab import ICD10, LAB_PLAUSIBLE, LABS, LOINC, LOINC_TO_KEY, PRIORITY, RXNORM, SNOMED

REF_PATTERN = r"^(Patient|Encounter)/[A-Za-z0-9\-\.]{1,64}$"


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, frozen=True)


class Coding(_Base):
    system: str | None = None
    code: str = Field(min_length=1)
    display: str | None = None


class CodeableConcept(_Base):
    coding: list[Coding] = Field(min_length=1)
    text: str | None = None

    def first(self) -> Coding:
        return self.coding[0]


class Reference(_Base):
    reference: str = Field(pattern=REF_PATTERN)

    @property
    def kind(self) -> str:
        return self.reference.split("/", 1)[0]

    @property
    def id(self) -> str:
        return self.reference.split("/", 1)[1]


class Period(_Base):
    start: datetime
    end: datetime | None = None

    @model_validator(mode="after")
    def _end_after_start(self) -> Period:
        if self.end is not None and _naive(self.end) < _naive(self.start):
            raise ValueError(f"period.end ({self.end.isoformat()}) is before period.start ({self.start.isoformat()})")
        return self


class Quantity(_Base):
    value: float
    unit: str = Field(min_length=1)


def _naive(dt: datetime) -> datetime:
    """Wall-clock time as recorded by the source system (clinical dates are local)."""
    return dt.replace(tzinfo=None)


class Patient(_Base):
    resourceType: Literal["Patient"]
    id: str = Field(min_length=1, max_length=64)
    gender: Literal["male", "female", "other", "unknown"]
    birthDate: date
    deceasedDateTime: datetime | None = None

    @field_validator("birthDate")
    @classmethod
    def _birth_not_future(cls, v: date) -> date:
        if v > WINDOW_END:
            raise ValueError(f"birthDate {v.isoformat()} is after the data extract date {WINDOW_END.isoformat()}")
        if v.year < 1900:
            raise ValueError(f"birthDate {v.isoformat()} is before 1900")
        return v

    @model_validator(mode="after")
    def _death_after_birth(self) -> Patient:
        if self.deceasedDateTime is not None and self.deceasedDateTime.date() < self.birthDate:
            raise ValueError("deceasedDateTime is before birthDate")
        return self


class Hospitalization(_Base):
    admitSource: CodeableConcept | None = None
    dischargeDisposition: CodeableConcept | None = None


class Encounter(_Base):
    resourceType: Literal["Encounter"]
    id: str = Field(min_length=1, max_length=64)
    status: Literal["planned", "in-progress", "finished", "cancelled", "entered-in-error"]
    class_: Coding = Field(alias="class")
    priority: CodeableConcept | None = None
    subject: Reference
    period: Period
    hospitalization: Hospitalization | None = None

    @field_validator("class_")
    @classmethod
    def _known_class(cls, v: Coding) -> Coding:
        if v.code not in {"IMP", "AMB", "EMER"}:
            raise ValueError(f"encounter class '{v.code}' is not one of IMP, AMB, EMER")
        return v

    @field_validator("priority")
    @classmethod
    def _known_priority(cls, v: CodeableConcept | None) -> CodeableConcept | None:
        if v is not None and v.first().code not in PRIORITY:
            raise ValueError(f"priority '{v.first().code}' is not one of {sorted(PRIORITY)}")
        return v

    @model_validator(mode="after")
    def _inpatient_complete(self) -> Encounter:
        if self.class_.code == "IMP" and self.status == "finished":
            if self.period.end is None:
                raise ValueError(f"finished inpatient encounter {self.id} has no period.end (discharge time)")
            if self.hospitalization is None or self.hospitalization.dischargeDisposition is None:
                raise ValueError(f"finished inpatient encounter {self.id} has no dischargeDisposition")
        return self


class Extension(_Base):
    url: str
    valueString: str | None = None


class Condition(_Base):
    resourceType: Literal["Condition"]
    id: str = Field(min_length=1, max_length=64)
    subject: Reference
    encounter: Reference | None = None
    code: CodeableConcept
    onsetDateTime: datetime
    extension: list[Extension] | None = None

    @property
    def rank(self) -> str | None:
        for e in self.extension or []:
            if e.url == "diagnosis-rank":
                return e.valueString
        return None

    @field_validator("code")
    @classmethod
    def _coded(cls, v: CodeableConcept) -> CodeableConcept:
        if v.first().system not in {ICD10, SNOMED}:
            raise ValueError(f"condition code system '{v.first().system}' is not ICD-10-CM or SNOMED CT")
        return v


class MedicationRequest(_Base):
    resourceType: Literal["MedicationRequest"]
    id: str = Field(min_length=1, max_length=64)
    status: str
    intent: str
    subject: Reference
    encounter: Reference | None = None
    medicationCodeableConcept: CodeableConcept
    authoredOn: datetime

    @field_validator("medicationCodeableConcept")
    @classmethod
    def _rxnorm(cls, v: CodeableConcept) -> CodeableConcept:
        if v.first().system != RXNORM:
            raise ValueError(f"medication code system '{v.first().system}' is not RxNorm")
        return v


class Observation(_Base):
    resourceType: Literal["Observation"]
    id: str = Field(min_length=1, max_length=64)
    status: Literal["registered", "preliminary", "final", "amended", "corrected"]
    code: CodeableConcept
    subject: Reference
    encounter: Reference | None = None
    effectiveDateTime: datetime
    valueQuantity: Quantity

    @model_validator(mode="after")
    def _units_and_range(self) -> Observation:
        c = self.code.first()
        if c.system != LOINC:
            raise ValueError(f"observation code system '{c.system}' is not LOINC")
        key = LOINC_TO_KEY.get(c.code)
        if key is None:
            return self  # unknown LOINC: kept, mapped to concept 0 downstream
        units = LABS[key][4]
        if self.valueQuantity.unit not in units:
            raise ValueError(
                f"unit '{self.valueQuantity.unit}' is not allowed for LOINC {c.code} ({key}); expected one of "
                f"{sorted(units)}"
            )
        canon = self.valueQuantity.value * units[self.valueQuantity.unit]
        lo, hi = LAB_PLAUSIBLE[key]
        if not lo <= canon <= hi:
            raise ValueError(
                f"{key} value {self.valueQuantity.value} {self.valueQuantity.unit} is outside the plausible range "
                f"{lo}-{hi} {LABS[key][3]} (unit or entry error?)"
            )
        return self


Resource = Annotated[
    Patient | Encounter | Condition | MedicationRequest | Observation, Field(discriminator="resourceType")
]


class BundleEntry(_Base):
    fullUrl: str | None = None
    resource: Resource


class BundleMeta(_Base):
    source: str = Field(min_length=1)


class Bundle(_Base):
    resourceType: Literal["Bundle"]
    id: str = Field(min_length=1)
    type: Literal["collection", "transaction", "searchset", "batch"]
    meta: BundleMeta
    entry: list[BundleEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _integrity(self) -> Bundle:
        res = [e.resource for e in self.entry]
        patients = [r for r in res if isinstance(r, Patient)]
        if len(patients) != 1:
            raise ValueError(f"bundle must contain exactly one Patient, found {len(patients)}")
        pat = patients[0]
        seen: set[tuple[str, str]] = set()
        for r in res:
            k = (r.resourceType, r.id)
            if k in seen:
                raise ValueError(f"duplicate resource id {r.resourceType}/{r.id}")
            seen.add(k)
        enc_ids = {r.id for r in res if isinstance(r, Encounter)}
        death = pat.deceasedDateTime.date() if pat.deceasedDateTime else None
        for r in res:
            if isinstance(r, Patient):
                continue
            if r.subject.kind != "Patient" or r.subject.id != pat.id:
                raise ValueError(
                    f"{r.resourceType}/{r.id}.subject references {r.subject.reference}, not Patient/{pat.id}"
                )
            enc = getattr(r, "encounter", None)
            if enc is not None and (enc.kind != "Encounter" or enc.id not in enc_ids):
                raise ValueError(f"{r.resourceType}/{r.id}.encounter references {enc.reference}, not in bundle")
            if isinstance(r, Encounter) and death and r.period.start.date() > death:
                raise ValueError(
                    f"Encounter/{r.id} starts {r.period.start.date().isoformat()}, after the patient's death "
                    f"{death.isoformat()}"
                )
        return self


class BundleRejected(Exception):
    """Raised when a bundle fails validation; ``errors`` holds human-readable messages."""

    def __init__(self, bundle_id: str, errors: list[str]) -> None:
        self.bundle_id = bundle_id
        self.errors = errors
        super().__init__(f"bundle {bundle_id} rejected: " + "; ".join(errors))


def format_errors(exc: ValidationError) -> list[str]:
    """Turn a pydantic ValidationError into short 'location: message' strings."""
    out = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err["loc"])
        msg = err["msg"].removeprefix("Value error, ")
        out.append(f"{loc}: {msg}" if loc else msg)
    return out


def validate_bundle(raw: str | bytes | dict) -> Bundle:
    """Validate one bundle (JSON text or dict). Raises ``BundleRejected`` with clear errors."""
    try:
        if isinstance(raw, dict):
            return Bundle.model_validate(raw)
        return Bundle.model_validate_json(raw)
    except ValidationError as exc:
        bid = "?"
        try:
            d = raw if isinstance(raw, dict) else json.loads(raw)
            bid = str(d.get("id", "?"))
        except Exception:  # noqa: BLE001 - malformed JSON keeps the '?' id
            pass
        raise BundleRejected(bid, format_errors(exc)) from exc
