"""SYNTHETIC FHIR-like bundle generator.

Each patient becomes one ``Bundle`` (type ``collection``) holding Patient, Encounter, Condition,
MedicationRequest and Observation resources. Three simulated EHRs differ in the ways real feeds do:

* ``ehr_a``: ICD-10-CM conditions, conventional lab units, UTI ``Z`` timestamps.
* ``ehr_b``: ICD-10-CM conditions, SI lab units (g/L, umol/L, mmol/L, 10*9/L), ``-05:00`` offsets.
* ``ehr_c``: SNOMED CT conditions, sodium in mEq/L, timestamps without seconds or zone.

Clinical correlations are simulated, not learned from real data: age and a latent frailty score
drive chronic disease; disease and frailty drive admission rates, length of stay, labs and
disposition; a logistic hazard of the discharge state drives unplanned 30-day readmission.
Planned (elective) admissions, transfers, AMA discharges and deaths are generated so the cohort
exclusions have something to exclude. A small share of bundles is corrupted on purpose to
exercise validation; the corruption types are recorded in the manifest.
"""

from __future__ import annotations

import copy
import gzip
import json
import math
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from .config import EHR_SOURCES, WINDOW_END, WINDOW_START, Scale
from .vocab import CHRONIC, CHRONIC_DRUGS, CONDITIONS, DRUGS, ICD10, LABS, LOINC, RXNORM, SNOMED

EHR_WEIGHTS = (0.45, 0.35, 0.20)
CORRUPTIONS = (
    "missing_subject",
    "end_before_start",
    "bad_gender",
    "unknown_unit",
    "dangling_encounter_ref",
    "future_birthdate",
    "implausible_value",
    "two_patients",
    "unknown_resource_type",
    "encounter_after_death",
)

# Chronic prevalence model: logit p = a + b * (age - 65) / 10 + c * frailty
PREV = {
    "chf": (-1.9, 0.45, 0.45),
    "copd": (-1.8, 0.20, 0.40),
    "diabetes": (-1.0, 0.10, 0.30),
    "ckd": (-2.6, 0.40, 0.40),
    "cancer": (-2.5, 0.25, 0.30),
    "dementia": (-3.2, 0.90, 0.30),
    "liver": (-3.6, -0.20, 0.40),
    "prior_mi": (-2.2, 0.35, 0.30),
    "hypertension": (0.2, 0.35, 0.10),
    "afib": (-2.1, 0.55, 0.20),
    "depression": (-1.7, -0.10, 0.30),
}
LOS_BASE = {
    "sepsis": 6.0,
    "pneumonia": 4.5,
    "chf_exac": 4.5,
    "copd_exac": 4.0,
    "uti": 3.5,
    "ami": 4.0,
    "gi_bleed": 3.5,
    "hip_fracture": 6.0,
    "chemo": 1.0,
    "elective_joint": 2.5,
    "cad_staged": 1.5,
}
INPATIENT_DRUGS = {
    "pneumonia": ["ceftriaxone"],
    "sepsis": ["ceftriaxone"],
    "uti": ["ceftriaxone"],
    "copd_exac": ["prednisone", "albuterol"],
    "chf_exac": ["furosemide"],
    "gi_bleed": ["pantoprazole"],
    "hip_fracture": ["oxycodone"],
    "ami": ["aspirin", "atorvastatin"],
    "chemo": ["cisplatin"],
    "elective_joint": ["oxycodone"],
    "cad_staged": ["aspirin"],
}


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class _PatientSim:
    """Simulates one patient's encounters over the window and renders them as a bundle."""

    def __init__(self, idx: int, rng: np.random.Generator, ehr: str) -> None:
        self.rng = rng
        self.ehr = ehr
        self.pid = f"{ehr[-1]}{idx:06d}"
        self.resources: list[dict] = []
        self.n_enc = self.n_cond = self.n_med = self.n_obs = 0
        self.drug_events: list[tuple[date, str]] = []
        self.admissions: list[tuple[date, date]] = []
        self.truth: list[dict] = []  # per-admission generator truth (for diagnostics only)

    # ---------- rendering helpers ----------
    def _ts(self, d: date, hour: int, minute: int = 0) -> str:
        dt = datetime(d.year, d.month, d.day, hour, minute)
        if self.ehr == "ehr_a":
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if self.ehr == "ehr_b":
            return dt.strftime("%Y-%m-%dT%H:%M:%S-05:00")
        return dt.strftime("%Y-%m-%dT%H:%M")

    def _hour(self) -> int:
        return int(self.rng.integers(6, 22))

    def _ref(self, kind: str, rid: str) -> dict:
        return {"reference": f"{kind}/{rid}"}

    def _add(self, res: dict) -> None:
        self.resources.append({"fullUrl": f"urn:{self.ehr}:{res['resourceType']}/{res['id']}", "resource": res})

    def encounter(
        self,
        cls: str,
        start: date,
        end: date,
        priority: str | None = None,
        admit_source: str | None = None,
        disposition: str | None = None,
    ) -> str:
        self.n_enc += 1
        eid = f"enc-{self.n_enc}"
        h0 = self._hour()
        h1 = min(23, h0 + 2) if start == end else self._hour()
        res: dict = {
            "resourceType": "Encounter",
            "id": eid,
            "status": "finished",
            "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode", "code": cls},
            "subject": self._ref("Patient", self.pid),
            "period": {"start": self._ts(start, h0), "end": self._ts(end, h1)},
        }
        if priority:
            res["priority"] = {
                "coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ActPriority", "code": priority}]
            }
        if admit_source or disposition:
            hosp: dict = {}
            if admit_source:
                hosp["admitSource"] = {
                    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/admit-source", "code": admit_source}]
                }
            if disposition:
                hosp["dischargeDisposition"] = {
                    "coding": [
                        {"system": "http://terminology.hl7.org/CodeSystem/discharge-disposition", "code": disposition}
                    ]
                }
            res["hospitalization"] = hosp
        self._add(res)
        return eid

    def condition(self, key: str, onset: date, enc: str | None, primary: bool = False) -> None:
        self.n_cond += 1
        _, icd, snomed, display, _, chronic = CONDITIONS[key]
        coding = {"system": SNOMED, "code": snomed} if self.ehr == "ehr_c" else {"system": ICD10, "code": icd}
        coding["display"] = display
        res: dict = {
            "resourceType": "Condition",
            "id": f"cond-{self.n_cond}",
            "subject": self._ref("Patient", self.pid),
            "code": {"coding": [coding]},
            "category": [{"coding": [{"code": "encounter-diagnosis" if not chronic else "problem-list-item"}]}],
            "onsetDateTime": self._ts(onset, self._hour()),
        }
        if enc:
            res["encounter"] = self._ref("Encounter", enc)
            res["extension"] = [{"url": "diagnosis-rank", "valueString": "primary" if primary else "secondary"}]
        self._add(res)

    def med(self, key: str, d: date, enc: str | None) -> None:
        self.n_med += 1
        _, rx, display = DRUGS[key]
        res: dict = {
            "resourceType": "MedicationRequest",
            "id": f"med-{self.n_med}",
            "status": "active",
            "intent": "order",
            "subject": self._ref("Patient", self.pid),
            "medicationCodeableConcept": {"coding": [{"system": RXNORM, "code": rx, "display": display}]},
            "authoredOn": self._ts(d, self._hour()),
        }
        if enc:
            res["encounter"] = self._ref("Encounter", enc)
        self._add(res)
        self.drug_events.append((d, key))

    def obs(self, key: str, value: float, d: date, enc: str | None) -> None:
        self.n_obs += 1
        _, loinc, display, unit, _ = LABS[key]
        if self.ehr == "ehr_b":
            si = {"hemoglobin": ("g/L", 10.0), "creatinine": ("umol/L", 88.42), "glucose": ("mmol/L", 1 / 18.016)}
            if key in si:
                unit, f = si[key]
                value = value * f
            elif key == "wbc":
                unit = "10*9/L"
        elif self.ehr == "ehr_c" and key == "sodium":
            unit = "mEq/L"
        res = {
            "resourceType": "Observation",
            "id": f"obs-{self.n_obs}",
            "status": "final",
            "code": {"coding": [{"system": LOINC, "code": loinc, "display": display}]},
            "subject": self._ref("Patient", self.pid),
            "effectiveDateTime": self._ts(d, self._hour()),
            "valueQuantity": {"value": round(float(value), 2), "unit": unit},
        }
        if enc:
            res["encounter"] = self._ref("Encounter", enc)
        self._add(res)


def _days(a: date, b: date) -> int:
    return (b - a).days


def simulate_patient(idx: int, seed: int) -> tuple[dict, list[dict]]:
    """Return (bundle, admission truth rows) for patient ``idx``. Deterministic in (idx, seed)."""
    rng = np.random.default_rng([seed, idx])
    ehr = EHR_SOURCES[int(rng.choice(3, p=EHR_WEIGHTS))]
    sim = _PatientSim(idx, rng, ehr)
    span = _days(WINDOW_START, WINDOW_END)

    # ---- demographics ----
    r = rng.random()
    if r < 0.01:
        age0 = float(rng.uniform(2, 17))
    elif r < 0.16:
        age0 = float(rng.uniform(18, 45))
    elif r < 0.46:
        age0 = float(rng.uniform(45, 65))
    else:
        age0 = float(rng.triangular(65, 72, 97))
    g = rng.random()
    gender = "female" if g < 0.52 else ("male" if g < 0.997 else "unknown")
    birth = WINDOW_START - timedelta(days=int(age0 * 365.25))
    frailty = float(rng.normal())

    def age_at(d: date) -> float:
        return _days(birth, d) / 365.25

    # ---- chronic conditions with onset dates ----
    onset: dict[str, date] = {}
    for k in CHRONIC:
        a, b, c = PREV[k]
        logit = a + b * (age0 - 65) / 10 + c * frailty
        if k == "ckd" and "diabetes" in onset:
            logit += 0.7
        if k == "chf" and ("prior_mi" in onset or "afib" in onset):
            logit += 0.6
        if age0 < 18:
            logit -= 4
        if rng.random() < _sig(logit):
            onset[k] = (
                WINDOW_START - timedelta(days=int(rng.integers(30, 3650)))
                if rng.random() < 0.7
                else WINDOW_START + timedelta(days=int(rng.integers(0, span)))
            )
    maint: dict[str, list[str]] = {}
    for k in onset:
        drugs = CHRONIC_DRUGS[k]
        chosen = [drugs[0]] + [d for d in drugs[1:] if rng.random() < 0.6]
        maint[k] = chosen

    # ---- death from background mortality ----
    hazard = 0.004 * math.exp(0.075 * (age0 - 65) + 0.35 * frailty + 0.15 * len(onset))
    t_death = rng.exponential(365.25 / max(hazard, 1e-6))
    death: date | None = WINDOW_START + timedelta(days=int(t_death)) if t_death < span else None

    def active(d: date) -> set[str]:
        return {k for k, o in onset.items() if o <= d}

    # ---- ambulatory: onset visits, annual renewals, extra visits ----
    amb_dates: list[tuple[date, str]] = []
    for k, o in onset.items():
        if o >= WINDOW_START:
            amb_dates.append((o, "onset:" + k))
    for y in range(WINDOW_START.year, WINDOW_END.year + 1):
        amb_dates.append((date(y, 1, 1) + timedelta(days=int(rng.integers(0, 364))), "renew"))
    for _ in range(int(rng.poisson(1.0 * span / 365.25))):
        amb_dates.append((WINDOW_START + timedelta(days=int(rng.integers(0, span))), "visit"))

    # ---- ED treat-and-release visits ----
    ed_rate = 0.22 * math.exp(0.45 * frailty + 0.12 * len(onset))
    ed_dates = sorted(
        WINDOW_START + timedelta(days=int(rng.integers(0, span))) for _ in range(int(rng.poisson(ed_rate * 4)))
    )

    # ---- planned admissions schedule ----
    electives: list[tuple[date, str]] = []
    if "cancer" in onset and rng.random() < 0.5:
        d = max(onset["cancer"], WINDOW_START) + timedelta(days=int(rng.integers(5, 400)))
        for _ in range(int(rng.integers(4, 7))):
            electives.append((d, "chemo"))
            d += timedelta(days=int(rng.integers(19, 24)))
    if 55 <= age0 <= 85 and rng.random() < 0.04:
        electives.append((WINDOW_START + timedelta(days=int(rng.integers(0, span))), "elective_joint"))
    electives.sort()

    # ---- inpatient process ----
    end_of_life = death if death else WINDOW_END + timedelta(days=1)

    def admit_rate(d: date) -> float:
        act = active(d)
        major = sum(k in act for k in ("chf", "copd", "ckd", "cancer", "dementia", "liver", "diabetes"))
        lam = math.exp(-2.25 + 0.30 * major + 0.15 * (age_at(d) - 65) / 10 + 0.55 * frailty)
        return lam / 365.25

    def gap(d: date) -> date:
        return d + timedelta(days=1 + int(rng.exponential(1 / admit_rate(d))))

    t = WINDOW_START
    next_unplanned = gap(t - timedelta(days=1))
    while True:
        while electives and electives[0][0] <= t:
            electives.pop(0)
        ne = electives[0][0] if electives else date.max
        start = min(next_unplanned, ne)
        if start >= end_of_life or start > WINDOW_END:
            break
        planned = ne < next_unplanned
        act = active(start)
        age = age_at(start)
        if planned:
            dx = electives.pop(0)[1]
            priority, source = "EL", ("outp" if dx == "chemo" else "gp")
        else:
            w = {
                "chf_exac": 3.0 * ("chf" in act),
                "copd_exac": 3.0 * ("copd" in act),
                "pneumonia": 2.0 + 1.0 * ("copd" in act),
                "sepsis": 1.0 + 0.5 * ("ckd" in act) + 0.5 * ("diabetes" in act) + 0.5 * ("cancer" in act),
                "uti": 1.5 + 1.0 * ("dementia" in act),
                "ami": 0.6 + 1.0 * ("prior_mi" in act),
                "gi_bleed": 0.6 + 2.0 * ("liver" in act),
                "hip_fracture": 0.4 + 1.0 * (age > 80) + 0.8 * ("dementia" in act),
            }
            keys = list(w)
            p = np.array([w[k] for k in keys], dtype=float)
            dx = keys[int(rng.choice(len(keys), p=p / p.sum()))]
            via_ed = rng.random() < 0.8
            priority, source = ("EM", "emd") if via_ed else ("UR", "gp")
        los = LOS_BASE[dx] * float(rng.lognormal(0, 0.45)) * math.exp(0.10 * frailty + 0.06 * (age - 65) / 10)
        los_days = 0 if (not planned and rng.random() < 0.03) else max(1, int(round(los)))
        if planned:
            los_days = max(1, int(round(LOS_BASE[dx] * float(rng.lognormal(0, 0.2)))))
        disch = start + timedelta(days=los_days)

        # labs: baseline + acute effect at admission, partial recovery by discharge
        female = gender == "female"
        hgb_b = (13.0 if female else 14.5) - 1.2 * ("ckd" in act) - 1.0 * ("cancer" in act) - 0.8 * ("liver" in act)
        hgb_b += float(rng.normal(0, 1.1))
        hgb_a = hgb_b - 3.0 * (dx == "gi_bleed") - 0.8 * (dx == "sepsis") - 0.6 * (dx == "chemo")
        na_b = 139 - 2.5 * ("chf" in act) - 2.0 * ("liver" in act) + float(rng.normal(0, 2.3))
        na_a = na_b - 2.5 * (dx in ("pneumonia", "sepsis")) - 1.5 * (dx == "chf_exac")
        cr_b = 0.85 * (1.2 if not female else 1.0) * (2.8 if "ckd" in act else 1.0) * float(rng.lognormal(0, 0.2))
        cr_a = cr_b * (1.6 if dx == "sepsis" else 1.3 if dx == "chf_exac" else 1.0) * float(rng.lognormal(0, 0.1))
        glu_b = 105 + 55 * ("diabetes" in act) + float(rng.normal(0, 18 + 25 * ("diabetes" in act)))
        glu_a = glu_b + 25 * (dx == "sepsis")
        wbc_b = 7.5 * float(rng.lognormal(0, 0.2))
        wbc_a = wbc_b * (1.8 if dx in ("pneumonia", "sepsis", "uti") else 1.0)
        rec = float(rng.uniform(0.3, 0.8))  # share of the acute deviation still present at discharge
        dis_vals = {
            "hemoglobin": hgb_b + rec * (hgb_a - hgb_b) + float(rng.normal(0, 0.4)),
            "sodium": na_b + rec * (na_a - na_b) + float(rng.normal(0, 1.0)),
            "creatinine": cr_b + rec * (cr_a - cr_b),
            "glucose": glu_b + rec * (glu_a - glu_b) + float(rng.normal(0, 10)),
            "wbc": wbc_b + rec * (wbc_a - wbc_b),
        }
        adm_vals = {"hemoglobin": hgb_a, "sodium": na_a, "creatinine": cr_a, "glucose": glu_a, "wbc": wbc_a}
        dis_vals = {
            k: float(np.clip(v, lo, hi))
            for (k, v), (lo, hi) in zip(
                dis_vals.items(), [(4, 19), (115, 155), (0.3, 12), (40, 600), (1, 40)], strict=True
            )
        }
        adm_vals = {
            k: float(np.clip(v, lo, hi))
            for (k, v), (lo, hi) in zip(
                adm_vals.items(), [(4, 19), (115, 155), (0.3, 12), (40, 600), (1, 40)], strict=True
            )
        }

        # outcome of the stay
        p_die = _sig(-4.6 + 0.5 * (age - 65) / 10 + 1.3 * (dx == "sepsis") + 0.4 * frailty + 0.5 * ("cancer" in act))
        if planned:
            p_die *= 0.2
        u = rng.random()
        if age < 18:
            disposition = "home"
        elif u < p_die:
            disposition = "exp"
        elif u < p_die + 0.03:
            disposition = "other-hcf"
        elif u < p_die + 0.03 + (0.012 + 0.04 * ("liver" in act) + 0.01 * (age < 50)) * (not planned):
            disposition = "aadvice"
        else:
            p_snf = _sig(
                -2.9
                + 0.8 * (age - 75) / 10
                + 1.2 * (dx == "hip_fracture")
                + 0.6 * ("dementia" in act)
                + 0.12 * los_days
            )
            p_hospice = (0.07 if "cancer" in act else 0.004) * (not planned)
            v = rng.random()
            if v < p_hospice:
                disposition = "hosp"
            elif v < p_hospice + p_snf:
                disposition = "snf"
            elif v < p_hospice + p_snf + 0.04 + 0.25 * (dx == "hip_fracture"):
                disposition = "rehab"
            elif v < p_hospice + p_snf + 0.06 + 0.25 * (dx == "hip_fracture"):
                disposition = "long"
            else:
                disposition = "home"
        if disposition == "exp":
            death = disch
            end_of_life = disch

        # render the stay
        eid = sim.encounter("IMP", start, disch, priority, source, disposition)
        sim.condition(dx, start, eid, primary=True)
        if dx == "chemo":
            sim.condition("cancer", start, eid)
        for k in sorted(act)[:4]:
            sim.condition(k, start, eid)
        for key in ("hemoglobin", "sodium", "creatinine", "glucose", "wbc"):
            if rng.random() > 0.05:
                sim.obs(key, adm_vals[key], start, eid)
            if los_days > 4 and rng.random() > 0.3:
                mid = start + timedelta(days=los_days // 2)
                sim.obs(key, 0.5 * (adm_vals[key] + dis_vals[key]), mid, eid)
            if los_days >= 1 and rng.random() > 0.08:
                sim.obs(key, dis_vals[key], disch, eid)
        for dkey in INPATIENT_DRUGS.get(dx, []):
            sim.med(dkey, start, eid)

        # readmission hazard at discharge (synthetic ground truth)
        prior365 = sum(1 for s, _ in sim.admissions if 0 < _days(s, start) <= 365)
        ed180 = sum(1 for e in ed_dates if 0 < _days(e, start) <= 180)
        lo = disch - timedelta(days=365)
        n_drugs = len({k for d, k in sim.drug_events if lo <= d <= disch} | {x for k in act for x in maint[k]})
        hgb_d, na_d, cr_d = dis_vals["hemoglobin"], dis_vals["sodium"], dis_vals["creatinine"]
        z = (
            -3.90
            + 0.18 * (age - 65) / 10
            + 0.45 * ("chf" in act)
            + 0.35 * ("copd" in act)
            + 0.40 * ("ckd" in act)
            + 0.55 * ("cancer" in act)
            + 0.20 * ("diabetes" in act)
            + 0.30 * ("dementia" in act)
            + 0.45 * ("liver" in act)
            + 0.30 * min(prior365, 4)
            + 0.12 * min(ed180, 4)
            + 0.20 * math.log1p(los_days)
            + 0.65 * (hgb_d < 10)
            + 0.25 * (10 <= hgb_d < 11.5)
            + 0.55 * (na_d < 133)
            + 0.35 * (cr_d > 2.0)
            + 0.30 * (n_drugs >= 10)
            + {"snf": 0.35, "hosp": -1.2, "rehab": 0.10, "aadvice": 0.6, "long": 0.2}.get(disposition, 0.0)
            + 0.35 * ("chf" in act and age >= 75)
            + 0.40 * frailty
            + {"chf_exac": 0.25, "copd_exac": 0.20, "sepsis": 0.20, "hip_fracture": -0.20}.get(dx, 0.0)
            - 0.9 * planned
        )
        p_readmit = _sig(z)
        sim.admissions.append((start, disch))
        sim.truth.append(
            {
                "key": f"{ehr}:{sim.pid}|{ehr}:{eid}",
                "discharge": disch.isoformat(),
                "p_true": p_readmit,
                "planned": planned,
                "disposition": disposition,
            }
        )
        if disposition == "exp":
            break
        # discharge prescriptions of maintenance drugs
        for k in sorted(act):
            for dkey in maint[k]:
                if rng.random() < 0.5:
                    sim.med(dkey, disch, eid)
        # staged elective procedure after AMI (planned readmission)
        if dx == "ami" and rng.random() < 0.3:
            electives.append((disch + timedelta(days=int(rng.integers(7, 26))), "cad_staged"))
            electives.sort()
        t = disch
        p_d30 = _sig(-4.4 + 0.5 * (age - 65) / 10 + 2.8 * (disposition == "hosp") + 0.5 * ("cancer" in act))
        if disposition != "other-hcf" and rng.random() < p_d30:
            death = disch + timedelta(days=int(rng.integers(1, 31)))
            end_of_life = death
            break
        if disposition != "other-hcf" and rng.random() < p_readmit:
            delay = 1 + int(29 * rng.beta(0.9, 1.8))
            next_unplanned = disch + timedelta(days=delay)
        else:
            next_unplanned = gap(disch)

    if death and death > WINDOW_END:
        death = None

    # ---- render ambulatory and ED encounters (skip those inside stays or after death) ----
    def free(d: date) -> bool:
        if death and d >= death:
            return False
        return all(not (s <= d <= e) for s, e in sim.admissions)

    for d, kind in sorted(amb_dates):
        if d > WINDOW_END or (death and d >= death):
            continue
        if not free(d):
            if kind.startswith("onset:"):  # diagnosed during a stay: record on the problem list
                sim.condition(kind.split(":", 1)[1], d, None)
            continue
        eid = sim.encounter("AMB", d, d, "R")
        if rng.random() < 0.6:
            sim.obs("systolic_bp", float(rng.normal(132 + 8 * ("hypertension" in onset), 14)), d, eid)
        if kind.startswith("onset:"):
            k = kind.split(":", 1)[1]
            sim.condition(k, d, eid, primary=True)
            for dkey in maint[k]:
                sim.med(dkey, d, eid)
        elif kind == "renew":
            for k in sorted(active(d)):
                for dkey in maint[k]:
                    sim.med(dkey, d, eid)
    for d in ed_dates:
        if not free(d):
            continue
        eid = sim.encounter("EMER", d, d, "EM")
        sim.obs("systolic_bp", float(rng.normal(138, 18)), d, eid)

    # chronic conditions diagnosed before the window: problem-list entries without encounter
    for k, o in onset.items():
        if o < WINDOW_START:
            sim.condition(k, o, None)

    patient: dict = {"resourceType": "Patient", "id": sim.pid, "gender": gender, "birthDate": birth.isoformat()}
    if death:
        patient["deceasedDateTime"] = sim._ts(death, 12)
    entries = [{"fullUrl": f"urn:{ehr}:Patient/{sim.pid}", "resource": patient}] + sim.resources
    bundle = {
        "resourceType": "Bundle",
        "id": f"bundle-{sim.pid}",
        "type": "collection",
        "meta": {"source": ehr},
        "entry": entries,
    }
    return bundle, sim.truth


def corrupt(bundle: dict, kind: str, rng: np.random.Generator) -> dict:
    """Return a copy of ``bundle`` with one deliberate defect of type ``kind``."""
    b = copy.deepcopy(bundle)
    res = [e["resource"] for e in b["entry"]]
    by_type: dict[str, list[dict]] = {}
    for r in res:
        by_type.setdefault(r["resourceType"], []).append(r)
    patient = by_type["Patient"][0]
    enc = by_type.get("Encounter", [])
    if kind == "missing_subject" and enc:
        del enc[int(rng.integers(len(enc)))]["subject"]
    elif kind == "end_before_start" and enc:
        e = enc[int(rng.integers(len(enc)))]
        s = datetime.fromisoformat(e["period"]["start"].replace("Z", "+00:00"))
        e["period"]["end"] = e["period"]["start"].replace(
            s.strftime("%Y-%m-%d"), (s - timedelta(days=3)).strftime("%Y-%m-%d")
        )
    elif kind == "bad_gender":
        patient["gender"] = "M"
    elif kind == "unknown_unit" and by_type.get("Observation"):
        by_type["Observation"][0]["valueQuantity"]["unit"] = "mg"
    elif kind == "dangling_encounter_ref" and by_type.get("Condition"):
        by_type["Condition"][0]["encounter"] = {"reference": "Encounter/enc-99999"}
    elif kind == "future_birthdate":
        patient["birthDate"] = "2090-05-01"
    elif kind == "implausible_value" and by_type.get("Observation"):
        hgb = [o for o in by_type["Observation"] if o["code"]["coding"][0]["code"] == "718-7"]
        target = hgb[0] if hgb else by_type["Observation"][0]
        target["code"]["coding"][0]["code"] = "718-7"
        target["valueQuantity"] = {"value": 140.0, "unit": "g/dL"}  # g/L value labelled g/dL
    elif kind == "two_patients":
        p2 = copy.deepcopy(patient)
        p2["id"] = patient["id"] + "x"
        b["entry"].append({"fullUrl": "urn:dup", "resource": p2})
    elif kind == "unknown_resource_type":
        b["entry"].append({"resource": {"resourceType": "Procedur", "id": "x1"}})
    elif kind == "encounter_after_death" and enc:
        last = max(datetime.fromisoformat(e["period"]["start"].replace("Z", "+00:00")).date() for e in enc)
        d = last - timedelta(days=1)
        patient["deceasedDateTime"] = d.isoformat() + "T00:00:00"
    else:  # fallback when the bundle lacks the needed resource
        patient["gender"] = "M"
        kind = "bad_gender"
    b.setdefault("meta", {})["tag"] = [{"code": "synthetic-corruption"}]
    return b


def generate(scale: Scale) -> tuple[dict[str, list[dict]], dict, list[dict]]:
    """Generate bundles per EHR, a manifest and the admission-level ground truth."""
    by_ehr: dict[str, list[dict]] = {e: [] for e in EHR_SOURCES}
    truth: list[dict] = []
    crng = np.random.default_rng([scale.seed, 10**9])
    corrupted: dict[str, str] = {}
    for i in range(scale.n_patients):
        bundle, t = simulate_patient(i, scale.seed)
        truth.extend(t)
        if crng.random() < scale.invalid_rate:
            kind = CORRUPTIONS[int(crng.integers(len(CORRUPTIONS)))]
            bundle = corrupt(bundle, kind, crng)
            corrupted[bundle["id"]] = kind
        by_ehr[bundle["meta"]["source"]].append(bundle)
    n_res = Counter(e["resource"]["resourceType"] for bs in by_ehr.values() for b in bs for e in b["entry"])
    manifest = {
        "synthetic": True,
        "scale": scale.name,
        "seed": scale.seed,
        "n_patients": scale.n_patients,
        "window": [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        "bundles_per_ehr": {e: len(v) for e, v in by_ehr.items()},
        "resources": dict(sorted(n_res.items())),
        "corrupted_bundles": corrupted,
        "corruption_counts": dict(sorted(Counter(corrupted.values()).items())),
    }
    return by_ehr, manifest, truth


def write(by_ehr: dict[str, list[dict]], manifest: dict, out_dir: Path, level: int = 6) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ehr, bundles in by_ehr.items():
        with gzip.GzipFile(out_dir / f"{ehr}.ndjson.gz", "wb", compresslevel=level, mtime=0) as fh:
            for b in bundles:
                fh.write((json.dumps(b, separators=(",", ":")) + "\n").encode())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def read_lines(data_dir: Path) -> list[tuple[str, bytes]]:
    """Return (source file, raw NDJSON line) pairs for every bundle file in ``data_dir``."""
    out = []
    for ehr in EHR_SOURCES:
        p = data_dir / f"{ehr}.ndjson.gz"
        if p.exists():
            with gzip.open(p, "rb") as fh:
                out.extend((ehr, line) for line in fh if line.strip())
    return out
