# Day 03: Hospital Readmission Risk on an OMOP-style Model

**Tier I - Advanced** | **Domain:** Healthcare | **Folder:** `day-03-healthcare-readmission-omop/`

**Core tools:** pandas, Pydantic, scikit-learn, DuckDB, matplotlib

**Method:** FHIR-like JSON normalisation into a common data model, cohort definition, calibrated risk modelling.

> This prompt is self-contained. Before starting, read `prompts/_conventions.md` in this repository and follow it exactly. Build everything inside `day-03-healthcare-readmission-omop/`. Do not modify other days' folders.

## Business context

A hospital network must reduce 30-day readmissions under value-based payment rules. Clinical data arrives as nested FHIR-style JSON from several EHRs.

## Data

Use Synthea synthetic patients if downloadable; otherwise write a generator producing nested FHIR-like bundles (Patient, Encounter, Condition, MedicationRequest, Observation) for ~20k patients with plausible clinical correlations.

Data rules: public or synthetic data only. Never commit raw data over 5 MB; commit the generator/download script plus a small sample (at most 1 MB) so the project runs offline. All randomness is seeded. If a public source is unreachable, fall back to the synthetic generator automatically and record which path was used in the README.

## What to build

- Pydantic schemas validating incoming bundles and a normaliser into OMOP-like tables (person, visit_occurrence, condition_occurrence, drug_exposure, measurement).
- SQL cohort definition for index admissions with exclusions (deaths, transfers, planned readmissions).
- Readmission model (regularised logistic regression and gradient boosting) with calibration plots, Brier score, and decision-curve analysis.
- Subgroup performance by age band and sex, with a short clinical-safety discussion.

## Acceptance criteria

- Invalid bundles are rejected with clear errors (tests)
- Cohort counts are reproducible from a single command
- Calibration slope reported and between 0.8 and 1.2 on test
- `make test` (or the documented equivalent) passes from a clean checkout of this folder.
- `make run` reproduces every number and figure quoted in this folder's README.

## Stretch goals (attempt only after acceptance criteria pass)

- LACE index baseline for comparison
- Survival framing of time to readmission

## Skills this project demonstrates

Healthcare data standards, Schema validation, Clinical risk modelling, Calibration

## Definition of done

1. Folder `day-03-healthcare-readmission-omop/` follows the standard layout in `prompts/_conventions.md`.
2. The folder README has: executive summary (3 sentences a VP can read), architecture diagram (Mermaid), data provenance, method, results table with real numbers produced by your run, limitations, and how to run.
3. Tests pass and are run in CI for this folder.
4. The root `README.md` progress table row for Day 03 is updated to ✅ with a one-line headline result and a link to the folder.
5. Everything is committed and pushed to `main`.
