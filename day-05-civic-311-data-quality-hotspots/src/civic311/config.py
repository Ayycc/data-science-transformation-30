"""Constants shared by the generator, the pipeline and the tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
SAMPLE_DIR = DATA_DIR / "sample"
RESULTS_DIR = ROOT / "results"
REFERENCE_DIR = ROOT / "reference"
REVIEWED_MAPPING = REFERENCE_DIR / "category_mapping_reviewed.csv"

# Public 311 sources probed before falling back to the synthetic generator (Socrata, no key needed).
PUBLIC_PROBES = {
    "chicago_311": "https://data.cityofchicago.org/resource/v6vf-nfxy.json?$limit=1",
    "nyc_311": "https://data.cityofnewyork.us/resource/erm2-nwe9.json?$limit=1",
    "sf_311": "https://data.sfgov.org/resource/vw6y-z8j6.json?$limit=1",
}

CATALOG_SEED = 2025  # fixed: the set of raw category spellings does not depend on the data seed
WINDOW_START = "2025-01-01 00:00:00"  # first valid created timestamp
EXTRACT_TS = "2026-01-01 00:00:00"  # the extract is taken at this instant; nothing may be later
PLACEHOLDER_CUTOFF = "2000-01-01 00:00:00"  # dates before this are sentinel values (1900-01-01, 1970-01-01)
N_WARDS = 50
N_NEIGHBOURHOODS = 40

H3_RES = 8  # ~0.74 km2 hexagons
MIN_CELL_N = 30  # cells with fewer completed requests are not scored
FDR_Q = 0.05  # Benjamini-Hochberg false discovery rate for Gi* hot/cold spots
MATERIAL_EXCESS = 0.25  # a hotspot must also be >= 25% slower than the city (local Gi* window mean)

# KPI anomaly detection
ANOMALY_WINDOW = 28  # trailing days used as the baseline
ANOMALY_Z = 6.0  # robust z threshold for data-quality rule rates
ANOMALY_MIN_LIFT = 0.01  # and the rate must exceed baseline by >= 1 percentage point
ANOMALY_MIN_FAILS = 5  # and at least 5 rows must fail that day (guards low-volume days)
VOLUME_Z = 5.0  # robust z threshold on log(volume / same-weekday baseline)

SLO_PASS_TARGET = 0.90  # data-quality SLO: >= 90% of each quarter's records pass the contract


@dataclass(frozen=True)
class Scale:
    name: str
    n_target: int  # approximate number of requests before duplicates are appended
    seed: int


FULL = Scale("full", 1_000_000, 42)
SAMPLE = Scale("sample", 8_000, 43)
SCALES = {s.name: s for s in (FULL, SAMPLE)}


def data_dir(scale: Scale) -> Path:
    return SAMPLE_DIR if scale.name == "sample" else GENERATED_DIR
