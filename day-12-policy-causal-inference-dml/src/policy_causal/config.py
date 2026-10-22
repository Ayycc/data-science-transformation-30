"""Central configuration: seeds, scales and paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_SAMPLE = ROOT / "data" / "sample"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

SEED = 20261022

# Public LaLonde / NSW replication (Rdatasets mirror on GitHub).
LALONDE_URLS = {
    # Dehejia-Wahba experimental sample: NSW treated + NSW randomised controls.
    "nsw_experimental": "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/causaldata/nsw_mixtape.csv",
    # NSW treated + PSID comparison group (observational design, MatchIt::lalonde).
    "nsw_psid": "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MatchIt/lalonde.csv",
}

# Backdoor adjustment set for the individual-level study (derived from the DAG, verified in dag.py).
CONFOUNDERS = ["age", "female", "educ", "married", "urban", "unemployed_base", "prior_earn_1", "prior_earn_2"]
# Effect modifiers handed to the causal forest.
MODIFIERS = ["unemployed_base", "educ", "age", "female", "prior_earn_1", "urban"]


@dataclass(frozen=True)
class RunConfig:
    seed: int = SEED
    n_people: int = 20_000
    n_counties: int = 60
    n_quarters: int = 24
    # first treated quarter index -> number of counties; the remainder are never treated
    cohorts: dict[int, int] = field(default_factory=lambda: {8: 15, 12: 15, 16: 15})
    n_boot: int = 200  # bootstrap reps for IPW / matching
    n_boot_did: int = 499  # county-cluster bootstrap reps for Callaway-Sant'Anna
    dml_folds: int = 5
    forest_trees: int = 600
    refute_sims: int = 20
    coverage_reps: int = 10  # extra seeds for the headline-CI coverage check
    quick: bool = False  # tiny settings for smoke tests


def quick_config() -> RunConfig:
    return RunConfig(
        n_people=2_500,
        n_counties=36,
        cohorts={8: 9, 12: 9, 16: 9},
        n_boot=20,
        n_boot_did=49,
        dml_folds=3,
        forest_trees=100,
        refute_sims=3,
        coverage_reps=2,
        quick=True,
    )
