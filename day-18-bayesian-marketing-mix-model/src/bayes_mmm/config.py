"""Central configuration. Every random process is seeded from SEED."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_SAMPLE = ROOT / "data" / "sample"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

SEED = 20261028
CHANNELS: tuple[str, ...] = ("tv", "search", "social", "retail_media")
N_WEEKS = 156  # 3 years
N_REGIONS = 10
L_MAX = 8  # adstock window (weeks)
START_DATE = "2023-01-02"
FOURIER_ORDER = 2
HDI_PROB = 0.94


@dataclass(frozen=True)
class SamplerConfig:
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9
    init: str = "jitter+adapt_diag"
    prior_samples: int = 500
    seed: int = SEED

    @classmethod
    def from_env(cls) -> SamplerConfig:
        """Allow quick smoke runs via MMM_DRAWS / MMM_TUNE / MMM_CHAINS; defaults reproduce results/."""
        return cls(
            draws=int(os.environ.get("MMM_DRAWS", cls.draws)),
            tune=int(os.environ.get("MMM_TUNE", cls.tune)),
            chains=int(os.environ.get("MMM_CHAINS", cls.chains)),
        )
