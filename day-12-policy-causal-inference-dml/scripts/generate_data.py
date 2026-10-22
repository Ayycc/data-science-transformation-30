"""Entry point for `make data`: write the full SYNTHETIC datasets to data/raw/ (git-ignored),
refresh the small committed samples in data/sample/, and try to cache the public LaLonde files."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policy_causal.config import DATA_RAW, RunConfig  # noqa: E402
from policy_causal.lalonde import load_lalonde  # noqa: E402
from policy_causal.pipeline import write_samples  # noqa: E402
from policy_causal.simulate import simulate_individuals, simulate_panel  # noqa: E402

if __name__ == "__main__":
    cfg = RunConfig()
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    simulate_individuals(cfg.n_people, cfg.seed).to_csv(DATA_RAW / "individuals.csv", index=False)
    simulate_panel(cfg.n_counties, cfg.n_quarters, cfg.cohorts, cfg.seed).to_csv(DATA_RAW / "county_panel.csv", index=False)
    write_samples(cfg)
    _, path = load_lalonde("auto")
    print(f"SYNTHETIC data written to {DATA_RAW}; samples refreshed; LaLonde path: {path}")
