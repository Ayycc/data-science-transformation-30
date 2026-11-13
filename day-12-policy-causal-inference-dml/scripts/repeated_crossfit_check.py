"""Follow-up check: does repeated cross-fitting move the headline employment estimate?

The committed run fits the headline LinearDML once, with one 5-fold split. This script refits it with
econml's repeated cross-fitting (``mc_iters`` different fold splits, nuisance predictions aggregated by
the median) on the SAME committed seed and data, and writes results/repeated_crossfit_check.json.
It only reports; the headline numbers in metrics.json are unchanged.

    PYTHONPATH=src python scripts/repeated_crossfit_check.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from econml.dml import LinearDML

from policy_causal.config import CONFOUNDERS, MODIFIERS, RESULTS, RunConfig
from policy_causal.dml import nuisance_models
from policy_causal.simulate import observed, simulate_individuals, true_effects

MC_ITERS = 10


def fit(df, outcome: str, controls: list[str], folds: int, seed: int, mc_iters: int | None) -> dict[str, float]:
    reg, clf = nuisance_models(seed)
    m = LinearDML(
        model_y=reg,
        model_t=clf,
        discrete_treatment=True,
        cv=folds,
        random_state=seed,
        mc_iters=mc_iters,
        mc_agg="median",
    )
    x = df[MODIFIERS].to_numpy(dtype=float)
    m.fit(df[outcome].to_numpy(dtype=float), df["treated"].to_numpy(), X=x, W=df[controls].to_numpy(dtype=float))
    lo, hi = m.ate_interval(x, alpha=0.05)
    return {"estimate": float(m.ate(x)), "ci_low": float(np.ravel(lo)[0]), "ci_high": float(np.ravel(hi)[0])}


def main() -> None:
    cfg = RunConfig()
    full = simulate_individuals(cfg.n_people, cfg.seed)
    df = observed(full)
    truth = true_effects(full)
    controls = [c for c in CONFOUNDERS if c not in MODIFIERS]
    out: dict[str, object] = {"seed": cfg.seed, "folds": cfg.dml_folds, "mc_iters": MC_ITERS, "mc_agg": "median"}
    for y, key in [("employed_post", "ate_employment"), ("earnings_post", "ate_earnings")]:
        t0 = time.time()
        single = fit(df, y, controls, cfg.dml_folds, cfg.seed, None)
        rep = fit(df, y, controls, cfg.dml_folds, cfg.seed, MC_ITERS)
        tv = truth[key]
        out[y] = {
            "true_sate": tv,
            "single_split": single | {"covers_truth": single["ci_low"] <= tv <= single["ci_high"]},
            "repeated_crossfit": rep | {"covers_truth": rep["ci_low"] <= tv <= rep["ci_high"]},
            "seconds": round(time.time() - t0, 1),
        }
        print(y, json.dumps(out[y]))
    Path(RESULTS, "repeated_crossfit_check.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
