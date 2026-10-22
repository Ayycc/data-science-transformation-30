"""Refutation tests for the DML estimate (DoWhy refuters + a negative-control outcome) and E-values."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .dag import TREATMENT, build_dag
from .dml import linear_dml_ate, nuisance_models


def dowhy_dml_refutations(
    df: pd.DataFrame, outcome: str, modifiers: list[str], folds: int, sims: int, seed: int, quick: bool = False
) -> dict[str, object]:
    """Estimate the ATE with DoWhy's econml LinearDML backend and run three DoWhy refuters on it."""
    from dowhy import CausalModel
    from econml.dml import LinearDML  # noqa: F401  (imported so DoWhy can resolve the backend)

    g = build_dag()
    cols = [c for c in df.columns if c in g.nodes]
    model = CausalModel(data=df[cols].copy(), treatment=TREATMENT, outcome=outcome, graph=g, effect_modifiers=modifiers)
    estimand = model.identify_effect(proceed_when_unidentifiable=False)
    reg, clf = nuisance_models(seed, quick)
    params = {
        "init_params": {"model_y": reg, "model_t": clf, "discrete_treatment": True, "cv": folds, "random_state": seed},
        "fit_params": {},
    }
    est = model.estimate_effect(estimand, method_name="backdoor.econml.dml.LinearDML", method_params=params, effect_modifiers=modifiers)
    base = float(est.value)
    out: dict[str, object] = {"dowhy_estimate": base, "backdoor_variables": sorted(estimand.get_backdoor_variables())}
    specs = {
        "placebo_treatment": {"method_name": "placebo_treatment_refuter", "placebo_type": "permute"},
        "random_common_cause": {"method_name": "random_common_cause"},
        "data_subset_80pct": {"method_name": "data_subset_refuter", "subset_fraction": 0.8},
    }
    for name, kw in specs.items():
        ref = model.refute_estimate(estimand, est, num_simulations=sims, random_seed=seed, show_progress_bar=False, **kw)
        new = float(ref.new_effect)
        rr = ref.refutation_result or {}
        if name == "placebo_treatment":
            ok = abs(new) < 0.10 * abs(base)
            rule = "|placebo effect| < 10% of the estimate"
        elif name == "random_common_cause":
            ok = abs(new - base) < 0.05 * abs(base)
            rule = "estimate moves < 5%"
        else:
            ok = abs(new - base) < 0.10 * abs(base)
            rule = "estimate moves < 10%"
        out[name] = {
            "original": base,
            "new_effect": new,
            "change_pct": float(100 * (new - base) / base) if base else float("nan"),
            "dowhy_p_value": float(rr.get("p_value", np.nan)),
            "simulations": sims,
            "pass_rule": rule,
            "passed": bool(ok),
        }
    return out


def negative_control_outcome(
    df: pd.DataFrame, modifiers: list[str], confounders: list[str], folds: int, seed: int, quick: bool = False
) -> dict[str, object]:
    """Effect of enrolment on earnings two years *before* enrolment; must be zero if the design is right.

    ``prior_earn_2`` is not a cause of enrolment once ``prior_earn_1`` is known (see the DAG), so
    the adjusted estimate should be indistinguishable from zero.
    """
    ctrl = [c for c in confounders if c not in modifiers and c != "prior_earn_2"]
    mods = [m for m in modifiers if m != "prior_earn_2"]
    r = linear_dml_ate(df, "prior_earn_2", mods, ctrl, folds, seed, quick)
    naive = float(df.loc[df[TREATMENT] == 1, "prior_earn_2"].mean() - df.loc[df[TREATMENT] == 0, "prior_earn_2"].mean())
    return {
        "outcome": "prior_earn_2 (pre-programme earnings)",
        "naive_difference": naive,
        "estimate": r["estimate"],
        "ci_low": r["ci_low"],
        "ci_high": r["ci_high"],
        "pass_rule": "95% CI contains 0",
        "passed": bool(r["ci_low"] <= 0 <= r["ci_high"]),
    }


def evalue_rr(rr: float) -> float:
    """VanderWeele & Ding (2017) E-value for a risk ratio."""
    if rr < 1:
        rr = 1 / rr
    return float(rr + math.sqrt(rr * (rr - 1)))


def evalue_from_risks(p1: float, p0: float, ci_low_diff: float) -> dict[str, float]:
    """E-value for a risk difference converted to a risk ratio at the estimated baseline risk p0."""
    rr = p1 / p0
    rr_lo = (p0 + ci_low_diff) / p0
    return {"risk_ratio": rr, "evalue_point": evalue_rr(rr), "evalue_ci": evalue_rr(rr_lo) if rr_lo > 1 else 1.0}


def evalue_continuous(effect: float, ci_low: float, sd: float) -> dict[str, float]:
    """E-value for a mean difference via the standardised-difference approximation RR ~ exp(0.91 d)."""
    d = effect / sd
    dl = ci_low / sd
    rr = math.exp(0.91 * d)
    rr_lo = math.exp(0.91 * dl)
    return {"std_mean_diff": d, "approx_risk_ratio": rr, "evalue_point": evalue_rr(rr), "evalue_ci": evalue_rr(rr_lo) if rr_lo > 1 else 1.0}
