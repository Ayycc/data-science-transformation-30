"""Full analysis of one experiment: SRM, naive vs CUPED, ratio metric, sequential trace,
Bayesian view, novelty check, segment slices, power, and comparison to ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd

from expengine.bayes import bayes_decision, beta_binomial, normal_posterior
from expengine.cuped import cuped_estimate, diff_in_means, regression_adjusted, stats_from_sums
from expengine.hte import segment_effects
from expengine.power import n_per_arm_means
from expengine.ratio import ratio_metric_test
from expengine.sequential import msprt_sequence
from expengine.simulate import COVARIATE, METRICS, ExperimentSpec
from expengine.srm import srm_test

ALPHA = 0.05
LOSS_TOLERANCE_REL = 0.005  # accept at most 0.5% of the control mean as expected loss when shipping


def sequential_trace(df: pd.DataFrame, metric: str, n_looks: int, tau2: float, cuped: bool = True) -> dict:
    """Daily looks: at look d the analysis includes every user who arrived before day d."""
    t = df["treat"].to_numpy(bool)
    y = df[metric].to_numpy(float)
    x = df[COVARIATE[metric]].to_numpy(float)
    day = df["arrival_day"].to_numpy()

    def cum(mask):
        def bc(w):
            return np.cumsum(np.bincount(day[mask], weights=w, minlength=n_looks))

        ym, xm = y[mask], x[mask]
        return {
            "n": bc(np.ones(mask.sum())),
            "sy": bc(ym),
            "syy": bc(ym * ym),
            "sx": bc(xm),
            "sxx": bc(xm * xm),
            "sxy": bc(xm * ym),
        }

    s = stats_from_sums(cum(~t), cum(t), cuped=cuped)
    seq = msprt_sequence(s["diff"], s["var"], tau2, ALPHA)
    seq["diff"] = s["diff"]
    seq["var"] = s["var"]
    seq["n_cum"] = cum(~t)["n"] + cum(t)["n"]
    return seq


def novelty_check(df: pd.DataFrame) -> dict:
    """Compare the per-day session lift in exposure days 0-2 with days 7-13 (within user)."""
    t = df["treat"].to_numpy(bool)
    early = df["sessions_early"].to_numpy(float) / 3
    late = df["sessions_late"].to_numpy(float) / 7
    e = diff_in_means(early, t, ALPHA, "early")
    lt = diff_in_means(late, t, ALPHA, "late")
    d = diff_in_means(early - late, t, ALPHA, "early_minus_late")
    flagged = bool(d.p_value < ALPHA and e.rel_lift > lt.rel_lift)
    return {
        "early_rel_lift": e.rel_lift,
        "late_rel_lift": lt.rel_lift,
        "late_rel_ci": [lt.rel_ci_low, lt.rel_ci_high],
        "decay_p_value": d.p_value,
        "flagged": flagged,
    }


def analyze_experiment(
    df: pd.DataFrame, spec: ExperimentSpec, truth: dict, tau2: dict[str, float], seed: int = 0
) -> dict:
    df = df.copy()
    df["treat"] = df["variant"].eq("treatment")
    t = df["treat"].to_numpy(bool)
    n_c, n_t = int((~t).sum()), int(t.sum())
    srm = srm_test([n_c, n_t], [0.5, 0.5])

    metrics = {}
    for m in METRICS:
        y = df[m].to_numpy(float)
        x = df[COVARIATE[m]].to_numpy(float)
        naive = diff_in_means(y, t, ALPHA, m)
        cup = cuped_estimate(y, x, t, ALPHA, m)
        ols = regression_adjusted(y, x, t)
        seq = sequential_trace(df, m, spec.enrollment_days, tau2[m])
        true = truth["effects"][m]
        metrics[m] = {
            "naive": naive.to_dict(),
            "cuped": cup.to_dict(),
            "ols_check": ols,
            "always_valid_p_final": float(seq["p_always_valid"][-1]),
            "always_valid_ci_final": [float(seq["ci_low"][-1]), float(seq["ci_high"][-1])],
            "first_reject_day": None if seq["first_reject_look"] is None else seq["first_reject_look"] + 1,
            "first_naive_reject_day": None
            if seq["first_naive_reject_look"] is None
            else seq["first_naive_reject_look"] + 1,
            "true_ate": true["ate"],
            "true_rel": true["rel"],
            "cuped_ci_covers_truth": bool(cup.ci_low <= true["ate"] <= cup.ci_high),
            "always_valid_ci_covers_truth": bool(seq["ci_low"][-1] <= true["ate"] <= seq["ci_high"][-1]),
        }
    primary = spec.primary_metric
    trace = sequential_trace(df, primary, spec.enrollment_days, tau2[primary])
    pm = metrics[primary]
    ctrl_mean = pm["naive"]["control_mean"]

    if primary == "converted":
        conv = df.groupby("treat")["converted"].sum()
        bayes = beta_binomial(int(conv[False]), n_c, int(conv[True]), n_t, seed=seed)
        prob, loss = bayes["prob_t_better"], bayes["expected_loss_t"]
    else:
        bayes = normal_posterior(pm["cuped"]["diff"], pm["cuped"]["se"])
        prob, loss = bayes["prob_positive"], bayes["expected_loss_t"]
    bayes["decision"] = bayes_decision(prob, loss, LOSS_TOLERANCE_REL * ctrl_mean)
    bayes["loss_tolerance"] = LOSS_TOLERANCE_REL * ctrl_mean

    ratio = ratio_metric_test(df["revenue"].to_numpy(), df["sessions"].to_numpy(), t, ALPHA)
    novelty = novelty_check(df)
    hte = segment_effects(df, {primary: COVARIATE[primary]}, ["platform", "tenure"], "treat", ALPHA)

    sd_c = float(df.loc[~df["treat"], primary].std(ddof=1))
    vr = max(pm["cuped"]["variance_reduction"], 0.0)
    mde_abs = spec.mde_rel * ctrl_mean
    users_per_day_arm = len(df) / spec.enrollment_days / 2
    n_naive = n_per_arm_means(sd_c, mde_abs)
    n_cuped = n_per_arm_means(sd_c, mde_abs, variance_reduction=min(vr, 0.99))
    power = {
        "mde_rel": spec.mde_rel,
        "n_per_arm_naive": n_naive,
        "n_per_arm_cuped": n_cuped,
        "days_naive": float(n_naive / users_per_day_arm),
        "days_cuped": float(n_cuped / users_per_day_arm),
    }

    guardrail_harm = [
        m
        for m in METRICS
        if m != primary and metrics[m]["always_valid_p_final"] < ALPHA and metrics[m]["cuped"]["diff"] < 0
    ]
    if srm.flagged:
        decision = "INVALID: sample-ratio mismatch. Do not read the metrics; fix assignment/logging and rerun."
    elif pm["always_valid_p_final"] < ALPHA and pm["cuped"]["diff"] > 0:
        if guardrail_harm:
            decision = f"DO NOT SHIP: primary metric up but guardrail(s) {', '.join(guardrail_harm)} regressed."
        elif novelty["flagged"]:
            decision = "SHIP WITH CAUTION: significant lift, but it decays (novelty). Plan on the late-window lift."
        else:
            decision = "SHIP: significant lift on the primary metric (always-valid p < 0.05), no guardrail harm."
    elif pm["always_valid_p_final"] < ALPHA:
        decision = "DO NOT SHIP: primary metric significantly worse."
    else:
        decision = "NO DETECTABLE EFFECT at alpha = 0.05 (always-valid). Do not ship on this evidence."

    return {
        "exp_id": spec.exp_id,
        "name": spec.name,
        "hypothesis": spec.hypothesis,
        "primary_metric": primary,
        "enrollment_days": spec.enrollment_days,
        "n_control": n_c,
        "n_treatment": n_t,
        "srm": {
            "chi2": srm.chi2,
            "p_value": srm.p_value,
            "flagged": srm.flagged,
            "share_treatment": srm.observed_share[1],
        },
        "metrics": metrics,
        "trace": {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in trace.items()},
        "bayes": bayes,
        "ratio_revenue_per_session": ratio,
        "novelty": novelty,
        "hte": hte.to_dict(orient="records"),
        "power": power,
        "guardrail_harm": guardrail_harm,
        "decision": decision,
        "truth": truth,
    }
