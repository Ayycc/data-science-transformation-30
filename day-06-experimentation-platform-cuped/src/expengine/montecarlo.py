"""Monte Carlo studies: false positives under daily peeking, power, and CUPED variance reduction.

Each replicate is a full SYNTHETIC experiment: ``n_looks`` daily cohorts of ``users_per_look``
users per arm drawn from the simulator, analysed after every day. All four metrics are
analysed on the same draws, with and without CUPED.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from expengine.cuped import stats_from_sums
from expengine.sequential import always_valid_ci, msprt_pvalues, naive_pvalues
from expengine.simulate import COVARIATE, METRICS, EffectSpec, draw_outcomes, draw_users, true_effects

TAU_REL = 0.05  # mSPRT mixing sd = 5% of the historical baseline mean of each metric


def historical_baselines(n: int = 200_000, seed: int = 7) -> dict[str, float]:
    """Baseline (control) means from a 'historical' SYNTHETIC cohort, used to fix tau before any test."""
    rng = np.random.default_rng(seed)
    users = draw_users(n, rng)
    out = draw_outcomes(users, np.zeros(n, bool), EffectSpec(), rng)
    return {m: float(out[m].mean()) for m in METRICS}


def tau2_for(baselines: dict[str, float], tau_rel: float = TAU_REL) -> dict[str, float]:
    return {m: float((tau_rel * baselines[m]) ** 2) for m in baselines}


def _cum_sums(y: np.ndarray, x: np.ndarray) -> dict[str, np.ndarray]:
    """y, x shaped (reps, looks, users_per_look) -> cumulative per-look sufficient statistics."""
    m = y.shape[2]
    return {
        "n": np.cumsum(np.full(y.shape[:2], float(m)), axis=1),
        "sy": np.cumsum(y.sum(2), axis=1),
        "syy": np.cumsum((y * y).sum(2), axis=1),
        "sx": np.cumsum(x.sum(2), axis=1),
        "sxx": np.cumsum((x * x).sum(2), axis=1),
        "sxy": np.cumsum((x * y).sum(2), axis=1),
    }


def simulate_looks(
    effect: EffectSpec,
    n_reps: int,
    n_looks: int,
    users_per_look: int,
    seed: int,
    max_users_per_chunk: int = 1_000_000,
) -> dict[str, dict[str, np.ndarray]]:
    """Per-replicate, per-look (diff, var) for every metric, naive and CUPED. Arrays are (reps, looks)."""
    chunk = max(1, max_users_per_chunk // (n_looks * users_per_look))
    rng = np.random.default_rng(seed)
    res: dict[str, dict[str, list]] = {
        m: {k: [] for k in ("d_naive", "v_naive", "d_cuped", "v_cuped")} for m in METRICS
    }
    done = 0
    while done < n_reps:
        r = min(chunk, n_reps - done)
        size = r * n_looks * users_per_look
        arms = {}
        for name, flag in (("c", False), ("t", True)):
            users = draw_users(size, rng)
            out = draw_outcomes(users, np.full(size, flag), effect, rng)
            arms[name] = (users, out)
        for m in METRICS:
            sums = {}
            for name, (users, out) in arms.items():
                y = out[m].reshape(r, n_looks, users_per_look)
                x = users[COVARIATE[m]].reshape(r, n_looks, users_per_look)
                sums[name] = _cum_sums(y, x)
            naive = stats_from_sums(sums["c"], sums["t"])
            cup = stats_from_sums(sums["c"], sums["t"], cuped=True)
            res[m]["d_naive"].append(naive["diff"])
            res[m]["v_naive"].append(naive["var"])
            res[m]["d_cuped"].append(cup["diff"])
            res[m]["v_cuped"].append(cup["var"])
        done += r
    return {m: {k: np.concatenate(v, axis=0) for k, v in d.items()} for m, d in res.items()}


def evaluate(
    sims: dict[str, dict[str, np.ndarray]],
    tau2: dict[str, float],
    alpha: float = 0.05,
    truth: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Rejection rates by method, stopping looks, CI coverage and CUPED variance reduction."""
    out: dict[str, dict] = {}
    for m, s in sims.items():
        dn, vn, dc, vc = s["d_naive"], s["v_naive"], s["d_cuped"], s["v_cuped"]
        p_naive = naive_pvalues(dn, vn)
        p_seq = msprt_pvalues(dn, vn, tau2[m])
        p_seq_c = msprt_pvalues(dc, vc, tau2[m])
        p_fixed_c = naive_pvalues(dc[:, -1], vc[:, -1])
        prob_pos = stats.norm.cdf(dn / np.sqrt(vn))  # flat-prior Normal posterior P(T > C)
        rej = {
            "naive_peeking": (p_naive < alpha).any(1),
            "fixed_horizon": p_naive[:, -1] < alpha,
            "fixed_horizon_cuped": p_fixed_c < alpha,
            "msprt": (p_seq < alpha).any(1),
            "msprt_cuped": (p_seq_c < alpha).any(1),
            "bayes_peeking_p95": (prob_pos >= 0.95).any(1),
        }
        curves = {
            "naive_peeking": np.maximum.accumulate(p_naive < alpha, axis=1).mean(0),
            "msprt": (p_seq < alpha).mean(0),
            "msprt_cuped": (p_seq_c < alpha).mean(0),
            "bayes_peeking_p95": np.maximum.accumulate(prob_pos >= 0.95, axis=1).mean(0),
        }

        def first_look(p):
            hit = p < alpha
            idx = np.where(hit.any(1), hit.argmax(1) + 1, np.nan)
            return float(np.nanmedian(idx)) if np.isfinite(idx).any() else None

        var_ratio_emp = float(np.var(dc[:, -1], ddof=1) / np.var(dn[:, -1], ddof=1))
        entry = {
            "rejection_rate": {k: float(v.mean()) for k, v in rej.items()},
            "rejection_curve": {k: v.tolist() for k, v in curves.items()},
            "median_stopping_look": {"msprt": first_look(p_seq), "msprt_cuped": first_look(p_seq_c)},
            "cuped_variance_reduction_empirical": 1 - var_ratio_emp,
            "cuped_variance_reduction_insample_mean": float(np.mean(1 - vc[:, -1] / vn[:, -1])),
            "n_reps": int(dn.shape[0]),
            "n_looks": int(dn.shape[1]),
        }
        if truth is not None:
            lo, hi = always_valid_ci(dc, vc, tau2[m], alpha)
            covered = ((lo <= truth[m]) & (hi >= truth[m])).all(1)
            z = stats.norm.ppf(1 - alpha / 2)
            fixed_cov = np.abs(dc[:, -1] - truth[m]) <= z * np.sqrt(vc[:, -1])
            entry["true_effect"] = float(truth[m])
            entry["always_valid_ci_coverage_all_looks"] = float(covered.mean())
            entry["fixed_ci_coverage_final"] = float(fixed_cov.mean())
            entry["bias_cuped_final"] = float(dc[:, -1].mean() - truth[m])
        out[m] = entry
    return out


def population_truth(effect: EffectSpec, n: int = 400_000, seed: int = 11) -> dict[str, float]:
    """Population average treatment effect of ``effect`` (absolute units) from a large cohort."""
    users = draw_users(n, np.random.default_rng(seed))
    return {m: v["ate"] for m, v in true_effects(users, effect).items()}
