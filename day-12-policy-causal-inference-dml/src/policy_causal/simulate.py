"""SYNTHETIC data generators with known causal truth.

Two linked designs:

* ``simulate_individuals`` - one row per adult in counties where the job-training programme is open.
  Enrolment is voluntary and confounded (people with low prior earnings, the unemployed, the young
  and people living near a training centre enrol more). Outcomes are next-year earnings and
  employment. Effects are heterogeneous and every row carries both potential outcomes, so the true
  sample average treatment effect (SATE) is known exactly.
* ``simulate_panel`` - a county-by-quarter panel with staggered adoption (three cohorts plus
  never-treated counties), an interactive factor structure, and dynamic effects that grow with
  exposure and differ by cohort.

All randomness flows from a single seed. Everything produced here is SYNTHETIC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

TRUTH_PREFIX = "true_"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def true_propensity(df: pd.DataFrame) -> np.ndarray:
    """The data-generating enrolment probability (used only for diagnostics and tests)."""
    pe1 = df["prior_earn_1"].to_numpy() / 10_000.0
    logit = (
        -0.35
        + 1.10 * df["unemployed_base"].to_numpy()
        - 0.55 * (pe1 - 2.0)
        + 0.10 * (pe1 - 2.0) ** 2
        + 0.70 * (df["age"].to_numpy() < 30)
        - 0.012 * (df["age"].to_numpy() - 35)
        - 0.12 * (df["educ"].to_numpy() - 12)
        + 0.40 * df["urban"].to_numpy()
        + 0.25 * df["female"].to_numpy()
        - 0.045 * (df["distance_km"].to_numpy() - 15)
        + 0.35 * df["unemployed_base"].to_numpy() * (df["age"].to_numpy() < 30)
    )
    return _sigmoid(np.clip(logit, -3.2, 3.2))


def true_tau_earnings(df: pd.DataFrame) -> np.ndarray:
    """Direct (non-mediated) programme effect on annual earnings, in dollars."""
    return (
        1_400.0
        + 1_900.0 * df["unemployed_base"].to_numpy()
        + 220.0 * (12 - df["educ"].to_numpy())
        - 40.0 * (df["age"].to_numpy() - 35)
        + 300.0 * df["female"].to_numpy()
    )


def simulate_individuals(n: int, seed: int) -> pd.DataFrame:
    """Individual-level SYNTHETIC records with confounded enrolment and known effects.

    Observed columns: covariates, ``distance_km`` (affects enrolment only), ``treated``, the
    post-treatment ``certificate`` (a mediator and collider: never adjust for it) and two outcomes
    ``earnings_post`` and ``employed_post``. Columns starting with ``true_`` are oracle values.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 61, n)
    female = rng.binomial(1, 0.5, n)
    educ = np.clip(np.round(rng.normal(12.3, 2.2, n)), 8, 18).astype(int)
    married = rng.binomial(1, _sigmoid(-0.8 + 0.05 * (age - 30)))
    urban = rng.binomial(1, 0.55, n)
    unemployed = rng.binomial(1, _sigmoid(-1.3 - 0.18 * (educ - 12) - 0.02 * (age - 35) + 0.2 * (1 - urban)))
    # prior earnings: two years back first, then last year depends on it (annual dollars)
    base = 9_000 + 1_900 * (educ - 8) + 260 * (age - 18) - 4.0 * (age - 18) ** 2 + 2_500 * urban - 1_500 * female
    pe2 = np.clip(base * rng.lognormal(0.0, 0.35, n), 0, None)
    pe1 = np.clip(0.75 * pe2 + 0.25 * base - 7_500 * unemployed + rng.normal(0, 2_500, n), 0, None)
    distance = np.round(rng.gamma(2.0, 7.5, n), 1)
    ability = rng.normal(0, 1, n)  # UNOBSERVED; affects outcomes and certificate, not enrolment

    df = pd.DataFrame(
        {
            "person_id": np.arange(n),
            "age": age,
            "female": female,
            "educ": educ,
            "married": married,
            "urban": urban,
            "unemployed_base": unemployed,
            "prior_earn_1": np.round(pe1, 0),
            "prior_earn_2": np.round(pe2, 0),
            "distance_km": distance,
        }
    )
    ps = true_propensity(df)
    t = rng.binomial(1, ps)

    # certificate: potential values under T=0 / T=1 from one uniform draw (monotone)
    u_cert = rng.random(n)
    p_c0 = _sigmoid(-3.0 + 0.8 * ability)
    p_c1 = _sigmoid(0.6 + 0.8 * ability + 0.1 * (educ - 12))
    c0 = (u_cert < p_c0).astype(int)
    c1 = (u_cert < np.maximum(p_c0, p_c1)).astype(int)

    # earnings potential outcomes
    tau_direct = true_tau_earnings(df)
    cert_bonus = 1_000.0
    mu0 = (
        3_000
        + 0.50 * pe1
        + 0.20 * pe2
        + 700 * (educ - 12)
        + 90 * (age - 35)
        - 3.0 * (age - 35) ** 2
        - 2_200 * unemployed
        + 900 * urban
        + 400 * married
        + 2_000 * ability
    )
    eps = rng.normal(0, 3_500, n)
    # earnings cannot be negative: censor both potential outcomes at zero
    y0 = np.maximum(mu0 + cert_bonus * c0 + eps, 0.0)
    y1 = np.maximum(mu0 + tau_direct + cert_bonus * c1 + eps, 0.0)
    # employment potential outcomes (probit); effects are expected over the idiosyncratic shock
    lat0 = 0.15 + 0.25 * (pe1 / 10_000 - 2.0) + 0.06 * (educ - 12) - 0.55 * unemployed + 0.35 * ability + 0.15 * urban - 0.012 * (age - 35)
    tau_lat = 0.22 + 0.30 * unemployed - 0.008 * (age - 35) + 0.02 * (12 - educ)
    m0 = lat0 + 0.15 * c0
    m1 = lat0 + tau_lat + 0.15 * c1
    e_emp = rng.normal(0, 1, n)
    emp0 = (m0 + e_emp > 0).astype(int)
    emp1 = (m1 + e_emp > 0).astype(int)

    df["treated"] = t
    df["certificate"] = np.where(t == 1, c1, c0)
    df["earnings_post"] = np.round(np.where(t == 1, y1, y0), 0)
    df["employed_post"] = np.where(t == 1, emp1, emp0)
    df[f"{TRUTH_PREFIX}propensity"] = ps
    df[f"{TRUTH_PREFIX}ability"] = ability
    df[f"{TRUTH_PREFIX}tau_earnings"] = y1 - y0  # total effect (direct + via certificate)
    df[f"{TRUTH_PREFIX}tau_employment"] = norm.cdf(m1) - norm.cdf(m0)
    return df


def observed(df: pd.DataFrame) -> pd.DataFrame:
    """Drop oracle columns: what an analyst would actually see."""
    return df.loc[:, [c for c in df.columns if not c.startswith(TRUTH_PREFIX)]]


def true_effects(df: pd.DataFrame) -> dict[str, float]:
    t = df["treated"] == 1
    return {
        "ate_earnings": float(df[f"{TRUTH_PREFIX}tau_earnings"].mean()),
        "att_earnings": float(df.loc[t, f"{TRUTH_PREFIX}tau_earnings"].mean()),
        "ate_employment": float(df[f"{TRUTH_PREFIX}tau_employment"].mean()),
        "att_employment": float(df.loc[t, f"{TRUTH_PREFIX}tau_employment"].mean()),
    }


# ----------------------------------------------------------------------------------------------
# County x quarter panel with staggered adoption
# ----------------------------------------------------------------------------------------------

COHORT_MULT = {8: 1.3, 12: 1.0, 16: 0.7}


def true_dynamic_effect(event_time: np.ndarray, cohort: np.ndarray) -> np.ndarray:
    """Employment-rate effect (percentage points) at ``event_time`` quarters after adoption."""
    e = np.asarray(event_time, dtype=float)
    mult = np.array([COHORT_MULT.get(int(g), 1.0) for g in np.broadcast_to(cohort, e.shape).ravel()]).reshape(e.shape)
    eff = (0.6 + 0.15 * np.minimum(e, 8)) * mult
    return np.where(e >= 0, eff, 0.0)


def simulate_panel(n_counties: int, n_quarters: int, cohorts: dict[int, int], seed: int, noise_sd: float = 0.35) -> pd.DataFrame:
    """Balanced SYNTHETIC county-quarter panel of employment rates (%) with staggered rollout.

    Early adopters have higher baseline unemployment (selection on *levels*, which DiD tolerates);
    factor loadings are independent of cohort, so parallel trends holds by construction.
    """
    rng = np.random.default_rng(seed + 1)
    n_treated = sum(cohorts.values())
    if n_treated >= n_counties:
        raise ValueError("need at least one never-treated county")
    first = np.full(n_counties, np.inf)
    idx = 0
    for g, k in sorted(cohorts.items()):
        first[idx : idx + k] = g
        idx += k
    rng.shuffle(first)
    # cohort-dependent levels: earlier adopters are poorer counties
    level_shift = np.where(np.isinf(first), 0.0, -0.25 * (20 - np.where(np.isinf(first), 20, first)))
    alpha = 68 + rng.normal(0, 3.0, n_counties) + level_shift
    lam = rng.normal(1.0, 0.35, n_counties)
    quarters = np.arange(n_quarters)
    delta = 0.05 * quarters + 0.6 * np.sin(2 * np.pi * quarters / 4)
    f = np.cumsum(rng.normal(0, 0.35, n_quarters))
    rows = []
    for i in range(n_counties):
        g = first[i]
        if np.isfinite(g):
            e = quarters - g
            tau = true_dynamic_effect(e, np.full(n_quarters, g))
            on = (e >= 0).astype(int)
        else:
            tau = np.zeros(n_quarters)
            on = np.zeros(n_quarters, dtype=int)
        y0 = alpha[i] + delta + lam[i] * f + rng.normal(0, noise_sd, n_quarters)
        rows.append(
            pd.DataFrame(
                {
                    "county_id": i,
                    "quarter": quarters,
                    "first_treated": first[i],
                    "treated_now": on,
                    "employment_rate": y0 + tau,
                    f"{TRUTH_PREFIX}effect": tau,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def panel_truth(panel: pd.DataFrame) -> dict[str, object]:
    post = panel[panel["treated_now"] == 1]
    ev = panel[np.isfinite(panel["first_treated"])].copy()
    ev["event_time"] = ev["quarter"] - ev["first_treated"]
    by_e = ev[ev["event_time"] >= 0].groupby("event_time")[f"{TRUTH_PREFIX}effect"].mean()
    return {
        "att_overall": float(post[f"{TRUTH_PREFIX}effect"].mean()),
        "event_time": {int(k): float(v) for k, v in by_e.items()},
    }
