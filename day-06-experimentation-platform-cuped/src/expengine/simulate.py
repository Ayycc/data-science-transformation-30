"""SYNTHETIC subscription-app experiment simulator with known ground truth.

Every user has a latent daily session rate and purchase propensity that drive both a 28-day
pre-period (the CUPED covariates) and a 14-day post-assignment window. Treatment changes the
session rate (steady lift plus an optional decaying novelty bump), the daily purchase
probability and the basket size, optionally by user tenure. A logging bug can silently drop
low-engagement treatment users, which creates a sample-ratio mismatch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

WINDOW_DAYS = 14
PRE_DAYS = 28
PLATFORMS = np.array(["ios", "android", "web"])
PLATFORM_P = (0.45, 0.40, 0.15)
NEW_USER_SHARE = 0.30
BASKET_SIGMA = 0.5
DRIFT_SD = 0.35
METRICS = ("sessions", "converted", "revenue", "retained_w2")
COVARIATE = {
    "sessions": "pre_sessions",
    "converted": "pre_purchases",
    "revenue": "pre_revenue",
    "retained_w2": "pre_sessions",
}


@dataclass(frozen=True)
class EffectSpec:
    lift_sessions: float = 0.0
    novelty_sessions: float = 0.0
    novelty_decay_days: float = 2.5
    lift_purchase: float = 0.0
    lift_basket: float = 0.0
    new_user_multiplier: float = 1.0
    existing_user_multiplier: float = 1.0


@dataclass(frozen=True)
class ExperimentSpec:
    exp_id: str
    name: str
    hypothesis: str
    primary_metric: str
    n_users: int
    enrollment_days: int
    effect: EffectSpec = field(default_factory=EffectSpec)
    srm_drop_rate: float = 0.0
    seed: int = 0
    mde_rel: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)


def _sum_lognormal(k: np.ndarray, mean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-user sum of ``k`` lognormal baskets with the given per-user mean."""
    idx = np.repeat(np.arange(len(k)), k)
    if len(idx) == 0:
        return np.zeros(len(k))
    vals = mean[idx] * np.exp(BASKET_SIGMA * rng.standard_normal(len(idx)) - BASKET_SIGMA**2 / 2)
    return np.bincount(idx, weights=vals, minlength=len(k))


def draw_users(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Latent user traits (experiment-window rates) plus observed 28-day pre-period behaviour."""
    platform = rng.choice(3, size=n, p=PLATFORM_P)
    is_new = rng.random(n) < NEW_USER_SHARE
    log_lam = (
        np.log(0.55) + 0.15 * (platform == 0) - 0.25 * (platform == 2) - 0.3 * is_new + 0.75 * rng.standard_normal(n)
    )
    logit_p = -4.8 + 1.0 * log_lam + 1.3 * rng.standard_normal(n)
    basket = np.exp(3.2 + 0.1 * (platform == 0) + 0.45 * rng.standard_normal(n))
    pre_sessions = rng.poisson(PRE_DAYS * np.exp(log_lam))
    pre_purchases = rng.binomial(PRE_DAYS, 1 / (1 + np.exp(-logit_p)))
    pre_revenue = _sum_lognormal(pre_purchases, basket, rng)
    # habits drift between the pre-period and the experiment window
    lam = np.exp(log_lam + DRIFT_SD * rng.standard_normal(n))
    p = 1 / (1 + np.exp(-(logit_p + DRIFT_SD * rng.standard_normal(n))))
    return {
        "platform": platform,
        "is_new": is_new,
        "lam": lam,
        "p": p,
        "basket": basket,
        "pre_sessions": pre_sessions.astype(float),
        "pre_purchases": pre_purchases.astype(float),
        "pre_revenue": pre_revenue,
    }


def _session_factors(users: dict, treat: np.ndarray, eff: EffectSpec) -> np.ndarray:
    """(n, WINDOW_DAYS) multiplier on the daily session rate."""
    mult = np.where(users["is_new"], eff.new_user_multiplier, eff.existing_user_multiplier)
    k = np.arange(WINDOW_DAYS)
    novelty = (
        eff.novelty_sessions * np.exp(-k / eff.novelty_decay_days) if eff.novelty_sessions else np.zeros(WINDOW_DAYS)
    )
    f = 1 + treat[:, None] * (eff.lift_sessions * mult[:, None] + novelty[None, :])
    return np.maximum(f, 0.0)


def _purchase(users: dict, treat: np.ndarray, eff: EffectSpec) -> tuple[np.ndarray, np.ndarray]:
    mult = np.where(users["is_new"], eff.new_user_multiplier, eff.existing_user_multiplier)
    p = np.clip(users["p"] * (1 + treat * eff.lift_purchase * mult), 0, 1)
    basket = users["basket"] * (1 + treat * eff.lift_basket * mult)
    return p, basket


def draw_outcomes(
    users: dict, treat: np.ndarray, eff: EffectSpec, rng: np.random.Generator, daily: bool = False
) -> dict[str, np.ndarray]:
    """Draw the 14-day outcomes. ``daily=True`` also returns the per-day session panel."""
    treat = np.asarray(treat, bool)
    lam = users["lam"]
    f = _session_factors(users, treat, eff)
    out: dict[str, np.ndarray] = {}
    if daily:
        panel = rng.poisson(lam[:, None] * f)
        s_a, s_b = panel[:, :7].sum(1), panel[:, 7:].sum(1)
        out["sessions_early"] = panel[:, :3].sum(1).astype(float)
        out["panel"] = panel
    else:
        s_a = rng.poisson(lam * f[:, :7].sum(1))
        s_b = rng.poisson(lam * f[:, 7:].sum(1))
    out["sessions"] = (s_a + s_b).astype(float)
    out["sessions_late"] = s_b.astype(float)
    out["retained_w2"] = (s_b > 0).astype(float)
    p, basket = _purchase(users, treat, eff)
    purchases = rng.binomial(WINDOW_DAYS, p)
    out["purchases"] = purchases.astype(float)
    out["converted"] = (purchases > 0).astype(float)
    out["revenue"] = _sum_lognormal(purchases, basket, rng)
    return out


def expected_outcomes(users: dict, treated: bool, eff: EffectSpec) -> dict[str, np.ndarray]:
    """Per-user expected outcomes under control (False) or treatment (True): the ground truth."""
    t = np.full(len(users["lam"]), treated)
    f = _session_factors(users, t, eff)
    lam = users["lam"]
    p, basket = _purchase(users, t, eff)
    return {
        "sessions": lam * f.sum(1),
        "converted": 1 - (1 - p) ** WINDOW_DAYS,
        "revenue": WINDOW_DAYS * p * basket,
        "retained_w2": 1 - np.exp(-lam * f[:, 7:].sum(1)),
    }


def true_effects(users: dict, eff: EffectSpec) -> dict[str, dict[str, float]]:
    """Sample average treatment effect (absolute and relative) of every metric."""
    y0 = expected_outcomes(users, False, eff)
    y1 = expected_outcomes(users, True, eff)
    return {m: {"ate": float((y1[m] - y0[m]).mean()), "rel": float(y1[m].mean() / y0[m].mean() - 1)} for m in METRICS}


def generate_experiment(spec: ExperimentSpec) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Simulate one experiment. Returns (user table, per-exposure-day session curve, truth)."""
    rng = np.random.default_rng(spec.seed)
    n = spec.n_users
    users = draw_users(n, rng)
    arrival = rng.integers(0, spec.enrollment_days, n)
    treat = rng.random(n) < 0.5
    truth = true_effects(users, spec.effect)
    out = draw_outcomes(users, treat, spec.effect, rng, daily=True)
    keep = np.ones(n, bool)
    if spec.srm_drop_rate > 0:
        low = users["pre_sessions"] < np.quantile(users["pre_sessions"], 0.30)
        keep = ~(treat & low & (rng.random(n) < spec.srm_drop_rate))
    df = pd.DataFrame(
        {
            "exp_id": spec.exp_id,
            "user_id": np.arange(n),
            "arrival_day": arrival,
            "variant": np.where(treat, "treatment", "control"),
            "platform": PLATFORMS[users["platform"]],
            "tenure": np.where(users["is_new"], "new", "existing"),
            "pre_sessions": users["pre_sessions"],
            "pre_purchases": users["pre_purchases"],
            "pre_revenue": users["pre_revenue"].round(2),
            **{k: (v.round(2) if k == "revenue" else v) for k, v in out.items() if k != "panel"},
        }
    )
    panel = out["panel"][keep]
    t_keep = treat[keep]
    curve = pd.DataFrame(
        {
            "exp_id": spec.exp_id,
            "exposure_day": np.tile(np.arange(WINDOW_DAYS), 2),
            "variant": np.repeat(["control", "treatment"], WINDOW_DAYS),
            "mean_sessions": np.concatenate([panel[~t_keep].mean(0), panel[t_keep].mean(0)]),
            "sd_sessions": np.concatenate([panel[~t_keep].std(0, ddof=1), panel[t_keep].std(0, ddof=1)]),
            "n": np.repeat([(~t_keep).sum(), t_keep.sum()], WINDOW_DAYS),
        }
    )
    truth_out = {
        "effects": truth,
        "users_dropped_by_bug": int((~keep).sum()),
        "users_assigned": {"control": int((~treat).sum()), "treatment": int(treat.sum())},
    }
    return df[keep].reset_index(drop=True), curve, truth_out


def default_portfolio(scale: float = 1.0) -> list[ExperimentSpec]:
    """The six SYNTHETIC experiments analysed in the README (``scale`` shrinks n for tests)."""

    def n(x: int) -> int:
        return max(int(x * scale), 400)

    return [
        ExperimentSpec(
            "E01",
            "Onboarding checklist",
            "A guided checklist raises engagement, mostly for new users.",
            "sessions",
            n(60_000),
            21,
            EffectSpec(lift_sessions=0.04, new_user_multiplier=2.5, existing_user_multiplier=0.4),
            seed=101,
            mde_rel=0.02,
        ),
        ExperimentSpec(
            "E02",
            "Annual-plan upsell banner",
            "Showing the annual plan raises basket size.",
            "revenue",
            n(80_000),
            21,
            EffectSpec(lift_basket=0.06),
            seed=102,
            mde_rel=0.05,
        ),
        ExperimentSpec(
            "E03",
            "Button colour (A/A)",
            "No change: an A/A test that should find nothing.",
            "converted",
            n(60_000),
            21,
            EffectSpec(),
            seed=103,
            mde_rel=0.03,
        ),
        ExperimentSpec(
            "E04",
            "Feed redesign",
            "A new feed lifts sessions; much of it is novelty that fades.",
            "sessions",
            n(60_000),
            21,
            EffectSpec(lift_sessions=0.01, novelty_sessions=0.25, novelty_decay_days=2.5),
            seed=104,
            mde_rel=0.02,
        ),
        ExperimentSpec(
            "E05",
            "Checkout v2 (logging bug)",
            "Checkout v2 has no real effect; a tracking bug drops low-engagement treatment users.",
            "converted",
            n(60_000),
            21,
            EffectSpec(),
            srm_drop_rate=0.20,
            seed=105,
            mde_rel=0.03,
        ),
        ExperimentSpec(
            "E06",
            "Trial-reminder email",
            "A reminder email gives a small lift to purchase probability.",
            "converted",
            n(120_000),
            28,
            EffectSpec(lift_purchase=0.04),
            seed=106,
            mde_rel=0.03,
        ),
    ]
