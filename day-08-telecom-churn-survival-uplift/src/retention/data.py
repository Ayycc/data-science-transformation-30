"""SYNTHETIC subscriber panel with churn, censoring and a randomised retention campaign.

Everything here is simulated from a seeded generator. The data-generating process is a
discrete-time (monthly) logistic hazard model with contract-specific baseline hazards
(which deliberately breaks proportional hazards for ``contract``), three competing churn
causes, independent random dropout, administrative censoring, and a 50/50 randomised
retention offer at calendar month 30 whose effect on the hazard differs by customer.
Because the process is known, the true uplift and the true value of treating every
campaign customer can be computed exactly (``campaign_truth``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CONTRACTS = ["month_to_month", "one_year", "two_year"]
REGIONS = ["urban", "suburban", "small_town", "rural"]
CAUSES = ["voluntary", "port_out", "non_payment"]
FEATURES = [
    "contract_one_year",
    "contract_two_year",
    "fiber",
    "monthly_charges",
    "log_data_gb",
    "dropped_call_rate",
    "support_tickets",
    "autopay",
    "senior",
    "region_suburban",
    "region_small_town",
    "region_rural",
]


@dataclass(frozen=True)
class GenConfig:
    n_customers: int = 200_000
    n_months: int = 36
    max_start_month: int = 29
    dropout_monthly: float = 0.004
    campaign_month: int = 30
    campaign_window: int = 6
    treat_share: float = 0.5
    seed: int = 20261018


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def baseline_logit(tenure: np.ndarray, contract: np.ndarray) -> np.ndarray:
    """Monthly baseline churn logit for tenure month ``tenure`` (1-based).

    Month-to-month hazard is high early and decays; one- and two-year contracts are low
    with spikes when the term ends (months 12-13 and 24-25). Non-proportional by design.
    """
    t = np.asarray(tenure, dtype=float)
    c = np.asarray(contract)
    m2m = -3.3 + 1.0 * np.exp(-(t - 1.0) / 5.0)
    one = -5.2 + 2.6 * np.isin(t, (12, 13))
    two = -5.8 + 2.9 * np.isin(t, (24, 25))
    return np.where(c == 0, m2m, np.where(c == 1, one, two))


def _features(n: int, rng: np.random.Generator) -> pd.DataFrame:
    contract = rng.choice(3, size=n, p=[0.55, 0.25, 0.20])
    fiber = rng.random(n) < 0.45
    region = rng.choice(4, size=n, p=[0.35, 0.30, 0.20, 0.15])
    charges = np.clip(35 + 40 * fiber + 6 * contract + rng.normal(0, 12, n), 18, 140)
    data_gb = np.exp(rng.normal(2.3 + 0.4 * fiber, 0.6, n))
    dropped = np.clip(rng.gamma(2.0, 0.9, n) + 0.9 * (region == 3), 0, 12)
    tickets = rng.poisson(0.6 + 0.35 * dropped)
    autopay = rng.random(n) < (0.30 + 0.25 * (contract > 0))
    senior = rng.random(n) < 0.16
    return pd.DataFrame(
        {
            "customer_id": np.arange(n, dtype=np.int64),
            "contract": contract,
            "fiber": fiber.astype(np.int8),
            "monthly_charges": charges.round(2),
            "data_gb": data_gb.round(2),
            "dropped_call_rate": dropped.round(3),
            "support_tickets": tickets.astype(np.int16),
            "autopay": autopay.astype(np.int8),
            "senior": senior.astype(np.int8),
            "region": region,
        }
    )


def risk_logit(df: pd.DataFrame) -> np.ndarray:
    """Time-invariant covariate part of the churn logit (the 'true' risk score)."""
    region_eff = np.array([0.0, 0.05, 0.10, 0.15])
    return (
        0.015 * (df["monthly_charges"].to_numpy() - 70)
        + 0.22 * (df["dropped_call_rate"].to_numpy() - 1.8)
        + 0.15 * (df["support_tickets"].to_numpy() - 1.2)
        - 0.40 * df["autopay"].to_numpy()
        + 0.20 * df["senior"].to_numpy()
        + 0.15 * df["fiber"].to_numpy()
        - 0.15 * (np.log(df["data_gb"].to_numpy()) - 2.5)
        + region_eff[df["region"].to_numpy()]
    )


def treatment_logit_shift(df: pd.DataFrame, tenure_at_campaign: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """True heterogeneous effect of the retention offer on the monthly churn logit.

    Negative = offer lowers churn. Price-sensitive customers respond (persuadables); customers
    whose problem is the network barely respond (lost causes); quiet long-tenure / two-year
    customers with no tickets are nudged to shop around (sleeping dogs).
    """
    ch = df["monthly_charges"].to_numpy()
    c = df["contract"].to_numpy()
    ps = _sigmoid(0.06 * (ch - 70) + 0.5 * (c == 0) - 0.5 * df["autopay"].to_numpy() + 0.3 * df["senior"].to_numpy() + noise)
    # network / service problems: a discount does not fix dropped calls
    lost = (df["dropped_call_rate"].to_numpy() > 3.5) | (df["support_tickets"].to_numpy() >= 3)
    sleeping = ((c == 2) | (tenure_at_campaign >= 20)) & (df["support_tickets"].to_numpy() == 0)
    return -1.6 * ps * (1.0 - 0.9 * lost) + 0.9 * sleeping


def generate(cfg: GenConfig, economics: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (panel, campaign_truth). ``panel`` has one row per customer.

    ``economics`` (optional) overrides the keyword defaults of :func:`campaign_truth`.
    """
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_customers
    df = _features(n, rng)
    df.insert(1, "start_month", rng.integers(0, cfg.max_start_month + 1, n))
    lin = risk_logit(df)
    contract = df["contract"].to_numpy()
    start = df["start_month"].to_numpy()

    # cause-of-churn logits (relative to voluntary)
    ch = df["monthly_charges"].to_numpy()
    cause_logits = np.column_stack(
        [
            np.zeros(n),
            -0.3 + 0.35 * (df["dropped_call_rate"].to_numpy() - 1.8) + 0.01 * (ch - 70),
            -1.0 - 1.5 * df["autopay"].to_numpy() + 0.012 * (ch - 70),
        ]
    )
    cause_p = np.exp(cause_logits)
    cause_p /= cause_p.sum(1, keepdims=True)

    tenure_at_campaign = np.maximum(cfg.campaign_month - start, 0)
    effect_noise = rng.normal(0, 0.8, n)
    delta = treatment_logit_shift(df, tenure_at_campaign, effect_noise)

    alive = np.ones(n, dtype=bool)
    duration = np.zeros(n, dtype=np.int16)
    event = np.zeros(n, dtype=np.int8)
    cause = np.full(n, -1, dtype=np.int8)
    in_campaign = np.zeros(n, dtype=bool)
    treat = np.zeros(n, dtype=np.int8)
    churn_window = np.zeros(n, dtype=np.int8)

    for m in range(cfg.n_months):
        if m == cfg.campaign_month:
            in_campaign = alive & (start < m)
            treat = (in_campaign & (rng.random(n) < cfg.treat_share)).astype(np.int8)
        at_risk = alive & (start <= m)
        tenure = m - start + 1
        logit = baseline_logit(tenure, contract) + lin
        in_window = cfg.campaign_month <= m < cfg.campaign_month + cfg.campaign_window
        if in_window:
            logit = logit + treat * delta
        u = rng.random(n)
        churn = at_risk & (u < _sigmoid(logit))
        duration[at_risk] = tenure[at_risk]
        event[churn] = 1
        alive &= ~churn
        if churn.any():
            cu = rng.random(n)
            cc = (cu[:, None] > np.cumsum(cause_p, 1)).sum(1).astype(np.int8)
            cause[churn] = cc[churn]
        if in_window:
            churn_window[churn & in_campaign] = 1
        if m < cfg.campaign_month:  # dropout only before the campaign: campaign outcomes are complete
            drop = alive & (start <= m) & (rng.random(n) < cfg.dropout_monthly)
            alive &= ~drop

    df["duration"] = duration
    df["event"] = event
    df["cause"] = np.where(cause >= 0, np.array(CAUSES, dtype=object)[np.maximum(cause, 0)], "censored")
    df["tenure_at_campaign"] = np.where(in_campaign, tenure_at_campaign, -1).astype(np.int16)
    df["in_campaign"] = in_campaign.astype(np.int8)
    df["treatment"] = treat
    df["churn_6m"] = np.where(in_campaign, churn_window, -1).astype(np.int8)
    df["data_source"] = "SYNTHETIC"

    truth = campaign_truth(df[df["in_campaign"] == 1], delta[in_campaign], lin[in_campaign], cfg, **(economics or {}))
    return df, truth


def campaign_truth(
    camp: pd.DataFrame,
    delta: np.ndarray,
    lin: np.ndarray,
    cfg: GenConfig,
    margin_rate: float = 0.55,
    monthly_discount_rate: float = 0.01,
    horizon: int = 36,
    contact_cost: float = 2.0,
    discount_share: float = 0.20,
    discount_months: int = 3,
) -> pd.DataFrame:
    """Exact ground truth for each campaign customer from the known hazard model."""
    t0 = camp["tenure_at_campaign"].to_numpy()
    c = camp["contract"].to_numpy()
    k = np.arange(1, horizon + 1)
    ten = t0[:, None] + k[None, :]  # tenure month being survived
    base = baseline_logit(ten, np.repeat(c[:, None], horizon, 1)) + lin[:, None]
    shift = np.where(k[None, :] <= cfg.campaign_window, delta[:, None], 0.0)
    s0 = np.cumprod(1 - _sigmoid(base), 1)
    s1 = np.cumprod(1 - _sigmoid(base + shift), 1)
    w = cfg.campaign_window
    p0, p1 = 1 - s0[:, w - 1], 1 - s1[:, w - 1]
    ch = camp["monthly_charges"].to_numpy()
    disc = (1 + monthly_discount_rate) ** -k
    margin = margin_rate * ch
    clv0 = (s0 * disc).sum(1) * margin
    dv = ((s1 - s0) * disc).sum(1) * margin
    cost = contact_cost + discount_share * ch * (s1[:, :discount_months] * disc[:discount_months]).sum(1)
    return pd.DataFrame(
        {
            "customer_id": camp["customer_id"].to_numpy(),
            "p_churn0": p0,
            "p_churn1": p1,
            "tau_true": p0 - p1,  # true 6-month RETENTION uplift (positive = offer helps)
            "clv0_true": clv0,
            "dv_true": dv,
            "cost_true": cost,
            "net_true": dv - cost,
            "delta_logit": delta,
        }
    )


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Model features (one-hot, numeric) used by every survival and uplift model."""
    out = pd.DataFrame(index=df.index)
    out["contract_one_year"] = (df["contract"] == 1).astype(float)
    out["contract_two_year"] = (df["contract"] == 2).astype(float)
    out["fiber"] = df["fiber"].astype(float)
    out["monthly_charges"] = df["monthly_charges"].astype(float)
    out["log_data_gb"] = np.log(df["data_gb"].astype(float))
    out["dropped_call_rate"] = df["dropped_call_rate"].astype(float)
    out["support_tickets"] = df["support_tickets"].astype(float)
    out["autopay"] = df["autopay"].astype(float)
    out["senior"] = df["senior"].astype(float)
    out["region_suburban"] = (df["region"] == 1).astype(float)
    out["region_small_town"] = (df["region"] == 2).astype(float)
    out["region_rural"] = (df["region"] == 3).astype(float)
    return out[FEATURES]
