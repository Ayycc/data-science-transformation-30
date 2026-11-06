"""SYNTHETIC hotel market simulator.

One hotel with ``capacity`` rooms per night. Each stay night is sold over three booking
windows (lead-time buckets): 22-60 days out, 4-21 days out and 0-3 days out. In every
bucket the hotel posts one price; shoppers from two segments (business, leisure) arrive
as a Poisson stream, and each books if their willingness-to-pay (WTP) is at least the
posted price. Rooms that are unsold when the night arrives perish.

WTP is lognormal and depends on lead time, season, day of week, events, the competitor's
price and segment. An unobserved nightly demand shock ``u`` raises both the number of
shoppers and their WTP. The historical revenue manager saw ``u`` (group enquiries, phone
calls) and priced on it, which is what confounds naive elasticity estimates.

Everything here is SYNTHETIC and seeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

BUCKETS = ("22-60d", "4-21d", "0-3d")
SEGMENTS = ("business", "leisure")
REF_COMP = 150.0
# Gauss-Hermite (probabilists') nodes to integrate the unobserved shock u ~ N(0, 1)
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(15)
_GH_W = _GH_W / np.sqrt(2 * np.pi)


@dataclass
class MarketConfig:
    capacity: int = 120
    # mean shoppers per night by segment x bucket on a weekday at mid season
    shoppers_weekday: dict = field(default_factory=lambda: {"business": (10.0, 26.0, 34.0), "leisure": (60.0, 42.0, 14.0)})
    weekend_mult: dict = field(default_factory=lambda: {"business": 0.35, "leisure": 1.6})
    season_amp_shoppers: dict = field(default_factory=lambda: {"business": 0.10, "leisure": 0.45})
    # lognormal WTP: log-median by segment x bucket, and dispersion
    wtp_median: dict = field(default_factory=lambda: {"business": (215.0, 230.0, 250.0), "leisure": (150.0, 140.0, 125.0)})
    wtp_sigma: dict = field(default_factory=lambda: {"business": 0.38, "leisure": 0.28})
    season_amp_wtp: float = 0.16
    comp_wtp_coef: float = 0.35  # log-WTP response to log(competitor / reference)
    event_prob: float = 0.06
    event_shopper_mult: float = 1.8
    event_wtp_shift: float = 0.22
    shock_shopper_coef: float = 0.22  # unobserved u
    shock_wtp_coef: float = 0.07
    comp_noise: float = 0.07
    comp_reaction: float = 0.0  # stretch: competitor partly follows our previous-night price
    start_weekday: int = 0  # day 0 is a Monday


def season_wave(day: np.ndarray | int) -> np.ndarray:
    """Seasonal wave in [-1, 1], peaking in mid July (day of year 196)."""
    doy = np.asarray(day) % 365
    return np.sin(2 * np.pi * (doy - 105) / 365.0)


def is_weekend(day: np.ndarray | int, cfg: MarketConfig) -> np.ndarray:
    """Friday and Saturday nights."""
    dow = (np.asarray(day) + cfg.start_weekday) % 7
    return np.isin(dow, (4, 5)).astype(float)


def static_rate_table(day: np.ndarray | int, cfg: MarketConfig) -> np.ndarray:
    """The incumbent static seasonal table: a rate per month (rounded to $5) plus a 10% weekend uplift."""
    doy = np.asarray(day) % 365
    month = np.minimum(doy // 31, 11)
    mid = month * 31 + 15
    base = np.round(150.0 * (1 + 0.22 * season_wave(mid)) / 5.0) * 5.0
    return base * (1 + 0.10 * is_weekend(day, cfg))


@dataclass
class World:
    """Exogenous state for a run of nights: calendar, events, competitor prices, hidden shocks."""

    days: np.ndarray
    event: np.ndarray
    comp_base: np.ndarray
    comp_noise: np.ndarray
    u: np.ndarray
    cfg: MarketConfig

    @property
    def n(self) -> int:
        return len(self.days)


def make_world(first_day: int, n_nights: int, seed: int, cfg: MarketConfig) -> World:
    rng = np.random.default_rng(seed)
    days = np.arange(first_day, first_day + n_nights)
    event = (rng.random(n_nights) < cfg.event_prob).astype(float)
    comp_base = 155.0 * (1 + 0.20 * season_wave(days)) * (1 + 0.25 * event) * (1 + 0.08 * is_weekend(days, cfg))
    noise = rng.normal(0, cfg.comp_noise, n_nights)
    u = rng.normal(0, 1, n_nights)
    return World(days, event, comp_base, noise, u, cfg)


def competitor_price(world: World, i: int, our_prev_ratio: float = 1.0) -> float:
    """Competitor's posted price for night i. With comp_reaction > 0 it partly follows our last price move."""
    react = our_prev_ratio**world.cfg.comp_reaction if world.cfg.comp_reaction else 1.0
    return float(world.comp_base[i] * np.exp(world.comp_noise[i]) * react)


@dataclass
class Context:
    """Everything the hotel observes when it prices night ``day`` in lead bucket ``bucket``."""

    day: int
    bucket: int
    weekend: float
    event: float
    comp: float
    static: float
    pace: float = 0.0  # log(1 + unconstrained demand already seen for this night in earlier windows)
    u_weights: np.ndarray | None = field(default=None, repr=False, compare=False)  # posterior over u (oracle / regret only)

    def features(self) -> np.ndarray:
        """Bandit controls x (without price). Booking pace proxies the night's hidden demand shock."""
        s = float(season_wave(self.day))
        c = float(np.cos(2 * np.pi * ((self.day % 365) - 105) / 365.0))
        return np.array([1.0, s, c, self.weekend, self.event, np.log(self.comp / REF_COMP), self.pace])


def context_for(world: World, i: int, bucket: int, comp: float) -> Context:
    d = int(world.days[i])
    return Context(d, bucket, float(is_weekend(d, world.cfg)), float(world.event[i]), comp, float(static_rate_table(d, world.cfg)))


def _segment_params(ctx: Context, cfg: MarketConfig, seg: str) -> tuple[float, float, float]:
    """(mean shoppers, log-median WTP, sigma) at u = 0."""
    s = float(season_wave(ctx.day))
    lam = cfg.shoppers_weekday[seg][ctx.bucket]
    lam *= cfg.weekend_mult[seg] if ctx.weekend else 1.0
    lam *= 1 + cfg.season_amp_shoppers[seg] * s
    lam *= cfg.event_shopper_mult if ctx.event else 1.0
    mu = np.log(cfg.wtp_median[seg][ctx.bucket]) + cfg.season_amp_wtp * s + cfg.comp_wtp_coef * np.log(ctx.comp / REF_COMP)
    mu += cfg.event_wtp_shift * ctx.event
    return lam, mu, cfg.wtp_sigma[seg]


def true_demand_given_u(ctx: Context, prices: np.ndarray, u: np.ndarray | float, cfg: MarketConfig) -> np.ndarray:
    """Expected unconstrained bookings (shoppers with WTP >= price) given the hidden shock u.

    Broadcasting: prices (..., P) and u (..., 1) or scalar.
    """
    prices = np.asarray(prices, dtype=float)
    out = 0.0
    for seg in SEGMENTS:
        lam, mu, sig = _segment_params(ctx, cfg, seg)
        lam_u = lam * np.exp(cfg.shock_shopper_coef * u - 0.5 * cfg.shock_shopper_coef**2)
        mu_u = mu + cfg.shock_wtp_coef * u
        out = out + lam_u * (1 - ndtr((np.log(prices) - mu_u) / sig))
    return out


def shock_posterior(history: list, cfg: MarketConfig) -> np.ndarray:
    """Posterior weights over the quadrature nodes of u given this night's earlier (ctx, price, demand)."""
    logw = np.log(_GH_W)
    for ctx, price, d in history:
        lam = true_demand_given_u(ctx, np.array([price]), _GH_X[:, None], cfg)[:, 0]
        logw = logw + d * np.log(np.maximum(lam, 1e-12)) - lam
    w = np.exp(logw - logw.max())
    return w / w.sum()


def true_expected_demand(ctx: Context, prices: np.ndarray, cfg: MarketConfig, weights: np.ndarray | None = None) -> np.ndarray:
    """Expected demand given only what the hotel observes: u integrated out under its prior, or under the
    posterior ``weights`` from :func:`shock_posterior` when earlier windows of the night have been seen."""
    prices = np.atleast_1d(np.asarray(prices, dtype=float))
    lam = true_demand_given_u(ctx, prices[None, :], _GH_X[:, None], cfg)
    return (_GH_W if weights is None else weights) @ lam


def true_semi_elasticity(ctx: Context, price: float, cfg: MarketConfig, h: float = 1e-3) -> float:
    """d log E[demand] / d m at multiplier m = price / static (u integrated out)."""
    m = price / ctx.static
    p = ctx.static * np.array([m - h, m + h])
    d = true_expected_demand(ctx, p, cfg)
    return float((np.log(d[1]) - np.log(d[0])) / (2 * h))


def sample_demand(ctx: Context, price: float, u: float, cfg: MarketConfig, rng: np.random.Generator) -> int:
    """Realised unconstrained demand (bookings + denials recorded by the booking engine)."""
    lam = float(true_demand_given_u(ctx, np.array([price]), u, cfg)[0])
    return int(rng.poisson(lam))
