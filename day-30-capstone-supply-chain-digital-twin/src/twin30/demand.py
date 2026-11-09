"""Bayesian demand and promo-effect model (PyMC), fitted on the observable sales history.

log mu[s,c,t] = a[s,c] + weekday[dow] + Fourier_c(doy) + trend_c * years + promo_c * promo + post_c * post
units ~ NegativeBinomial(mu, alpha_c), fitted on store-category-days that did NOT stock out
(on stockout days sales are censored demand, so they are dropped rather than treated as demand).

The promo coefficients are the causal piece: promo weeks were chosen by the merchandisers in
high-season weeks, so a naive promo-vs-non-promo comparison is confounded by seasonality. The model
adjusts for seasonality, weekday, trend and the post-promo dip; the SYNTHETIC truth lets us check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CATEGORIES, DAYS_PER_YEAR, HISTORY_START, N_CATS, N_STORES, STORES

K_FOURIER = 2


def design(day0: int, n_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns dow (T,), Fourier features (T, 2K) and trend in years (T,) for days day0..day0+n."""
    dates = pd.date_range(HISTORY_START, periods=day0 + n_days, freq="D")[day0:]
    doy = dates.dayofyear.to_numpy()
    w = 2 * np.pi / 365.25
    four = np.column_stack([f(k * w * doy) for k in range(1, K_FOURIER + 1) for f in (np.cos, np.sin)])
    years = (day0 + np.arange(n_days)) / DAYS_PER_YEAR
    return dates.dayofweek.to_numpy(), four, years


@dataclass
class Posterior:
    a: np.ndarray  # (N, S, C)
    weekday: np.ndarray  # (N, 7)
    fourier: np.ndarray  # (N, C, 2K)
    trend: np.ndarray  # (N, C)
    promo: np.ndarray  # (N, C) log lift
    post: np.ndarray  # (N, C) log post-promo effect
    alpha: np.ndarray  # (N, C) NB dispersion

    @property
    def n(self) -> int:
        return self.a.shape[0]

    def log_mu(self, i: int, day0: int, n_days: int, promo: np.ndarray, post: np.ndarray) -> np.ndarray:
        dow, four, years = design(day0, n_days)
        season = four @ self.fourier[i].T  # (T, C)
        lm = (
            self.a[i][None, :, :]
            + self.weekday[i][dow][:, None, None]
            + (season + self.trend[i][None, :] * years[:, None] + promo * self.promo[i] + post * self.post[i])[:, None, :]
        )
        return lm

    def mean(self, i: int, day0: int, n_days: int, promo: np.ndarray, post: np.ndarray) -> np.ndarray:
        return np.exp(self.log_mu(i, day0, n_days, promo, post))

    def sample_demand(self, i: int, mu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        alpha = self.alpha[i][None, None, :]
        lam = rng.gamma(alpha, mu / alpha)
        return rng.poisson(lam).astype(float)

    def to_npz(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in ("a", "weekday", "fourier", "trend", "promo", "post", "alpha")}

    @classmethod
    def from_npz(cls, d) -> Posterior:
        return cls(**{k: np.asarray(d[k]) for k in ("a", "weekday", "fourier", "trend", "promo", "post", "alpha")})


def training_frame(sales: pd.DataFrame, day_from: int, day_to: int) -> pd.DataFrame:
    s = sales[(sales.day >= day_from) & (sales.day < day_to) & (sales.stockout == 0)].copy()
    s["si"] = s.store.map({st.name: i for i, st in enumerate(STORES)})
    s["ci"] = s.category.map({c.name: i for i, c in enumerate(CATEGORIES)})
    return s


def fit(
    sales: pd.DataFrame,
    day_from: int,
    day_to: int,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    seed: int = 1,
    max_rows: int | None = None,
    progressbar: bool = False,
) -> tuple[Posterior, dict]:
    """Fit the NB demand model with NUTS. Returns the posterior and convergence diagnostics."""
    import arviz as az
    import pymc as pm

    df = training_frame(sales, day_from, day_to)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).sort_values(["day", "si", "ci"])
    dow_all, four_all, years_all = design(day_from, day_to - day_from)
    t = (df.day - day_from).to_numpy()
    si, ci = df.si.to_numpy(), df.ci.to_numpy()
    y = df.units_sold.to_numpy()
    dow, four, years = dow_all[t], four_all[t], years_all[t]
    promo, post = df.promo.to_numpy().astype(float), df.post_promo.to_numpy().astype(float)
    cat_mean = np.log(df.groupby("ci").units_sold.mean().reindex(range(N_CATS)).to_numpy() + 1.0)

    yc = float(years_all.mean())
    with pm.Model():
        # centred parameterisation: every store-category stream has hundreds of observations
        a = pm.Normal("a", mu=np.tile(cat_mean, (N_STORES, 1)), sigma=1.0, shape=(N_STORES, N_CATS))
        wd = pm.ZeroSumNormal("weekday", sigma=0.3, shape=7)
        fb = pm.Normal("fourier", 0.0, 0.5, shape=(N_CATS, 2 * K_FOURIER))
        tr = pm.Normal("trend", 0.0, 0.1, shape=N_CATS)
        pr = pm.Normal("promo", 0.0, 0.5, shape=N_CATS)
        po = pm.Normal("post", 0.0, 0.3, shape=N_CATS)
        alpha = pm.LogNormal("alpha", mu=np.log(50.0), sigma=1.5, shape=N_CATS)
        season = (four * fb[ci]).sum(axis=1)
        lm = a[si, ci] + wd[dow] + season + tr[ci] * (years - yc) + pr[ci] * promo + po[ci] * post
        pm.NegativeBinomial("y", mu=pm.math.exp(lm), alpha=alpha[ci], observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1, random_seed=seed, progressbar=progressbar, target_accept=0.85)
    names = ["a", "weekday", "fourier", "trend", "promo", "post", "alpha"]
    summ = az.summary(idata, var_names=names, kind="diagnostics")
    diag = {
        "mean_tree_depth": float(idata.sample_stats["tree_depth"].mean()),
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum()),
        "n_obs": int(len(y)),
        "draws": draws,
        "tune": tune,
        "chains": chains,
    }
    p = idata.posterior.stack(sample=("chain", "draw"))

    def arr(name: str) -> np.ndarray:
        v = p[name].values
        return np.moveaxis(v, -1, 0)

    trend_ = arr("trend")
    post_ = Posterior(
        a=arr("a") - (trend_ * yc)[:, None, :],
        weekday=arr("weekday"),
        fourier=arr("fourier"),
        trend=trend_,
        promo=arr("promo"),
        post=arr("post"),
        alpha=arr("alpha"),
    )
    return post_, diag


def naive_promo_lift(sales: pd.DataFrame, day_from: int, day_to: int) -> np.ndarray:
    """Promo-week mean sales / non-promo-week mean sales per category (no adjustment)."""
    df = training_frame(sales, day_from, day_to)
    g = df.groupby(["ci", "promo"]).units_sold.mean().unstack()
    return (g[1] / g[0]).reindex(range(N_CATS)).to_numpy()


def hdi(x: np.ndarray, prob: float = 0.9) -> tuple[float, float]:
    xs = np.sort(np.asarray(x))
    n = len(xs)
    k = int(np.floor(prob * n))
    widths = xs[k:] - xs[: n - k]
    j = int(np.argmin(widths))
    return float(xs[j]), float(xs[j + k])
