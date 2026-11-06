"""Historical logs and price-elasticity estimation with confounding addressed.

The logged year was priced by a revenue manager who raised rates on nights with a strong
(unobserved to the analyst) demand shock u, so price is correlated with the demand error and
a naive regression of demand on price is biased towards zero (or even positive slope).
On a random 40% of (night, window) cells a randomised price test overrode the manager with a
multiplier z drawn uniformly from {0.80, 0.90, 1.00, 1.10, 1.20} of the static rate.

Estimators of the semi-log demand model  log(D + 0.5) = x'beta + theta * m,  m = price / static:
  * naive OLS on all cells (confounded);
  * OLS on randomised test cells only (unbiased, fewer rows);
  * 2SLS on all cells with the randomised multiplier z as the instrument for m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .simulator import BUCKETS, MarketConfig, competitor_price, context_for, make_world, sample_demand, true_semi_elasticity

TEST_SHARE = 0.40
TEST_MULTS = np.array([0.80, 0.90, 1.00, 1.10, 1.20])
RM_SHOCK_COEF = 0.12


def generate_logs(cfg: MarketConfig, seed: int = 20261106, n_nights: int = 365) -> pd.DataFrame:
    """One SYNTHETIC year of booking logs, one row per (stay night, booking window)."""
    world = make_world(0, n_nights, seed, cfg)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for i in range(n_nights):
        comp = competitor_price(world, i)
        R = cfg.capacity
        seen = 0
        for k in range(len(BUCKETS)):
            ctx = context_for(world, i, k, comp)
            test = rng.random() < TEST_SHARE
            z = float(rng.choice(TEST_MULTS)) if test else 1.0
            m = z if test else float(np.exp(RM_SHOCK_COEF * world.u[i]))
            price = round(ctx.static * m, 2)
            demand = sample_demand(ctx, price, float(world.u[i]), cfg, rng)
            sold = min(demand, R)
            rows.append(
                dict(
                    night=int(world.days[i]),
                    bucket=k,
                    window=BUCKETS[k],
                    weekend=int(ctx.weekend),
                    event=int(ctx.event),
                    comp_price=round(comp, 2),
                    static_price=ctx.static,
                    price=price,
                    price_test=int(test),
                    test_multiplier=z,
                    demand_unconstrained=demand,
                    rooms_left_before=R,
                    pace=round(float(np.log1p(seen)), 6),
                    sold=sold,
                    true_semi_elasticity=true_semi_elasticity(ctx, ctx.static, cfg),
                )
            )
            R -= sold
            seen += demand
    return pd.DataFrame(rows)


def control_arrays(night, bucket, weekend, event, comp, test) -> np.ndarray:
    d = np.asarray(night) % 365
    s = np.sin(2 * np.pi * (d - 105) / 365.0)
    c = np.cos(2 * np.pi * (d - 105) / 365.0)
    wk = np.asarray(weekend, dtype=float)
    b1 = (np.asarray(bucket) == 1).astype(float)
    b2 = (np.asarray(bucket) == 2).astype(float)
    logc = np.log(np.asarray(comp, dtype=float) / 150.0)
    one = np.ones_like(s, dtype=float)
    return np.column_stack(
        [one, s, c, wk, np.asarray(event, dtype=float), logc, b1, b2, wk * b1, wk * b2, s * b1, s * b2, np.asarray(test, dtype=float) * one]
    )


def control_matrix(df: pd.DataFrame) -> np.ndarray:
    return control_arrays(df["night"].to_numpy(), df["bucket"].to_numpy(), df["weekend"], df["event"], df["comp_price"], df["price_test"])


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = (X * resid[:, None] ** 2).T @ X
    n, k = X.shape
    cov = XtX_inv @ meat @ XtX_inv * n / (n - k)  # HC1
    return beta, np.sqrt(np.diag(cov))


def two_sls(y: np.ndarray, endog: np.ndarray, exog: np.ndarray, instr: np.ndarray) -> dict:
    """Just-identified 2SLS with HC1 standard errors and first-stage F."""
    Z = np.column_stack([exog, instr])
    X = np.column_stack([exog, endog])
    pi, *_ = np.linalg.lstsq(Z, endog, rcond=None)
    # first-stage partial F for the instrument
    fs_r = endog - Z @ pi
    pi_r, *_ = np.linalg.lstsq(exog, endog, rcond=None)
    rss_r, rss_u = np.sum((endog - exog @ pi_r) ** 2), np.sum(fs_r**2)
    n = len(y)
    f_stat = (rss_r - rss_u) / (rss_u / (n - Z.shape[1]))
    Xhat = np.column_stack([exog, Z @ pi])
    beta, *_ = np.linalg.lstsq(Xhat, y, rcond=None)
    resid = y - X @ beta
    bread = np.linalg.inv(Xhat.T @ Xhat)
    meat = (Xhat * resid[:, None] ** 2).T @ Xhat
    cov = bread @ meat @ bread * n / (n - X.shape[1])
    return dict(beta=beta, se=np.sqrt(np.diag(cov)), first_stage_F=float(f_stat))


@dataclass
class SemiLogDemand:
    """log E[D] = x'beta + theta*m (with smearing); used by the greedy optimiser and the RM layer."""

    beta: np.ndarray
    theta: float
    smear: float
    name: str

    def predict(self, ctx, prices: np.ndarray) -> np.ndarray:
        xb = float(control_arrays([ctx.day], [ctx.bucket], [ctx.weekend], [ctx.event], [ctx.comp], [0.0])[0] @ self.beta)
        m = np.asarray(prices, dtype=float) / ctx.static
        return np.maximum(np.exp(xb + self.theta * m) * self.smear - 0.5, 0.0)


def estimate_elasticities(logs: pd.DataFrame) -> dict:
    y = np.log(logs["demand_unconstrained"].to_numpy() + 0.5)
    m = (logs["price"] / logs["static_price"]).to_numpy()
    X = control_matrix(logs)
    out: dict = {}
    b, se = _ols(np.column_stack([X, m]), y)
    out["naive_ols"] = dict(theta=float(b[-1]), se=float(se[-1]), n=len(y), beta=b[:-1])
    t = logs["price_test"].to_numpy() == 1
    Xt = np.delete(X[t], -1, axis=1)  # price_test constant on this subset
    bt, set_ = _ols(np.column_stack([Xt, m[t]]), y[t])
    out["randomised_test_ols"] = dict(theta=float(bt[-1]), se=float(set_[-1]), n=int(t.sum()))
    z = logs["test_multiplier"].to_numpy()
    iv = two_sls(y, m, X, z)
    out["iv_2sls"] = dict(theta=float(iv["beta"][-1]), se=float(iv["se"][-1]), n=len(y), beta=iv["beta"][:-1], first_stage_F=iv["first_stage_F"])
    truth = logs["true_semi_elasticity"].to_numpy()
    out["truth"] = dict(theta_mean=float(truth.mean()), theta_p10=float(np.quantile(truth, 0.1)), theta_p90=float(np.quantile(truth, 0.9)))
    for key in ("naive_ols", "iv_2sls"):
        beta = out[key]["beta"]
        resid = y - np.column_stack([X, m]) @ np.append(beta, out[key]["theta"])
        out[key]["smear"] = float(np.mean(np.exp(resid)))
    return out


def demand_models(est: dict) -> dict[str, SemiLogDemand]:
    return {k: SemiLogDemand(est[k]["beta"], est[k]["theta"], est[k]["smear"], k) for k in ("naive_ols", "iv_2sls")}
