"""Diagnostics, prior/posterior predictive checks, parameter recovery and ROAS summaries."""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd

from .config import CHANNELS, HDI_PROB
from .data import Panel, logistic_saturation_np
from .model import ModelInputs


def hdi(samples: np.ndarray, prob: float = HDI_PROB) -> tuple[float, float]:
    """Highest-density interval of a 1-D sample."""
    lo, hi = az.hdi(np.asarray(samples, dtype=float).reshape(-1), hdi_prob=prob)
    return float(lo), float(hi)


def flat(idata, var: str, group: str = "posterior") -> np.ndarray:
    """Draws stacked over (chain, draw) -> leading axis."""
    da = idata[group][var]
    return da.stack(sample=("chain", "draw")).transpose("sample", ...).to_numpy()


def convergence(idata) -> dict:
    post = idata.posterior
    rhat = az.rhat(post)
    ess_b = az.ess(post, method="bulk")
    ess_t = az.ess(post, method="tail")
    per_var = []
    for v in post.data_vars:
        per_var.append(
            {
                "variable": v,
                "size": int(np.prod(post[v].shape[2:])) if post[v].ndim > 2 else 1,
                "max_rhat": float(rhat[v].max()),
                "min_ess_bulk": float(ess_b[v].min()),
                "min_ess_tail": float(ess_t[v].min()),
            }
        )
    table = pd.DataFrame(per_var)
    n_div = int(idata.sample_stats["diverging"].sum())
    n_draws = int(post.sizes["chain"] * post.sizes["draw"])
    return {
        "table": table,
        "max_rhat": float(table["max_rhat"].max()),
        "n_quantities": int(table["size"].sum()),
        "n_rhat_ge_1_01": int(sum(int((rhat[v] >= 1.01).sum()) for v in post.data_vars)),
        "min_ess_bulk": float(table["min_ess_bulk"].min()),
        "min_ess_tail": float(table["min_ess_tail"].min()),
        "divergences": n_div,
        "n_draws": n_draws,
        "mean_tree_depth": float(idata.sample_stats["tree_depth"].mean()),
    }


def to_sales_k(y_scaled: np.ndarray, inp: ModelInputs) -> np.ndarray:
    """(..., T, R) scaled per-capita -> $k sales."""
    return y_scaled * inp.y_scale * inp.population


def predictive_check(y_draws: np.ndarray, panel: Panel, inp: ModelInputs) -> dict:
    """y_draws: (S, T, R) in scaled units. Coverage, R^2 and MAPE on region-week sales ($k)."""
    sales_draws = to_sales_k(y_draws, inp)
    lo, hi = np.quantile(sales_draws, [(1 - HDI_PROB) / 2, 1 - (1 - HDI_PROB) / 2], axis=0)
    obs = panel.sales
    mean = sales_draws.mean(axis=0)
    nat = sales_draws.sum(axis=2)
    nat_lo, nat_hi = np.quantile(nat, [(1 - HDI_PROB) / 2, 1 - (1 - HDI_PROB) / 2], axis=0)
    nat_obs = obs.sum(axis=1)
    ss_res = ((obs - mean) ** 2).sum()
    ss_tot = ((obs - obs.mean()) ** 2).sum()
    return {
        "coverage_94_region_week": float(((obs >= lo) & (obs <= hi)).mean()),
        "coverage_94_national_week": float(((nat_obs >= nat_lo) & (nat_obs <= nat_hi)).mean()),
        "r2_region_week": float(1 - ss_res / ss_tot),
        "mape_region_week": float(np.mean(np.abs(mean - obs) / obs)),
        "mape_national_week": float(np.mean(np.abs(nat.mean(axis=0) - nat_obs) / nat_obs)),
        "interval_width_ratio": float(np.median((hi - lo) / obs)),
        "draw_min_sales_k": float(sales_draws.min()),
        "draw_max_sales_k": float(sales_draws.max()),
        "share_negative_sales": float((sales_draws < 0).mean()),
    }


def true_values(panel: Panel, inp: ModelInputs) -> dict[str, np.ndarray]:
    """Ground truth mapped into the model's scaled parameterisation."""
    tr = panel.truth
    ys = inp.y_scale
    lam = np.array(tr["lam"])
    beta = np.array(tr["beta_pc"]) / ys
    sat_ref = logistic_saturation_np(inp.x_mean, lam)
    phase = 2 * np.pi * 0.92
    return {
        "alpha": np.array(tr["alpha"]),
        "lam": lam,
        "beta": beta,
        "effect_at_mean": beta * sat_ref[None, :],
        "gamma_price": np.array(tr["price_effect_pc"] / ys),
        "gamma_comp": np.array(tr["competitor_effect_pc"] / ys),
        "gamma_holiday": np.array(tr["holiday_effect_pc"] / ys),
        "gamma_fourier": np.array([8 * np.sin(phase), 8 * np.cos(phase), 3.0, 0.0]) / ys,
        "trend": np.array(6.0 / ys),
        "sigma": np.array(tr["noise_sd_pc"] / ys),
        "roas": np.array(tr["roas"]),
    }


def recovery(idata, panel: Panel, inp: ModelInputs) -> pd.DataFrame:
    rows = []
    truth = true_values(panel, inp)
    for var, tv in truth.items():
        draws = flat(idata, var)
        tv = np.asarray(tv)
        draws = draws.reshape(draws.shape[0], -1)
        labels = _labels(var, tv.shape, panel)
        for j, (lab, t) in enumerate(zip(labels, tv.reshape(-1), strict=True)):
            lo, hi = hdi(draws[:, j])
            rows.append(
                {
                    "parameter": lab,
                    "group": var,
                    "truth": float(t),
                    "mean": float(draws[:, j].mean()),
                    "hdi_low": lo,
                    "hdi_high": hi,
                    "in_hdi": bool(lo <= t <= hi),
                }
            )
    return pd.DataFrame(rows)


def _labels(var: str, shape: tuple, panel: Panel) -> list[str]:
    if shape == ():
        return [var]
    if var == "gamma_fourier":
        return [f"{var}[{k}]" for k in ("sin1", "cos1", "sin2", "cos2")]
    if len(shape) == 1:
        return [f"{var}[{c}]" for c in CHANNELS]
    return [f"{var}[{r},{c}]" for r in panel.regions for c in CHANNELS]


def roas_table(idata, panel: Panel) -> pd.DataFrame:
    draws = flat(idata, "roas")
    contrib = flat(idata, "channel_contribution").sum(axis=1)  # (S, C)
    spend = panel.spend.sum(axis=(0, 1))
    rows = []
    for k, c in enumerate(CHANNELS):
        lo, hi = hdi(draws[:, k])
        clo, chi = hdi(contrib[:, k])
        rows.append(
            {
                "channel": c,
                "spend_k": float(spend[k]),
                "true_roas": float(panel.truth["roas"][k]),
                "roas_mean": float(draws[:, k].mean()),
                "roas_hdi_low": lo,
                "roas_hdi_high": hi,
                "true_in_hdi": bool(lo <= panel.truth["roas"][k] <= hi),
                "last_click_roas": float(panel.truth["last_click"]["roas"][k]),
                "contribution_mean_k": float(contrib[:, k].mean()),
                "contribution_hdi_low_k": clo,
                "contribution_hdi_high_k": chi,
                "true_contribution_k": float(panel.truth["contribution_total_k"][k]),
                "contribution_share_of_sales": float(contrib[:, k].mean() / panel.sales.sum()),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, digits: int = 3) -> str:
    """Minimal GitHub-markdown table (avoids a tabulate dependency)."""

    def fmt(v):
        if isinstance(v, float | np.floating):
            return f"{v:.{digits}f}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])
