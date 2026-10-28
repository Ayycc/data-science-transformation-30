"""Hierarchical Bayesian MMM in PyMC with PyMC-Marketing's geometric adstock and logistic saturation.

Per-capita sales (scaled by their mean) for region r in week t:

    y[t,r] = intercept[r] + trend * t + fourier[t] @ gamma + holiday[t] * g_h + price[t,r] * g_p + comp[t,r] * g_c
             + sum_c beta[r,c] * logistic(adstock(x[t,r,c] / scale_c; alpha_c); lam_c) + noise

Channel effects are partially pooled across regions. To break the beta-lam ridge of the saturation
curve, the model is parameterised by the effect at the channel's mean scaled spend x_mean_c:
log effect[r,c] = mu_c + sigma_c * z[r,c] and beta[r,c] = effect[r,c] / logistic(x_mean_c; lam_c).
Adstock decay (alpha) and saturation (lam) are shared by all regions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
from pytensor.xtensor import as_xtensor

from .config import CHANNELS, FOURIER_ORDER, L_MAX, SamplerConfig
from .data import Panel, fourier_features


@dataclass
class ModelInputs:
    y_scaled: np.ndarray  # (T, R)
    x_scaled: np.ndarray  # (T, R, C)
    y_scale: float  # $k per million people per week
    channel_scale: np.ndarray  # (C,) per-capita $k
    x_mean: np.ndarray  # (C,) mean scaled spend
    population: np.ndarray  # (R,)
    spend_total: np.ndarray  # (C,) $k over the sample
    t_norm: np.ndarray
    fourier: np.ndarray
    holiday: np.ndarray
    price_c: np.ndarray
    competitor: np.ndarray
    coords: dict


def prepare_inputs(panel: Panel) -> ModelInputs:
    pop = panel.population
    y_pc = panel.sales / pop[None, :]
    x_pc = panel.spend / pop[None, :, None]
    y_scale = float(y_pc.mean())
    channel_scale = x_pc.reshape(-1, x_pc.shape[-1]).max(axis=0)
    t = panel.n_weeks
    fourier = fourier_features(panel.dates, FOURIER_ORDER)
    coords = {
        "date": panel.dates.strftime("%Y-%m-%d").tolist(),
        "region": list(panel.regions),
        "channel": list(CHANNELS),
        "fourier": [f"{kind}{k}" for k in range(1, FOURIER_ORDER + 1) for kind in ("sin", "cos")],
    }
    return ModelInputs(
        y_scaled=y_pc / y_scale,
        x_scaled=x_pc / channel_scale,
        y_scale=y_scale,
        channel_scale=channel_scale,
        x_mean=(x_pc / channel_scale).reshape(-1, x_pc.shape[-1]).mean(axis=0),
        population=pop,
        spend_total=panel.spend.sum(axis=(0, 1)),
        t_norm=np.arange(t) / max(t - 1, 1),
        fourier=fourier,
        holiday=panel.holiday.astype(float),
        price_c=panel.price - 1.0,
        competitor=panel.competitor,
        coords=coords,
    )


def media_response(x_scaled: pt.TensorVariable, alpha: pt.TensorVariable, lam: pt.TensorVariable) -> pt.TensorVariable:
    """PyMC-Marketing geometric adstock (normalised, l_max weeks) then logistic saturation, (T, R, C)."""
    xx = as_xtensor(x_scaled, dims=("date", "region", "channel"))
    ad = geometric_adstock(xx, as_xtensor(alpha, dims=("channel",)), l_max=L_MAX, dim="date", normalize=True)
    sat = logistic_saturation(ad, lam=as_xtensor(lam, dims=("channel",)))
    return sat.transpose("date", "region", "channel").values


def build_model(inp: ModelInputs) -> pm.Model:
    with pm.Model(coords=inp.coords) as model:
        x = pm.Data("x_scaled", inp.x_scaled, dims=("date", "region", "channel"))

        alpha = pm.Beta("alpha", 2.0, 2.0, dims="channel")
        lam = pm.Gamma("lam", alpha=3.0, beta=1.0, dims="channel")
        # Effect at the channel's mean spend (well identified), pooled across regions on the log scale.
        mu_log_effect = pm.Normal("mu_log_effect", -3.0, 1.0, dims="channel")
        sigma_log_effect = pm.HalfNormal("sigma_log_effect", 0.3, dims="channel")
        z_effect = pm.Normal("z_effect", 0.0, 1.0, dims=("region", "channel"))
        effect = pm.Deterministic("effect_at_mean", pt.exp(mu_log_effect + sigma_log_effect * z_effect), dims=("region", "channel"))
        # Implied ceiling (effect at full saturation): beta = effect / logistic(lam * x_mean).
        e_ref = pt.exp(-lam * inp.x_mean)
        beta = pm.Deterministic("beta", effect / ((1.0 - e_ref) / (1.0 + e_ref))[None, :], dims=("region", "channel"))

        intercept_mu = pm.Normal("intercept_mu", 0.9, 0.3)
        intercept_sigma = pm.HalfNormal("intercept_sigma", 0.1)
        # Centred: each region has 156 informative weeks, so the non-centred form would create a ridge.
        intercept = pm.Normal("intercept", intercept_mu, intercept_sigma, dims="region")

        trend = pm.Normal("trend", 0.0, 0.2)
        gamma_fourier = pm.Normal("gamma_fourier", 0.0, 0.2, dims="fourier")
        gamma_holiday = pm.Normal("gamma_holiday", 0.0, 0.2)
        gamma_price = pm.Normal("gamma_price", 0.0, 2.0)
        gamma_comp = pm.Normal("gamma_comp", 0.0, 0.2)
        sigma = pm.HalfNormal("sigma", 0.05)

        sat = media_response(x, alpha, lam)
        media = sat * beta[None, :, :]
        base = (
            intercept[None, :]
            + (trend * inp.t_norm + pt.dot(inp.fourier, gamma_fourier) + gamma_holiday * inp.holiday)[:, None]
            + gamma_price * inp.price_c
            + gamma_comp * inp.competitor
        )
        mu = base + media.sum(axis=-1)
        pm.Normal("y", mu=mu, sigma=sigma, observed=inp.y_scaled, dims=("date", "region"))

        # Contributions back in $k: scaled per-capita units x y_scale x population.
        contrib = pm.Deterministic("channel_contribution", (media * inp.population[None, :, None]).sum(axis=1) * inp.y_scale, dims=("date", "channel"))
        pm.Deterministic("roas", contrib.sum(axis=0) / inp.spend_total, dims="channel")
    return model


def sample_prior(model: pm.Model, cfg: SamplerConfig):
    with model:
        return pm.sample_prior_predictive(draws=cfg.prior_samples, random_seed=cfg.seed)


def fit(model: pm.Model, cfg: SamplerConfig, progressbar: bool = False):
    with model:
        idata = pm.sample(
            draws=cfg.draws,
            tune=cfg.tune,
            chains=cfg.chains,
            cores=min(cfg.chains, 4),
            target_accept=cfg.target_accept,
            init=cfg.init,
            random_seed=cfg.seed,
            progressbar=progressbar,
            compute_convergence_checks=False,
        )
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=cfg.seed, progressbar=progressbar)
    return idata
