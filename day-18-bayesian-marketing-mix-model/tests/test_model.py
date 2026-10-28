"""Model tests on a tiny panel with very short chains: shapes and plumbing, not convergence."""

import numpy as np
import pytensor.tensor as pt
import pytest

from bayes_mmm import analysis
from bayes_mmm.config import CHANNELS, SamplerConfig
from bayes_mmm.data import TRUE_ALPHA, TRUE_LAM, geometric_adstock_np, logistic_saturation_np
from bayes_mmm.model import build_model, fit, media_response, prepare_inputs, sample_prior


def test_pymc_marketing_transform_matches_numpy(small_panel):
    inp = prepare_inputs(small_panel)
    got = media_response(pt.as_tensor(inp.x_scaled), pt.as_tensor(TRUE_ALPHA), pt.as_tensor(TRUE_LAM)).eval()
    ref = logistic_saturation_np(geometric_adstock_np(inp.x_scaled, TRUE_ALPHA), TRUE_LAM)
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)


@pytest.fixture(scope="module")
def tiny_fit(small_panel):
    inp = prepare_inputs(small_panel)
    model = build_model(inp)
    cfg = SamplerConfig(draws=40, tune=40, chains=2, prior_samples=50, seed=1)
    prior = sample_prior(model, cfg)
    idata = fit(model, cfg)
    return inp, prior, idata


def test_prior_predictive_is_plausible(small_panel, tiny_fit):
    inp, prior, _ = tiny_fit
    y = prior.prior_predictive["y"].stack(s=("chain", "draw")).transpose("s", ...).to_numpy()
    chk = analysis.predictive_check(y, small_panel, inp)
    assert chk["coverage_94_region_week"] > 0.9  # priors are wide enough to include the data
    assert chk["share_negative_sales"] < 0.1  # ...without being absurd


def test_posterior_outputs(small_panel, tiny_fit):
    inp, _, idata = tiny_fit
    assert idata.posterior["beta"].shape[2:] == (3, len(CHANNELS))
    roas = analysis.roas_table(idata, small_panel)
    assert len(roas) == len(CHANNELS) and np.isfinite(roas["roas_mean"]).all()
    conv = analysis.convergence(idata)
    assert conv["n_draws"] == 80 and np.isfinite(conv["max_rhat"])
    rec = analysis.recovery(idata, small_panel, inp)
    assert {"alpha", "lam", "beta", "roas", "gamma_price"} <= set(rec["group"])
    y = analysis.flat(idata, "y", group="posterior_predictive")
    assert y.shape == (80, *small_panel.sales.shape)
