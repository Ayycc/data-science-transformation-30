from __future__ import annotations

import numpy as np

from claims_mlops import config as C
from claims_mlops import data as D
from claims_mlops import models as M


def test_gini_extremes():
    rng = np.random.default_rng(1)
    y = rng.gamma(0.3, 100, 5_000) * (rng.random(5_000) < 0.2)
    w = rng.uniform(0.1, 1, 5_000)
    assert abs(M.gini(y, y, w) - 1) < 1e-9
    assert abs(M.gini(y, rng.random(5_000), w)) < 0.1


def test_models_balance_and_beat_null(small_book, glm):
    train = D.split_months(small_book, range(1, 13))
    valid = D.split_months(small_book, range(13, 19))
    pipe, cat_idx = M.build_gbm(n_estimators=150)
    gbm = M.fit(pipe, train, cat_idx)
    for model in (glm, gbm):
        tot_pred = (model.predict(train) * train[C.WEIGHT]).sum()
        tot_act = (train[C.TARGET] * train[C.WEIGHT]).sum()
        assert abs(tot_pred / tot_act - 1) < 1e-6  # balance property on training data
        ev = M.evaluate(model, valid)
        assert ev["tweedie_deviance"] < M.null_deviance(train, valid)
        assert ev["gini"] > 0.15
        assert (model.predict(valid) > 0).all()


def test_models_run_on_committed_sample(sample):
    model = M.fit(M.build_glm(n_bins=4), sample)
    ev = M.evaluate(model, sample)
    assert np.isfinite(ev["tweedie_deviance"]) and ev["n_rows"] == len(sample)
