from __future__ import annotations

import h3
import numpy as np
import pandas as pd
from scipy import stats

from civic311 import hotspots, kpi


def test_bh_fdr_matches_hand_computation():
    q = hotspots.bh_fdr(np.array([0.01, 0.04, 0.03, 0.5]))
    # sorted p = .01 .03 .04 .5 -> p*n/rank = .04 .06 .0533 .5 -> running min from the top = .04 .0533 .0533 .5
    np.testing.assert_allclose(q, [0.04, 0.16 / 3, 0.16 / 3, 0.5], rtol=1e-9)


def _grid():
    centre = h3.latlng_to_cell(41.85, -87.70, 8)
    return sorted(h3.grid_disk(centre, 8)), centre


def test_gi_star_finds_planted_cluster():
    cells, centre = _grid()
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.05, len(cells))
    cluster = set(h3.grid_disk(centre, 1))
    x[[i for i, c in enumerate(cells) if c in cluster]] += 0.5
    z, local = hotspots.getis_ord_gi_star(cells, x)
    q = hotspots.bh_fdr(2 * (1 - stats.norm.cdf(np.abs(z))))
    hot = {c for c, zi, qi in zip(cells, z, q, strict=True) if qi < 0.05 and zi > 0}
    assert centre in hot and hot <= set(h3.grid_disk(centre, 2))


def test_gi_star_null_has_no_hotspots():
    cells, _ = _grid()
    x = np.random.default_rng(1).normal(0, 0.05, len(cells))
    z, _ = hotspots.getis_ord_gi_star(cells, x)
    q = hotspots.bh_fdr(2 * (1 - stats.norm.cdf(np.abs(z))))
    assert (q < 0.05).sum() == 0


def test_rate_anomaly_detector():
    rng = np.random.default_rng(3)
    n = pd.Series(2700.0, index=range(120))
    rate = pd.Series(rng.binomial(2700, 0.03, 120) / 2700)
    a = kpi.robust_rate_anomalies(rate, n)
    assert not a.flag.any()
    rate.iloc[90] = 0.15
    a = kpi.robust_rate_anomalies(rate, n)
    assert a.flag.sum() == 1 and a.flag.iloc[90]


def test_slo_burndown_budget():
    idx = pd.date_range("2025-01-01", periods=365, freq="D")
    daily = pd.DataFrame({"rows": 100.0, "failed": 5.0}, index=idx)
    s = kpi.slo_burndown(daily, target=0.9)
    end = s.groupby("quarter").budget_left.last()
    np.testing.assert_allclose(end.to_numpy(), 0.5)
