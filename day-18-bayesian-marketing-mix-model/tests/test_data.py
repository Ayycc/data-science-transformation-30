import numpy as np
import pytest

from bayes_mmm.config import CHANNELS, DATA_SAMPLE, N_REGIONS, N_WEEKS
from bayes_mmm.data import TARGET_ROAS, geometric_adstock_np, load, logistic_saturation_np, simulate


def test_shapes_and_labels(panel):
    assert panel.sales.shape == (N_WEEKS, N_REGIONS)
    assert panel.spend.shape == (N_WEEKS, N_REGIONS, len(CHANNELS))
    assert panel.truth["label"].startswith("SYNTHETIC")
    assert (panel.sales > 0).all() and (panel.spend >= 0).all()


def test_seeded_determinism():
    a, b = simulate(seed=11), simulate(seed=11)
    np.testing.assert_array_equal(a.sales, b.sales)
    assert not np.allclose(a.sales, simulate(seed=12).sales)


def test_true_roas_calibrated(panel):
    # Region multipliers are applied after calibration, so national ROAS is exact to rounding.
    np.testing.assert_allclose(panel.truth["roas"], TARGET_ROAS, rtol=1e-9)


def test_last_click_overcredits_search(panel):
    lc = np.array(panel.truth["last_click"]["roas"])
    true = np.array(panel.truth["roas"])
    k = CHANNELS.index("search")
    assert lc[k] > 2 * true[k]
    assert lc[CHANNELS.index("tv")] < true[CHANNELS.index("tv")]


def test_adstock_reference():
    x = np.zeros((6, 1))
    x[0] = 1.0
    out = geometric_adstock_np(x, np.array([0.5]), l_max=3, normalize=False)
    np.testing.assert_allclose(out[:, 0], [1, 0.5, 0.25, 0, 0, 0])
    # normalised adstock preserves a constant (steady-state) level after l_max weeks
    const = geometric_adstock_np(np.ones((10, 2)), np.array([0.3, 0.8]), l_max=4)
    np.testing.assert_allclose(const[4:], 1.0)


def test_saturation_reference():
    assert logistic_saturation_np(np.array(0.0), 2.0) == 0.0
    v = logistic_saturation_np(np.linspace(0, 5, 20), 1.5)
    assert np.all(np.diff(v) > 0) and v[-1] < 1


@pytest.mark.skipif(not (DATA_SAMPLE / "panel_sample.csv.gz").exists(), reason="sample not generated")
def test_committed_sample_matches_generator(panel):
    path = DATA_SAMPLE / "panel_sample.csv.gz"
    assert path.stat().st_size <= 1_000_000
    loaded = load(path, DATA_SAMPLE / "truth.json")
    np.testing.assert_allclose(loaded.sales, panel.sales, atol=1e-3)
    np.testing.assert_allclose(loaded.spend, panel.spend, atol=1e-3)
    assert loaded.truth["roas"] == pytest.approx(panel.truth["roas"])
