from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import ROOT
from retention.data import FEATURES, GenConfig, baseline_logit, design_matrix, generate
from retention.pipeline import make_data


def test_generator_is_seeded(small, cfg):
    panel, truth = small
    p2, t2 = make_data(cfg, n_customers=20_000)
    pd.testing.assert_frame_equal(panel, p2)
    pd.testing.assert_frame_equal(truth, t2)
    assert (panel["data_source"] == "SYNTHETIC").all()


def test_panel_structure(small):
    panel, truth = small
    assert panel["customer_id"].is_unique
    assert panel["duration"].between(1, 36).all()
    assert 0.2 < panel["event"].mean() < 0.6  # plenty of events AND censoring
    assert set(panel.loc[panel["event"] == 1, "cause"]) == {"voluntary", "port_out", "non_payment"}
    camp = panel[panel["in_campaign"] == 1]
    assert abs(camp["treatment"].mean() - 0.5) < 0.02  # randomised 50/50
    assert (panel.loc[panel["in_campaign"] == 0, "treatment"] == 0).all()
    assert len(truth) == len(camp)
    assert list(design_matrix(panel).columns) == FEATURES


def test_truth_is_consistent_with_simulated_outcomes(small):
    panel, truth = small
    assert np.allclose(truth["tau_true"], truth["p_churn0"] - truth["p_churn1"])
    camp = panel[panel["in_campaign"] == 1].merge(truth, on="customer_id")
    # simulated churn in the window matches the analytic probabilities in each arm
    for arm, col in ((0, "p_churn0"), (1, "p_churn1")):
        g = camp[camp["treatment"] == arm]
        se = np.sqrt(g[col].mean() * (1 - g[col].mean()) / len(g))
        assert abs(g["churn_6m"].mean() - g[col].mean()) < 4 * se
    # heterogeneous effects: persuadables and sleeping dogs both exist
    assert (truth["tau_true"] > 0.05).mean() > 0.05
    assert (truth["tau_true"] < 0).mean() > 0.03


def test_baseline_breaks_proportional_hazards():
    t = np.arange(1, 37)
    one = baseline_logit(t, np.ones_like(t))
    m2m = baseline_logit(t, np.zeros_like(t))
    ratio = one - m2m  # log-odds ratio vs month-to-month changes over tenure
    assert ratio.max() - ratio.min() > 2.0


def test_committed_sample_matches_generator(cfg):
    sample = pd.read_csv(ROOT / "data" / "sample" / "panel_sample.csv.gz")
    d = cfg["data"]
    panel, _ = generate(
        GenConfig(
            d["n_customers"],
            d["n_months"],
            d["max_start_month"],
            d["dropout_monthly"],
            d["campaign_month"],
            d["campaign_window"],
            d["treat_share"],
            cfg["seed"],
        )
    )
    head = panel.head(len(sample)).reset_index(drop=True)
    pd.testing.assert_frame_equal(sample, head, check_dtype=False, check_exact=False, rtol=1e-6)
    assert (ROOT / "data" / "sample" / "panel_sample.csv.gz").stat().st_size < 1_000_000
