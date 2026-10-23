from __future__ import annotations

import numpy as np

from aml_graph.config import DetectorConfig
from aml_graph.motifs import detect_all, flow_links, material, motif_node_features, window_counts


def test_flow_links_respect_window_and_ratio(toy_tx):
    links = flow_links(material(toy_tx, DetectorConfig()), DetectorConfig())
    pivots = set(links["v"].tolist())
    assert 51 not in pivots  # 100 h hold > 72 h window
    assert 54 not in pivots  # onward amount is 20% of incoming
    assert {1, 2, 11, 12, 13}.issubset(pivots)
    assert (links["hold_s"] > 0).all()


def test_each_typology_detected(toy_tx):
    det, _, _ = detect_all(toy_tx)
    assert any(d.accounts == frozenset({0, 1, 2}) for d in det["cycle"])
    assert any(frozenset({10, 11, 12, 13, 14}) <= d.accounts for d in det["mule_chain"])
    sg = [d for d in det["scatter_gather"] if d.hub == 20]
    assert sg and sg[0].extra["sink"] == 25 and sg[0].extra["n_mid"] == 4
    assert any(d.hub == 36 for d in det["fan_in"])
    assert any(d.hub == 41 for d in det["fan_out"])
    # the cycle is not double-reported as a chain
    assert not any(d.accounts <= frozenset({0, 1, 2}) for d in det["mule_chain"])


def test_window_counts_vectorised(toy_tx):
    wc = window_counts(material(toy_tx, DetectorConfig()), "dst", 72 * 3600)
    assert wc.loc[36] == 6 and wc.loc[25] == 4


def test_motif_node_features(toy_tx):
    cfg = DetectorConfig()
    det, links, mat = detect_all(toy_tx, cfg)
    f = motif_node_features(60, det, links, mat, cfg)
    assert f.loc[36, "motif_fan_in_hub"] == 1
    assert f.loc[1, "motif_cycle"] >= 1
    assert np.isclose(f.loc[11, "flow_median_hold_h"], 5.0)
