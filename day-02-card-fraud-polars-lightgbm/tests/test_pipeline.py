"""Time split + end-to-end smoke run on the committed sample (writes to a temp dir, never to results/)."""

from __future__ import annotations

import json

import polars as pl

from card_fraud.config import SAMPLE_DIR, SCALES
from card_fraud.model import downsample_negatives, time_split
from card_fraud.pipeline import run


def test_time_split_is_ordered_and_disjoint(feats: pl.DataFrame) -> None:
    p = time_split(feats)
    assert p["train"]["ts"].max() < p["val"]["ts"].min()
    assert p["val"]["ts"].max() < p["test"]["ts"].min()
    assert sum(d.height for d in p.values()) == feats.height
    ds = downsample_negatives(p["train"], 0.3)
    assert ds["is_fraud"].sum() == p["train"]["is_fraud"].sum()
    assert ds.height < p["train"].height


def test_end_to_end_on_sample(raw: pl.DataFrame, tmp_path) -> None:
    manifest = json.loads((SAMPLE_DIR / "manifest.json").read_text())
    m = run(raw, manifest, SCALES["sample"], tmp_path / "results", model_dir=tmp_path / "models")
    assert (tmp_path / "results" / "metrics.json").exists()
    assert m["feature_benchmark"]["max_relative_difference"] < 1e-9
    test = m["ranking"]["lightgbm"]["test"]
    assert test["pr_auc"] > m["split"]["test"]["fraud_rate"] * 5
    pol = m["policies_test"]
    assert pol["lightgbm_cost_optimal"]["total_cost"] < pol["approve_all"]["total_cost"]
    assert m["threshold"]["val_total_cost_at_threshold"] <= m["threshold"]["val_total_cost_approve_all"]
    assert m["explainability"]["shap_vs_lightgbm_pred_contrib_max_abs_diff"] < 1e-6
    assert m["explainability"]["alerts_explained"] == pol["lightgbm_cost_optimal"]["alerts"]
