"""Feature correctness, Polars/pandas parity and proof that no feature uses future information."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from card_fraud import features_pandas as fpd
from card_fraud.config import FEATURES
from card_fraud.features_polars import build_features

KEY = ["txn_id", *FEATURES]


def _cmp(a: pl.DataFrame, b: pl.DataFrame) -> None:
    assert_frame_equal(a.select(KEY), b.select(KEY), check_exact=False, rel_tol=1e-9, abs_tol=1e-9)


def test_polars_matches_pandas(raw: pl.DataFrame, feats: pl.DataFrame) -> None:
    p = fpd.build_features(raw.to_pandas())
    assert np.array_equal(p["txn_id"].to_numpy(), feats["txn_id"].to_numpy())
    for c in FEATURES:
        a = feats[c].cast(pl.Float64).to_numpy()
        b = p[c].astype("float64").to_numpy()
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=c)


@pytest.mark.parametrize("q", [0.15, 0.5, 0.83])
def test_truncating_the_future_does_not_change_past_features(raw: pl.DataFrame, feats: pl.DataFrame, q: float):
    cutoff = raw["ts"].sort()[int(q * raw.height)]
    trunc = build_features(raw.filter(pl.col("ts") <= cutoff))
    _cmp(feats.filter(pl.col("ts") <= cutoff), trunc)


def test_perturbing_future_rows_does_not_change_past_features(raw: pl.DataFrame, feats: pl.DataFrame) -> None:
    cutoff = raw["ts"].sort()[raw.height // 2]
    fut = pl.col("ts") > cutoff
    tampered = raw.with_columns(
        amount=pl.when(fut).then(pl.col("amount") * 7 + 13).otherwise(pl.col("amount")),
        lat=pl.when(fut).then(pl.col("lat") + 10).otherwise(pl.col("lat")),
        device_id=pl.when(fut).then(pl.col("device_id") + 999).otherwise(pl.col("device_id")),
        is_fraud=pl.when(fut).then(1 - pl.col("is_fraud")).otherwise(pl.col("is_fraud")),
    )
    out = build_features(tampered)
    _cmp(feats.filter(pl.col("ts") <= cutoff), out.filter(pl.col("ts") <= cutoff))
    # sanity: the perturbation is visible after the cutoff, so the test can fail
    after = feats.filter(pl.col("ts") > cutoff)
    assert not out.filter(pl.col("ts") > cutoff)["amount"].equals(after["amount"])


def test_own_and_same_month_labels_are_not_used(raw: pl.DataFrame, feats: pl.DataFrame) -> None:
    flipped = build_features(
        raw.with_columns(
            is_fraud=pl.when(pl.col("ts").dt.month() == 3).then(1 - pl.col("is_fraud")).otherwise(pl.col("is_fraud"))
        )
    )
    _cmp(feats.filter(pl.col("fold") <= 3), flipped.filter(pl.col("fold") <= 3))
    # April's encoding does use March labels -> it must change (the test is sensitive)
    assert not np.allclose(
        feats.filter(pl.col("fold") == 4)["merchant_risk_te"].to_numpy(),
        flipped.filter(pl.col("fold") == 4)["merchant_risk_te"].to_numpy(),
    )


def test_validation_and_test_labels_never_reach_features(raw: pl.DataFrame, feats: pl.DataFrame) -> None:
    flipped = build_features(
        raw.with_columns(
            is_fraud=pl.when(pl.col("ts").dt.month() >= 5).then(1 - pl.col("is_fraud")).otherwise(pl.col("is_fraud"))
        )
    )
    _cmp(feats, flipped)


def test_first_month_has_no_merchant_history(feats: pl.DataFrame) -> None:
    jan = feats.filter(pl.col("fold") == 1)
    assert jan["merchant_risk_te"].null_count() == jan.height
    assert feats.filter(pl.col("fold") > 1)["merchant_risk_te"].null_count() == 0


def _tiny() -> pl.DataFrame:
    ts = [datetime(2025, 3, 1, 10, 0), datetime(2025, 3, 1, 10, 30), datetime(2025, 3, 1, 11, 0),
          datetime(2025, 3, 1, 11, 0), datetime(2025, 3, 3, 9, 0)]  # fmt: skip
    return pl.DataFrame(
        {
            "txn_id": [0, 1, 2, 3, 4],
            "ts": pl.Series(ts).cast(pl.Datetime("ms")),
            "card_id": pl.Series([7, 7, 7, 7, 7], dtype=pl.Int32),
            "merchant_id": pl.Series([1, 1, 2, 3, 1], dtype=pl.Int32),
            "mcc": ["grocery", "grocery", "digital_goods", "digital_goods", "grocery"],
            "channel": ["in_person", "in_person", "online", "online", "in_person"],
            "amount": [100.0, 2.0, 3.0, 4.0, 50.0],
            "lat": pl.Series([40.71, 40.71, 51.51, 51.51, 40.71], dtype=pl.Float32),
            "lon": pl.Series([-74.01, -74.01, -0.13, -0.13, -74.01], dtype=pl.Float32),
            "device_id": [0, 0, 55, 55, 0],
            "is_fraud": pl.Series([0, 0, 1, 1, 0], dtype=pl.Int8),
            "attack_type": ["none", "none", "card_testing", "card_testing", "none"],
        }
    )


def test_hand_computed_values() -> None:
    f = build_features(_tiny()).sort("txn_id")
    # windows are [t - w, t): the current transaction and same-timestamp transactions are excluded
    assert f["card_txn_count_1h"].to_list() == [0, 1, 2, 2, 0]
    assert f["card_amount_sum_1h"].to_list() == [0, 100, 102, 102, 0]
    assert f["card_small_txn_count_1h"].to_list() == [0, 0, 1, 1, 0]
    assert f["card_txn_count_7d"].to_list() == [0, 1, 2, 2, 4]
    assert f["secs_since_prev"].to_list()[:3] == [None, 1800.0, 1800.0]
    assert f["is_new_device"].to_list() == [0, 0, 1, 0, 0]
    assert f["is_new_merchant_for_card"].to_list() == [1, 0, 1, 1, 0]
    # NYC -> London in 30 minutes: ~5,570 km at >10,000 km/h
    d, s = f["dist_prev_km"][2], f["speed_kmh"][2]
    assert 5_500 < d < 5_650 and s > 10_000
    assert f["card_amount_mean_30d"][4] == pytest.approx((100 + 2 + 3 + 4) / 4)
