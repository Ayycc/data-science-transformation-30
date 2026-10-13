from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from card_fraud.config import SCALES
from card_fraud.generate import generate


def test_generator_is_deterministic_and_matches_committed_sample(raw: pl.DataFrame) -> None:
    a, ma = generate(SCALES["sample"], seed=42)
    b, mb = generate(SCALES["sample"], seed=42)
    assert_frame_equal(a, b)
    assert ma == mb
    assert_frame_equal(a, raw)


def test_different_seed_changes_data() -> None:
    a, _ = generate(SCALES["sample"], seed=1)
    b, _ = generate(SCALES["sample"], seed=2)
    assert a.height != b.height or not a.equals(b)


def test_schema_and_attack_patterns(raw: pl.DataFrame) -> None:
    assert raw["txn_id"].is_unique().all()
    assert raw["ts"].is_sorted()
    assert set(raw["attack_type"].unique()) == {"none", "card_testing", "account_takeover", "geo_impossible"}
    assert ((raw["attack_type"] != "none") == (raw["is_fraud"] == 1)).all()
    ato = raw.filter(pl.col("attack_type") == "account_takeover")
    assert (ato["channel"] == "online").all() and (ato["amount"] >= 5).all()
    geo = raw.filter(pl.col("attack_type") == "geo_impossible")
    assert (geo["channel"] == "in_person").all()
    ct = raw.filter(pl.col("attack_type") == "card_testing")
    assert (ct.filter(pl.col("mcc") == "digital_goods")["amount"] < 10).all()
    # card testers always use a fresh device; most (not all) account takeovers do too
    assert (ct["device_id"] >= 10_000_000).all()
    fresh = ato.group_by("card_id").agg(fresh=pl.col("device_id").min() >= 10_000_000)["fresh"].mean()
    assert 0.4 < fresh < 1.0


def test_full_scale_parameters_target_brief() -> None:
    s = SCALES["full"]
    assert s.n_cards == 100_000
    expected_txn = s.n_cards * 181 * s.txn_rate_per_day
    assert 4.5e6 < expected_txn < 5.5e6
