from pathlib import Path

from semsql import generator


def test_deterministic():
    a = generator.generate(generator.GenConfig(n_accounts=50))
    b = generator.generate(generator.GenConfig(n_accounts=50))
    for k in a:
        assert a[k].equals(b[k]), k


def test_semantic_traps_present(tables):
    subs, acc, inv = tables["subscriptions"], tables["accounts"], tables["invoices"]
    assert acc["is_test"].any() and (inv["status"] == "void").any()
    assert inv["currency"].nunique() >= 5
    # plan changes: a cancelled subscription followed by another one for the same account
    multi = subs.groupby("account_id").size()
    assert (multi > 1).any()
    cancelled = (subs["status"] == "cancelled").sum()
    churned = subs.groupby("account_id")["end_date"].apply(lambda s: s.notna().all()).sum()
    assert cancelled > churned  # naive "cancelled subscriptions" overstates churn


def test_committed_sample_is_small():
    sample = Path(__file__).resolve().parents[1] / "data" / "sample"
    files = list(sample.glob("*_SYNTHETIC_sample.csv"))
    assert files and sum(f.stat().st_size for f in files) <= 1_000_000
