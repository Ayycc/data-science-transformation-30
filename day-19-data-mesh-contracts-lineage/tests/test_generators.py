import filecmp

from mesh import generators as g
from mesh.config import SAMPLE


def test_generator_is_seeded():
    a = g.simulate_orders(n_days=5, base_rate=20)
    b = g.simulate_orders(n_days=5, base_rate=20)
    assert a.equals(b)


def test_raw_schema_evolves_per_domain():
    assert "order_total" in g.raw_release_on("orders", 48).columns
    assert "order_total" not in g.raw_release_on("orders", 49).columns
    assert "amount_cents" in g.raw_release_on("payments", 35).columns
    assert "carrier_code" in g.raw_release_on("logistics", 56).columns


def test_committed_sample_matches_generator(tmp_path):
    counts = g.write_raw(tmp_path, base_rate=6.0)
    assert all(v > 300 for v in counts.values())
    for domain in ("orders", "payments", "logistics"):
        cmp = filecmp.dircmp(tmp_path / domain, SAMPLE / domain)
        assert not cmp.left_only and not cmp.right_only and not cmp.diff_files, domain
    size = sum(f.stat().st_size for f in SAMPLE.rglob("*") if f.is_file())
    assert size <= 1_000_000


def test_logistics_lands_in_batches(tmp_path):
    g.write_raw(tmp_path, base_rate=6.0)
    files = sorted((tmp_path / "logistics").glob("*.csv"))
    assert files[-1].name <= "logistics_2026-09-25.csv"  # day 86: the last, incomplete 3-day batch has not landed
