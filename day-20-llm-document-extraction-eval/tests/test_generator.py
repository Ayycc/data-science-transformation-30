from pathlib import Path

from extract_eval.generator import add_months, fmt_num, generate, ocr_noise

ROOT = Path(__file__).resolve().parents[1]


def test_seeded_and_deterministic():
    a, b = generate(40), generate(40)
    assert [d.gold for d in a] == [d.gold for d in b] and [d.text for d in a] == [d.text for d in b]
    assert generate(40, seed=1)[0].gold != a[0].gold


def test_corpus_has_variety():
    docs = generate(500)
    assert {d.doc_type for d in docs} == {"contract", "invoice"}
    assert len({d.meta["template"] for d in docs}) == 8
    share_scanned = sum(d.meta["scanned"] for d in docs) / len(docs)
    assert 0.15 < share_scanned < 0.35


def test_helpers():
    import random

    assert add_months(__import__("datetime").date(2024, 1, 31), 1).isoformat() == "2024-02-29"
    assert fmt_num(1234.5, "eu") == "1.234,50"
    assert ocr_noise("hello world", 0.0, random.Random(0)) == "hello world"


def test_committed_sample_is_small():
    sample = ROOT / "data" / "sample"
    files = [p for p in sample.rglob("*") if p.is_file()]
    assert files, "data/sample must be committed"
    assert sum(p.stat().st_size for p in files) <= 1_000_000
