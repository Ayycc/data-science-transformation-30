import json
from pathlib import Path

from regrag.data import build, load
from regrag.ecfr import cloze_questions, parse_ecfr
from regrag.pipeline import run

ROOT = Path(__file__).resolve().parents[1]

XML = """<ECFR><DIV5 N="312" TYPE="PART"><DIV8 N="312.10" TYPE="SECTION"><HEAD>§ 312.10 Data retention and deletion requirements.</HEAD>
<P>(a) An operator of a website shall retain personal information collected online from a child for only as long as is reasonably necessary.</P>
<P>(b) An operator must respond to a parental request within 30 days of receipt of the request from the parent.</P>
</DIV8></DIV5></ECFR>"""


def test_parse_ecfr_fixture():
    clauses = parse_ecfr(XML)
    assert [c.clause_id for c in clauses] == ["§312.10(a)", "§312.10(b)"]
    assert clauses[0].heading == "Data retention and deletion requirements"
    qs = cloze_questions(clauses)
    assert len(qs) == 1 and qs[0].answer == "30 days" and "30 days" not in qs[0].question


def test_committed_sample_matches_generator(tmp_path):
    prov = build(tmp_path / "raw", None, source="synthetic")
    assert prov["path_used"] == "synthetic_generator"
    for name in ("corpus.jsonl", "questions.jsonl"):
        assert (tmp_path / "raw" / name).read_text() == (ROOT / "data" / "sample" / name).read_text()


def test_end_to_end_on_sample(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # the offline fallback must work without a key
    m = run(ROOT / "data" / "sample", tmp_path, make_figures=False)
    assert m["llm"]["used"] is False
    g = m["generation"]["extractive_structure"]
    assert g["n_answered"] > 0 and g["answers_with_valid_citation"] == 1.0
    assert set(m["retrieval"]) == {"structure", "fixed"}
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert (
        saved["acceptance_hybrid_vs_single"]["structure"]["hybrid_recall@5"]
        == m["acceptance_hybrid_vs_single"]["structure"]["hybrid_recall@5"]
    )
    assert load(ROOT / "data" / "sample")[2]["path_used"] == "synthetic_generator"
