"""Regression snapshot: rules_repair outputs on the committed sample must not change silently.
Regenerate deliberately with: PYTHONPATH=src python tests/make_snapshot.py"""

import json
from pathlib import Path

from extract_eval.extractors.rules import RulesExtractor
from extract_eval.pdf_text import read_pdf

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "tests" / "fixtures" / "rules_repair_snapshot.json"


def current() -> dict:
    ex = RulesExtractor("rules")
    out = {}
    for pdf in sorted((ROOT / "data" / "sample" / "pdf").glob("*.pdf")):
        r = ex.extract(read_pdf(pdf))
        out[r.doc_id] = {"doc_type": r.doc_type, "fields": r.fields, "line_items": r.line_items}
    return out


def test_rules_outputs_match_snapshot():
    snap = json.loads(SNAP.read_text())
    cur = json.loads(json.dumps(current()))
    diffs = [(d, k) for d in snap for k in snap[d]["fields"] if snap[d]["fields"][k] != cur[d]["fields"].get(k)]
    diffs += [(d, "line_items") for d in snap if snap[d]["line_items"] != cur[d]["line_items"]]
    assert set(snap) == set(cur) and not diffs, diffs[:10]
