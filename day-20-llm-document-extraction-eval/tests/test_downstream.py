import copy
import json
import random

import pytest

from extract_eval.downstream import DownstreamStore, InvalidRecordError
from extract_eval.extractors.base import ExtractionResult
from extract_eval.generator import generate
from extract_eval.schemas import validate_record


def _result(doc, fields):
    f = {k: v for k, v in fields.items() if k not in ("doc_type", "line_items")}
    return ExtractionResult(doc.doc_id, "test", doc.doc_type, f, fields.get("line_items"), {})


def test_invalid_record_is_refused(tmp_path):
    doc = next(d for d in generate(4) if d.doc_type == "invoice")
    bad = copy.deepcopy(doc.gold)
    bad["total"] = bad["total"] + 1
    store = DownstreamStore(tmp_path)
    with pytest.raises(InvalidRecordError):
        store.write_accepted(_result(doc, bad))
    assert store.accepted_path.read_text() == ""
    assert store.route(_result(doc, bad)) == "quarantine"
    assert store.accepted_path.read_text() == ""


def test_random_corruptions_never_reach_accepted(tmp_path):
    rng = random.Random(7)
    store = DownstreamStore(tmp_path)
    n_invalid = 0
    for doc in generate(120):
        rec = copy.deepcopy(doc.gold)
        k = rng.choice(list(rec))
        if k == "doc_type":
            continue
        rec[k] = rng.choice([None, "garbage", -5, 10**12, "2020-13-45", [] if k == "line_items" else 3.14159])
        res = _result(doc, rec)
        if validate_record(doc.doc_type, res.record_payload())[0] is None:
            n_invalid += 1
        store.route(res)
    assert n_invalid > 30 and store.n_quarantined == n_invalid
    for line in store.accepted_path.read_text().splitlines():
        r = json.loads(line)["record"]
        assert validate_record(r["doc_type"], r)[0] is not None
