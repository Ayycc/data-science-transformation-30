import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from regrag.chunking import answer_span, render_with_offsets, structure_chunks  # noqa: E402
from regrag.corpus import generate_corpus, generate_questions  # noqa: E402


@pytest.fixture(scope="session")
def corpus():
    clauses = generate_corpus()
    questions = generate_questions(clauses)
    doc, offsets = render_with_offsets(clauses)
    spans = {c.clause_id: answer_span(c, offsets) for c in clauses}
    return {
        "clauses": clauses,
        "questions": questions,
        "doc": doc,
        "offsets": offsets,
        "spans": spans,
        "by_id": {c.clause_id: c for c in clauses},
    }


@pytest.fixture(scope="session")
def chunks(corpus):
    return structure_chunks(corpus["clauses"])
