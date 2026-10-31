"""Build / load the corpus and gold set. Public eCFR text if reachable, else the seeded SYNTHETIC generator."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from .chunking import render_with_offsets
from .corpus import SEED, TITLE, Clause, GoldQuestion, generate_corpus, generate_questions


def build(raw_dir: Path, sample_dir: Path | None = None, source: str | None = None) -> dict:
    source = source or os.environ.get("DATA_SOURCE", "auto")
    provenance = {"requested": source, "public_attempted": False, "public_reachable": None}
    clauses: list[Clause] | None = None
    questions: list[GoldQuestion] | None = None
    if source in ("auto", "public"):
        from .ecfr import ECFR_URL, load_public

        provenance["public_attempted"] = True
        provenance["public_url"] = ECFR_URL
        got = load_public()
        provenance["public_reachable"] = got is not None
        if got is not None:
            clauses, questions = got
            provenance["path_used"] = "public_ecfr_16cfr312"
            provenance["title"] = "16 CFR Part 312 (eCFR, public domain)"
        elif source == "public":
            raise RuntimeError("DATA_SOURCE=public but the eCFR API is unreachable")
    if clauses is None:
        clauses = generate_corpus(SEED)
        questions = generate_questions(clauses, SEED)
        provenance["path_used"] = "synthetic_generator"
        provenance["title"] = TITLE
        provenance["seed"] = SEED
    provenance["n_clauses"] = len(clauses)
    provenance["n_questions"] = len(questions)
    provenance["n_in_scope"] = sum(q.in_scope for q in questions)
    for d in [raw_dir] + ([sample_dir] if sample_dir and provenance["path_used"] == "synthetic_generator" else []):
        write(d, clauses, questions, provenance)
    return provenance


def write(d: Path, clauses: list[Clause], questions: list[GoldQuestion], provenance: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "corpus.jsonl", "w") as f:
        for c in clauses:
            f.write(json.dumps(asdict(c)) + "\n")
    with open(d / "questions.jsonl", "w") as f:
        for q in questions:
            f.write(json.dumps(asdict(q)) + "\n")
    doc, _ = render_with_offsets(clauses)
    (d / "regulation.txt").write_text(doc)
    (d / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def load(d: Path) -> tuple[list[Clause], list[GoldQuestion], dict]:
    clauses = [Clause(**json.loads(line)) for line in open(d / "corpus.jsonl")]
    questions = [GoldQuestion(**json.loads(line)) for line in open(d / "questions.jsonl")]
    provenance = json.loads((d / "provenance.json").read_text())
    return clauses, questions, provenance
