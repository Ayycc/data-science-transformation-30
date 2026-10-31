"""Two chunking strategies over the same regulation.

* ``structure`` - one chunk per lettered clause, prefixed with its section number and heading, so the chunk is
  self-describing ("§ 4.2 Customer complaints (b) ...") and maps to exactly one citable clause id.
* ``fixed``     - sliding windows of ``size`` whitespace tokens with ``overlap`` tokens of overlap over the flattened
  document, ignoring structure (the common default in RAG tutorials). A window can straddle clauses and sections.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import TITLE, Clause


@dataclass
class Chunk:
    chunk_id: str
    text: str  # text that is indexed and shown to the generator
    clause_ids: list[str]  # clauses whose text overlaps this chunk (the citable units)
    start: int  # char span in the flattened document
    end: int
    heading: str = ""


def render_with_offsets(clauses: list[Clause]) -> tuple[str, dict[str, tuple[int, int]]]:
    """Flatten clauses into one document and return char offsets of every clause body."""
    parts: list[str] = [TITLE, "\n"]
    pos = sum(len(p) for p in parts)
    offsets: dict[str, tuple[int, int]] = {}
    last = None
    for c in clauses:
        if c.section_id != last:
            header = f"\n§ {c.section_id} {c.heading}\n"
            parts.append(header)
            pos += len(header)
            last = c.section_id
        prefix = f"({c.letter}) "
        parts.append(prefix)
        pos += len(prefix)
        offsets[c.clause_id] = (pos, pos + len(c.text))
        parts.append(c.text + "\n")
        pos += len(c.text) + 1
    return "".join(parts), offsets


def structure_chunks(clauses: list[Clause]) -> list[Chunk]:
    _, offsets = render_with_offsets(clauses)
    out = []
    for c in clauses:
        s, e = offsets[c.clause_id]
        out.append(Chunk(c.clause_id, f"§ {c.section_id} {c.heading} ({c.letter}) {c.text}", [c.clause_id], s, e, c.heading))
    return out


def fixed_chunks(clauses: list[Clause], size: int = 80, overlap: int = 20) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    doc, offsets = render_with_offsets(clauses)
    tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", doc)]
    out: list[Chunk] = []
    step = size - overlap
    for i, first in enumerate(range(0, len(tokens), step)):
        window = tokens[first : first + size]
        s, e = window[0][0], window[-1][1]
        cids = [cid for cid, (cs, ce) in offsets.items() if cs < e and ce > s]
        out.append(Chunk(f"F{i:04d}", doc[s:e], cids, s, e))
        if first + size >= len(tokens):
            break
    return out


def answer_span(clause: Clause, offsets: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """Char span of the gold answer inside the flattened document (falls back to the whole clause)."""
    s, e = offsets[clause.clause_id]
    key = clause.answer.split(",")[0].replace("the ", "", 1) if clause.answer else ""
    idx = clause.text.find(key) if key else -1
    if idx < 0:
        return s, e
    return s + idx, s + idx + len(key)


def relevant_chunk_ids(chunks: list[Chunk], gold_clause_ids: list[str], spans: dict[str, tuple[int, int]]) -> set[str]:
    """A chunk is relevant if it contains the full answer span of a gold clause."""
    rel = set()
    for ch in chunks:
        for cid in gold_clause_ids:
            a, b = spans[cid]
            if ch.start <= a and b <= ch.end:
                rel.add(ch.chunk_id)
    return rel
