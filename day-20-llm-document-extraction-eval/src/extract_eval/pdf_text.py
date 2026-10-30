"""PDF -> text + tables with pdfplumber (the shared input for every extractor)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocText:
    doc_id: str
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    n_pages: int = 1
    parse_seconds: float = 0.0

    def as_prompt_text(self) -> str:
        """Text plus any ruled tables rendered as pipe rows, the representation sent to the LLM."""
        parts = [self.text]
        for k, t in enumerate(self.tables, 1):
            parts.append(f"[TABLE {k}]")
            parts.extend(" | ".join(c or "" for c in row) for row in t)
        return "\n".join(parts)


def read_pdf(path: Path) -> DocText:
    import pdfplumber

    t0 = time.perf_counter()
    texts: list[str] = []
    tables: list[list[list[str]]] = []
    with pdfplumber.open(str(path)) as pdf:
        n = len(pdf.pages)
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
            for tb in page.extract_tables():
                tables.append([[(c or "").strip() for c in row] for row in tb])
    return DocText(Path(path).stem, "\n".join(texts), tables, n, time.perf_counter() - t0)
