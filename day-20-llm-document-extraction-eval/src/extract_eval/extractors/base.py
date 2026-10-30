"""Extractor interface shared by the offline rules baselines and the Claude extractor."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ..pdf_text import DocText


@dataclass
class ExtractionResult:
    doc_id: str
    extractor: str
    doc_type: str
    fields: dict[str, Any]
    line_items: list[dict[str, Any]] | None
    confidence: dict[str, float]
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    attempts: int = 1
    valid: bool = False
    validation_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    prompt_version: str | None = None

    def record_payload(self) -> dict[str, Any]:
        out = dict(self.fields)
        if self.doc_type == "invoice":
            out["line_items"] = self.line_items or []
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Extractor(Protocol):
    name: str

    def extract(self, doc: DocText) -> ExtractionResult: ...


def detect_doc_type(text: str) -> str:
    """Shared deterministic router: invoices announce themselves in the header lines."""
    head = " ".join(text.splitlines()[:4]).lower().replace(" ", "")
    return "invoice" if re.search(r"inv[o0]ice|statement&|rechnung|bill\b", head) else "contract"
