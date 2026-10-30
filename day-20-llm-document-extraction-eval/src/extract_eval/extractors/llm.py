"""Claude extractor: schema-constrained output (structured outputs), retries and a validation-repair loop.

Only used when ANTHROPIC_API_KEY is set (or a client is injected, as the tests do with a fake client).
Configuration via environment variables:

  ANTHROPIC_MODEL            model id (default ``claude-opus-5-5``)
  ANTHROPIC_EFFORT           output_config.effort (default ``low``: extraction is a routine task)
  ANTHROPIC_FALLBACKS        "1" (default) enables server-side refusal fallbacks, "0" disables
  ANTHROPIC_PRICE_IN/OUT     USD per million input/output tokens for cost tracking (defaults: Opus 5.5 list price)
  LLM_MAX_DOCS               cap on documents sent per run (default: all)
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from ..pdf_text import DocText
from ..prompts import load_prompt
from ..schemas import FIELDS_BY_TYPE, validate_record
from .base import ExtractionResult, detect_doc_type

DEFAULT_MODEL = "claude-opus-5-5"
PRICES_PER_MTOK = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}

_NULLABLE = {
    "string": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    "number": {"anyOf": [{"type": "number"}, {"type": "null"}]},
    "integer": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
}
CONTRACT_PROPS: dict[str, Any] = {
    "contract_number": _NULLABLE["string"],
    "supplier": {"type": "string"},
    "customer": {"type": "string"},
    "effective_date": {"type": "string", "description": "YYYY-MM-DD"},
    "expiration_date": {"anyOf": [{"type": "string", "description": "YYYY-MM-DD"}, {"type": "null"}]},
    "total_value": _NULLABLE["number"],
    "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
    "payment_terms_days": _NULLABLE["integer"],
    "auto_renewal": {"type": "boolean"},
    "renewal_term_months": _NULLABLE["integer"],
    "renewal_notice_days": _NULLABLE["integer"],
    "governing_law": _NULLABLE["string"],
}
LINE_ITEM = {
    "type": "object",
    "additionalProperties": False,
    "required": ["description", "quantity", "unit_price", "amount"],
    "properties": {
        "description": {"type": "string"},
        "quantity": {"type": "number"},
        "unit_price": {"type": "number"},
        "amount": {"type": "number"},
    },
}
INVOICE_PROPS: dict[str, Any] = {
    "invoice_number": {"type": "string"},
    "po_number": _NULLABLE["string"],
    "supplier": {"type": "string"},
    "customer": {"type": "string"},
    "invoice_date": {"type": "string", "description": "YYYY-MM-DD"},
    "due_date": {"type": "string", "description": "YYYY-MM-DD"},
    "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
    "subtotal": {"type": "number"},
    "tax": {"type": "number"},
    "total": {"type": "number"},
    "payment_terms_days": _NULLABLE["integer"],
    "line_items": {"type": "array", "items": LINE_ITEM},
}


def output_schema(doc_type: str) -> dict[str, Any]:
    props = dict(CONTRACT_PROPS if doc_type == "contract" else INVOICE_PROPS)
    conf_fields = FIELDS_BY_TYPE[doc_type] + (["line_items"] if doc_type == "invoice" else [])
    conf = {
        "type": "object",
        "additionalProperties": False,
        "required": conf_fields,
        "properties": {f: {"type": "number"} for f in conf_fields},
    }
    record = {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["record", "field_confidence"],
        "properties": {"record": record, "field_confidence": conf},
    }


class ClaudeExtractor:
    name = "claude"

    def __init__(
        self, client: Any | None = None, model: str | None = None, max_repairs: int = 2, prompt_version: str = "extract_v1"
    ) -> None:
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.effort = os.environ.get("ANTHROPIC_EFFORT", "low")
        self.fallbacks = os.environ.get("ANTHROPIC_FALLBACKS", "1") == "1"
        default_in, default_out = PRICES_PER_MTOK.get(self.model, PRICES_PER_MTOK[DEFAULT_MODEL])
        self.price_in = float(os.environ.get("ANTHROPIC_PRICE_IN", default_in))
        self.price_out = float(os.environ.get("ANTHROPIC_PRICE_OUT", default_out))
        self.max_repairs = max_repairs
        self.prompt_version = prompt_version
        self.system_prompt = load_prompt(prompt_version)
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=4)  # SDK retries 408/409/429/5xx and connection errors with backoff
        self.client = client

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _call(self, messages: list[dict[str, Any]], doc_type: str) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": self.system_prompt,
            "messages": messages,
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": output_schema(doc_type)}},
        }
        if self.fallbacks:
            return self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        return self.client.messages.create(**kwargs)

    def extract(self, doc: DocText) -> ExtractionResult:
        t0 = time.perf_counter()
        doc_type = detect_doc_type(doc.text)
        res = ExtractionResult(
            doc.doc_id,
            self.name,
            doc_type,
            {f: None for f in FIELDS_BY_TYPE[doc_type]},
            [] if doc_type == "invoice" else None,
            {},
            prompt_version=self.prompt_version,
            attempts=0,
        )
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Document type: {doc_type}\n<document>\n{doc.as_prompt_text()}\n</document>"}
        ]
        for attempt in range(1 + self.max_repairs):
            res.attempts = attempt + 1
            try:
                resp = self._call(messages, doc_type)
            except Exception as exc:  # SDK has already retried transient errors; record and stop
                res.notes.append(f"api_error: {type(exc).__name__}: {exc}"[:300])
                break
            usage = getattr(resp, "usage", None)
            res.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            res.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            stop = getattr(resp, "stop_reason", None)
            if stop in ("refusal", "max_tokens"):
                res.notes.append(f"stop_reason={stop}")
                break
            text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
            try:
                payload = json.loads(text)
                record, conf = payload["record"], payload.get("field_confidence", {})
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                errors = [f"output was not the required JSON object: {exc}"]
            else:
                items = record.pop("line_items", None) if doc_type == "invoice" else None
                res.fields = {f: record.get(f) for f in FIELDS_BY_TYPE[doc_type]}
                res.line_items = items
                res.confidence = {k: min(1.0, max(0.0, float(v))) for k, v in conf.items() if isinstance(v, int | float)}
                _, errors = validate_record(doc_type, res.record_payload())
                if not errors:
                    res.valid, res.validation_errors = True, []
                    break
            res.validation_errors = errors
            messages = messages + [
                {"role": "assistant", "content": resp.content},
                {
                    "role": "user",
                    "content": "The extraction failed validation:\n- "
                    + "\n- ".join(errors)
                    + "\nRe-read the document and return the corrected JSON. Use null for values the document does not state.",
                },
            ]
        res.latency_s = time.perf_counter() - t0
        res.cost_usd = round(res.input_tokens / 1e6 * self.price_in + res.output_tokens / 1e6 * self.price_out, 6)
        return res
