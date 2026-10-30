"""The Claude extractor is exercised with a fake client: no key, no network. Checks the repair loop and accounting."""

import copy
import json
from types import SimpleNamespace

from extract_eval.extractors.llm import ClaudeExtractor, output_schema
from extract_eval.pdf_text import DocText

GOOD = {
    "invoice_number": "INV-2026-00001",
    "po_number": None,
    "supplier": "Acme Metals Inc.",
    "customer": "Beta Foods Ltd",
    "invoice_date": "2026-01-10",
    "due_date": "2026-02-09",
    "currency": "USD",
    "subtotal": 30.0,
    "tax": 2.5,
    "total": 32.5,
    "payment_terms_days": 30,
    "line_items": [{"description": "Widget", "quantity": 3, "unit_price": 10.0, "amount": 30.0}],
}
CONF = {k: 0.9 for k in list(GOOD)}
DOC = DocText("I9999", "INVOICE\nFrom: Acme Metals Inc.\nInvoice No: INV-2026-00001")


def _resp(payload, stop="end_turn", text=None):
    body = text if text is not None else json.dumps({"record": payload, "field_confidence": CONF})
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=body)],
        stop_reason=stop,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(copy.deepcopy({k: v for k, v in kw.items() if k != "messages"}) | {"messages": list(kw["messages"])})
        return self.responses.pop(0)


def test_repair_loop_fixes_invalid_output(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-5-5")
    bad = dict(GOOD, total=99.0)
    client = FakeClient([_resp(bad), _resp(GOOD)])
    r = ClaudeExtractor(client=client).extract(DOC)
    assert r.valid and r.attempts == 2 and r.fields["total"] == 32.5
    repair_msg = client.calls[1]["messages"][-1]["content"]
    assert "failed validation" in repair_msg and "subtotal" in repair_msg
    assert r.input_tokens == 2000 and r.output_tokens == 400
    assert abs(r.cost_usd - (2000 * 4.0 + 400 * 20.0) / 1e6) < 1e-12
    assert client.calls[0]["model"] == "claude-opus-5-5"
    assert client.calls[0]["output_config"]["format"]["type"] == "json_schema"
    assert client.calls[0]["fallbacks"] == "default"
    assert r.prompt_version == "extract_v1"


def test_gives_up_after_max_repairs_and_stays_invalid():
    bad = dict(GOOD, total=99.0)
    client = FakeClient([_resp(bad), _resp(None, text="not json"), _resp(bad)])
    r = ClaudeExtractor(client=client, max_repairs=2).extract(DOC)
    assert not r.valid and r.attempts == 3 and r.validation_errors


def test_refusal_and_api_errors_are_recorded_not_raised(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_FALLBACKS", "0")
    r = ClaudeExtractor(client=FakeClient([_resp(GOOD, stop="refusal")])).extract(DOC)
    assert not r.valid and "stop_reason=refusal" in r.notes

    class Boom(FakeClient):
        def _create(self, **kw):
            raise RuntimeError("network down")

    r = ClaudeExtractor(client=Boom([])).extract(DOC)
    assert not r.valid and r.notes[0].startswith("api_error")


def test_model_is_configurable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    ex = ClaudeExtractor(client=FakeClient([]))
    assert ex.model == "claude-sonnet-5-5" and ex.price_in == 2.0


def test_output_schema_is_closed():
    def walk(s):
        if isinstance(s, dict):
            if s.get("type") == "object":
                assert s["additionalProperties"] is False and set(s["required"]) == set(s["properties"])
            for v in s.values():
                walk(v)
        elif isinstance(s, list):
            for v in s:
                walk(v)

    for t in ("contract", "invoice"):
        walk(output_schema(t))


def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not ClaudeExtractor.available()
