import json

import pytest

from ticket_intel.config import SAMPLE_DIR
from ticket_intel.pii import Redactor, evaluate_redaction, load_handlabelled, parse_marked, regex_spans

needs_ner = pytest.mark.skipif(not Redactor().has_ner, reason="spaCy en_core_web_md not installed")


def test_email_and_phone_regex():
    text = "mail j.doe+x@example.co.uk or call (415) 555-0134 / +44 20 7946 0958 / +49 30 123456"
    labs = [(text[s.start : s.end], s.label) for s in regex_spans(text)]
    assert ("j.doe+x@example.co.uk", "EMAIL") in labs
    assert ("(415) 555-0134", "PHONE") in labs
    assert ("+44 20 7946 0958", "PHONE") in labs
    assert ("+49 30 123456", "PHONE") in labs


def test_ids_amounts_dates_are_not_phones():
    for text in ["order #123456 is late", "ORD-554433 status", "refund of 49.99 please", "billed on 2026-03-14", "invoice INV-12345"]:
        assert not [s for s in regex_spans(text) if s.label == "PHONE"], text


def test_name_cues():
    r = Redactor(use_ner=False)
    assert r.redact("Hello, my name is Wei Zhang. help") == "Hello, my name is [NAME]. help"
    assert r.redact("thanks\n\nRegards,\nAmara Osei") == "thanks\n\nRegards,\n[NAME]"


@needs_ner
def test_ner_names(redactor):
    out = redactor.redact("Please ask Margaret Thompson in accounts to resend it.")
    assert "Margaret" not in out and "Thompson" not in out and "[NAME]" in out


def test_parse_marked():
    t, g = parse_marked("call [[NAME:Ann Lee]] at [[PHONE:555-867-5309]]")
    assert t == "call Ann Lee at 555-867-5309"
    assert [t[s:e] for s, e, _ in g] == ["Ann Lee", "555-867-5309"]


def test_augment_gold_spans_are_exact(tickets):
    for text, spans in zip(tickets.text, tickets.pii_spans, strict=True):
        for s, e, lab in json.loads(spans):
            frag = text[s:e]
            assert frag.strip() == frag and frag
            if lab == "EMAIL":
                assert "@" in frag


@needs_ner
def test_recall_on_generator_labelled_tickets(tickets, redactor):
    texts = tickets.text.tolist()
    ev = evaluate_redaction(texts, tickets.pii_spans.tolist(), redactor.detect_many(texts))
    assert ev["n_gold_spans"] > 100
    assert ev["recall_any_label"] >= 0.95
    assert ev["per_type"]["EMAIL"]["recall_any_label"] == 1.0


@needs_ner
def test_recall_on_hand_labelled_set(redactor):
    texts, gold = load_handlabelled(SAMPLE_DIR / "pii_handlabelled.txt")
    assert len(texts) >= 40
    ev = evaluate_redaction(texts, gold, redactor.detect_many(texts))
    assert ev["recall_any_label"] >= 0.95


def test_evaluate_counts_partial_redaction_as_miss():
    from ticket_intel.pii import Span

    ev = evaluate_redaction(["hi Ann Lee"], [[(3, 10, "NAME")]], [[Span(3, 6, "NAME")]])
    assert ev["recall_any_label"] == 0.0
