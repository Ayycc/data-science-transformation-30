"""PII redaction: regex for emails and phone numbers, spaCy NER plus cue patterns for names."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .config import SPACY_MODEL

EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
# Optional country code, optional (area) group, then 2+ digit groups separated by space/dot/dash.
PHONE_RE = re.compile(
    r"(?<![\w#$€£-])"
    r"(?:\+\d{1,3}[\s.-]?)?"
    r"(?:\(\d{2,4}\)[\s.-]?)?"
    r"\d{1,5}(?:[\s.-]\d{2,8}){1,4}"
    r"(?![\w-])"
)
ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NAME_TOKEN = r"[A-Z][a-zà-ÿ'’-]+"
NAME_CUE_RE = re.compile(
    r"(?i:(?:\b(?:my name is|this is|i am|i'm|im|name:|regards,|thanks,|thank you,|cheers,|best,|"
    r"sincerely,|sent by|signed,?|hey,|hi,|hello,|dear)|--))\s+"
    rf"({_NAME_TOKEN}(?:\s{_NAME_TOKEN}){{0,2}})"
)
# "from X" is a weaker cue ("invoice from Ann Lee" vs "order from Leeds"): kept only if NER does not call X a place.
NAME_FROM_RE = re.compile(rf"(?i:\bfrom)\s+({_NAME_TOKEN}(?:\s{_NAME_TOKEN}){{0,2}})")
NAME_HERE_RE = re.compile(rf"(?:^|[,.!]\s+)({_NAME_TOKEN}\s{_NAME_TOKEN}) here\b")
# Capitalised words after a cue that are not names (to keep precision up).
NAME_STOP = {
    "I",
    "Im",
    "The",
    "A",
    "An",
    "My",
    "Your",
    "Our",
    "This",
    "That",
    "Please",
    "Support",
    "Premium",
    "Business",
    "Free",
    "Pro",
    "Family",
    "Gold",
    "Standard",
    "Enterprise",
    "Team",
    "Starter",
    "Order",
    "Invoice",
    "Account",
    "Customer",
    "Service",
    "Hi",
    "Hello",
    "Thanks",
    "Cheers",
    "Regards",
    "Best",
    # common non-English greetings / courtesy words that NER mistakes for people
    "Bonjour",
    "Hola",
    "Olá",
    "Ola",
    "Hallo",
    "Ciao",
    "Gracias",
    "Grazie",
    "Danke",
    "Merci",
    "Obrigado",
    "Obrigada",
    "Voraus",
    "Por",
    "Buenas",
    "Salut",
    "Gracias de antemano",
    "Grazie mille",
    "Danke im Voraus",
}

TAGS = {"EMAIL": "[EMAIL]", "PHONE": "[PHONE]", "NAME": "[NAME]"}


@dataclass
class Span:
    start: int
    end: int
    label: str
    weak: bool = False


def _load_nlp():
    import spacy

    try:
        return spacy.load(SPACY_MODEL, disable=["parser", "lemmatizer", "attribute_ruler", "tagger", "senter"])
    except OSError:
        return None


def regex_spans(text: str) -> list[Span]:
    spans = [Span(m.start(), m.end(), "EMAIL") for m in EMAIL_RE.finditer(text)]
    for m in PHONE_RE.finditer(text):
        digits = sum(ch.isdigit() for ch in m.group())
        if 7 <= digits <= 15 and not ISO_DATE_RE.fullmatch(m.group()) and not _overlaps(m.start(), m.end(), spans):
            spans.append(Span(m.start(), m.end(), "PHONE"))
    for rx in (NAME_CUE_RE, NAME_HERE_RE, NAME_FROM_RE):
        for m in rx.finditer(text):
            s, e = m.span(1)
            toks = m.group(1).split()
            while toks and toks[-1] in NAME_STOP:
                toks.pop()
                e = s + len(" ".join(toks))
            if toks and toks[0] not in NAME_STOP:
                spans.append(Span(s, e, "NAME", weak=rx is NAME_FROM_RE))
    return spans


_CAP_WORD_LEFT = re.compile(rf"({_NAME_TOKEN})\s$")
_CAP_WORD_RIGHT = re.compile(rf"^\s({_NAME_TOKEN})")


def _extend_name(text: str, s: int, e: int) -> tuple[int, int]:
    """Grow an NER PERSON span by one adjacent capitalised word on each side (partial-name fix)."""
    m = _CAP_WORD_LEFT.search(text[max(0, s - 30) : s])
    if m and m.group(1) not in NAME_STOP and s - len(m.group(0)) > 0:
        s -= len(m.group(0))
    m = _CAP_WORD_RIGHT.match(text[e : e + 30])
    if m and m.group(1) not in NAME_STOP:
        e += len(m.group(0))
    return s, e


def _overlaps(s: int, e: int, spans: Iterable[Span]) -> bool:
    return any(s < sp.end and sp.start < e for sp in spans)


def merge_spans(spans: Sequence[Span]) -> list[Span]:
    """Merge overlapping spans; the earliest-starting span's label wins, priority EMAIL > PHONE > NAME."""
    prio = {"EMAIL": 0, "PHONE": 1, "NAME": 2}
    out: list[Span] = []
    for sp in sorted(spans, key=lambda x: (x.start, -x.end)):
        if out and sp.start < out[-1].end:
            last = out[-1]
            label = last.label if prio[last.label] <= prio[sp.label] else sp.label
            out[-1] = Span(last.start, max(last.end, sp.end), label)
        else:
            out.append(Span(sp.start, sp.end, sp.label))
    return out


class Redactor:
    """Detect and mask PII. `use_ner=False` (or a missing spaCy model) gives a regex-only redactor."""

    def __init__(self, use_ner: bool = True, use_cues: bool = True) -> None:
        self.nlp = _load_nlp() if use_ner else None
        self.use_cues = use_cues

    @property
    def has_ner(self) -> bool:
        return self.nlp is not None

    def detect_many(self, texts: Sequence[str], batch_size: int = 512) -> list[list[Span]]:
        results = []
        docs = self.nlp.pipe(texts, batch_size=batch_size) if self.nlp is not None else (None for _ in texts)
        for text, doc in zip(texts, docs, strict=True):
            spans = regex_spans(text)
            if not self.use_cues:
                spans = [s for s in spans if s.label != "NAME"]
            if doc is not None:
                places = [(e.start_char, e.end_char) for e in doc.ents if e.label_ in ("GPE", "LOC", "NORP", "DATE")]
                persons = [(e.start_char, e.end_char) for e in doc.ents if e.label_ == "PERSON"]
                # a cue-matched "name" that NER calls a place/date (e.g. "order from Leeds") is dropped
                spans = [
                    sp
                    for sp in spans
                    if not sp.weak
                    or not any(a <= sp.start and sp.end <= b for a, b in places)
                    or any(sp.start < b and a < sp.end for a, b in persons)
                ]
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        s, e = ent.start_char, ent.end_char
                        # trim leading/trailing punctuation that NER sometimes includes
                        while s < e and not text[s].isalnum():
                            s += 1
                        while e > s and not text[e - 1].isalnum():
                            e -= 1
                        s, e = _extend_name(text, s, e)
                        if e > s and text[s:e] not in NAME_STOP and not any(ch.isdigit() or ch == "@" for ch in text[s:e]):
                            spans.append(Span(s, e, "NAME"))
            results.append(merge_spans(spans))
        return results

    def detect(self, text: str) -> list[Span]:
        return self.detect_many([text])[0]

    @staticmethod
    def apply(text: str, spans: Sequence[Span]) -> str:
        out = text
        for sp in sorted(spans, key=lambda x: x.start, reverse=True):
            out = out[: sp.start] + TAGS[sp.label] + out[sp.end :]
        return out

    def redact(self, text: str) -> str:
        return self.apply(text, self.detect(text))

    def redact_many(self, texts: Sequence[str]) -> tuple[list[str], list[list[Span]]]:
        spans = self.detect_many(texts)
        return [self.apply(t, s) for t, s in zip(texts, spans, strict=True)], spans


def _covered(gs: int, ge: int, text: str, pred: Sequence[Span], label: str | None = None) -> bool:
    """A gold span counts as redacted if every alphanumeric character in it is masked."""
    for i in range(gs, ge):
        if not text[i].isalnum():
            continue
        if not any(p.start <= i < p.end and (label is None or p.label == label) for p in pred):
            return False
    return True


def evaluate_redaction(texts: Sequence[str], gold: Sequence[str | list], pred: Sequence[Sequence[Span]]) -> dict:
    """Span recall (any label and same label) per type, char-level precision, and false-positive rate."""
    per: dict[str, dict[str, int]] = {}
    pred_chars = true_pred_chars = 0
    fp_spans = 0
    clean_tickets = clean_with_redaction = 0
    for text, g, p in zip(texts, gold, pred, strict=True):
        g = json.loads(g) if isinstance(g, str) else g
        gold_mask = np.zeros(len(text), dtype=bool)
        for s, e, lab in g:
            gold_mask[s:e] = True
            d = per.setdefault(lab, {"n": 0, "hit_any": 0, "hit_typed": 0})
            d["n"] += 1
            d["hit_any"] += _covered(s, e, text, p)
            d["hit_typed"] += _covered(s, e, text, p, lab)
        for sp in p:
            pred_chars += sp.end - sp.start
            true_pred_chars += int(gold_mask[sp.start : sp.end].sum())
            if not gold_mask[sp.start : sp.end].any():
                fp_spans += 1
        if not g:
            clean_tickets += 1
            clean_with_redaction += bool(p)
    n = sum(d["n"] for d in per.values())
    out = {
        "n_gold_spans": n,
        "recall_any_label": sum(d["hit_any"] for d in per.values()) / max(n, 1),
        "recall_typed": sum(d["hit_typed"] for d in per.values()) / max(n, 1),
        "char_precision": true_pred_chars / max(pred_chars, 1),
        "false_positive_spans": fp_spans,
        "clean_tickets": clean_tickets,
        "clean_ticket_false_redaction_rate": clean_with_redaction / max(clean_tickets, 1),
        "per_type": {
            k: {"n": v["n"], "recall_any_label": v["hit_any"] / v["n"], "recall_typed": v["hit_typed"] / v["n"]}
            for k, v in sorted(per.items())
        },
    }
    return out


MARK_RE = re.compile(r"\[\[(NAME|EMAIL|PHONE):(.+?)\]\]")


def parse_marked(line: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Parse '[[TYPE:value]]' markup into (plain text, gold spans)."""
    out, spans, pos = [], [], 0
    cur = 0
    for m in MARK_RE.finditer(line):
        out.append(line[pos : m.start()])
        cur += m.start() - pos
        spans.append((cur, cur + len(m.group(2)), m.group(1)))
        out.append(m.group(2))
        cur += len(m.group(2))
        pos = m.end()
    out.append(line[pos:])
    return "".join(out), spans


def load_handlabelled(path) -> tuple[list[str], list[list[tuple[int, int, str]]]]:
    texts, golds = [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        t, g = parse_marked(line)
        texts.append(t)
        golds.append(g)
    return texts, golds
