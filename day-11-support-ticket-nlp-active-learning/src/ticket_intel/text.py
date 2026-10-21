"""Text normalisation applied after redaction, before any model sees the ticket."""

from __future__ import annotations

import re
import unicodedata

_TAG_RE = re.compile(r"\[(EMAIL|PHONE|NAME)\]")
_ORDER_RE = re.compile(r"(?:#|\b(?:ord|inv)-)\d{3,}\b", re.I)
_MONEY_RE = re.compile(r"[$€£]?\s?\d+(?:[.,]\d{2})\b")
_NUM_RE = re.compile(r"\b\d+\b")
_REPEAT_RE = re.compile(r"([!?.])\1+")
_WS_RE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """NFKC, lowercase, typed placeholders for PII tags / ids / amounts / numbers, collapse noise."""
    t = unicodedata.normalize("NFKC", text)
    t = _TAG_RE.sub(lambda m: f" __{m.group(1).lower()}__ ", t)
    t = _ORDER_RE.sub(" __id__ ", t)
    t = _MONEY_RE.sub(" __amount__ ", t)
    t = _NUM_RE.sub(" __num__ ", t)
    t = t.replace("’", "'").lower()
    t = _REPEAT_RE.sub(r"\1", t)
    return _WS_RE.sub(" ", t).strip()


_TOKEN_RE = re.compile(r"__\w+__|[a-z0-9']+|[^\sa-z0-9']")
_BOILER_WORDS = set(
    "hi hello hey dear there support team good morning afternoon evening thanks thank you regards best cheers "
    "sincerely sent by my name is this i am i'm im here from the a an and or me at on to back can reach call "
    "text email number phone contact please sign signed".split()
)
_PII_TOKEN = re.compile(r"__(?:name|email|phone)__")


def strip_boilerplate(norm_text: str) -> str:
    """Remove introduction / contact / sign-off phrases around redacted PII (for topic discovery).

    Each redacted name/email/phone token is deleted together with the run of courtesy or contact
    words and punctuation directly before and after it ("thanks, __name__", "text me at __phone__").
    """
    toks = _TOKEN_RE.findall(norm_text)
    drop = [False] * len(toks)

    def is_boiler(t: str) -> bool:
        return t in _BOILER_WORDS or not t[0].isalnum() and not t.startswith("__")

    for i, t in enumerate(toks):
        if _PII_TOKEN.fullmatch(t):
            drop[i] = True
            j = i - 1
            while j >= 0 and (is_boiler(toks[j]) or _PII_TOKEN.fullmatch(toks[j])):
                drop[j] = True
                j -= 1
            j = i + 1
            while j < len(toks) and (is_boiler(toks[j]) or _PII_TOKEN.fullmatch(toks[j])):
                drop[j] = True
                j += 1
    kept = [t for t, d in zip(toks, drop, strict=True) if not d]
    out = re.sub(r"\s+([^\w\s])", r"\1", " ".join(kept)).strip()
    return out or norm_text
