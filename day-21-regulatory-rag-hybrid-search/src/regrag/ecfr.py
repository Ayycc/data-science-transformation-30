"""Public-domain path: US eCFR text (16 CFR Part 312, the COPPA Rule) with an automatically built cloze gold set.

``load_public()`` tries the eCFR versioner API with a short timeout. If it is unreachable (the build environment's
proxy blocked it), the caller falls back to the SYNTHETIC generator and records which path was used.

Gold set on the public path: for every paragraph that states a number (days, years, dollars, ...), the number is
masked and the paragraph's sentence becomes the question; the paragraph is the supporting passage. This is
weaker than hand-written questions (the question shares most words with the passage), so it mainly tests lexical
retrieval. It is only used when the public source is reachable.
"""

from __future__ import annotations

import random
import re
import urllib.request
import xml.etree.ElementTree as ET

from .corpus import LETTERS, Clause, GoldQuestion

ECFR_URL = "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-16.xml?part=312"
NUMERIC_FACT = re.compile(r"\b(\d[\d,]*\s+(?:business |calendar )?(?:days?|years?|months?|hours?)|\$[\d,]+)")


def fetch_ecfr(date: str = "2025-01-01", timeout: float = 10.0) -> str | None:
    try:
        with urllib.request.urlopen(ECFR_URL.format(date=date), timeout=timeout) as r:  # noqa: S310 - fixed https URL
            if r.status != 200:
                return None
            return r.read().decode("utf-8")
    except Exception:  # noqa: BLE001 - any network failure means "use the synthetic fallback"
        return None


def parse_ecfr(xml_text: str) -> list[Clause]:
    root = ET.fromstring(xml_text)
    clauses: list[Clause] = []
    for div in root.iter("DIV8"):
        sid = (div.get("N") or "").strip()
        if not re.fullmatch(r"\d+\.\d+", sid):
            continue
        head = " ".join((div.findtext("HEAD") or "").split())
        heading = re.sub(r"^§\s*\d+\.\d+\s*", "", head).rstrip(".")
        part = int(sid.split(".")[0])
        i = 0
        for p in div.iter("P"):
            text = " ".join("".join(p.itertext()).split())
            if len(text) < 30:
                continue
            m = re.match(r"^\(([a-z])\)\s*", text)
            letter = m.group(1) if m else LETTERS[i % len(LETTERS)] + ("" if i < len(LETTERS) else LETTERS[i // len(LETTERS) - 1])
            body = text[m.end() :] if m else text
            cid = f"§{sid}({letter})"
            if any(c.clause_id == cid for c in clauses):
                cid = f"§{sid}({letter}{LETTERS[i % len(LETTERS)]})"
            fact = NUMERIC_FACT.search(body)
            clauses.append(Clause(cid, sid, part, heading, letter, body, "public", heading, fact.group(1) if fact else ""))
            i += 1
    return clauses


def cloze_questions(clauses: list[Clause], seed: int = 0, test_fraction: float = 0.5) -> list[GoldQuestion]:
    rng = random.Random(seed)
    sections = sorted({c.section_id for c in clauses})
    rng.shuffle(sections)
    test = set(sections[: int(round(len(sections) * test_fraction))])
    out = []
    for c in clauses:
        if not c.answer:
            continue
        sent = next((s for s in re.split(r"(?<=[.;])\s+", c.text) if c.answer in s), c.text)
        q = sent.replace(c.answer, "what amount or period", 1).rstrip(".;") + "?"
        out.append(
            GoldQuestion(
                f"Q{len(out) + 1:04d}",
                q,
                "cloze",
                "public",
                [c.clause_id],
                c.answer,
                True,
                "test" if c.section_id in test else "dev",
            )
        )
    return out


def load_public() -> tuple[list[Clause], list[GoldQuestion]] | None:
    xml_text = fetch_ecfr()
    if not xml_text:
        return None
    try:
        clauses = parse_ecfr(xml_text)
    except ET.ParseError:
        return None
    qs = cloze_questions(clauses)
    if len(clauses) < 20 or len(qs) < 100:  # the prompt requires >= 100 gold questions
        return None
    return clauses, qs
