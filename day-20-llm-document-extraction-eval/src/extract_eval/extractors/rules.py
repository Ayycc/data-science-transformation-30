"""Offline extractors: a strict regex baseline and a rules+repair extractor (the documented offline fallback).

``RulesExtractor(mode="strict")``  - first-match regexes, US number format only, no OCR repair, no derivation.
``RulesExtractor(mode="rules")``   - label synonyms with fuzzy matching, OCR repair in numeric/date context,
                                     European number parsing, derived fields (term -> expiry, terms -> due date),
                                     entity-suffix normalisation and arithmetic cross-checks that drive confidence.

Raw confidences are heuristic (pattern tier x OCR-repair penalty x cross-check outcome); the evaluation
harness calibrates them with isotonic regression on a held-out dev split before routing.
"""

from __future__ import annotations

import re
import time
from datetime import date, timedelta
from typing import Any

from rapidfuzz import fuzz, process

from ..generator import LAWS, MONTHS, add_months
from ..pdf_text import DocText
from .base import ExtractionResult, detect_doc_type

DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8"})
WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "twelve": 12}
SUFFIX_CANON = {"incorporated": "Inc.", "corporation": "Corp.", "limited": "Ltd", "company": "Co."}
SUFFIX_CASE = {"GMBH": "GmbH", "PLC": "plc", "LLC": "LLC", "INC.": "Inc.", "LTD": "Ltd", "CORP.": "Corp.", "CO.": "Co."}
VOCAB = [m.lower() for m in MONTHS] + [
    "agreement",
    "effective",
    "purchaser",
    "vendor",
    "supplier",
    "customer",
    "between",
    "invoice",
    "total",
    "subtotal",
    "commencement",
    "expiry",
    "renewal",
    "automatically",
]

MONTH_RE = r"[A-Za-z]{3,10}"
DATE_PATTERNS = [
    ("iso", r"(\d{4})-(\d{2})-(\d{2})"),
    ("us", r"(\d{2})/(\d{2})/(\d{4})"),
    ("legal", rf"the (\d{{1,2}})(?:st|nd|rd|th) day of ({MONTH_RE}),? (\d{{4}})"),
    ("long", rf"({MONTH_RE}) (\d{{1,2}}), (\d{{4}})"),
    ("abbr", rf"(\d{{2}})-({MONTH_RE})-(\d{{4}})"),
    ("dmy", rf"(\d{{1,2}}) ({MONTH_RE}) (\d{{4}})"),
]
ANY_DATE = (
    r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|the \d{1,2}(?:st|nd|rd|th) day of [A-Za-z]{3,10},? \d{4}"
    r"|[A-Za-z]{3,10} \d{1,2}, \d{4}|\d{2}-[A-Za-z]{3,10}-\d{4}|\d{1,2} [A-Za-z]{3,10} \d{4})"
)
MONEY = r"((?:USD|EUR|GBP|[$€£])?\s?\d[\d.,]*\d(?:\s?(?:USD|EUR|GBP|€))?)"


class RulesExtractor:
    def __init__(self, mode: str = "rules") -> None:
        if mode not in {"rules", "strict"}:
            raise ValueError(mode)
        self.mode = mode
        self.name = "rules_repair" if mode == "rules" else "regex_strict"
        self._repairs = 0

    # ------------------------------------------------------------------ text normalisation
    def _normalise(self, text: str) -> str:
        flat = re.sub(r"[ \t]+", " ", text)
        if self.mode == "strict":
            return flat

        # re-join tokens split by OCR or line wrapping when the join is a known keyword ("Octo ber" -> "October")
        def join(m: re.Match[str]) -> str:
            a, b = m.group(1), m.group(2)
            if (a + b).lower() in VOCAB and a.lower() not in VOCAB:
                self._repairs += 1
                return a + b
            return m.group(0)

        flat = re.sub(r"\b([A-Za-z]{2,9}) ([a-z]{1,8})\b", join, flat)

        # OCR digit confusions inside numeric-looking tokens (2O24 -> 2024, l,234.56 -> 1,234.56)
        def fix(m: re.Match[str]) -> str:
            tok = m.group(0)
            if sum(c.isdigit() for c in tok) >= max(1, len(re.sub(r"[.,]", "", tok)) // 2) and any(c in "OolISB" for c in tok):
                self._repairs += 1
                return tok.translate(DIGIT_FIX)
            return tok

        flat = re.sub(r"(?<![A-Za-z])[0-9OlISB][0-9OlISB.,]*[0-9OlISB](?![A-Za-z])", fix, flat)
        # a date glued to the next word ("2024between") -> insert the lost space
        flat = re.sub(r"(\d{4})([a-z]{3,})", r"\1 \2", flat)
        return flat

    # ------------------------------------------------------------------ value parsers
    def _month(self, s: str) -> int | None:
        for i, m in enumerate(MONTHS):
            if s.lower() in (m.lower(), m[:3].lower()):
                return i + 1
        if self.mode == "rules":
            hit = process.extractOne(s.lower(), [m.lower() for m in MONTHS], scorer=fuzz.ratio, score_cutoff=75)
            if hit:
                self._repairs += 1
                return hit[2] + 1
        return None

    def parse_date(self, s: str) -> date | None:
        for style, pat in DATE_PATTERNS:
            m = re.fullmatch(pat, s.strip()) or re.search(pat, s)
            if not m:
                continue
            g = m.groups()
            try:
                if style == "iso":
                    return date(int(g[0]), int(g[1]), int(g[2]))
                if style == "us":
                    return date(int(g[2]), int(g[0]), int(g[1]))
                if style == "long":
                    mo = self._month(g[0])
                    return date(int(g[2]), mo, int(g[1])) if mo else None
                if style in ("legal", "dmy", "abbr"):
                    mo = self._month(g[1])
                    return date(int(g[2]), mo, int(g[0])) if mo else None
            except ValueError:
                return None
        return None

    def parse_money(self, s: str) -> tuple[float | None, str | None]:
        cur = None
        for sym, code in (("$", "USD"), ("€", "EUR"), ("£", "GBP"), ("USD", "USD"), ("EUR", "EUR"), ("GBP", "GBP")):
            if sym in s:
                cur = code
                break
        num = re.sub(r"[^\d.,]", "", s)
        if not num:
            return None, cur
        if self.mode == "strict":
            try:
                return float(num.replace(",", "")), cur
            except ValueError:
                return None, cur
        last = max(num.rfind("."), num.rfind(","))
        if last >= 0 and len(num) - last - 1 == 2:
            whole, frac = num[:last], num[last + 1 :]
            whole = re.sub(r"[.,]", "", whole)
            return float(f"{whole or 0}.{frac}"), cur
        return float(re.sub(r"[.,]", "", num)), cur

    @staticmethod
    def _int_in(s: str) -> int | None:
        m = re.search(r"\((\d+)\)|(\d+)", s)
        return int(m.group(1) or m.group(2)) if m else None

    def canon_party(self, s: str) -> str:
        s = s.strip().strip(",;:\"' ").strip()
        if self.mode == "strict":
            return s
        m = re.match(r"^(.*?),?\s+(Incorporated|Corporation|Limited|Company)$", s, flags=re.I)
        if m:
            s = f"{m.group(1)} {SUFFIX_CANON[m.group(2).lower()]}"
        if s.isupper():
            words = s.split()
            s = " ".join(SUFFIX_CASE.get(w, w.capitalize()) for w in words)
        return s

    # ------------------------------------------------------------------ search helpers
    def _find(self, text: str, patterns: list[str], flags: int = 0) -> tuple[re.Match[str] | None, int]:
        for tier, p in enumerate(patterns):
            m = re.search(p, text, flags)
            if m:
                return m, tier
        return None, -1

    def _label(self, lines: list[str], labels: list[str]) -> tuple[str | None, float]:
        """Value after 'Label:' on a line; fuzzy label matching tolerates OCR damage in rules mode."""
        for ln in lines:
            if ":" not in ln:
                continue
            key, val = ln.split(":", 1)
            key = key.strip()
            for lab in labels:
                if key.lower() == lab.lower():
                    return val.strip(), 1.0
                if self.mode == "rules" and len(key) <= len(lab) + 3 and fuzz.ratio(key.lower(), lab.lower()) >= 85:
                    return val.strip(), 0.85
        return None, 0.0

    def _conf(self, base: float, before: int) -> float:
        """Penalise a value whose span needed OCR repair."""
        return round(base * (0.8 if self._repairs > before else 1.0) * self._doc_noise_factor, 4)

    # ------------------------------------------------------------------ main entry
    def extract(self, doc: DocText) -> ExtractionResult:
        t0 = time.perf_counter()
        doc_type = detect_doc_type(doc.text)
        self._repairs = 0
        flat = self._normalise(re.sub(r"\s*\n\s*", " ", doc.text))
        n_tok = max(1, len(flat.split()))
        self._doc_noise = self._repairs / n_tok
        flat_lines = [self._normalise(ln) for ln in doc.text.splitlines()]
        self._repairs = 0
        self._doc_noise_factor = 1.0 if self.mode == "strict" else max(0.55, 1.0 - 4.0 * self._doc_noise)
        if doc_type == "contract":
            fields, conf = self._contract(flat, flat_lines)
            items = None
        else:
            fields, conf, items = self._invoice(flat, flat_lines, doc)
        res = ExtractionResult(doc.doc_id, self.name, doc_type, fields, items, conf)
        res.latency_s = time.perf_counter() - t0
        res.notes.append(f"ocr_repairs_per_token={self._doc_noise:.3f}")
        return res

    # ------------------------------------------------------------------ contracts
    def _contract(self, flat: str, lines: list[str]) -> tuple[dict[str, Any], dict[str, float]]:
        f: dict[str, Any] = {}
        c: dict[str, float] = {}
        rules = self.mode == "rules"
        I = re.I  # noqa: E741

        r0 = self._repairs
        m, tier = self._find(
            flat,
            [
                r"(?:Agreement No\.?|Contract Ref:|reference:)\s*([A-Z]{2,4}-\d{4}-\d{3,4})",
                r"\b((?:MSA|SA|PA|CTR)-\d{4}-\d{4})\b",
            ],
        )
        f["contract_number"] = m.group(1) if m else None
        c["contract_number"] = self._conf([0.95, 0.75][tier] if m else 0.7, r0)

        # parties
        r0 = self._repairs
        sup = cus = None
        base = 0.0
        pats = [
            (r'by and between (.+?), an? [A-Za-z ]+? corporation \(\s?"?Supplier"?\s?\),? and (.+?) \(\s?"?Customer', ("s", "c")),
            (r'between (.+?) \(the "?Purchaser"?\) and (.+?) \(the "?Vendor', ("c", "s")),
        ]
        for p, order in pats:
            m = re.search(p, flat, I if rules else 0)
            if m:
                a, b = m.group(1), m.group(2)
                sup, cus = (a, b) if order == ("s", "c") else (b, a)
                base = 0.93
                break
        if sup is None:
            s_val, s_q = self._label(lines, ["Supplier", "Vendor"])
            c_val, c_q = self._label(lines, ["Buyer", "Customer", "Purchaser"])
            if s_val and c_val:
                sup, cus, base = s_val, c_val, 0.93 * min(s_q, c_q)
        if sup is None and rules:
            ms = re.search(r'([A-Z][\w&., ]+?) \((?:the )?"?(?:Supplier|Vendor|Seller)"?\)', flat)
            mc = re.search(r'([A-Z][\w&., ]+?) \((?:the )?"?(?:Customer|Purchaser|Buyer)"?\)', flat)
            if ms and mc:
                sup, cus, base = ms.group(1).split(" and ")[-1], mc.group(1).split(" and ")[-1], 0.6
        f["supplier"] = self.canon_party(sup) if sup else None
        f["customer"] = self.canon_party(cus) if cus else None
        c["supplier"] = c["customer"] = self._conf(base if sup else 0.3, r0)

        # effective date
        r0 = self._repairs
        if rules:
            m, tier = self._find(
                flat,
                [
                    rf"as of {ANY_DATE} \(the \"?Effective Date",
                    rf"Commencement Date: {ANY_DATE}",
                    rf"made effective {ANY_DATE}",
                    rf"[Ee]ffective(?: Date)?:? {ANY_DATE}",
                ],
            )
        else:
            m, tier = self._find(flat, [ANY_DATE])
        eff = self.parse_date(m.group(1)) if m else None
        base = ([0.95, 0.95, 0.93, 0.7][tier] if rules else 0.9) if eff else 0.2
        if eff is None and rules and (m := re.search(ANY_DATE, flat)):  # generic low-confidence fallback for unknown layouts
            eff, base = self.parse_date(m.group(1)), 0.35
        f["effective_date"] = eff.isoformat() if eff else None
        c["effective_date"] = self._conf(base, r0)

        # expiration (explicit or derived from the initial term)
        r0 = self._repairs
        m, tier = self._find(flat, [rf"continues until {ANY_DATE}", rf"Expiry Date: {ANY_DATE}", rf"Schedule A until {ANY_DATE}"])
        exp = self.parse_date(m.group(1)) if m else None
        base = 0.94 if exp else 0.0
        if exp is None and rules and eff is not None:
            mt = re.search(r"(?:initial term of|Initial Term:)\s*([\w-]+(?: \(\d+\))?) months", flat, I)
            if mt and (n := self._int_in(mt.group(1))):
                exp, base = add_months(eff, n), 0.9
        f["expiration_date"] = exp.isoformat() if exp else None
        c["expiration_date"] = self._conf(base if exp else 0.3, r0)

        # total value and currency
        r0 = self._repairs
        if rules:
            m, tier = self._find(
                flat,
                [
                    rf"total contract value of {MONEY}",
                    rf"Contract Value: {MONEY}",
                    rf"total commitment under this Agreement shall be a maximum of {MONEY}",
                ],
                I,
            )
        else:
            m, tier = self._find(flat, [rf"(?:value of|exceed|maximum of|Value:)\s*{MONEY}"])
        val, cur = self.parse_money(m.group(1)) if m else (None, None)
        f["total_value"] = val
        c["total_value"] = self._conf(0.93 if val is not None else 0.75, r0)
        if cur is None:
            mc = re.search(r"(USD|EUR|GBP|[$€£])", flat)
            cur = {"$": "USD", "€": "EUR", "£": "GBP"}.get(mc.group(1), mc.group(1)) if mc else None
        f["currency"] = cur
        c["currency"] = self._conf(0.95 if cur else 0.2, r0)

        # payment terms
        r0 = self._repairs
        f["payment_terms_days"], c["payment_terms_days"] = self._terms(flat)
        c["payment_terms_days"] = self._conf(c["payment_terms_days"], r0)

        # renewal
        r0 = self._repairs
        if rules:
            no = re.search(r"not renew automatically|mutual written agreement only|Auto-Renewal: No", flat, I)
            yes = re.search(r"automatically renew|renews automatically|Auto-Renewal: Yes", flat, I)
            auto = bool(yes) and not no
            base = 0.95 if (yes or no) else 0.55
        else:
            auto = "automatic" in flat.lower()
            base = 0.9
        f["auto_renewal"] = auto
        c["auto_renewal"] = self._conf(base, r0)
        r0 = self._repairs
        term = None
        if auto or not rules:
            mt = re.search(r"(?:successive|additional) ([\w-]+(?: \(\d+\))?)[ -](year|month)|Yes - (\d+) month", flat, I)
            if mt:
                if mt.group(3):
                    term = int(mt.group(3))
                else:
                    raw = mt.group(1)
                    n = self._int_in(raw) or WORD_NUM.get(raw.split()[0].lower())
                    if n:
                        term = n * (12 if mt.group(2).lower() == "year" else 1)
        f["renewal_term_months"] = term
        c["renewal_term_months"] = self._conf(0.92 if (term or not auto) else 0.35, r0)
        r0 = self._repairs
        notice = None
        if rules:
            if auto:
                mn = re.search(
                    r"at least ([\w-]+ \(\d+\)|\d+) days'? prior|no later than (\d+) days|Non-Renewal Notice: (\d+) days", flat, I
                )
                if mn:
                    notice = self._int_in(next(g for g in mn.groups() if g))
        else:
            mn = re.search(r"(\d+)\)? days'? (?:prior )?(?:written )?notice", flat)
            notice = int(mn.group(1)) if mn else None
        f["renewal_notice_days"] = notice
        c["renewal_notice_days"] = self._conf(0.92 if (notice or not auto) else 0.35, r0)

        # governing law
        r0 = self._repairs
        m = re.search(
            r"laws of (?:the )?(?:State of |Province of |Federal Republic of |Republic of )?([A-Z][A-Za-z ]+?)[.,;]", flat
        )
        law_raw = m.group(1) if m else None
        if law_raw is None:
            law_raw, _ = self._label(lines, ["Governing Law"])
        law = None
        if law_raw:
            if rules:
                hit = process.extractOne(law_raw, LAWS, scorer=fuzz.WRatio, score_cutoff=80)
                law = hit[0] if hit else law_raw.strip()
            else:
                law = law_raw.strip()
        f["governing_law"] = law
        c["governing_law"] = self._conf(0.94 if law else 0.65, r0)
        return f, c

    def _terms(self, flat: str) -> tuple[int | None, float]:
        if self.mode == "strict":
            m = re.search(r"[Nn]et (\d+)", flat)
            return (int(m.group(1)), 0.9) if m else (None, 0.6)
        m = re.search(r"\b[Nn]et (\d+)|within ([\w-]+ \(\d+\)|\d+) days(?! of the effective)|Payment Terms: Net (\d+)", flat)
        if m:
            g = next(x for x in m.groups() if x)
            return self._int_in(g), 0.93
        if re.search(r"upon receipt|on receipt", flat, re.I):
            return 0, 0.9
        return None, 0.7

    # ------------------------------------------------------------------ invoices
    def _invoice(
        self, flat: str, lines: list[str], doc: DocText
    ) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
        f: dict[str, Any] = {}
        c: dict[str, float] = {}
        rules = self.mode == "rules"

        r0 = self._repairs
        m, tier = self._find(
            flat,
            [
                r"Invoice No:?\s*([A-Z]{2,3}[-/]\d{4}[-/]\d{5})",
                r"Invoice #\s*([A-Z]{2,3}[-/]\d{4}[-/]\d{5})",
                r"Ref:\s*([A-Z]{2,3}[-/]\d{4}[-/]\d{5})",
                r"\b((?:INV|IN|SI)[-/]\d{4}[-/]\d{5})\b",
            ],
        )
        f["invoice_number"] = m.group(1) if m else None
        c["invoice_number"] = self._conf([0.96, 0.96, 0.94, 0.7][tier] if m else 0.2, r0)
        r0 = self._repairs
        m = re.search(r"\b(PO-\d{5})\b", flat)
        f["po_number"] = m.group(1) if m else None
        c["po_number"] = self._conf(0.95 if m else 0.75, r0)

        r0 = self._repairs
        sup, sq = self._label(lines, ["From"])
        if sup is None:
            m = re.search(r"Statement & Invoice from (.+?)(?: To:|$)", flat)
            sup, sq = (m.group(1), 1.0) if m else (None, 0.0)
        if sup is None and lines:
            sup, sq = lines[0], 0.8  # letterhead
        cus, cq = self._label(lines, ["Bill To", "Customer", "To"])
        f["supplier"] = self.canon_party(sup) if sup else None
        f["customer"] = self.canon_party(cus) if cus else None
        c["supplier"] = self._conf(0.93 * sq if sup else 0.2, r0)
        c["customer"] = self._conf(0.93 * cq if cus else 0.2, r0)

        r0 = self._repairs
        if rules:
            m, tier = self._find(flat, [rf"Invoice Date: {ANY_DATE}", rf"(?<!Order )Date: {ANY_DATE}", rf"Issued {ANY_DATE}"])
        else:
            m, tier = self._find(flat, [rf"Date: {ANY_DATE}", ANY_DATE])
        inv = self.parse_date(m.group(1)) if m else None
        base = 0.95 if inv else 0.2
        if inv is None and rules and (m := re.search(ANY_DATE, flat)):  # generic low-confidence fallback for unknown layouts
            inv, base = self.parse_date(m.group(1)), 0.35
        f["invoice_date"] = inv.isoformat() if inv else None
        c["invoice_date"] = self._conf(base, r0)

        r0 = self._repairs
        f["payment_terms_days"], tc = self._terms(flat)
        c["payment_terms_days"] = self._conf(tc, r0)
        r0 = self._repairs
        m, tier = self._find(flat, [rf"Due Date: {ANY_DATE}", rf"Payment Due: {ANY_DATE}", rf"pay by {ANY_DATE}"])
        due = self.parse_date(m.group(1)) if m else None
        base = 0.95 if due else 0.0
        if due is None and rules and inv is not None and f["payment_terms_days"] is not None:
            due, base = inv + timedelta(days=f["payment_terms_days"]), 0.88
        f["due_date"] = due.isoformat() if due else None
        c["due_date"] = self._conf(base if due else 0.2, r0)

        r0 = self._repairs
        cur = None
        val, _ = self._label(lines, ["Currency"])
        if val and val[:3] in ("USD", "EUR", "GBP"):
            cur = val[:3]
        amounts: dict[str, float | None] = {}
        pats = {
            "subtotal": [rf"(?:Subtotal:|Net amount|Sub-total)\s*{MONEY}"],
            "tax": [rf"(?:VAT|Sales Tax|MwSt|GST|Tax exempt)\s*\(?[\d.,]*%?\)?:?\s*{MONEY}"],
            "total": [rf"(?:Total Due:|Invoice total|Amount Due:)\s*{MONEY}"]
            if rules
            else [rf"Total[^0-9]*{MONEY}", rf"Amount Due:\s*{MONEY}"],
        }
        for k, ps in pats.items():
            m, _ = self._find(flat, ps, re.I if rules else 0)
            v, cc = self.parse_money(m.group(1)) if m else (None, None)
            if v is None and rules and k == "total":  # generic fallback: last amount after the word "total"
                hits = re.findall(rf"total\b[^0-9$€£]{{0,12}}{MONEY}", flat, re.I)
                v, cc = self.parse_money(hits[-1]) if hits else (None, None)
            amounts[k] = v
            cur = cur or cc
        if cur is None:
            mc = re.search(r"(USD|EUR|GBP|[$€£])", flat)
            cur = {"$": "USD", "€": "EUR", "£": "GBP"}.get(mc.group(1), mc.group(1)) if mc else None
        f["currency"] = cur
        c["currency"] = self._conf(0.95 if cur else 0.2, r0)

        r0 = self._repairs
        items = self._line_items(flat, lines, doc)
        item_sum = round(sum(i["amount"] for i in items), 2) if items else None
        st, tx, tt = amounts["subtotal"], amounts["tax"], amounts["total"]
        arith_ok = st is not None and tx is not None and tt is not None and abs(st + tx - tt) <= 0.011
        items_ok = item_sum is not None and st is not None and abs(item_sum - st) <= 0.011
        if rules and not arith_ok:
            # deterministic repair: trust the two values that agree with the line items
            if items_ok and tx is not None and (tt is None or abs(st + tx - tt) > 0.011):
                tt = round(st + tx, 2)
            elif tt is not None and tx is not None and item_sum is not None and abs(item_sum + tx - tt) <= 0.011:
                st = item_sum
        arith_ok2 = st is not None and tx is not None and tt is not None and abs(st + tx - tt) <= 0.011
        f["subtotal"], f["tax"], f["total"] = st, tx, tt
        if self.mode == "strict":
            for k in ("subtotal", "tax", "total"):
                c[k] = 0.9 if f[k] is not None else 0.3
        else:
            q = 0.98 if (arith_ok and items_ok) else (0.8 if arith_ok2 else 0.45)
            for k in ("subtotal", "tax", "total"):
                c[k] = self._conf(q if f[k] is not None else 0.2, r0)
        ok_items = sum(abs(round(i["quantity"] * i["unit_price"], 2) - i["amount"]) <= 0.011 for i in items)
        c["line_items"] = (
            self._conf((0.97 if items_ok else 0.5) * (ok_items / len(items)) if items else 0.1, r0)
            if rules
            else (0.9 if items else 0.3)
        )
        return f, c, items

    def _line_items(self, flat: str, lines: list[str], doc: DocText) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        num = r"([\d.,OlISB]*\d[\d.,OlISB]*)" if self.mode == "rules" else r"([\d.,]+)"

        def mk(desc: str, q: str, p: str, a: str) -> dict[str, Any] | None:
            if self.mode == "rules":
                q, p, a = (x.translate(DIGIT_FIX) for x in (q, p, a))
            try:
                qty = float(q.replace(",", "")) if self.mode == "strict" else self.parse_money(q)[0]
                price, amt = self.parse_money(p)[0], self.parse_money(a)[0]
            except ValueError:
                return None
            if qty is None or price is None or amt is None:
                return None
            return {"description": re.sub(r"\s+", " ", desc).strip(), "quantity": qty, "unit_price": price, "amount": amt}

        for t in doc.tables:
            if t and len(t[0]) >= 4 and "desc" in (t[0][0] or "").lower().replace(" ", ""):
                for row in t[1:]:
                    if len(row) >= 4 and (it := mk(row[0], row[1], row[2], row[3])):
                        items.append(it)
        if items:
            return items
        header_seen = False
        for ln in lines:
            if re.match(r"\s*Item\s+Qty", ln):
                header_seen = True
                continue
            m = re.match(rf"^\s*{num} x (.+?) @ {num} = {num}\s*$", ln)
            if m:
                if it := mk(m.group(2), m.group(1), m.group(3), m.group(4)):
                    items.append(it)
                continue
            if header_seen:
                m = re.match(rf"^(.+?)\s+{num}\s+{num}\s+{num}\s*$", ln)
                if m and not re.match(r"(Net amount|Invoice total)", ln):
                    if it := mk(m.group(1), m.group(2), m.group(3), m.group(4)):
                        items.append(it)
                elif ln.lower().startswith(("net amount", "net arnount")):
                    header_seen = False
        return items
