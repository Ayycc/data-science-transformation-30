"""Deterministic, rules-based planner: English question -> QuerySpec | refusal | clarification.

This is the documented offline fallback used when ANTHROPIC_API_KEY is not set. It only
knows what the semantic layer tells it (metric/dimension synonyms and dimension values),
so it cannot invent a metric: anything it cannot ground is refused or clarified.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any

from .semantic import QuerySpec, SemanticLayer

WRITE_RE = re.compile(
    r"\b(delete|drop|truncate|update|insert|alter|create\s+(?:a\s+)?(?:table|view|index)|grant|revoke|"
    r"remove|overwrite|modify|rename|merge\s+into|wipe|purge)\b"
)
PII_RE = re.compile(
    r"\b(e-?mails?|e-?mail addresses?|phone(?: numbers?)?|contact names?|contact details|"
    r"personal (?:data|details|information)|home address(?:es)?|ssn|social security)\b"
)
INJECTION_RE = re.compile(r"\b(ignore|disregard|forget)\b.{0,30}\b(instructions|rules|guardrails|prompt)\b")
UNDEFINED_TERMS = {
    "net promoter": "NPS",
    "nps": "NPS",
    "gross margin": "gross margin",
    "margin": "margin",
    "profit": "profit",
    "ebitda": "EBITDA",
    "lifetime value": "customer lifetime value",
    "ltv": "LTV",
    "cac": "CAC",
    "acquisition cost": "customer acquisition cost",
    "pipeline": "sales pipeline",
    "quota": "quota attainment",
    "headcount": "headcount",
    "forecast": "forecast",
    "cost": "cost",
    "costs": "costs",
    "expenses": "expenses",
}
VAGUE_RE = re.compile(
    r"\b(how are we doing|how is (?:the )?business|numbers|performance|kpis?|metrics|dashboard|"
    r"overview|stats|statistics|update me|health)\b"
)
STOP_WORDS = {
    "in",
    "for",
    "at",
    "during",
    "of",
    "and",
    "with",
    "from",
    "on",
    "as",
    "over",
    "since",
    "the",
    "?",
}
MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ],
        start=1,
    )
}
VALUE_ALIASES = {
    "region": {
        "north america": "NA",
        "europe": "EMEA",
        "latin america": "LATAM",
        "asia pacific": "APAC",
    },
    "segment": {
        "mid market": "Mid-Market",
        "midmarket": "Mid-Market",
        "small business": "SMB",
    },
}


@dataclass
class Plan:
    kind: str  # "query" | "refuse" | "clarify"
    spec: QuerySpec | None = None
    code: str | None = None
    message: str = ""
    notes: list[str] = field(default_factory=list)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-']*|\?", text.lower())


def _find_phrase(tokens: list[str], phrase: list[str], max_gap: int = 1) -> tuple[int, int] | None:
    """Find phrase tokens in order, allowing up to max_gap extra tokens between them."""
    n = len(tokens)
    for start in range(n):
        if tokens[start] != phrase[0]:
            continue
        pos, ok = start, True
        for p in phrase[1:]:
            for j in range(pos + 1, min(n, pos + 2 + max_gap)):
                if tokens[j] == p:
                    pos = j
                    break
            else:
                ok = False
                break
        if ok:
            return start, pos
    return None


def fiscal_year_end(fy: int) -> dt.date:
    return dt.date(fy, 1, 31)


class RulePlanner:
    def __init__(self, layer: SemanticLayer, dimension_values: dict[str, list[str]]):
        self.layer = layer
        self.values = dimension_values
        self.metric_syn = sorted(
            ((s.lower().split(), m) for m, d in layer.metrics.items() for s in d.get("synonyms", [])),
            key=lambda x: -len(x[0]),
        )
        self.dim_syn = sorted(
            ((s.lower().split(), d) for d, dd in layer.dimensions.items() for s in dd.get("synonyms", [])),
            key=lambda x: -len(x[0]),
        )

    # ----------------------------------------------------------------- helpers
    def _match_metrics(self, tokens: list[str]) -> list[tuple[int, str]]:
        found: dict[str, int] = {}
        used: set[int] = set()
        for syn, metric in self.metric_syn:
            hit = _find_phrase(tokens, syn)
            if hit and not (set(range(hit[0], hit[1] + 1)) & used):
                used |= set(range(hit[0], hit[1] + 1))
                found.setdefault(metric, hit[0])
        return sorted(((p, m) for m, p in found.items()))

    def _exact_metric(self, words: list[str]) -> bool:
        return any(words == syn for syn, _ in self.metric_syn)

    def _match_dimension(self, words: list[str]) -> str | None:
        for syn, dim in self.dim_syn:
            if words[: len(syn)] == syn:
                return dim
        return None

    def _grouping(self, tokens: list[str]) -> tuple[list[str], list[str], dict[str, Any]]:
        """Return (dimensions, unknown_phrases, ranking)."""
        dims: list[str] = []
        unknown: list[str] = []
        ranking: dict[str, Any] = {}
        text = " ".join(tokens)
        # ranking: "which X had the highest ...", "top N Xs by ...", "bottom N ..."
        m = re.search(r"\b(top|bottom)\s+(\d+)\s+(.+)$", text)
        if m:
            words = m.group(3).split()
            d = self._match_dimension(words)
            if d:
                dims.append(d)
                ranking = {
                    "direction": "desc" if m.group(1) == "top" else "asc",
                    "limit": int(m.group(2)),
                }
        m = re.search(r"\bwhich\s+(.+?)\s+(?:had|has|have|was|is|did|got|generated)\b(.*)$", text)
        if m:
            d = self._match_dimension(m.group(1).split())
            rest = m.group(2)
            if d:
                dims.append(d)
                low = re.search(r"\b(lowest|least|fewest|smallest|worst|bottom)\b", rest)
                ranking = {"direction": "asc" if low else "desc", "limit": 1}
        # "by X", "per X", "for each X", "broken down by X", "split by X"
        for m in re.finditer(r"\b(?:by|per|each|across)\s+(.+)", text):
            words = []
            for w in m.group(1).split():
                if w in STOP_WORDS:
                    break
                words.append(w)
            if not words:
                continue
            if self._exact_metric(words):
                continue  # "top regions by revenue": an ordering, not a grouping
            d = self._match_dimension(words)
            if d:
                if d not in dims:
                    dims.append(d)
            else:
                unknown.append(" ".join(words))
        for adj, dim in (
            ("monthly", "calendar_month"),
            ("quarterly", "fiscal_quarter"),
        ):
            if adj in tokens and dim not in dims:
                dims.append(dim)
        return dims, unknown, ranking

    def _filters(self, text: str, tokens: list[str]) -> tuple[dict[str, list[Any]], dict[str, Any]]:
        filters: dict[str, list[Any]] = {}
        extra: dict[str, Any] = {}
        q = re.search(r"\bq([1-4])\s+(?:of\s+)?(?:fy|fiscal(?:\s+year)?)\s*(\d{4})\b", text) or re.search(
            r"\b(?:fy|fiscal(?:\s+year)?)\s*(\d{4})\s*-?\s*q([1-4])\b", text
        )
        if q:
            a, b = q.groups()
            qn, fy = (a, b) if len(a) == 1 else (b, a)
            filters["fiscal_quarter"] = [f"FY{fy}-Q{qn}"]
        else:
            fys = re.findall(r"\b(?:fy|fiscal(?:\s+year)?)\s*(\d{4})\b", text)
            if fys:
                end = re.search(
                    r"\b(?:end|close)\s+of\s+(?:fy|fiscal(?:\s+year)?)\s*(\d{4})\b",
                    text,
                )
                if end:
                    extra["as_of_fy"] = int(end.group(1))
                else:
                    filters["fiscal_year"] = sorted({int(y) for y in fys})
        cal = re.search(r"\bcalendar(?:\s+year)?\s*(\d{4})\b", text)
        if cal:
            filters["calendar_year"] = [int(cal.group(1))]
        mon = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(\d{4})\b", text)
        if mon:
            filters["calendar_month"] = [f"{mon.group(2)}-{MONTHS[mon.group(1)]:02d}"]
        if not filters and not extra:
            bare = re.search(r"\b(?:in|during)\s+(20\d{2})\b", text)
            if bare:
                filters["calendar_year"] = [int(bare.group(1))]
                extra["note"] = (
                    f"Interpreted '{bare.group(1)}' as calendar year {bare.group(1)}; say FY{bare.group(1)} for the fiscal year."
                )
        asof = re.search(r"\bas\s+of\s+(\d{4}-\d{2}-\d{2})\b", text)
        if asof:
            extra["as_of"] = asof.group(1)
        # categorical values
        for dim, vals in self.values.items():
            for v in vals:
                vl = v.lower()
                pat = r"(?<![a-z0-9\-])" + re.escape(vl) + r"(?![a-z0-9\-])"
                if dim == "plan":
                    pat += r"(?=\s+(?:plan|tier)s?\b)"
                if dim == "priority":
                    pat += r"(?=\s+(?:priority|tickets?|cases?)\b)"
                if dim == "region" and vl == "na":
                    pat = r"\bNA\b"
                    if not re.search(pat, self._raw):
                        continue
                    filters.setdefault(dim, []).append(v)
                    continue
                if re.search(pat, text):
                    filters.setdefault(dim, []).append(v)
            for alias, v in VALUE_ALIASES.get(dim, {}).items():
                if re.search(r"\b" + re.escape(alias) + r"\b", text) and v not in filters.get(dim, []):
                    filters.setdefault(dim, []).append(v)
        return filters, extra

    # ----------------------------------------------------------------- main entry
    def plan(self, question: str) -> Plan:
        self._raw = question
        text = " ".join(_tokens(question))
        tokens = text.split()
        if INJECTION_RE.search(text) and PII_RE.search(text):
            return Plan(
                "refuse",
                code="pii",
                message="Request tries to override instructions to reveal personal data; PII columns are blocked.",
            )
        if WRITE_RE.search(text):
            return Plan(
                "refuse",
                code="write_request",
                message="This assistant is read-only: it cannot modify, delete or create data.",
            )
        if PII_RE.search(text):
            return Plan(
                "refuse",
                code="pii",
                message="Personal contact data (names, e-mails, phone numbers) is blocked by policy.",
            )
        if INJECTION_RE.search(text):
            return Plan(
                "refuse",
                code="prompt_injection",
                message="Instructions embedded in the question are ignored; please ask an analytics question.",
            )
        for term, label in UNDEFINED_TERMS.items():
            if re.search(r"\b" + re.escape(term) + r"\b", text):
                return Plan(
                    "refuse",
                    code="undefined_metric",
                    message=f"'{label}' is not a governed metric in the semantic layer. Defined metrics: "
                    + ", ".join(self.layer.metrics),
                )
        metrics = [m for _, m in self._match_metrics(tokens)]
        if not metrics:
            if VAGUE_RE.search(text):
                return Plan(
                    "clarify",
                    code="ambiguous",
                    message="Which metric do you mean? Available: " + ", ".join(m["label"] for m in self.layer.list_metrics()),
                )
            return Plan(
                "refuse",
                code="undefined_metric",
                message="No governed metric matches this question. Defined metrics: " + ", ".join(self.layer.metrics),
            )
        dims, unknown, ranking = self._grouping(tokens)
        if unknown:
            return Plan(
                "refuse",
                code="undefined_dimension",
                message=f"Cannot break down by '{unknown[0]}': not a dimension in the semantic layer. Available: "
                + ", ".join(self.layer.dimensions),
            )
        filters, extra = self._filters(text, tokens)
        notes = [extra["note"]] if "note" in extra else []
        snapshot = any(self.layer.is_snapshot(m) for m in metrics)
        as_of = extra.get("as_of")
        if "as_of_fy" in extra:
            if snapshot:
                as_of = fiscal_year_end(extra["as_of_fy"]).isoformat()
            else:
                filters["fiscal_year"] = [extra["as_of_fy"]]
        if snapshot and "fiscal_year" in filters and len(filters["fiscal_year"]) == 1 and not dims:
            fy = filters.pop("fiscal_year")[0]
            as_of = fiscal_year_end(fy).isoformat()
            notes.append(f"Point-in-time metric: evaluated at the end of FY{fy} ({as_of}).")
        # drop a filter on the dimension we group by (e.g. "by region" + no value)
        spec = QuerySpec(metrics=metrics, dimensions=dims, filters=filters, as_of=as_of)
        if ranking:
            spec.order_by = [[metrics[0], ranking["direction"]]]
            spec.limit = ranking["limit"]
        return Plan("query", spec=spec, notes=notes)
