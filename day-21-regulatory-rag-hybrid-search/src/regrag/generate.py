"""Citation-grounded answer generation: extractive (offline default) and Claude (when ANTHROPIC_API_KEY is set).

Both generators see the same context: the citable units (clause id + the clause text that is inside the retrieved
chunk) of the top re-ranked chunks. Both are wrapped by the same out-of-scope guardrail: if the best re-ranker
probability is below ``tau`` (tuned on dev), the system refuses instead of answering.

Citation format: ``[§4.2(b)]`` after every claim.

Claude configuration (environment variables)
  ANTHROPIC_API_KEY     enables the Claude generator (otherwise the extractive fallback is used)
  ANTHROPIC_MODEL       model id (default ``claude-opus-5-5``)
  ANTHROPIC_EFFORT      output_config.effort (default ``low``: short grounded answers are a routine task)
  ANTHROPIC_FALLBACKS   "1" (default) enables server-side refusal fallbacks, "0" disables
  LLM_MAX_QUESTIONS     cap on test questions sent to the API (default 80)
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

from .chunking import Chunk
from .corpus import Clause
from .rerank import PASSAGE_TYPE_PATTERNS, Scored, passage_types, question_type
from .retrieval import tokenize

CITATION_RE = re.compile(r"\[(§\d+\.\d+\([a-z]+\))\]")
DEFAULT_MODEL = "claude-opus-5-5"
REFUSAL_TOKEN = "INSUFFICIENT_CONTEXT"


@dataclass
class Answer:
    qid: str
    text: str
    citations: list[str]
    refused: bool
    context: dict[str, str]  # clause id -> unit text the generator was shown
    generator: str
    top_prob: float
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    notes: list[str] = field(default_factory=list)


def units_for_chunk(
    chunk: Chunk, clauses: dict[str, Clause], offsets: dict[str, tuple[int, int]], doc: str
) -> list[tuple[str, str]]:
    """The citable units inside a chunk: (clause id, the part of that clause's text that the chunk contains)."""
    out = []
    for cid in chunk.clause_ids:
        cs, ce = offsets[cid]
        if chunk.chunk_id == cid:  # structure chunk: exactly one whole clause
            out.append((cid, clauses[cid].text))
            continue
        s, e = max(cs, chunk.start), min(ce, chunk.end)
        if e > s:
            out.append((cid, doc[s:e]))
    return out


def build_context(
    ranked: list[Scored], chunks: dict[str, Chunk], clauses: dict[str, Clause], offsets, doc: str, n: int = 5
) -> list[tuple[str, str, float]]:
    """Units of the top-n chunks as (clause id, text, chunk probability), de-duplicated by clause id (longest text)."""
    seen: dict[str, tuple[str, str, float]] = {}
    for s in ranked[:n]:
        for cid, text in units_for_chunk(chunks[s.chunk_id], clauses, offsets, doc):
            if cid not in seen or len(text) > len(seen[cid][1]):
                seen[cid] = (cid, text, max(s.prob, seen.get(cid, ("", "", 0.0))[2]))
    return list(seen.values())


def _fact(text: str, qtype: str | None) -> str | None:
    if qtype is None:
        return None
    m = re.search(PASSAGE_TYPE_PATTERNS[qtype], text)
    if not m:
        return None
    if qtype == "ROLE":
        r = re.search(r"by (the [A-Za-z ]+?) at least (annually|semi-annually|quarterly|every 24 months)", m.group(0))
        return f"{r.group(1)}, at least {r.group(2)}" if r else m.group(0)
    return m.group(0)


def _best_unit(question: str, units: list[tuple[str, str]]) -> tuple[str, str]:
    qtype = question_type(question)
    q = set(tokenize(question))

    def key(u: tuple[str, str]) -> tuple[int, float]:
        return (1 if qtype in passage_types(u[1]) else 0, len(q & set(tokenize(u[1]))) / max(1, len(q)))

    return max(units, key=key)


class ExtractiveGenerator:
    """Offline fallback: quote the best-matching clause from the top chunk and lift the asked-for fact out of it."""

    name = "extractive"

    def __init__(self, tau: float, second_ratio: float = 0.5) -> None:
        self.tau, self.second_ratio = tau, second_ratio

    def answer(self, qid: str, question: str, ranked: list[Scored], chunks, clauses, offsets, doc) -> Answer:
        t0 = time.perf_counter()
        context = {cid: text for cid, text, _ in build_context(ranked, chunks, clauses, offsets, doc)}
        top = ranked[0].prob if ranked else 0.0
        if top < self.tau:
            return Answer(
                qid,
                f"I cannot answer this from the regulation: no passage is relevant enough (score {top:.2f} < {self.tau:.2f}).",
                [],
                True,
                context,
                self.name,
                top,
                (time.perf_counter() - t0) * 1000,
            )
        cid, unit = _best_unit(question, units_for_chunk(chunks[ranked[0].chunk_id], clauses, offsets, doc))
        fact = _fact(unit, question_type(question))
        parts = []
        if fact:
            parts.append(f"Answer: {fact} [{cid}].")
        parts.append(f'Source: "{unit.strip()}" [{cid}].')
        cites = [cid]
        if len(ranked) > 1 and ranked[1].prob >= self.second_ratio * top:  # independent of tau, so the guardrail only gates
            cid2, unit2 = _best_unit(question, units_for_chunk(chunks[ranked[1].chunk_id], clauses, offsets, doc))
            if cid2 != cid and question_type(question) in passage_types(unit2):
                parts.append(f'Also relevant: "{unit2.strip()}" [{cid2}].')
                cites.append(cid2)
        return Answer(qid, " ".join(parts), cites, False, context, self.name, top, (time.perf_counter() - t0) * 1000)


SYSTEM_PROMPT = f"""You answer compliance questions using only the numbered regulation passages provided.
Rules:
- Answer in one to three short sentences.
- End every sentence with the citation(s) of the passage(s) that support it, written exactly as labelled, e.g. [§4.2(b)].
- State only facts that appear in the cited passages. Do not add advice, caveats or outside knowledge.
- If the passages do not answer the question, reply with exactly {REFUSAL_TOKEN} and nothing else."""


class ClaudeGenerator:  # network path; unit-tested with a fake client
    name = "claude"

    def __init__(self, tau: float, client=None) -> None:
        self.tau = tau
        self.model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.effort = os.environ.get("ANTHROPIC_EFFORT", "low")
        self.fallbacks = os.environ.get("ANTHROPIC_FALLBACKS", "1") == "1"
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=4)  # SDK retries 408/409/429/5xx and connection errors
        self.client = client

    def _create(self, system: str, user: str, max_tokens: int = 1024):
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.effort},
        }
        if self.fallbacks:
            return self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        return self.client.messages.create(**kwargs)

    @staticmethod
    def _text(resp) -> str:
        return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text").strip()

    def answer(self, qid: str, question: str, ranked: list[Scored], chunks, clauses, offsets, doc) -> Answer:
        t0 = time.perf_counter()
        ctx = build_context(ranked, chunks, clauses, offsets, doc)
        context = {cid: text for cid, text, _ in ctx}
        top = ranked[0].prob if ranked else 0.0
        if top < self.tau:  # guardrail runs before any API call
            return Answer(
                qid,
                "I cannot answer this from the regulation: no passage is relevant enough.",
                [],
                True,
                context,
                self.name,
                top,
                (time.perf_counter() - t0) * 1000,
            )
        passages = "\n".join(f"[{cid}] {text}" for cid, text, _ in ctx)
        user = f"Passages:\n{passages}\n\nQuestion: {question}"
        notes = []
        try:
            resp = self._create(SYSTEM_PROMPT, user)
        except Exception as exc:  # noqa: BLE001 - recorded, never raised: one failed call must not stop the eval
            return Answer(
                qid,
                "",
                [],
                True,
                context,
                self.name,
                top,
                (time.perf_counter() - t0) * 1000,
                notes=[f"error={type(exc).__name__}"],
            )
        stop = getattr(resp, "stop_reason", None)
        if stop in ("refusal", "max_tokens"):
            notes.append(f"stop_reason={stop}")
        text = self._text(resp)
        usage = getattr(resp, "usage", None)
        refused = text.strip() == REFUSAL_TOKEN or stop == "refusal" or not text
        cites = list(dict.fromkeys(CITATION_RE.findall(text)))
        return Answer(
            qid,
            text,
            cites,
            refused,
            context,
            self.name,
            top,
            (time.perf_counter() - t0) * 1000,
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            notes,
        )

    def hypothetical_clause(self, question: str) -> str:
        """HyDE: write a plausible regulation clause that would answer the question (used only as a dense query)."""
        system = "Write one sentence in the style of a regulation clause that would answer the user's question. Output only the sentence."
        try:
            return self._text(self._create(system, question, max_tokens=256))
        except Exception:  # noqa: BLE001
            return ""
