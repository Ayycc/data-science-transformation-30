"""Second-stage re-ranking of hybrid candidates.

Default (offline): a **learned feature re-ranker** - logistic regression over query-chunk pair features, trained
on the dev questions only. Features: BM25 and dense scores and ranks, the RRF score, exact query-term coverage,
heading coverage, soft (embedding MaxSim) coverage, and an answer-type match between what the question asks for
(a deadline, a retention period, a penalty, an approver, an exemption threshold) and what the chunk states.

Optional: ``RERANKER=cross-encoder`` uses a sentence-transformers ``CrossEncoder`` (``CROSS_ENCODER_MODEL``,
default ``cross-encoder/ms-marco-MiniLM-L-6-v2``) if the package and weights are available. Hugging Face downloads
were blocked in the build environment, so no reported number uses it.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .chunking import Chunk
from .retrieval import BM25Retriever, DenseRetriever, Hit, HybridRetriever, _words, tokenize

# What the question asks for. Checked in order; the first matching cue wins.
QUESTION_TYPE_CUES: list[tuple[str, str]] = [
    ("YEARS", r"\b(retain|retained|keep|kept|preserve|preserved|storage|disposal|years?)\b"),
    ("MONEY", r"\b(penalty|penalties|fine|fines|sanction|sanctions)\b"),
    ("ROLE", r"\b(who|whom|whose|officer|sign-off|signing off|approve|approved)\b"),
    ("COUNT", r"\b(exempt|exemption|excused|not apply|threshold|small firms|ignore)\b"),
    ("DAYS", r"\b(days?|deadline|quickly|soon|notify|inform|alert)\b"),
]
# What a passage states.
PASSAGE_TYPE_PATTERNS: dict[str, str] = {
    "DAYS": r"\b\d+ (?:business|calendar) days\b",
    "YEARS": r"\b\d+ years\b",
    "MONEY": r"\$[\d,]+",
    "ROLE": r"\breviewed and (?:approved|signed off|endorsed) by the [A-Za-z ]+? at least (?:annually|semi-annually|quarterly|every 24 months)",
    "COUNT": r"\bfewer than [\d,]+",
}


def question_type(question: str) -> str | None:
    q = question.lower()
    for t, pat in QUESTION_TYPE_CUES:
        if re.search(pat, q):
            return t
    return None


def passage_types(text: str) -> set[str]:
    return {t for t, pat in PASSAGE_TYPE_PATTERNS.items() if re.search(pat, text)}


FEATURES = ["bm25_norm", "bm25_rr", "dense_sim", "dense_rr", "rrf", "coverage", "heading_cov", "soft_cov", "type_match", "typed"]


@dataclass
class Scored:
    chunk_id: str
    prob: float
    rank: int


class FeatureReranker:
    name = "learned"

    def __init__(
        self, chunks: list[Chunk], bm25: BM25Retriever, dense: DenseRetriever, hybrid: HybridRetriever, depth: int = 30
    ) -> None:
        self.chunks = {c.chunk_id: c for c in chunks}
        self.bm25, self.dense, self.hybrid, self.depth = bm25, dense, hybrid, depth
        self.model = None
        self._chunk_tokens = {c.chunk_id: set(tokenize(c.text)) for c in chunks}
        self._heading_tokens = {c.chunk_id: set(tokenize(c.heading)) for c in chunks}
        self._chunk_types = {c.chunk_id: passage_types(c.text) for c in chunks}
        self._chunk_wordvecs: dict[str, np.ndarray | None] = {}

    def _wordvecs(self, text: str) -> np.ndarray | None:
        if self.dense.backend != "static":
            return None
        vecs = [v for w in _words(text) if (v := self.dense._word_vec(w)) is not None]
        return np.vstack(vecs) if vecs else None

    def candidates(self, question: str, extra_dense_text: str | None = None) -> list[Hit]:
        return self.hybrid.search(question, self.depth, extra_dense_text=extra_dense_text)

    def features(self, question: str, cands: list[Hit]) -> np.ndarray:
        q_toks = set(tokenize(question))
        bm = self.bm25.scores(question)
        bm_by_id = dict(zip(self.bm25.ids, bm, strict=True))
        bm_rank = {cid: r for r, cid in enumerate(np.array(self.bm25.ids)[np.argsort(-bm, kind="stable")])}
        ds = self.dense.scores(question)
        ds_by_id = dict(zip(self.dense.ids, ds, strict=True))
        ds_rank = {cid: r for r, cid in enumerate(np.array(self.dense.ids)[np.argsort(-ds, kind="stable")])}
        max_bm = max((bm_by_id[h.chunk_id] for h in cands), default=1.0) or 1.0
        qtype = question_type(question)
        qv = self._wordvecs(question)
        rows = []
        for h in cands:
            cid = h.chunk_id
            ctoks = self._chunk_tokens[cid]
            if qv is not None:
                if cid not in self._chunk_wordvecs:
                    self._chunk_wordvecs[cid] = self._wordvecs(self.chunks[cid].text)
                cv = self._chunk_wordvecs[cid]
                soft = float(np.mean(np.max(qv @ cv.T, axis=1))) if cv is not None else 0.0
            else:
                soft = 0.0
            types = self._chunk_types[cid]
            rows.append(
                [
                    bm_by_id[cid] / max_bm,
                    1.0 / (1 + bm_rank[cid]),
                    ds_by_id[cid],
                    1.0 / (1 + ds_rank[cid]),
                    h.score * 60,
                    len(q_toks & ctoks) / max(1, len(q_toks)),
                    len(q_toks & self._heading_tokens[cid]) / max(1, len(q_toks)),
                    soft,
                    1.0 if qtype in types else 0.0,
                    1.0 if types else 0.0,
                ]
            )
        return np.asarray(rows, dtype=float)

    def fit(self, questions: list[str], relevant: list[set[str]], seed: int = 0) -> FeatureReranker:
        xs, ys = [], []
        for q, rel in zip(questions, relevant, strict=True):
            cands = self.candidates(q)
            xs.append(self.features(q, cands))
            ys.extend(1 if h.chunk_id in rel else 0 for h in cands)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
        self.model.fit(np.vstack(xs), np.asarray(ys))
        return self

    def coefficients(self) -> dict[str, float]:
        lr = self.model[-1]
        return {f: round(float(c), 4) for f, c in zip(FEATURES, lr.coef_[0], strict=True)}

    def rerank(self, question: str, k: int = 10, extra_dense_text: str | None = None) -> list[Scored]:
        if self.model is None:
            raise RuntimeError("call fit() first")
        cands = self.candidates(question, extra_dense_text)
        probs = self.model.predict_proba(self.features(question, cands))[:, 1]
        order = sorted(range(len(cands)), key=lambda i: (-probs[i], cands[i].rank))[:k]
        return [Scored(cands[i].chunk_id, float(probs[i]), r) for r, i in enumerate(order)]


class CrossEncoderReranker:  # pragma: no cover - needs weights from the Hugging Face Hub
    name = "cross-encoder"

    def __init__(self, chunks: list[Chunk], hybrid: HybridRetriever, depth: int = 30) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"), device="cpu")
        self.chunks = {c.chunk_id: c for c in chunks}
        self.hybrid, self.depth = hybrid, depth

    def fit(self, *args, **kwargs) -> CrossEncoderReranker:
        return self

    def rerank(self, question: str, k: int = 10, extra_dense_text: str | None = None) -> list[Scored]:
        cands = self.hybrid.search(question, self.depth, extra_dense_text=extra_dense_text)
        logits = self.model.predict([(question, self.chunks[h.chunk_id].text) for h in cands])
        probs = [1 / (1 + math.exp(-float(x))) for x in logits]
        order = sorted(range(len(cands)), key=lambda i: -probs[i])[:k]
        return [Scored(cands[i].chunk_id, probs[i], r) for r, i in enumerate(order)]


def make_reranker(chunks: list[Chunk], bm25: BM25Retriever, dense: DenseRetriever, hybrid: HybridRetriever):
    """Cross-encoder if requested and available, else the learned feature re-ranker. Returns (reranker, name)."""
    if os.environ.get("RERANKER") == "cross-encoder":
        try:
            return CrossEncoderReranker(chunks, hybrid), "cross-encoder"
        except Exception:  # noqa: BLE001
            pass
    return FeatureReranker(chunks, bm25, dense, hybrid), "learned"
