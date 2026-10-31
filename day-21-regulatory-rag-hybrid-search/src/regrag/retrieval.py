"""Lexical (BM25), dense (in-process FAISS), and hybrid (reciprocal rank fusion) retrieval.

Dense backends (``DENSE_BACKEND``)
---------------------------------
* ``static`` (default) - pretrained 300-d static word vectors from spaCy's ``en_core_web_md`` package (installed
  from its GitHub release wheel, no Hugging Face download). A chunk or query vector is the IDF-weighted mean of
  its unit-normalised content-word vectors (a SIF-style sentence embedding), L2-normalised.
* ``lsa`` - TF-IDF over the chunk collection projected with truncated SVD. Needs no pretrained weights at all;
  used automatically if the spaCy model is not installed.
* ``sentence-transformers`` - optional, not in ``requirements.txt``; used only if the package and model weights
  (``DENSE_MODEL``, default ``sentence-transformers/all-MiniLM-L6-v2``) are available. Hugging Face downloads were
  blocked in the build environment, so no reported number uses it.

Every backend is searched with an exact inner-product FAISS index (``IndexFlatIP``) held in process memory.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .chunking import Chunk

STOPWORDS = set(
    """a an and are as at be by can do does for from has have how i if in into is it its of on or that the their
    there these this to was what when where which who whom whose why will with within must shall any all each
    under than not no need needs being been also may""".split()
)


def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower().replace(",", ""))
    return [_stem(t) for t in toks if t not in STOPWORDS]


@dataclass
class Hit:
    chunk_id: str
    score: float
    rank: int


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75) -> None:
        self.chunks = chunks
        self.ids = [c.chunk_id for c in chunks]
        self.docs = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.docs, k1=k1, b=b)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str | list[str]) -> np.ndarray:
        toks = tokenize(query) if isinstance(query, str) else query
        return np.asarray(self.bm25.get_scores(toks), dtype=float)

    def search(self, query: str | list[str], k: int = 50) -> list[Hit]:
        return _top(self.ids, self.scores(query), k)

    def expand_query(self, query: str, fb_docs: int = 3, fb_terms: int = 5) -> list[str]:
        """Query rewriting by pseudo-relevance feedback (RM3-style): add the highest tf-idf terms of the top
        ``fb_docs`` BM25 hits that are not already in the query."""
        q = tokenize(query)
        hits = self.search(q, fb_docs)
        weights: Counter[str] = Counter()
        for h in hits:
            doc = self.docs[self.ids.index(h.chunk_id)]
            tf = Counter(doc)
            for t, f in tf.items():
                if t not in q and not t.isdigit():
                    weights[t] += (f / len(doc)) * self.idf.get(t, 0.0)
        return q + [t for t, _ in weights.most_common(fb_terms)]


_SPACY_NLP = None


def load_static_vectors():
    """Return spaCy's en_core_web_md vocab (vectors only), or None if the model package is not installed."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        try:
            import spacy

            _SPACY_NLP = spacy.load(
                "en_core_web_md", exclude=["tok2vec", "tagger", "parser", "senter", "attribute_ruler", "lemmatizer", "ner"]
            )
        except Exception:  # noqa: BLE001 - missing package/model means "fall back to LSA"
            _SPACY_NLP = False
    return _SPACY_NLP or None


class DenseRetriever:
    name = "dense"

    def __init__(
        self, chunks: list[Chunk], backend: str | None = None, dim: int = 128, seed: int = 0, idf: dict[str, float] | None = None
    ) -> None:
        self.chunks = chunks
        self.ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        requested = backend or os.environ.get("DENSE_BACKEND", "static")
        self.backend = requested
        self._st = None
        self._nlp = None
        if requested == "sentence-transformers":
            try:  # optional: not in requirements.txt; needs model weights from the Hugging Face Hub
                from sentence_transformers import SentenceTransformer

                self._st = SentenceTransformer(
                    os.environ.get("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"), device="cpu"
                )
            except Exception:  # noqa: BLE001
                self.backend = "static"
        if self.backend == "static":
            self._nlp = load_static_vectors()
            if self._nlp is None:
                self.backend = "lsa"
        if self.backend == "static":
            if idf is None:
                df = Counter(t for c in texts for t in set(_words(c)))
                n = len(texts)
                idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
            self.idf = idf
            self._default_idf = 1.0  # words never seen in the corpus (question-only wording) get a low, neutral weight
            self._cache: dict[str, np.ndarray | None] = {}
        elif self.backend == "lsa":
            self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)
            x = self.vec.fit_transform(texts)
            n_comp = max(2, min(dim, x.shape[1] - 1, x.shape[0] - 1))
            self.svd = TruncatedSVD(n_components=n_comp, random_state=seed)
            self.svd.fit(x)
        emb = self.embed(texts)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.dim = emb.shape[1]

    def _word_vec(self, w: str) -> np.ndarray | None:
        if w not in self._cache:
            lex = self._nlp.vocab[w]
            v = lex.vector if lex.has_vector else None
            self._cache[w] = None if v is None else (v / (np.linalg.norm(v) + 1e-9)).astype("float32")
        return self._cache[w]

    def embed(self, texts: list[str]) -> np.ndarray:
        if self._st is not None:
            return _l2(np.asarray(self._st.encode(texts, normalize_embeddings=True), dtype="float32"))
        if self.backend == "lsa":
            return _l2(np.asarray(self.svd.transform(self.vec.transform(texts)), dtype="float32"))
        out = np.zeros((len(texts), 300), dtype="float32")
        for i, t in enumerate(texts):
            for w in _words(t):
                v = self._word_vec(w)
                if v is not None:
                    out[i] += v * self.idf.get(w, self._default_idf)
        return _l2(out)

    def scores(self, query: str) -> np.ndarray:
        q = self.embed([query])
        sims, idx = self.index.search(q, len(self.ids))
        out = np.zeros(len(self.ids), dtype=float)
        out[idx[0]] = sims[0]
        return out

    def search(self, query: str, k: int = 50) -> list[Hit]:
        q = self.embed([query])
        sims, idx = self.index.search(q, min(k, len(self.ids)))
        return [Hit(self.ids[i], float(s), r) for r, (i, s) in enumerate(zip(idx[0], sims[0], strict=True)) if i >= 0]


def _words(text: str) -> list[str]:
    """Content words for the static-vector encoder (no stemming: vectors are looked up by surface form)."""
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS]


def _l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _top(ids: list[str], scores: np.ndarray, k: int) -> list[Hit]:
    order = np.lexsort((np.arange(len(scores)), -scores))[:k]  # stable tie-break by index
    return [Hit(ids[i], float(scores[i]), r) for r, i in enumerate(order)]


def rrf(rankings: list[list[Hit]], k: int = 60, top: int = 50) -> list[Hit]:
    """Reciprocal rank fusion: score(d) = sum_r 1 / (k + rank_r(d) + 1)."""
    agg: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for h in ranking:
            agg[h.chunk_id] = agg.get(h.chunk_id, 0.0) + 1.0 / (k + h.rank + 1)
            first_seen.setdefault(h.chunk_id, len(first_seen))
    order = sorted(agg, key=lambda c: (-agg[c], first_seen[c]))[:top]
    return [Hit(c, agg[c], r) for r, c in enumerate(order)]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60, depth: int = 50, prf: bool = False) -> None:
        self.bm25, self.dense, self.rrf_k, self.depth, self.prf = bm25, dense, rrf_k, depth, prf

    def search(self, query: str, k: int = 50, extra_dense_text: str | None = None) -> list[Hit]:
        lex_q = self.bm25.expand_query(query) if self.prf else query
        rankings = [self.bm25.search(lex_q, self.depth), self.dense.search(query, self.depth)]
        if extra_dense_text:  # HyDE: also search with the embedding of a hypothetical answer clause
            rankings.append(self.dense.search(extra_dense_text, self.depth))
        return rrf(rankings, k=self.rrf_k, top=k)
