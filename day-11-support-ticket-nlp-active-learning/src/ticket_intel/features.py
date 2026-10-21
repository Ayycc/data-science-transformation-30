"""Feature spaces: sparse TF-IDF and dense sentence embeddings (with an offline fallback)."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from .config import SPACY_MODEL, ST_MODEL


def make_tfidf() -> FeatureUnion:
    """Word 1-2 grams plus character 2-5 grams (robust to typos)."""
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=60000)),
        ]
    )


class Embedder:
    """Dense sentence vectors.

    backend (env TICKET_EMBEDDER): auto | st | spacy.
    - st: sentence-transformers `all-MiniLM-L6-v2` (needs torch and Hugging Face access; optional).
    - spacy: mean of `en_core_web_md` 300-d static word vectors (offline: the model is a pinned wheel).
    auto tries st and falls back to spacy, recording why in `self.note`.
    """

    def __init__(self, backend: str | None = None) -> None:
        backend = (backend or os.environ.get("TICKET_EMBEDDER", "auto")).lower()
        self.note = ""
        self.backend = ""
        self._st = None
        self._nlp = None
        if backend in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer  # optional dependency

                self._st = SentenceTransformer(ST_MODEL, device="cpu")
                self.backend = f"sentence-transformers:{ST_MODEL}"
            except Exception as e:  # noqa: BLE001 - any failure -> offline fallback
                self.note = f"sentence-transformers unavailable ({type(e).__name__}: {str(e)[:120]})"
                if backend == "st":
                    raise
        if self._st is None:
            import spacy

            self._nlp = spacy.load(SPACY_MODEL, exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner", "senter"])
            self.backend = f"spacy:{SPACY_MODEL} mean word vectors"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._st is not None:
            return np.asarray(self._st.encode(list(texts), batch_size=128, normalize_embeddings=True, show_progress_bar=False))
        vecs = np.vstack([d.vector for d in self._nlp.pipe(texts, batch_size=1000)]).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1.0, norms)
