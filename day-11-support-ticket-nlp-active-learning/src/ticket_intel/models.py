"""Linear classifiers and evaluation helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def make_clf(kind: str, seed: int) -> LogisticRegression:
    """kind: 'tfidf' (sparse) or 'emb' (dense, unit-normalised)."""
    c = 20.0 if kind == "tfidf" else 50.0
    return LogisticRegression(C=c, solver="newton-cg", max_iter=1000, random_state=seed)


def full_proba(clf: LogisticRegression, X, labels: Sequence[str]) -> np.ndarray:
    """predict_proba with columns for every label (zeros for classes absent from training)."""
    p = clf.predict_proba(X)
    out = np.zeros((p.shape[0], len(labels)))
    idx = {lab: i for i, lab in enumerate(labels)}
    for j, c in enumerate(clf.classes_):
        out[:, idx[c]] = p[:, j]
    return out


def macro_f1(y_true, y_pred, labels: Sequence[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def evaluate(y_true, y_pred, labels: Sequence[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), zero_division=0)
    return {
        "macro_f1": macro_f1(y_true, y_pred, labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_intent": {
            lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, lab in enumerate(labels)
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
    }
