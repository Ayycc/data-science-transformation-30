"""Evaluation harness: per-field exact/fuzzy match, line-item F1, calibration, routing, failure taxonomy."""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .schemas import DATE_FIELDS, FIELDS_BY_TYPE, INT_FIELDS, MONEY_FIELDS, PARTY_FIELDS

FUZZY_FIELDS = PARTY_FIELDS | {"governing_law"}
FUZZY_THRESHOLD = 90
LEGAL_SUFFIX = {"incorporated": "inc", "corporation": "corp", "limited": "ltd", "company": "co"}


def norm_text(v: Any) -> str:
    s = str(v).casefold()
    s = re.sub(r"[.,;:\"']", "", s)
    return re.sub(r"\s+", " ", s).strip()


def compare(field: str, pred: Any, gold: Any) -> tuple[bool, bool]:
    """(exact, fuzzy) correctness for one field. None == None counts as correct (true negative)."""
    if pred is None or gold is None:
        ok = pred is None and gold is None
        return ok, ok
    if field in MONEY_FIELDS:
        try:
            ok = abs(float(pred) - float(gold)) <= 0.005
        except (TypeError, ValueError):
            ok = False
        return ok, ok
    if field in INT_FIELDS:
        try:
            ok = int(pred) == int(gold)
        except (TypeError, ValueError):
            ok = False
        return ok, ok
    if field in DATE_FIELDS:
        ok = str(pred)[:10] == str(gold)[:10]
        return ok, ok
    if isinstance(gold, bool):
        ok = pred is gold or pred == gold
        return ok, ok
    exact = norm_text(pred) == norm_text(gold)
    if field in FUZZY_FIELDS:
        p, g = (" ".join(LEGAL_SUFFIX.get(w, w) for w in norm_text(x).split()) for x in (pred, gold))
        return exact, exact or fuzz.token_sort_ratio(p, g) >= FUZZY_THRESHOLD
    return exact, exact


def match_line_items(pred: list[dict[str, Any]] | None, gold: list[dict[str, Any]]) -> dict[str, int]:
    """Greedy one-to-one matching. Loose match: fuzzy description (>=90) and amount within a cent.
    Strict match additionally needs quantity and unit price exact."""
    pred = list(pred or [])
    used: set[int] = set()
    tp = tp_strict = 0
    for g in gold:
        best, best_s = None, -1.0
        for j, p in enumerate(pred):
            if j in used:
                continue
            s = fuzz.token_sort_ratio(norm_text(p.get("description", "")), norm_text(g["description"]))
            try:
                amt_ok = abs(float(p.get("amount", -1)) - g["amount"]) <= 0.011
            except (TypeError, ValueError):
                amt_ok = False
            if s >= FUZZY_THRESHOLD and amt_ok and s > best_s:
                best, best_s = j, s
        if best is not None:
            used.add(best)
            tp += 1
            p = pred[best]
            try:
                if abs(float(p["quantity"]) - g["quantity"]) < 1e-9 and abs(float(p["unit_price"]) - g["unit_price"]) <= 0.005:
                    tp_strict += 1
            except (KeyError, TypeError, ValueError):
                pass
    return {"tp": tp, "tp_strict": tp_strict, "n_pred": len(pred), "n_gold": len(gold)}


def prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def _is_ratio(a: float, b: float) -> bool:
    if not a or not b:
        return False
    r = a / b
    return any(abs(r - k) / k < 1e-6 for k in (100, 1000, 0.01, 0.001, 10, 0.1))


def error_mode(pred: Any, gold: Any) -> str:
    if pred is None:
        return "missed"
    if gold is None:
        return "spurious"
    return "wrong_value"


def classify_error(field: str, pred: Any, gold: Any, meta: dict[str, Any]) -> str:
    """Root-cause taxonomy for a wrong field (first matching rule wins, most specific first)."""
    distract = meta.get("distractors", {}).get(field, [])
    if pred is not None and any(compare(field, pred, d)[1] for d in distract):
        return "distractor_confusion"
    if field in MONEY_FIELDS and pred is not None and gold is not None:
        try:
            if _is_ratio(float(pred), float(gold)):
                return "number_format"
        except (TypeError, ValueError):
            pass
    if field in DATE_FIELDS and pred is not None and gold is not None:
        try:
            p, g = date.fromisoformat(str(pred)[:10]), date.fromisoformat(str(gold)[:10])
            if (p.month, p.day) == (g.day, g.month) and p != g:
                return "date_format"
        except ValueError:
            pass
    if str(meta.get("template", "")).endswith("_unseen"):
        return "unseen_layout"
    if meta.get("scanned"):
        return "ocr_noise"
    if meta.get("derived", {}).get(field):
        return "derived_value_error"
    if field in PARTY_FIELDS and meta.get("name_variant", {}).get(field):
        return "entity_normalisation"
    return "other"


def field_rows(results: list[Any], golds: dict[str, dict[str, Any]], split: dict[str, str]) -> pd.DataFrame:
    """One row per (extractor, doc, field) with correctness, confidence, error category."""
    rows: list[dict[str, Any]] = []
    for r in results:
        g = golds[r.doc_id]
        label, meta = g["label"], g["meta"]
        gtype = label["doc_type"]
        for fld in FIELDS_BY_TYPE[gtype]:
            pred = r.fields.get(fld) if r.doc_type == gtype else None
            exact, fz = compare(fld, pred, label.get(fld))
            rows.append(
                {
                    "extractor": r.extractor,
                    "doc_id": r.doc_id,
                    "doc_type": gtype,
                    "field": fld,
                    "split": split[r.doc_id],
                    "template": meta["template"],
                    "scanned": bool(meta.get("scanned")),
                    "pred": pred,
                    "gold": label.get(fld),
                    "exact": exact,
                    "correct": fz,
                    "confidence": float(r.confidence.get(fld, 0.0)),
                    "valid_record": r.valid,
                    "error_category": None if fz else classify_error(fld, pred, label.get(fld), meta),
                    "error_mode": None if fz else error_mode(pred, label.get(fld)),
                }
            )
    return pd.DataFrame(rows)


def per_field_table(df: pd.DataFrame) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        nonnull_gold = g["gold"].notna()
        nonnull_pred = g["pred"].notna()
        return pd.Series(
            {
                "n": len(g),
                "exact_acc": g["exact"].mean(),
                "fuzzy_acc": g["correct"].mean(),
                "coverage": nonnull_pred.mean(),
                "precision_nonnull": g.loc[nonnull_pred, "correct"].mean() if nonnull_pred.any() else math.nan,
                "recall_nonnull": g.loc[nonnull_gold, "correct"].mean() if nonnull_gold.any() else math.nan,
                "acc_clean": g.loc[~g["scanned"], "correct"].mean(),
                "acc_scanned": g.loc[g["scanned"], "correct"].mean() if g["scanned"].any() else math.nan,
                "mean_conf": g["confidence"].mean(),
            }
        )

    return df.groupby(["extractor", "doc_type", "field"], sort=False).apply(agg, include_groups=False).reset_index()


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def reliability(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out.append(
                {
                    "bin": b,
                    "lo": b / n_bins,
                    "hi": (b + 1) / n_bins,
                    "n": int(m.sum()),
                    "mean_conf": conf[m].mean(),
                    "accuracy": correct[m].mean(),
                }
            )
    return pd.DataFrame(out)


class IsotonicCalibrator:
    """Per-field isotonic calibration fitted on the dev split, with a pooled fallback for sparse fields."""

    def __init__(self, min_points: int = 60) -> None:
        self.min_points = min_points
        self.models: dict[str, Any] = {}
        self.pooled: Any = None

    @staticmethod
    def _fit(x: np.ndarray, y: np.ndarray) -> Any:
        from sklearn.isotonic import IsotonicRegression

        return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x, y)

    def fit(self, df: pd.DataFrame) -> IsotonicCalibrator:
        self.pooled = self._fit(df["confidence"].to_numpy(), df["correct"].to_numpy(float))
        for fld, g in df.groupby("field"):
            if len(g) >= self.min_points and g["correct"].nunique() > 1:
                self.models[fld] = self._fit(g["confidence"].to_numpy(), g["correct"].to_numpy(float))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(df))
        for i, (fld, c) in enumerate(zip(df["field"], df["confidence"], strict=True)):
            out[i] = (self.models.get(fld) or self.pooled).predict([c])[0]
        return out


def choose_threshold(conf: np.ndarray, correct: np.ndarray, target: float = 0.99) -> float:
    """Lowest threshold whose auto-accepted set has accuracy >= target (maximises automation)."""
    best = 1.01
    for t in sorted(set(np.round(conf, 4)), reverse=True):
        m = conf >= t
        if m.sum() >= 20 and correct[m].mean() >= target:
            best = float(t)
    return best


def routing_summary(df: pd.DataFrame, cal_col: str, thr: float) -> dict[str, Any]:
    """Field-level HITL rule: auto-accept a field iff its record validated AND calibrated confidence >= thr."""
    auto = (df[cal_col] >= thr) & df["valid_record"]
    docs = df.assign(auto=auto).groupby("doc_id").agg(all_auto=("auto", "all"), all_correct=("correct", "all"))
    stp = docs["all_auto"]
    return {
        "threshold": thr,
        "fields_total": int(len(df)),
        "auto_accept_rate": float(auto.mean()),
        "auto_accepted_accuracy": float(df.loc[auto, "correct"].mean()) if auto.any() else None,
        "review_rate": float(1 - auto.mean()),
        "errors_total": int((~df["correct"]).sum()),
        "errors_caught_by_review": int((~df["correct"] & ~auto).sum()),
        "errors_leaked": int((~df["correct"] & auto).sum()),
        "error_capture_rate": float((~df["correct"] & ~auto).sum() / max(1, (~df["correct"]).sum())),
        "straight_through_doc_rate": float(stp.mean()),
        "straight_through_doc_accuracy": float(docs.loc[stp, "all_correct"].mean()) if stp.any() else None,
    }
