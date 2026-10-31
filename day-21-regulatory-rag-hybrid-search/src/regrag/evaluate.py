"""Retrieval metrics, citation metrics, and a deterministic faithfulness checker with a hallucination stress test."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

import numpy as np

from .generate import CITATION_RE, Answer
from .retrieval import STOPWORDS, tokenize

# ----------------------------------------------------------------------------------------------- retrieval


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Hit rate: 1 if any relevant chunk is in the top k (each in-scope question has one supporting clause)."""
    return 1.0 if relevant & set(ranked[:k]) else 0.0


def mrr_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    for i, c in enumerate(ranked[:k]):
        if c in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, c in enumerate(ranked[:k]) if c in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean of a-b with a 95% percentile bootstrap CI over questions (paired)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boots = d[idx].mean(axis=1)
    return {
        "diff": float(d.mean()),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "p_le_0": float((boots <= 0).mean()),
    }


# ----------------------------------------------------------------------------------------------- answers

FRAMING = {"answer", "source", "also", "relevant", "least", "according", "passage", "clause", "states", "reads", "regulation"}
NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in NUM_RE.findall(text)}


@dataclass
class SegmentCheck:
    claim: str
    cited: list[str]
    supported: bool
    reason: str


def split_claims(text: str) -> list[tuple[str, list[str]]]:
    """Split an answer into (claim text, citations) segments. Text after the last citation becomes an uncited claim."""
    out: list[tuple[str, list[str]]] = []
    pos = 0
    for m in re.finditer(r"((?:\s*\[§\d+\.\d+\([a-z]+\)\])+)\.?", text):
        claim = text[pos : m.start()].strip()
        if claim:
            out.append((claim, CITATION_RE.findall(m.group(1))))
        pos = m.end()
    tail = text[pos:].strip()
    if tail and set(tokenize(tail)) - FRAMING:
        out.append((tail, []))
    return out


def check_faithfulness(answer_text: str, context: dict[str, str], min_token_recall: float = 0.8) -> list[SegmentCheck]:
    """A claim is supported iff it cites >=1 clause shown to the generator, every number in it appears in the cited
    text, and >= ``min_token_recall`` of its content tokens appear in the cited text."""
    checks = []
    for claim, cited in split_claims(answer_text):
        if not cited:
            checks.append(SegmentCheck(claim, cited, False, "uncited"))
            continue
        if any(c not in context for c in cited):
            checks.append(SegmentCheck(claim, cited, False, "citation_not_in_context"))
            continue
        src = " ".join(context[c] for c in cited)
        missing_nums = _numbers(claim) - _numbers(src)
        if missing_nums:
            checks.append(SegmentCheck(claim, cited, False, "number_not_in_source"))
            continue
        toks = [t for t in tokenize(claim) if t not in FRAMING and t not in STOPWORDS]
        src_toks = set(tokenize(src))
        rec = sum(t in src_toks for t in toks) / len(toks) if toks else 1.0
        checks.append(
            SegmentCheck(claim, cited, rec >= min_token_recall, "ok" if rec >= min_token_recall else f"token_recall={rec:.2f}")
        )
    return checks


def answer_correct(answer_text: str, gold: str) -> bool:
    a = answer_text.lower().replace(",", "")
    return all(part.strip().lower().replace(",", "") in a for part in gold.split(", ") if part.strip())


def evaluate_answers(answers: list[Answer], gold: dict[str, dict], all_clause_ids: set[str]) -> tuple[dict, list[dict]]:
    rows = []
    for a in answers:
        g = gold[a.qid]
        checks = check_faithfulness(a.text, a.context) if not a.refused else []
        valid = [c for c in a.citations if c in all_clause_ids and c in a.context]
        gold_ids = set(g["gold_clause_ids"])
        rows.append(
            {
                "qid": a.qid,
                "style": g["style"],
                "in_scope": g["in_scope"],
                "refused": a.refused,
                "n_citations": len(a.citations),
                "n_valid_citations": len(valid),
                "has_valid_citation": bool(valid),
                "citation_precision": (len(set(a.citations) & gold_ids) / len(a.citations)) if a.citations else None,
                "citation_recall": (1.0 if gold_ids & set(a.citations) else 0.0) if gold_ids else None,
                "correct": answer_correct(a.text, g["answer"]) if g["in_scope"] and not a.refused else None,
                "n_claims": len(checks),
                "n_supported": sum(c.supported for c in checks),
                "fully_faithful": all(c.supported for c in checks) if checks else None,
                "top_prob": round(a.top_prob, 4),
                "answer": a.text,
            }
        )
    ins = [r for r in rows if r["in_scope"]]
    oos = [r for r in rows if not r["in_scope"]]
    ans = [r for r in rows if not r["refused"]]
    ans_in = [r for r in ins if not r["refused"]]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.mean(xs)), 4) if xs else None

    refused_oos = sum(r["refused"] for r in oos)
    refused_in = sum(r["refused"] for r in ins)
    summary = {
        "n_questions": len(rows),
        "n_in_scope": len(ins),
        "n_out_of_scope": len(oos),
        "n_answered": len(ans),
        "answered_rate_in_scope": mean([0.0 if r["refused"] else 1.0 for r in ins]),
        "accuracy_answered_in_scope": mean([r["correct"] for r in ans_in]),
        "accuracy_all_in_scope": mean([bool(r["correct"]) for r in ins]),
        "answers_with_valid_citation": mean([1.0 if r["has_valid_citation"] else 0.0 for r in ans]),
        "n_answers_without_valid_citation": sum(not r["has_valid_citation"] for r in ans),
        "citation_precision": mean([r["citation_precision"] for r in ans_in]),
        "citation_recall": mean([r["citation_recall"] for r in ans_in]),
        "claim_support_rate": round(sum(r["n_supported"] for r in ans) / max(1, sum(r["n_claims"] for r in ans)), 4),
        "fully_faithful_answers": mean([1.0 if r["fully_faithful"] else 0.0 for r in ans]),
        "oos_refusal_recall": round(refused_oos / len(oos), 4) if oos else None,
        "oos_refusal_precision": round(refused_oos / (refused_oos + refused_in), 4) if (refused_oos + refused_in) else None,
        "in_scope_false_refusal_rate": round(refused_in / len(ins), 4) if ins else None,
    }
    return summary, rows


# ----------------------------------------------------------------------------------------------- stress test

OFFICERS_RE = re.compile(
    r"(Chief Compliance Officer|Data Protection Officer|Chief Risk Officer|board audit committee|Head of Internal Audit)"
)


def corrupt(answer: Answer, kind: str, rng: random.Random, other_clause: str | None) -> str | None:
    """Inject one hallucination into the first claim of an extractive answer. Returns None if not applicable."""
    text = answer.text
    segs = split_claims(text)
    if not segs:
        return None
    first_claim, cites = segs[0]
    if kind == "fact_swap":
        nums = NUM_RE.findall(first_claim)
        if nums:
            old = nums[0]
            val = int(old.replace(",", ""))
            new = (
                f"{val + rng.choice([1, 2, 5, 10]) * max(1, val // 10):,}" if "," in old else str(val + rng.choice([1, 2, 5, 10]))
            )
            bad = first_claim.replace(old, new, 1)
        else:
            m = OFFICERS_RE.search(first_claim)
            if not m:
                return None
            choices = [
                o
                for o in [
                    "Chief Compliance Officer",
                    "Data Protection Officer",
                    "Chief Risk Officer",
                    "board audit committee",
                    "Head of Internal Audit",
                ]
                if o != m.group(1)
            ]
            bad = first_claim.replace(m.group(1), rng.choice(choices), 1)
        return text.replace(first_claim, bad, 1)
    if kind == "citation_swap":
        if not other_clause or other_clause in cites:
            return None
        return text.replace(f"[{cites[0]}]", f"[{other_clause}]", 1)
    if kind == "added_claim":
        return (
            text
            + f" Firms may also apply to the regulator for a {rng.choice([30, 60, 90])}-day extension in exceptional circumstances [{cites[0]}]."
        )
    raise ValueError(kind)


def stress_test(answers: list[Answer], seed: int = 0) -> dict:
    """Detection rate of the faithfulness checker on corrupted answers, and its false-alarm rate on clean ones."""
    rng = random.Random(seed)
    out: dict[str, dict] = {}
    clean = [a for a in answers if not a.refused]
    flagged_clean = sum(not all(c.supported for c in check_faithfulness(a.text, a.context)) for a in clean)
    out["clean"] = {"n": len(clean), "flagged": flagged_clean, "false_alarm_rate": round(flagged_clean / max(1, len(clean)), 4)}
    for kind in ("fact_swap", "citation_swap", "added_claim"):
        n = caught = 0
        for a in clean:
            others = [c for c in a.context if c not in a.citations]
            bad = corrupt(a, kind, rng, rng.choice(others) if others else None)
            if bad is None or bad == a.text:
                continue
            n += 1
            caught += not all(c.supported for c in check_faithfulness(bad, a.context))
        out[kind] = {"n": n, "detected": caught, "detection_rate": round(caught / max(1, n), 4)}
    return out
