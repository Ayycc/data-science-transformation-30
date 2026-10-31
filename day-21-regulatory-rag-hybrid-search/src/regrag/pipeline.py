"""End-to-end experiment: chunk -> retrieve (BM25 / dense / hybrid / +PRF / +re-rank) -> guardrail -> generate
with citations -> evaluate retrieval, citations, faithfulness -> results/."""

from __future__ import annotations

import csv
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk, answer_span, fixed_chunks, relevant_chunk_ids, render_with_offsets, structure_chunks
from .corpus import Clause, GoldQuestion
from .evaluate import answer_correct, evaluate_answers, mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k, stress_test
from .generate import Answer, ClaudeGenerator, ExtractiveGenerator
from .rerank import make_reranker
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever

KS = (1, 3, 5, 10)
RRF_GRID = (10, 30, 60, 100)


def _metrics(ranked: list[list[str]], rel: list[set[str]]) -> dict[str, float]:
    m = {f"recall@{k}": float(np.mean([recall_at_k(r, s, k) for r, s in zip(ranked, rel, strict=True)])) for k in KS}
    m["mrr@10"] = float(np.mean([mrr_at_k(r, s) for r, s in zip(ranked, rel, strict=True)]))
    m["ndcg@10"] = float(np.mean([ndcg_at_k(r, s) for r, s in zip(ranked, rel, strict=True)]))
    return {k: round(v, 4) for k, v in m.items()}


class Setup:
    """All retrievers for one chunking strategy."""

    def __init__(self, name: str, chunks: list[Chunk], dev: list[GoldQuestion], spans, seed: int = 0) -> None:
        self.name, self.chunks = name, chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.spans = spans
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(chunks)  # default backend: static pretrained vectors (falls back to LSA)
        self.lsa = DenseRetriever(chunks, backend="lsa", seed=seed)
        dev_rel = [self.relevant(q) for q in dev]
        # Tune the single RRF constant on dev (recall@5, then MRR), never on test.
        best = None
        for k in RRF_GRID:
            h = HybridRetriever(self.bm25, self.dense, rrf_k=k)
            ranked = [[x.chunk_id for x in h.search(q.question, 10)] for q in dev]
            m = _metrics(ranked, dev_rel)
            key = (m["recall@5"], m["mrr@10"])
            if best is None or key > best[0]:
                best = (key, k)
        self.rrf_k = best[1]
        self.hybrid = HybridRetriever(self.bm25, self.dense, rrf_k=self.rrf_k)
        self.hybrid_prf = HybridRetriever(self.bm25, self.dense, rrf_k=self.rrf_k, prf=True)
        self.reranker, self.reranker_name = make_reranker(chunks, self.bm25, self.dense, self.hybrid)
        self.reranker.fit([q.question for q in dev], dev_rel, seed=seed)

    def relevant(self, q: GoldQuestion) -> set[str]:
        return relevant_chunk_ids(self.chunks, q.gold_clause_ids, self.spans)

    def methods(self) -> dict:
        return {
            "bm25": lambda q: [h.chunk_id for h in self.bm25.search(q, 10)],
            "bm25+prf": lambda q: [h.chunk_id for h in self.bm25.search(self.bm25.expand_query(q), 10)],
            "dense_lsa": lambda q: [h.chunk_id for h in self.lsa.search(q, 10)],
            f"dense_{self.dense.backend}": lambda q: [h.chunk_id for h in self.dense.search(q, 10)],
            "hybrid_rrf": lambda q: [h.chunk_id for h in self.hybrid.search(q, 10)],
            "hybrid_rrf+prf": lambda q: [h.chunk_id for h in self.hybrid_prf.search(q, 10)],
            f"hybrid_rrf+rerank_{self.reranker_name}": lambda q: [s.chunk_id for s in self.reranker.rerank(q, 10)],
        }


def tune_tau(setup: Setup, dev_all: list[GoldQuestion]) -> tuple[float, dict]:
    """Guardrail threshold on the top re-ranker probability, chosen on dev to maximise balanced accuracy of
    (answer in-scope, refuse out-of-scope)."""
    tops = [(setup.reranker.rerank(q.question, 1)[0].prob, q.in_scope) for q in dev_all]
    cands = sorted({round(p, 4) for p, _ in tops} | {0.0})
    best = None
    for t in cands:
        tpr = np.mean([p >= t for p, ins in tops if ins])
        tnr = np.mean([p < t for p, ins in tops if not ins])
        bal = (tpr + tnr) / 2
        if best is None or bal > best[0] + 1e-12:
            best = (bal, t, tpr, tnr)
    return float(best[1]), {
        "dev_balanced_accuracy": round(float(best[0]), 4),
        "dev_in_scope_answer_rate": round(float(best[2]), 4),
        "dev_oos_refusal_rate": round(float(best[3]), 4),
    }


def run(data_dir: Path, out_dir: Path, seed: int = 0, make_figures: bool = True) -> dict:
    from .data import load

    t_start = time.perf_counter()
    clauses, questions, provenance = load(data_dir)
    clause_by_id: dict[str, Clause] = {c.clause_id: c for c in clauses}
    doc, offsets = render_with_offsets(clauses)
    spans = {c.clause_id: answer_span(c, offsets) for c in clauses}
    dev = [q for q in questions if q.split == "dev" and q.in_scope]
    test = [q for q in questions if q.split == "test" and q.in_scope]
    dev_all = [q for q in questions if q.split == "dev"]
    test_all = [q for q in questions if q.split == "test"]
    gold = {q.qid: asdict(q) for q in questions}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    chunkings = {"structure": structure_chunks(clauses), "fixed": fixed_chunks(clauses, size=40, overlap=10)}
    metrics: dict = {
        "data": provenance,
        "scale": {
            "n_clauses": len(clauses),
            "n_dev_in_scope": len(dev),
            "n_test_in_scope": len(test),
            "n_test_out_of_scope": len(test_all) - len(test),
            "n_dev_out_of_scope": len(dev_all) - len(dev),
        },
        "retrieval": {},
        "environment": {"python": platform.python_version(), "cpu_only": True},
    }
    rows_csv = []
    setups: dict[str, Setup] = {}
    per_q: dict[str, dict[str, list[float]]] = {}
    for cname, chunks in chunkings.items():
        st = Setup(cname, chunks, dev, spans, seed)
        setups[cname] = st
        rel = [st.relevant(q) for q in test]
        res = {
            "n_chunks": len(chunks),
            "mean_chunk_words": round(float(np.mean([len(c.text.split()) for c in chunks])), 1),
            "rrf_k": st.rrf_k,
            "dense_backend": st.dense.backend,
            "reranker": st.reranker_name,
            "methods": {},
        }
        if hasattr(st.reranker, "coefficients"):
            res["reranker_coefficients"] = st.reranker.coefficients()
        for mname, fn in st.methods().items():
            t0 = time.perf_counter()
            ranked = [fn(q.question) for q in test]
            ms = (time.perf_counter() - t0) * 1000 / len(test)
            m = _metrics(ranked, rel)
            for style in ("lexical", "paraphrase", "cloze"):
                idx = [i for i, q in enumerate(test) if q.style == style]
                if idx:
                    m[f"recall@5_{style}"] = round(float(np.mean([recall_at_k(ranked[i], rel[i], 5) for i in idx])), 4)
            m["latency_ms_per_query"] = round(ms, 2)
            res["methods"][mname] = m
            per_q.setdefault(cname, {})[mname] = [recall_at_k(r, s, 5) for r, s in zip(ranked, rel, strict=True)]
            rows_csv.append({"chunking": cname, "method": mname, **m})
        metrics["retrieval"][cname] = res

    # Acceptance check: hybrid beats both single methods on recall@5 (with paired bootstrap CIs).
    acc = {}
    for cname, st in setups.items():
        dname = f"dense_{st.dense.backend}"
        r = metrics["retrieval"][cname]["methods"]
        acc[cname] = {
            "hybrid_recall@5": r["hybrid_rrf"]["recall@5"],
            "bm25_recall@5": r["bm25"]["recall@5"],
            "dense_recall@5": r[dname]["recall@5"],
            "hybrid_beats_both": r["hybrid_rrf"]["recall@5"] > max(r["bm25"]["recall@5"], r[dname]["recall@5"]),
            "hybrid_minus_bm25": paired_bootstrap(per_q[cname]["hybrid_rrf"], per_q[cname]["bm25"], seed=seed),
            "hybrid_minus_dense": paired_bootstrap(per_q[cname]["hybrid_rrf"], per_q[cname][dname], seed=seed),
        }
    metrics["acceptance_hybrid_vs_single"] = acc

    # Guardrail + generation (both chunkings; structure is the production configuration).
    gen_metrics = {}
    answers_rows = []
    stress = {}
    all_ids = set(clause_by_id)
    extractive_answers: dict[str, list[Answer]] = {}
    for cname, st in setups.items():
        tau, tau_info = tune_tau(st, dev_all)
        gen = ExtractiveGenerator(tau)
        answers = [
            gen.answer(q.qid, q.question, st.reranker.rerank(q.question, 5), st.by_id, clause_by_id, offsets, doc)
            for q in test_all
        ]
        extractive_answers[cname] = answers
        summary, rows = evaluate_answers(answers, gold, all_ids)
        gen_metrics[f"extractive_{cname}"] = {"tau": round(tau, 4), **tau_info, **summary}
        stress[cname] = stress_test(answers, seed=seed)
        if cname == "structure":
            # Transparency only (tau itself was fixed on dev): how the operating point trades coverage for safety.
            # The extractive answer for a question does not depend on tau, so re-thresholding the rows is exact.
            unguarded = ExtractiveGenerator(0.0)
            ug_correct = {}
            for q in test:
                ua = unguarded.answer(q.qid, q.question, st.reranker.rerank(q.question, 5), st.by_id, clause_by_id, offsets, doc)
                ug_correct[q.qid] = answer_correct(ua.text, q.answer)
            sweep = []
            for t in (0.0, 0.1, 0.2, 0.3, 0.4, round(tau, 4), 0.5, 0.6, 0.7):
                ins = [r for r in rows if r["in_scope"]]
                oos = [r for r in rows if not r["in_scope"]]
                answered = [r for r in ins if r["top_prob"] >= t]
                ok = [r for r in answered if ug_correct[r["qid"]]]
                sweep.append(
                    {
                        "tau": t,
                        "in_scope_answer_rate": round(len(answered) / len(ins), 4),
                        "accuracy_answered": round(len(ok) / max(1, len(answered)), 4),
                        "oos_refusal_rate": round(sum(r["top_prob"] < t for r in oos) / max(1, len(oos)), 4),
                    }
                )
            metrics["guardrail_tradeoff_test"] = sweep
        for r in rows:
            answers_rows.append(
                {
                    "chunking": cname,
                    "generator": "extractive",
                    "question": gold[r["qid"]]["question"],
                    "gold_answer": gold[r["qid"]]["answer"],
                    "gold_clauses": ";".join(gold[r["qid"]]["gold_clause_ids"]),
                    **r,
                }
            )

    llm_info = {"api_key_present": bool(os.environ.get("ANTHROPIC_API_KEY")), "used": False}
    if llm_info["api_key_present"]:  # pragma: no cover - network path
        st = setups["structure"]
        tau = gen_metrics["extractive_structure"]["tau"]
        cg = ClaudeGenerator(tau)
        cap = int(os.environ.get("LLM_MAX_QUESTIONS", "80"))
        subset = [q for q in test_all if not q.in_scope] + [q for q in test_all if q.in_scope][
            : max(0, cap - sum(not q.in_scope for q in test_all))
        ]
        use_hyde = os.environ.get("HYDE") == "1"
        answers = []
        for q in subset:
            hyde = cg.hypothetical_clause(q.question) if use_hyde else None
            answers.append(
                cg.answer(
                    q.qid,
                    q.question,
                    st.reranker.rerank(q.question, 5, extra_dense_text=hyde),
                    st.by_id,
                    clause_by_id,
                    offsets,
                    doc,
                )
            )
        summary, rows = evaluate_answers(answers, gold, all_ids)
        lat = [a.latency_ms for a in answers if not a.refused]
        gen_metrics["claude_structure"] = {
            "model": cg.model,
            "effort": cg.effort,
            "hyde": use_hyde,
            **summary,
            "input_tokens": sum(a.input_tokens for a in answers),
            "output_tokens": sum(a.output_tokens for a in answers),
            "latency_ms_p50": round(float(np.percentile(lat, 50)), 1) if lat else None,
            "errors": sum(any(n.startswith("error=") for n in a.notes) for a in answers),
        }
        llm_info["used"] = True
        llm_info["model"] = cg.model
        for r in rows:
            answers_rows.append(
                {
                    "chunking": "structure",
                    "generator": "claude",
                    "question": gold[r["qid"]]["question"],
                    "gold_answer": gold[r["qid"]]["answer"],
                    "gold_clauses": ";".join(gold[r["qid"]]["gold_clause_ids"]),
                    **r,
                }
            )
    metrics["generation"] = gen_metrics
    metrics["faithfulness_stress_test"] = stress
    metrics["llm"] = llm_info
    metrics["runtime_seconds"] = round(time.perf_counter() - t_start, 1)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _write_csv(out_dir / "retrieval_results.csv", rows_csv)
    _write_csv(out_dir / "answers_test.csv", answers_rows)
    (out_dir / "report.md").write_text(render_report(metrics, extractive_answers["structure"], gold))
    if make_figures:
        from .figures import make_all

        make_all(metrics, out_dir / "figures")
    return metrics


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def render_report(m: dict, answers: list[Answer], gold: dict) -> str:
    lines = [
        "# Day 21 results report",
        "",
        f"Data path: `{m['data']['path_used']}` ({m['data']['title']}). All text is SYNTHETIC unless the path says public.",
        "",
    ]
    for cname, res in m["retrieval"].items():
        lines += [
            f"## Retrieval, {cname} chunking ({res['n_chunks']} chunks, mean {res['mean_chunk_words']} words, dense backend `{res['dense_backend']}`, RRF k={res['rrf_k']})",
            "",
        ]
        cols = [
            "recall@1",
            "recall@3",
            "recall@5",
            "recall@10",
            "mrr@10",
            "ndcg@10",
            "recall@5_lexical",
            "recall@5_paraphrase",
            "latency_ms_per_query",
        ]
        lines.append("| method | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for name, mm in res["methods"].items():
            lines.append(f"| {name} | " + " | ".join(str(mm.get(c, "")) for c in cols) + " |")
        lines.append("")
    lines += ["## Hybrid vs single methods (recall@5, paired bootstrap 95% CI)", ""]
    for cname, a in m["acceptance_hybrid_vs_single"].items():
        lines.append(
            f"- **{cname}**: hybrid {a['hybrid_recall@5']} vs BM25 {a['bm25_recall@5']} vs dense {a['dense_recall@5']}; beats both: **{a['hybrid_beats_both']}**. hybrid-BM25 = {a['hybrid_minus_bm25']['diff']:+.4f} [{a['hybrid_minus_bm25']['ci_low']:+.4f}, {a['hybrid_minus_bm25']['ci_high']:+.4f}], hybrid-dense = {a['hybrid_minus_dense']['diff']:+.4f} [{a['hybrid_minus_dense']['ci_low']:+.4f}, {a['hybrid_minus_dense']['ci_high']:+.4f}]"
        )
    lines += ["", "## Answers (test split)", ""]
    for g, s in m["generation"].items():
        lines += [f"### {g}", "", "| metric | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s.items()] + [""]
    lines += [
        "## Faithfulness checker stress test",
        "",
        "| chunking | clean false-alarm rate | fact_swap detected | citation_swap detected | added_claim detected |",
        "|---|---|---|---|---|",
    ]
    for cname, s in m["faithfulness_stress_test"].items():
        lines.append(
            f"| {cname} | {s['clean']['false_alarm_rate']} ({s['clean']['flagged']}/{s['clean']['n']}) | {s['fact_swap']['detection_rate']} ({s['fact_swap']['detected']}/{s['fact_swap']['n']}) | {s['citation_swap']['detection_rate']} ({s['citation_swap']['detected']}/{s['citation_swap']['n']}) | {s['added_claim']['detection_rate']} ({s['added_claim']['detected']}/{s['added_claim']['n']}) |"
        )
    if "guardrail_tradeoff_test" in m:
        lines += [
            "",
            "## Guardrail trade-off (structure, test; tau was chosen on dev)",
            "",
            "| tau | in-scope answer rate | accuracy of answered | OOS refusal rate |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {r['tau']} | {r['in_scope_answer_rate']} | {r['accuracy_answered']} | {r['oos_refusal_rate']} |"
            for r in m["guardrail_tradeoff_test"]
        ]
    lines += ["", "## Example answers (structure chunking, extractive)", ""]
    shown = {"lexical": 0, "paraphrase": 0, "out_of_scope": 0}
    for a in answers:
        st = gold[a.qid]["style"]
        if st in shown and shown[st] < 2:
            shown[st] += 1
            lines += [
                f"- **[{st}] {gold[a.qid]['question']}**  ",
                f"  gold: `{gold[a.qid]['answer'] or 'refuse'}` {gold[a.qid]['gold_clause_ids']}  ",
                f"  system: {a.text}",
            ]
    lines.append("")
    lines.append(
        f"LLM: {'used, model ' + m['llm'].get('model', '') if m['llm']['used'] else 'not used (no ANTHROPIC_API_KEY); all answers come from the extractive fallback'}."
    )
    return "\n".join(lines) + "\n"
