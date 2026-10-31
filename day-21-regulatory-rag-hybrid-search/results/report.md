# Day 21 results report

Data path: `synthetic_generator` (Model Data Protection and Conduct Regulation (SYNTHETIC)). All text is SYNTHETIC unless the path says public.

## Retrieval, structure chunking (319 chunks, mean 27.9 words, dense backend `static`, RRF k=100)

| method | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 | ndcg@10 | recall@5_lexical | recall@5_paraphrase | latency_ms_per_query |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.54 | 0.648 | 0.68 | 0.732 | 0.6002 | 0.6317 | 1.0 | 0.36 | 0.43 |
| bm25+prf | 0.348 | 0.584 | 0.66 | 0.712 | 0.477 | 0.5346 | 0.976 | 0.344 | 1.12 |
| dense_lsa | 0.512 | 0.628 | 0.676 | 0.748 | 0.5855 | 0.6244 | 1.0 | 0.352 | 0.8 |
| dense_static | 0.508 | 0.656 | 0.704 | 0.752 | 0.5927 | 0.6315 | 1.0 | 0.408 | 0.05 |
| hybrid_rrf | 0.572 | 0.676 | 0.716 | 0.784 | 0.6368 | 0.6717 | 1.0 | 0.432 | 0.5 |
| hybrid_rrf+prf | 0.484 | 0.68 | 0.704 | 0.784 | 0.5883 | 0.6357 | 1.0 | 0.408 | 1.14 |
| hybrid_rrf+rerank_learned | 0.656 | 0.764 | 0.812 | 0.86 | 0.723 | 0.7562 | 1.0 | 0.624 | 2.86 |

## Retrieval, fixed chunking (257 chunks, mean 39.9 words, dense backend `static`, RRF k=30)

| method | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 | ndcg@10 | recall@5_lexical | recall@5_paraphrase | latency_ms_per_query |
|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.416 | 0.6 | 0.652 | 0.716 | 0.5176 | 0.509 | 0.952 | 0.352 | 0.29 |
| bm25+prf | 0.284 | 0.528 | 0.604 | 0.68 | 0.4229 | 0.4398 | 0.904 | 0.304 | 0.82 |
| dense_lsa | 0.372 | 0.584 | 0.636 | 0.716 | 0.4885 | 0.4918 | 0.944 | 0.328 | 0.71 |
| dense_static | 0.344 | 0.536 | 0.596 | 0.696 | 0.4564 | 0.4586 | 0.88 | 0.312 | 0.05 |
| hybrid_rrf | 0.424 | 0.588 | 0.648 | 0.74 | 0.5233 | 0.5181 | 0.952 | 0.344 | 0.43 |
| hybrid_rrf+prf | 0.364 | 0.596 | 0.668 | 0.72 | 0.495 | 0.499 | 0.968 | 0.368 | 1.03 |
| hybrid_rrf+rerank_learned | 0.6 | 0.684 | 0.764 | 0.848 | 0.6645 | 0.6518 | 0.984 | 0.544 | 2.05 |

## Hybrid vs single methods (recall@5, paired bootstrap 95% CI)

- **structure**: hybrid 0.716 vs BM25 0.68 vs dense 0.704; beats both: **True**. hybrid-BM25 = +0.0360 [+0.0040, +0.0720], hybrid-dense = +0.0120 [-0.0120, +0.0360]
- **fixed**: hybrid 0.648 vs BM25 0.652 vs dense 0.596; beats both: **False**. hybrid-BM25 = -0.0040 [-0.0360, +0.0280], hybrid-dense = +0.0520 [+0.0200, +0.0840]

## Answers (test split)

### extractive_structure

| metric | value |
|---|---|
| tau | 0.4405 |
| dev_balanced_accuracy | 0.7963 |
| dev_in_scope_answer_rate | 0.676 |
| dev_oos_refusal_rate | 0.9167 |
| n_questions | 274 |
| n_in_scope | 250 |
| n_out_of_scope | 24 |
| n_answered | 173 |
| answered_rate_in_scope | 0.684 |
| accuracy_answered_in_scope | 0.8655 |
| accuracy_all_in_scope | 0.592 |
| answers_with_valid_citation | 1.0 |
| n_answers_without_valid_citation | 0 |
| citation_precision | 0.8333 |
| citation_recall | 0.8538 |
| claim_support_rate | 1.0 |
| fully_faithful_answers | 1.0 |
| oos_refusal_recall | 0.9167 |
| oos_refusal_precision | 0.2178 |
| in_scope_false_refusal_rate | 0.316 |

### extractive_fixed

| metric | value |
|---|---|
| tau | 0.3653 |
| dev_balanced_accuracy | 0.7823 |
| dev_in_scope_answer_rate | 0.648 |
| dev_oos_refusal_rate | 0.9167 |
| n_questions | 274 |
| n_in_scope | 250 |
| n_out_of_scope | 24 |
| n_answered | 159 |
| answered_rate_in_scope | 0.624 |
| accuracy_answered_in_scope | 0.8077 |
| accuracy_all_in_scope | 0.504 |
| answers_with_valid_citation | 1.0 |
| n_answers_without_valid_citation | 0 |
| citation_precision | 0.7179 |
| citation_recall | 0.7628 |
| claim_support_rate | 1.0 |
| fully_faithful_answers | 1.0 |
| oos_refusal_recall | 0.875 |
| oos_refusal_precision | 0.1826 |
| in_scope_false_refusal_rate | 0.376 |

## Faithfulness checker stress test

| chunking | clean false-alarm rate | fact_swap detected | citation_swap detected | added_claim detected |
|---|---|---|---|---|
| structure | 0.0 (0/173) | 1.0 (173/173) | 0.9017 (156/173) | 1.0 (173/173) |
| fixed | 0.0 (0/159) | 1.0 (159/159) | 0.9497 (151/159) | 1.0 (159/159) |

## Guardrail trade-off (structure, test; tau was chosen on dev)

| tau | in-scope answer rate | accuracy of answered | OOS refusal rate |
|---|---|---|---|
| 0.0 | 1.0 | 0.736 | 0.0 |
| 0.1 | 0.9 | 0.7689 | 0.5833 |
| 0.2 | 0.832 | 0.7981 | 0.7917 |
| 0.3 | 0.756 | 0.836 | 0.8333 |
| 0.4 | 0.712 | 0.8596 | 0.875 |
| 0.4405 | 0.684 | 0.8655 | 0.9167 |
| 0.5 | 0.632 | 0.8734 | 0.9167 |
| 0.6 | 0.532 | 0.8872 | 0.9167 |
| 0.7 | 0.448 | 0.8929 | 0.9583 |

## Example answers (structure chunking, extractive)

- **[lexical] Who must review and approve supplier disputes, and how often?**  
  gold: `the board audit committee, quarterly` ['§2.1(a)']  
  system: Answer: the board audit committee, at least quarterly [§2.1(a)]. Source: "All supplier disputes shall be reviewed and endorsed by the board audit committee at least quarterly." [§2.1(a)].
- **[paraphrase] Which officer is responsible for signing off contractor complaints?**  
  gold: `the board audit committee, quarterly` ['§2.1(a)']  
  system: I cannot answer this from the regulation: no passage is relevant enough (score 0.16 < 0.44).
- **[lexical] Which firms are exempt from the requirements on contractor complaints?**  
  gold: `fewer than 250` ['§2.1(b)']  
  system: Answer: fewer than 250 [§2.1(b)]. Source: "The requirements of this section on contractor complaints do not apply to a covered institution that serves fewer than 250 data subjects in a calendar year." [§2.1(b)].
- **[paraphrase] Are small firms excused from the contractor disputes rules?**  
  gold: `fewer than 250` ['§2.1(b)']  
  system: I cannot answer this from the regulation: no passage is relevant enough (score 0.25 < 0.44).
- **[out_of_scope] What is the deadline for filing school enrolment forms?**  
  gold: `refuse` []  
  system: I cannot answer this from the regulation: no passage is relevant enough (score 0.34 < 0.44).
- **[out_of_scope] Who must approve wine import labels?**  
  gold: `refuse` []  
  system: I cannot answer this from the regulation: no passage is relevant enough (score 0.02 < 0.44).

LLM: not used (no ANTHROPIC_API_KEY); all answers come from the extractive fallback.
