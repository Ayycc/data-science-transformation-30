# Day 24: Recommender System with Two-Tower Retrieval and Off-Policy Evaluation

**Tier IV - Distinguished** | **Domain:** Media streaming | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-24-streaming-recsys-off-policy-eval/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-24-streaming-recsys-off-policy-eval/README.md)

- **Key metric:** NDCG@10 two-tower + re-ranker: **0.0798** (ALS NDCG@10 0.0825)
- **Data:** synthetic fallback (public source blocked); public sources attempted: MovieLens-1M; reachable: none
- **Tests (static):** 16 functions, 16 cases (16 stated in README)
- **Lines of code:** 1,602 Python source, 189 tests, 0 SQL
- **CI:** success
- **Tools:** faiss-cpu, implicit, lightgbm, matplotlib, numpy, pandas, scikit-learn, scipy, threadpoolctl, torch

## Acceptance criteria not met

- two-tower retrieval + re-ranker beats the ALS baseline on NDCG@10: 0.0798 vs ALS 0.0825 (-3.3%)
