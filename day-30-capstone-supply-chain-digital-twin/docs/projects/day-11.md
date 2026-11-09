# Day 11: Support Ticket Intelligence: NLP Classification with Active Learning

**Tier II - Expert** | **Domain:** Customer operations | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-11-support-ticket-nlp-active-learning/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-11-support-ticket-nlp-active-learning/README.md)

- **Key metric:** macro-F1 at 2% labels (margin AL): **0.974** (random sampling at 2% 0.913)
- **Data:** public data (headline); public sources attempted: Bitext via Hugging Face, Bitext via GitHub; reachable: Bitext via GitHub
- **Tests (static):** 20 functions, 23 cases
- **Lines of code:** 1,760 Python source, 150 tests, 0 SQL
- **CI:** success
- **Tools:** en-core-web-md, joblib, llvmlite, matplotlib, numba, numpy, pandas, pynndescent, scikit-learn, scipy, spacy, umap-learn

## Acceptance criteria not met

- active learning matches random-sampling F1 with <= 50% of the labels: needed 63.7% of the labels

## Caveats

- sentence-transformers unreachable (Hugging Face blocked); spaCy static vectors used as fallback
