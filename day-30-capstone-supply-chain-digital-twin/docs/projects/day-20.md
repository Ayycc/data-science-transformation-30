# Day 20: LLM Structured Extraction with a Rigorous Evaluation Harness

**Tier III - Principal** | **Domain:** Legal & procurement | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-20-llm-document-extraction-eval/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-20-llm-document-extraction-eval/README.md)

- **Key metric:** field accuracy (offline rules extractor): **88.5%** (strict regex baseline 73.4%)
- **Data:** synthetic by design
- **Tests (static):** 32 functions, 32 cases (32 stated in README)
- **Lines of code:** 2,336 Python source, 404 tests, 0 SQL
- **CI:** success
- **Tools:** anthropic, matplotlib, numpy, pandas, pdfplumber, pydantic, rapidfuzz, reportlab, scikit-learn

## Caveats

- No live LLM evaluated (no ANTHROPIC_API_KEY); offline rules extractor only
