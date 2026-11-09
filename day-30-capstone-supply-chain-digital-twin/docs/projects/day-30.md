# Day 30: Capstone: Decision-Intelligence Digital Twin and Portfolio Retrospective

**Tier IV - Distinguished** | **Domain:** Cross-industry (retail supply chain) | [Folder on GitHub](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-30-capstone-supply-chain-digital-twin/) | [README](https://github.com/Ayycc/data-science-transformation-30/tree/main/day-30-capstone-supply-chain-digital-twin/README.md)

- **Key metric:** max \|twin - history\| KPI error, hold-out year: **2.9%** (strategic scenarios compared 4)
- **Data:** synthetic by design
- **Tests (static):** 40 functions, 40 cases (40 stated in README)
- **Lines of code:** 2,750 Python source, 387 tests, 0 SQL
- **CI:** pending
- **Tools:** arviz, matplotlib, mkdocs, numpy, ortools, pandas, pymc, pytensor, pyyaml, scipy, simpy, streamlit

## Caveats

- Twin is conservative: historical KPI inside the twin's 5-95% band for only 3 of 8 KPIs
- Optimiser's store safety factor at the top of its grid for the DC-East plan
- Reinforcement-learning replenishment agent (stretch) not attempted
