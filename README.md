# Data Science & Transformation: 30 Days, 30 Projects

A public portfolio of 30 end-to-end data science and data transformation projects, one per day, progressing from **advanced** to **distinguished**-level work across 30 industries and the modern data stack.

Each project is built autonomously by a scheduled Claude Code cloud agent from a self-contained prompt in [`prompts/`](./prompts/), following shared [build conventions](./prompts/_conventions.md) and the [agent runbook](./AGENT_RUNBOOK.md). Every number in every project README is produced by the committed code.

## Progression

| Tier | Level | Days | Focus |
|---|---|---|---|
| I | Advanced | 1-7 | Production-grade foundations: modern SQL/dataframe engines, rigorous evaluation, data quality. |
| II | Expert | 8-15 | Scale and specialised methods: distributed compute, orchestration, causal, graph, streaming, geospatial. |
| III | Principal | 16-23 | Platforms and systems: MLOps, feature stores, Bayesian decision models, data contracts, LLM and RAG pipelines. |
| IV | Distinguished | 24-30 | Field-defining builds: counterfactual evaluation, privacy, responsible AI, sequential decisions, legacy modernisation, agentic analytics, and a capstone digital twin. |

## Progress

<!-- PROGRESS:START -->
| Day | Tier | Project | Domain | Key tools | Status | Prompt | Headline result |
|---|---|---|---|---|---|---|---|
| 01 | I Advanced | [Retail Medallion Lakehouse with dbt + DuckDB](./day-01-retail-medallion-lakehouse/) | Retail | DuckDB, dbt-duckdb, SQL | ✅ | [prompt](./prompts/day-01.md) | Gold reconciles to bronze to $0.00 on $124.6M over 2.0M synthetic order lines; 67/67 dbt tests pass |
| 02 | I Advanced | [Card Fraud Detection at Speed with Polars + LightGBM](./day-02-card-fraud-polars-lightgbm/) | Financial services | Polars, LightGBM, SHAP | ⏳ Planned | [prompt](./prompts/day-02.md) | |
| 03 | I Advanced | [Hospital Readmission Risk on an OMOP-style Model](./day-03-healthcare-readmission-omop/) | Healthcare | pandas, Pydantic, scikit-learn | ⏳ Planned | [prompt](./prompts/day-03.md) | |
| 04 | I Advanced | [Grid Load Forecasting with Probabilistic Backtesting](./day-04-energy-load-forecasting-conformal/) | Energy & utilities | statsforecast, LightGBM, MAPIE | ⏳ Planned | [prompt](./prompts/day-04.md) | |
| 05 | I Advanced | [City 311 Data Quality and Service Hotspots](./day-05-civic-311-data-quality-hotspots/) | Public sector | pandera, Great Expectations, H3 | ⏳ Planned | [prompt](./prompts/day-05.md) | |
| 06 | I Advanced | [Experimentation Engine: CUPED, Sequential and Bayesian Tests](./day-06-experimentation-platform-cuped/) | Digital product & marketing | SciPy, statsmodels, NumPy | ⏳ Planned | [prompt](./prompts/day-06.md) | |
| 07 | I Advanced | [Demand-Driven Inventory Optimisation](./day-07-supply-chain-demand-inventory-opt/) | Supply chain & logistics | OR-Tools, PuLP, statsforecast | ⏳ Planned | [prompt](./prompts/day-07.md) | |
| 08 | II Expert | [Telecom Retention: Survival Analysis + Uplift Modelling](./day-08-telecom-churn-survival-uplift/) | Telecommunications | lifelines, scikit-survival, CausalML | ⏳ Planned | [prompt](./prompts/day-08.md) | |
| 09 | II Expert | [Clickstream Sessionisation at Scale with PySpark + Delta Lake](./day-09-clickstream-spark-delta/) | Media & e-commerce | PySpark, Delta Lake, Spark SQL | ⏳ Planned | [prompt](./prompts/day-09.md) | |
| 10 | II Expert | [Air Quality Intelligence with Dagster Software-Defined Assets](./day-10-air-quality-dagster-assets/) | Environment & climate | Dagster, DuckDB, pandas | ⏳ Planned | [prompt](./prompts/day-10.md) | |
| 11 | II Expert | [Support Ticket Intelligence: NLP Classification with Active Learning](./day-11-support-ticket-nlp-active-learning/) | Customer operations | spaCy, sentence-transformers, scikit-learn | ⏳ Planned | [prompt](./prompts/day-11.md) | |
| 12 | II Expert | [Did the Programme Work? Causal Inference with Double ML](./day-12-policy-causal-inference-dml/) | Public policy & workforce | DoWhy, EconML, statsmodels | ⏳ Planned | [prompt](./prompts/day-12.md) | |
| 13 | II Expert | [Anti-Money-Laundering with Transaction Graph Analytics](./day-13-aml-transaction-graph-analytics/) | Banking & compliance | NetworkX, igraph, PyTorch Geometric (optional) | ⏳ Planned | [prompt](./prompts/day-13.md) | |
| 14 | II Expert | [Streaming Predictive Maintenance for Manufacturing IoT](./day-14-iot-streaming-anomaly-detection/) | Manufacturing | River, Bytewax or asyncio, Kafka (optional) | ⏳ Planned | [prompt](./prompts/day-14.md) | |
| 15 | II Expert | [Bike-Share Demand and Fleet Rebalancing](./day-15-bikeshare-geospatial-rebalancing/) | Urban mobility & transport | GeoPandas, H3, OSMnx | ⏳ Planned | [prompt](./prompts/day-15.md) | |
| 16 | III Principal | [Insurance Claims Severity: Full MLOps Lifecycle](./day-16-insurance-claims-mlops/) | Insurance | MLflow, FastAPI, Evidently | ⏳ Planned | [prompt](./prompts/day-16.md) | |
| 17 | III Principal | [Point-in-Time Correct Feature Store for Ride-Hailing ETA](./day-17-ridehail-feature-store-pit/) | Mobility marketplace | Feast, DuckDB, Redis or SQLite online store | ⏳ Planned | [prompt](./prompts/day-17.md) | |
| 18 | III Principal | [Bayesian Marketing Mix Model and Budget Optimiser](./day-18-bayesian-marketing-mix-model/) | Consumer goods & marketing | PyMC, PyMC-Marketing, ArviZ | ⏳ Planned | [prompt](./prompts/day-18.md) | |
| 19 | III Principal | [Data Mesh in Practice: Contracts, Versioning and Lineage](./day-19-data-mesh-contracts-lineage/) | Enterprise data platform | dbt (model contracts), OpenLineage, JSON Schema | ⏳ Planned | [prompt](./prompts/day-19.md) | |
| 20 | III Principal | [LLM Structured Extraction with a Rigorous Evaluation Harness](./day-20-llm-document-extraction-eval/) | Legal & procurement | Anthropic SDK (optional), Pydantic, pdfplumber | ⏳ Planned | [prompt](./prompts/day-20.md) | |
| 21 | III Principal | [Regulatory Q&A: Hybrid-Search RAG with Measured Faithfulness](./day-21-regulatory-rag-hybrid-search/) | Government & regulation | rank-bm25, sentence-transformers, FAISS or LanceDB | ⏳ Planned | [prompt](./prompts/day-21.md) | |
| 22 | III Principal | [Crop Yield and Climate Risk with xarray + Dask](./day-22-agri-climate-xarray-dask/) | Agriculture & climate | xarray, Dask, rioxarray | ⏳ Planned | [prompt](./prompts/day-22.md) | |
| 23 | III Principal | [Observability Analytics: Multivariate Anomaly Detection and Root Cause](./day-23-sre-anomaly-root-cause/) | Technology operations | ruptures, PyOD, statsmodels | ⏳ Planned | [prompt](./prompts/day-23.md) | |
| 24 | IV Distinguished | [Recommender System with Two-Tower Retrieval and Off-Policy Evaluation](./day-24-streaming-recsys-off-policy-eval/) | Media streaming | PyTorch, FAISS, implicit | ⏳ Planned | [prompt](./prompts/day-24.md) | |
| 25 | IV Distinguished | [Privacy-Preserving Analytics: Differential Privacy and Synthetic Data](./day-25-privacy-dp-synthetic-data/) | Census & health statistics | OpenDP or diffprivlib, SDV, scikit-learn | ⏳ Planned | [prompt](./prompts/day-25.md) | |
| 26 | IV Distinguished | [Responsible Credit Decisioning: Fairness, Explainability and Model Governance](./day-26-lending-fairness-model-governance/) | Lending | Fairlearn, SHAP, scikit-learn | ⏳ Planned | [prompt](./prompts/day-26.md) | |
| 27 | IV Distinguished | [Dynamic Pricing for Hospitality with Contextual Bandits](./day-27-hospitality-dynamic-pricing-bandits/) | Travel & hospitality | NumPy, PyTorch (optional), SciPy | ⏳ Planned | [prompt](./prompts/day-27.md) | |
| 28 | IV Distinguished | [Legacy Reporting Modernisation: SAS/Excel to Iceberg + Ibis](./day-28-legacy-reporting-modernisation-iceberg/) | Financial reporting & insurance | Apache Iceberg (pyiceberg), Ibis, DuckDB | ⏳ Planned | [prompt](./prompts/day-28.md) | |
| 29 | IV Distinguished | [Agentic Analytics: Text-to-SQL over a Semantic Layer with Guardrails](./day-29-agentic-text-to-sql-semantic-layer/) | Enterprise business intelligence | Anthropic SDK (optional), DuckDB, sqlglot | ⏳ Planned | [prompt](./prompts/day-29.md) | |
| 30 | IV Distinguished | [Capstone: Decision-Intelligence Digital Twin and Portfolio Retrospective](./day-30-capstone-supply-chain-digital-twin/) | Cross-industry (retail supply chain) | SimPy, OR-Tools, PyMC | ⏳ Planned | [prompt](./prompts/day-30.md) | |

Last build: 2026-10-12
<!-- PROGRESS:END -->

## How each project is structured

Every `day-NN-*/` folder has a README (executive summary, architecture, data provenance, method, results, limitations), a `Makefile` with `setup`, `run` and `test`, pinned requirements, tests, and committed results. Data is public or synthetic; large data is regenerated by script.

```bash
cd day-01-retail-medallion-lakehouse
make setup && make run && make test
```

## End state (day 30)

- 30 runnable projects covering 30 domains and more than 60 tools.
- A portfolio report (`day-30-*/PORTFOLIO_REPORT.md`) with a tool coverage matrix, test status and headline metric per project.
- A docs site indexing all projects.

## License

MIT
