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
| 02 | I Advanced | [Card Fraud Detection at Speed with Polars + LightGBM](./day-02-card-fraud-polars-lightgbm/) | Financial services | Polars, LightGBM, SHAP | ✅ | [prompt](./prompts/day-02.md) | Test PR-AUC 0.961 on 5.2M synthetic txns; cost-optimal threshold cuts fraud cost $439.6k to $41.1k; Polars 8.85x faster than pandas |
| 03 | I Advanced | [Hospital Readmission Risk on an OMOP-style Model](./day-03-healthcare-readmission-omop/) | Healthcare | pandas, Pydantic, scikit-learn | ✅ | [prompt](./prompts/day-03.md) | Test AUROC 0.782 vs LACE 0.707, calibration slope 1.02 on 9,301 synthetic index admissions; 199/199 invalid FHIR bundles rejected |
| 04 | I Advanced | [Grid Load Forecasting with Probabilistic Backtesting](./day-04-energy-load-forecasting-conformal/) | Energy & utilities | statsforecast, LightGBM, MAPIE | ✅ | [prompt](./prompts/day-04.md) | Day-ahead MASE 0.445 vs seasonal-naive 0.874 (49.1% lower) on 5 synthetic regions x 4 yrs; conformal 90% intervals cover 89.8% over a 12-fold backtest |
| 05 | I Advanced | [City 311 Data Quality and Service Hotspots](./day-05-civic-311-data-quality-hotspots/) | Public sector | pandera, Great Expectations, H3 | ✅ | [prompt](./prompts/day-05.md) | 8.7% of 1.0M synthetic 311 rows quarantined: 15/15 defect classes caught at 100%, 0 clean rows rejected; Gi* finds 36/36 planted slow H3 cells |
| 06 | I Advanced | [Experimentation Engine: CUPED, Sequential and Bayesian Tests](./day-06-experimentation-platform-cuped/) | Digital product & marketing | SciPy, statsmodels, NumPy | ✅ | [prompt](./prompts/day-06.md) | Daily peeking: naive FPR 25.9% vs mSPRT 1.35% over 2,000 synthetic A/A tests; CUPED cuts engagement variance 65.7% (46,268 to 15,087 users/arm) |
| 07 | I Advanced | [Demand-Driven Inventory Optimisation](./day-07-supply-chain-demand-inventory-opt/) | Supply chain & logistics | OR-Tools, PuLP, statsforecast | ✅ | [prompt](./prompts/day-07.md) | SimPy test year on 800 synthetic SKUs x 3 WH: cost -41.5% ($2.32M to $1.35M), fill rate 91.1% to 97.4%; 52/52 weekly MILPs within 0.01% gap |
| 08 | II Expert | [Telecom Retention: Survival Analysis + Uplift Modelling](./day-08-telecom-churn-survival-uplift/) | Telecommunications | lifelines, scikit-survival, CausalML | ✅ | [prompt](./prompts/day-08.md) | Uplift targeting nets $500k vs $359k for churn-score targeting (+39%, ROI 1.19 vs 0.86) on a $400k budget, 65k-customer synthetic holdout; PH violation fixed by contract strata |
| 09 | II Expert | [Clickstream Sessionisation at Scale with PySpark + Delta Lake](./day-09-clickstream-spark-delta/) | Media & e-commerce | PySpark, Delta Lake, Spark SQL | ✅ | [prompt](./prompts/day-09.md) | 13.9M synthetic event deliveries: 99.995% exact session recovery, idempotent replay (0 rows changed), AQE skew join 1.76x faster |
| 10 | II Expert | [Air Quality Intelligence with Dagster Software-Defined Assets](./day-10-air-quality-dagster-assets/) | Environment & climate | Dagster, DuckDB, pandas | ✅ | [prompt](./prompts/day-10.md) | 300 synthetic stations x 30 daily partitions: blocking checks withheld 130 false alerts on a unit-bug day; alert precision 0.966 vs 0.430 naive; late-data sensor re-ran exactly 4 partitions |
| 11 | II Expert | [Support Ticket Intelligence: NLP Classification with Active Learning](./day-11-support-ticket-nlp-active-learning/) | Customer operations | spaCy, sentence-transformers, scikit-learn | ✅ | [prompt](./prompts/day-11.md) | Margin active learning with 2% of labels: macro-F1 0.974 vs 0.913 random on 27 Bitext intents (random needs 3x labels); PII redaction recall 99.93% on 23,664 synthetic spans |
| 12 | II Expert | [Did the Programme Work? Causal Inference with Double ML](./day-12-policy-causal-inference-dml/) | Public policy & workforce | DoWhy, EconML, statsmodels | ✅ | [prompt](./prompts/day-12.md) | DML earnings ATE $2,243 (95% CI $2,111-$2,375) vs true $2,283 on 20k synthetic people; naive bias -$5,101; CS DiD 1.35 pp vs TWFE 1.04 pp (truth 1.47); 7/7 refutations pass |
| 13 | II Expert | [Anti-Money-Laundering with Transaction Graph Analytics](./day-13-aml-transaction-graph-analytics/) | Banking & compliance | NetworkX, igraph, PyTorch Geometric (optional) | ✅ | [prompt](./prompts/day-13.md) | PR-AUC 0.911 graph vs 0.536 tabular on 100k-account / 2.0M-txn synthetic graph; false alerts at 80% recall 6,649 to 82; typology detector recall 0.75-1.00 |
| 14 | II Expert | [Streaming Predictive Maintenance for Manufacturing IoT](./day-14-iot-streaming-anomaly-detection/) | Manufacturing | River, Bytewax or asyncio, Kafka (optional) | ✅ | [prompt](./prompts/day-14.md) | Warnings precision 0.912 / recall 0.877, median lead 17.3 h on 500-machine synthetic stream; 63,942 events/s end to end; drift resets cut false warnings 0.099 to 0.009 per machine-day |
| 15 | II Expert | [Bike-Share Demand and Fleet Rebalancing](./day-15-bikeshare-geospatial-rebalancing/) | Urban mobility & transport | GeoPandas, H3, OSMnx | ✅ | [prompt](./prompts/day-15.md) | Forecast-driven OR-Tools routes cut simulated stockout hours 45.9% vs fixed routes (21,760 to 11,770) replaying 885k real Capital Bikeshare trips; station-hour MAE 0.663 vs 0.747 historical average |
| 16 | III Principal | [Insurance Claims Severity: Full MLOps Lifecycle](./day-16-insurance-claims-mlops/) | Insurance | MLflow, FastAPI, Evidently | ✅ | [prompt](./prompts/day-16.md) | Gated MLflow registry auto-promoted 4/4 passing models, rejected 1; Evidently flagged injected shift in 6/6 months, 0/6 false alarms; retraining cut \|A/E-1\| 11.1% to 4.3% on 600k synthetic policies |
| 17 | III Principal | [Point-in-Time Correct Feature Store for Ride-Hailing ETA](./day-17-ridehail-feature-store-pit/) | Mobility marketplace | Feast, DuckDB, Redis or SQLite online store | ✅ | [prompt](./prompts/day-17.md) | Naive hour-bucket join: offline MAE 2.075 min became 2.633 in production (+26.9%); Feast PIT model 2.159 both, 0 online/offline skew cells, online p99 1.87 ms (SYNTHETIC) |
| 18 | III Principal | [Bayesian Marketing Mix Model and Budget Optimiser](./day-18-bayesian-marketing-mix-model/) | Consumer goods & marketing | PyMC, PyMC-Marketing, ArviZ | ✅ | [prompt](./prompts/day-18.md) | True ROAS in 94% HDI for 4/4 channels, max R-hat 1.004; reallocating the same budget adds $1.03M/yr (94% HDI $0.52M-$1.48M; true +$1.13M) while last-click guidance loses $1.61M (SYNTHETIC) |
| 19 | III Principal | [Data Mesh in Practice: Contracts, Versioning and Lineage](./day-19-data-mesh-contracts-lineage/) | Enterprise data platform | dbt (model contracts), OpenLineage, JSON Schema | ✅ | [prompt](./prompts/day-19.md) | CI gate blocked 15/15 breaking contract changes; versioned data products cut dashboard breakage from 130/370 to 0 dashboard-days (SYNTHETIC); net_revenue traced to raw columns via OpenLineage |
| 20 | III Principal | [LLM Structured Extraction with a Rigorous Evaluation Harness](./day-20-llm-document-extraction-eval/) | Legal & procurement | Anthropic SDK (optional), Pydantic, pdfplumber | ✅ | [prompt](./prompts/day-20.md) | Offline rules extractor 88.5% field accuracy vs 73.4% regex on 300 synthetic test docs (47.4% on held-out layouts); calibrated routing auto-accepts 64.7% of fields at 99.7% accuracy; 0 invalid records written (no live LLM: no API key) |
| 21 | III Principal | [Regulatory Q&A: Hybrid-Search RAG with Measured Faithfulness](./day-21-regulatory-rag-hybrid-search/) | Government & regulation | rank-bm25, sentence-transformers, FAISS or LanceDB | ✅ | [prompt](./prompts/day-21.md) | Hybrid search + learned re-ranker finds the right clause in the top 5 for 81.2% of held-out SYNTHETIC questions (BM25 68.0%); 100% of answers carry a valid citation (offline, no LLM key) |
| 22 | III Principal | [Crop Yield and Climate Risk with xarray + Dask](./day-22-agri-climate-xarray-dask/) | Agriculture & climate | xarray, Dask, rioxarray | ⏳ Planned | [prompt](./prompts/day-22.md) | |
| 23 | III Principal | [Observability Analytics: Multivariate Anomaly Detection and Root Cause](./day-23-sre-anomaly-root-cause/) | Technology operations | ruptures, PyOD, statsmodels | ⏳ Planned | [prompt](./prompts/day-23.md) | |
| 24 | IV Distinguished | [Recommender System with Two-Tower Retrieval and Off-Policy Evaluation](./day-24-streaming-recsys-off-policy-eval/) | Media streaming | PyTorch, FAISS, implicit | ⏳ Planned | [prompt](./prompts/day-24.md) | |
| 25 | IV Distinguished | [Privacy-Preserving Analytics: Differential Privacy and Synthetic Data](./day-25-privacy-dp-synthetic-data/) | Census & health statistics | OpenDP or diffprivlib, SDV, scikit-learn | ⏳ Planned | [prompt](./prompts/day-25.md) | |
| 26 | IV Distinguished | [Responsible Credit Decisioning: Fairness, Explainability and Model Governance](./day-26-lending-fairness-model-governance/) | Lending | Fairlearn, SHAP, scikit-learn | ⏳ Planned | [prompt](./prompts/day-26.md) | |
| 27 | IV Distinguished | [Dynamic Pricing for Hospitality with Contextual Bandits](./day-27-hospitality-dynamic-pricing-bandits/) | Travel & hospitality | NumPy, PyTorch (optional), SciPy | ⏳ Planned | [prompt](./prompts/day-27.md) | |
| 28 | IV Distinguished | [Legacy Reporting Modernisation: SAS/Excel to Iceberg + Ibis](./day-28-legacy-reporting-modernisation-iceberg/) | Financial reporting & insurance | Apache Iceberg (pyiceberg), Ibis, DuckDB | ⏳ Planned | [prompt](./prompts/day-28.md) | |
| 29 | IV Distinguished | [Agentic Analytics: Text-to-SQL over a Semantic Layer with Guardrails](./day-29-agentic-text-to-sql-semantic-layer/) | Enterprise business intelligence | Anthropic SDK (optional), DuckDB, sqlglot | ⏳ Planned | [prompt](./prompts/day-29.md) | |
| 30 | IV Distinguished | [Capstone: Decision-Intelligence Digital Twin and Portfolio Retrospective](./day-30-capstone-supply-chain-digital-twin/) | Cross-industry (retail supply chain) | SimPy, OR-Tools, PyMC | ⏳ Planned | [prompt](./prompts/day-30.md) | |

Last build: 2026-10-31
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
