# 30 Days, 30 Projects

Index of the 30 projects in the portfolio. Each page summarises the project from its committed files and links to the folder on GitHub.
See the [portfolio report](portfolio-report.md) for the tools matrix, provenance and unmet criteria.

| Day | Project | Tier | Domain | Key metric |
|---|---|---|---|---|
| 01 | [Retail Medallion Lakehouse with dbt + DuckDB](projects/day-01.md) | I - Advanced | Retail | gold vs bronze revenue difference: **$0.00** (gold revenue $124,647,651) |
| 02 | [Card Fraud Detection at Speed with Polars + LightGBM](projects/day-02.md) | I - Advanced | Financial services | test PR-AUC (LightGBM): **0.961** (test-month fraud cost at chosen threshold $41,073) |
| 03 | [Hospital Readmission Risk on an OMOP-style Model](projects/day-03.md) | I - Advanced | Healthcare | test AUROC (logistic): **0.782** (LACE baseline AUROC 0.707) |
| 04 | [Grid Load Forecasting with Probabilistic Backtesting](projects/day-04.md) | I - Advanced | Energy & utilities | day-ahead MASE (best model): **0.445** (seasonal-naive MASE 0.874) |
| 05 | [City 311 Data Quality and Service Hotspots](projects/day-05.md) | I - Advanced | Public sector | share of rows quarantined: **8.7%** (defect classes fully caught 15) |
| 06 | [Experimentation Engine: CUPED, Sequential and Bayesian Tests](projects/day-06.md) | I - Advanced | Digital product & marketing | mSPRT false-positive rate under daily peeking: **1.35%** (naive peeking FPR 25.9%) |
| 07 | [Demand-Driven Inventory Optimisation](projects/day-07.md) | I - Advanced | Supply chain & logistics | total cost reduction vs heuristic: **41.5%** (fill rate (optimised) 97.4%) |
| 08 | [Telecom Retention: Survival Analysis + Uplift Modelling](projects/day-08.md) | II - Expert | Telecommunications | uplift-policy net value (true): **$500,314** (churn-score policy net value $359,254) |
| 09 | [Clickstream Sessionisation at Scale with PySpark + Delta Lake](projects/day-09.md) | II - Expert | Media & e-commerce | exact session recovery rate: **99.995%** (AQE skew-join speed-up 1.76x) |
| 10 | [Air Quality Intelligence with Dagster Software-Defined Assets](projects/day-10.md) | II - Expert | Environment & climate | alert precision (pipeline): **0.966** (false alerts withheld on unit-bug day 130) |
| 11 | [Support Ticket Intelligence: NLP Classification with Active Learning](projects/day-11.md) | II - Expert | Customer operations | macro-F1 at 2% labels (margin AL): **0.974** (random sampling at 2% 0.913) |
| 12 | [Did the Programme Work? Causal Inference with Double ML](projects/day-12.md) | II - Expert | Public policy & workforce | DML earnings ATE: **$2,243** (95% CI low $2,111) |
| 13 | [Anti-Money-Laundering with Transaction Graph Analytics](projects/day-13.md) | II - Expert | Banking & compliance | PR-AUC graph features: **0.911** (PR-AUC tabular only 0.536) |
| 14 | [Streaming Predictive Maintenance for Manufacturing IoT](projects/day-14.md) | II - Expert | Manufacturing | warning precision: **0.912** (warning recall 0.877) |
| 15 | [Bike-Share Demand and Fleet Rebalancing](projects/day-15.md) | II - Expert | Urban mobility & transport | stockout-hour reduction vs fixed routes: **45.9%** (station-hour MAE (LightGBM) 0.663) |
| 16 | [Insurance Claims Severity: Full MLOps Lifecycle](projects/day-16.md) | III - Principal | Insurance | \|A/E - 1\| after retraining: **4.3%** (\|A/E - 1\| static model 11.1%) |
| 17 | [Point-in-Time Correct Feature Store for Ride-Hailing ETA](projects/day-17.md) | III - Principal | Mobility marketplace | naive join offline-to-production MAE gap: **26.9%** (PIT model production MAE (min) 2.159) |
| 18 | [Bayesian Marketing Mix Model and Budget Optimiser](projects/day-18.md) | III - Principal | Consumer goods & marketing | budget reallocation lift ($k/yr, posterior mean): **$1,028k** (max R-hat 1.004) |
| 19 | [Data Mesh in Practice: Contracts, Versioning and Lineage](projects/day-19.md) | III - Principal | Enterprise data platform | breaking contract changes blocked: **15** of 15 |
| 20 | [LLM Structured Extraction with a Rigorous Evaluation Harness](projects/day-20.md) | III - Principal | Legal & procurement | field accuracy (offline rules extractor): **88.5%** (strict regex baseline 73.4%) |
| 21 | [Regulatory Q&A: Hybrid-Search RAG with Measured Faithfulness](projects/day-21.md) | III - Principal | Government & regulation | recall@5 hybrid + learned re-ranker: **81.2%** (BM25 recall@5 68.0%) |
| 22 | [Crop Yield and Climate Risk with xarray + Dask](projects/day-22.md) | III - Principal | Agriculture & climate | +2C/-20% rain mean yield change (model): **-22.1%** (truth -23.8%) |
| 23 | [Observability Analytics: Multivariate Anomaly Detection and Root Cause](projects/day-23.md) | III - Principal | Technology operations | alert volume reduction (raw alerts to incidents): **88.2%** (raw threshold alerts 730) |
| 24 | [Recommender System with Two-Tower Retrieval and Off-Policy Evaluation](projects/day-24.md) | IV - Distinguished | Media streaming | NDCG@10 two-tower + re-ranker: **0.0798** (ALS NDCG@10 0.0825) |
| 25 | [Privacy-Preserving Analytics: Differential Privacy and Synthetic Data](projects/day-25.md) | IV - Distinguished | Census & health statistics | TSTR AUC, DP synthetic data (eps=1): **0.830** (AUC on real data 0.875) |
| 26 | [Responsible Credit Decisioning: Fairness, Explainability and Model Governance](projects/day-26.md) | IV - Distinguished | Lending | min adverse impact ratio (chosen model): **0.776** (unmitigated baseline 0.599) |
| 27 | [Dynamic Pricing for Hospitality with Contextual Bandits](projects/day-27.md) | IV - Distinguished | Travel & hospitality | Thompson sampling revenue uplift vs static table: **5.79%** (95% CI low 5.65%) |
| 28 | [Legacy Reporting Modernisation: SAS/Excel to Iceberg + Ibis](projects/day-28.md) | IV - Distinguished | Financial reporting & insurance | unexplained parallel-run differences: **0** (parallel-run closes 8) |
| 29 | [Agentic Analytics: Text-to-SQL over a Semantic Layer with Guardrails](projects/day-29.md) | IV - Distinguished | Enterprise business intelligence | execution accuracy, semantic layer (answerable): **47** of 47 |
| 30 | [Capstone: Decision-Intelligence Digital Twin and Portfolio Retrospective](projects/day-30.md) | IV - Distinguished | Cross-industry (retail supply chain) | max \|twin - history\| KPI error, hold-out year: **2.9%** (strategic scenarios compared 4) |
