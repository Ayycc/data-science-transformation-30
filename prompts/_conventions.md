# Build conventions (apply to every day)

These rules keep 30 projects, built by 30 independent sessions, consistent and reviewable.

## Folder layout

```
day-NN-slug/
  README.md            # executive summary, Mermaid architecture, provenance, method, results, limitations, how to run
  Makefile             # targets: setup, data, run, test, clean (run must reproduce all reported results)
  requirements.txt     # pinned versions (==) for everything imported
  src/<package>/       # importable code; no logic hidden in notebooks
  tests/               # pytest; fast (< 3 min total), use small fixtures
  data/                # git-ignored except data/sample/ (<= 1 MB)
  results/             # committed: metrics.json, figures (PNG/HTML), reports (MD); each file <= 5 MB
  notebooks/           # optional, exploratory only, outputs cleared
  scripts/             # data generation / download entry points
```

## Engineering standards

- Python 3.11+. Type hints on public functions. Format with ruff (`ruff format`, `ruff check`).
- Seed every random process. A single `make run` regenerates data (or uses the sample), trains/evaluates, and writes `results/metrics.json`.
- Keep compute laptop-class: CPU only, finish `make run` in under 20 minutes. Scale data down rather than skipping steps, and state the scale used.
- No secrets in the repo. Optional external APIs (for example the Anthropic API) must be read from environment variables, and the project must still run end-to-end without them through a documented offline fallback.
- No Docker requirement at runtime. Dockerfiles and docker-compose files are welcome as artefacts but tests must not depend on a running Docker daemon.
- Prefer well-maintained, widely used libraries. If a library fails to install, choose an equivalent and record the substitution in the README.

## Honesty standards

- Every number in a README comes from `results/` produced by the committed code. Never invent or round up results.
- If an acceptance criterion is not met, say so plainly in the README under "Limitations" with what was tried.
- Synthetic data is labelled as synthetic everywhere it appears.

## README results table

Each day's README includes a "Results" table with the headline metrics, and a "What this demonstrates" section mapping the work to the skills listed in the prompt.

## Root README progress table

After finishing, update exactly one row of the root `README.md` table (between the `<!-- PROGRESS:START -->` and `<!-- PROGRESS:END -->` markers): set Status to ✅, fill in the headline result (one short line with a real number), and keep the folder link. Also set the `Last build` line under the table to today's date.

## Git

- Commit with message `Day NN: <title>` and push directly to `main`. Do not open a pull request; this repository is built autonomously.
- If the push is rejected because `main` moved, `git pull --rebase origin main` and push again.
