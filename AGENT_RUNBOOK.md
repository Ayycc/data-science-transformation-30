# Agent runbook: daily build

This file is read by the scheduled cloud agent each day. Humans can read it to see exactly what the automation does.

1. Read `prompts/_conventions.md` and the progress table in `README.md`.
2. Pick the target day: the lowest-numbered row whose Status is not ✅. If a folder for that day already exists but is incomplete, continue it rather than starting over. If all 30 rows are ✅, stop and do nothing else.
3. Read `prompts/day-NN.md` for the target day and build the project to its definition of done. Work only inside that day's folder, plus the single progress row and `Last build` line in `README.md`.
4. Run `make test` and `make run` inside the folder. Fix failures. Report real results only.
5. Commit as `Day NN: <title>` and push to `main` (pull --rebase and retry if rejected).
6. Build exactly one day per run, even if time remains. Quality over quantity.
