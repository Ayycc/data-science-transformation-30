# Day 20 run report (SYNTHETIC data)

Data path: `synthetic_generator`, 500 documents, split 200 dev / 300 test. LLM: skipped: ANTHROPIC_API_KEY not set (offline rules extractors only).

| extractor | field acc (fuzzy) | field acc (exact) | docs fully correct | schema-valid | line-item F1 | ECE raw -> cal | auto-accept | auto-accepted acc | p50 ms/doc | $/doc |
|---|---|---|---|---|---|---|---|---|---|---|
| regex_strict | 0.734 | 0.716 | 0.083 | 0.493 | 0.703 | 0.082 -> 0.023 | 0.269 | 0.995 | 27.2 | 0.0000 |
| rules_repair | 0.885 | 0.882 | 0.703 | 0.767 | 0.901 | 0.062 -> 0.020 | 0.647 | 0.997 | 27.7 | 0.0000 |

## Accuracy by slice (test)

| extractor | seen clean | seen scanned | unseen clean | unseen scanned |
|---|---|---|---|---|
| regex_strict | 0.814 (n=180) | 0.777 (n=66) | 0.425 (n=47) | 0.361 (n=7) |
| rules_repair | 1.000 (n=180) | 0.919 (n=66) | 0.474 (n=47) | 0.398 (n=7) |

## Active sampling for review (test)

| extractor | budget | docs | errors found (active) | errors found (random, mean of 500) |
|---|---|---|---|---|
| regex_strict | 10pct_docs | 30 | 215 (23.4%) | 91.9 (10.0%) |
| regex_strict | 20pct_docs | 60 | 395 (43.0%) | 183.8 (20.0%) |
| regex_strict | 30pct_docs | 90 | 491 (53.4%) | 276.8 (30.1%) |
| rules_repair | 10pct_docs | 30 | 212 (53.4%) | 40.1 (10.1%) |
| rules_repair | 20pct_docs | 60 | 353 (88.9%) | 78.4 (19.7%) |
| rules_repair | 30pct_docs | 90 | 389 (98.0%) | 119.5 (30.1%) |

## Validation gate

{
 "regex_strict": {
  "accepted": 255,
  "quarantined": 245,
  "accepted_with_no_review_fields": 0,
  "invalid_written_downstream": 0
 },
 "rules_repair": {
  "accepted": 373,
  "quarantined": 127,
  "accepted_with_no_review_fields": 52,
  "invalid_written_downstream": 0
 }
}

