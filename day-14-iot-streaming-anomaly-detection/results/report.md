# Day 14 technical report (SYNTHETIC data)

Scale: 500 machines x 14 days, 5,804,889 raw sensor events, 562 failures.

## Warning evaluation (all variants x rules)

| variant | rule | n_warnings | tp_warnings | precision | recall | recall_gradual | recall_sudden | n_failures | detected | median_lead_h | p25_lead_h | p75_lead_h | min_lead_h | false_warnings_per_machine_day | f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adaptive | combined | 704 | 642.000 | 0.912 | 0.877 | 0.991 | 0.016 | 520 | 456.000 | 17.261 | 10.832 | 30.565 | 0.526 | 0.009 | 0.894 |
| adaptive | hst_only | 517 | 489.000 | 0.946 | 0.810 | 0.917 | 0.000 | 520 | 421.000 | 10.881 | 6.767 | 17.190 | 0.186 | 0.004 | 0.872 |
| adaptive | rul_only | 547 | 498.000 | 0.910 | 0.662 | 0.747 | 0.016 | 520 | 344.000 | 21.054 | 13.266 | 33.530 | 0.584 | 0.007 | 0.766 |
| no_reset | combined | 1511 | 820.000 | 0.543 | 0.910 | 1.000 | 0.230 | 520 | 473.000 | 24.327 | 14.993 | 42.786 | 0.717 | 0.099 | 0.680 |
| no_reset | hst_only | 731 | 642.000 | 0.878 | 0.879 | 0.996 | 0.000 | 520 | 457.000 | 17.916 | 10.726 | 26.331 | 0.364 | 0.013 | 0.879 |
| no_reset | rul_only | 1285 | 639.000 | 0.497 | 0.687 | 0.747 | 0.230 | 520 | 357.000 | 29.830 | 18.472 | 47.771 | 0.217 | 0.092 | 0.577 |
| static | combined | 2120 | 854.000 | 0.403 | 0.923 | 1.000 | 0.344 | 520 | 480.000 | 25.094 | 14.601 | 43.823 | 1.777 | 0.181 | 0.561 |
| static | hst_only | 1781 | 800.000 | 0.449 | 0.910 | 1.000 | 0.230 | 520 | 473.000 | 22.106 | 12.549 | 37.762 | 1.107 | 0.140 | 0.601 |
| static | rul_only | 1198 | 595.000 | 0.497 | 0.683 | 0.743 | 0.230 | 520 | 355.000 | 27.335 | 15.416 | 44.539 | 0.717 | 0.086 | 0.575 |

## Online RUL accuracy (hours, vs simulator ground truth capped at 72 h)

| variant | n_scored | mae_capped_h | n_within_cap | mae_within_cap_h | mae_last_24h_h |
|---|---|---|---|---|---|
| adaptive | 174755 | 7.310 | 44555 | 14.940 | 14.360 |
| no_reset | 174755 | 18.560 | 44555 | 14.620 | 11.320 |
| static | 174755 | 14.600 | 44555 | 14.310 | 12.820 |

## Throughput

| stage | events | seconds | events_per_sec |
|---|---|---|---|
| windowing only (source + event-time windows) | 5804889 | 13.500 | 428,435.400 |
| end to end (windows + HST + RUL + drift) | 5804889 | 90.800 | 63,942.100 |

## Windowing correctness

| check | value |
|---|---|
| events_in | 5,804,889 |
| events_out_of_order | 3,456,708 |
| late_events_within_allowed_lateness | 3,405,864 |
| late_within_lateness_assigned | 3,405,864 |
| events_beyond_allowed_lateness | 50,844 |
| beyond_lateness_dropped | 29,408 |
| beyond_lateness_still_assigned_window_open | 21,436 |
| oracle_mismatches | 0 |
| window_records_fired | 329,690 |
| stream_vs_batch_sql.batch_windows | 329690 |
| stream_vs_batch_sql.stream_windows | 329690 |
| stream_vs_batch_sql.matched | 329690 |
| stream_vs_batch_sql.count_mismatches | 0 |
| stream_vs_batch_sql.max_abs_diff | 0.0 |

## Exactly-once sink demo (sample stream)

```json
{
  "checkpoint_every_batches": 5,
  "crash_after_batch": 23,
  "batch_size": 1024,
  "clean_rows": 2180,
  "transactional": {
    "rows_after_restart": 2180,
    "duplicate_rows": 0,
    "missing_rows": 0,
    "identical_to_clean_run": true
  },
  "at_least_once": {
    "rows_after_restart": 2355,
    "duplicate_rows": 175,
    "missing_rows": 0,
    "identical_to_clean_run": false
  }
}
```
