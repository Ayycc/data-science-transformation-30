# Error analysis (test split, SYNTHETIC documents)

Each wrong field is assigned the first matching category:

| category | rule |
|---|---|
| distractor_confusion | prediction equals a decoy value planted in the document (liability cap, superseded date, order date, balance forward, termination notice) |
| number_format | amount off by a power of ten (decimal/thousands separator confusion) |
| date_format | day and month swapped |
| unseen_layout | document uses a template the rules were not written for (held-out layouts) |
| ocr_noise | wrong or missing value on a scanned-style document |
| derived_value_error | field had to be computed (expiry from term, due date from terms) and is wrong |
| entity_normalisation | party name printed in a long-suffix variant and not normalised |
| other | anything else |

Each error also has a mode: missed (null predicted, value expected), spurious (value predicted, null expected) or wrong_value.

## regex_strict: 919 wrong fields of 3450

| category | count | share of errors | missed / spurious / wrong | top fields |
|---|---|---|---|---|
| unseen_layout | 364 | 39.6% | 326 / 0 / 38 | supplier (54), customer (54), payment_terms_days (52) |
| other | 182 | 19.8% | 158 / 1 / 23 | payment_terms_days (79), total (36), renewal_notice_days (29) |
| ocr_noise | 133 | 14.5% | 111 / 0 / 22 | total (20), payment_terms_days (18), due_date (16) |
| distractor_confusion | 95 | 10.3% | 0 / 59 / 36 | renewal_notice_days (52), effective_date (34), total_value (7) |
| derived_value_error | 76 | 8.3% | 76 / 0 / 0 | expiration_date (49), due_date (27) |
| number_format | 69 | 7.5% | 0 / 0 / 69 | subtotal (25), tax (19), total (15) |

Errors by template: order_form_unseen 238, text_table 141, eu_layout_unseen 128, msa_prose 101, cover_sheet 98, procurement_letter 94, compact_lines 71, ruled_table 48

Examples (first two per category):

| doc | template | field | gold | predicted | category |
|---|---|---|---|---|---|
| C0005 | msa_prose | expiration_date | 2026-09-29 | None | derived_value_error |
| C0012 | msa_prose | expiration_date | 2029-04-28 | None | derived_value_error |
| C0001 | procurement_letter | renewal_notice_days | None | 90 | distractor_confusion |
| C0007 | msa_prose | renewal_notice_days | None | 30 | distractor_confusion |
| C0001 | procurement_letter | total_value | 56900.0 | 56.9 | number_format |
| C0005 | msa_prose | total_value | 83500.0 | 83.5 | number_format |
| C0001 | procurement_letter | customer | Ironbridge Fleet Services Corp. | Ironbridge F leet Services Corp. | ocr_noise |
| C0001 | procurement_letter | effective_date | 2024-10-22 | 2028-10-22 | ocr_noise |
| C0005 | msa_prose | renewal_notice_days | 60 | None | other |
| C0012 | msa_prose | renewal_notice_days | 60 | None | other |
| C0002 | order_form_unseen | supplier | Ironbridge Facilities Corp. | None | unseen_layout |
| C0002 | order_form_unseen | customer | Northwind Industrial Co. | None | unseen_layout |

Human-in-the-loop routing (threshold 0.895 on calibrated confidence, chosen on dev for >= 99% auto-accept accuracy): 914 of 919 wrong fields are sent to review; 5 are auto-accepted.

## rules_repair: 397 wrong fields of 3450

| category | count | share of errors | missed / spurious / wrong | top fields |
|---|---|---|---|---|
| unseen_layout | 336 | 84.6% | 298 / 0 / 38 | supplier (54), customer (54), payment_terms_days (47) |
| ocr_noise | 59 | 14.9% | 42 / 0 / 17 | supplier (10), total_value (6), expiration_date (5) |
| distractor_confusion | 2 | 0.5% | 0 / 0 / 2 | effective_date (2) |

Errors by template: order_form_unseen 231, eu_layout_unseen 105, msa_prose 19, procurement_letter 12, compact_lines 10, text_table 8, cover_sheet 6, ruled_table 6

Examples (first two per category):

| doc | template | field | gold | predicted | category |
|---|---|---|---|---|---|
| C0015 | msa_prose | effective_date | 2025-03-08 | 2021-03-04 | distractor_confusion |
| C0062 | cover_sheet | effective_date | 2024-04-16 | 2024-03-28 | distractor_confusion |
| C0001 | procurement_letter | customer | Ironbridge Fleet Services Corp. | Ironbridge F leet Services Corp. | ocr_noise |
| C0015 | msa_prose | expiration_date | 2029-03-08 | 2025-03-04 | ocr_noise |
| C0002 | order_form_unseen | supplier | Ironbridge Facilities Corp. | None | unseen_layout |
| C0002 | order_form_unseen | customer | Northwind Industrial Co. | None | unseen_layout |

Human-in-the-loop routing (threshold 0.900 on calibrated confidence, chosen on dev for >= 99% auto-accept accuracy): 390 of 397 wrong fields are sent to review; 7 are auto-accepted.

