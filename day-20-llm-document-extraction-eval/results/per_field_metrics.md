# Per-field metrics (test split, SYNTHETIC documents)

exact = normalised exact match; fuzzy = rapidfuzz token_sort_ratio >= 90 for party names and governing law (other fields: exact).
coverage = share of documents with a non-null prediction; acc_clean / acc_scanned split by scanned-style noise.

## regex_strict

| doc type | field | n | exact | fuzzy | coverage | acc clean | acc scanned | mean conf |
|---|---|---|---|---|---|---|---|---|
| contract | contract_number | 150 | 0.980 | 0.980 | 0.827 | 1.000 | 0.900 | 0.871 |
| contract | supplier | 150 | 0.627 | 0.733 | 0.733 | 0.775 | 0.567 | 0.762 |
| contract | customer | 150 | 0.653 | 0.720 | 0.733 | 0.775 | 0.500 | 0.762 |
| contract | effective_date | 150 | 0.767 | 0.767 | 1.000 | 0.775 | 0.733 | 0.900 |
| contract | expiration_date | 150 | 0.360 | 0.360 | 0.360 | 0.367 | 0.333 | 0.530 |
| contract | total_value | 150 | 0.660 | 0.660 | 0.660 | 0.642 | 0.733 | 0.869 |
| contract | currency | 150 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.950 |
| contract | payment_terms_days | 150 | 0.607 | 0.607 | 0.553 | 0.575 | 0.733 | 0.766 |
| contract | auto_renewal | 150 | 0.707 | 0.707 | 1.000 | 0.700 | 0.733 | 0.900 |
| contract | renewal_term_months | 150 | 0.873 | 0.873 | 0.413 | 0.892 | 0.800 | 0.810 |
| contract | renewal_notice_days | 150 | 0.293 | 0.293 | 0.527 | 0.292 | 0.300 | 0.779 |
| contract | governing_law | 150 | 0.760 | 0.760 | 0.700 | 0.792 | 0.633 | 0.853 |
| invoice | invoice_number | 150 | 0.993 | 0.993 | 0.993 | 1.000 | 0.977 | 0.914 |
| invoice | po_number | 150 | 1.000 | 1.000 | 0.567 | 1.000 | 1.000 | 0.863 |
| invoice | supplier | 150 | 0.707 | 0.833 | 1.000 | 0.813 | 0.884 | 0.841 |
| invoice | customer | 150 | 0.747 | 0.847 | 0.847 | 0.813 | 0.930 | 0.818 |
| invoice | invoice_date | 150 | 0.913 | 0.913 | 0.953 | 0.935 | 0.860 | 0.915 |
| invoice | due_date | 150 | 0.573 | 0.573 | 0.573 | 0.561 | 0.605 | 0.630 |
| invoice | currency | 150 | 0.993 | 0.993 | 0.993 | 1.000 | 0.977 | 0.945 |
| invoice | subtotal | 150 | 0.660 | 0.660 | 0.833 | 0.645 | 0.698 | 0.800 |
| invoice | tax | 150 | 0.813 | 0.813 | 0.953 | 0.850 | 0.721 | 0.872 |
| invoice | total | 150 | 0.387 | 0.387 | 0.487 | 0.364 | 0.442 | 0.592 |
| invoice | payment_terms_days | 150 | 0.400 | 0.400 | 0.327 | 0.318 | 0.605 | 0.698 |

## rules_repair

| doc type | field | n | exact | fuzzy | coverage | acc clean | acc scanned | mean conf |
|---|---|---|---|---|---|---|---|---|
| contract | contract_number | 150 | 1.000 | 1.000 | 0.847 | 1.000 | 1.000 | 0.869 |
| contract | supplier | 150 | 0.720 | 0.740 | 0.760 | 0.775 | 0.600 | 0.768 |
| contract | customer | 150 | 0.740 | 0.747 | 0.760 | 0.775 | 0.633 | 0.768 |
| contract | effective_date | 150 | 0.987 | 0.987 | 1.000 | 1.000 | 0.933 | 0.786 |
| contract | expiration_date | 150 | 0.747 | 0.747 | 0.760 | 0.775 | 0.633 | 0.768 |
| contract | total_value | 150 | 0.800 | 0.800 | 0.627 | 0.825 | 0.700 | 0.859 |
| contract | currency | 150 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.946 |
| contract | payment_terms_days | 150 | 0.813 | 0.813 | 0.760 | 0.833 | 0.733 | 0.866 |
| contract | auto_renewal | 150 | 0.873 | 0.873 | 1.000 | 0.892 | 0.800 | 0.851 |
| contract | renewal_term_months | 150 | 0.860 | 0.860 | 0.400 | 0.892 | 0.733 | 0.908 |
| contract | renewal_notice_days | 150 | 0.867 | 0.867 | 0.407 | 0.892 | 0.767 | 0.912 |
| contract | governing_law | 150 | 0.780 | 0.780 | 0.700 | 0.792 | 0.733 | 0.850 |
| invoice | invoice_number | 150 | 0.993 | 0.993 | 0.993 | 1.000 | 0.977 | 0.905 |
| invoice | po_number | 150 | 1.000 | 1.000 | 0.567 | 1.000 | 1.000 | 0.855 |
| invoice | supplier | 150 | 0.813 | 0.833 | 1.000 | 0.813 | 0.884 | 0.833 |
| invoice | customer | 150 | 0.840 | 0.860 | 0.860 | 0.813 | 0.977 | 0.818 |
| invoice | invoice_date | 150 | 0.987 | 0.987 | 1.000 | 1.000 | 0.953 | 0.834 |
| invoice | due_date | 150 | 0.840 | 0.840 | 0.847 | 0.813 | 0.907 | 0.808 |
| invoice | currency | 150 | 0.993 | 0.993 | 0.993 | 1.000 | 0.977 | 0.936 |
| invoice | subtotal | 150 | 0.840 | 0.840 | 0.840 | 0.813 | 0.907 | 0.818 |
| invoice | tax | 150 | 0.967 | 0.967 | 0.967 | 1.000 | 0.884 | 0.850 |
| invoice | total | 150 | 0.973 | 0.973 | 0.980 | 1.000 | 0.907 | 0.853 |
| invoice | payment_terms_days | 150 | 0.853 | 0.853 | 0.773 | 0.813 | 0.953 | 0.866 |
