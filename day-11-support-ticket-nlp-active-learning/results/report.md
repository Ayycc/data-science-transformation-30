# Support ticket intelligence: run report

Data: PUBLIC: Bitext customer-support corpus (CDLA-Sharing-1.0) + SYNTHETIC PII/snippet layer (26872 tickets, 27 intents). All PII is SYNTHETIC.

## Per-intent F1 (test)

| Intent | Support | TF-IDF 100% | Embedding 100% | Random 2% | Margin AL 2% |
|---|---|---|---|---|---|
| cancel_order | 200 | 0.997 | 0.967 | 0.943 | 0.965 |
| change_order | 199 | 1.000 | 0.936 | 0.798 | 0.955 |
| change_shipping_address | 195 | 1.000 | 0.918 | 0.957 | 0.985 |
| check_cancellation_fee | 190 | 1.000 | 0.960 | 0.976 | 0.995 |
| check_invoice | 200 | 0.998 | 0.939 | 0.923 | 0.908 |
| check_payment_methods | 200 | 1.000 | 0.980 | 0.957 | 0.997 |
| check_refund_policy | 199 | 1.000 | 0.922 | 0.930 | 0.995 |
| complaint | 200 | 1.000 | 0.982 | 0.977 | 0.995 |
| contact_customer_service | 200 | 1.000 | 0.961 | 0.932 | 0.992 |
| contact_human_agent | 200 | 1.000 | 0.930 | 0.913 | 0.988 |
| create_account | 199 | 0.995 | 0.931 | 0.912 | 0.950 |
| delete_account | 199 | 0.995 | 0.883 | 0.911 | 0.944 |
| delivery_options | 199 | 1.000 | 0.934 | 0.892 | 0.978 |
| delivery_period | 200 | 1.000 | 0.927 | 0.957 | 0.992 |
| edit_account | 200 | 1.000 | 0.945 | 0.951 | 0.970 |
| get_invoice | 200 | 0.997 | 0.937 | 0.922 | 0.903 |
| get_refund | 199 | 1.000 | 0.914 | 0.899 | 0.972 |
| newsletter_subscription | 200 | 1.000 | 0.987 | 0.997 | 1.000 |
| payment_issue | 200 | 1.000 | 0.852 | 0.969 | 0.997 |
| place_order | 200 | 1.000 | 0.958 | 0.890 | 0.980 |
| recover_password | 199 | 1.000 | 0.980 | 0.982 | 0.995 |
| registration_problems | 200 | 0.998 | 0.850 | 0.980 | 0.987 |
| review | 199 | 1.000 | 0.962 | 0.956 | 0.995 |
| set_up_shipping_address | 199 | 1.000 | 0.977 | 0.962 | 0.985 |
| switch_account | 200 | 1.000 | 0.962 | 0.935 | 0.968 |
| track_order | 199 | 1.000 | 0.956 | 0.833 | 0.973 |
| track_refund | 200 | 1.000 | 0.964 | 0.927 | 0.977 |

## Active learning (TF-IDF, mean over repeats)

| Strategy | F1 @ 2% | F1 @ 1% | Labels to match random @ 2% | Share of budget |
|---|---|---|---|---|
| entropy | 0.9264 | 0.7713 | 395 | 0.9186 |
| margin | 0.9737 | 0.8655 | 274 | 0.6372 |
| margin_diverse | 0.9785 | 0.8760 | 274 | 0.6372 |
| random | 0.9132 | 0.7926 | 430 | 1.0 |

## Topic clusters (test set)

| Cluster | Size | Majority intent | Purity | Keywords |
|---|---|---|---|---|
| 6 | 417 | set_up_shipping_address | 0.47 | address, delivery address, shipping address, shipping, delivery, set |
| 2 | 393 | registration_problems | 0.50 | inform, notify, report, payments, online, issues |
| 31 | 202 | contact_human_agent | 0.95 | agent, speak, talk, person, operator, human |
| 5 | 180 | complaint | 1.00 | claim, complaint, lodge, reclamation, consumer, make |
| 20 | 174 | get_invoice | 0.54 | bills, download, download bills, look, bills download, mail |
| 14 | 165 | recover_password | 0.99 | user, key, pin, retrieve, reset, pin code |
| 21 | 151 | get_refund | 0.58 | money, es urgente, favor es, es, favor, urgente |
| 49 | 125 | switch_account | 0.36 | freemium account, freemium, free account, free, account, switch |
| 41 | 118 | contact_customer_service | 1.00 | customer, customer service, contact customer, service, contact, talk customer |
| 40 | 111 | review | 1.00 | feedback, send feedback, feedback product, send, submit feedback, submit |
| 19 | 109 | delete_account | 0.26 | information, question, accounts, account information, address, information opening |
| 18 | 107 | check_cancellation_fee | 1.00 | early, termination, exit, early exit, early termination, check early |
| 3 | 101 | track_refund | 0.71 | dollars, waiting, expect, paid, rebate, refund |
| 34 | 100 | change_order | 0.48 | purchase, cancelling purchase, purchase cancelling, cancelling, product purchase, item purchase |
| 43 | 99 | cancel_order | 0.68 | order, cancel order, order cancel, cancelling order, canceling order, cancel |
| 32 | 92 | track_order | 0.61 | status, status order, order status, current, current status, order |
| 44 | 91 | edit_account | 0.33 | premium, premium account, account, open premium, open, change premium |
| 48 | 84 | place_order | 0.94 | item, product, products, items, acquire, earn |
| 28 | 83 | get_invoice | 0.52 | invoice, invoices, getting, invoice check, getting invoice, invoice getting |
| 39 | 83 | review | 1.00 | review, leave, comment, opinion, write, service |
| 45 | 81 | place_order | 0.65 | articles, article, articles order, ur article, article order, article purchase |
| 1 | 80 | delivery_period | 1.00 | soon, soon expect, expect, check soon, expect shipment, expect product |
| 8 | 78 | check_payment_methods | 0.65 | methods, payment methods, payment, shipping methods, accepted, accepted payment |
| 46 | 78 | edit_account | 0.33 | gold account, gold, account, change gold, info gold, open gold |
| 17 | 77 | check_cancellation_fee | 0.99 | withdrawal, cancellation, penalty, fee, cancellation penalty, penalties |
| 27 | 76 | get_invoice | 0.68 | invoice, download invoice, download, invoice don, invoice download, invoice invoice |
| 36 | 76 | create_account | 0.32 | standard account, standard, account, change standard, create standard, change |
| 13 | 74 | delivery_options | 0.41 | possible, place, possible place, place order, possible order, order possible |
| 16 | 74 | newsletter_subscription | 1.00 | unsubscribe, newsletter, subscribe, newsletter unsubscribe, subscribe newsletter, corporate |
| 4 | 69 | check_payment_methods | 0.97 | modalities, payment modalities, payment, available payment, modalities list, available |
| 0 | 65 | delivery_period | 0.98 | arrive, gonna, gonna arrive, going, going arrive, arrive check |
| 9 | 63 | newsletter_subscription | 1.00 | subscription, newsletter subscription, newsletter, cancel subscription, company newsletter, cancel |
| 29 | 62 | place_order | 0.45 | don, buy, consumer, don shop, claim, file |
| 24 | 61 | track_order | 1.00 | eta, eta order, eta purchase, check eta, order eta, order |
| 25 | 61 | get_refund | 0.48 | reimbursement, reimbursements, reimbursement money, reimbursement policy, policy, money |
| 47 | 58 | change_order | 0.86 | order, item order, item, add, items order, order delete |
| 26 | 57 | delivery_period | 0.77 | long, takes, long takes, arrive, check long, long refunds |
| 50 | 57 | create_account | 0.56 | pro, pro account, create pro, new pro, account, create |
| 42 | 56 | contact_customer_service | 1.00 | hours, reach, customer, reach customer, time, check hours |
| 7 | 54 | delivery_options | 0.67 | methods, available, methods delivery, delivery, delivery methods, methods available |
| 51 | 53 | edit_account | 1.00 | data, included, pro account, update, user, details |
| 15 | 51 | newsletter_subscription | 1.00 | newsletter, receive, sign, newsletter sign, newsletter receive, corporate newsletter |
| 38 | 50 | switch_account | 0.98 | profile, use, profile use, standard profile, pro profile, profile using |
| 35 | 49 | cancel_order | 0.92 | cancel purchase, cancel, purchase cancel, purchase, purchase don, got cancel |
| 23 | 47 | check_refund_policy | 1.00 | ask, cases ask, cases, ask refund, situations ask, ask refunds |
| 30 | 47 | track_order | 1.00 | status, status purchase, purchase status, purchase current, current status, current |
| 12 | 44 | check_payment_methods | 1.00 | payment options, options, payment, allowed, accepted, accepted payment |
| 33 | 44 | get_refund | 0.73 | compensation, compensations, compensation money, money, compensations money, status compensation |
| 10 | 41 | delete_account | 0.37 | enterprise, enterprise account, account, remove enterprise, change enterprise, create enterprise |
| 22 | 41 | check_refund_policy | 1.00 | request, cases request, request refunds, cases, situations request, request refund |
| 11 | 40 | delivery_options | 1.00 | options, shipping options, shipping, shipment options, delivery options, options choose |
| 37 | 40 | switch_account | 0.30 | business, business account, use business, account, business profile, information business |
