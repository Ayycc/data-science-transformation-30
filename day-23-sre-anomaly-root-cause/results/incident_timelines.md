# Incident timelines (auto-generated, SYNTHETIC)

First 12 detected incidents. Ranking = ensemble + deploy prior.

## Incident 0: day 3 01:53

- **First alert** day 3 01:53; 2 services alerting: user-profile, web.
- **Change-point onsets (PELT)**: user-profile (01:46).
- **Deploys nearby**: user-profile at 01:46.
- **Likely root cause**: user-profile (95% of posterior mass), then web, user-db.
- *Ground truth (synthetic)*: user-profile / cpu_saturation, injected 01:51 -> correct.

## Incident 1: day 3 09:36

- **First alert** day 3 09:36; 2 services alerting: notification, shipping.
- **Change-point onsets (PELT)**: notification (09:26) -> shipping (09:37).
- **Deploys nearby**: notification at 09:26.
- **Likely root cause**: notification (98% of posterior mass), then message-queue, shipping.
- *Ground truth (synthetic)*: notification / cpu_saturation, injected 09:34 -> correct.

## Incident 2: day 3 14:42

- **First alert** day 3 14:42; 2 services alerting: api-gateway, cache.
- **Change-point onsets (PELT)**: cache (14:35) -> catalog (14:38).
- **Deploys nearby**: message-queue at 14:27.
- **Likely root cause**: cache (51% of posterior mass), then api-gateway, catalog.
- *Ground truth (synthetic)*: message-queue / queue_backlog, injected 14:34 -> missed.

## Incident 3: day 3 14:42

- **First alert** day 3 14:42; 2 services alerting: api-gateway, cache.
- **Change-point onsets (PELT)**: cache (14:35) -> catalog (14:38).
- **Deploys nearby**: message-queue at 14:27.
- **Likely root cause**: cache (51% of posterior mass), then api-gateway, catalog.
- *Ground truth (synthetic)*: cache / queue_backlog, injected 14:35 -> correct.

## Incident 4: day 3 23:54

- **First alert** day 3 23:54; 1 services alerting: web.
- **Change-point onsets (PELT)**: web (23:15).
- **Deploys nearby**: none in the 30 minutes before the first alert.
- **Likely root cause**: web (76% of posterior mass), then api-gateway.
- *Ground truth (synthetic)*: web / slow_leak, injected 23:22 -> correct.

## Incident 5: day 4 03:38

- **First alert** day 4 03:38; 2 services alerting: api-gateway, inventory.
- **Change-point onsets (PELT)**: cart (03:33) -> inventory (03:36).
- **Deploys nearby**: none in the 30 minutes before the first alert.
- **Likely root cause**: inventory (39% of posterior mass), then cart, catalog-db.
- *Ground truth (synthetic)*: inventory / cpu_saturation, injected 03:36 -> correct.

## Incident 6: day 4 11:30

- **First alert** day 4 11:30; 1 services alerting: notification.
- **Change-point onsets (PELT)**: notification (11:19).
- **Deploys nearby**: notification at 11:19, mobile-bff at 11:28.
- **Likely root cause**: notification (98% of posterior mass), then message-queue.
- *Ground truth (synthetic)*: notification / latency_regression, injected 11:26 -> correct.

## Incident 7: day 4 17:14

- **First alert** day 4 17:14; 1 services alerting: shipping.
- **Change-point onsets (PELT)**: shipping (17:10).
- **Deploys nearby**: none in the 30 minutes before the first alert.
- **Likely root cause**: shipping (83% of posterior mass), then notification.
- *Ground truth (synthetic)*: shipping / queue_backlog, injected 17:09 -> correct.

## Incident 8: day 4 22:22

- **First alert** day 4 22:22; 2 services alerting: api-gateway, auth.
- **Change-point onsets (PELT)**: auth (22:05).
- **Deploys nearby**: auth at 22:05.
- **Likely root cause**: auth (92% of posterior mass), then user-db, catalog.
- *Ground truth (synthetic)*: auth / slow_leak, injected 22:12 -> correct.

## Incident 9: day 5 06:27

- **First alert** day 5 06:27; 2 services alerting: api-gateway, inventory.
- **Change-point onsets (PELT)**: inventory (06:23).
- **Deploys nearby**: none in the 30 minutes before the first alert.
- **Likely root cause**: inventory (60% of posterior mass), then api-gateway, catalog-db.
- *Ground truth (synthetic)*: inventory / cpu_saturation, injected 06:23 -> correct.

## Incident 10: day 5 12:29

- **First alert** day 5 12:29; 2 services alerting: cache, catalog-db.
- **Change-point onsets (PELT)**: cache (12:16) -> catalog-db (12:23).
- **Deploys nearby**: cache at 12:16, catalog-db at 12:23, api-gateway at 12:24, catalog at 12:34.
- **Likely root cause**: cache (66% of posterior mass), then catalog-db.
- *Ground truth (synthetic)*: cache / slow_leak, injected 12:22 -> correct.

## Incident 11: day 5 12:29

- **First alert** day 5 12:29; 2 services alerting: cache, catalog-db.
- **Change-point onsets (PELT)**: cache (12:16) -> catalog-db (12:23).
- **Deploys nearby**: cache at 12:16, catalog-db at 12:23, api-gateway at 12:24, catalog at 12:34.
- **Likely root cause**: cache (66% of posterior mass), then catalog-db.
- *Ground truth (synthetic)*: catalog-db / queue_backlog, injected 12:26 -> in top 3.
