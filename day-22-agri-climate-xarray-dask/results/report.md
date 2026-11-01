# Day 22 run report (SYNTHETIC data)

Data path used: **synthetic_generator** (public probe: URLError: <urlopen error Tunnel connection failed: 403 Forbidden>).
Cube: {'time_ndvi': 1104, 'lat': 80, 'lon': 120, 'time': 8766} = 0.98 GB uncompressed, 689 MB in Zarr. 300 fields, 7200 field-years.

## Yield model cross-validation

| Model | CV scheme | RMSE (t/ha) | MAE (t/ha) | R2 |
|---|---|---|---|---|
| nowcast_with_ndvi | random | 0.653 | 0.520 | 0.897 |
| nowcast_with_ndvi | spatial | 0.819 | 0.644 | 0.838 |
| climate_only | random | 0.640 | 0.509 | 0.901 |
| climate_only | spatial | 0.996 | 0.763 | 0.760 |
| predict the mean | - | 2.034 | - | 0 |

Spatial-block RMSE is 25.5% higher than random-CV RMSE for the nowcast model and 55.5% higher for the climate-only model (1.0 degree blocks).

## Climate scenario: +2.0 C, precipitation x0.8

| Source | Climate | Mean (t/ha) | P5 (t/ha) | YaR95 = mean - P5 (t/ha) | CVaR95 (t/ha) | P(yield < 80% of field baseline mean) |
|---|---|---|---|---|---|---|
| model | baseline | 10.15 | 6.65 | 3.50 | 5.83 | 11.3% |
| model | scenario | 7.90 | 4.82 | 3.08 | 4.54 | 58.3% |
| truth | baseline | 10.09 | 6.62 | 3.48 | 5.79 | 11.9% |
| truth | scenario | 7.69 | 4.04 | 3.65 | 3.55 | 59.7% |

Model mean change -22.1% vs truth -23.8% (gap +1.7 pp). Fields with a mean loss over 20%: 195.

Mean feature shift under the scenario: gdd +265.0, heat_days +20.8, precip_jja -67.8, spi3_aug -0.7.

Share of scenario field-years outside the training range: gdd 20.4%, heat_days 3.6%, precip_jja 0.2%, spi3_aug 0.8%.
