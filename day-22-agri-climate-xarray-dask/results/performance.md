# Performance notes

## Storage chunking

| Variable | Chunk shape | Chunk MB | Chunks | Array MB |
|---|---|---|---|---|
| ndvi | (1104, 40, 60) | 10.11 | 4 | 40 |
| precip | (365, 40, 60) | 3.34 | 100 | 321 |
| tmax | (365, 40, 60) | 3.34 | 100 | 321 |
| tmin | (365, 40, 60) | 3.34 | 100 | 321 |

## Dask chunk size vs graph size, time and memory (annual GDD on the main cube, each in a fresh process, local threaded scheduler)

| Dask chunking | Chunk MB | Chunks per variable | Tasks (raw) | Tasks (optimised) | Wall s | Peak RSS increase MB |
|---|---|---|---|---|---|---|
| tiny (73, 20, 20) | 0.11 | 2904 | 36,003 | 4,059 | 26.80 | 34 |
| stored (365, 40, 60) | 3.34 | 100 | 1,271 | 171 | 1.85 | 32 |
| large (1460, 80, 120) | 53.47 | 7 | 97 | 20 | 3.30 | 512 |

## Out-of-core demonstration

- Cube: 3.92 GB uncompressed (2.68 GB on disk), chunks (730, 40, 60) = 6.68 MB.
- Every stage ran in its own process with a hard address-space cap of 1.00 GB (RLIMIT_AS): the cube is 3.9x the cap.
- Writing the cube tile by tile: 114 s, peak RSS 751 MB.
- Naive `ds.load()`: **MemoryError**.
- Chunked GDD + heat days + JJA precip + SPI + annual precip: **succeeded** in 13.5 s, peak RSS 193 MB (4.8% of the cube).
- Graph tasks per feature: {'gdd': 2607, 'heat_days': 1150, 'precip_jja': 734, 'precip_annual': 734}.

## Seasonal sums on the big cube: `resample().sum()` vs indicator-matrix contraction (uncapped child processes)

| Method | Tasks (raw) | Wall s | Peak RSS increase MB |
|---|---|---|---|
| resample | 5,693 | 7.32 | 623 |
| dot | 2,607 | 5.79 | 77 |

Same result (checksums agree to 1e-6): True.
