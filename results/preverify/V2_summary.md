# V2 — Classical-ML baselines on Client C

5-seed × LOOCV. Morgan radius=2, nBits=2048.

## hole (n=53)

| model | MAE (mean ± std) | RMSE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| RF | 0.4084 ± 0.0038 | 0.5729 ± 0.0051 | -0.334 ± 0.024 |
| GP | 0.3965 ± 0.0000 | 0.5319 ± 0.0000 | -0.150 ± 0.000 |
| KNN | 0.3674 ± 0.0000 | 0.5206 ± 0.0000 | -0.101 ± 0.000 |
| XGB | 0.4694 ± 0.0070 | 0.6459 ± 0.0062 | -0.696 ± 0.032 |

## triplet (n=49)

| model | MAE (mean ± std) | RMSE (mean ± std) | R² (mean ± std) |
|---|---|---|---|
| RF | 0.5518 ± 0.0050 | 0.7134 ± 0.0047 | +0.084 ± 0.012 |
| GP | 0.5921 ± 0.0000 | 0.7608 ± 0.0000 | -0.042 ± 0.000 |
| KNN | 0.6371 ± 0.0000 | 0.7874 ± 0.0000 | -0.116 ± 0.000 |
| XGB | 0.5841 ± 0.0079 | 0.7196 ± 0.0081 | +0.068 ± 0.021 |

## Verdict

- Best classical R² on C-hole: **-0.101** (vs E52 SchNet+KAN+FedPer: 0.139)
- Best classical R² on C-triplet: **+0.084**

**Possible data ceiling.** All non-deep methods fail too; investigate label noise / DFT protocol heterogeneity / conformer dependence on Client C before more model work.