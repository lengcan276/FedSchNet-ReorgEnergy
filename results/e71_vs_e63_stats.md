# E71 vs E63

Reference: **E71** = FedPer + calibration only. Compared to: **E63** = FedPer legacy baseline (no calibration).

| Target | n | MAE_E71 | MAE_other | RMSE_E71 | RMSE_other | R²_E71 | R²_other | ΔMAE (E71-other; 95% CI) | wins/loses | Wilcoxon p | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loocv_c_hole | 53 | 0.3736 | 0.3682 | 0.5083 | 0.5068 | -0.050 | -0.044 | +0.0054 (-0.0271, +0.0391) | 24/29 | 0.5984 |  |
| loocv_c_triplet | 49 | 0.6536 | 0.7621 | 0.8255 | 0.9307 | -0.227 | -0.559 | -0.1085 (-0.2083, -0.0097) | 32/17 | 0.0277 | * |

## Notes

- `wins`/`loses`: number of molecules where E71 has lower / higher abs error than the comparison.
- significance: `*` p<0.05, `**` p<0.01, `***` p<0.001 (unadjusted nominal). Bonferroni threshold for 3 comparisons would be α/3 ≈ 0.017.
- All comparisons are PAIRED on the same molecule set.
- Per-molecule errors aggregated as the MEDIAN across the 5 seeds.