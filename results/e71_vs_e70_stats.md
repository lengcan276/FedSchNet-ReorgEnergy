# E71 vs E70

Reference: **E71** = FedPer + calibration only. Compared to: **E70** = PC² with adapter + T-gate but WITHOUT calibration.

| Target | n | MAE_E71 | MAE_other | RMSE_E71 | RMSE_other | R²_E71 | R²_other | ΔMAE (E71-other; 95% CI) | wins/loses | Wilcoxon p | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loocv_c_hole | 53 | 0.3736 | 0.3773 | 0.5083 | 0.5229 | -0.050 | -0.111 | -0.0036 (-0.0398, +0.0331) | 27/26 | 0.8421 |  |
| loocv_c_triplet | 49 | 0.6536 | 0.7785 | 0.8255 | 0.9480 | -0.227 | -0.618 | -0.1249 (-0.2242, -0.0272) | 33/16 | 0.0206 | * |

## Notes

- `wins`/`loses`: number of molecules where E71 has lower / higher abs error than the comparison.
- significance: `*` p<0.05, `**` p<0.01, `***` p<0.001 (unadjusted nominal). Bonferroni threshold for 3 comparisons would be α/3 ≈ 0.017.
- All comparisons are PAIRED on the same molecule set.
- Per-molecule errors aggregated as the MEDIAN across the 5 seeds.