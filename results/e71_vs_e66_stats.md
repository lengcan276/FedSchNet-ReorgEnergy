# E71 vs E66

Reference: **E71** = FedPer + calibration only. Compared to: **E66** = PC²-FedReorg (full architecture).

| Target | n | MAE_E71 | MAE_other | RMSE_E71 | RMSE_other | R²_E71 | R²_other | ΔMAE (E71-other; 95% CI) | wins/loses | Wilcoxon p | sig |
|---|---|---|---|---|---|---|---|---|---|---|---|
| loocv_c_hole | 53 | 0.3736 | 0.3820 | 0.5083 | 0.5220 | -0.050 | -0.107 | -0.0084 (-0.0264, +0.0093) | 31/22 | 0.4230 |  |
| loocv_c_triplet | 49 | 0.6536 | 0.6595 | 0.8255 | 0.8230 | -0.227 | -0.220 | -0.0059 (-0.0437, +0.0293) | 23/26 | 0.9293 |  |

## Notes

- `wins`/`loses`: number of molecules where E71 has lower / higher abs error than the comparison.
- significance: `*` p<0.05, `**` p<0.01, `***` p<0.001 (unadjusted nominal). Bonferroni threshold for 3 comparisons would be α/3 ≈ 0.017.
- All comparisons are PAIRED on the same molecule set.
- Per-molecule errors aggregated as the MEDIAN across the 5 seeds.