# PC²-FedReorg Phase 5 statistical comparison

Reference: **E66** (PC²-FedReorg)

Per-molecule abs-error aggregated as median across seeds. 
Negative `mae_diff` (= MAE_ref − MAE_other) means PC² has lower MAE.

Wilcoxon signed-rank: paired, two-sided. Bootstrap CI on MAE diff with N=5000 resamples.

## Target: `loocv_c_hole` (n=53 molecules)

| Comparison | MAE_ref | MAE_other | mae_diff (95% CI) | wins/loses | Wilcoxon p | sig? |
|---|---|---|---|---|---|---|
| E66 vs E60 | 0.3820 | 0.4137 | -0.0316 (-0.1148, +0.0500) | 25/28 | 0.5801 |  |
| E66 vs E61 | 0.3820 | 0.3846 | -0.0026 (-0.0421, +0.0391) | 27/26 | 0.8979 |  |
| E66 vs E62 | 0.3820 | 0.3861 | -0.0041 (-0.0555, +0.0444) | 28/25 | 0.8214 |  |
| E66 vs E63 | 0.3820 | 0.3682 | +0.0138 (-0.0214, +0.0487) | 23/30 | 0.1977 |  |
| E66 vs E65 | 0.3820 | 0.4028 | -0.0208 (-0.0992, +0.0552) | 28/25 | 0.7600 |  |
| E66 vs E68 | 0.3820 | 0.3821 | -0.0001 (-0.0173, +0.0175) | 30/23 | 0.5620 |  |
| E66 vs E69 | 0.3820 | 0.3700 | +0.0120 (-0.0203, +0.0427) | 26/27 | 0.4282 |  |
| E66 vs E70 | 0.3820 | 0.3773 | +0.0048 (-0.0333, +0.0425) | 23/30 | 0.5801 |  |

## Target: `loocv_c_triplet` (n=49 molecules)

| Comparison | MAE_ref | MAE_other | mae_diff (95% CI) | wins/loses | Wilcoxon p | sig? |
|---|---|---|---|---|---|---|
| E66 vs E60 | 0.6595 | 0.7471 | -0.0876 (-0.2297, +0.0534) | 28/21 | 0.1745 |  |
| E66 vs E61 | 0.6595 | 0.7630 | -0.1035 (-0.2195, +0.0043) | 31/18 | 0.0449 | * |
| E66 vs E62 | 0.6595 | 0.7837 | -0.1243 (-0.2542, -0.0069) | 30/19 | 0.0449 | * |
| E66 vs E63 | 0.6595 | 0.7621 | -0.1026 (-0.2049, -0.0023) | 32/17 | 0.0417 | * |
| E66 vs E65 | 0.6595 | 0.6699 | -0.0104 (-0.1421, +0.1168) | 27/22 | 0.8514 |  |
| E66 vs E68 | 0.6595 | 0.6329 | +0.0266 (+0.0006, +0.0551) | 17/32 | 0.0700 |  |
| E66 vs E69 | 0.6595 | 0.6162 | +0.0433 (-0.0007, +0.0916) | 21/28 | 0.1809 |  |
| E66 vs E70 | 0.6595 | 0.7785 | -0.1190 (-0.2251, -0.0153) | 35/14 | 0.0427 | * |

## Notes

- `wins/loses`: number of molecules where reference has lower / higher abs error than the comparison.
- significance: `*` p<0.05, `**` p<0.01, `***` p<0.001.
- All comparisons are PAIRED on the same molecule set.
- Per-molecule errors are the MEDIAN across the 5 seeds (robust to seed outliers).