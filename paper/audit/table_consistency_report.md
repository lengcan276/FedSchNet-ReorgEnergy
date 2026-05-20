# Table consistency report

- Total checks: **260**
- PASS: **249**
- FAIL: **11**
- NA: **0**

Tolerances: MAE/RMSE/R² ≤ 0.001 abs; mae_diff ≤ 0.001; CI bounds ≤ 0.002; p ≤ 0.001; wins/losses exact.

## FAIL rows

| table | claim | field | paper | recomputed | abs_diff | tol | note |
|---|---|---|---|---|---|---|---|
| table3 | E66 vs E60 / loocv_c_triplet | ci_low | -0.2297 | -0.232017 | 0.002317 | 0.002 | bootstrap 5000 resamples; seed may differ |
| table3 | E66 vs E61 / loocv_c_triplet | ci_low | -0.2195 | -0.224296 | 0.004796 | 0.002 | bootstrap 5000 resamples; seed may differ |
| table3 | E66 vs E61 / loocv_c_triplet | ci_high | +0.0043 | 0.012279 | 0.007979 | 0.002 |  |
| table3 | E66 vs E62 / loocv_c_triplet | ci_low | -0.2542 | -0.258529 | 0.004329 | 0.002 | bootstrap 5000 resamples; seed may differ |
| table3 | E66 vs E62 / loocv_c_triplet | ci_high | -0.0069 | 0.002385 | 0.009285 | 0.002 |  |
| table3 | E66 vs E63 / loocv_c_triplet | ci_low | -0.2049 | -0.208644 | 0.003744 | 0.002 | bootstrap 5000 resamples; seed may differ |
| table3 | E66 vs E63 / loocv_c_triplet | ci_high | -0.0023 | 0.007449 | 0.009749 | 0.002 |  |
| table3 | E66 vs E70 / loocv_c_triplet | ci_high | -0.0153 | -0.009139 | 0.006161 | 0.002 |  |
| table3 | E71 vs E63 / loocv_c_triplet | ci_low | -0.2083 | -0.210883 | 0.002583 | 0.002 | bootstrap 5000 resamples; seed may differ |
| table3 | E71 vs E63 / loocv_c_triplet | ci_high | -0.0097 | -0.001956 | 0.007744 | 0.002 |  |
| table3 | E71 vs E70 / loocv_c_triplet | ci_high | -0.0272 | -0.021381 | 0.005819 | 0.002 |  |
