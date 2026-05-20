# Ablation summary (mean ± SEM across 5 seeds)

| experiment | target | n_seeds | MAE_mean | MAE_sem |
| --- | --- | --- | --- | --- |
| E66 PC²-FedReorg (reference) | C-hole | 5 | 0.3842 | 0.0058 |
| E66 PC²-FedReorg (reference) | C-triplet | 5 | 0.6709 | 0.0096 |
| E65 D→C supervised pretrain | C-hole | 5 | 0.4078 | 0.0099 |
| E65 D→C supervised pretrain | C-triplet | 5 | 0.6820 | 0.0222 |
| E68 PC² − T_repr C-cross removed | C-hole | 5 | 0.3851 | 0.0083 |
| E68 PC² − T_repr C-cross removed | C-triplet | 5 | 0.6395 | 0.0128 |
| E69 PC² − uniform FedAvg gate | C-hole | 5 | 0.3690 | 0.0058 |
| E69 PC² − uniform FedAvg gate | C-triplet | 5 | 0.6253 | 0.0147 |
| E70 PC² − no calibration | C-hole | 5 | 0.3791 | 0.0039 |
| E70 PC² − no calibration | C-triplet | 5 | 0.7857 | 0.0061 |
| E71 FedPer + calibration only (positive control) | C-hole | 5 | 0.3757 | 0.0029 |
| E71 FedPer + calibration only (positive control) | C-triplet | 5 | 0.6527 | 0.0111 |
