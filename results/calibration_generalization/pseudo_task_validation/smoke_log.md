# Phase 2 — smoke run log

Config:
```
{
  "experiment_id": "pseudo_task_validation",
  "phase": "phase2",
  "data_source": "data/client_a_b/public_reorg_energy_15210.csv",
  "n_per_task": 30,
  "n_tasks": 4,
  "n_folds": 3,
  "seeds": [
    42
  ],
  "n_rounds": 2,
  "n_local_epochs": 1,
  "lr": 0.001,
  "batch_size": 64,
  "methods": [
    "local",
    "fedavg_no_cal",
    "fedper",
    "fedper_cal"
  ],
  "bin_strategy": "label_quantile",
  "device": "cuda:0",
  "bootstrap_seed_for_stats": 20260520,
  "smoke": true,
  "timestamp_utc": "2026-05-20T04:50:18Z"
}
```

## Per-method mean MAE over all pseudo-targets and seeds

| method | MAE | n |
|---|---|---|
| local | 1.1839 | 12 |
| fedavg_no_cal | 0.6579 | 12 |
| fedper | 0.6763 | 12 |
| fedper_cal | 0.1226 | 12 |

## Calibration vs FedPer improvement rate: 4/4 (target × seed)

## Runtime
Total: 15 s (0.2 min)
