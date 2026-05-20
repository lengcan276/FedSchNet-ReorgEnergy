# Phase 2 — full run log

Config:
```
{
  "experiment_id": "pseudo_task_validation",
  "phase": "phase2",
  "data_source": "data/client_a_b/public_reorg_energy_15210.csv",
  "n_per_task": 50,
  "n_tasks": 10,
  "n_folds": 3,
  "seeds": [
    42,
    123,
    456
  ],
  "n_rounds": 8,
  "n_local_epochs": 3,
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
  "smoke": false,
  "timestamp_utc": "2026-05-20T04:51:34Z"
}
```

## Per-method mean MAE over all pseudo-targets and seeds

| method | MAE | n |
|---|---|---|
| local | 0.1008 | 90 |
| fedavg_no_cal | 0.3586 | 90 |
| fedper | 0.0827 | 90 |
| fedper_cal | 0.0640 | 90 |

## Calibration vs FedPer improvement rate: 26/30 (target × seed)

## Runtime
Total: 1218 s (20.3 min)
