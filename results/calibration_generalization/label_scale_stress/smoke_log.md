# Phase 1 — smoke run log

Started 2026-05-20T04:47:20Z

Config:
```
{
  "experiment_id": "label_scale_stress",
  "phase": "phase1",
  "data_source": "data/client_a_b/public_reorg_energy_15210.csv",
  "n_target": 50,
  "n_source": 80,
  "n_source_clients": 4,
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
  "kan_grid": 5,
  "encoder_out_dim": 256,
  "node_feat_dim": 11,
  "head_type": "kan",
  "device": "cuda:0",
  "bootstrap_seed_for_stats": 20260520,
  "smoke": true,
  "timestamp_utc": "2026-05-20T04:47:20Z"
}
```

## Summary by method (averaged across 1 seeds)

| method | MAE (mean ± SEM over seeds) | RMSE | R² |
|---|---|---|---|
| local | 1.1264 (n=1) | 1.1858 | -13.359 |
| fedavg_no_cal | 0.5865 (n=1) | 0.6859 | -2.831 |
| fedper | 1.0157 (n=1) | 1.0940 | -13.604 |
| fedper_cal | 0.3377 (n=1) | 0.4213 | -0.594 |

## Runtime

Total: 6.8 s = 0.11 min
