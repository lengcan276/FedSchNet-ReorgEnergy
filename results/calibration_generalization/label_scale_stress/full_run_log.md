# Phase 1 — full run log

Started 2026-05-20T04:47:48Z

Config:
```
{
  "experiment_id": "label_scale_stress",
  "phase": "phase1",
  "data_source": "data/client_a_b/public_reorg_energy_15210.csv",
  "n_target": 100,
  "n_source": 200,
  "n_source_clients": 4,
  "n_folds": 5,
  "seeds": [
    42,
    123,
    456,
    789,
    1000
  ],
  "n_rounds": 12,
  "n_local_epochs": 5,
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
  "smoke": false,
  "timestamp_utc": "2026-05-20T04:47:48Z"
}
```

## Summary by method (averaged across 5 seeds)

| method | MAE (mean ± SEM over seeds) | RMSE | R² |
|---|---|---|---|
| local | 0.3090 ± 0.0096 | 0.4129 | -0.335 |
| fedavg_no_cal | 0.3422 ± 0.0123 | 0.4011 | -0.247 |
| fedper | 0.3575 ± 0.0402 | 0.4525 | -0.963 |
| fedper_cal | 0.3402 ± 0.0210 | 0.4344 | -0.649 |

## Runtime

Total: 1238.3 s = 20.64 min
