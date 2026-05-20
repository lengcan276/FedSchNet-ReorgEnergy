# Calibration-generalization experiments — result files

> **Scope.** Outputs of the controlled pseudo-federation experiments
> described in `experiments/calibration_generalization/README.md`.
> **No file in this directory contains private C-hole / C-triplet
> raw labels.** Every prediction is on QM9-derived public molecules.

## Layout

```
results/calibration_generalization/
├── phase0_repo_survey.md                                 # Phase 0 read-only repo audit
├── label_scale_stress/                                   # Phase 1
│   ├── config_label_scale_stress.json
│   ├── summary_by_seed.csv                               # per-seed × method aggregates
│   ├── summary_by_target.csv                             # per-method mean ± SEM over seeds  ← paper-cite
│   ├── pairwise_stats.csv                                # paired Wilcoxon + bootstrap CI    ← paper-cite
│   ├── per_fold_results.csv                              # per-fold raw MAE / RMSE / R²
│   ├── calibration_effect_by_scale_ratio.csv             # source-vs-target std ratios
│   ├── runtime_log.csv                                   # wall-clock per (seed, method)
│   ├── smoke_log.md / full_run_log.md                    # markdown run logs
│   └── predictions/                                      # per (seed, method) y_true/y_pred JSON
│                                                         #   (20 files; pseudo-federation, public mols only)
├── pseudo_task_validation/                               # Phase 2
│   ├── config_pseudo_task_validation.json
│   ├── pseudo_task_definitions.csv                       # per-task label stats
│   ├── summary_by_seed.csv
│   ├── summary_by_task.csv                               # per-task × method mean ± SEM      ← paper-cite
│   ├── pairwise_stats.csv                                # per-task paired Wilcoxon + CI     ← paper-cite
│   ├── per_fold_results.csv
│   ├── improvement_distribution.csv                      # per (task, seed) ΔMAE cal − FedPer
│   ├── improvement_summary.csv                           # aggregate improvement rate        ← paper-cite
│   ├── runtime_log.csv
│   ├── smoke_log.md / full_run_log.md
│   └── predictions/                                      # 120 (task × method × seed) JSON files
└── calibration_variants/                                 # Phase 3 (NOT run)
    ├── config_calibration_variants.json                  # status = PROPOSAL_ONLY
    └── variant_feasibility_report.md                     # shrinkage + learnable-affine plan
```

## What to cite in the manuscript / SI

The SI §S11 cites:

- `label_scale_stress/summary_by_target.csv` (Phase 1 per-method MAE table)
- `label_scale_stress/pairwise_stats.csv` (calibration vs FedPer, calibration vs FedAvg paired stats)
- `pseudo_task_validation/improvement_summary.csv` (9/10 headline; median/IQR ΔMAE)
- `pseudo_task_validation/pairwise_stats.csv` (per-task Wilcoxon p < 0.001 on 7/10 tasks)

`paper/calibration_generalization/calibration_generalization_report.md`
auto-renders these into the unified report.

## What is logging, not result

- `*.csv` files of class `per_fold_results.csv`: per-fold raw metrics,
  useful for debugging or replication but not cited.
- `runtime_log.csv`: per-(seed, method) wall-clock seconds.
- `smoke_log.md`, `full_run_log.md`: human-readable run logs (markdown).

## What does NOT belong here

- Model checkpoints. Not produced by these experiments.
- Optimizer states. Not produced.
- Private TADF labels. None present.
- Aggregations across seeds-and-folds in headline-table form: those live
  under `paper/calibration_generalization/` so the source-CSVs stay close
  to the raw data.

## File sizes

```
868 K  total
164 K  label_scale_stress/predictions/  (20 small JSONs)
492 K  pseudo_task_validation/predictions/  (120 small JSONs)
```

No file exceeds 50 MB.

## Privacy / data-leak checks

Every prediction file is derived from
`data/client_a_b/public_reorg_energy_15210.csv` only. The construction
of pseudo-clients in `experiments/calibration_generalization/pseudo_federation.py`
draws molecule subsets from that public CSV and applies synthetic
label transforms (Phase 1) or label-quantile bins (Phase 2); it never
reads `data/client_c/*.csv`. Confirmed by code inspection and by the
`predictions/*.json` `y_true` ranges (all match the public QM9 mean ≈
0.72 eV after the documented transforms).
