# Calibration-generalization experiments — code

> **Scope.** These experiments test whether the task-specific calibration
> mechanism that distinguishes E71 from FedPer on the real TADF C-triplet
> target also helps on *controlled* pseudo-federations built from public
> QM9-derived data. **The result interpretation is regime-dependent: it
> is *not* a universal calibration superiority claim.**

## Layout

```
experiments/calibration_generalization/
├── __init__.py
├── pseudo_federation.py            # builds synthetic 5-client / 10-target federations
├── fed_train_minimal.py            # wraps src/ to train Local / FedAvg / FedPer / FedPer+cal
├── run_label_scale_stress.py       # Phase 1 driver
├── run_pseudo_task_validation.py   # Phase 2 driver
├── compute_pairwise_stats.py       # paired Wilcoxon + bootstrap CI on per-target abs errors
└── render_phase4_report.py         # auto-renders the unified report from CSVs
```

None of these files modify `src/`. They import from `src/` (data_utils,
models, federated, train_eval) read-only and add new code under this
directory only.

## Phases

- **Phase 0 — repo survey.** Read-only check of imports and data; recorded in
  `results/calibration_generalization/phase0_repo_survey.md`.
- **Phase 1 — label-scale stress.** Five clients: one target with the
  identity label transform (n_target = 100); four sources with deliberately
  heterogeneous affine + noise transforms (2y+0.5, 0.5y−0.2, 3y+1.0,
  y+N(0, 0.05)). 5 seeds × 4 methods × 5-fold CV on target.
- **Phase 2 — pseudo-task multi-target validation.** 10 label-quantile-binned
  pseudo-targets (n_per_task = 50), identity label transforms. Each
  pseudo-target is evaluated as the small client in the federation with
  the other 9 acting as sources. 3 seeds × 4 methods × 3-fold CV per target.
- **Phase 3 — calibration variants.** Specification only; **not executed**.
  See `results/calibration_generalization/calibration_variants/variant_feasibility_report.md`.

## Running

Smoke runs (sanity-check; minutes, not hours):

```
python -m experiments.calibration_generalization.run_label_scale_stress --smoke
python -m experiments.calibration_generalization.run_pseudo_task_validation --smoke --device cuda:1
```

Full runs (Phase 1 ≈ 20 min on one A30; Phase 2 ≈ 30 min on one A30):

```
python -m experiments.calibration_generalization.run_label_scale_stress --full
python -m experiments.calibration_generalization.run_pseudo_task_validation --full --device cuda:1
```

Paired statistics:

```
python -m experiments.calibration_generalization.compute_pairwise_stats --phase both
```

Unified report:

```
python -m experiments.calibration_generalization.render_phase4_report
```

## Environment

Same environment as the main paper: `H-CAAN` conda env (Python 3.10,
PyTorch 2.7.1+cu126, PyG 2.3.0, RDKit, efficient_kan).

## Seeds

- Phase 1: `[42, 123, 456, 789, 1000]`.
- Phase 2: `[42, 123, 456]`.
- Bootstrap RNG seed for `compute_pairwise_stats.py`: `20260520`.

These are hard-coded in the drivers; per-fold seeds are derived as
`seed * 100 + fold_index`. All seeding is reproducible.

## Output files

```
results/calibration_generalization/label_scale_stress/
results/calibration_generalization/pseudo_task_validation/
results/calibration_generalization/calibration_variants/
paper/calibration_generalization/calibration_generalization_report.md
```

See `results/calibration_generalization/README.md` for the result-file
layout, and `paper/calibration_generalization/README.md` for how the
report fits the manuscript.

## How to read the verdict

- **PASS-STRONG** — Phase 1 + Phase 2 both support calibration. Would
  warrant updating the manuscript abstract.
- **PASS-LIMITED** — *this study's outcome.* Phase 2 supports calibration
  (9/10 tasks improved), Phase 1 does not (n_target = 100 is large
  enough that local-only is competitive and the federation is not
  load-bearing). The conclusion is therefore **regime-dependent**:
  calibration helps when the target is small enough that federation
  matters AND when sources carry naturally heterogeneous label means.
- **FAIL** — calibration has no detectable advantage over FedPer in any
  controlled stress test. Would prevent expanding the main-paper claim
  beyond the C-triplet positive control.

## What this is *not*

- This is **not** evidence that PC²-FedReorg is universally superior
  to other federated baselines.
- This is **not** evidence that task-specific calibration universally
  improves heterogeneous federated regression.
- The Phase 1 null result is **not** a contradiction of the C-triplet
  result — it identifies the *boundary* of the calibration regime
  (n_target ≲ 50, naturally heterogeneous source label means).
- These results live alongside, not in place of, the real TADF C-hole /
  C-triplet experiments that are the manuscript's headline.
