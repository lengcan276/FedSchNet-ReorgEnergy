# Supporting Information — PC²-FedReorg

**Manuscript:** *Protocol-Calibrated Personalized Federated Learning for Molecular Reorganization Energy Prediction*

> Outline only — to be expanded into a full SI document after the main manuscript is finalised.  Every numbered subsection lists the artefacts that already exist in this repository so the SI prose can be backed by the same files that produced the main-text figures and tables.

---

## S1. Dataset and task-client details

### S1.1 Per-node definitions

For each of the five task-client nodes (A, B, C-hole, C-triplet, D), provide:

- formal source citation and dataset URL / DOI;
- SMILES preprocessing rules (RDKit MolFromSmiles, charge sanitisation, valence check, removal of invalid entries);
- aromatic-ring split rule for A/B (`Chem.rdMolDescriptors.CalcNumAromaticRings`);
- per-node sample count after filtering (`n_kept` from `results/preverify/T_transferability.json`):
  A = 6,020 / B = 9,190 / C-hole = 53 / C-triplet = 49 / D = 5,876.

**Artefacts:** `data/client_a_b/public_reorg_energy_15210.csv`, `../logs/hole_reorg_molecular.csv`, `../logs/reorganization_energy_summary.csv`, `../logs/atahan_reorg_5876.csv`.

### S1.2 Label distributions

- Per-node λ histogram and Q-Q plot.
- Per-node summary statistics (mean, std, min, max, IQR) — mirror Table 1 in the main text with extended quantiles.
- Distribution-overlap visualisation: per-pair Wasserstein-1 distances on the label space.

**Artefacts:** `paper/figures/Fig1_task_client_molecules.{pdf,svg,png}`; numerical accompaniment in `paper/tables/table1_task_clients.{csv,md}`.

### S1.3 Representative molecules

- Reproduce the deterministic median-λ selection rule used in main-text Figure 1.
- List the chosen SMILES, label, and chemistry description for each node (e.g. for C-triplet the median molecule has λ = 2.747 eV).
- Show alternative representatives at quartile boundaries (Q1, Q3) for chemists who want a richer scaffold view.

**Artefacts:** chosen SMILES already printed by `paper/make_figures.py` (logged in stdout).

### S1.4 Protocol notes

- Full DFT-protocol metadata table extracted from `T_transferability.json` (`protocol_meta` entries) plus literature references for each public source.
- Explicit listing of *unknown* fields per node — particularly for D (geometry, charge state, conformer policy).

**Artefacts:** `results/preverify/T_transferability.json` keys `protocol_meta.*`.

---

## S2. Pre-verification results (V1 / V2 / V3)

### S2.1 V1: label and protocol divergence between D and C

- Two-sample Kolmogorov–Smirnov test on D-hole-λ vs. C-hole-λ label distributions; report KS statistic, p-value (≈ 1.1 × 10⁻¹¹⁶), and 95 % CI on the difference of means.
- DFT-protocol divergence summary: functional, basis, geometry, charge state, conformer policy.
- Conclusion: the V1 divergence motivated the conservative T_head clauses in main-text §2.4.

**Artefacts:** `results/preverify/V1_label_protocol.md` if present in the repo, otherwise reconstruct from raw CSVs.

### S2.2 V2: classical C-only baselines and data-ceiling diagnosis

For C-hole (n = 53) and C-triplet (n = 49), train and evaluate:

- Morgan fingerprint + Random Forest;
- Morgan fingerprint + Gaussian Process;
- Morgan fingerprint + K-Nearest Neighbours;
- XGBoost on Morgan + RDKit descriptors (if `xgboost` is available).

Report MAE / RMSE / R² with 5-fold and LOOCV protocols.  Headline numbers used in main-text §4.3:

- C-hole best classical R² = −0.101 (KNN);
- C-triplet best classical R² = +0.084 (RF).

These define the small-sample data ceiling against which the federated methods are positioned.

**Artefacts:** `results/preverify/V2_classical_baselines.csv` (to be regenerated from `experiments/preverify.py --v2`).

### S2.3 V3: chemical-space overlap

- Morgan-Tanimoto K matrix and Murcko-scaffold S matrix for all 5 × 5 task-client pairs.
- Per-C-molecule median max-Tanimoto to D (≈ 0.120 / 0.118 for hole / triplet).
- Scaffold-overlap statistics: 0.0 % of C scaffolds appear in D (and vice versa).

**Artefacts:** `results/preverify/T_chemistry.json`.

### S2.4 T_chemistry and T_transferability artefacts

- Full 5 × 5 K and S matrices (Table SX).
- Full 5 × 5 T_repr, T_adapter, T_head matrices (Table SX) with SHA-256 of the JSON file for reproducibility.
- List of every off-diagonal entry of T_head that equals zero, with the firing clause noted next to each.

**Artefacts:** `results/preverify/T_chemistry.json`, `results/preverify/T_transferability.json`.

---

## S3. Model and implementation details

### S3.1 GIN encoder

- Layer-by-layer description (5 × GINConv → BatchNorm → ReLU; hidden dim 300; JumpingKnowledge concat; mean+max pooling; projection MLP to 256-d).
- Node feature definition: 11-dim atomic descriptor (atom-type one-hot, degree, formal charge, hybridisation, aromaticity, in-ring flag, total Hs, Gasteiger partial charge) extended to 15-d when HOMO/LUMO/gap/dipole node features are enabled — zero-padded for nodes outside C.
- Codebase note: SchNet 3D encoder class exists in `src/models.py:413` and shares the same `forward(...)` signature; *no SchNet results are reported in this manuscript* — included here only as repository documentation.

**Artefacts:** `src/models.py:22-260` (GINEncoder + ReorgEnergyModel).

### S3.2 KAN head settings

- `efficient-kan` with `grid = 5`, `spline_order = 3`, scale_noise = 0.1, scale_base = 1.0.
- Three-layer KAN: 256 → 64 → 32 → 1.
- Initialisation: B-spline coefficients standardised to unit norm; base linear layer Xavier-normal.

**Artefacts:** `src/models.py:323-410`.

### S3.3 Residual adapter

- 256 → 128 → 256 with zero-initialised final projection so that the adapter is initially the identity.
- Activation: GELU; LayerNorm before residual addition.
- Optional toggle: `use_adapter=False` recovers the FedPer-equivalent architecture.

**Artefacts:** `src/models.py`, `AdapterLayer` class.

### S3.4 Calibration buffer

- Two non-trainable buffers `μ`, `σ` registered via `register_buffer`.
- `σ` floor = 1 × 10⁻⁸ to avoid division-by-zero at degenerate folds.
- Install function: `model.set_calibration(mu, sigma)` is called from each LOOCV fold's training routine with the *training-only* labels of that fold.
- Forward path: `z_pred = forward(x)`; `λ_pred = σ · z_pred + μ`.

**Artefacts:** `CalibrationHead` class in `src/models.py`; unit tests in `tests/test_calibration_isolation.py`.

### S3.5 z-space loss / eV-space metrics

- Training loss `L = MSE(model_z(x), (λ − μ_c) / σ_c)`, with `μ_c` and `σ_c` installed from the current fold's training data.
- Evaluation always performed in eV-space after `λ̂ = σ · z + μ` inversion.
- Verification: round-trip identity test in `tests/test_calibration_isolation.py` — fixed input → forward → inverse → original label tolerance 1 × 10⁻⁵ eV.

**Artefacts:** `src/train_eval.py:train_one_epoch`, `compute_metrics`, `evaluate`.

### S3.6 Aggregation-exclusion rules

- `_get_exclude_keys(state_dict)` returns the set of keys to skip during aggregation.
- For PC²: `*calibration*` ∪ `*.bn.*` ∪ `*.norm.*` are universally excluded; head/adapter/encoder are routed per the T matrix (Section 2.3 of main text).
- For each legacy strategy (FedAvg / FedProx / FedPer / FedBN) the same exclusion filter is applied via the dispatch in `get_aggregation_fn`.

**Artefacts:** `src/federated.py:_get_exclude_keys`, `src/federated.py:get_aggregation_fn`.

### S3.7 Unit tests and leakage checks

Include a table listing the 93 unit tests (22 transferability + 18 PC² aggregator + 33 train-eval + 20 calibration), each with:

- short description (e.g. *T_head[D, C-hole] == 0 invariant*);
- pass / fail status (all 93 pass);
- file path within `tests/`.

Particular emphasis on LOOCV-leakage checks:

- per-fold calibration is computed strictly from `(n − 1)` training labels;
- the held-out test molecule never appears in `set_calibration(mu, sigma)`;
- the aggregated state_dict re-installs calibration values from each task-client's own current fold, *not* from any aggregated source.

**Artefacts:** `tests/test_*.py` (paths to be enumerated).

---

## S4. Full main results

### S4.1 5-seed summaries

For every (method, target, seed) cell, report:

- MAE_mean ± std (across 5 folds for A/B, single value for C LOOCV);
- RMSE_mean ± std;
- R² mean ± std (auxiliary — explicit caveat that small-n LOOCV R² is unreliable).

**Artefacts:** `results/pc2_batch1_multiseed_summary.csv` (25 rows × 4 targets) and `results/pc2_phase5_ablation_summary.csv` (20 rows × 2 C targets).

### S4.2 A and B 5-fold breakdown

- One row per (method, A_5fold) and (method, B_5fold) cell across the 5 seeds.
- Compare against table 2 of the main text; SI version reports the per-seed numbers individually rather than the mean ± SEM aggregate.

### S4.3 C-hole and C-triplet LOOCV breakdown

- One row per (method, C-hole / C-triplet, seed) cell.
- Per-seed MAE for each method; visualise as a small-multiples strip plot to give reviewers a feel for seed-to-seed noise.

### S4.4 R² caveat figure

- Per-seed R² distributions for C-hole and C-triplet under each method, demonstrating that R² occasionally drifts below zero and is therefore unsuitable as a primary metric at this sample size.

---

## S5. Paired statistical tests

### S5.1 Methodology

- Paired Wilcoxon signed-rank (two-sided, `zero_method='wilcox'`) on the per-molecule absolute errors aggregated as the median across seeds.
- Bootstrap 95 % CI on the paired MAE difference: 5,000 resamples of molecule indices.
- Win / loss / tie counts per molecule.

**Artefacts:** `experiments/run_phase5_step5_stats.py`.

### S5.2 Full paired table for C-hole

Reproduce all eight paired comparisons (PC² vs. {E60, E61, E62, E63, E65, E68, E69, E70}) with mae_ref, mae_other, mae_diff, CI low/high, wins, loses, Wilcoxon p, significance label.  Confirm all p ≥ 0.20 after rounding.

**Artefacts:** `results/pc2_phase5_stats.csv` rows where `target == 'loocv_c_hole'`; also rendered in `paper/tables/table3_paired_tests.{csv,md}`.

### S5.3 Full paired table for C-triplet

Reproduce all eight paired comparisons on C-triplet.  Highlight the four statistically significant comparisons (vs. E61 p = 0.045; vs. E62 p = 0.045; vs. E63 p = 0.042; vs. E70 p = 0.043) and note the four non-significant comparisons (vs. E60 p = 0.17; vs. E65 p = 0.85; vs. E68 p = 0.07; vs. E69 p = 0.18).

**Artefacts:** `results/pc2_phase5_stats.csv` rows where `target == 'loocv_c_triplet'`.

### S5.4 C-hole non-significance interpretation

Direct narrative:

- MAE band on C-hole spans only 0.368–0.414 eV across nine evaluated methods (E60–E70).
- Bootstrap CIs straddle zero in every paired comparison.
- This matches the V2 classical-baseline ceiling (best classical R² = −0.10).
- We explicitly do *not* claim that PC² improves C-hole.

### S5.5 C-triplet significance interpretation

- PC² achieves MAE 0.659 vs. 0.762–0.784 eV for FedAvg / FedProx / FedPer.
- Relative reduction approximately 13–16 %.
- All three paired tests fall below the 0.05 significance threshold (p = 0.042–0.045).
- Win counts 30–32 out of 49 molecules.

---

## S6. Ablation results

### S6.1 Tabular summary

For each ablation (E65, E68, E69, E70) on each C target, report per-seed and aggregated MAE/RMSE/R², plus the paired statistic vs. PC² (E66).

**Artefacts:** `results/pc2_phase5_ablation_summary.csv`, `results/pc2_phase5_ablation_predictions.json`, `paper/tables/table4_ablation.{csv,md}`.

### S6.2 E70 — task-specific calibration removed

Specification:

- `use_calibration=False` in the PC² config;
- pc2_fed routing unchanged; adapter on; encoder/adapter/head shared as in main PC²;
- training and evaluation both performed in raw eV-space.

Result:

- C-triplet: ΔMAE = −0.119 eV, 95 % CI (−0.225, −0.015), Wilcoxon p = 0.043, 35 / 14 wins.
- C-hole: ΔMAE = +0.005 eV, 95 % CI (−0.033, +0.042), Wilcoxon p = 0.58, 23 / 30 wins.

This is the only ablation with a statistically significant effect; the effect is confined to C-triplet.

### S6.3 E68 — T_repr C-cross removed

Specification:

- A modified `T_transferability_no_ccross.json` artefact in which T_repr[C-hole][C-triplet] and T_repr[C-triplet][C-hole] are floored to 10⁻³;
- All other entries identical to the main T matrix.

Result: no statistically significant difference vs. PC² on either C target (C-triplet p = 0.07, C-hole p = 0.56).

### S6.4 E69 — uniform FedAvg aggregation

Specification:

- `use_adapter=True`, `use_calibration=True` retained;
- `aggregation_strategy` set to `fedavg` (uniform weights);
- T matrices not consulted during aggregation.

Result: no statistically significant difference vs. PC² on either C target.  Empirically establishes that the chemistry-driven gate weighting is not the source of PC²'s C-triplet improvement.

### S6.5 E65 — D → C supervised pretrain control

Specification:

- Stage 1: train a single GIN + KAN model on all 5,876 D molecules for 50 epochs;
- Stage 2: per-fold LOOCV fine-tune on C (each fold uses the n − 1 training C molecules, with the D-pretrained weights as the starting state).

Result: no statistically significant difference vs. PC² on either C target.  Per-fold MAE comparable to PC².  Conclusion: a naive D → C transfer is not significantly worse than PC² in our setting — we therefore avoid claiming that D necessarily produces negative transfer to C.

### S6.6 Composite forest plot

Reproduce main-text Figure 5 with all four ablation × two-target cells; include numerical labels for ΔMAE and CI on each row.

---

## S7. Per-molecule prediction artefacts

### S7.1 Reference table

Full 49-row C-triplet and 53-row C-hole tables listing:

- molecule index, SMILES, true λ;
- per-method per-seed predicted λ (E60 / E61 / E62 / E63 / E66, plus ablations);
- per-method median absolute error across the 5 seeds.

**Artefacts:** `results/pc2_batch1_multiseed_predictions.json`, `results/pc2_phase5_ablation_predictions.json`.

### S7.2 Top-improved and top-degraded molecules

- Re-render the six panels of main-text Figure 6 with extended captions including the SMILES.
- Optional: rank-ordered list of all 49 C-triplet molecules by ΔMAE (PC² − FedAvg) with chemistry annotations from RDKit (functional groups, scaffold class).

**Artefacts:** `paper/figures/Fig6_molecule_error_cases.{pdf,svg,png}`.

### S7.3 Failure-case analysis

For each of the top-3 degraded molecules:

- highlight the chemistry feature that may explain the degradation;
- check whether the molecule's true λ falls outside the C-triplet training-fold distribution (potential out-of-distribution case).

---

## S8. Reproducibility checklist

### S8.1 Code paths

List of every file that participates in producing main-text figures and tables:

- `src/models.py`, `src/federated.py`, `src/train_eval.py`, `src/data_utils.py`, `src/transferability.py`;
- `experiments/configs.py`, `experiments/run_all.py`, `experiments/run_pc2_multiseed.py`, `experiments/run_phase5_step3_4.py`, `experiments/run_phase5_step5_stats.py`;
- `paper/make_figures.py`.

### S8.2 Seeds

Five-seed set: 42, 123, 456, 789, 1000.  `set_all_seeds(seed)` calls `random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and `torch.backends.cudnn.deterministic = True`.

### S8.3 Hardware

- GPU box: dual NVIDIA A30 24 GB, CUDA 12.4, dual Intel Xeon Platinum 8260.
- Per-run wall-clock approximately 30 minutes; total runtime for the present manuscript approximately 20 GPU-hours across main + ablation batches.

### S8.4 Logs

- tmux session names `pc2_b1`, `pc2_multiseed`, `pc2_phase5_chain`.
- log files in `logs/pc2_batch1/`, `logs/pc2_multiseed/`, `logs/pc2_multiseed/step3_4_*.log`, `logs/pc2_multiseed/step5_*.log`.
- frozen-info file `logs/pc2_batch1/FROZEN_INFO.txt` recording the commit hash and SHA-256 of `T_transferability.json` at the freeze point.

### S8.5 Generated figures and tables

| Asset | File | Generated by |
|---|---|---|
| Figure 1 | `paper/figures/Fig1_task_client_molecules.{pdf,svg,png}` | `paper/make_figures.py:fig1_task_clients` |
| Figure 2 | `paper/figures/Fig2_overall_pc2_federated_architecture.{pdf,svg,png}` | `paper/make_figures.py:fig2_architecture` |
| Figure 3 | `paper/figures/Fig3_transferability_matrices.{pdf,svg,png}` | `paper/make_figures.py:fig3_transferability` |
| Figure 4 | `paper/figures/Fig4_main_performance.{pdf,svg,png}` | `paper/make_figures.py:fig4_main_performance` |
| Figure 5 | `paper/figures/Fig5_ablation_forest.{pdf,svg,png}` | `paper/make_figures.py:fig5_ablation_forest` |
| Figure 6 | `paper/figures/Fig6_molecule_error_cases.{pdf,svg,png}` | `paper/make_figures.py:fig6_molecule_cases` |
| Table 1 | `paper/tables/table1_task_clients.{csv,md}` | `paper/make_figures.py:table1_task_clients` |
| Table 2 | `paper/tables/table2_main_performance.{csv,md}` | `paper/make_figures.py:table2_main_performance` |
| Table 3 | `paper/tables/table3_paired_tests.{csv,md}` | `paper/make_figures.py:table3_paired_tests` |
| Table 4 | `paper/tables/table4_ablation.{csv,md}` | `paper/make_figures.py:table4_ablation` |

### S8.6 No-leakage calibration tests

- Unit tests in `tests/test_calibration_isolation.py` (20 cases) verify that:
  - calibration is installed strictly from `(n − 1)` training labels per LOOCV fold;
  - aggregating two models with different `(μ, σ)` does not mix their calibration buffers;
  - the held-out test molecule's label never enters `set_calibration`;
  - round-trip z-space ↔ eV-space identity holds to 1 × 10⁻⁵ eV.

### S8.7 Transferability matrix invariants

After Phase 1, the generated `T_transferability.json` was checked against:

- diag(T_*) = 1 for all three matrices;
- T_head[D, C-hole] = 0, T_head[D, C-triplet] = 0;
- T_head[C-hole, C-triplet] = T_head[C-triplet, C-hole] = 0;
- T_head[A, C-triplet] = T_head[B, C-triplet] = 0;
- T_repr in [ε_r, 1] inclusive; T_adapter consistent with √(T_repr · T_head).

**Artefacts:** `tests/test_transferability.py` (22 cases).

---

## S9. Additional limitations and future work

### S9.1 Deferred baselines

- MOON [Li 2021] — model-contrastive FL targeting representation drift; deferred to follow-up work since PC² already addresses personalization.
- D-MPNN / ChemProp [Yang 2019] — strong message-passing baseline; not run in this manuscript due to GPU-hours budget and to keep the comparison anchored to the FL family.
- Reviewer-requested additions can be reported in a revision SI section.

### S9.2 Learnable calibration

E72 (planned): replace the fixed (μ, σ) buffer with a learnable affine (a, b) over the standardised output.  Deferred because at n = 49–53 the two extra degrees of freedom risk absorbing noise.

### S9.3 C-hole data ceiling

- All evaluated methods land in MAE 0.368–0.414 eV on n = 53 LOOCV.
- Best classical baseline R² on the same data = −0.10.
- Expanding C-hole through additional ωB97X-D conformer-resolved measurements is the most direct route to breaking this ceiling; alternatively, a larger external hole-λ source measured at matching protocols could be added as a new task-client node.

### S9.4 Chemistry gate testing under more diverse pairs

- T_repr between C-hole and C-triplet is near-unity (0.98–1.00) because the underlying molecule sets largely overlap.  The chemistry-driven gate is therefore not strongly probed in this manuscript.
- Future work: include task-clients whose chemistry is moderately divergent (Tanimoto-K ≈ 0.4–0.6) so that the gate's weighting math actively shapes parameter routing.

### S9.5 Other future directions

- Quantity-aware multi-head extension: one task-client owning multiple physically distinct quantities, with intra-client per-quantity heads.
- Communication compression: top-k sparsification of per-key updates, particularly relevant for slow private-lab uplinks.
- Federated active learning: use the calibration σ as an uncertainty proxy to prioritise which next-batch molecules each laboratory should label.

---

*End of SI outline.*
