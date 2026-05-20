# Phase 0 — Repo survey for calibration-generalization experiments

**Date:** 2026-05-20  
**Host:** dell-PowerEdge-R940xa (GPU box)  
**Scope:** read-only. No file under `src/`, `experiments/`, `results/`, or
`paper/` is modified by this survey.

## 1. Data sources usable for new experiments

| source | path | rows | columns | notes |
|---|---|---|---|---|
| Public QM9-derived reorg energy | `data/client_a_b/public_reorg_energy_15210.csv` | 15,210 SMILES + label | `smiles`, `reorg_energy_eV` | label mean = 0.722 eV, std = 0.372 eV, range [0.011, 2.939]. Used by main paper as Clients A (non-aromatic) and B (aromatic). |
| Client C-hole (TADF private) | `data/client_c/hole_reorg_molecular.csv` | 53 | `smiles`, `lambda_hole_eV`, … | reserved for E60-E71 main-paper LOOCV; do **not** reuse for synthetic stress tests. |
| Client C-triplet (TADF private) | `data/client_c/reorganization_energy_summary.csv` | 49 | `molecule`, `lambda_T_total_eV`, … | same — reserved. |
| Client D (Atahan-Evrenk 2019) | `data/client_d/atahan_reorg_5876.csv` | 5,876 | `smiles`, `reorg_eV` | available as a chemistry-overlap-low federated source; cheap to import. |

**Decision for Phase 1 (label-scale stress test):** use the **public QM9 file
only** (`public_reorg_energy_15210.csv`). It is the cheapest, has no
domain-shift confound, and we will *deliberately* apply synthetic
label transforms — so any TADF/D contamination would be a confound.

**Decision for Phase 2 (pseudo-task multi-target validation):** use the
same public QM9 file plus possibly a tiny aromaticity / molecular-size
slice from it. Same reasoning.

## 2. Training entry points reusable without modification

| function | file | reusable as-is? |
|---|---|---|
| `data_utils.load_public_data()` | `src/data_utils.py:216` | ✅ returns the 15,210-row dataframe. |
| `data_utils.smiles_to_graph(smi, y)` | `src/data_utils.py:45` | ✅ builds an 11-dim node-feature PyG `Data` object (8 atom one-hot + degree + formal charge + Gasteiger charge). |
| `data_utils.build_graph_dataset(df, label_col)` | `src/data_utils.py:309` | ✅ batches smiles_to_graph over a dataframe. |
| `models.ReorgEnergyModel(head_type, use_adapter, use_calibration, …)` | `src/models.py:269` | ✅ supports all four variants we need: vanilla / +adapter / +calibration / +adapter+calibration. |
| `train_eval.install_calibration_from_train(model, train_data)` | `src/train_eval.py:154` | ✅ sets `(μ, σ)` from training labels per client. |
| `train_eval.train_one_epoch(model, loader, …)` | `src/train_eval.py:82` | ✅ already handles z-space loss when `model.use_calibration=True`. |
| `train_eval.evaluate(model, loader, device)` | `src/train_eval.py:123` | ✅ inverse-standardizes back to eV-space when calibration is active. |
| `federated.fedavg_aggregate(client_models, weights, exclude_keys=[])` | `src/federated.py:94` | ✅ generic weighted parameter averaging. |
| `federated.fedper_aggregate(client_models, weights)` | `src/federated.py:136` | ✅ FedPer (excludes `head`/`bn`/`norm`/`calibration` keys via `_get_exclude_keys`). |
| `federated.distribute_params(client_models, avg_dict)` | `src/federated.py:181` | ✅ broadcast aggregated state back. |
| `federated._get_exclude_keys(state_dict, strategy)` | `src/federated.py:32` | ✅ already excludes `calibration` keys under both FedAvg and FedPer (confirmed by grep). |
| `train_eval.train_federated(client_data, fed_strategy, …)` | `src/train_eval.py:475` | ⚠️ usable but assumes client keys are `client_a` / `client_b` / `client_c` / `client_d`; `target_type` checks key off `'client_c'`. Workable by naming pseudo-clients accordingly. |

**Conclusion:** all four target methods (Local-only, FedPer, FedPer+calibration,
no-calibration ablation) are reachable through the public `src/` API
without modification. Only the PC²-FedReorg full-routing path requires
a transferability JSON pre-computed via `src/transferability.py` —
see §3.

## 3. PC²-FedReorg reuse cost

PC²-FedReorg (full routing) requires a 5×5 transferability matrix JSON
at `results/preverify/T_transferability.json` keyed on the canonical
five task-client nodes (A, B, C-hole, C-triplet, D). To run PC² on a
*synthetic* federation:

- Either rebuild `T_transferability.json` with the synthetic client
  identities and chemistries → expensive, requires re-running
  `src/transferability.py:compute_T_matrices()` with pseudo-meta;
- Or skip PC²-fed-routing for the synthetic experiments and report
  results only on the four cheaper variants (Local, FedPer,
  FedPer+calibration, no-calibration ablation).

**Phase-1 / Phase-2 decision:** **skip PC²-fed-routing** in the synthetic
runs. The user-side task brief permits this ("如果 PC² 路径重用成本太高，
则先只做 Local / FedPer / FedPer+calibration"). The PC²-vs-calibration
question is already settled on the real TADF federation by E71 (PC² ≈
FedPer+calibration, paired Wilcoxon p = 0.929); re-running it on a
synthetic federation would not change the paper's stance. We will
document this skip in the per-phase report so it is not mistaken for
a missing result.

If Phase 1 + 2 land cleanly and time remains, the easiest cheap
PC²-comparison is to invoke `fedper_pc2_aggregate` (which currently
ships in `src/federated.py:165`) without the transferability gate —
it differs from FedPer only in installing the adapter. Not equivalent
to full PC², but a useful "adapter alone" proxy.

## 4. Calibration code paths to reuse

The existing calibration plumbing already implements the v1 spec:

1. `models.CalibrationHead` (line 213): non-trainable `(μ, σ)` buffers,
   no gradient, applied in forward as eV-space inverse-standardize.
2. `models.ReorgEnergyModel(use_calibration=True)` (line 296): installs
   the head.
3. `train_eval.install_calibration_from_train(model, train_data)`
   (line 154): sets `(μ, σ)` from train labels. **Call after the train
   split is decided in each CV fold.**
4. `train_eval.train_one_epoch` (line 82): auto-detects
   `model.use_calibration` and runs the loss in z-space.
5. `train_eval.evaluate` (line 123): auto-detects calibration and
   returns metrics in eV-space.
6. `federated._get_exclude_keys(state_dict, strategy)` (line 32):
   excludes any state-dict key containing `'calibration'` from BOTH
   FedAvg and FedPer aggregation. **Confirmed working.**

**Implication.** A FedPer+calibration training loop can be assembled
from these primitives in ~150 lines without touching `src/`. That is
exactly what Phase 1 will do.

## 5. Wrappers we need to write

Plan for Phase 1 + 2 — every script lives in `experiments/calibration_generalization/`,
every output in `results/calibration_generalization/`. None of these
touch `src/`, `experiments/run_all.py`, or existing `results/*.json`/`*.csv`.

| script | purpose | LoC est. |
|---|---|---|
| `experiments/calibration_generalization/pseudo_federation.py` | helper module: build pseudo-client splits from `public_reorg_energy_15210.csv` with synthetic label transforms, return list of `Data` objects per pseudo-client. | 180 |
| `experiments/calibration_generalization/fed_train_minimal.py` | thin wrapper around `src/`: per-method (Local / FedPer / FedPer+calibration / no-calibration) federated training loop. | 220 |
| `experiments/calibration_generalization/run_label_scale_stress.py` | Phase 1 driver: pseudo-federation × N transforms × 4 methods × 5 seeds; writes summary CSVs. | 200 |
| `experiments/calibration_generalization/run_pseudo_task_validation.py` | Phase 2 driver: 10 pseudo targets × 4 methods × 5 seeds; writes summary CSVs. | 200 |
| `experiments/calibration_generalization/compute_pairwise_stats.py` | independent paired Wilcoxon + bootstrap 95% CI script (analogous to `paper/audit/recompute_all_reported_numbers.py` but for synthetic-federation outputs). | 150 |

Each script will:

- Use `argparse` with a `--smoke` flag (1 seed, 2 rounds, n_train=100) and
  `--full` flag (5 seeds, N_rounds=10–15 depending on phase, full
  pseudo-client sizes).
- Read its hyperparameters from a JSON config saved alongside the
  results.
- Seed every RNG (python `random`, numpy, torch CPU + CUDA) from the
  per-run seed argument.
- Emit per-run CSV rows containing `experiment_id`, `seed`,
  `pseudo_task_id`, `method`, `target`, `n_train`, `n_test`,
  `label_transform`, `MAE`, `RMSE`, `R2`.

## 6. Constraints honoured by this survey

- ✅ No file in `src/`, `experiments/` (existing files), `results/`,
  `paper/audit/`, `paper/figures/`, `paper/tables/`, `paper/manuscript.md`,
  `paper/latex/main.tex`, `paper/SI.md`, `paper/latex/SI.tex`, or
  `paper/latex/CalibrationAware_FedReorg_Overleaf.zip` is modified.
- ✅ No `git commit`, no `.docx` generation.
- ✅ No training has been launched.
- ✅ All future work writes to `experiments/calibration_generalization/`,
  `results/calibration_generalization/`, `paper/calibration_generalization/`.

## 7. Environment

- **Conda env for training:** `H-CAAN` (verified to import `src/`,
  `torch_geometric`, `rdkit`, `efficient_kan`). Python 3.10, PyTorch 2.7.1+cu126.
- **GPU:** 2× A30 24 GB available. Will use cuda:0 for one pseudo-client
  pair, cuda:1 for the second, per the main-paper device-map convention.
- **Conda env for audit/stats only:** `py312_env` (numpy/scipy only).

## 8. Risk summary

| risk | mitigation |
|---|---|
| Synthetic federation may be too easy → calibration always wins by construction. | Report calibration's effect *size* (ΔMAE) as a function of scale ratio; if ΔMAE saturates at large ratios, that is the headline message. Also report cases where calibration does **not** help (small ratios). |
| Smoke test passes but full run hits an OOM or hang. | Cap pseudo-client sizes at n ≤ 500; 30-round upper bound; tmux+nohup for the full runs so a hang doesn't block the audit. |
| Method labels in the report could conflate Phase 1 / Phase 2 results with the main-paper results. | Every CSV row carries `phase` field. Every report explicitly states "controlled synthetic heterogeneity / NOT real TADF". |
| Bootstrap CI seed reuse from main paper. | Use a *different* bootstrap seed for the synthetic runs (we already document the previous seed in `paper/audit/data_provenance_audit.md`); record it in the config JSON. |

## 9. Smoke-test plan (executed before Phase 1 full run)

1. Build a tiny pseudo-federation: 3 pseudo-clients × n=50 molecules each.
2. Methods: Local-only, FedPer, FedPer+calibration. Skip PC² for smoke.
3. Rounds: 2. Local epochs: 1. Seeds: 1 (= 42).
4. Expected runtime: < 5 minutes on a single A30.
5. Pass criterion: each method produces a finite MAE / RMSE on each
   pseudo-client, and FedPer+calibration's MAE is less than or equal to
   FedPer's MAE on at least 2 of 3 clients (qualitative direction check
   only; no statistical claim).

If smoke passes, proceed to Phase 1 full. If it fails, debug the
wrapper, do not run the full experiment.

---

**Phase 0 verdict:** Repo is fully ready for the proposed Phase 1 +
Phase 2 work; no `src/` modification required; PC²-fed-routing skipped
for synthetic federations (justified above). Proceeding to Phase 1.
