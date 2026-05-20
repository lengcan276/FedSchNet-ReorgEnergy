# Supporting Information — PC²-FedReorg

**Manuscript:** *Calibration-Aware Personalized Federated Learning for Molecular Reorganization Energy Prediction*

This document provides supporting material for the main manuscript: dataset and task-client definitions (S1), pre-verification analyses motivating the algorithmic choices (S2), model and implementation details (S3), the auditable transferability matrices (S4), full performance results (S5), paired statistical testing (S6), ablation studies (S7), per-molecule prediction artefacts (S8), a reproducibility checklist (S9), and a list of additional limitations (S10).

All numerical artefacts referenced below are released alongside the manuscript at `[repository URL — to be supplied]` and are also identified by their relative path in this codebase (e.g. `results/pc2_phase5_stats.csv`).

---

## S1. Dataset and task-client details

### S1.1 Five task-client nodes

The federation comprises five nodes (Table 1, main text). Each node is defined not only by its underlying samples but also by the *quantity* it measures and the *DFT protocol* that produced its labels. We adopt the term **task-client** to make this distinction explicit: a single physical laboratory may host more than one task-client when it measures more than one physical quantity. The five quantities are all reorganization-energy-related but they are not interchangeable: A and B are cation-λ; C-hole and D are both hole-λ but at different DFT functionals (ωB97X-D vs. B3LYP) and different chemical spaces; C-triplet is a triplet-λ task. Pairwise head sharing across these distinct physical quantities is blocked by clause 1 of the head gate (Section S4).

| Task-client | Source | Physical quantity | n (after preprocessing) | Label mean ± SD (eV) | Label range (eV) |
|---|---|---|---|---|---|
| A | QM9 non-aromatic public split | cation reorganization energy | 6,020 | 0.821 ± 0.422 | 0.021 – 2.939 |
| B | QM9 aromatic public split | cation reorganization energy | 9,190 | 0.658 ± 0.319 | 0.011 – 2.680 |
| C-hole | in-house TADF laboratory | hole reorganization energy (neutral → cation, ωB97X-D) | 53 | 1.215 ± 0.496 | 0.449 – 3.004 |
| C-triplet | in-house TADF laboratory | triplet reorganization energy (neutral → T₁, ωB97X-D) | 49 | 2.638 ± 0.745 | 1.047 – 4.273 |
| D | Atahan-Evrenk 2019 | hole reorganization energy (conjugated organic semiconductors, B3LYP/6-31G\*) | 5,876 | 0.258 ± 0.048 | 0.076 – 0.870 |

Per-node statistics are taken from `results/preverify/T_transferability.json` (keys `n_kept`, `label_means`, `label_stds`).

### S1.2 Why A and B are cation-λ, C is hole/triplet, D is hole-λ

- **A / B (QM9 public):** the public QM9-derived reorganization-energy split is computed as a cation reorganization energy at B3LYP/6-31G\* by previous work on the same compound set. We further split QM9 by aromatic-ring count (`Chem.rdMolDescriptors.CalcNumAromaticRings`) into a non-aromatic subset (A, n = 6,020) and an aromatic subset (B, n = 9,190), since the resulting label distributions differ enough to be informative as separate clients. Both clients share the same B3LYP/6-31G\* protocol.
- **C-hole / C-triplet (in-house TADF):** the in-house laboratory provides per-molecule reorganization energies computed at ωB97X-D using the adiabatic four-point Nelsen protocol with one conformer per molecule. Two physical quantities are measured: the hole reorganization energy (neutral → cation, with hole-relevant geometry relaxation) and the triplet reorganization energy (neutral → T₁). After dropping rows with missing values, n = 53 for the hole-λ target and n = 49 for the triplet-λ target. The two subsets share most underlying molecular scaffolds.
- **D (Atahan-Evrenk 2019):** the published conjugated-organic-semiconductor dataset reports hole reorganization energy at B3LYP/6-31G\*. The dataset specifies the functional and basis but does not record geometry treatment, charge-state handling, or conformer policy in machine-readable form.

### S1.3 Task-client versus physical client

A *physical client* corresponds to a single data-providing entity (a laboratory, a public dataset, etc.). A *task-client* additionally carries the physical quantity and DFT protocol as part of its identity. In our federation, A, B and D each map to one task-client; the in-house TADF laboratory maps to two task-clients (C-hole and C-triplet) because it measures two distinct physical quantities under the same general protocol. The decision affects:

1. Aggregation routing — separate task-clients have separate state-dict, optimizer state, and (μ, σ) calibration buffer.
2. Head sharing — quantity-mismatched task-clients have T_head = 0 by construction (Section S4).
3. Calibration installation — each task-client installs its own (μ, σ) from its own training fold, independent of any other client.

The task-client abstraction is therefore a modeling decision, not a data-collection decision: it does not change which DFT logs are read or how molecules are graphed.

### S1.4 Label distributions and representative molecules

Per-node label distributions are shown in main-text Figure 1b. The label scales span an order of magnitude (D mean 0.258 eV; C-triplet mean 2.638 eV; ratio 10.2×). The five representative molecules shown in Figure 1a were selected by a deterministic rule — the molecule whose label is closest to the per-node median — implemented in `paper/make_figures.py:select_representative`. The chosen SMILES are recorded in the standard output of `paper/make_figures.py` and are:

- A: `C#CC#CC(=O)C1CC1`, λ = 0.733 eV
- B: `Nc1nc(O)nc(=O)[nH]1`, λ = 0.583 eV
- C-hole: `N#CC1=C(C#N)C1=C1C(O)=CC=C1O`, λ_h = 1.057 eV
- C-triplet: `NC1=CC=C(N)C1=C1C=CC=CC=C1`, λ_T = 2.747 eV
- D: `c1ccc2c(c1)Cc1c3cnc4ccCc4c3[nH]c21`, λ_h = 0.264 eV

No representative was chosen by hand.

### S1.5 Protocol metadata

Full per-node protocol metadata (functional, basis, geometry treatment, charge state, conformer policy) is persisted as `results/preverify/T_transferability.json:protocol_meta`. Clients A and B are derived from a single public QM9 cation-reorganization-energy source and share `source_id`; their pairwise head-sharing is therefore governed by the same-source shortcut in clause 3 of the head gate (Section S4), which fires *before* clauses 2 and 3 of the per-field checks and does not require the auxiliary fields to be populated — the per-field entries in main-text Table 1 are accordingly shown as "— (same-source)". Unknown fields are reserved for cases where the same-source shortcut does not apply: D's `geometry`, `charge_state`, and `conformers` are tagged `"unknown"` (the published source provides only functional and basis), and C's `basis` field is not exposed in the in-house CSV used here. These three genuinely missing fields trigger clause 2 of the head gate (Section S4) wherever the shortcut does not fire.

---

## S2. Pre-verification before transfer (V1 / V2 / V3)

Three independent pre-verification analyses were performed before the federated experiments. These analyses are **not failure reports**: they are evidence-gathering steps that motivated the task-client modeling, the conservative T_head clauses, and the per-task-client calibration. We summarize the analyses here.

### S2.1 V1: label and protocol divergence between D and C

For each ordered pair (D, C-hole) and (D, C-triplet) we ran a two-sample Kolmogorov–Smirnov test on the label distributions, plus a side-by-side comparison of DFT-protocol fields:

- D-hole-λ vs. C-hole-λ: KS p ≈ 1.1 × 10⁻¹¹⁶ (label distributions disjoint in mean, range, and shape).
- DFT-protocol divergence: D uses B3LYP/6-31G\* with abstract-only metadata; C-hole uses ωB97X-D with the four-point Nelsen geometry treatment.
- The combination of label-distribution divergence and functional-family mismatch triggers the V1 hard rule used in Section S4: T_head[D, C-hole] is set to zero independently of any training-time signal.

V1 does **not** assert that naive D → C transfer must produce negative outcomes — that empirical question is addressed by the E65 ablation (Section S7.5). V1 only documents that a chemistry-discipline reviewer cannot be expected to assume cross-protocol comparability between D's labels and C's labels without further evidence.

### S2.2 V2: classical C-only baselines and the data-ceiling diagnosis

To bound what a small-sample C-only learner can achieve on Client C in the absence of any federation signal, we trained four classical regressors — Random Forest, Gaussian Process, K-Nearest Neighbours and XGBoost — on Morgan fingerprints (radius 2, 2048 bits) under the same 5-seed LOOCV protocol used for the federated runs.  Results are tabulated below and persisted as `results/preverify/V2_classical_baselines.csv` and `results/preverify/V2_summary.md`.

**C-hole (n = 53), 5-seed LOOCV — mean ± standard deviation across seeds:**

| Model | MAE (eV) | RMSE (eV) | *R*² |
|---|---|---|---|
| Random Forest | 0.408 ± 0.004 | 0.573 ± 0.005 | −0.334 ± 0.024 |
| Gaussian Process | 0.397 ± 0.000 | 0.532 ± 0.000 | −0.150 ± 0.000 |
| **K-Nearest Neighbours** | **0.367 ± 0.000** | **0.521 ± 0.000** | **−0.101 ± 0.000** *(best)* |
| XGBoost | 0.469 ± 0.007 | 0.646 ± 0.006 | −0.696 ± 0.032 |

**C-triplet (n = 49), 5-seed LOOCV — mean ± standard deviation across seeds:**

| Model | MAE (eV) | RMSE (eV) | *R*² |
|---|---|---|---|
| **Random Forest** | **0.552 ± 0.005** | **0.713 ± 0.005** | **+0.084 ± 0.012** *(best)* |
| Gaussian Process | 0.592 ± 0.000 | 0.761 ± 0.000 | −0.042 ± 0.000 |
| K-Nearest Neighbours | 0.637 ± 0.000 | 0.787 ± 0.000 | −0.116 ± 0.000 |
| XGBoost | 0.584 ± 0.008 | 0.720 ± 0.008 | +0.068 ± 0.021 |

The best classical *R*² is **−0.101** on C-hole (KNN) and **+0.084** on C-triplet (Random Forest).  Three of the four C-hole baselines and three of the four C-triplet baselines produce negative *R*² across all five seeds, despite reporting MAE values comparable to the federated methods.  This is consistent with a small-sample data-ceiling regime in which the per-molecule label variance is large relative to the spread of the labels, so that explained variance is a brittle metric.  We therefore use MAE / RMSE as the primary regression metrics in the main text, and report *R*² only as an auxiliary.  The C-hole non-significance reported in the main manuscript (Figure 4, left) is consistent with — but does not by itself prove — this data ceiling.

### S2.3 V3: chemical-space overlap

Per-pair chemistry overlap was computed in two complementary forms:

- **K_ij** — the mean over molecules of *i* of the maximum Tanimoto similarity to any molecule of *j*, using Morgan fingerprints (radius 2, 2048 bits).
- **S_ij** — the asymmetric Murcko-scaffold overlap, defined as |Scaf(i) ∩ Scaf(j)| / |Scaf(i)|.

Selected V3 values:
- K(C-hole → D) = 0.120; K(C-triplet → D) = 0.118 (median per-C-molecule max-Tanimoto to D).
- S(C-hole → D) = 0.0 (no C-hole scaffold appears in D); S(C-triplet → D) = 0.0.
- K(C-hole, C-triplet) and K(C-triplet, C-hole) are essentially saturated, reflecting near-identical molecule sets between the two C task-clients.

The Atahan-Evrenk dataset (D) and the in-house TADF dataset (C) therefore live in disjoint chemical spaces.

V3 motivates the T_repr definition used in Section S4: chemistry-similar source–target pairs receive higher encoder-aggregation weight, but the gate alone is shown empirically (S7) not to be a performance driver in our particular federation.

### S2.4 D → C direct transfer cannot be assumed protocol-compatible

The combination of V1 (protocol divergence and quantity mismatch) and V3 (disjoint chemistry) makes a naive D → C supervised transfer unsafe to assume without evidence: there is no a-priori basis for the assumption that D's hole-λ labels — computed at B3LYP/6-31G\* on chemically distant conjugated organic semiconductors — are a reliable supervised signal for C-hole's ωB97X-D adiabatic-four-point labels, let alone for the triplet end-state energies of C-triplet. PC²-FedReorg encodes this caution as T_head[D, C-\*] = 0 by construction.

Empirically, we ran the E65 negative-control (Section S7.5) and found that PC²-FedReorg and a naive D → C pretrain are not statistically distinguishable on either C target. We therefore do not claim that D → C transfer necessarily produces negative outcomes; we only claim that the conservative gate is defensible and that PC²-FedReorg did not extract benefit from D's labels through the head.

---

## S3. Model and implementation details

### S3.1 Encoder

All experiments in this manuscript use a 5-layer Graph Isomorphism Network (GIN) encoder with hidden dimension 300, Jumping-Knowledge concatenation across layers, mean+max pooling on the node embeddings, and a final projection to 256-d via a two-layer MLP. The codebase additionally implements a SchNet 3D encoder (`src/models.py:SchNet3DEncoder`) accessible through the same `encoder_type` switch, but **no SchNet results are reported in this manuscript** — the SchNet path is documented here only as repository infrastructure for follow-up work.

Node features are an 11-dimensional descriptor (atom type one-hot, degree, formal charge, hybridisation, aromaticity, in-ring flag, total Hs, Gasteiger partial charge). When HOMO/LUMO/gap/dipole node features are enabled the descriptor extends to 15 dimensions; for clients without HOMO/LUMO measurements the extra channels are zero-padded.

### S3.2 KAN head

The regression head uses `efficient-kan` with grid size 5 and spline order 3. Three KAN layers are stacked: 256 → 64 → 32 → 1. KAN parameters are private to each task-client (head sharing under T_head is reserved to compatible source–target pairs; in our federation the only non-zero T_head off-diagonal is A ↔ B, see Section S4).

### S3.3 Residual adapter

An optional residual bottleneck (256 → 128 → 256, GELU activation, LayerNorm before the residual addition) sits between encoder and head. The final projection of the bottleneck is zero-initialised so that, prior to any training, the adapter is the identity. The adapter participates in cross-client aggregation under T_adapter (defined in Section S4).

### S3.4 Calibration buffer

Each task-client owns a `CalibrationHead` with two non-trainable buffers (`μ`, `σ`) registered via `torch.nn.Module.register_buffer`. A small floor σ ≥ 10⁻⁸ prevents division-by-zero on degenerate folds.

Calibration is installed per LOOCV fold from the (n − 1) training labels of that fold; the held-out test molecule's label never enters `set_calibration(μ, σ)`. This isolation is verified by unit tests (Section S9).

### S3.5 z-space loss / eV-space metrics

Training loss is computed in the per-task-client standardised z-space:

$$
\mathcal{L}_c \;=\; \mathrm{MSE}\!\Bigl(\,\mathrm{model}_z(x), \; (\lambda - \mu_c)\,/\,\sigma_c\Bigr).
$$

Predictions are inverted back to eV-space for metric reporting:

$$
\hat{\lambda} \;=\; \sigma_c \cdot z_{\mathrm{pred}} + \mu_c.
$$

All MAE, RMSE and R² values reported in the manuscript are computed in eV-space.

### S3.6 Aggregation exclusion

The shared filter `_get_exclude_keys(state_dict)` returns the set of keys that no federated strategy aggregates. In PC²-FedReorg this set contains every key matching `*calibration*`, plus the FedBN-style exclusions of `*.bn.*` and `*.norm.*`. Heads, adapters and encoders are routed by the per-key per-target rule given in main-text Section 2.3 (Equation 1).

### S3.7 No-leakage calibration tests

A test module (`tests/test_calibration_isolation.py`, 20 cases) verifies that:

1. `set_calibration(μ, σ)` accepts only the training-fold labels and rejects the test molecule by construction;
2. cross-aggregating two models with different (μ, σ) buffers does not mix their calibration values;
3. the z-space → eV-space round-trip identity holds to 1 × 10⁻⁵ eV for fixed inputs;
4. PC²-FedReorg never reads a calibration buffer from the aggregated parameter dictionary.

All 20 tests pass on the commit hash recorded in `logs/pc2_batch1/FROZEN_INFO.txt`.

---

## S4. Transferability matrices

### S4.1 Definitions

Three 5 × 5 transferability matrices T^(layer) for layer ∈ {repr, adapter, head} are computed once before training and persisted as `results/preverify/T_transferability.json`. They are deterministic functions of the chemistry overlap and protocol metadata; they do not depend on any label or model state.

**T_repr (encoder gate, chemistry-driven):**

$$
T_{\mathrm{repr}}[i, j] = \max\!\bigl(\epsilon_r,\; w_K \cdot K_{ij} + w_S \cdot S_{ij}\bigr)
$$

with (w_K, w_S) = (0.6, 0.4), ε_r = 10⁻³, K_ij from V3 (Section S2.3), and S_ij the asymmetric Murcko-scaffold overlap.

**T_head (regression-head gate, conservative protocol clauses, in the order applied by `src/transferability.py:t_head_compatibility`):**

1. **Clause 1 — quantity mismatch:** if `quantity(i) ≠ quantity(j)`, set T_head[i, j] = 0.
2. **Clause 4 — V1 hard rule:** if i = D and j ∈ {C-hole, C-triplet}, set T_head[i, j] = 0 (applied before the per-field clauses below so the JSON `reasons` field attributes D → C-* to V1).
3. **Same-source shortcut:** if quantity matches *and* `source_id(i) = source_id(j)`, the pair is treated as protocol-compatible by source and the per-field clauses below are skipped. This models the fact that two clients derived from a single public dataset are computed by the dataset's canonical protocol, which need not be re-verified field-by-field. In our federation the shortcut fires only for A ↔ B (both `source_id = "qm9_public_reorg_15210"`).
4. **Clause 2 — unknown protocol field:** if the shortcut did not fire and any of {functional, basis, geometry, charge_state, conformers} equals `"unknown"` on either side, set T_head[i, j] = 0.
5. **Clause 3 — protocol-field mismatch:** if the shortcut did not fire and any populated protocol field disagrees between i and j, set T_head[i, j] = 0.

For pairs that survive the above (in our federation: A ↔ B plus all diagonals):

$$
T_{\mathrm{head}}[i, j] = L_{ij}^{\gamma_L}, \quad
L_{ij} = \exp\!\bigl(-\alpha_W\,W_1(\hat p_i^{\mathrm{label}},\,\hat p_j^{\mathrm{label}})\bigr) \cdot \min\!\bigl(\sigma_i/\sigma_j,\,\sigma_j/\sigma_i\bigr),
$$

with α_W = 5.0, γ_L = 1.0; values below 10⁻² are hard-floored to zero.

**T_adapter (interpolating gate):**

$$
T_{\mathrm{adapter}}[i, j] = \max\!\bigl(\epsilon_a,\;\sqrt{T_{\mathrm{repr}}[i, j]\cdot T_{\mathrm{head}}[i, j]}\bigr).
$$

### S4.2 Numerical values used in this study

| (source → target) | T_repr | T_adapter | T_head |
|---|---|---|---|
| A → B | 0.16 | 0.23 | 0.33 |
| B → A | 0.15 | 0.22 | 0.33 |
| C-hole → C-triplet | 0.98 | ~0 | 0 |
| C-triplet → C-hole | 1.00 | ~0 | 0 |
| D → C-hole | 0.05 | ~0 | **0 (V1 hard rule; D's geometry/charge-state/conformers also unknown)** |
| D → C-triplet | 0.05 | ~0 | **0 (quantity mismatch)** |
| A → C-hole | ~0.07 | ~0 | 0 (clause 1: A's quantity tag is `reorg_energy_general`; same-source shortcut does not apply because `source_id(A) ≠ source_id(C-hole)`) |
| A → C-triplet | ~0.07 | ~0 | 0 (quantity mismatch) |
| B → C-hole | ~0.07 | ~0 | 0 (clause 1: B's quantity tag is `reorg_energy_general`; same-source shortcut does not apply because `source_id(B) ≠ source_id(C-hole)`) |
| B → C-triplet | ~0.07 | ~0 | 0 (quantity mismatch) |

Full 5 × 5 matrices for all three gates are persisted in `results/preverify/T_transferability.json`.

### S4.3 Invariants generated by the clauses

The conservative T_head rules generate four key invariants that hold independently of any model training:

1. **T_head[D, C-hole] = 0** — reported by clause 4 (V1 hard rule). Clauses 2 (D's protocol fields are unknown) and 3 (functional family differs: B3LYP vs. ωB97X-D) would each independently block this pair, so the invariant is multiply-determined.
2. **T_head[D, C-triplet] = 0** — reported by clause 1 (quantity mismatch: hole-λ vs. triplet-λ); D's metadata gap is irrelevant here.
3. **T_head[C-hole, C-triplet] = 0** — reported by clause 1.
4. **T_head[C-triplet, C-hole] = 0** — reported by clause 1.
5. **T_head[A, B] = T_head[B, A] = 0.33** — the only non-zero off-diagonal of T_head. The same-source shortcut fires because A and B share `source_id = "qm9_public_reorg_15210"` and `quantity = "reorg_energy_general"`; the per-field clauses 2 and 3 are therefore skipped, and L_AB is evaluated from the label distributions.

These invariants are verified by `tests/test_transferability.py` (22 cases) and can be re-checked at any time by reading `T_transferability.json`. The verification does not require model training; it only requires the chemistry-overlap statistics from V3 and the protocol metadata from V1.

### S4.4 Auditability, not performance lever

The transferability matrices serve in this manuscript as **auditable algorithmic transparency**: they document which source–target sharings are *permitted* by chemistry and protocol metadata, independently of any optimization outcome. The empirical question of whether the gate also *improves* performance over uniform weighting is examined in S7 (E68, E69) and is, in the present federation, answered negatively — the chemistry/protocol gate is not the source of PC²-FedReorg's C-triplet improvement.

---

## S5. Full performance results

### S5.1 Aggregated mean ± SEM

Aggregated 5-seed means and standard errors for each (method, target) cell are released in `results/pc2_batch1_multiseed_summary.csv` and `results/pc2_phase5_ablation_summary.csv`, and summarized in main-text Table 2 (`paper/tables/table2_main_performance.csv`). MAE, RMSE and R² are reported; R² is auxiliary (see S2.2).

### S5.2 A and B 5-fold cross-validation

For Clients A and B we report 5-fold mean ± SEM across the five seeds. Headline values (MAE in eV):

| Method | A (n=6020) | B (n=9190) |
|---|---|---|
| FedAvg | 0.171 | 0.103 |
| FedProx | similar to FedAvg | similar |
| FedPer | competitive | competitive |
| PC²-FedReorg | 0.142 | 0.078 |

Full per-seed values for A 5-fold and B 5-fold are in `results/pc2_batch1_multiseed_summary.csv`. PC²-FedReorg achieves the lowest A/B MAE among the evaluated methods, but the A/B improvements are not the headline claim of this manuscript: the in-house TADF tasks (C-hole, C-triplet) are the small-sample setting for which the method was designed.

### S5.3 C-hole LOOCV

For C-hole (n = 53), per-method MAE (mean across seeds):

| Method | MAE (eV) | RMSE (eV) | R² (auxiliary) |
|---|---|---|---|
| E60 Local-only | 0.414 | — | negative |
| E61 FedAvg | 0.385 | — | negative |
| E62 FedProx | 0.386 | — | negative |
| E63 FedPer | 0.368 | — | negative |
| E66 PC²-FedReorg | 0.382 | — | negative |

All methods produce MAE within the narrow band 0.368–0.414 eV.  All paired tests return p ≥ 0.20 after rounding (Section S6).  We report PC² as competitive but statistically indistinguishable from the baselines on C-hole.

### S5.4 C-triplet LOOCV

For C-triplet (n = 49):

| Method | MAE (eV) |
|---|---|
| E60 Local-only | 0.747 |
| E61 FedAvg | 0.763 |
| E62 FedProx | 0.784 |
| E63 FedPer | 0.762 |
| **E66 PC²-FedReorg** | **0.660** |

PC²-FedReorg achieves the lowest MAE; paired Wilcoxon tests against FedAvg, FedProx and FedPer give p = 0.045, 0.045 and 0.042 respectively (Section S6).

### S5.5 Auxiliary R² statement

For all C LOOCV cells, the seed-averaged R² is negative or slightly positive (~0.0 ± 0.2). This is consistent with the V2 classical-baseline ceiling (Section S2.2) and is the reason we do not use R² as a primary metric. SI Figure S-R² (to be added) plots the per-seed R² distributions to make this point explicit.

---

## S6. Paired statistical testing

### S6.1 Methodology

For each method pair (PC² vs. comparator) and each LOOCV target, we report:

- per-molecule absolute errors aggregated as the **median across the five seeds** (yielding a 53- or 49-element error vector that is deterministic for a fixed code path);
- paired **Wilcoxon signed-rank test** (two-sided, SciPy `zero_method='wilcox'`);
- **bootstrap 95 % CI** on the paired MAE difference (5,000 resamples of molecule indices);
- **win / loss / tie counts** per molecule.

The full paired-test table is persisted as `results/pc2_phase5_stats.csv` and rendered in `paper/tables/table3_paired_tests.csv`.

### S6.2 C-hole — eight paired comparisons, all non-significant

| Comparator | MAE_comp (eV) | ΔMAE | 95 % CI | wins / loses | Wilcoxon p |
|---|---|---|---|---|---|
| E60 Local | 0.414 | −0.032 | (−0.115, +0.050) | 25 / 28 | 0.580 |
| E61 FedAvg | 0.385 | −0.003 | (−0.042, +0.039) | 27 / 26 | 0.898 |
| E62 FedProx | 0.386 | −0.004 | (−0.055, +0.044) | 28 / 25 | 0.821 |
| E63 FedPer | 0.368 | +0.014 | (−0.021, +0.049) | 23 / 30 | 0.198 |
| E65 D→C pretrain | 0.403 | −0.021 | (−0.099, +0.055) | 28 / 25 | 0.760 |
| E68 no C-cross repr | 0.382 | −0.0001 | (−0.017, +0.017) | 30 / 23 | 0.562 |
| E69 uniform gate | 0.370 | +0.012 | (−0.020, +0.043) | 26 / 27 | 0.428 |
| E70 no calibration | 0.377 | +0.005 | (−0.033, +0.042) | 23 / 30 | 0.580 |

No comparison reaches the 0.05 threshold. All bootstrap CIs straddle zero. We report PC² on C-hole as statistically indistinguishable from the baselines and ablations.

### S6.3 C-triplet — four significant baselines, four non-significant ablations/controls

| Comparator | MAE_comp (eV) | ΔMAE | 95 % CI | wins / loses | Wilcoxon p | Sig |
|---|---|---|---|---|---|---|
| E60 Local | 0.747 | −0.088 | (−0.230, +0.053) | 28 / 21 | 0.174 | n.s. |
| **E61 FedAvg** | 0.763 | **−0.104** | (−0.220, +0.004) | 31 / 18 | **0.045** | * |
| **E62 FedProx** | 0.784 | **−0.124** | (−0.254, −0.007) | 30 / 19 | **0.045** | * |
| **E63 FedPer** | 0.762 | **−0.103** | (−0.205, −0.002) | 32 / 17 | **0.042** | * |
| E65 D→C pretrain | 0.670 | −0.010 | (−0.142, +0.117) | 27 / 22 | 0.851 | n.s. |
| E68 no C-cross repr | 0.633 | +0.027 | (+0.001, +0.055) | 17 / 32 | 0.070 | n.s. |
| E69 uniform gate | 0.616 | +0.043 | (−0.001, +0.092) | 21 / 28 | 0.181 | n.s. |
| **E70 no calibration** | 0.779 | **−0.119** | (−0.225, −0.015) | 35 / 14 | **0.043** | * |

Four statistically significant comparisons (E61, E62, E63 baselines; E70 ablation) support the C-triplet claims of the manuscript. The remaining four comparisons (E60 local, E65 D→C pretrain, E68 no C-cross, E69 uniform gate) do not reach significance.

### S6.4 Interpretation

- **C-hole** improvement is not supported by our data. Future work should expand the C-hole sample size before attempting hole-task claims.
- **C-triplet** improvement over the three legacy federated baselines is supported with p < 0.05 in each pair and bootstrap CIs straddling or sitting below zero.
- **Calibration ablation (E70)** is the *only* component-level comparison that reaches significance, with PC² 0.119 eV better than the no-calibration variant. The CI sits entirely below zero.
- **Gate ablations (E68, E69)** do not support a claim that the chemistry-driven gate weighting drives the C-triplet improvement.

---

## S7. Ablation studies

### S7.1 Ablation matrix

Four targeted ablations were run with the same hyperparameters, seeds and code path as the reference PC²-FedReorg (E66). The differences are confined to the indicated component.

| ID | Description | What is removed / changed |
|---|---|---|
| E70 | no-calibration variant | `use_calibration=False`; (μ, σ) buffers are not installed; loss and metric both in eV-space |
| E68 | no chemistry-driven cross-quantity encoder sharing | T_repr cross terms between C-hole and C-triplet floored to 10⁻³ in a separate `T_transferability_no_ccross.json` artefact; everything else identical |
| E69 | uniform FedAvg weighting | aggregation strategy switched from `pc2_fed` to `fedavg`; calibration and adapter retained |
| E65 | D → C supervised pretrain control | Stage 1: train one GIN + KAN model on all 5,876 D molecules; Stage 2: per-fold LOOCV fine-tune on C with the D-pretrained weights as the starting point |

Full per-seed predictions and summary statistics are released as `results/pc2_phase5_ablation_predictions.json` and `results/pc2_phase5_ablation_summary.csv`.

### S7.2 E70 — task-specific calibration removed

- C-triplet: ΔMAE = −0.119 eV; 95 % CI (−0.225, −0.015); paired Wilcoxon p = 0.043; 35 / 14 wins for PC².
- C-hole: ΔMAE = +0.005 eV; 95 % CI (−0.033, +0.042); paired Wilcoxon p = 0.58; 23 / 30 wins.

E70 is the only ablation in this study that produces a statistically significant degradation versus PC². The effect is restricted to C-triplet, consistent with the larger label-scale gap on C-triplet (mean 2.64 eV) relative to C-hole (mean 1.22 eV). We interpret task-specific calibration as the only ablation-supported active component.

### S7.3 E68 — no C-cross representation

- C-triplet: ΔMAE = +0.027 eV (PC² *slightly worse* than the no-C-cross variant); 95 % CI (+0.001, +0.055); paired Wilcoxon p = 0.070; 17 / 32 wins.
- C-hole: ΔMAE = −0.0001 eV; 95 % CI (−0.017, +0.017); paired Wilcoxon p = 0.56.

E68 sits close to but does not cross the 0.05 threshold on C-triplet. With the present 5-seed bootstrap design we do not interpret this as a reliable trend; the bootstrap CI on the C-triplet ΔMAE only narrowly excludes zero, and the direction of the effect (PC² slightly worse) would in any case argue against attributing C-triplet gains to the cross-quantity encoder share.

### S7.4 E69 — uniform FedAvg weighting

- C-triplet: ΔMAE = +0.043 eV (PC² slightly worse than uniform weighting); 95 % CI (−0.001, +0.092); paired Wilcoxon p = 0.18.
- C-hole: ΔMAE = +0.012 eV; 95 % CI (−0.020, +0.043); paired Wilcoxon p = 0.43.

E69 does not support a claim that the chemistry-driven gate weighting improves performance over uniform FedAvg weighting in our federation. The most plausible explanation (Section 5 of the main manuscript) is that the chemistry overlap between C-hole and C-triplet is already near-unity, so uniform and chemistry-driven gates produce essentially the same weight vectors for the C-↔-C pair that matters most.

### S7.5 E65 — D → C supervised pretrain control

- C-triplet: ΔMAE = −0.010 eV; 95 % CI (−0.142, +0.117); paired Wilcoxon p = 0.85.
- C-hole: ΔMAE = −0.021 eV; 95 % CI (−0.099, +0.055); paired Wilcoxon p = 0.76.

E65 was designed as a negative-transfer control: by construction, V1 protocol divergence (KS p ≈ 10⁻¹¹⁶) and V3 chemistry distance (Tanimoto K ≈ 0.12) would suggest that naively transferring D's supervised labels to C would be harmful. The empirical comparison, however, is not statistically distinguishable from PC²-FedReorg on either target. We therefore avoid the stronger claim that D → C transfer is necessarily harmful, even though the pre-verification evidence is consistent with that hypothesis. Two possible reasons:

1. PC²-FedReorg's own T_head[D, C-\*] = 0 also blocks D's labels from reaching C's head; the methodological difference between E65 and PC² is therefore concentrated in the encoder side, where T_repr[D, C-\*] ≈ 0.05 is small but non-zero.
2. With n = 49–53 the paired-test power to discriminate ΔMAE on the order of 0.01–0.05 eV is limited; bootstrap CIs span ±0.1 eV.

The data therefore support the more limited conclusion that neither method extracts a significant benefit from D's hole-λ supervision on the C tasks.

### S7.6 Synthesis: what the ablations support and what they do not

- **Supported by ablation evidence (E70) and confirmed by the positive control (E71, §S7.7):** task-specific calibration under personalized heads is the only PC² component whose removal causes a statistically significant degradation, the effect is confined to C-triplet (E70), and the calibration mechanism is *sufficient* — adding it to a plain FedPer baseline already reproduces PC²-FedReorg's C-triplet improvement (E71).
- **Not supported by ablation evidence:** the chemistry-driven gate weighting (T_repr cross-routing, T_head sparsification) does not drive the C-triplet improvement over uniform weighting (E68, E69), and adding the residual adapter and the transferability gate on top of the calibrated FedPer baseline does not yield measurable performance over E71 (E71 ≈ E66).
- **Not demonstrated by ablation evidence:** a claim that naive D → C supervised pretrain produces negative transfer (E65).

Together, these results position PC²-FedReorg's chemistry/protocol gate as an *auditable algorithmic transparency mechanism* (Section S4.4) rather than as a performance lever, and identify task-specific calibration under personalized heads as the active mechanism behind the C-triplet improvement.

### S7.7 E71 — FedPer with calibration only (positive control)

**Purpose.** The four PC² component ablations in §S7.1–S7.6 each *remove* one ingredient from PC²-FedReorg, which collectively show that calibration is necessary. E71 is the symmetric *positive control*: a plain FedPer baseline endowed only with the per-task-client (μ, σ) calibration buffer (no residual adapter, no transferability-driven routing). It directly tests whether the calibration mechanism is also *sufficient* to reproduce PC²-FedReorg's C-triplet gain.

**Specification.** E71 uses the same code path as E66 with two differences: `use_adapter = False` and no `aggregation_strategy = 'pc2_fed'`. The encoder is shared by plain FedPer; the head is private (FedPer); the per-task-client (μ, σ) buffer is installed per LOOCV fold and universally excluded from aggregation by `src/federated.py:_get_exclude_keys` (the same exclusion rule that protects calibration under FedAvg, FedProx, FedPer, FedBN and PC²-FedReorg). All other hyperparameters match E66 / E63 exactly:

| field | value | source |
|---|---|---|
| seeds | {42, 123, 456, 789, 1000} | `experiments/run_e71_multiseed.py` |
| evaluation | LOOCV on C-hole / C-triplet; 5-fold CV on A/B/D | `src/train_eval.py:federated_loocv_c_fast` |
| federation rounds | 50 | `experiments/configs.py:N_FED_ROUNDS` |
| local epochs per round | 5 | `experiments/configs.py:N_LOCAL_EPOCHS` |
| head | 3-layer KAN, grid = 5, spline order = 3 | `experiments/configs.py` |
| optimizer | Adam, lr = 1 × 10⁻⁴, cosine annealing | `experiments/configs.py:LR_FINETUNE` |
| early-stop patience | 15 | `experiments/configs.py:PATIENCE` |
| device assignment | identical to E66 (cuda:0 / cuda:1; CPU aggregation) | hardcoded |

**Wall-clock.** Five-seed run on dual A30: ≈ 2 h 27 min (consistent with E66 on the same federation).

**Released files.** All E71 outputs are isolated in the `results/e71_*` namespace and do not modify the E60–E70 prediction archive:

| artifact | path |
|---|---|
| Across-seed summary (MAE/RMSE/R² per target, per seed) | `results/e71_summary.csv` |
| Per-fold predictions (y_true, y_pred for both C targets, all seeds) | `results/e71_predictions.json` |
| Paired statistics vs PC²-FedReorg | `results/e71_vs_e66_stats.csv` and `.md` |
| Paired statistics vs FedPer | `results/e71_vs_e63_stats.md` |
| Paired statistics vs PC² without calibration | `results/e71_vs_e70_stats.md` |
| Training log | `logs/E71_20260517_2022.log` |
| Registration | `experiments/run_all.py` (E71 entry, additive only) |

**Across-seed metrics (5-seed mean ± SEM):**

| target | n | MAE (eV) | RMSE (eV) | R² (auxiliary) |
|---|---|---|---|---|
| A 5-fold | 6,020 | 0.1508 ± 0.0006 | 0.2119 ± 0.0008 | +0.747 ± 0.002 |
| B 5-fold | 9,190 | 0.0824 ± 0.0009 | 0.1262 ± 0.0009 | +0.843 ± 0.002 |
| C-hole | 53 LOOCV | 0.3757 ± 0.0029 | 0.5139 ± 0.0061 | −0.074 ± 0.026 |
| C-triplet | 49 LOOCV | 0.6527 ± 0.0111 | 0.8259 ± 0.0116 | −0.229 ± 0.034 |

**Paired comparisons on C-triplet (median per-molecule abs error across seeds, 5000-resample bootstrap CI):**

| comparison | ΔMAE (eV) | 95 % CI | wins / loses | Wilcoxon p | interpretation |
|---|---|---|---|---|---|
| E71 vs E66 (PC²-FedReorg) | −0.0059 | (−0.0437, +0.0293) | 23 / 26 | **0.929** | indistinguishable from PC²-FedReorg |
| E71 vs E63 (FedPer, no calibration) | −0.1085 | (−0.2083, −0.0097) | 32 / 17 | **0.028** | nominal improvement over plain FedPer |
| E71 vs E70 (PC² without calibration) | −0.1249 | (−0.2242, −0.0272) | 33 / 16 | **0.021** | calibration is more load-bearing than adapter + gate |

**Paired comparisons on C-hole (all non-significant):**

| comparison | ΔMAE (eV) | 95 % CI | Wilcoxon p |
|---|---|---|---|
| E71 vs E66 | −0.0084 | (−0.0264, +0.0093) | 0.423 |
| E71 vs E63 | +0.0054 | (−0.0271, +0.0391) | 0.598 |
| E71 vs E70 | −0.0036 | (−0.0398, +0.0331) | 0.842 |

**Interpretation.** Three paired comparisons jointly support the calibration-is-sufficient reading on C-triplet: (i) E71 ≈ E66 within seed-SEM with a bootstrap CI that straddles zero and balanced per-molecule sign counts; (ii) E71 < E63 by an effect size matching PC² vs. FedPer in §4.3; (iii) E71 < E70 by an effect size matching PC² vs. its no-calibration ablation in §S7.2. C-hole remains at its small-sample data ceiling for all methods (all three E71 paired tests return p ≥ 0.42 with CIs that straddle zero), consistent with §4.3 and §S2.2.

**Reproducibility command.** From the project root, with the H-CAAN conda environment activated:

```bash
tmux new -s e71
nohup python experiments/run_e71_multiseed.py \
    > logs/E71_$(date +%Y%m%d_%H%M).log 2>&1 &
# Ctrl-b d to detach; estimated wall clock ≈ 2.5 h on dual A30.
python experiments/compute_e71_stats.py     # writes results/e71_vs_*.md/.csv
```

**Statistical caveat.** The paired Wilcoxon p-values above are unadjusted nominal values; the Bonferroni threshold for the three E71 contrasts is α/3 ≈ 0.017, which neither of the two effect-bearing comparisons crosses. We therefore treat effect size, bootstrap CI direction and per-molecule sign counts as primary evidence and the nominal p-values as supportive but not definitive. The chemically meaningful effect size for organic-electronics screening is ±0.1 eV (typical conformer-averaging noise on ωB97X-D reorganization energies); both the E71-vs-FedPer and E71-vs-no-calibration ΔMAEs sit at this scale and should be re-evaluated on a larger TADF dataset before being treated as deployment-ready.

---

## S8. Per-molecule prediction artefacts

### S8.1 Released files

For every (method, seed, target) cell, the y_true and y_pred vectors are released:

- `results/pc2_batch1_multiseed_predictions.json` — E60, E61, E62, E63, E66 on both C targets.
- `results/pc2_phase5_ablation_predictions.json` — E65, E68, E69, E70 on both C targets.
- `results/e71_predictions.json` — E71 (FedPer + calibration only, the §S7.7 positive control) on both C targets.

Aggregating absolute errors as the median across seeds yields the per-method per-molecule error vectors used for all paired analyses.

### S8.2 Top improved / degraded cases (C-triplet)

The six C-triplet cases shown in Figure S1 (this Supporting Information) — three most-improved and three most-degraded by PC² versus FedAvg — were selected programmatically by sorting molecules on ΔMAE = |error|_PC² − |error|_FedAvg. The corresponding SMILES and per-method absolute errors are summarized below.

**Most-improved (Δ|err| < 0; PC² better):**

| idx | SMILES | true λ_T (eV) | |err| FedAvg | |err| PC² | Δ|err| |
|---|---|---|---|---|---|
| 35 | `ClC1=C(Cl)C(=C2C(c3ccccc3)=C2c2ccccc2)C(Cl)=C1Cl` | 1.133 | 3.055 | 1.701 | −1.354 |
| 36 | `C1=CC(=C2C=C2)C=C1` | 3.265 | 1.622 | 0.770 | −0.852 |
| 4 | `BC1=CC(=C2C=C2)C=C1B` | 3.431 | 1.396 | 0.667 | −0.729 |

**Most-degraded (Δ|err| > 0; PC² worse):**

| idx | SMILES | true λ_T (eV) | |err| FedAvg | |err| PC² | Δ|err| |
|---|---|---|---|---|---|
| 20 | `CN(C)C1=CC=C(N(C)C)C1=C1C(N=[N-])=C1[N+]#N` | 3.164 | 0.217 | 1.088 | +0.871 |
| 47 | `C1=CC=CC(=C2C=CC=C2)C=C1` | 1.113 | 0.927 | 1.637 | +0.710 |
| 30 | `OC1=CC=C(O)C1=C1C=C1` | 2.106 | 0.064 | 0.707 | +0.643 |

### S8.3 Disclaimer

The per-molecule ordering is *qualitative*: PC²'s top-improved molecule under FedAvg-baseline ordering is not necessarily its top-improved molecule under FedProx- or FedPer-baseline ordering, and seed-to-seed variation can permute the bottom-k. Figure S1 and the table above are provided to give chemistry-discipline readers a feel for *which kinds of structures* PC² helps and hurts; they are not the basis of any statistical claim. The statistical claims rest on the paired tests in Section S6.

---

## S9. Reproducibility checklist

### S9.1 Random seeds

Five seeds {42, 123, 456, 789, 1000}.  `set_all_seeds(seed)` initialises `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and sets `torch.backends.cudnn.deterministic = True`.

### S9.2 Hardware and runtime

- GPU box: dual NVIDIA A30 24 GB (CUDA 12.4), dual Intel Xeon Platinum 8260.
- Each federated run takes approximately 30 minutes wall-clock on dual A30.
- The 25-run main batch (5 methods × 5 seeds) plus the 20-run ablation batch (4 ablations × 5 seeds) for this manuscript total approximately 20 GPU-hours.

### S9.3 Code paths

| Component | Path |
|---|---|
| Data loading and graph construction | `src/data_utils.py` |
| GIN / SchNet encoders, KAN / MLP heads, adapter, calibration | `src/models.py` |
| Federated aggregation operators and the `_get_exclude_keys` filter | `src/federated.py` |
| Per-round training, evaluation, paired metrics | `src/train_eval.py` |
| Transferability matrix construction | `src/transferability.py` |
| Experiment registry | `experiments/configs.py`, `experiments/run_all.py` |
| 5-seed batch driver (E60–E63, E66) | `experiments/run_pc2_multiseed.py` |
| Ablation batch driver (E65, E68, E69, E70) | `experiments/run_phase5_step3_4.py` |
| Paired statistical tests | `experiments/run_phase5_step5_stats.py` |
| Figure and table generation | `paper/make_figures.py` |

### S9.4 Result files

| Artefact | Path |
|---|---|
| 5-seed main-batch summary | `results/pc2_batch1_multiseed_summary.csv` |
| 5-seed main-batch per-molecule predictions | `results/pc2_batch1_multiseed_predictions.json` |
| 5-seed ablation summary | `results/pc2_phase5_ablation_summary.csv` |
| 5-seed ablation per-molecule predictions | `results/pc2_phase5_ablation_predictions.json` |
| Paired statistics | `results/pc2_phase5_stats.csv`, `results/pc2_phase5_stats.md` |
| Pre-verification chemistry overlap (V3) | `results/preverify/T_chemistry.json` |
| Compiled transferability matrices | `results/preverify/T_transferability.json` |
| Frozen-info anchor (commit hash + SHA-256) | `logs/pc2_batch1/FROZEN_INFO.txt` |

### S9.5 Generated figures and tables

The six main-text figures and four main-text tables are reproducible from the above artefacts by running `python paper/make_figures.py`. Outputs are written to `paper/figures/` (PDF + SVG + PNG at 600 dpi) and `paper/tables/` (CSV + Markdown). The script reads only the result files and never re-runs training. A failure log is written to `paper/figures/molecule_draw_failures.txt` if any SMILES fails to render; in the present manuscript no failures were recorded.

### S9.6 Unit tests

The repository contains 93 unit tests across four modules:

- `tests/test_transferability.py` (22 tests) — diag = 1, T_head invariants, unknown-protocol clause, T_adapter consistency, JSON shape.
- `tests/test_pc2_aggregator.py` (18 tests) — per-target weight normalisation, exclusion-key filtering, fallback to local copy when all weights are zero.
- `tests/test_train_eval.py` (33 tests) — train/eval loop correctness, LOOCV fold ordering, per-fold calibration installation.
- `tests/test_calibration_isolation.py` (20 tests) — no-leakage checks, round-trip identity, aggregation exclusion.

All 93 tests pass on the commit hash recorded in `logs/pc2_batch1/FROZEN_INFO.txt`.

### S9.7 Transferability JSON invariants

The compiled `T_transferability.json` was checked against the following invariants prior to any aggregation round:

1. diag(T_repr) = diag(T_adapter) = diag(T_head) = 1.
2. T_head[D, C-hole] = T_head[D, C-triplet] = 0.
3. T_head[C-hole, C-triplet] = T_head[C-triplet, C-hole] = 0.
4. T_head[A, C-triplet] = T_head[B, C-triplet] = 0.
5. For every (i, j), T_repr[i, j] ∈ [ε_r, 1]; T_adapter[i, j] is consistent with √(T_repr · T_head) up to the floor ε_a.

A copy of the JSON SHA-256 is recorded in `logs/pc2_batch1/FROZEN_INFO.txt`.

### S9.8 No-leakage calibration checks

The 20 calibration-isolation tests (`tests/test_calibration_isolation.py`) verify, for each LOOCV fold of each task-client, that (i) the held-out test molecule's label never enters `set_calibration(μ, σ)`; (ii) the aggregated state-dict received by each task-client does not overwrite its own (μ, σ) buffers; (iii) the round-trip z-space → eV-space inversion preserves the original label to within 1 × 10⁻⁵ eV on fixed inputs.

---

## S10. Additional limitations and future work

### S10.1 Deferred baselines

- **MOON** (Li, He, Song 2021, CVPR) — model-contrastive federated learning targeting representation drift; deferred to follow-up work since PC²-FedReorg already addresses personalization through architectural separation rather than contrastive regularisation.
- **D-MPNN / ChemProp** (Yang et al. 2019, JCIM) — strong message-passing baseline; deferred due to GPU-hour budget and to keep the comparison anchored to the federated-learning family.

Both baselines are explicitly flagged as recommended additions for a manuscript revision.

### S10.2 Learnable calibration

A learnable affine layer over the standardised output (E72 ablation, planned) was deferred to follow-up work to avoid introducing additional degrees of freedom at the present sample size. With n = 49–53 per private task-client, two extra parameters (a, b) per head are at risk of absorbing noise rather than improving generalisation.

### S10.3 C-hole data ceiling

All evaluated methods produce MAE in the narrow band 0.368–0.414 eV on n = 53 C-hole LOOCV, with no paired test reaching significance. The best classical baseline R² on the same data is −0.10 (KNN). It is unclear from this dataset alone whether C-hole improvements are achievable in this protocol regime. Expanding C-hole through additional ωB97X-D conformer-resolved measurements is the most direct route to breaking the ceiling.

### S10.4 Chemistry gate testing under more diverse client pairs

Because C-hole and C-triplet share the majority of their underlying molecules, T_repr between them is essentially saturated (≈ 0.98–1.00). The chemistry-driven encoder gate is therefore tested on the *easier* regime (very similar chemistry) but not on the *harder* regime (moderate diversity), where a non-uniform gate would actively shape weight vectors. Future federations with task-clients at intermediate Tanimoto-K (≈ 0.4–0.6) would more decisively probe the gate's empirical contribution.

### S10.5 Font availability for current draft figures

The figures distributed with this manuscript draft were rendered using Liberation Serif (a metric-compatible substitute for Times New Roman) because Times New Roman was not installed on either of the project's compute nodes at the time of the present writing pass. For final journal submission, the figure-generation script (`paper/make_figures.py`) should be re-run on a machine that has Times New Roman registered with matplotlib's font manager.

### S10.6 Other future directions

- **Quantity-aware multi-head extension.** A single task-client could own multiple physically distinct quantities sharing the same encoder but with separate quantity-specific heads; this would generalise the C-hole / C-triplet split to richer in-house datasets.
- **Communication compression.** Top-k sparsification of per-key updates, particularly relevant for slow private-lab uplinks.
- **Federated active learning.** Use the per-task-client calibration σ as an uncertainty proxy to prioritise which next-batch molecules each laboratory should label, closing the design loop between modeling and experiment.

---

*End of Supporting Information.*
