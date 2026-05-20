# Calibration-Aware Personalized Federated Learning for Molecular Reorganization Energy Prediction

*[Author list, affiliations, and corresponding-author email — to be supplied]*

---

## Abstract

Molecular reorganization energy is a primary descriptor in the design of organic semiconductors and thermally activated delayed fluorescence (TADF) emitters, yet a single experimental laboratory typically produces only a few dozen high-fidelity samples — far below what deep models require for robust generalisation. Federated learning offers a path to pool laboratories and public datasets without exchanging raw structures, but the prevailing toolkit (FedAvg, FedProx, FedPer, FedBN, MOON) was designed for statistical heterogeneity, not for the systematic heterogeneity that distinguishes quantum-chemical regression: distinct physical quantities (hole-λ versus triplet-λ), distinct DFT protocols (B3LYP/6-31G\* versus ωB97X-D with adiabatic four-point Nelsen), and label scales that can differ by more than a factor of two.

We introduce **PC²-FedReorg** (Protocol-Calibrated Personalized Federated Learning), a federation that distinguishes *physical clients* from *task-clients*: distinct quantities measured at the same laboratory are modeled as distinct task-client nodes, each with a private (μ, σ) calibration buffer that is computed strictly from its own training fold and universally excluded from cross-client aggregation. Encoders and an optional residual adapter are shared under a per-key routing scheme defined by an auditable chemistry/protocol transferability matrix; regression heads and calibration buffers remain private.

On a five-node federation (A: QM9 non-aromatic, n = 6,020; B: QM9 aromatic, n = 9,190; C-hole: in-house TADF, n = 53; C-triplet: in-house TADF, n = 49; D: Atahan-Evrenk 2019, n = 5,876) evaluated by five-seed leave-one-out cross-validation, PC²-FedReorg achieves nominally significant MAE reductions on triplet reorganization energy compared with FedAvg, FedProx and FedPer (paired Wilcoxon p = 0.045, 0.045 and 0.042; MAE 0.66 vs. 0.76–0.78 eV), while no statistically separable improvement is observed on hole reorganization energy, consistent with the small-sample data ceiling already evident in classical baselines. A targeted control (E71) further shows that a FedPer baseline endowed only with task-specific calibration — without the residual adapter and without the transferability-driven routing — yields a C-triplet MAE that is not statistically distinguishable from PC²-FedReorg's at our sample size (per-molecule median-aggregated MAE 0.654 vs. 0.660 eV; paired Wilcoxon p = 0.929; 95% bootstrap CI on ΔMAE straddles zero), while yielding nominally lower MAEs than FedPer without calibration (paired Wilcoxon p = 0.028) and than the no-calibration ablation of PC²-FedReorg (paired Wilcoxon p = 0.021). The corresponding 5-seed mean MAEs follow the same ordering (0.653 ± 0.011 vs. 0.671 ± 0.010 eV for E71 vs. PC²-FedReorg). These results are consistent with task-specific calibration under personalized heads carrying the only ablation-detectable signal in our federation; we did not pre-register an equivalence margin and therefore do not interpret the n.s. comparison as evidence that the adapter and the chemistry/protocol gate contribute nothing, only that any additional benefit they confer is below detection at our seed budget and sample size. The transferability matrix is retained as a deterministic, inspectable record of which source–target sharings the algorithm permits — persisted before any training begins — independently of the optimization outcome.

**Keywords:** personalized federated learning; molecular reorganization energy; task-specific calibration; label-scale heterogeneity; quantum chemistry; organic semiconductors; TADF emitters; small-sample molecular learning.

---

## 1. Introduction

The Marcus picture of charge-transfer kinetics identifies the molecular reorganization energy λ — the energetic cost of relaxing the equilibrium geometry following an electron- or hole-transfer event — as a primary determinant of charge mobility in organic field-effect transistors, hole/electron transport layers in organic light-emitting diodes (OLEDs), and the radiative–nonradiative balance in TADF emitters [1–4]. Reliable in-silico λ-screening accelerates the experimental design loop and reduces wet-lab attrition.

In practice, however, λ prediction sits at an uncomfortable intersection. High-fidelity reference values require multi-point geometry optimization and DFT single-point evaluation at consistent functional/basis combinations; a single experimental TADF laboratory typically generates 50–100 such labels per year. Conjugated-molecule λ predictions from neural networks have therefore relied on aggregating laboratories' data into a single pool [5,6], at the cost of losing protocol consistency and obscuring the resulting label-scale heterogeneity.

Federated learning (FL) [7] is a natural response to this asymmetry: a private TADF laboratory can contribute gradient updates to a shared model without releasing its raw molecules, while industrial precompetitive consortia and public-data hosts can participate symmetrically. The standard FL tool-kit — FedAvg [7], FedProx [8], FedPer [9], FedBN [10], and contrastive variants such as MOON [11] — was developed for statistical heterogeneity across clients drawn from the same underlying distribution, not for the systematic heterogeneities present in quantum-chemical labels: (i) the physical quantity being regressed (hole-λ vs. triplet-λ) may differ across clients hosted at the same laboratory; (ii) DFT protocols are unique fingerprints (functional, basis, geometry treatment, charge state, conformer policy), and protocol-mismatched labels are not interchangeable; (iii) label scales can differ by a factor of two or more — a head trained with mean-squared-error loss against z = 0.26 eV (D) and z = 2.64 eV (C-triplet) optimizes primarily toward the larger-scale client, regardless of protocol compatibility.

A natural remedy is to model these heterogeneities explicitly. We propose **PC²-FedReorg**, a personalized federation that (i) distinguishes physical clients from *task-clients* defined by quantity/protocol/scale; (ii) attaches a private, fold-isolated (μ, σ) calibration buffer to each task-client so that loss and metric live in compatible spaces; (iii) maintains a layer-wise separation between universally shared encoder/adapter parameters and quantity-specific private head/calibration parameters; and (iv) encodes the chemistry- and protocol-derived compatibility information into an auditable 5×5 transferability matrix that documents which source–target sharings are permitted at aggregation. Section 4.4 reports a targeted control (E71) showing that, in our federation, the operative element is component (ii) under personalized heads; components (iii) and (iv) serve as auditability scaffolding rather than additional performance drivers, and the framework is therefore better described as *calibration-aware personalized federated learning made auditable*.

PC²-FedReorg has five design components. We list them in decreasing order of empirical support in this study, with explicit boundaries on what the present ablations can and cannot conclude:

1. **Task-specific calibration under personalized heads as the only ablation-detectable component.** A non-trainable (μ, σ) buffer per task-client, computed strictly from each LOOCV training fold and universally excluded from every federated aggregation operator, paired with a private regression head. Loss is computed in z-space; metrics are reported in eV-space. A targeted control (§4.4, E71) shows that a FedPer baseline endowed only with this buffer — without the residual adapter and without the transferability-driven routing — yields a C-triplet MAE that is not statistically distinguishable from PC²-FedReorg's at our sample size (paired Wilcoxon p = 0.929; 95% bootstrap CI on ΔMAE straddles zero), while yielding nominally lower MAEs than FedPer without calibration (p = 0.028) and than the no-calibration ablation of PC²-FedReorg (p = 0.021). Calibration under a private head is therefore the only component whose presence has measurable support in our ablation set; we did not pre-register an equivalence margin and therefore do not claim that the adapter or the gate contribute nothing, only that any additional benefit they confer is below detection at our sample size.
2. **Task-client modeling for heterogeneous cation/hole/triplet λ tasks.** A federated formulation in which hole-λ, triplet-λ and cation-λ measurements are treated as distinct task-client nodes throughout aggregation, head parameters and calibration, even when two task-clients are hosted by the same physical laboratory.
3. **A calibration-aware personalized federated baseline that reproduces PC² gains.** The control variant in (1) — FedPer with a per-task-client (μ, σ) buffer, no residual adapter, no transferability matrix consulted at aggregation — serves as a transparent baseline for heterogeneous-quantity federations and recovers the headline C-triplet improvement of the full PC² architecture.
4. **Deterministic, inspectable compatibility metadata for protocol/quantity constraints.** A 5×5 transferability matrix derived from Morgan-Tanimoto similarity, Murcko scaffold overlap and conservative DFT-protocol clauses, providing a training-data-independent record of which source–target pairs are *permitted* to share parameters. The matrix is computed before any aggregation round, persisted to `T_transferability.json` with the failing-clause `reasons` field, and can be inspected without any model state. We claim three concrete properties for this artifact: it is *deterministic* (a fixed function of declared metadata), *inspectable* (a single JSON file rather than a model-state dump), and *aggregation-operator-independent* (the same exclusions apply under FedAvg, FedProx, FedPer, FedBN and PC²-FedReorg through a shared `_get_exclude_keys` filter). We do **not** claim that the matrix is a performance lever — the ablations and the E71 control in §4.4 do not detect a performance contribution from the chemistry-driven gate weighting over uniform weighting in our particular federation at the present sample size — and we do not test downstream compliance-violation or contamination-prevention scenarios in this manuscript.
5. **A reliable small-sample evaluation protocol.** Five-seed LOOCV with paired Wilcoxon signed-rank tests and bootstrap 95 % confidence intervals on the paired MAE difference; MAE / RMSE are primary, R² is auxiliary.

Section 2 specifies the federation; Section 3 presents the datasets; Section 4 reports the main empirical findings; Section 5 discusses the active ingredient and the role of the gate; Section 6 lists the limitations of the present study; Section 7 concludes.

---

## 2. Methods

### 2.1 Task-client formulation

Let an organic-electronics federation consist of five task-client nodes (Figure 1; Table 1):

| Node | Source | Physical quantity | DFT protocol summary |
|---|---|---|---|
| A | QM9 non-aromatic public | cation reorg λ | same public cation-reorg source protocol; head-sharing with B permitted by same-source shortcut (§2.4) |
| B | QM9 aromatic public | cation reorg λ | same public cation-reorg source protocol; head-sharing with A permitted by same-source shortcut (§2.4) |
| C-hole | in-house TADF lab | hole-λ (neutral → cation) | ωB97X-D, adiabatic 4-point Nelsen [14] |
| C-triplet | in-house TADF lab | triplet-λ (neutral → T₁) | ωB97X-D, adiabatic 4-point Nelsen [14] |
| D | Atahan-Evrenk 2019 [12] | hole-λ | B3LYP/6-31G\* reported; geometry treatment, charge-state handling, and conformer policy not available in metadata |

Critically, C-hole and C-triplet are different *task-clients* even though their underlying molecular structures largely overlap. They measure different physical quantities (different end states), and their label distributions differ by a factor of 2.17 (means 1.215 vs. 2.638 eV; Figure 1b). Merging them into a single regression task corrupts the head's quantity-specific mapping.

The task-client abstraction is a modeling assumption. It does not change which DFT logs are read or how molecules are graphed; it changes how parameters are routed and where (μ, σ) buffers are installed.

### 2.2 Model architecture

Each task-client owns a model decomposed into four logical stages (Figure 2):

```
Mol → Encoder (5-layer GIN, projected to 256-d)    [shared, T_repr-gated]
    → Adapter (residual 256→128→256, zero-initialised to identity)
                                [optionally shared, T_adapter-gated]
    → Head (KAN or MLP, quantity-specific)        [PRIVATE]
    → Calibration (frozen (μ, σ) buffer)           [PRIVATE]
    → λ̂ (eV)
```

- The encoder produces a 256-d chemical-representation vector; we use a five-layer GIN [15] with Jumping-Knowledge concatenation and mean+max pooling, consistent with prior molecular FL benchmarks. Our codebase supports both GIN and SchNet encoders through a common interface; all reported PC² and baseline runs in this manuscript use the 5-layer GIN encoder for fair comparison.
- The adapter is a residual bottleneck initialised to the identity so that, prior to training, removing it has no effect on the network output.
- The head is a three-layer KAN [16] (`efficient-kan` implementation [17]) or three-layer MLP; head parameters are *never* shared across task-clients.
- The calibration carries non-trainable buffers (μ_c, σ_c) per task-client c, installed from the training fold's labels. Training loss is computed in z-space; predictions are de-standardised at inference (Figure 2, formula band).

### 2.3 Federated training and per-key aggregation

Training alternates between local supervised epochs and a round of aggregation. We define aggregation as a per-state-dict-key operation routed by layer kind:

| key substring | weighting | source |
|---|---|---|
| `encoder.*` | T_repr · n_i, renormalised | shared with chemistry gate |
| `adapter.*` | T_adapter · n_i, renormalised | shared with geom. mean gate |
| `head.*` | T_head · n_i, renormalised | quantity-compatible sources only |
| `calibration.*` | skipped | never aggregated |
| `bn.*` / `*.norm.*` | skipped | FedBN-style local statistics |

Aggregation is **per-target**: for each target task-client *j* and each shared parameter key belonging to layer kind ℓ ∈ {encoder, adapter, head}, the contribution of source task-client *i* is weighted by

$$
w_{i \to j}^{(\ell)} \;=\; \frac{T^{(\ell)}_{ij}\, n_i}{\sum_{k} T^{(\ell)}_{kj}\, n_k},
$$

where *n_i* is the supervised sample count of task-client *i* and *T*^(ℓ) is the corresponding transferability matrix (Section 2.4). When all *T*^(ℓ)_kj equal zero for a given target *j*, the target keeps its local copy of the layer (an identity routing fallback that prevents division-by-zero and preserves the FedPer "private when nothing is compatible" intuition). PC²-FedReorg's universal exclusion of calibration is enforced through a shared `_get_exclude_keys` filter applied under every federated strategy (FedAvg / FedProx / FedPer / FedBN / PC²), so that strategy-specific differences are isolated to the aggregation operator itself.

### 2.4 Auditable transferability matrices

Three 5×5 transferability matrices are pre-computed once before training and persisted as `T_transferability.json`. They are deterministic functions of the chemistry and protocol metadata — *independent of any label or model state* — and can be inspected and audited prior to any aggregation round.

**T_repr (encoder gate, chemistry-driven):**
T_repr\[i, j] = max(ε_r, w_K · K_ij + w_S · S_ij),
with K_ij the mean over i-molecules of max-Tanimoto-similarity to j-molecules (Morgan radius 2, 2048 bits; computed with RDKit [18]), S_ij the asymmetric Murcko scaffold overlap, (w_K, w_S) = (0.6, 0.4), and ε_r = 10⁻³.

**T_head (head gate, conservative protocol clauses, evaluated in this order):**

1. **Clause 1 — quantity mismatch:** if quantity(i) ≠ quantity(j), set T_head[i, j] = 0.
2. **Clause 4 — V1 hard rule:** if i = D and j ∈ {C-hole, C-triplet}, set T_head[i, j] = 0 (applied early so that D → C is always attributed to the V1 evidence rather than to any later metadata clause).
3. **Same-source shortcut:** if quantity matches *and* source_id(i) = source_id(j), the pair is treated as protocol-compatible by source and the per-field check (clauses 2 and 3 below) is skipped. This shortcut models the fact that public datasets distributed from a single source are computed by a single canonical protocol, which need not be re-verified field-by-field. In our federation, the shortcut fires only for A ↔ B (both derived from the QM9-public split).
4. **Clause 2 — unknown protocol field:** if the shortcut did not fire and *any* of {functional, basis, geometry, charge_state, conformers} equals `"unknown"` on either side, set T_head[i, j] = 0.
5. **Clause 3 — protocol-field mismatch:** if the shortcut did not fire and any populated protocol field disagrees between i and j, set T_head[i, j] = 0.

For pairs that survive the above, T_head[i, j] equals a label-distribution similarity term L_ij = exp(−α_W · W₁(p̂_i, p̂_j)) · min(σ_i/σ_j, σ_j/σ_i)^γ_L, with α_W = 5.0 and γ_L = 1.0; values below 0.01 are hard-floored to zero. The clause order above matches the deterministic order in `src/transferability.py:t_head_compatibility`, so the `reasons` field of `T_transferability.json` always reports the first failing clause; in our federation this is clause 1 for A/B → {D, C-hole, C-triplet}, clause 4 for D → C-hole, and clause 1 for D → C-triplet.

**T_adapter (interpolating gate):**
T_adapter\[i, j] = max(ε_a, √(T_repr · T_head)).

These rules generate several invariants automatically (Figure 3, T_head matrix):
- T_head\[D, C-hole] = 0 (clauses 2 + 4),
- T_head\[D, C-triplet] = 0 (clause 1),
- T_head\[C-hole, C-triplet] = T_head\[C-triplet, C-hole] = 0 (clause 1),
- T_head\[A, C-triplet] = T_head\[B, C-triplet] = 0 (clause 1).

We emphasize that this matrix is intended as **auditable algorithmic transparency**: it documents which source–target sharings are permitted, independently of any optimization outcome. Whether the gate also improves performance over uniform weighting is a separate empirical question, examined in §4.4.

### 2.5 Evaluation protocol

For Clients A, B and D we use 5-fold cross-validation. For C-hole (n = 53) and C-triplet (n = 49) we use leave-one-out cross-validation (LOOCV). Both protocols are repeated for 5 random seeds {42, 123, 456, 789, 1000}; the per-molecule LOOCV ordering is deterministic.

Per-molecule absolute errors are aggregated as the median across the five seeds, yielding a single 53- or 49-element per-method error vector for paired analysis. Method pairs are compared by:

1. Paired Wilcoxon signed-rank test (two-sided, `zero_method='wilcox'`) on the per-molecule absolute errors.
2. Bootstrap 95 % confidence interval on the paired MAE difference (5,000 resamples of molecule indices).
3. Win / loss / tie counts per molecule.

MAE and RMSE are reported as primary metrics. R² is reported only as an auxiliary; for n ≤ 53 LOOCV regimes, classical baselines (RF, KNN, GP) already produced negative R² (best classical R² = −0.10 on C-hole, +0.08 on C-triplet), which we interpret as a small-sample data ceiling rather than a methodological flaw.

---

## 3. Datasets and Experimental Setup

### 3.1 Datasets

Table 1 summarizes the five task-client nodes. Total label volume is approximately 21,180 molecules, but only ~100 reside in the two private TADF nodes that form the small-sample target of this study (Figure 1).

Public Clients A and B are derived from a single public QM9 cation-reorganization-energy split (`reorg_energy_eV` column of `data/client_a_b/public_reorg_energy_15210.csv`), partitioned by aromatic-ring count (`NumAromaticRings == 0` for A, `≥ 1` for B). Both clients are assigned the same `source_id` in the implementation; their pairwise A ↔ B head-sharing is permitted via the same-source shortcut documented in Section 2.4. Client D (Atahan-Evrenk 2019 [12]) measures hole reorganization energy on a conjugated-organic-semiconductor library; the published metadata report B3LYP/6-31G* but geometry treatment, charge-state handling, and conformer policy are not stated.

The deterministic outcomes of the head gate for D and the C-pair are:
- **D → C-hole:** clause 4 (V1 hard rule) fires and is reported first; clauses 2 (protocol fields unknown for D) and 3 (functional family differs: B3LYP vs. ωB97X-D) would each also block this pair independently. The manuscript therefore treats T_head[D, C-hole] = 0 as a multiply-determined invariant, not contingent on any single piece of evidence.
- **D → C-triplet:** clause 1 (quantity mismatch: hole-λ vs. triplet-λ) is sufficient on its own; the metadata gap for D is irrelevant here.

These outcomes can be audited directly from the `reasons` field of `results/preverify/T_transferability.json` (Figure 3).

The in-house TADF laboratory provides Clients C-hole (n = 53 molecules with valid hole-λ at ωB97X-D) and C-triplet (n = 49 molecules with valid triplet-λ at ωB97X-D). The two C clients share the majority of underlying molecular scaffolds (Figure 3, T_repr) but measure different end-state geometries and therefore yield label means differing by 2.17× (Figure 1, label-scale heterogeneity panel).

### 3.2 Training setup

Encoders are five-layer GIN networks (hidden dim 300, JK-concat, mean+max pooling, projection to 256-d). All KAN heads use grid = 5 and spline order = 3, following the efficient-KAN configuration used throughout this codebase. Adapters are 256-128-256 residual bottlenecks (zero-initialised). Optimiser Adam, learning rate 1 × 10⁻³ for local epochs, cosine annealing. Per-round local epochs = 5 for public clients, 10 for private clients; communication rounds = 30. Batch sizes B = 512 (A, B, D) and full-batch (C). Training is performed on dual NVIDIA A30 (24 GB each) with Client A and D placed on cuda:0 and Clients B, C-hole, C-triplet placed on cuda:1; aggregation is performed on CPU.

The same code path runs all federated strategies (FedAvg / FedProx / FedPer / FedBN / PC²) through a common `get_aggregation_fn` dispatch and `_get_exclude_keys` filter, so that strategy-specific differences are isolated to the aggregation operator and the per-key routing logic.

---

## 4. Results

### 4.1 Task-client heterogeneity along quantity, protocol, and label-scale axes

Figure 1 visualises the five task-client nodes with their representative molecules — each selected deterministically as the molecule whose label is closest to the per-node median. The accompanying bar in Figure 1a quantifies label-scale heterogeneity: D-hole sits at 0.26 eV (mean), A and B at 0.66–0.82 eV, C-hole at 1.22 eV, and C-triplet at 2.64 eV — a 10× spread. Figure 1b casts all five label distributions onto a common axis, demonstrating that no two task-clients share a comparable label scale.

Table 1 lists the per-node DFT protocols. The five task-clients fall into three distinct quantity classes: A and B are cation-reorganization-energy task-clients (public QM9-derived); C-hole and D are hole-reorganization-energy task-clients but they differ in functional (ωB97X-D vs. B3LYP), in published protocol fields (full Nelsen four-point treatment vs. abstract-only metadata), and in chemical space; C-triplet is a triplet-reorganization-energy task-client. All five quantities are reorganization-energy-related but they are not interchangeable regression targets. The conservative T_head clauses (Section 2.4) detect each form of incompatibility: clause 1 (quantity mismatch) blocks every A/B/D pair against C-triplet; clause 1 also blocks A/B against the hole-λ pair (D, C-hole) in the current code, which labels the public QM9 split with a generic "reorg-energy" quantity tag rather than a specific cation label to keep public-private head sharing conservative; clauses 2--4 block D → C-hole jointly (Section 2.4).

These observations motivate the modeling decision that the same lambda symbol does not refer to the same quantity across our five nodes, and that label scales prevent a naive shared regression head from being well-posed. Chemical-space heterogeneity is examined separately through the transferability matrices in §4.2.

### 4.2 Auditable compatibility metadata generates non-trivial constraints from evidence

Figure 3 plots the three transferability matrices. T_repr shows that the C-hole and C-triplet molecule sets are essentially identical from the chemistry perspective (T_repr ≈ 0.98–1.00, asymmetric under our K + S definition), while D is moderately distant from the C nodes (T_repr ≈ 0.05) and the A/B public sets sit in between. T_adapter, as the geometric mean, inherits the same general topology.

T_head shows a deliberately sparser pattern: only A↔B receives a non-zero off-diagonal (0.33), reflecting their shared quantity and DFT protocol. All other off-diagonals are zero. Importantly, the four invariants T_head\[D, C-hole] = 0, T_head\[D, C-triplet] = 0, T_head\[C-hole, C-triplet] = 0, and T_head\[C-triplet, C-hole] = 0 (highlighted in Figure 3) follow deterministically from the protocol metadata and quantity definitions; they do not depend on any training-time signal and can be audited from `T_transferability.json` prior to any aggregation round.

We emphasize the framing: the chemistry/protocol gate is presented here as an algorithmic transparency mechanism, *not* as a claim that the gate per se improves performance over uniform weighting. The empirical decomposition is the subject of §4.4.

### 4.3 PC²-FedReorg achieves nominally significant gains on the triplet task; the hole task is at a data ceiling

Figure 4 reports the main performance comparison on the two private TADF targets, with full numerical breakdown in Tables 2 and 3.

**On C-triplet (n = 49 LOOCV)**, PC²-FedReorg achieves MAE = 0.660 eV, lower than FedAvg (0.763 eV), FedProx (0.784 eV) and FedPer (0.762 eV) by approximately 13–16 % relative. Paired Wilcoxon signed-rank tests on per-molecule absolute errors give p = 0.045 vs. FedAvg, p = 0.045 vs. FedProx, and p = 0.042 vs. FedPer (Table 3; Figure 4 right). We describe these as **nominally significant** rather than significant: the p-values are unadjusted, sit close to the conventional 0.05 threshold, and would *not* survive a Bonferroni correction for the three baseline comparisons performed in this section (α/3 ≈ 0.017). The bootstrap 95 % confidence interval on the PC²-vs-FedAvg paired MAE difference is (−0.220, +0.004) — the upper bound only just touches zero. For the PC²-vs-FedProx and PC²-vs-FedPer comparisons the CI upper bound sits just below zero ((−0.254, −0.007) and (−0.205, −0.002) respectively); both upper bounds are within ~0.01 eV of zero and shift across the zero line under a different bootstrap RNG seed (independent reseed in `paper/audit/` yields (−0.259, +0.002) and (−0.209, +0.007) respectively), so the CI-exclusion-of-zero criterion is fragile at these effect sizes and the Wilcoxon p-values remain the primary inferential anchor. Per-molecule win/loss counts (PC² better / worse) are 31 / 18, 30 / 19 and 32 / 17 respectively (Table 3). We therefore interpret the triplet results as a small-sample but reproducible signal rather than a broad performance guarantee.

**On C-hole (n = 53 LOOCV)**, all evaluated methods produce MAE in a narrow band of 0.368–0.414 eV (Figure 4 left, Table 2). The eight paired comparisons we performed all return p ≥ 0.20 after rounding (Table 3); the bootstrap CIs straddle zero in every case. PC²-FedReorg is statistically indistinguishable from FedAvg, FedProx, FedPer or local-only training on this target.

We attribute the C-hole non-significance to the small-sample data ceiling already identified by classical baselines on the same n = 53 set (best classical R² = −0.101, as also reported in Supporting Information Section S2); we do not claim a hole-task improvement, and we recommend that future work on this regime should report effect sizes alongside p-values for transparency.

**On the public Clients A and B (5-fold CV)** PC²-FedReorg remains competitive with the legacy federated baselines (Table 2). The intention of these clients in our federation is to provide chemistry diversity through encoder sharing rather than direct prediction targets, and the A/B results confirm that PC²'s personalization does not harm performance on the larger public splits.

The targeted FedPer-with-calibration control (E71) introduced in §4.4 reaches the same headline numbers on C-triplet as PC²-FedReorg (MAE 0.654 vs. 0.660 eV; n.s. by paired Wilcoxon, p = 0.929), so the C-triplet improvement reported in this section is not specific to the full PC² architecture and is fully reproduced by a FedPer baseline endowed with the per-task-client (μ, σ) buffer. Section 4.4 develops this decomposition.

### 4.4 Calibration under private heads explains the C-triplet gain

Figure 5 organises the empirical case in two layers. Panel (a) reports a targeted *positive control* — E71 = FedPer + per-task-client (μ, σ) calibration only — that isolates the calibration mechanism without the residual adapter or the transferability-driven routing. Panel (b) reports the four planned PC² component ablations on both C-hole and C-triplet, with numerical detail in Tables 3 and 4. Each ablation row is a paired difference (PC² minus ablation, eV) with bootstrap 95 % CI; negative ΔMAE indicates that PC² is better than the ablation.

**E71 — FedPer + calibration-only reproduces the full C-triplet gain.** Holding the federation, head, optimiser and seed budget fixed but replacing PC²-FedReorg's adapter and per-key transferability gate with a plain FedPer aggregation while *retaining* the per-task-client calibration buffer (E71; specification in `paper/next_experiments_plan.md`, raw outputs in `results/e71_*`), the C-triplet per-molecule median-aggregated MAE moves to 0.654 eV versus 0.660 eV for the full PC²-FedReorg architecture (corresponding 5-seed mean MAE ± seed-SD: 0.653 ± 0.025 vs. 0.671 ± 0.022 eV; the median-aggregated estimand matches Table 3 and Figure 5 panel b, the 5-seed-mean estimand matches Table 2 and Figure 5 panel a). Three paired comparisons jointly support the calibration-is-sufficient reading:

- **E71 vs. PC²-FedReorg (E66):** ΔMAE = −0.006 eV (PC²-FedReorg minus E71: +0.006 eV); bootstrap 95% CI (−0.044, +0.029); per-molecule wins 23 vs. 26; paired Wilcoxon p = 0.929 (n.s.). E71's effect size is well within seed-SEM, the CI straddles zero by a wide margin, and the per-molecule sign count is essentially balanced. By any of the three criteria, E71 and PC²-FedReorg are statistically indistinguishable on C-triplet.
- **E71 vs. FedPer (E63):** ΔMAE = −0.109 eV; bootstrap 95% CI (−0.208, −0.010); per-molecule wins 32 vs. 17; paired Wilcoxon p = 0.028 (nominal). Adding the per-task-client (μ, σ) buffer to a plain FedPer baseline therefore reproduces, with comparable effect size, the nominally significant C-triplet improvement that PC²-FedReorg delivers over FedPer in §4.3.
- **E71 vs. PC²-FedReorg without calibration (E70):** ΔMAE = −0.125 eV; bootstrap 95% CI (−0.224, −0.027); per-molecule wins 33 vs. 16; paired Wilcoxon p = 0.021 (nominal). Stripping calibration out of the full PC² architecture is more harmful on C-triplet than stripping the adapter and the gate out of the calibrated baseline.

These three comparisons report unadjusted nominal p-values; under a Bonferroni correction for the three tests (α/3 ≈ 0.017), neither of the two against-baseline comparisons crosses the corrected threshold, and we therefore treat them as effect-size evidence supported by sign-counts and bootstrap CIs rather than as strict significance. **On C-hole**, E71, E66, E63 and E70 produce MAEs in the same narrow 0.368–0.382 eV band that all other methods occupy (Table 2), and the three E71 paired comparisons return p ≥ 0.42 with bootstrap CIs that straddle zero (`results/e71_vs_*.md`). The C-hole regime remains at its small-sample data ceiling regardless of whether calibration is present.

**Together, E71 and the four PC² ablations are consistent with task-specific calibration under personalized heads carrying the only ablation-detectable signal in our federation. E71 reaches a C-triplet MAE that is not statistically distinguishable from PC²-FedReorg's at our sample size; we did not pre-register an equivalence margin, so this is descriptive evidence of "no detectable difference" rather than a formal claim that the adapter and the chemistry/protocol gate fail to contribute (an equivalence test such as TOST would require a tolerance margin committed to in advance).** The implementations of E66 and E71 differ only in (a) the `use_adapter` flag and (b) the `aggregation_strategy='pc2_fed'` flag in `experiments/run_all.py`; all other code paths, hyperparameters, seeds, optimizer states and LOOCV fold orderings are identical, so the comparison is not confounded by unrelated implementation differences. The supporting four-ablation analysis (Figure 5b, Tables 3 and 4) is consistent with the same reading:

**Removing task-specific calibration (E70):** On C-triplet, ΔMAE = −0.119 eV (95 % CI (−0.225, −0.015), paired Wilcoxon p = 0.043; 35 / 14 per-molecule wins for PC²). The CI sits entirely below zero. On C-hole, the direction is reversed: PC² is 0.005 eV worse than the no-calibration variant, but the difference is well within noise (p = 0.58). Calibration is therefore the **only** ablation that produces a statistically significant change on either target, and the effect is restricted to C-triplet. We conclude that, within our federation, the task-specific (μ, σ) buffers are the active ingredient driving PC²'s C-triplet improvement.

**Removing C-cross chemistry-based encoder sharing (E68):** ΔMAE = +0.027 eV on C-triplet (PC² *slightly worse* than removing the cross-quantity encoder share; p = 0.07, ns) and −0.0001 eV on C-hole (p = 0.56, ns). The full 95 % CI on C-triplet is (+0.001, +0.055), which is consistent with no effect.

**Replacing the chemistry-driven gate with uniform FedAvg weighting (E69):** ΔMAE = +0.043 eV on C-triplet (PC² slightly worse than uniform weighting; bootstrap 95 % CI (−0.001, +0.092); p = 0.18, ns; per-molecule wins 21 / 28 in favor of E69) and +0.012 eV on C-hole (p = 0.43, ns). Although the uniform-gate ablation has a numerically lower MAE than PC²-FedReorg on C-triplet, the paired comparison is not statistically significant and the bootstrap CI crosses zero. We therefore do **not** interpret this numerical difference as evidence either against or in favor of the compatibility gate — it is a small-sample observation that is consistent with no effect at our seed budget. The gate is retained only as auditable compatibility metadata, not as the source of any performance gain.

**D → C supervised pretrain → fine-tune (E65), the negative-transfer control:** ΔMAE = −0.010 eV on C-triplet (p = 0.85, ns) and −0.021 eV on C-hole (p = 0.76, ns). PC²-FedReorg and a naive D → C pretrain are not statistically distinguishable on either target in our setting; we therefore do **not** claim that D's labels necessarily produce negative transfer to C, only that PC² did not extract benefit from D's hole-λ labels (consistent with T_head\[D, C-\*] = 0 by construction).

The combined picture: among the four PC² components inspected here, the task-specific calibration carries the only ablation-detectable signal, and the targeted E71 control above reaches a C-triplet MAE that is not statistically distinguishable from PC²-FedReorg's at our sample size. The chemistry/protocol gate (T_repr cross-routing, T_head sparsification) is not detected as a performance contributor either as an isolated PC² component (E68, E69) or relative to the calibrated FedPer control (E71 versus E66 paired Wilcoxon p = 0.929); we therefore position it in this paper as a deterministic, inspectable record of compatibility decisions rather than as a performance lever (§5).

### 4.5 Per-molecule error analysis (Supporting Information)

For completeness, the per-molecule error decomposition on C-triplet — the three molecules where PC²-FedReorg improves most over the FedAvg baseline and the three where it degrades most — is provided as **Figure S1** in the Supporting Information. The largest improvement reduces |error| from 3.06 eV (FedAvg) to 1.70 eV (PC²) on a halogenated polycyclic conjugated system with a true triplet-λ of 1.13 eV; the largest degradation moves a different system in the wrong direction by 0.87 eV.

These illustrative cases are qualitative observations rather than mechanistic claims; the paired statistics in §4.3 / §4.4 (and not Figure S1) are the basis for the manuscript's claims.

---

## 5. Discussion

Our results decompose into six statements that are independent of each other and should be evaluated separately.

**(i) Task-specific calibration under personalized heads carries the only ablation-detectable signal.** The targeted E71 control (FedPer + per-task-client (μ, σ) calibration only, no adapter, no transferability gate) yields a C-triplet MAE that is nominally lower than FedPer without calibration (paired Wilcoxon p = 0.028, ΔMAE = −0.109 eV) and not statistically distinguishable from the full PC²-FedReorg architecture (p = 0.929; 95 % CI on ΔMAE straddles zero); the symmetric ablation (PC² without calibration; E70) sits 0.125 eV above E71 on C-triplet (paired p = 0.021). These three comparisons jointly support the descriptive statement that, in our federation, the combination "private head + per-task-client (μ, σ) buffer + universal aggregation exclusion of that buffer" is the only configuration whose ablation produces a paired-Wilcoxon-detectable change. We did not pre-register an equivalence margin and therefore do not interpret the n.s. E71-vs-PC² comparison as proving that the adapter and the chemistry/protocol gate contribute nothing; the appropriate reading is that any additional benefit they confer is below detection at our seed budget and sample size.

**(ii) The construct is not merely a single global standardization step in this implementation.** Six concrete implementation properties distinguish the (μ, σ) buffers from a per-dataset global normalization, although we do not claim each property is individually causally identified: (a) they are *fold-isolated* — μ and σ are computed strictly from the (n − 1) training labels of each LOOCV fold and never see the held-out molecule's label; (b) they are *task-client-specific*, so C-hole, C-triplet, A, B and D each carry their own buffer with different means and dispersions; (c) they are *private non-trainable buffers* rather than learnable parameters; (d) they are *universally excluded* from every federated aggregation operator through the shared `_get_exclude_keys` filter, so the buffer is never overwritten by aggregation regardless of the federated strategy in use; (e) they are *paired with private heads* so that the standardized output is consumed by a quantity-specific predictor rather than a shared one; and (f) loss is in z-space while metrics and reporting are in *eV-space*. Properties (a)–(c) keep the calibration leakage-free under LOOCV; property (d) preserves the buffer across federated strategies; properties (e)–(f) couple the construct to the personalized-head architecture. The present ablations identify (a)–(f) as a load-bearing combination but cannot disentangle which subset alone would suffice; we treat this as a follow-up question rather than a proven mechanism statement.

**(iii) Chemistry/protocol gating is retained as a deterministic, inspectable record of compatibility decisions, with no detected performance contribution.** The chemistry-driven encoder routing (E68) and the chemistry-driven weighting versus uniform FedAvg (E69) both leave PC²'s C-triplet performance unchanged within paired-test noise (Table 3); the targeted E71 comparison does not detect an additional C-triplet gain from the gate over the calibrated FedPer baseline at our sample size. We therefore present the gate not as a performance contributor. The narrower set of claims we make about it are properties that can be verified directly from the artifact rather than from training outcomes: the transferability matrices are *deterministic* (computed before any aggregation round as a fixed function of the declared chemistry and protocol metadata), *inspectable* (a single JSON file with a `reasons` field that names the first failing clause for every blocked source–target pair, rather than a model-state dump), and *aggregation-operator-independent* (the per-key T_head routing applies the same exclusion rules whether the federated strategy is FedAvg, FedProx, FedPer, FedBN or PC²-FedReorg, through the shared `_get_exclude_keys` filter). When a reviewer or downstream user asks why D's labels do not contaminate C's head, the answer is a single file (`T_transferability.json`) rather than a model-state inspection. We have not validated downstream compliance-violation scenarios or independently demonstrated label-contamination prevention beyond what the deterministic exclusion rule guarantees by construction; those are out of scope for the present manuscript.

**(iv) E71 narrows the interpretation of PC²-FedReorg.** Prior to E71 the most defensible reading of §4.3 was that PC²-FedReorg's chemistry/protocol-aware per-key routing might be *necessary* for the observed C-triplet improvement. E71 narrows that reading: the same C-triplet performance is reached without consulting the transferability matrix at aggregation time. The framework is therefore better described as a *calibration-aware personalized federation made auditable*: the empirical performance is captured by the private (μ, σ) buffer paired with the private head, while the per-key routing functions as auditability scaffolding. We retain the PC²-FedReorg name to refer to the full instrumented framework (calibration + adapter + gate) and use "FedPer + calibration" or "the calibrated personalized baseline" to refer to the E71 mechanism in isolation. Both are valid configurations of the same code path; the choice between them is a transparency/parsimony trade-off, since at our sample size the added components are not detected to change performance.

**(v) C-hole shows no method separation at the present sample size.** All federated methods evaluated in this study — FedAvg, FedProx, FedPer, PC²-FedReorg, E70, E71 and the local-only baseline — produce C-hole MAEs in a narrow 0.368–0.414 eV band (Table 2). The eight PC²-vs-baseline paired comparisons (Table 3) return p ≥ 0.20 and the three E71 paired comparisons (`results/e71_vs_*.md`) return p ≥ 0.42; the bootstrap CIs straddle zero in every case (p ≥ 0.20 therefore holds across all eleven C-hole paired tests we report). The classical baselines we ran for V2 (random forest, KNN, Gaussian process) already produced negative R² on the same n = 53 set (best classical R² = −0.10; Supporting Information Section S2), which is consistent with a data-limited regime at n = 53 / LOOCV given our descriptor set and seed budget rather than with a method-specific failure. We do not interpret the C-hole non-significance as a deficiency of any single component, nor do we claim a definitive theoretical ceiling; instead we report it as the observed outcome at the present sample size and recommend that future work on C-hole expand the molecule set before attempting method-comparison claims.

**(vi) Reported p-values are nominal and should not be over-claimed.** All p-values in this manuscript are unadjusted paired Wilcoxon signed-rank values on per-molecule absolute errors, where the paired sample unit is each individual molecule (n = 49 for C-triplet, n = 53 for C-hole) and the per-molecule error is the median across the five seeds. The headline C-triplet comparisons in §4.3 — PC²-FedReorg vs. FedAvg / FedProx / FedPer (p = 0.045, 0.045, 0.042) — sit just below the conventional 0.05 threshold and would not survive a Bonferroni correction across the three tests (α/3 ≈ 0.017). The three E71 comparisons in §4.4 (p = 0.929 for E71 vs. PC², p = 0.028 for E71 vs. FedPer, p = 0.021 for E71 vs. no-calibration) carry the same nominal character. The paired Wilcoxon test is our pre-specified primary inferential procedure; we additionally report bootstrap 95 % CIs and per-molecule win/loss counts so that effect size and effect direction can be inspected alongside the p-values. We did not pre-register an equivalence margin, so the n.s. p = 0.929 for E71 vs. PC² should be read as "no detectable difference at our sample size" rather than as a formal equivalence proof. The minimum chemically meaningful improvement in λ prediction for organic-electronics screening is approximately ±0.1 eV (the typical conformer-averaging noise on ωB97X-D reorganization energies for medium-sized π-systems); the C-triplet ΔMAE of −0.10 to −0.12 eV is at the edge of this practical threshold and should be re-evaluated on a larger TADF dataset before being treated as a deployment-ready improvement.

---

## 6. Limitations

We list six concrete limitations that future work should address.

1. **Small-sample data ceiling on C-hole.** All evaluated methods produce indistinguishable MAE on n = 53 LOOCV (Table 3), already above the best classical baseline R² (−0.10). It is unclear from this dataset alone whether C-hole improvements are achievable in this protocol regime; expanding C-hole through additional ωB97X-D conformer-resolved measurements is a plausible next experiment.

2. **Chemistry homogeneity at the high-T_repr pair.** Because C-hole and C-triplet share the majority of their underlying molecules, T_repr between them is essentially saturated. The chemistry gate is therefore tested on the easier regime (very similar chemistry) but not on the harder regime (moderate diversity) where a non-uniform gate would matter more. Future work should test whether the auditable compatibility gate becomes beneficial in federations with more chemically diverse but protocol-compatible clients — for example, multiple cation-reorganization-energy laboratories with the same DFT protocol but distinct scaffold libraries.

3. **MOON and ChemProp baselines deferred.** Our comparison includes FedAvg, FedProx, FedPer and a local-only baseline. We did not run model-contrastive variants (MOON [11]) or message-passing baselines (D-MPNN / ChemProp [13]) in the present manuscript; both are recommended reviewer additions.

4. **Seed budget.** Each federated run on dual A30 takes approximately 30 minutes; the present 25-run main batch plus 20-run ablation batch totals approximately 20 GPU-hours. We use 5 seeds rather than 10; the bootstrap CIs partially compensate for this but cannot fully replace additional repeats.

5. **D's role.** The V1 protocol divergence motivated the T_head clause that excludes D from both C heads. We empirically verified (E65) that a naive D → C supervised transfer is not statistically distinguishable from PC²; the corollary is that we cannot empirically *prove* that the gate decision was correct, only that it was deterministic and auditable. Whether D's structural information contributes useful encoder signal — beyond the FedAvg-like baseline — remains an open question that the E69 ablation does not fully resolve.

6. **Calibration restricted to fixed (μ, σ).** A learnable affine head over the standardized output was deferred to follow-up work to avoid introducing additional degrees of freedom at the present sample size.

7. **Component-isolation ablation settled by E71; further isolation deferred.** The four ablations we ran (no calibration, no C-cross encoder share, uniform gate, D → C pretrain control) jointly disable groups of mechanisms rather than isolating each one. The single most diagnostic missing control — a FedPer baseline with the calibration buffer retained but the adapter and the transferability gate removed — has now been completed as E71 (§4.4). The E71 control settles this question in the present federation: FedPer + calibration is statistically indistinguishable from PC²-FedReorg on C-triplet (paired Wilcoxon p = 0.929; bootstrap 95 % CI on ΔMAE straddles zero), and the residual adapter / transferability gate do not contribute measurable performance over this baseline. The remaining open ablation isolates the *adapter alone* (FedPer + calibration + adapter, no transferability gate); we leave it as a follow-up since the bracketing evidence already in §4.4 (E66 with adapter, E71 without) places the adapter contribution within paired-test noise. We similarly defer a learnable-affine calibration head over the standardized output (item 6 above) and a Probe-augmented T_repr (footnote-level mention in §4.4) to follow-up work.

8. **Small-sample statistical power and multiple-comparison treatment.** With n = 49–53 LOOCV per private target, the paired Wilcoxon test detects ΔMAE on the order of 0.05 eV at p ≈ 0.05. We hedge this regime by reporting bootstrap 95 % CIs alongside p-values rather than relying on a single significance threshold. We also did not apply Bonferroni or other family-wise correction across the three baseline comparisons in §4.3; under Bonferroni (α/3 ≈ 0.017) none of the C-triplet p-values would cross the corrected threshold. The CIs make this borderline character explicit (upper bound +0.004 eV on the PC²-vs-FedAvg comparison). The minimum chemically meaningful improvement in λ prediction for organic-electronics screening is approximately ±0.1 eV (the typical conformer-averaging noise on ωB97X-D reorganization energies for medium-sized π-systems); the C-triplet ΔMAE of −0.10 to −0.12 eV is at the edge of this practical threshold and should be re-evaluated on a larger TADF dataset before being treated as a deployment-ready improvement.

9. **Paired-error independence assumption.** All paired Wilcoxon and bootstrap comparisons in §4.3, §4.4 and Tables 3 and S6 treat per-molecule median-across-seeds absolute errors as paired samples indexed by molecule. Under LOOCV the (n − 1)-fold training sets overlap by (n − 2)/(n − 1) ≈ 96–98 %, so the per-molecule errors are not strictly independent across folds, and seed-aggregation reduces but does not eliminate seed-by-method interaction. We treat this as a standard limitation of LOOCV inference rather than as a defect of any particular method; the bootstrap resampling is over molecule indices (not over fold-level errors) and inherits the same dependence structure. A more conservative inferential framework — for example, a hierarchical model that explicitly models fold-level correlation, or a permutation test over method labels — would be a useful follow-up but does not change the descriptive ordering reported here.

10. **Estimand consistency between summary tables and paired tests.** Table 2 (and Figure 5a) reports the 5-seed mean of per-method MAE_mean (the natural unit for a bar chart with seed-level SEM). Tables 3 and the paired analyses in §4.3 / §4.4 use per-molecule median-across-seeds absolute errors (the natural paired unit for Wilcoxon testing). These two estimands differ in our data by ≤ 0.003 eV (well below seed-SEM), but they are statistically distinct objects, and we have therefore stated the convention explicitly in the Figure 5 caption rather than mixing the two numbers in headline statements.

11. **Calibration generalisation is regime-dependent.** Additional controlled pseudo-federation experiments, reported in the Supporting Information (§S11), support the calibration-centered interpretation in small-target pseudo-tasks: task-specific calibration improved FedPer on 9 of 10 label-quantile pseudo-targets at n = 50 (median ΔMAE = −0.020 eV; paired Wilcoxon p < 0.001 on 7 of 10 tasks). However, an aggressive label-scale stress test at n_target = 100 did not show an advantage of calibration over plain FedPer (ΔMAE = +0.010 eV, paired Wilcoxon p = 0.73), indicating that the calibration benefit observed on the real C-triplet target is regime-dependent rather than universal.

---

## 7. Conclusions

PC²-FedReorg addresses three forms of heterogeneity — physical quantity, DFT protocol, and label scale — that the standard federated-learning toolkit was not designed to handle. The targeted E71 control in §4.4 identifies task-specific calibration under personalized heads — a fold-isolated (μ, σ) buffer per task-client, universally excluded from every federated aggregation operator and paired with a private regression head — as sufficient to reproduce the observed C-triplet performance. A chemistry/protocol transferability matrix and a residual adapter complete the architecture as auditability scaffolding; the ablation evidence and the E71 control do not detect an additional C-triplet contribution from these layers on top of the calibrated FedPer baseline at our sample size, an absence of evidence that we report as such rather than as evidence of absence.

On a five-node organic-electronics federation, the method achieves nominally significant MAE reductions on triplet-reorganization-energy prediction over FedAvg, FedProx and FedPer (paired Wilcoxon p = 0.042–0.045; Figure 4, Table 3) while remaining statistically indistinguishable on hole-reorganization-energy, where all methods sit in a narrow 0.368–0.414 eV band at the present sample size. The E71 control further shows that adding the per-task-client calibration buffer to a plain FedPer baseline (no adapter, no transferability gate) is sufficient to reproduce the observed C-triplet performance (paired Wilcoxon p = 0.929 vs. PC²-FedReorg; p = 0.028 vs. FedPer; Figure 5). The chemistry/protocol gate is positioned as deterministic compatibility metadata (Figure 3) that can be inspected from `T_transferability.json` prior to any aggregation step; a uniform-gate ablation (E69) is numerically slightly better on the triplet task but not statistically distinguishable from PC²-FedReorg (p = 0.18), and the targeted E71 control does not detect an additional C-triplet contribution from the gate beyond the calibrated FedPer baseline at our sample size. We therefore make no performance claim for the gate and recommend that calibration-aware personalized federated learning, with optional auditable compatibility metadata, be considered an appropriate baseline for heterogeneous-quantity quantum-chemistry federations.

We release the source code, configuration files, transferability matrices, and per-molecule prediction outputs to support reproducibility, and we encourage future federated quantum-chemistry work to report effect sizes and paired bootstrap intervals alongside p-values in small-sample regimes.

---

## Data and Code Availability

Source code, configuration files, transferability matrices, and per-molecule prediction outputs are released at `[repository URL — to be supplied]`. The in-house C-hole / C-triplet labels are made available upon request and within the constraints of the originating laboratory's data-sharing agreement; public Clients A, B and D use third-party datasets and are linked in `data/README.md`.

---

## Author Contributions

*[To be completed: lead author, supervisor, computational contributions, chemistry-side contributions, manuscript drafting and revision.]*

---

## Acknowledgements

*[Compute resources, funding sources, data-providing laboratories — to be supplied.]*

---

## References (placeholder list)

[1] Marcus, R. A. *Electron Transfer Reactions in Chemistry: Theory and Experiment.* Rev. Mod. Phys. **65**, 599–610 (1993).
[2] Coropceanu, V. *et al.* Charge Transport in Organic Semiconductors. *Chem. Rev.* **107**, 926–952 (2007).
[3] Sasabe, H. & Kido, J. Multifunctional Materials in High-Performance OLEDs. *Chem. Mater.* **23**, 621–630 (2011).
[4] Uoyama, H. *et al.* Highly Efficient Organic Light-Emitting Diodes from Delayed Fluorescence. *Nature* **492**, 234–238 (2012).
[5] Atahan-Evrenk, S. & Atalay, F. B. Prediction of Intramolecular Reorganization Energy Using Machine Learning. *J. Phys. Chem. A* **123**, 7855–7863 (2019).
[6] Schütt, K. T. *et al.* SchNet: a Continuous-Filter Convolutional Neural Network for Modeling Quantum Interactions. *NeurIPS* (2017).
[7] McMahan, H. B. *et al.* Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS* (2017).
[8] Li, T. *et al.* Federated Optimization in Heterogeneous Networks. *MLSys* (2020).
[9] Arivazhagan, M. G. *et al.* Federated Learning with Personalization Layers. arXiv:1912.00818 (2019).
[10] Li, X. *et al.* FedBN: Federated Learning on Non-IID Features via Local Batch Normalization. *ICLR* (2021).
[11] Li, Q. *et al.* Model-Contrastive Federated Learning. *CVPR* (2021).
[12] Atahan-Evrenk, S. *Sci. Data* dataset, 5,876 conjugated organic semiconductors (2019).
[13] Yang, K. *et al.* Analyzing Learned Molecular Representations for Property Prediction (Chemprop). *J. Chem. Inf. Model.* **59**, 3370–3388 (2019).
[14] Nelsen, S. F., Blackstock, S. C. & Kim, Y. Estimation of Inner Shell Marcus Terms for Amino Nitrogen Compounds by Molecular Orbital Calculations. *J. Am. Chem. Soc.* **109**, 677–682 (1987).
[15] Xu, K., Hu, W., Leskovec, J. & Jegelka, S. How Powerful are Graph Neural Networks? *ICLR* (2019).
[16] Liu, Z. *et al.* KAN: Kolmogorov–Arnold Networks. arXiv:2404.19756 (2024).
[17] Blealtan. efficient-KAN: an efficient pure-PyTorch implementation of Kolmogorov–Arnold Networks. GitHub repository (2024). [DOI / Zenodo release: TODO.]
[18] Landrum, G. *et al.* RDKit: Open-Source Cheminformatics. https://www.rdkit.org (2023).

---

## Figure & Table Index (cross-reference summary)

| Asset | Manuscript section(s) | Function in argument |
|---|---|---|
| **Figure 1** — task-client heterogeneity & median molecules | §1, §3, §4.1 | Establishes the four heterogeneity axes; motivates the task-client abstraction. |
| **Figure 2** — PC² architecture | §2 | Visual summary of encoder + adapter + head + calibration + per-key routing. |
| **Figure 3** — transferability matrices | §2.4, §4.2 | Shows that the protocol/chemistry gate is auditable and deterministic. |
| **Figure 4** — main MAE bars + significance | §4.3 | Headline result: C-triplet significant, C-hole n.s. |
| **Figure 5** — component ablation forest | §4.4 | Calibration is the only ablation-significant component. |
| **Figure S1** — per-molecule cases *(Supporting Information)* | §4.5 | Qualitative illustration; not the basis of claims. |
| **Table 1** — task-client dataset summary | §3, §4.1 | Numerical accompaniment to Figure 1. |
| **Table 2** — main performance | §4.3 | Numerical accompaniment to Figure 4. |
| **Table 3** — paired statistical tests | §4.3, §4.4 | Primary evidence base for all p-value claims. |
| **Table 4** — ablation summary | §4.4 | Numerical accompaniment to Figure 5. |

---

*End of manuscript draft.*
