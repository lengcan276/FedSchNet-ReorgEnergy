# Next Experiments Plan — PC²-FedReorg

**Status:** plan only, no experiments executed.

**Purpose:** describe the minimum-viable set of follow-up runs that would close the empirical gaps surfaced in the Phase 1 audit, ranked by reviewer-attack severity. None of these experiments has been run yet; this document is for the lead author to schedule.

**Hard constraints retained from the parent revision:** no changes to `src/`, `experiments/`, `results/`; no claim of negative D → C transfer; no claim that the chemistry gate drives performance.

---

## Summary table

| ID | Title | Pre-submission? | Estimated GPU-hours |
|---|---|---|---|
| **E71** | FedPer + calibration-only (calibration as the active ingredient under FedPer) | **Strongly recommended** | 2.5 h |
| **E72** | Local-only + calibration-only (does calibration help without federation?) | **Strongly recommended** | 0.5 h |
| **E73** | Uniform-gate + calibration-only (does the gate's chemistry weighting add anything on top of calibration?) | Recommended | 2.5 h |
| **E74** | MOON baseline (model-contrastive FL) | Optional (major revision) | 2.5 h |
| **E75** | ChemProp / D-MPNN C-only baseline | Optional (chemistry-reviewer satisfaction) | 1.5 h |

All estimates assume the same dual-A30 setup, 5 seeds, LOOCV on C-hole and C-triplet.

---

## E71 — FedPer + calibration-only  *(runnable spec; not yet executed)*

> **One-line motivation.** PC²-FedReorg's only ablation-supported active component is the per-task-client (μ, σ) calibration buffer (E70 ablation, paired-Wilcoxon p = 0.043 on C-triplet). The natural counter-question is whether the *entire* C-triplet gain over FedPer is explained by adding that same buffer to FedPer — i.e. whether the transferability gate and the residual adapter contribute anything on top. E71 is the minimum-viable control that answers this; without it the §4.4 narrative is open to the attack "PC² simplifies to FedPer + calibration".

### Specification

| component | E63 (FedPer, baseline) | **E71 (this entry)** | E66 (PC²-FedReorg) |
|---|---|---|---|
| encoder aggregation | uniform FedAvg-style, FedPer mask | **uniform FedAvg-style, FedPer mask** | T_repr-gated per-key |
| adapter (residual 256→128→256) | absent | **absent** | present, T_adapter-gated |
| head (KAN, 3-layer) | private (FedPer-excluded) | **private (FedPer-excluded)** | private + T_head-gated for A↔B |
| calibration (μ, σ) buffer | absent | **present, task-client-private** | present, task-client-private |
| transferability matrix | not consulted | **not consulted** | consulted for every layer |

### experiments/run_all.py registration *(insert below E70, ~line 138)*

```python
    # E71: FedPer + per-task-client calibration buffer. No adapter, no T-gate.
    # Purpose: tests whether PC²'s C-triplet gain over FedPer is fully
    # explained by the calibration buffer alone. Differs from E63 only in
    # use_calibration=True; differs from E66 in use_adapter=False AND no
    # aggregation_strategy ('fedper' routing handles encoder/head split).
    'E71': {'desc': 'PC2 ablation: FedPer + calibration only (no adapter, no T-gate)',
            'head': 'kan', 'fed': 'fedper', 'ssl': None,
            'pc2': {'use_adapter': False,
                    'use_calibration': True,
                    # NO aggregation_strategy -> uses fed='fedper' aggregation
                    # NO transferability_path -> uniform mask
                    }},
```

No other code changes required: `train_federated`'s existing branch on `pc2_kwargs.get('use_calibration')` installs the (μ, σ) buffer per LOOCV fold; the calibration key is universally excluded from aggregation by `_get_exclude_keys`, and the FedPer head-private routing is identical to E63's.

### Hyperparameters (identical to E63 / E66, no new tuning)

| field | value | source |
|---|---|---|
| seeds | {42, 123, 456, 789, 1000} | `experiments/run_all.py:get_seeds()` |
| evaluation | LOOCV on C-hole (n=53) and C-triplet (n=49); 5-fold CV on A/B/D | `src/train_eval.py:federated_loocv_c_fast` |
| federation rounds | 50 | `configs.N_FED_ROUNDS` |
| local epochs per round | 5 | `configs.N_LOCAL_EPOCHS` |
| SSL pretrain | none (parity with E63/E66) | `'ssl': None` |
| head | 3-layer KAN, grid = 5, spline order = 3 | `configs.KAN_GRID`, `configs.KAN_SPLINE_ORDER` |
| optimizer | Adam, lr = 1 × 10⁻⁴, cosine annealing | `configs.LR_FINETUNE` |
| early-stop patience | 15 | `configs.PATIENCE` |
| device assignment | A → cuda:0; B/C-hole/C-triplet → cuda:1; D → cuda:0 | `src/train_eval.py` (hardcoded, do not modify) |

### Runnable command (do not auto-execute)

```bash
# On dell-PowerEdge-R940xa (GPU box), in a detached tmux session:
tmux new -s e71
nohup python experiments/run_all.py --exp E71 \
    > logs/E71_$(date +%Y%m%d_%H%M).log 2>&1 &
echo $! > logs/E71.pid
# Ctrl-b d to detach
# Estimated wall clock: ~2.5 h on dual A30 (matches E66 on the same federation).
```

### Outputs to capture

| artifact | path | how it enters the paper |
|---|---|---|
| per-fold predictions | `results/E71_predictions.json` | SI Table S5.x (raw) |
| seed-aggregated summary | `results/E71_summary.csv` | Main-text Table 2 / Table 4 |
| paired absolute-error vector | `results/E71_paired_abs_errors.npy` | feeds the paired-Wilcoxon and bootstrap-CI scripts in `experiments/run_phase5_step5_stats.py` |
| stats vs PC²-FedReorg | `results/pc2_phase5_stats.csv` (appended row) | §4.4 / Figure 5 ablation forest plot |

### Decision tree — what the paper says after E71 lands

The decision uses the **paired-Wilcoxon signed-rank test on per-molecule absolute errors (median across 5 seeds)** on **C-triplet**, with PC²-FedReorg (E66) as the reference. Effect size is the median paired Δ|error|.

**Outcome A — Calibration explains the entire gain.**
*Condition:* `MAE(E71) ≈ MAE(E66)` within seed-SEM, and paired Wilcoxon p ≥ 0.10 vs E66 on per-molecule absolute errors, and bootstrap 95 % CI on the paired ΔMAE straddles zero.
*Action on the manuscript:* **rewrite the mainline as a "calibration-aware personalized FL" framework.** The PC²-FedReorg method becomes a transparency-instrumented variant of "FedPer + calibration"; the adapter and the chemistry/protocol gate become auditability scaffolding rather than performance-affecting components. Specifically:
1. Abstract: replace "PC²-FedReorg achieves nominally significant MAE reductions … on triplet reorganization energy" with "We show that adding a per-task-client (μ, σ) calibration buffer to FedPer accounts for the entire C-triplet MAE reduction observed with our richer PC²-FedReorg architecture, and we therefore position PC²-FedReorg's per-key gate as an auditable transparency layer on top of the simpler calibration-aware FedPer baseline".
2. §1 contributions: demote contributions 2 (adapter) and 4 (gate); promote calibration to *the* contribution; add E71 as the principal evidence.
3. §4.3 / §4.4: replace "the layer-wise gate produces nominally significant gains" with "the gain is fully attributable to calibration, which is the only active component identified by E70 and confirmed by E71".
4. Figure 5 ablation forest: add the E71 vs FedPer (E63) bar to the same panel; the bar should show the calibration-only effect cleanly.

**Outcome B — PC² architecture adds value beyond calibration.**
*Condition:* `MAE(E71) > MAE(E66)` with paired Wilcoxon p < 0.05 (preferably p < 0.017 to survive Bonferroni for the three already-reported comparisons), AND the bootstrap 95 % CI on the paired ΔMAE between E71 and E66 sits strictly below zero.
*Action on the manuscript:* **retain the PC² layer-wise routing claim** with stronger evidence than the current draft. Specifically:
1. Add a sentence to §4.3 / §4.4: "A FedPer-with-calibration control (E71) leaves a residual gap to PC²-FedReorg of Δ MAE = X.XXX eV (paired Wilcoxon p = 0.0XX; bootstrap 95 % CI [−Y.YY, −Y.YY]), demonstrating that the adapter and/or transferability gate contribute beyond the calibration mechanism."
2. Run E72 (Local-only + calibration; already specified below) to further isolate the federation contribution.
3. Keep the Abstract's headline phrasing; add E71 as an additional row to Figure 5.

**Outcome C — Calibration adds nothing to FedPer (E71 ≈ E63).**
*Condition:* `MAE(E71) ≈ MAE(E63)` (paired Wilcoxon p ≥ 0.10 vs E63 on per-molecule absolute errors).
*Action on the manuscript:* **revise the calibration mechanism narrative.** The calibration buffer's benefit then appears only in the *full PC²-FedReorg context* (interaction with the adapter or with T-gated encoder sharing), which complicates §4.4 substantially. In this case:
1. Add a §5 paragraph headed "Calibration appears mechanism-conditional" describing the interaction.
2. Run E72 (Local-only + calibration) — if E72 ≈ E60, calibration's benefit is *strictly* in heterogeneous-federation context.
3. The headline calibration-is-the-only-active-component claim survives the §4.4 paired-Wilcoxon evidence but is reframed as "calibration is active *only inside the layer-wise PC² scaffold*".

**Edge case D — bootstrap CI of E71 vs E66 straddles zero but p < 0.05 in some seeds.**
Treat as Outcome A with a footnote: report both the paired test and the bootstrap CI, note that the test is borderline at our sample size, and frame the calibration-explains-it claim as "consistent with but not strongly proven by" the data.

### Reviewer attack this addresses

> "You claim calibration is the active ingredient (Section 4.4). Then your method's gain over FedPer should be reproducible by adding calibration to FedPer alone. Why isn't this comparison in the paper?"

After E71 lands, the answer is one of: (A) it is, and the comparison is in Table 4 / Figure 5; (B) the comparison shows the architecture adds value; (C) the comparison reveals an interaction worth a dedicated paragraph. All three are defensible; the current draft without E71 is not.

### Mandatory for submission? **Yes — strongly recommended.**

**Why:** the §4.4 conclusion currently rests on E70 alone (no-calibration causes a MAE degradation). The complementary control "yes-calibration, no-rest-of-PC²" is missing; that gap is the single most likely reviewer attack. Cost is ~2.5 GPU-hours on the dual-A30 box. **Do not auto-execute** — the user should schedule and start the tmux session manually.

---

## E72 — Local-only + calibration-only

**Specification:**
- A single GIN+KAN model trained only on the C client's data (no federation, no other clients).
- Adds a task-specific (μ, σ) calibration buffer installed per LOOCV fold from C's training labels.
- Differs from E60 Local-only only in: calibration added.

**Purpose / question:**
Does the calibration buffer help even without federation? If E72 ≈ E60, calibration is acting purely as a *training-side regulariser* in the heterogeneous-federation setting and would not help a single-client experiment.

**Reviewer attack this addresses:**
"Calibration is just per-client standardisation. It should be neutral in the no-federation case where there is no cross-client label-scale conflict."

**Expected outcomes and paper-claim branches:**

| Outcome | Interpretation | What the paper should say |
|---|---|---|
| (A) E72 MAE on C-triplet ≈ E60 Local-only | Calibration is neutral without federation; its benefit lives in the cross-scale interaction with other clients during aggregation | This is the cleanest narrative: calibration matters *because* it isolates the federated head from cross-client scale dominance. Strengthens the "calibration under heterogeneous quantum-chemical FL" framing. |
| (B) E72 MAE < E60 Local-only with p < 0.05 | Calibration helps even single-client training; the federation context is not necessary | Acknowledge that calibration is partly a single-client benefit; revise the SI calibration-mechanism explanation. |
| (C) E72 MAE > E60 Local-only | Calibration hurts in the single-client case — degree of freedom issue | Worth investigating but probably not surprising; the calibration buffer is fixed (μ, σ) so it should not hurt local training in principle. |

**Mandatory for submission?** **Yes, strongly recommended.** This is a single-client run (no federation) at LOOCV n=49/53 — extremely cheap (~30 min total). The result tells us whether the calibration narrative is *federation-context-specific* or *general*.

---

## E73 — Uniform-gate + calibration-only

**Specification:**
- Same as PC²-FedReorg but with uniform FedAvg weighting in the encoder/adapter aggregation (no T-matrix), keeping the adapter and the calibration buffer.
- Essentially: E69 (uniform-gate) is already this run if E69's spec retains calibration. **Confirm.** If E69 does not include calibration, this is a new run.

**Note:** Looking at `results/pc2_phase5_ablation_summary.csv` and the manuscript §S7.4, E69 spec says "calibration and adapter retained", so E69 = E73. No new experiment needed; just re-describe E69 with this framing in the next paper revision.

**Purpose / question:**
Already answered in current results: PC²-FedReorg (E66) vs E69 (uniform-gate + calibration + adapter) is statistically indistinguishable on C-triplet (p = 0.18, E69 numerically slightly better).

**Reviewer attack this addresses:**
"The chemistry gate's weighting carries no measurable contribution on top of calibration."

**What the paper should say:** already says it (current §4.4). No additional run.

**Mandatory for submission?** No — already covered by E69. Use this entry to clarify in the next manuscript revision that **E69 already includes calibration** so it serves as the gate-ablation-with-calibration control.

---

## E74 — MOON baseline (model-contrastive federated learning)

**Specification:**
- MOON [Li, He, Song 2021] applied to the 5-node federation with the same encoder/head/training protocol as PC²-FedReorg.
- MOON's contrastive loss is added at each local epoch on the encoder representation.
- Calibration is **not** included (MOON's original spec does not assume per-client calibration).
- 5 seeds, 5-fold CV on A/B/D, LOOCV on C-hole/C-triplet.

**Purpose / question:**
Does a state-of-the-art personalization-by-regularisation method (MOON) match or beat PC²-FedReorg without the calibration buffer? This would suggest PC²'s edge comes from calibration rather than from architectural personalization.

**Reviewer attack this addresses:**
"You compare against FedAvg/FedProx/FedPer but omit MOON, which is the most-cited recent FL baseline. Why?"

**Expected outcomes and paper-claim branches:**

| Outcome | Paper-claim adjustment |
|---|---|
| (A) MOON MAE on C-triplet > PC²-FedReorg (p < 0.05) | PC²-FedReorg outperforms a contemporary FL baseline; strengthen the comparison narrative. |
| (B) MOON MAE ≈ PC²-FedReorg | Adjust framing to "PC²-FedReorg is competitive with MOON while being architecturally simpler and having an auditable transferability record". |
| (C) MOON MAE < PC²-FedReorg (p < 0.05) | Reduce headline claim. PC²-FedReorg becomes a transparency-focused alternative to a more performant method. Major revision needed. |

**Mandatory for submission?** **Optional but high-value.** If submitting to JCIM / Digital Discovery, the absence of MOON is a likely reviewer-1 complaint. If GPU-hours are tight, defer to the revision round and **acknowledge MOON as a missing baseline** in the manuscript's Limitations §6 (already done).

---

## E75 — ChemProp / D-MPNN C-only baseline

**Specification:**
- ChemProp / D-MPNN [Yang et al. 2019] trained on C-hole and C-triplet only (no federation, no public clients).
- 5 seeds, LOOCV.
- This is a **chemistry-baseline** comparison, not a federation comparison.

**Purpose / question:**
Does the GIN+KAN choice carry over to chemistry-discipline reviewers, or would a more chemistry-canonical message-passing baseline (D-MPNN) close the gap or reverse it on C-triplet?

**Reviewer attack this addresses:**
"Why GIN+KAN and not the standard D-MPNN? Did you compare?"

**Expected outcomes:**

| Outcome | Paper-claim adjustment |
|---|---|
| (A) D-MPNN MAE > FedAvg / similar | GIN+KAN+federation provides genuine improvement over the chemistry-canonical baseline. |
| (B) D-MPNN MAE ≈ PC²-FedReorg on C-triplet | Either federation does not add value beyond a stronger chemistry baseline, or D-MPNN happens to be unusually good at this regime. Discuss honestly. |
| (C) D-MPNN MAE < PC²-FedReorg significantly | Reduce headline claim to "PC²-FedReorg is competitive with single-task chemistry baselines while offering an auditable federation architecture". |

**Mandatory for submission?** **Optional, supplementary.** If the reviewer is chemistry-discipline, this is the second-most-likely complaint after MOON. Cheap to run (~1.5 h on dual A30). Defer to revision if GPU-hours are tight.

---

## Recommended pre-submission action

**Strongly run before submission:** E71 + E72 (total ~3 GPU-hours; both extend the calibration narrative).

**Defer to major revision:** E74 (MOON) and E75 (ChemProp). Acknowledge upfront in §6 Limitations.

**Already covered, no new run:** E73 — clarify that E69 in the current ablation set already provides the gate-ablation-with-calibration control.

---

## Reviewer-attack matrix vs proposed experiments

| Reviewer concern | Addressed by | Defensible without it? |
|---|---|---|
| "Calibration is just standardisation, not novel" | E71 (FedPer + cal), E72 (local + cal) | Partially — current §S4.4 + Discussion (i) frames it; ablations strengthen |
| "Why is the gate even there if E69 ≥ PC²?" | Already in §4.4, §5(ii); E71 to confirm calibration alone explains | Yes if §4.4 phrasing is clear |
| "No MOON baseline" | E74 | Yes — §6 already flags it as deferred |
| "No ChemProp baseline" | E75 | Yes — §6 already flags it as deferred |
| "Quantity-confounded protocol claim" | Already fixed in §2.4 (same-source shortcut documented) | Yes after the Phase 2 revisions |
| "C-hole non-significance" | Future data expansion (cannot fix at this manuscript) | Yes if honestly reported |

---

*This plan is advisory. None of these experiments have been executed; the manuscript and SI numerical tables remain unchanged.*
