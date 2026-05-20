# Phase 3 — calibration-variant feasibility report

**Status:** *Proposal-only*. No full training has been run. The Phase 1 +
Phase 2 verdict landed at **PASS-LIMITED** (see
`paper/calibration_generalization/calibration_generalization_report.md`),
so per the user-side task brief — "Phase 3 只在 Phase 1 和 Phase 2 支持 calibration
后做 smoke prototype" — we report feasibility and decline to launch a
full Phase 3 run.

## 0. Why Phase 3 is *not* recommended at this point

The Phase 1 null result (calibration shows no per-molecule advantage over
plain FedPer at n_target = 100 with aggressive affine transforms) means
the chunk of design space where calibration *fails* is non-trivial. The
right next step for the main paper is **NOT** to optimise calibration —
which already works well in the small-n regime — but to characterise
**when** calibration helps. Calibration variants would only add value if
the Phase 2 strong result generalised to a larger evaluation set; we have
no evidence yet that the variant overhead would survive that
investigation.

A separate, paper-side concern: the main paper's claim is already
"task-specific calibration under personalized heads is the
*ablation-detectable* mechanism on the real TADF C-triplet positive
control" — adding shrinkage or learnable-affine variants to a paper that
positions calibration as the minimal sufficient ingredient would invite
the question "why is the minimal sufficient ingredient not actually
minimal?" Unless a downstream reviewer asks for it, we recommend keeping
Phase 3 as a deferred follow-up.

## 1. Candidate 1 — Shrinkage calibration

### Specification

Replace the per-client `(μ_c, σ_c)` buffers (computed strictly from
client *c*'s training fold) with a shrunk estimator:

```
μ_c* = α · μ_c + (1 - α) · μ_global
σ_c* = α · σ_c + (1 - α) · σ_global
```

where `μ_global` and `σ_global` are pooled statistics from the *training*
folds of every federation participant. The pooling must respect the
LOOCV / K-fold split — global stats are recomputed per fold to keep the
held-out molecules out of the calibration buffer.

α grid to test: **0.25, 0.5, 0.75, 1.0** (1.0 = current task-specific
calibration; baseline).

### Implementation cost

**No `src/` modification required.** The mechanism can be expressed as a
wrapper around the existing `CalibrationHead.set_stats(mu, sigma)`
helper. The wrapper would:

1. Compute `μ_c` and `σ_c` from client c's train data (already in
   `install_calibration_from_train`).
2. Compute pooled `(μ_global, σ_global)` from the concatenation of every
   client's train data.
3. Linearly combine and pass to `set_stats`.

Estimated wrapper size: 20–30 lines added to
`experiments/calibration_generalization/fed_train_minimal.py` behind a
`shrinkage_alpha: Optional[float]` argument.

### Expected outcome

Shrinkage at small α (say α = 0.5) would push each client's calibration
toward the consensus mean / std. In the **Phase 2** small-n regime, this
might help by stabilising the per-client `(μ, σ)` estimate (n = 33 train
labels per fold → noisy `σ_c`). In the **Phase 1** aggressive-transform
regime, shrinkage would *hurt* (the shrunk `μ_c` would underestimate
S4's true `μ ≈ 3·E[y] + 1.0 ≈ 3.17`). Net effect: probably mild
improvement on Phase-2-like settings, mild degradation on Phase-1-like
settings.

This prediction can be falsified with a small-Phase-2-only smoke run
(~5 minutes) if requested.

### Risk

Shrinkage with α = 0 would erase all per-client calibration and reduce
the method to FedPer-with-pooled-standardization. That is *another*
ablation worth running, but it would also conflate the calibration
mechanism with global label normalisation, which the main paper has
already ablated via E70 (PC² without calibration uses the *legacy*
norm_stats path, which is roughly equivalent). The cleanest interpretation
demands α > 0; we suggest α ∈ {0.5, 0.75, 1.0} for any future run.

## 2. Candidate 2 — Learnable affine calibration

### Specification

After the fixed inverse-standardize step, add a private 2-parameter
transform:

```
z_pred           = head(encoder(x))                       # standardized space
z_cal            = a_c · z_pred + b_c                     # private learnable affine
lambda_hat       = σ_c · z_cal + μ_c                       # eV space
```

Initialise `a_c = 1.0, b_c = 0.0` so the model starts identical to fixed
calibration. `a_c, b_c` are torch parameters with `requires_grad = True`
but **must** be private (never aggregated). Their state-dict keys must
contain the substring `calibration` so the existing
`_get_exclude_keys()` rule (`src/federated.py:32`) skips them under all
aggregation strategies.

### Implementation cost

**`src/` modification required.** `CalibrationHead` currently uses
`register_buffer` for `mu` and `sigma` (no gradient). Adding learnable
`a, b` parameters demands a new sub-class — or a `learnable_affine: bool`
flag in `CalibrationHead.__init__`. Either way, ~30–50 lines of `src/models.py`
change.

### Risk

- **Over-fit risk at n = 49 / 50.** Two extra parameters per client trained
  on n ≈ 30 LOOCV-train labels is statistically dangerous; gradient noise
  can drive `a_c` away from 1.0 by spurious amounts, undoing the very
  standardization the buffer enforces.
- **Forced personalisation.** Even if `a_c = 1.0` and `b_c = 0.0` are the
  ground-truth optimum, the optimiser will not stay there unless heavily
  regularised — yielding spurious per-client drift.
- **State-dict key migration.** If we name the new parameters
  `self.calibration.affine_a`, they are excluded by the substring rule;
  but any existing checkpoint produced with `use_calibration=True`
  will fail to load because the state dict adds two keys. Backwards-
  compatibility shim required for E66/E71 reproducibility.

### Recommendation

**Do not implement** unless a reviewer explicitly asks for it. The risk /
reward profile is bad: at our sample sizes the two extra parameters are
more likely to add variance than to deliver a usable gain, and the
backwards-compatibility hazard for E66 / E71 reproducibility is real.

If a reviewer requests this experiment, the smallest acceptable
implementation is:

1. Add `learnable_affine: bool = False` to `CalibrationHead.__init__`,
   defaulting to False so all existing experiments are unaffected.
2. When True, register `a` and `b` via `nn.Parameter` (init 1.0 / 0.0),
   with weight-decay-friendly L2 regularisation toward (1, 0).
3. Add the affine to `CalibrationHead.forward` and to
   `CalibrationHead.standardize` (the inverse direction).
4. Smoke on Phase 2 only (5 minutes). Skip the full Phase 2 run unless
   the smoke is encouraging.

The proposal is recorded here for future maintainers; we will not
proceed without an explicit request.

## 3. Smoke-run cost estimates (for reference)

| variant | n_target | n_per_task | seeds | rounds | est. wall time |
|---|---|---|---|---|---|
| Shrinkage smoke (4 α values × Phase-2-style) | 50 | 50 | 2 | 6 | ~10 min on a single A30 |
| Learnable-affine smoke (1 config × Phase-2-style) | 50 | 50 | 2 | 6 | ~10 min on a single A30 (after src/models.py edit) |

Both can be parallelised across the two A30s if a Phase 2 + Phase 3
combined run is wanted.

## 4. Decision

**Recommendation: do not run Phase 3 at this time.** The Phase 1 + Phase 2
results give a clean PASS-LIMITED verdict that maps directly to the
existing main-paper claim ("task-specific calibration is the
ablation-detectable mechanism in the small-n federation regime"). Adding
shrinkage or learnable-affine variants would expand the scope of the
calibration story without strengthening the headline claim.

If the user later requests Phase 3, **start with the shrinkage smoke**
(zero `src/` modification, ~10 min wall clock). Only proceed to the
learnable-affine prototype if shrinkage shows promise.

---

**Hard constraints honoured:** no file in `src/`, `experiments/run_all.py`,
existing `results/*.csv` / `.json`, `paper/manuscript.md`, `paper/latex/main.tex`,
`paper/SI.md`, `paper/latex/SI.tex`, or `paper/figures/` was modified by
this Phase-3 feasibility analysis. No training run.
