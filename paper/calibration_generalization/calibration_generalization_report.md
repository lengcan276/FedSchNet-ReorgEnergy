# Calibration generalization study — unified report (Phase 4)

**Render date:** 2026-05-20T05:16:05Z  
**Phase 1 config:** seeds=[42, 123, 456, 789, 1000], n_target=100, n_source=200, n_rounds=12, n_folds=5  
**Phase 2 config:** seeds=[42, 123, 456], n_tasks=10, n_per_task=50, n_rounds=8, n_folds=3  

---

## 0. Verdict: **PASS-LIMITED**

Diagnostic numbers: Phase 1 ΔMAE(cal − FedPer) = +0.0101 eV, p = 0.731; Phase 2 improvement rate = 9/10 = 90 %; per-task Wilcoxon p < 0.05 with negative ΔMAE: 7 / 10 tasks.

Calibration generalises **in the small-n_target regime** that matches the main-paper C-triplet positive control. Phase 2 shows a strong and statistically significant calibration advantage on the large majority of n = 50 pseudo-targets. Phase 1 — with n_target = 100 and aggressive affine label transforms — shows **no detectable advantage of calibration over plain FedPer** at our seed budget; local-only training is already competitive in that regime, so the federation itself is not load-bearing. The combined reading: calibration helps when (a) the target is small enough that federation matters AND (b) the sources carry naturally heterogeneous label means. **Do not expand the main-paper claim to large-n targets or to artificial scale-amplification transforms.**

## 1. Phase 1 — label-scale stress test

Synthetic federation built from `data/client_a_b/public_reorg_energy_15210.csv`.

Target client S1 uses identity transform; source clients receive deliberately heterogeneous affine + noise label transforms (S2: y → 2y+0.5; S3: y → 0.5y−0.2; S4: y → 3y+1.0; S5: y + N(0,0.05)).

### Per-method 5-seed-mean MAE on the target (eV)

| method | MAE (mean ± SEM over seeds) | RMSE | R² | n_seeds |
|---|---|---|---|---|
| Local-only (target alone) | 0.3090 ± 0.0096 | 0.4129 | -0.335 | 5 |
| FedAvg, no calibration | 0.3422 ± 0.0123 | 0.4011 | -0.247 | 5 |
| FedPer (baseline) | 0.3575 ± 0.0402 | 0.4525 | -0.963 | 5 |
| **FedPer + calibration** | 0.3402 ± 0.0210 | 0.4344 | -0.649 | 5 |

### Paired statistics (per-target median across seeds; n_pairs = n_target × n_seeds)

| comparison | ΔMAE (eV) | 95% CI | wins/losses/ties | Wilcoxon p |
|---|---|---|---|---|
| fedper_cal vs fedper | +0.0101 | (-0.0250, +0.0478) | 45/55/0 | 0.7310 |
| fedper_cal vs fedavg_no_cal | -0.0431 | (-0.0773, -0.0086) | 62/38/0 | 0.0078 |
| fedper_cal vs local | +0.0433 | (+0.0036, +0.0850) | 48/52/0 | 0.0982 |
| fedper vs fedavg_no_cal | -0.0532 | (-0.0839, -0.0231) | 64/36/0 | 0.0011 |
| fedper vs local | +0.0332 | (+0.0013, +0.0654) | 44/56/0 | 0.0586 |

**Runtime:** 1238 s total (20.6 min) across all method × seed combinations.

## 2. Phase 2 — pseudo-task multi-target validation

10 label-quantile-binned pseudo-targets at n=50 per task. Identity transform on labels; heterogeneity is in label *mean* across targets (quantile slices of the QM9 reorg-energy distribution).

### Calibration vs FedPer headline

- Tasks improved by adding calibration: **9/10** (90%).
- Median ΔMAE (fedper_cal − fedper): **-0.0198 eV** (IQR [-0.0300, -0.0133]).

### Per-task summary (averaged across seeds)

| task | Local-only (target alone) | FedAvg, no calibration | FedPer (baseline) | **FedPer + calibration** |
|---|---|---|---|---|
| T00_quantile0 | 0.0663 | 0.1125 | 0.0541 | 0.0533 |
| T01_quantile1 | 0.0651 | 0.0397 | 0.0458 | 0.0235 |
| T02_quantile2 | 0.0557 | 0.0794 | 0.0450 | 0.0174 |
| T03_quantile3 | 0.0660 | 0.1356 | 0.0631 | 0.0206 |
| T04_quantile4 | 0.0725 | 0.1968 | 0.0545 | 0.0227 |
| T05_quantile5 | 0.0642 | 0.2746 | 0.0455 | 0.0235 |
| T06_quantile6 | 0.0645 | 0.3731 | 0.0540 | 0.0291 |
| T07_quantile7 | 0.0821 | 0.5108 | 0.0840 | 0.0441 |
| T08_quantile8 | 0.1103 | 0.7054 | 0.0939 | 0.0769 |
| T09_quantile9 | 0.3609 | 1.1580 | 0.2867 | 0.3290 |

### Per-task paired statistics (fedper_cal vs fedper)

| task | ΔMAE (eV) | 95% CI | wins/losses | Wilcoxon p |
|---|---|---|---|---|
| T00_quantile0 | -0.0014 | (-0.0119, +0.0089) | 26/24 | 0.7449 |
| T01_quantile1 | -0.0198 | (-0.0267, -0.0130) | 40/10 | 0.0000 |
| T02_quantile2 | -0.0285 | (-0.0350, -0.0221) | 42/8 | 0.0000 |
| T03_quantile3 | -0.0421 | (-0.0507, -0.0334) | 47/3 | 0.0000 |
| T04_quantile4 | -0.0306 | (-0.0371, -0.0237) | 43/7 | 0.0000 |
| T05_quantile5 | -0.0185 | (-0.0250, -0.0123) | 39/11 | 0.0000 |
| T06_quantile6 | -0.0198 | (-0.0270, -0.0129) | 40/10 | 0.0000 |
| T07_quantile7 | -0.0423 | (-0.0546, -0.0297) | 41/9 | 0.0000 |
| T08_quantile8 | -0.0116 | (-0.0285, +0.0045) | 25/25 | 0.2866 |
| T09_quantile9 | +0.0347 | (-0.0063, +0.0755) | 21/29 | 0.1117 |

## 3. Phase 3 status — calibration variants

**Not run.** Per the user-side task brief, Phase 3 (shrinkage and learnable-affine calibration variants) is conditional on Phase 1 + Phase 2 showing calibration to be stably effective. The PASS-LIMITED verdict — strong Phase 2 result paired with a null Phase 1 result — does not meet that bar. A standalone proposal-only feasibility report is at `results/calibration_generalization/calibration_variants/variant_feasibility_report.md`, recording the implementation cost (shrinkage: zero `src/` change; learnable affine: requires a `src/models.py:CalibrationHead` edit) and the risks (over-fit at n ≈ 30 LOOCV-train labels, backwards-compatibility hazard for E66 / E71 checkpoint loading).

## 4. Recommendations for paper integration

- **Keep the main-text headline claim unchanged.** The current manuscript positions E71 (FedPer + per-task-client calibration) as the ablation-detectable mechanism on the C-triplet positive control; this study supports that framing but does NOT extend it to broader regimes.
- **Do not modify the title or abstract.** Adding a controlled-heterogeneity result to the abstract would create an overclaim risk: a reviewer who reads Phase 1 will see a null in the moderate-n regime and conclude the paper is overstating.
- **Recommended SI addition (1 paragraph in §S7.x):** report Phase 2 only (the small-n / label-quantile experiment) as evidence that the calibration mechanism is *not* a C-triplet fluke. Cite Phase 1's null as a sensitivity check that **defines the regime of applicability** (small target, naturally heterogeneous source label distributions).
- **Do not promote E71 above PC²-FedReorg in the title.** The existing 'Calibration-Aware Personalized Federated Learning' framing already captures the mechanism; the PC²-FedReorg name remains an accurate description of the deployed system (auditable compatibility metadata + calibration).
- **Gate stays as audit-only.** Phase 2 demonstrates a calibration effect that PC²-fed-routing is unlikely to amplify (it was not detectable on the real federation either).
- **No update to Figures 4 / 5 / Tables 2 / 3 / 4 needed.** The new Phase-2 figure (if added) belongs in the SI; the main-text claim is unchanged.

## 5. Residual reviewer-risk (PASS-LIMITED specific)

- A reviewer might ask: *"Phase 2 shows calibration helps; why doesn't it transfer to Phase 1?"* Answer: in Phase 1, n_target = 100 is already enough for local-only training to match federated MAE (Local 0.309 vs FedPer 0.358; Wilcoxon Local-vs-FedPer p = 0.059). The federation is not load-bearing in that regime, so neither FedPer nor calibration can demonstrate an advantage. This is **not** a contradiction of the C-triplet result; it is a sample-size boundary.
- A reviewer might ask: *"Did you control for source label mean shift versus source label scale amplification?"* Answer: Phase 1's S2 (2y + 0.5), S3 (0.5y − 0.2), S4 (3y + 1.0) deliberately mix shift and scale; Phase 2's quantile bins are pure mean shift. The contrast is intentional. Both settings exercise the calibration mechanism in different ways.
- A reviewer might ask: *"What happens at n_target = 30?"* Answer: this study did not run that condition. The Phase 2 tasks at n = 50 already approximate the C-triplet n = 49 case; if a reviewer specifically requests n = 30, the same driver supports it via `--n-per-task 30`.

## 6. Hard constraints honoured

- ✅ No existing `results/*.csv` / `.json` was modified or overwritten.
- ✅ No `src/` file or main-paper `experiments/run_all.py` config was touched.
- ✅ No `paper/manuscript.md`, `paper/latex/main.tex`, `paper/SI.md`, `paper/latex/SI.tex`, `paper/figures/`, `paper/tables/`, or `paper/latex/CalibrationAware_FedReorg_Overleaf.zip` was modified.
- ✅ No `.docx` was generated. No `git commit` was issued. No training on the real TADF data was run.
- ✅ All new artefacts live under `experiments/calibration_generalization/`, `results/calibration_generalization/`, and `paper/calibration_generalization/`.
