# Two-commit staging plan — `feat/pc2-calibration-generalization`

**Branch:** `feat/pc2-calibration-generalization` (off `main@b7b224b`,
no remote yet)  
**Goal:** preserve the post-E71 FedSchNet-Reorg baseline as Commit 1, then
add the calibration-generalization robustness analysis as Commit 2.

---

## Commit 1 — Finalize post-E71 calibration-centered PC2 manuscript

### Stage (Commit 1)

**Tracked-and-modified (9):**

```
experiments/configs.py
experiments/preverify.py            # already staged as 'A'
experiments/run_all.py
plot_paper_figures.py
results/figures/fig4_tsne.pdf
results/figures/fig7_physics_ml_fusion.pdf
src/federated.py
src/models.py
src/train_eval.py
```

**Untracked new files (post-E71 build):**

```
.gitignore                          # appended with .claude/ and session.md
experiments/compute_e71_stats.py
experiments/pc2_fedreorg.py
experiments/run_batch1.sh
experiments/run_centralized.py
experiments/run_e71_multiseed.py
experiments/run_pc2_multiseed.py
experiments/run_phase5_chain.sh
experiments/run_phase5_step3_4.py
experiments/run_phase5_step5_stats.py
experiments/smoke_pc2.py
src/transferability.py
tests/__init__.py
tests/test_models_pc2.py
tests/test_pc2_aggregation.py
tests/test_pc2_train_integration.py
tests/test_transferability.py
generate_docx.py
generate_fig17_subgroup.py
generate_fig1_fig3.py
generate_fig7_physics_ml.py
generate_si.py
generate_si_figures.py
results/e71_predictions.json
results/e71_summary.csv
results/e71_vs_e63_stats.md
results/e71_vs_e66_stats.csv
results/e71_vs_e66_stats.md
results/e71_vs_e70_stats.md
results/pc2_batch1_multiseed_predictions.json
results/pc2_batch1_multiseed_summary.csv
results/pc2_phase5_ablation_predictions.json
results/pc2_phase5_ablation_summary.csv
results/pc2_phase5_stats.csv
results/pc2_phase5_stats.md
results/preverify/                   # T_transferability.json + V1/V2/V3 stats
results/tables/centralized_upper_bound.csv
results/tables/experiment_results.csv
results/tables/experiment_results_full.json
results/tables/subgroup_analysis.csv
results/tables/subgroup_optimized.csv
results/figures/fig1_overview.{drawio,pdf}
results/figures/fig3_fedper_workflow.{drawio,pdf}
results/figures/fig17_subgroup_analysis.pdf
results/figures/figS1_ablation_details.pdf
results/figures/figS5_centralized_comparison.pdf
results/figures/figS6_per_molecule_errors.pdf
paper/manuscript.md
paper/SI.md
paper/SI_outline.md
paper/figure_captions.md
paper/final_polish_plan.md
paper/final_pdf_audit.md
paper/next_experiments_plan.md
paper/references.bib
paper/references_unused.bib
paper/reference_todo_audit.md
paper/review_package.md
paper/submission_checklist.md
paper/main.tex                       # the OLD repo-root main.tex
paper/main.docx                      # the existing pre-E71 Word draft (3.5 MB)
paper/supplementary.docx             # 600 KB
paper/make_figures.py
paper/figures/Fig1_task_client_molecules.{pdf,png,svg}
paper/figures/Fig2_pc2_method.{pdf,png,svg}
paper/figures/Fig3_transferability_matrices.{pdf,png,svg}
paper/figures/Fig4_main_performance.{pdf,png,svg}
paper/figures/Fig5_ablation_forest.{pdf,png,svg}
paper/figures/FigS1_per_molecule_cases.{pdf,png,svg}
paper/figures/molecule_draw_failures.txt
paper/tables/table1_task_clients.{csv,md}
paper/tables/table2_main_performance.{csv,md}
paper/tables/table3_paired_tests.{csv,md}
paper/tables/table4_ablation.{csv,md}
paper/latex/main.tex                 # post-E71 LaTeX
paper/latex/SI.tex
paper/latex/main.pdf
paper/latex/SI.pdf
paper/latex/references.bib
paper/latex/README_Overleaf.md
paper/latex/README.md
paper/latex/final_pdf_audit.md
paper/latex/CalibrationAware_FedReorg_Overleaf.zip   # pre-Commit-2 version
paper/audit/                         # full audit suite (13 files)
```

> Note: `results/figures/*.png` is `.gitignored` — those will not show
> up in `git add` even if they exist on disk.

### Do NOT stage (Commit 1)

```
session.md                          # transcript export; added to .gitignore
.claude/                            # IDE state; added to .gitignore
data/conformer_3d_cache.pkl         # 19 MB, .gitignored
paper/__pycache__/                  # .gitignored
paper/latex/PC2-FedReorg-overleaf.zip   # superseded by CalibrationAware_*.zip
experiments/calibration_generalization/      # → Commit 2
results/calibration_generalization/          # → Commit 2
paper/calibration_generalization/            # → Commit 2
```

### Commit 1 message

```
Finalize post-E71 calibration-centered PC2 manuscript

- preserve post-E71 manuscript, SI, figures, tables, and Overleaf package
- include data-provenance audit outputs and corrected estimand wording
- keep claims centered on task-specific calibration rather than gate performance
```

---

## Commit 2 — Add calibration generalization robustness experiments

### Stage (Commit 2)

**Calibration-generalization code:**

```
experiments/calibration_generalization/__init__.py
experiments/calibration_generalization/pseudo_federation.py
experiments/calibration_generalization/fed_train_minimal.py
experiments/calibration_generalization/run_label_scale_stress.py
experiments/calibration_generalization/run_pseudo_task_validation.py
experiments/calibration_generalization/compute_pairwise_stats.py
experiments/calibration_generalization/render_phase4_report.py
experiments/calibration_generalization/README.md            (Step 5 will create)
```

**Calibration-generalization results:**

```
results/calibration_generalization/README.md                (Step 5 will create)
results/calibration_generalization/phase0_repo_survey.md
results/calibration_generalization/label_scale_stress/{config_label_scale_stress.json,
                                                       summary_by_seed.csv,
                                                       summary_by_target.csv,
                                                       pairwise_stats.csv,
                                                       per_fold_results.csv,
                                                       calibration_effect_by_scale_ratio.csv,
                                                       runtime_log.csv,
                                                       smoke_log.md,
                                                       full_run_log.md,
                                                       predictions/*.json}   # 20 JSONs, total ~164 KB
results/calibration_generalization/pseudo_task_validation/{config_pseudo_task_validation.json,
                                                          summary_by_seed.csv,
                                                          summary_by_task.csv,
                                                          pairwise_stats.csv,
                                                          per_fold_results.csv,
                                                          pseudo_task_definitions.csv,
                                                          improvement_distribution.csv,
                                                          improvement_summary.csv,
                                                          runtime_log.csv,
                                                          smoke_log.md,
                                                          full_run_log.md,
                                                          predictions/*.json}   # 120 JSONs, total ~492 KB
results/calibration_generalization/calibration_variants/{config_calibration_variants.json,
                                                         variant_feasibility_report.md}
```

**Calibration-generalization paper artefacts:**

```
paper/calibration_generalization/README.md                                    (Step 5 will create)
paper/calibration_generalization/calibration_generalization_report.md
paper/calibration_generalization/repo_update_preflight.md
paper/calibration_generalization/repo_update_preflight_clean_worktree.md
paper/calibration_generalization/branch_scope_audit.md
paper/calibration_generalization/staging_plan.md                              (this file)
paper/calibration_generalization/audit_generalization_text_numbers.py         (Step 7 creates)
paper/calibration_generalization/generalization_text_number_audit.csv         (Step 7 creates)
paper/calibration_generalization/generalization_text_number_audit.md          (Step 7 creates)
```

**Manuscript-side text additions (modifications of Commit-1 files):**

```
paper/manuscript.md          (+1 sentence in Limitations; +SI subsection link reference)
paper/SI.md                  (+1 subsection "Controlled pseudo-federation tests of task-specific calibration"; +1 table)
paper/latex/main.tex         (+1 sentence; matching LaTeX)
paper/latex/SI.tex           (+1 subsection + 1 table; LaTeX form)
paper/latex/main.pdf         (recompiled)
paper/latex/SI.pdf           (recompiled)
paper/latex/CalibrationAware_FedReorg_Overleaf.zip                            (Step 6 rebuilds)
paper/latex/CalibrationAware_FedReorg_Overleaf_pre_generalization.zip         (Step 6 backs up)
```

**Word exports:**

```
paper/word/CalibrationAware_FedReorg_with_calibration_generalization.docx     (Step 6 creates)
paper/word/SI_with_calibration_generalization.docx                            (Step 6 creates if applicable)
```

### Do NOT stage (Commit 2)

```
session.md
.claude/
paper/latex/PC2-FedReorg-overleaf.zip
paper/__pycache__/
```

### Commit 2 message

```
Add calibration generalization robustness experiments

- add controlled label-scale stress and pseudo-task validation scripts
- add calibration-generalization result summaries and reproducibility notes
- add SI robustness discussion of small-target pseudo-federation results
- keep main manuscript claims conservative and calibration-centered
- regenerate Overleaf package and Word export
```

---

## Open questions for the user (defer until Step 8)

1. The "post-E71 build" untracked files include some **older Word files**
   from March 2026 — `paper/main.docx` (3.5 MB), `paper/supplementary.docx`
   (600 KB). They predate the post-E71 rewrite and may not match the
   current `paper/manuscript.md`. We can either:
   - **(a) Include them in Commit 1** as historical artefacts and let
     Commit 2's new Word file (`paper/word/CalibrationAware_*.docx`)
     be the canonical Word version.
   - **(b) Skip them from Commit 1.** Cleaner repo history; older Word
     files stay untracked in the working tree.

   Default in this plan: **(a)**, on the grounds that they document a
   previous draft state and are not wrong, just superseded.

2. `paper/main.tex` (a 1.5 MB file at `paper/main.tex`, NOT in
   `paper/latex/`) is an older, pre-rename copy of the manuscript. The
   canonical post-E71 LaTeX source is `paper/latex/main.tex`. We can
   either:
   - **(a) Include both.** Mild duplication; the older file is the
     historical reference.
   - **(b) Skip the repo-root `paper/main.tex` from Commit 1.**

   Default in this plan: **(a)** for the same reason as #1.

3. **`paper/latex/PC2-FedReorg-overleaf.zip`** (2.0 MB) is the pre-rename
   draft Overleaf zip. **Default: exclude.** It is superseded by
   `paper/latex/CalibrationAware_FedReorg_Overleaf.zip`, and the
   `_pre_generalization` backup that Step 6 will create.

These defaults can be revisited in Step 8 (final go/no-go).

---

## Step 8 abort criteria (commit will NOT happen unless ALL pass)

- ✅ `git branch --show-current` returns `feat/pc2-calibration-generalization`
- ✅ `audit_generalization_text_numbers.py` reports `0 ERROR`
- ✅ `main.pdf` and `SI.pdf` compile cleanly (no `! ` errors, no undefined refs)
- ✅ Word file exists and a `grep` for `"9 of 10" "+0.010" "p = 0.73" "−0.020" "regime-dependent"` returns all five
- ✅ No `ghp_…` token anywhere in staged content
- ✅ No file > 50 MB staged
- ✅ No file matching the exclude patterns (`*.pkl` / `*.pt` / `*.aux` / etc.) staged
- ✅ User explicitly authorises commit after reading the Step 8 report
