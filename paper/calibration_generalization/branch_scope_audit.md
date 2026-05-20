# Branch scope audit — `feat/pc2-calibration-generalization`

**Date:** 2026-05-20  
**Working directory:** `/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reorganization/FedSchNet-ReorgEnergy`  
**Branch:** `feat/pc2-calibration-generalization` (created off `main@b7b224b`, no remote yet)  
**Origin:** `git@github.com:lengcan276/FedSchNet-ReorgEnergy.git` (SSH; no PAT)

## Verdict: **proceed to staging plan**

- ✅ no files > 50 MB in scope
- ✅ no checkpoint binaries (`.pt` / `.pth` / `.ckpt` / `.npy` / `.npz`)
- ✅ only one `.pkl` (`data/conformer_3d_cache.pkl`, 19 MB) and it is already
  excluded by `.gitignore` (line `*.pkl`)
- ✅ no `ghp_…` tokens found anywhere outside `.git/`
- ✅ no `*.aux` / `*.bbl` / `*.blg` / `*.out` / `*.toc` / `*.fls` /
  `*.fdb_latexmk` / `*.synctex.gz` LaTeX scratch files in the working
  tree
- ✅ all `*.log` and `logs/` already excluded by `.gitignore`
- ✅ `__pycache__/` excluded by `.gitignore`
- ✅ private TADF labels are not stored in any file in scope — every
  `results/calibration_generalization/predictions/*.json` contains only
  public-QM9-derived `y_true` arrays

## File inventory

`git status --short` reports 58 entries. They split into three buckets:

### Bucket A — modified tracked files (9 files, belong to the post-E71 baseline)

```
 M experiments/configs.py
 A experiments/preverify.py        # staged as A (was previously git add ... ed)
 M experiments/run_all.py
 M plot_paper_figures.py
 M results/figures/fig4_tsne.pdf
 M results/figures/fig7_physics_ml_fusion.pdf
 M src/federated.py
 M src/models.py
 M src/train_eval.py
```

→ **goes into Commit 1.** These are the source-side changes that made
E71 / PC²-FedReorg reproducible.

### Bucket B — untracked files that belong to the post-E71 baseline (Commit 1)

```
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
tests/                              # unit tests for PC² / transferability
generate_docx.py
generate_fig17_subgroup.py
generate_fig1_fig3.py
generate_fig7_physics_ml.py
generate_si.py
generate_si_figures.py
paper/                              # manuscript / SI / figures / tables / audit / latex
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
results/preverify/                  # T_transferability.json + V1/V2/V3 stats
results/tables/centralized_upper_bound.csv
results/tables/experiment_results.csv
results/tables/experiment_results_full.json
results/tables/subgroup_analysis.csv
results/tables/subgroup_optimized.csv
results/figures/fig17_subgroup_analysis.pdf
results/figures/fig1_overview.{drawio,pdf}
results/figures/fig3_fedper_workflow.{drawio,pdf}
results/figures/figS{1_ablation_details,5_centralized_comparison,6_per_molecule_errors}.pdf
```

→ **goes into Commit 1.** This is the complete post-E71 manuscript +
its reproducibility scaffold.

### Bucket C — calibration-generalization scope (Commit 2)

```
experiments/calibration_generalization/
results/calibration_generalization/
paper/calibration_generalization/
```

→ **goes into Commit 2.** Plus the manuscript-side text additions to
`paper/manuscript.md`, `paper/latex/main.tex`, `paper/SI.md`,
`paper/latex/SI.tex` that Step 5 will produce.

### Bucket D — do NOT commit

```
.claude/settings.json               # Claude Code IDE state; user-local
session.md                          # 70 KB conversation export; user-local
paper/__pycache__/make_figures.cpython-310.pyc  # already .gitignored
data/conformer_3d_cache.pkl         # 19 MB; .gitignored via *.pkl
```

→ **excluded.** I will add `.claude/` and `session.md` to `.gitignore`
in Commit 1 to keep them out permanently.

### Bucket E — superseded artefact decision

```
paper/latex/PC2-FedReorg-overleaf.zip          # 2.0 MB old draft zip (legacy name)
paper/latex/CalibrationAware_FedReorg_Overleaf.zip  # 542 KB current zip (post-E71)
```

→ **only ship `CalibrationAware_FedReorg_Overleaf.zip`.** The older
`PC2-FedReorg-overleaf.zip` was the pre-title-change draft from
2026-05-12 and is now stale. I will not stage it. (It will sit
untracked in the working tree; the user can `rm` it later.)

## Detailed scan results

```
$ find . -type f -size +50M -not -path "./.git/*"
(no output — clean)

$ find . -type f \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" \
                  -o -name "*.pkl" -o -name "*.npy" -o -name "*.npz" \) \
                  -not -path "./.git/*"
./data/conformer_3d_cache.pkl    # .gitignored via *.pkl

$ find . -type f \( -name "*.aux" -o -name "*.bbl" -o -name "*.blg" \
                  -o -name "*.out" -o -name "*.toc" -o -name "*.fls" \
                  -o -name "*.fdb_latexmk" -o -name "*.synctex.gz" \) \
                  -not -path "./.git/*"
(no output — clean)

$ grep -RnaIE "ghp_[A-Za-z0-9_]{20,}" . \
        --exclude-dir=.git --exclude-dir=__pycache__ \
        --exclude-dir=.claude --exclude="session.md" --exclude="*.log"
(no output — clean)
```

## Private-data check (TADF)

The C-hole and C-triplet raw labels live in `data/client_c/*.csv`:

```
data/client_c/hole_reorg_molecular.csv
data/client_c/reorganization_energy_summary.csv
data/client_c/hole_reorg_all_conformers.csv
```

These files are **currently tracked in `main`** (they were part of the
initial release `01772b2`). They are also referenced by `data_utils.py`
which the user explicitly authorised at the project-CLAUDE.md level for
the FedSchNet-Reorg release. **No action needed on this commit**; we
are not changing the data policy.

`results/calibration_generalization/.../predictions/*.json` contain
`y_true` arrays only for QM9-derived public molecules — confirmed by
inspecting `experiments/calibration_generalization/pseudo_federation.py`,
which reads exclusively from
`data/client_a_b/public_reorg_energy_15210.csv` and never touches the
TADF datasets.

## .gitignore additions proposed (Commit 1)

Append to existing `.gitignore`:

```
# Claude Code IDE state
.claude/

# Conversation transcripts
session.md
```

Both are user-local artefacts that should not flow to GitHub.

---

**Next step:** generate `staging_plan.md` (Step 4); then Step 5 writes
the manuscript additions, Step 6 recompiles LaTeX + exports Word, Step 7
runs the numeric audit. **No commit happens before Step 8 reports
back to the user.**
