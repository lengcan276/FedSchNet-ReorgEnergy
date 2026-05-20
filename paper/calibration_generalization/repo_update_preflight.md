# Phase A — preflight repo + branch safety check

**Date:** 2026-05-20  
**Host:** dell-PowerEdge-R940xa  
**Working directory:** `/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reorganization/FedSchNet-ReorgEnergy`

## Verdict: **BLOCKED — do not commit or push until the user resolves the issues below.**

I stopped Phase A at the safety-check stage. Per the task brief:

> "当前分支必须是 release-n13. 如果不是 release-n13，停止并报告，不要自动切换."
> "如果有未提交的非本任务修改，停止并报告."
> "如果远程不是 https://github.com/lengcan276/FedSchNet-ReorgEnergy 或 origin 指向不一致，停止并报告."

Four blockers fire. Three are upstream of any LaTeX / Word / commit work in
Phases B–G; the fourth (credential exposure) is a separate security issue
that I am flagging here because the embedded token is currently usable
by anyone who has read access to the local git config or to the local
checkout, and any `git push` from this checkout would silently
authenticate with that token.

---

## A1. Wrong local branch

```
$ git branch --show-current
main
```

The task expects `release-n13`. The local repo is on `main`.

- `main` local HEAD: `b7b224b` ("docs: add CLAUDE.md for Claude Code project context")
- `main` remote HEAD: `01772b2` ("Initial release: FedSchNet-KAN for cross-domain reorganization energy prediction")
- `main` is **1 commit ahead** of remote (the CLAUDE.md commit was made
  locally and never pushed; it is not part of this calibration-generalization task).

I did **not** switch branches. The task explicitly forbids auto-switching.

## A2. `release-n13` is not present locally; only on remote

```
$ git ls-remote --heads origin
01772b24...    refs/heads/main
3a72bc95...    refs/heads/release-n13
```

The remote branch exists at `3a72bc9599c8bda2bdf70f53303b01b76b47d9d3`,
but the local repository has not fetched it.

```
$ git branch -a
* main
  remotes/origin/main
```

To proceed, the user must decide whether to:

1. `git fetch origin release-n13` then `git checkout -b release-n13 origin/release-n13`
   — this brings the remote branch local and checks it out; this is the
   most likely correct action.
2. Reset the local repo so it matches `release-n13` directly, discarding
   the local `main`'s ahead-by-one commit (only if that commit should
   not be preserved).
3. Cherry-pick the local `main`'s `b7b224b` (the CLAUDE.md commit) onto
   `release-n13` so the docs change is not lost.

I will **not** take any of these actions without explicit authorization.

## A3. **Credential exposure in the local git config**

```
$ git config --get remote.origin.url
https://lengcan276:<REDACTED-TOKEN>@github.com/lengcan276/FedSchNet-ReorgEnergy.git
```

The remote URL embeds a personal access token (`ghp_…`). The token is
**not** in a committed file; it lives in `.git/config` only, but:

- The token is currently active. Any `git push` from this checkout will
  use it silently.
- If the local `.git` directory is ever zipped, copied to another host,
  or shared with a collaborator, the token leaks.
- The token starts with `ghp_…`, which is GitHub's prefix for
  fine-grained or classic personal access tokens. **It should be
  revoked at <https://github.com/settings/tokens> immediately**, and a
  new token (or, preferably, an SSH key) used going forward.

I will **not** modify `.git/config` (would risk breaking the user's
auth state) but I am flagging this strongly. Any push you authorise me
to do from this checkout will succeed silently using the embedded
token.

Recommended remediation (user-side, not by me):

```bash
git remote set-url origin https://github.com/lengcan276/FedSchNet-ReorgEnergy.git
# Then push with an SSH key, or use the GitHub CLI ("gh auth login")
# Then revoke the old PAT on github.com.
```

## A4. Large pre-existing uncommitted work outside the calibration-generalization scope

```
$ git status --short
 M experiments/configs.py
 A experiments/preverify.py
 M experiments/run_all.py
 M plot_paper_figures.py
 M results/figures/fig4_tsne.pdf
 M results/figures/fig7_physics_ml_fusion.pdf
 M src/federated.py
 M src/models.py
 M src/train_eval.py
?? .claude/
?? experiments/calibration_generalization/                        ← in scope
?? experiments/compute_e71_stats.py
?? experiments/pc2_fedreorg.py
?? experiments/run_batch1.sh
?? experiments/run_centralized.py
?? experiments/run_e71_multiseed.py
?? experiments/run_pc2_multiseed.py
?? experiments/run_phase5_chain.sh
?? experiments/run_phase5_step3_4.py
?? experiments/run_phase5_step5_stats.py
?? experiments/smoke_pc2.py
?? generate_docx.py
?? generate_fig17_subgroup.py
?? generate_fig1_fig3.py
?? generate_fig7_physics_ml.py
?? generate_si.py
?? generate_si_figures.py
?? paper/                                                          ← partially in scope
?? results/calibration_generalization/                             ← in scope
?? results/e71_predictions.json
?? results/e71_summary.csv
?? results/e71_vs_e63_stats.md
?? results/e71_vs_e66_stats.csv
?? results/e71_vs_e66_stats.md
?? results/e71_vs_e70_stats.md
?? results/figures/fig17_subgroup_analysis.pdf
?? results/figures/fig1_overview.drawio
?? results/figures/fig1_overview.pdf
?? results/figures/fig3_fedper_workflow.drawio
?? results/figures/fig3_fedper_workflow.pdf
... (truncated; full list available with `git status --short`)
```

What is here:

- **Modified tracked files (9)** — `src/federated.py`, `src/models.py`,
  `src/train_eval.py`, `experiments/configs.py`, `experiments/run_all.py`,
  `plot_paper_figures.py`, two `results/figures/*.pdf`,
  `experiments/preverify.py` (staged as `A`).
- **Untracked new directories** spanning the entire post-baseline project
  rebuild: `.claude/`, `paper/`, `experiments/*` Phase-5/E71/PC²
  scripts, all `results/e71_*`, all `results/calibration_generalization/`,
  all `results/figures/fig{1,3,17}*.pdf`, and many top-level Python
  files (`generate_docx.py`, `generate_si*.py`, `generate_fig*.py`).

Only **three** of these belong to the current task (calibration-generalization):

1. `experiments/calibration_generalization/` (8 source files + a
   `__pycache__` that must be excluded)
2. `results/calibration_generalization/` (23 CSV/JSON/MD files + the
   `predictions/` subdirectories: 20 + 120 small JSON files totalling
   ~656 KB; no checkpoints; no files > 50 MB)
3. `paper/calibration_generalization/` (currently 2 files; will grow to
   include the audit / README / new SI text once Phase B–E proceed)

Everything else under `git status` is **previous work that should
already be on `release-n13`** but has not been committed. Per the task
rule "如果有未提交的非本任务修改，停止并报告", I must stop.

I will **not** attempt to selectively stash or `git add` the
non-calibration-generalization files. The user needs to:

- Either confirm that `release-n13` already contains all of that
  pre-existing work (and the local `main` checkout is just a stale
  copy), in which case the correct action is to switch to
  `release-n13` first and then re-evaluate `git status` against that
  branch;
- Or commit / stash the pre-existing work first, on a separate branch,
  before this task continues.

## Files-in-scope sanity check (calibration-generalization only)

For reference, the **intended** commit scope is:

```
experiments/calibration_generalization/__init__.py
experiments/calibration_generalization/pseudo_federation.py
experiments/calibration_generalization/fed_train_minimal.py
experiments/calibration_generalization/run_label_scale_stress.py
experiments/calibration_generalization/run_pseudo_task_validation.py
experiments/calibration_generalization/compute_pairwise_stats.py
experiments/calibration_generalization/render_phase4_report.py
results/calibration_generalization/phase0_repo_survey.md
results/calibration_generalization/label_scale_stress/{config*.json,*.csv,*.md,predictions/*.json}
results/calibration_generalization/pseudo_task_validation/{config*.json,*.csv,*.md,predictions/*.json}
results/calibration_generalization/calibration_variants/{config*.json,*.md}
paper/calibration_generalization/calibration_generalization_report.md
paper/calibration_generalization/calibration_generalization_report.md.template
paper/calibration_generalization/repo_update_preflight.md  (this file)
```

Sizes:

```
868 K   results/calibration_generalization
176 K   experiments/calibration_generalization
 20 K   paper/calibration_generalization
```

No file in this scope is > 50 MB.  
No `.pt` / `.pth` / `.ckpt` / `.pkl` / `.npy` / `.npz` file exists in
`results/calibration_generalization/` (confirmed by `find`).  
The only "binary" artefacts are JSON prediction files in `predictions/`
(20 + 120 small files, ~656 KB total). They contain `y_true` arrays of
QM9-derived **public** labels only — **no C-hole / C-triplet private
labels are present**.

## What I am asking the user to confirm before Phase B–G

1. **Branch action.** Should I run
   `git fetch origin release-n13 && git checkout -b release-n13 origin/release-n13`?
   Or do you want to handle the branch transition manually?
2. **Pre-existing uncommitted work.** Is all of the `M` / `??` content
   from `git status` already committed to `release-n13` on the remote
   (i.e., the local `main` checkout is just stale)? If yes, after I
   check out `release-n13` the `git status` should be clean except for
   the calibration-generalization additions. If no, the rebuild work
   needs to be committed first on its own branch, **not** mixed into
   this calibration-generalization commit.
3. **Credential rotation.** Please revoke the `ghp_…` PAT visible in
   the remote URL, then re-issue authentication via SSH key or
   `gh auth login`. I will not modify `.git/config` without explicit
   permission. **Any commit you authorise me to push from this
   checkout uses that exposed token silently.**
4. **Commit scope.** Once `release-n13` is checked out and the
   pre-existing work is resolved, may I proceed with **only** the
   calibration-generalization additions listed above, or do you want
   me to bundle some of the pre-existing work into this commit?

## Phases B–G status

All Phases B–G are **on hold** pending the four resolutions above. None
of the LaTeX / Word / audit / push steps have been started, and I will
not start them until the user replies.
