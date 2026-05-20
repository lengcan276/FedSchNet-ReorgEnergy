# Phase A2 — clean worktree preflight (BLOCKED)

**Date:** 2026-05-20  
**Working repo:** `/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reorganization/FedSchNet-ReorgEnergy`  
**Worktree created at:** `/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reorganization/FedSchNet-ReorgEnergy_release_n13`  
**Worktree branch:** `release-n13` (local) tracking `origin/release-n13` (`3a72bc9`)

## Verdict: **BLOCKED — `release-n13` is a different paper. Do not push calibration-generalization there.**

Phase A2.1 and A2.2 succeeded. Phase A2.3 created the worktree
successfully, but the contents of `release-n13` are **a completely
different paper / project** from FedSchNet-ReorgEnergy. The task brief
assumed `release-n13` would host the FedSchNet-Reorg / E71 / PC² /
calibration-generalization manuscript and supporting files, but it
does not. Pushing the calibration-generalization additions onto
`release-n13` would mix two unrelated papers into one branch and
guarantee a confusing repo state. **Stopping.**

## A2.1 — origin URL repaired (success)

```
$ git remote get-url origin | sed -E 's/ghp_[A-Za-z0-9_]+/ghp_***REDACTED***/g'
https://lengcan276:ghp_***REDACTED***@github.com/lengcan276/FedSchNet-ReorgEnergy.git

$ git remote set-url origin git@github.com:lengcan276/FedSchNet-ReorgEnergy.git
$ git remote -v
origin    git@github.com:lengcan276/FedSchNet-ReorgEnergy.git (fetch)
origin    git@github.com:lengcan276/FedSchNet-ReorgEnergy.git (push)

$ ssh -T git@github.com
Hi lengcan276! You've successfully authenticated, but GitHub does not provide shell access.
```

✅ Origin URL no longer contains the PAT. SSH auth to GitHub works.

> **Reminder (user-side):** please revoke the `ghp_…` token on
> github.com/settings/tokens. The token was active in the local
> `.git/config` until just now and may have been used in prior pushes.

## A2.2 — fetched `release-n13` (success)

```
$ git fetch origin +refs/heads/release-n13:refs/remotes/origin/release-n13
 * [new branch]      release-n13 -> origin/release-n13

$ git rev-parse origin/release-n13
3a72bc9599c8bda2bdf70f53303b01b76b47d9d3

$ git branch -a | grep release-n13
  remotes/origin/release-n13
```

## A2.3 — worktree created (success), but the contents are wrong

```
$ git worktree add ../FedSchNet-ReorgEnergy_release_n13 origin/release-n13
Preparing worktree (detached HEAD 3a72bc9)
HEAD is now at 3a72bc9 docs: add exported session log

$ cd ../FedSchNet-ReorgEnergy_release_n13
$ git checkout -b release-n13
Switched to a new branch 'release-n13'
$ git status --short        # clean
$ git log --oneline -2
3a72bc9 docs: add exported session log
2441973 INVEST paper release — n=13 SCS-CC2 cross-check extension (audit-clean)
```

The worktree is clean. The local branch `release-n13` was created on
the worktree-detached HEAD and tracks `origin/release-n13` via the
prior fetch.

## Why this is BLOCKED

`release-n13` does **not** host the FedSchNet-Reorg / calibration-aware
federated learning paper. It hosts the **INVEST paper** — a different
project on n=13 SCS-CC2 cross-checks for an active-learning study with
delta transfer learning. Evidence:

| expected (per task brief) | actually on release-n13 |
|---|---|
| `paper/manuscript.md` | **absent** |
| `paper/SI.md` | **absent** |
| `paper/latex/main.tex` (Calibration-Aware Personalized FL) | absent (a `paper/main.tex` exists but it is the INVEST paper) |
| `paper/latex/SI.tex` | **absent** |
| `paper/latex/CalibrationAware_FedReorg_Overleaf.zip` | **absent** (a different zip `INVEST_paper_overleaf_n13_v3.zip` exists) |
| `paper/figures/Fig{1..5}_*` | **absent** (only INVEST `paper_overleaf/figures/` directory) |
| `paper/tables/table{1..4}_*.{csv,md}` | **absent** |
| `experiments/` (E60, E63, E66, E70, E71, …) | **absent** |
| `src/` (FedSchNet-Reorg encoder/head/federated.py) | **absent** |
| `experiments/calibration_generalization/` (this task) | **absent** |
| `results/calibration_generalization/` | **absent** (results/ has `adc2_batch2_summary.csv`, `round1_eval`, `round2_eval` etc. — INVEST artefacts) |

What `release-n13` actually contains:

```
AGENTS.md
CHANGELOG.md
CLAUDE.md
data/{source,processed}/        # INVEST data (446 Pollice mols + 33 INVEST)
figures/                        # INVEST figures
INVEST_paper_overleaf_n13_v3.zip  # large legacy zip (3.0 MB)
paper/main.tex                  # INVEST paper LaTeX
paper/audit_reports/            # INVEST audit
paper_overleaf/{main.tex, Table1_invest_candidates.tex, Table2_method_summary.tex, figures/}
README.md                       # INVEST README ("Delta Transfer Learning and Active Learning for INVEST Discovery")
results/{adc2_batch2_summary.csv, round1_eval/, round2_eval/, ...}
reviews/
scripts/{99_emit_canonical.py, audit_numbers.py, build_master_table.py, ...}
session.md
```

Two cleanly separated papers happen to share one git repo: FedSchNet-Reorg
is on `main`, INVEST is on `release-n13`. Pushing
calibration-generalization onto `release-n13` would put files like
`paper/calibration_generalization/` into the **INVEST paper's branch**,
which is not what the task intends.

## What I think happened

The task brief looks like it was written assuming the FedSchNet-Reorg
manuscript currently *lives* on `release-n13` (e.g., that the post-E71
rebuild had been pushed there already). It hasn't. Local `main` is at
`b7b224b` (1 commit ahead of `origin/main` at `01772b2`, "Initial
release: FedSchNet-KAN…") but the post-E71 rebuild — all of `paper/`,
`results/e71_*`, `src/federated.py` / `src/models.py` modifications,
the Overleaf zip, the audit, etc. — has **never** been committed,
either to `main` or to `release-n13`.

## What I need the user to confirm before any Phase B–G work

Pick one of these explicitly. **I will not act otherwise.**

**Option 1 — calibration-generalization goes on `main` (or a new branch off `main`).**

- The entire FedSchNet-Reorg project lives on `main` (currently `b7b224b`,
  which is just the initial release + the CLAUDE.md). All the post-E71
  rebuild work is uncommitted in the dirty working tree.
- Recommended sub-option:
  - First commit on `main`: the post-E71 rebuild + the calibration-generalization
    additions, *as separate commits* on a topic branch like
    `feat/post-e71-rebuild` and `feat/calibration-generalization`.
  - Then merge / PR / fast-forward as you prefer.
- I would do this in the **same** clean-worktree pattern: a new
  worktree off `main` (or a topic branch), and `rsync` in only the
  in-scope files for each commit.

**Option 2 — push calibration-generalization onto `release-n13` anyway.**

- Mix two unrelated papers into one branch's history. **Not recommended**;
  future readers of the INVEST paper repo will see FedSchNet-Reorg
  files appearing out of nowhere.
- If you want this anyway, please say so explicitly. I would still
  need a decision on whether the post-E71 rebuild also goes there.

**Option 3 — create a new branch dedicated to this task.**

- E.g., `git push origin HEAD:feat/calibration-generalization` from a
  worktree built on top of `main`'s `b7b224b`. This keeps both papers
  cleanly separated.
- This is probably the safest path while you decide where the
  FedSchNet-Reorg project should live long-term.

**Option 4 — there is a fourth branch I do not know about.**

- If a branch like `fedschnet-reorg/main` or `feat/post-e71` already
  exists on the remote and I missed it, tell me the branch name and I
  will fetch it. `git ls-remote --heads origin` currently shows only
  `main` and `release-n13`.

## Cleanup status

- ✅ origin URL switched from HTTPS-with-PAT to SSH (PAT no longer in
  `.git/config`).
- ✅ `release-n13` fetched into `origin/release-n13`.
- ✅ Clean worktree at `../FedSchNet-ReorgEnergy_release_n13/` on a
  local `release-n13` branch tracking `origin/release-n13`. **Nothing
  has been added to or modified in this worktree.** Safe to delete via
  `git worktree remove ../FedSchNet-ReorgEnergy_release_n13` if the
  user wants it gone.
- ✅ Phase A3 (rsync calibration-generalization into the worktree) was
  **not** executed — the worktree is the wrong target.
- ✅ Phases B–G all still on hold.

## What I am asking

> Please reply with Option 1 / 2 / 3 / 4 (or describe a custom path).
> I will not touch `release-n13`, `main`, or any branch until you do.
