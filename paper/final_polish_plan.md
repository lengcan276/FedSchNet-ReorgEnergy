# Final-Polish Plan — PC²-FedReorg

**Goal:** lift the current Major-Revision baseline from a careful draft to a submission-ready SCI manuscript by tightening framing, fixing residual figure/table risks, and removing project-report register from the prose. **No new data, no new experiments.**

## 1. Current strongest claim
> **Task-specific calibration is the only ablation-supported active component of PC²-FedReorg.** Removing the per-task-client (μ, σ) buffer (E70) produces a paired-Wilcoxon-significant degradation on C-triplet (Δ MAE = −0.119 eV; bootstrap 95 % CI entirely below zero; p = 0.043; per-molecule wins 35 / 14). This claim is robust under our seed budget, traceable to `results/pc2_phase5_stats.csv`, and consistent with the proposed mechanism (label-scale dominance prevention in eV-space loss).

Secondary supporting claims:
- C-triplet: PC² nominally lower MAE than FedAvg/FedProx/FedPer (p = 0.045/0.045/0.042; 13–16 % relative).
- Public clients A and B: PC²'s personalization does not harm public-split performance.
- Auditable transferability matrix exists deterministically before any aggregation round.

## 2. Current weakest risks

| Risk | Severity | Where surfaced |
|---|---|---|
| Table 1 (main-text inline §2.1) currently reads `source-consistent public QM9 protocol` for A/B's *protocol summary*, but `paper/tables/table1_task_clients.csv` still has separate columns for functional / basis / geometry that show `source-consistent public protocol` in all four — readable, but main-text Table 1 (`tab:task-clients` rendered in latex) has 8 separate columns including some unknowns from the CSV merge. Reviewer can spot inconsistency between main-text Table 1 and tables/table1_task_clients.* | Medium | §2.1, Table 1, tables/table1_task_clients.* |
| Phrase "We deliberately accept" (manuscript line 57) reads like project-report disclaimer | Low | §2.1 |
| Phrase "We agree:" (manuscript line 224) and "The honest report is..." (line 226, SI:401) read like reviewer-response prose | Low | §5 Discussion (ii) and (iii) |
| Main text carries 6 figures + 4 tables. Typical SCI main text supports ≤ 5 figures + 2–3 tables. Fig 6 (per-molecule cases) is already flagged as SI candidate but not actually demoted. | Medium | §4.5, Figure & Table Index |
| Figure 4 uses bar plot. Point plot with SEM and significance annotations is more scientific for nominal-significance reporting. | Low | Figure 4 |
| Figure 5 forest-plot annotations don't explicitly call out E69 numerical-but-not-significant superiority. | Low | Figure 5 |
| British / American spelling inconsistent — `optimisation`, `generalising`, `modelling` appear alongside US-style elsewhere. | Low | Throughout |

## 3. Manuscript sections to revise

### §Abstract
- Replace the third-person disclaimer chain with tighter framing. Lead with the problem, the method, and the ablation-supported claim — gate downgrade fits in one sentence.

### §1 Introduction contributions
- Already correctly ordered after Major Revision. Tighten language; remove the bullet-style list (rewrite as a short paragraph or a 5-item compact list, but with sharper verbs).

### §2.1 task-client formulation
- Delete the "We deliberately accept …" sentence; replace with one sentence of the form "The task-client abstraction is a modelling assumption that does not change which DFT logs are read; it changes how parameters are routed and how (μ, σ) buffers are installed."
- The §2.1 task-client inline Table needs to be fully consistent with `paper/tables/table1_task_clients.csv`. Either consolidate to a single Protocol/Head-sharing pair of columns or keep eight columns but make every cell consistent.

### §4.3 main performance
- Restructure to the user's four-paragraph plan (C-triplet → interpret nominal significance → C-hole n.s. → A/B unharmed).
- Adopt the user's recommended phrasing about nominal significance and reproducible-signal-vs-broad-guarantee.

### §4.4 ablations
- Reorder: E70 first (the only supported component); then E68/E69 grouped as the gate-not-driver evidence; then E65 as negative-transfer control with explicit "not demonstrated".
- Adopt the user's recommended single-paragraph summary that names the auditability-mechanism framing.

### §5 Discussion
- Refactor (i)/(ii)/(iii) framework but answer six explicit questions in narrative form:
  1. Why is calibration not "just standardisation"?
  2. Why keep the gate if uniform-gate is numerically better?
  3. Why useful for chemistry-domain FL?
  4. Why no C-hole improvement?
  5. Why not claim D→C negative transfer?
  6. Why nominal significance?
- Remove "We agree" and "The honest report is" registers.

### §6 Limitations
- Currently 6 items. Trim to ≤ 5 by merging item 1 (C-hole ceiling) with the calibration-only future-work direction.

### §7 Conclusions
- Already tightened in Major Revision. One additional copy-edit pass for word repetition.

## 4. Figures to redraw or demote

| Asset | Action | Rationale |
|---|---|---|
| Figure 1 | Redraw: enlarge molecules; remove mean/std/range from per-panel captions (move to Table 1); reduce point-plot density in panel (b) | Currently cluttered |
| Figure 2 | Redraw: clean architecture; emphasise calibration as the active component; remove dense in-figure text; mention per-key routing concisely | Currently busy |
| Figure 3 | Redraw: highlight A↔B same-source nonzero cell with explicit colour box; add sub-title "Compatibility metadata, not performance driver" | Currently reads as "the algorithm's key matrix" |
| Figure 4 | Redraw: change bar plot to point plot with SEM; significance brackets labelled "nominal p = …" | Bar plots oversell discrete contrast |
| Figure 5 | Redraw: explicit "nominal p = 0.043" on E70 row; explicit "uniform gate numerically lower, n.s." on E69 row; x-axis label "ΔMAE = MAE(PC²) − MAE(ablation), negative ⇒ PC² better" | Currently relies on caption to convey nuance |
| **Figure 6** | **Demote to SI as Figure S1.** Replace the §4.5 paragraph with a one-line pointer "Representative molecule-level cases are shown in Figure S1." | Main text is too crowded; Fig 6 is qualitative |

## 5. Main-text tables to keep / move

| Table | Action |
|---|---|
| Table 1 (task-client summary) | **Keep in main text.** Compact format: 6 columns (Task-client, Source, n, Quantity, λ mean ± std, Protocol note + Head-sharing note merged). |
| Table 2 (main performance) | **Keep in main text.** 5 methods × 4 targets = 20 rows. |
| Table 3 (paired statistical tests) | **Keep but reduce to key rows** in main: PC² vs E60, E61, E62, E63 on both targets + PC² vs E70 (calibration ablation) on both targets = 10 rows. Move the other 6 rows (E65, E68, E69 details) to SI Table S3. |
| Table 4 (full ablation) | **Move to SI as Table S4.** The main-text key rows are already in Table 3 (the merged "key paired statistics + ablation rows"). |

## 6. Language unification

- American English throughout (`optimization`, `generalize`, `model` not `modelling`, `behaviour` → `behavior`).
- Single notation for the method: `PC²-FedReorg` (no `PC2-FedReorg`, no line-break in `PC²`).
- Remove `we deliberately accept`, `we agree`, `the honest report is`, `we are careful to`.
- Limit `not a performance lever` and `only ablation-supported active component` to 1–2 occurrences each so they retain weight.
- Avoid em-dashes back-to-back; favour commas and short sentences.

## 7. MCP review checkpoints

- **Phase 0 (now):** `mcp__reviewer__review_plan` on this plan — sanity check before patching.
- **Phase 3 (after Abstract / Intro contributions / §4.3 / §4.4 / §5 / §6 / §7 are patched):** `mcp__reviewer__adversarial_review` on the patched sections.
- **Phase 4 (after `paper/make_figures.py` is edited):** `mcp__reviewer__review_code` on the edits.
- **Phase 8 (final):** `mcp__reviewer__adversarial_review` on the entire revised manuscript + captions.

For each MCP turn, blocking items will be patched; the loop is bounded to two rounds per phase. Any unresolved blocking item after round 2 is written to `paper/final_blocking_issues.md` rather than papered over.

## 8. Out of scope (explicit)

- `src/`, `experiments/`, `results/` are read-only.
- No new training, no new experiments, no docx, no git commit.
- The 5 references with `note = {TODO: verify DOI}` in `references.bib` remain TODO (Sasabe2011_OLED, AtahanEvrenk2019_RE, Nelsen1987, EfficientKAN2024, RDKit2023). No DOI is invented in this pass.

---

## 9. MCP review_plan feedback (round 1) and integration decisions

Round 1 reviewer (gpt-5.4) returned six points. Decisions documented here before patching.

| MCP item | Severity claimed | Decision |
|---|---|---|
| (a) Without `FedPer + calibration-only`, the "only driver" claim is overstated | BLOCKING | **Partial adopt.** New experiments are forbidden by the user. We instead narrow the wording from "only ablation-supported active component" to "the only component, among the four ablations evaluated in this study, whose removal produced a statistically supported degradation"; the missing FedPer+calibration control is explicitly added to §6 Limitations. |
| (b/d) Why keep the gate when E69 is numerically slightly better? Currently not transparent | RECOMMEND | **Adopt.** Add a dedicated subsection in §5 Discussion that answers this question explicitly (auditability before training, deterministic invariants, no measured cost beyond noise). |
| (c) Table 1 main vs CSV may be inconsistent | BLOCKING (misread) | **Reject** as blocking; the CSV and main-text table were already unified in the Major-Revision pass. We will run a final grep to confirm and document the result in the submission checklist. |
| C-hole non-improvement and D → C non-claim should be more visible | RECOMMEND | **Adopt.** Surface the non-claim of D → C negative transfer earlier in the Discussion; add a single concise sentence on C-hole non-improvement to the §4.3 closing. |
| Fig 4 bar→point may make differences look smaller | NICE-TO-HAVE | **Conditional adopt.** Keep bars in main but overlay seed-level points (jitter) so distribution and SEM are both visible. Re-evaluate after rendering. |
| LOOCV power analysis missing | RECOMMEND | **Adopt.** Add a brief power-budget sentence to §6 Limitations explaining that bootstrap CIs on 49/53 molecule indices are how we hedge the small-n regime. |
| (e) Figure redraw plan | NICE-TO-HAVE | **Adopt as planned.** |
| (f) E71 blocking for submission | BLOCKING | **Reject** as blocking. The user explicitly forbids new experiments in this polish pass. Instead: (i) narrow the calibration claim per (a) above, (ii) keep E71 in `next_experiments_plan.md` as strongly recommended pre-submission run, (iii) make the gap visible in §6 Limitations rather than hidden. |
| SHA-256 hash / audit trail for Table 1 | BLOCKING | **Reject.** Over-engineering for an SCI methods paper. The relevant artefact is `T_transferability.json` and its content is reproducible from `src/transferability.py`; the manuscript already points to both. |
| TADF / industrial FL motivation missing | (implicit) | **Adopt.** Add one sentence to §1 about TADF-laboratory data privacy / industrial precompetitive collaboration motivating the federation setup. |

**Plan moves to Phase 1 with the adopted integrations above.**

---

*End of plan.*
