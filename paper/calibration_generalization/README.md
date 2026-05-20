# Calibration-generalization — paper artefacts

> **Top-level report.** `calibration_generalization_report.md` is the
> single source of truth for the calibration-generalization study's
> verdict and recommendations. Everything else in this directory either
> feeds that report (audit / preflight) or is intermediate scaffolding.

## Layout

```
paper/calibration_generalization/
├── README.md                                          # this file
├── calibration_generalization_report.md               # ← TOP-LEVEL REPORT (verdict PASS-LIMITED)
├── calibration_generalization_report.md.template      # render template (do not edit by hand)
├── repo_update_preflight.md                           # 2026-05-20 branch / token / scope check
├── repo_update_preflight_clean_worktree.md            # 2026-05-20 clean-worktree decision (release-n13 was wrong)
├── branch_scope_audit.md                              # 2026-05-20 feat/pc2-calibration-generalization scope check
├── staging_plan.md                                    # 2026-05-20 two-commit plan for this branch
├── audit_generalization_text_numbers.py               # auto-check manuscript / SI / Word numbers vs CSV
├── generalization_text_number_audit.csv               # audit result (machine-readable)
└── generalization_text_number_audit.md                # audit result (human-readable; must be 0 ERROR)
```

## Re-generating the report

```
python -m experiments.calibration_generalization.render_phase4_report
```

reads from `results/calibration_generalization/{label_scale_stress,pseudo_task_validation,calibration_variants}/`
and rewrites `calibration_generalization_report.md`. Numbers are never
hand-typed in the report.

## Re-running the text-number audit

```
python paper/calibration_generalization/audit_generalization_text_numbers.py
```

reads the same CSVs and checks every claim in:

- `paper/manuscript.md` (Limitations item 11)
- `paper/SI.md` (§S11)
- `paper/latex/main.tex` (Limitations item 11, LaTeX form)
- `paper/latex/SI.tex` (§S11, LaTeX form)
- `paper/word/CalibrationAware_FedReorg_with_calibration_generalization.docx`
- `paper/word/SI_with_calibration_generalization.docx`

Audit must report **0 ERROR**; current status: **OK = 11 / 11**.

## Recommended SI wording (for reference)

The wording committed to `paper/SI.md` §S11 and `paper/latex/SI.tex` §S11
treats Phase 1 + Phase 2 as **robustness evidence for a regime-dependent
small-target calibration interpretation**, not as a universality claim.

Key sentences kept across surfaces:

- "These experiments do not modify any main-text result on the real
  C-hole or C-triplet targets."
- "Calibration generalises in the small-target regime that mirrors
  C-triplet, but not in the moderate-target / aggressive-transform
  regime where local-only training is already adequate."
- "These controlled experiments strengthen the interpretation that
  task-specific calibration is useful in small-target heterogeneous
  settings, while the Phase 1 null result prevents a universal
  performance claim."

## Why we do not recommend changing abstract / title / conclusion

The Phase 1 null result is informative: it identifies a regime
(n_target ≈ 100, aggressive synthetic transforms) where calibration
shows no advantage over plain FedPer. Promoting this study to the
abstract or title would invite a reviewer to read the Phase 1 null as
a contradiction of the main-text C-triplet claim. The SI placement
preserves the headline claim while disclosing the boundary.

## How this relates to the main manuscript

| manuscript surface | calibration-generalisation addition |
|---|---|
| Title                              | unchanged                                       |
| Abstract                           | unchanged                                       |
| §1 Introduction                    | unchanged                                       |
| §4 Results (Figs 4 & 5; Tables 2/3/4) | unchanged                                  |
| §5 Discussion                      | unchanged                                       |
| **§6 Limitations**                 | **+ 1 item ("regime-dependent")**              |
| §7 Conclusions                     | unchanged                                       |
| **SI §S11**                        | **new section + Table S11**                    |
| `paper/word/CalibrationAware_FedReorg_with_calibration_generalization.docx` | regenerated |
| `paper/word/SI_with_calibration_generalization.docx`                        | regenerated |
| `paper/latex/CalibrationAware_FedReorg_Overleaf.zip`                        | rebuilt (old version saved as `…_pre_generalization.zip`) |
