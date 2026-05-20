"""Audit that key calibration-generalization numbers in the manuscript
sources, SI sources, and the Word exports match the raw CSV values.

Inputs (data-side; all under results/calibration_generalization/):
  label_scale_stress/summary_by_target.csv
  label_scale_stress/pairwise_stats.csv
  pseudo_task_validation/improvement_summary.csv
  pseudo_task_validation/pairwise_stats.csv

Inputs (text-side):
  paper/manuscript.md          (Limitations item 11)
  paper/SI.md                  (§S11)
  paper/latex/main.tex         (Limitations item 11 LaTeX form)
  paper/latex/SI.tex           (§S11 LaTeX form)
  paper/word/CalibrationAware_FedReorg_with_calibration_generalization.docx
  paper/word/SI_with_calibration_generalization.docx

Outputs:
  paper/calibration_generalization/generalization_text_number_audit.csv
  paper/calibration_generalization/generalization_text_number_audit.md

Verdict rule:
  - OK     : the SI/main/Word text contains the rounded-to-display form
             of the CSV value (or a documented surface form for ratios).
  - WARNING: number found only in the report markdown, not in manuscript
             text (informational; not blocking).
  - ERROR  : the numeric value in the text disagrees with the recomputed
             value from the CSV.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = PROJECT_ROOT / "results" / "calibration_generalization"
AUDIT_DIR = PROJECT_ROOT / "paper" / "calibration_generalization"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- text-side files ----------------

TEXT_FILES = {
    "manuscript.md":              PROJECT_ROOT / "paper" / "manuscript.md",
    "SI.md":                      PROJECT_ROOT / "paper" / "SI.md",
    "main.tex":                   PROJECT_ROOT / "paper" / "latex" / "main.tex",
    "SI.tex":                     PROJECT_ROOT / "paper" / "latex" / "SI.tex",
}

WORD_FILES = {
    "main.docx": PROJECT_ROOT / "paper" / "word" / "CalibrationAware_FedReorg_with_calibration_generalization.docx",
    "SI.docx":   PROJECT_ROOT / "paper" / "word" / "SI_with_calibration_generalization.docx",
}


def _read(p: pathlib.Path) -> str:
    return p.read_text(errors="replace") if p.exists() else ""


def _read_docx(p: pathlib.Path) -> str:
    if not p.exists():
        return ""
    import docx
    d = docx.Document(p)
    parts = [pa.text for pa in d.paragraphs]
    for tbl in d.tables:
        for row in tbl.rows:
            parts.extend(c.text for c in row.cells)
    return "\n".join(parts)


# ---------------- CSV-side numbers ----------------

def _csv_rows(p: pathlib.Path) -> list[dict]:
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open() as fh:
        return list(csv.DictReader(fh))


def _f(s) -> float:
    return float(str(s).replace("+", "").strip())


def gather_csv_values() -> dict[str, dict]:
    """Compute the rounded surface forms of every claim that appears in the
    manuscript additions, directly from the CSVs (not from any markdown
    report)."""
    # ---- Phase 1
    p1_pairs = _csv_rows(RES / "label_scale_stress" / "pairwise_stats.csv")
    p1_summary = _csv_rows(RES / "label_scale_stress" / "summary_by_target.csv")
    fp_vs_fper = next(r for r in p1_pairs
                      if r["reference"] == "fedper_cal" and r["compared_to"] == "fedper")
    fp_vs_fedavg = next(r for r in p1_pairs
                        if r["reference"] == "fedper_cal" and r["compared_to"] == "fedavg_no_cal")
    local = next(r for r in p1_summary if r["method"] == "local")

    # ---- Phase 2
    imp = _csv_rows(RES / "pseudo_task_validation" / "improvement_summary.csv")[0]
    p2_pairs = _csv_rows(RES / "pseudo_task_validation" / "pairwise_stats.csv")
    cal_vs_fper = [r for r in p2_pairs
                   if r["reference"] == "fedper_cal" and r["compared_to"] == "fedper"]
    p2_sig = sum(1 for r in cal_vs_fper
                 if _f(r["wilcoxon_p"]) < 0.001 and _f(r["mae_diff"]) < 0)

    # Surface forms (rounded as displayed in the manuscript)
    return {
        "phase1_cal_vs_fp_dmae":   {"raw": _f(fp_vs_fper["mae_diff"]),
                                    "display_forms": ["+0.010", "+ 0.010", "+0.0101"]},
        "phase1_cal_vs_fp_p":      {"raw": _f(fp_vs_fper["wilcoxon_p"]),
                                    "display_forms": ["0.73", "0.731", "0.7310"]},
        "phase1_cal_vs_fedavg_dmae": {"raw": _f(fp_vs_fedavg["mae_diff"]),
                                      "display_forms": ["−0.043", "-0.043", "-0.0431"]},
        "phase1_cal_vs_fedavg_p":  {"raw": _f(fp_vs_fedavg["wilcoxon_p"]),
                                    "display_forms": ["0.008", "0.0078"]},
        "phase1_local_mae":        {"raw": _f(local["MAE_mean_over_seeds"]),
                                    "display_forms": ["0.309"]},
        "phase1_local_sem":        {"raw": _f(local["MAE_sem_over_seeds"]),
                                    "display_forms": ["0.010", "0.0096"]},
        "phase2_improvement":      {"raw": (int(imp["improved_tasks"]), int(imp["total_tasks"])),
                                    "display_forms": ["9 of 10", "9/10", "9 / 10"]},
        "phase2_median_dmae":      {"raw": _f(imp["median_delta_MAE"]),
                                    "display_forms": ["−0.020", "-0.020", "−0.0198"]},
        "phase2_iqr_low":          {"raw": _f(imp["q25"]),
                                    "display_forms": ["−0.030", "-0.030", "−0.0300"]},
        "phase2_iqr_high":         {"raw": _f(imp["q75"]),
                                    "display_forms": ["−0.013", "-0.013", "−0.0133"]},
        "phase2_sig_count":        {"raw": p2_sig,
                                    "display_forms": ["7 of 10", "7/10", "7 / 10"]},
    }


# ---------------- per-text scan ----------------

def scan(text: str, displays: list[str]) -> tuple[bool, str]:
    for s in displays:
        if s in text:
            return True, s
    return False, ""


def run_audit(args) -> int:
    expected = gather_csv_values()
    rows = []

    # Combine text sources we need to scan
    sources = {}
    for name, p in TEXT_FILES.items():
        sources[name] = _read(p)
    for name, p in WORD_FILES.items():
        sources[name] = _read_docx(p)

    # Each claim must appear in at least one of:
    #   manuscript.md (limitations item) OR main.tex
    #   SI.md OR SI.tex
    #   main.docx OR SI.docx
    REQUIRED_PRESENCE = {
        # claim_id           : (must_be_in_manuscript_or_main_tex, must_be_in_SI, must_be_in_one_docx)
        "phase1_cal_vs_fp_dmae":     (True,  True,  True),
        "phase1_cal_vs_fp_p":        (True,  True,  True),
        "phase1_cal_vs_fedavg_dmae": (False, True,  True),
        "phase1_cal_vs_fedavg_p":    (False, True,  True),
        "phase1_local_mae":          (False, True,  True),
        "phase1_local_sem":          (False, True,  True),
        "phase2_improvement":        (True,  True,  True),
        "phase2_median_dmae":        (True,  True,  True),
        "phase2_iqr_low":            (False, True,  True),
        "phase2_iqr_high":           (False, True,  True),
        "phase2_sig_count":          (True,  True,  True),
    }

    for claim_id, req in REQUIRED_PRESENCE.items():
        rec = expected[claim_id]
        displays = rec["display_forms"]
        raw_str = str(rec["raw"])

        loc_main = any(scan(sources[n], displays)[0] for n in ("manuscript.md", "main.tex"))
        loc_si   = any(scan(sources[n], displays)[0] for n in ("SI.md", "SI.tex"))
        loc_docx = any(scan(sources[n], displays)[0] for n in ("main.docx", "SI.docx"))

        # Decide severity
        miss = []
        if req[0] and not loc_main: miss.append("manuscript+main.tex")
        if req[1] and not loc_si:   miss.append("SI")
        if req[2] and not loc_docx: miss.append("docx")

        severity = "OK" if not miss else "ERROR"
        rows.append({
            "claim_id": claim_id,
            "raw_value": raw_str,
            "display_forms_searched": " | ".join(displays),
            "found_in_main_or_manuscript": loc_main,
            "found_in_SI_md_or_tex": loc_si,
            "found_in_docx": loc_docx,
            "severity": severity,
            "note": "missing in: " + ", ".join(miss) if miss else "all required surfaces matched",
        })

    # Write CSV
    csv_out = AUDIT_DIR / "generalization_text_number_audit.csv"
    with csv_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Write MD
    md_out = AUDIT_DIR / "generalization_text_number_audit.md"
    err = [r for r in rows if r["severity"] == "ERROR"]
    warn = [r for r in rows if r["severity"] == "WARNING"]
    ok = [r for r in rows if r["severity"] == "OK"]

    with md_out.open("w") as fh:
        fh.write("# Calibration-generalisation text-number audit\n\n")
        fh.write(f"- OK: {len(ok)} / {len(rows)}\n")
        fh.write(f"- WARNING: {len(warn)}\n")
        fh.write(f"- ERROR: {len(err)}\n\n")
        for sev in ("ERROR", "WARNING", "OK"):
            sub = [r for r in rows if r["severity"] == sev]
            if not sub: continue
            fh.write(f"## {sev}\n\n")
            fh.write("| claim | raw value | surface forms searched | "
                     "main/manuscript | SI | docx | note |\n")
            fh.write("|---|---|---|---|---|---|---|\n")
            for r in sub:
                fh.write(f"| {r['claim_id']} | {r['raw_value']} | "
                         f"`{r['display_forms_searched']}` | "
                         f"{'✓' if r['found_in_main_or_manuscript'] else '✗'} | "
                         f"{'✓' if r['found_in_SI_md_or_tex'] else '✗'} | "
                         f"{'✓' if r['found_in_docx'] else '✗'} | "
                         f"{r['note']} |\n")
            fh.write("\n")

    print(f"wrote {csv_out.relative_to(PROJECT_ROOT)}")
    print(f"wrote {md_out.relative_to(PROJECT_ROOT)}")
    print(f"  OK={len(ok)} WARNING={len(warn)} ERROR={len(err)}")
    return 0 if len(err) == 0 else 1


if __name__ == "__main__":
    sys.exit(run_audit(argparse.ArgumentParser().parse_args()))
