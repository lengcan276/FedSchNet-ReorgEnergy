#!/usr/bin/env python3
"""Step 4 of the data-provenance audit.

Scans `paper/manuscript.md` and `paper/latex/main.tex` for the headline
numbers listed in the audit spec and verifies each against the
recomputed numbers / pairwise stats. Also runs estimand-mix detectors
that emit WARNING / ERROR labels when a "mean ± seed-SD" wrapper is
applied to a per-molecule median-aggregated MAE.

The script is purely textual / regex-based; it does **not** reach into
any source data beyond the existing audit/recomputed_*.csv files.

Outputs:
- paper/audit/text_number_audit.csv
- paper/audit/text_number_audit.md
"""
from __future__ import annotations

import csv
import pathlib
import re
import sys
from dataclasses import dataclass, asdict

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "paper" / "audit"
MS_PATHS = [
    REPO_ROOT / "paper" / "manuscript.md",
    REPO_ROOT / "paper" / "latex" / "main.tex",
    REPO_ROOT / "paper" / "SI.md",
    REPO_ROOT / "paper" / "latex" / "SI.tex",
]
REC_NUMBERS = AUDIT / "recomputed_numbers.csv"
REC_PAIRS = AUDIT / "recomputed_pairwise_stats.csv"


# -------------------------- load recomputed numbers --------------------------

def _load_numbers():
    with REC_NUMBERS.open() as fh:
        rows = list(csv.DictReader(fh))
    idx = {(r["experiment"], r["target"]): r for r in rows}
    return idx


def _load_pairs():
    with REC_PAIRS.open() as fh:
        rows = list(csv.DictReader(fh))
    idx = {(r["target"], r["reference"], r["compared_to"]): r for r in rows}
    return idx


# -------------------------- verdict data class --------------------------

@dataclass
class Finding:
    severity: str         # OK / WARN / ERROR
    file: str
    line: int
    snippet: str
    claim: str
    paper_value: str
    recomputed_value: str
    note: str


# -------------------------- the actual checks --------------------------

# Each check returns (snippet, paper_value_str, recomputed_value_str,
# verdict_str, note)

NUMBERS_TO_CHECK = []  # populated below


def _check_value(name, paper_val, rec_val, tol, kind="metric") -> tuple[str, str]:
    """Returns (severity, note). severity ∈ OK / WARN / ERROR."""
    try:
        pv = float(paper_val.lstrip("+"))
        rv = float(rec_val)
        d = abs(pv - rv)
        if d <= tol:
            return "OK", f"|Δ|={d:.5f} ≤ tol={tol}"
        return "ERROR", f"|Δ|={d:.5f} > tol={tol}"
    except Exception as exc:
        return "ERROR", f"parse failure: {exc}"


def _open(p: pathlib.Path) -> list[str]:
    return p.read_text(errors="replace").splitlines()


# ---------- file-specific scans ----------

# Regex patterns we look for, with the associated expected value from
# recomputed data. Each entry: (pattern, severity, claim_label, evaluator)

# helpers
def _find_lines(text: list[str], pat: str) -> list[tuple[int, str]]:
    r = re.compile(pat)
    return [(i + 1, ln) for i, ln in enumerate(text) if r.search(ln)]


def scan_file(path: pathlib.Path, numbers, pairs, findings: list[Finding]):
    if not path.exists():
        return
    text = _open(path)
    fname = str(path.relative_to(REPO_ROOT))

    # -------- 1. C-triplet E66 paired-MAE band 0.660 vs FedAvg/Prox/Per (0.76/0.78/0.76) --------
    # Recomputed E66 C-triplet per-mol median = 0.6595 → rounds to 0.660
    for ln_no, ln in _find_lines(text, r"MAE\s*[=:]?\s*\\?\s*SI?\s*\{?\s*0\.66\s*\}?\s*\\?eV"):
        if "C-triplet" in ln or "triplet" in ln:
            rec_perMolMed_E66 = numbers[("E66", "loocv_c_triplet")]["MAE_perMolMedian"]
            sev, note = _check_value("E66 C-triplet MAE 0.660", "0.660", rec_perMolMed_E66, tol=0.001)
            findings.append(Finding(sev, fname, ln_no, ln.strip()[:200],
                                    "E66 C-triplet median-aggregated MAE",
                                    "0.660", rec_perMolMed_E66, note))

    # -------- 2. E66 vs FedAvg/FedProx/FedPer triplet p-values: 0.045/0.045/0.042 --------
    for ln_no, ln in _find_lines(text, r"0\.045[,\s]*0\.045[,\s]*(?:and\s*)?0\.042|p\s*=\s*0\.045[^0-9]*0\.045[^0-9]*0\.042"):
        rec_p1 = pairs[("loocv_c_triplet", "E66", "E61")]["wilcoxon_p"]
        rec_p2 = pairs[("loocv_c_triplet", "E66", "E62")]["wilcoxon_p"]
        rec_p3 = pairs[("loocv_c_triplet", "E66", "E63")]["wilcoxon_p"]
        # tol 0.001 each
        v1 = _check_value("p61", "0.045", rec_p1, 0.001)
        v2 = _check_value("p62", "0.045", rec_p2, 0.001)
        v3 = _check_value("p63", "0.042", rec_p3, 0.001)
        sev = "OK" if all(x[0] == "OK" for x in (v1, v2, v3)) else "ERROR"
        findings.append(Finding(sev, fname, ln_no, ln.strip()[:240],
                                "E66 vs FedAvg/FedProx/FedPer triplet p",
                                "0.045/0.045/0.042",
                                f"{rec_p1}/{rec_p2}/{rec_p3}",
                                f"{v1[1]} | {v2[1]} | {v3[1]}"))

    # -------- 3. E71 vs E66 stats: p=0.929, CI (-0.044, +0.029), wins 23/26 --------
    for ln_no, ln in _find_lines(text, r"E66\)[^.]{0,40}-?\s*\$?\s*[-+]?0\.006|E71[^.]{0,80}0\.929"):
        rec = pairs[("loocv_c_triplet", "E71", "E66")]
        sev, note = _check_value("p_E71_E66", "0.929", rec["wilcoxon_p"], 0.001)
        findings.append(Finding(sev, fname, ln_no, ln.strip()[:240],
                                "E71 vs E66 triplet paired Wilcoxon p",
                                "0.929", rec["wilcoxon_p"], note))

    # -------- 4. wins 23/26 (E71 vs E66 triplet) --------
    for ln_no, ln in _find_lines(text, r"wins[^0-9]*23[^0-9]+26"):
        rec = pairs[("loocv_c_triplet", "E71", "E66")]
        ok = (rec["ref_wins"] == "23" and rec["ref_loses"] == "26")
        findings.append(Finding("OK" if ok else "ERROR", fname, ln_no, ln.strip()[:240],
                                "E71 vs E66 triplet wins/losses",
                                "23/26",
                                f"{rec['ref_wins']}/{rec['ref_loses']}",
                                "exact match" if ok else "MISMATCH"))

    # -------- 5. wins 32/17 (E71 vs E63 triplet) --------
    for ln_no, ln in _find_lines(text, r"wins[^0-9]*32[^0-9]+17"):
        rec = pairs[("loocv_c_triplet", "E71", "E63")]
        ok = (rec["ref_wins"] == "32" and rec["ref_loses"] == "17")
        findings.append(Finding("OK" if ok else "ERROR", fname, ln_no, ln.strip()[:240],
                                "E71 vs E63 triplet wins/losses",
                                "32/17",
                                f"{rec['ref_wins']}/{rec['ref_loses']}",
                                "exact match" if ok else "MISMATCH"))

    # -------- 6. wins 33/16 (E71 vs E70 triplet) --------
    for ln_no, ln in _find_lines(text, r"wins[^0-9]*33[^0-9]+16"):
        rec = pairs[("loocv_c_triplet", "E71", "E70")]
        ok = (rec["ref_wins"] == "33" and rec["ref_loses"] == "16")
        findings.append(Finding("OK" if ok else "ERROR", fname, ln_no, ln.strip()[:240],
                                "E71 vs E70 triplet wins/losses",
                                "33/16",
                                f"{rec['ref_wins']}/{rec['ref_loses']}",
                                "exact match" if ok else "MISMATCH"))

    # -------- 7. wins 31/22 (E71 vs E66 HOLE) --------
    for ln_no, ln in _find_lines(text, r"wins[^0-9]*31[^0-9]+22"):
        rec = pairs[("loocv_c_hole", "E71", "E66")]
        ok = (rec["ref_wins"] == "31" and rec["ref_loses"] == "22")
        findings.append(Finding("OK" if ok else "ERROR", fname, ln_no, ln.strip()[:240],
                                "E71 vs E66 hole wins/losses",
                                "31/22",
                                f"{rec['ref_wins']}/{rec['ref_loses']}",
                                "exact match" if ok else "MISMATCH"))

    # -------- 8. ΔMAE -0.109 for E71 vs E63 triplet --------
    for ln_no, ln in _find_lines(text, r"-\s*0\.109"):
        rec = pairs[("loocv_c_triplet", "E71", "E63")]
        sev, note = _check_value("dMAE_E71_E63", "-0.109", rec["mae_diff"], 0.001)
        findings.append(Finding(sev, fname, ln_no, ln.strip()[:240],
                                "E71 vs E63 triplet ΔMAE",
                                "-0.109", rec["mae_diff"], note))

    # -------- 9. ΔMAE -0.125 for E71 vs E70 triplet --------
    for ln_no, ln in _find_lines(text, r"-\s*0\.125"):
        rec = pairs[("loocv_c_triplet", "E71", "E70")]
        sev, note = _check_value("dMAE_E71_E70", "-0.125", rec["mae_diff"], 0.001)
        findings.append(Finding(sev, fname, ln_no, ln.strip()[:240],
                                "E71 vs E70 triplet ΔMAE",
                                "-0.125", rec["mae_diff"], note))

    # -------- 10. C-hole MAE band 0.368–0.414 --------
    for ln_no, ln in _find_lines(text, r"0\.368[^0-9]+0\.414"):
        # Recompute: min C-hole seed-mean = E69=0.3690, max = E60=0.4227.
        # 0.368 is *per-mol-median* min (E63=0.368). 0.414 is per-mol-median max (E60=0.4137).
        # Both estimands are consistent: paper band uses per-mol-median bounds.
        min_paper = 0.368
        max_paper = 0.414
        # recompute per-mol-median for E60..E70 hole
        mins = []
        maxs = []
        for exp in ("E60", "E61", "E62", "E63", "E65", "E66", "E68", "E69", "E70", "E71"):
            v = numbers[(exp, "loocv_c_hole")]["MAE_perMolMedian"]
            mins.append(float(v))
            maxs.append(float(v))
        rec_min = min(mins)
        rec_max = max(maxs)
        sev1, n1 = _check_value("hole band min", str(min_paper), f"{rec_min:.6f}", 0.001)
        sev2, n2 = _check_value("hole band max", str(max_paper), f"{rec_max:.6f}", 0.001)
        sev = "OK" if (sev1, sev2) == ("OK", "OK") else "WARN"
        findings.append(Finding(sev, fname, ln_no, ln.strip()[:240],
                                "C-hole MAE band 0.368–0.414",
                                "0.368-0.414",
                                f"{rec_min:.4f}-{rec_max:.4f}",
                                f"{n1} | {n2}"))

    # -------- ESTIMAND-MIX DETECTOR --------
    # PATTERN: a number followed by "± X" then within ~80 chars the phrase
    # "mean ± seed-SD" — verify whether the leading number is the seed-mean
    # or the per-mol-median. If the leading number rounds to 0.660 or 0.654
    # (the median figures) AND the wrapper text says "mean", flag ERROR.
    # We use a narrow, file-specific check on lines that contain BOTH
    # "0.660" / "0.654" AND "mean ± seed-SD" (or LaTeX equivalent).
    mix_re = re.compile(
        r"(0\.65[34]|0\.66[01]).{0,80}(mean\s*\\?\$?±\\?\$?\s*seed[- ]SD|mean\s*\$?\\pm\$?\s*seed[- ]SD)",
        re.IGNORECASE,
    )
    for i, ln in enumerate(text):
        m = mix_re.search(ln)
        if not m:
            continue
        ln_no = i + 1
        leading_num = m.group(1)
        # E66 5-seed mean is 0.671 (≈ 0.671). E71 5-seed mean is 0.653.
        # If leading 0.660 → that's per-mol-median for E66, mislabeled as "mean".
        if leading_num == "0.660":
            findings.append(Finding("ERROR", fname, ln_no, ln.strip()[:240],
                                    "estimand mix: '0.660 (mean ± seed-SD)' for E66 C-triplet",
                                    "0.660 ± X (claimed mean)",
                                    "5-seed mean is 0.671 (≈0.671); 0.660 is per-mol median",
                                    "MIX: numeric value is the per-mol-median MAE, but the wrapper label says 'mean ± seed-SD'."))
        elif leading_num == "0.654":
            # 0.654 = E71 per-mol median (0.6536 rounds to 0.654);
            # E71 5-seed mean = 0.6527 (rounds to 0.653).
            findings.append(Finding("ERROR", fname, ln_no, ln.strip()[:240],
                                    "estimand mix: '0.654 (mean ± seed-SD)' for E71 C-triplet",
                                    "0.654 ± X (claimed mean)",
                                    "5-seed mean is 0.653; 0.654 is per-mol median",
                                    "MIX: numeric value is the per-mol-median MAE, but the wrapper label says 'mean ± seed-SD'."))
        elif leading_num == "0.653":
            # 0.653 IS the seed-mean for E71 (0.6527 → 0.653); not a mix.
            findings.append(Finding("OK", fname, ln_no, ln.strip()[:240],
                                    "E71 0.653 ± seed-SD (correct labelling)",
                                    "0.653 ± X (claimed mean)",
                                    "5-seed mean rounds to 0.653",
                                    "Labelling is correct."))
        elif leading_num == "0.661":
            findings.append(Finding("WARN", fname, ln_no, ln.strip()[:240],
                                    "ambiguous E66 number 0.661",
                                    "0.661 ± X",
                                    "5-seed mean is 0.671",
                                    "0.661 not on file; possible typo."))

    # -------- ALSO WARN: 0.660 and 0.671 both appear without estimand disclosure --------
    # Build a list of lines containing 0.660; for each, scan the surrounding
    # ±2 lines for 0.671 *without* the word 'estimand' / 'median' / '5-seed mean' / 'Reporting'.
    lines_660 = [i for i, ln in enumerate(text) if "0.660" in ln]
    for i in lines_660:
        window = " ".join(text[max(0, i - 2):min(len(text), i + 3)])
        if "0.671" in window:
            has_estimand = any(s in window.lower() for s in
                               ("estimand", "median-aggregated", "median-across-seeds",
                                "5-seed mean", "reporting convention", "match"))
            if has_estimand:
                continue
            findings.append(Finding("WARN", fname, i + 1, text[i].strip()[:240],
                                    "co-occurring 0.660 and 0.671 without explicit estimand label",
                                    "0.660 and 0.671 both appear nearby",
                                    "",
                                    "WARNING: 0.660 (per-mol median) and 0.671 (5-seed mean) co-occur without explicit disclosure of which estimand each is."))


def main() -> int:
    numbers = _load_numbers()
    pairs = _load_pairs()
    findings: list[Finding] = []

    for path in MS_PATHS:
        scan_file(path, numbers, pairs, findings)

    out_csv = AUDIT / "text_number_audit.csv"
    out_md = AUDIT / "text_number_audit.md"

    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(asdict(findings[0]).keys()) if findings else
                                            ["severity", "file", "line", "snippet", "claim",
                                             "paper_value", "recomputed_value", "note"])
        w.writeheader()
        for f in findings:
            w.writerow(asdict(f))

    sev_counts = {"OK": 0, "WARN": 0, "ERROR": 0}
    for f in findings:
        sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1

    md = ["# Text number audit",
          "",
          f"Total checks: **{len(findings)}**  ",
          f"- OK: **{sev_counts.get('OK', 0)}**",
          f"- WARN: **{sev_counts.get('WARN', 0)}**",
          f"- ERROR: **{sev_counts.get('ERROR', 0)}**",
          ""]

    for sev in ("ERROR", "WARN", "OK"):
        rows = [f for f in findings if f.severity == sev]
        if not rows:
            continue
        md.append(f"## {sev} ({len(rows)})")
        md.append("")
        md.append("| file | line | claim | paper | recomputed | note |")
        md.append("|---|---|---|---|---|---|")
        for f in rows:
            md.append(f"| {f.file} | {f.line} | {f.claim} | {f.paper_value} | {f.recomputed_value} | {f.note} |")
        md.append("")
    out_md.write_text("\n".join(md))

    print(f"wrote {out_csv.relative_to(REPO_ROOT)} ({len(findings)} rows)")
    print(f"  OK: {sev_counts.get('OK', 0)}  WARN: {sev_counts.get('WARN', 0)}  ERROR: {sev_counts.get('ERROR', 0)}")
    return 0 if sev_counts.get("ERROR", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
