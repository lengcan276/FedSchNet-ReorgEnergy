#!/usr/bin/env python3
"""Step 3 of the data-provenance audit.

Compares paper/tables/table{2,3,4}*.csv against the recomputed numbers
written by `recompute_all_reported_numbers.py`.

Tolerances (per the audit spec):
- MAE / RMSE / R²:   absolute tol 0.001
- p-value:            absolute tol 0.001
- CI bounds:          absolute tol 0.002
- wins / losses / ties: must match exactly
- mae_diff:           absolute tol 0.001

Outputs:
- paper/audit/table_consistency_report.csv  (one row per checked claim)
- paper/audit/table_consistency_report.md   (human-readable summary)
"""
from __future__ import annotations

import csv
import pathlib
import sys
from typing import Dict, List, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "paper" / "audit"
TABLES = REPO_ROOT / "paper" / "tables"

REC_NUMBERS = AUDIT / "recomputed_numbers.csv"
REC_PAIRS = AUDIT / "recomputed_pairwise_stats.csv"

TOL_METRIC = 1e-3
TOL_P = 1e-3
TOL_CI = 2e-3
TOL_DIFF = 1e-3

# Map paper table "method" -> experiment code
T2_METHOD_TO_EXP = {
    "E60 Local-only": "E60",
    "E61 FedAvg": "E61",
    "E62 FedProx": "E62",
    "E63 FedPer": "E63",
    "E66 PC²-FedReorg": "E66",
    "E71 FedPer + calibration only": "E71",
}

# Map paper table target strings -> recomputed target strings
T2_TGT_MAP = {
    "C-hole": "loocv_c_hole",
    "C-triplet": "loocv_c_triplet",
}

T4_EXP_MAP = {
    "E66 PC²-FedReorg (reference)": "E66",
    "E65 D→C supervised pretrain": "E65",
    "E68 PC² − T_repr C-cross removed": "E68",
    "E69 PC² − uniform FedAvg gate": "E69",
    "E70 PC² − no calibration": "E70",
    "E71 FedPer + calibration only (positive control)": "E71",
}


def _read_csv(p: pathlib.Path) -> List[Dict[str, str]]:
    with p.open() as fh:
        return list(csv.DictReader(fh))


def _index_recomputed_numbers(rows) -> Dict[Tuple[str, str], Dict[str, str]]:
    idx = {}
    for r in rows:
        idx[(r["experiment"], r["target"])] = r
    return idx


def _index_recomputed_pairs(rows) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    idx = {}
    for r in rows:
        idx[(r["target"], r["reference"], r["compared_to"])] = r
    return idx


def _fdiff(a: str | float, b: str | float) -> float:
    try:
        return abs(float(a) - float(b))
    except (TypeError, ValueError):
        return float("nan")


def _verdict(diff: float, tol: float) -> str:
    if diff != diff:  # NaN
        return "NA"
    return "PASS" if diff <= tol else "FAIL"


# ---------------------------- Table 2 (main perf) ----------------------------

def check_table2(numbers_idx, t2_rows) -> List[Dict]:
    out = []
    for r in t2_rows:
        exp = T2_METHOD_TO_EXP.get(r["method"])
        if exp is None:
            continue  # A_5fold / B_5fold rows are not in our recompute scope
        tgt = T2_TGT_MAP.get(r["target"])
        if tgt is None:
            continue
        key = (exp, tgt)
        if key not in numbers_idx:
            out.append({
                "table": "table2",
                "claim": f"{r['method']} / {r['target']}",
                "field": "MAE",
                "paper": r["MAE_mean"],
                "recomputed": "MISSING",
                "abs_diff": "nan",
                "tol": f"{TOL_METRIC}",
                "verdict": "FAIL",
                "note": "no matching recomputed row",
            })
            continue
        rec = numbers_idx[key]
        # MAE
        d = _fdiff(r["MAE_mean"], rec["MAE_seed_mean"])
        out.append({"table": "table2", "claim": f"{r['method']} / {r['target']}",
                    "field": "MAE_mean", "paper": r["MAE_mean"],
                    "recomputed": rec["MAE_seed_mean"], "abs_diff": f"{d:.6f}",
                    "tol": f"{TOL_METRIC}", "verdict": _verdict(d, TOL_METRIC),
                    "note": "5-seed mean of per-seed MAE"})
        # SEM
        d = _fdiff(r["MAE_sem"], rec["MAE_seed_sem"])
        out.append({"table": "table2", "claim": f"{r['method']} / {r['target']}",
                    "field": "MAE_sem", "paper": r["MAE_sem"],
                    "recomputed": rec["MAE_seed_sem"], "abs_diff": f"{d:.6f}",
                    "tol": f"{TOL_METRIC}", "verdict": _verdict(d, TOL_METRIC),
                    "note": ""})
        # RMSE
        d = _fdiff(r["RMSE_mean"], rec["RMSE_seed_mean"])
        out.append({"table": "table2", "claim": f"{r['method']} / {r['target']}",
                    "field": "RMSE_mean", "paper": r["RMSE_mean"],
                    "recomputed": rec["RMSE_seed_mean"], "abs_diff": f"{d:.6f}",
                    "tol": f"{TOL_METRIC}", "verdict": _verdict(d, TOL_METRIC),
                    "note": ""})
        d = _fdiff(r["RMSE_sem"], rec["RMSE_seed_sem"])
        out.append({"table": "table2", "claim": f"{r['method']} / {r['target']}",
                    "field": "RMSE_sem", "paper": r["RMSE_sem"],
                    "recomputed": rec["RMSE_seed_sem"], "abs_diff": f"{d:.6f}",
                    "tol": f"{TOL_METRIC}", "verdict": _verdict(d, TOL_METRIC),
                    "note": ""})
        # R2 (paper has +0.xxx / -0.xxx)
        paper_r2 = r["R2_mean"].lstrip("+")
        d = _fdiff(paper_r2, rec["R2_seed_mean"])
        out.append({"table": "table2", "claim": f"{r['method']} / {r['target']}",
                    "field": "R2_mean", "paper": r["R2_mean"],
                    "recomputed": rec["R2_seed_mean"], "abs_diff": f"{d:.6f}",
                    "tol": f"{TOL_METRIC}", "verdict": _verdict(d, TOL_METRIC),
                    "note": ""})
    return out


# ---------------------------- Table 3 (paired tests) ----------------------------

def check_table3(pairs_idx, t3_rows) -> List[Dict]:
    out = []
    for r in t3_rows:
        key = (r["target"], r["reference"], r["compared_to"])
        if key not in pairs_idx:
            out.append({"table": "table3", "claim": f"{r['reference']} vs {r['compared_to']} / {r['target']}",
                        "field": "ALL", "paper": "n/a", "recomputed": "MISSING",
                        "abs_diff": "nan", "tol": "n/a", "verdict": "FAIL",
                        "note": "no matching recomputed paired row"})
            continue
        rec = pairs_idx[key]

        def _add(field: str, paper_val: str, rec_val: str, tol: float, note: str = "") -> None:
            paper_clean = str(paper_val).lstrip("+")
            d = _fdiff(paper_clean, rec_val)
            out.append({"table": "table3", "claim": f"{r['reference']} vs {r['compared_to']} / {r['target']}",
                        "field": field, "paper": paper_val, "recomputed": rec_val,
                        "abs_diff": f"{d:.6f}", "tol": f"{tol}",
                        "verdict": _verdict(d, tol), "note": note})

        _add("mae_ref",    r["mae_ref"],    rec["mae_ref_perMolMedian"],   TOL_METRIC,
             "per-mol median-across-seeds MAE")
        _add("mae_other",  r["mae_other"],  rec["mae_other_perMolMedian"], TOL_METRIC,
             "per-mol median-across-seeds MAE")
        _add("mae_diff",   r["mae_diff"],   rec["mae_diff"],               TOL_DIFF, "")
        _add("ci_low",     r["mae_diff_ci_low"],  rec["ci_low"],           TOL_CI,
             "bootstrap 5000 resamples; seed may differ")
        _add("ci_high",    r["mae_diff_ci_high"], rec["ci_high"],          TOL_CI, "")
        # wins/losses must match exactly
        wins_match = (int(r["ref_wins"]) == int(rec["ref_wins"]))
        out.append({"table": "table3", "claim": f"{r['reference']} vs {r['compared_to']} / {r['target']}",
                    "field": "ref_wins", "paper": r["ref_wins"], "recomputed": rec["ref_wins"],
                    "abs_diff": "0" if wins_match else f"{abs(int(r['ref_wins'])-int(rec['ref_wins']))}",
                    "tol": "exact", "verdict": "PASS" if wins_match else "FAIL", "note": ""})
        losses_match = (int(r["ref_loses"]) == int(rec["ref_loses"]))
        out.append({"table": "table3", "claim": f"{r['reference']} vs {r['compared_to']} / {r['target']}",
                    "field": "ref_loses", "paper": r["ref_loses"], "recomputed": rec["ref_loses"],
                    "abs_diff": "0" if losses_match else f"{abs(int(r['ref_loses'])-int(rec['ref_loses']))}",
                    "tol": "exact", "verdict": "PASS" if losses_match else "FAIL", "note": ""})
        _add("wilcoxon_p", r["wilcoxon_p"], rec["wilcoxon_p"], TOL_P, "")
    return out


# ---------------------------- Table 4 (ablation) ----------------------------

def check_table4(numbers_idx, t4_rows) -> List[Dict]:
    out = []
    for r in t4_rows:
        exp = T4_EXP_MAP.get(r["experiment"])
        if exp is None:
            continue
        tgt = T2_TGT_MAP.get(r["target"])
        if tgt is None:
            continue
        key = (exp, tgt)
        if key not in numbers_idx:
            out.append({"table": "table4", "claim": f"{r['experiment']} / {r['target']}",
                        "field": "ALL", "paper": "n/a", "recomputed": "MISSING",
                        "abs_diff": "nan", "tol": "n/a", "verdict": "FAIL",
                        "note": "no matching recomputed row"})
            continue
        rec = numbers_idx[key]
        d = _fdiff(r["MAE_mean"], rec["MAE_seed_mean"])
        out.append({"table": "table4", "claim": f"{r['experiment']} / {r['target']}",
                    "field": "MAE_mean", "paper": r["MAE_mean"], "recomputed": rec["MAE_seed_mean"],
                    "abs_diff": f"{d:.6f}", "tol": f"{TOL_METRIC}", "verdict": _verdict(d, TOL_METRIC),
                    "note": "5-seed mean"})
        d = _fdiff(r["MAE_sem"], rec["MAE_seed_sem"])
        out.append({"table": "table4", "claim": f"{r['experiment']} / {r['target']}",
                    "field": "MAE_sem", "paper": r["MAE_sem"], "recomputed": rec["MAE_seed_sem"],
                    "abs_diff": f"{d:.6f}", "tol": f"{TOL_METRIC}", "verdict": _verdict(d, TOL_METRIC),
                    "note": ""})
    return out


# ---------------------------- main ----------------------------

def main() -> int:
    rec_numbers = _read_csv(REC_NUMBERS)
    rec_pairs = _read_csv(REC_PAIRS)
    numbers_idx = _index_recomputed_numbers(rec_numbers)
    pairs_idx = _index_recomputed_pairs(rec_pairs)

    t2 = _read_csv(TABLES / "table2_main_performance.csv")
    t3 = _read_csv(TABLES / "table3_paired_tests.csv")
    t4 = _read_csv(TABLES / "table4_ablation.csv")

    rows: List[Dict] = []
    rows += check_table2(numbers_idx, t2)
    rows += check_table3(pairs_idx, t3)
    rows += check_table4(numbers_idx, t4)

    out_csv = AUDIT / "table_consistency_report.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["table", "claim", "field", "paper",
                                           "recomputed", "abs_diff", "tol",
                                           "verdict", "note"])
        w.writeheader()
        w.writerows(rows)

    # Markdown summary
    n_total = len(rows)
    n_pass = sum(1 for r in rows if r["verdict"] == "PASS")
    n_fail = sum(1 for r in rows if r["verdict"] == "FAIL")
    n_na = sum(1 for r in rows if r["verdict"] == "NA")
    fails = [r for r in rows if r["verdict"] == "FAIL"]

    md = ["# Table consistency report",
          "",
          f"- Total checks: **{n_total}**",
          f"- PASS: **{n_pass}**",
          f"- FAIL: **{n_fail}**",
          f"- NA: **{n_na}**",
          "",
          "Tolerances: MAE/RMSE/R² ≤ 0.001 abs; mae_diff ≤ 0.001; CI bounds ≤ 0.002; p ≤ 0.001; wins/losses exact.",
          "",
          "## FAIL rows",
          ""]
    if not fails:
        md.append("_(none)_")
    else:
        md.append("| table | claim | field | paper | recomputed | abs_diff | tol | note |")
        md.append("|---|---|---|---|---|---|---|---|")
        for r in fails:
            md.append(f"| {r['table']} | {r['claim']} | {r['field']} | {r['paper']} | "
                      f"{r['recomputed']} | {r['abs_diff']} | {r['tol']} | {r['note']} |")
    (AUDIT / "table_consistency_report.md").write_text("\n".join(md) + "\n")

    print(f"wrote {out_csv.relative_to(REPO_ROOT)} ({n_total} rows)")
    print(f"  PASS: {n_pass}  FAIL: {n_fail}  NA: {n_na}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
