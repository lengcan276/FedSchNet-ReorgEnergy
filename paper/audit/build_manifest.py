#!/usr/bin/env python3
"""Step 1 of the data-provenance audit.

Walks the explicit list of files the audit cares about, computes sha256
hashes, captures size + mtime, and writes paper/audit/result_file_manifest.csv.
No file in `src/`, `experiments/`, `results/`, `paper/manuscript.md`,
`paper/latex/main.tex`, `paper/SI.md`, or `paper/figures/` is modified --
the script only reads.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "paper" / "audit" / "result_file_manifest.csv"

AUDIT_FILES: list[tuple[str, str]] = [
    # role, relative path (POSIX)
    ("raw_prediction",     "results/pc2_batch1_multiseed_predictions.json"),
    ("summary_table",      "results/pc2_batch1_multiseed_summary.csv"),
    ("raw_prediction",     "results/pc2_phase5_ablation_predictions.json"),
    ("summary_table",      "results/pc2_phase5_ablation_summary.csv"),
    ("paired_stats",       "results/pc2_phase5_stats.csv"),
    ("paired_stats_md",    "results/pc2_phase5_stats.md"),
    ("raw_prediction",     "results/e71_predictions.json"),
    ("summary_table",      "results/e71_summary.csv"),
    ("paired_stats",       "results/e71_vs_e66_stats.csv"),
    ("paired_stats_md",    "results/e71_vs_e66_stats.md"),
    ("paired_stats_md",    "results/e71_vs_e63_stats.md"),
    ("paired_stats_md",    "results/e71_vs_e70_stats.md"),
    ("paper_table",        "paper/tables/table2_main_performance.csv"),
    ("paper_table_md",     "paper/tables/table2_main_performance.md"),
    ("paper_table",        "paper/tables/table3_paired_tests.csv"),
    ("paper_table_md",     "paper/tables/table3_paired_tests.md"),
    ("paper_table",        "paper/tables/table4_ablation.csv"),
    ("paper_table_md",     "paper/tables/table4_ablation.md"),
    ("manuscript_source",  "paper/manuscript.md"),
    ("manuscript_source",  "paper/SI.md"),
    ("manuscript_source",  "paper/latex/main.tex"),
    ("manuscript_source",  "paper/latex/SI.tex"),
    ("figure_script",      "paper/make_figures.py"),
]


def _sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    rows: list[dict[str, str | int]] = []
    missing: list[str] = []
    for role, rel in AUDIT_FILES:
        p = REPO_ROOT / rel
        if not p.exists():
            missing.append(rel)
            rows.append({
                "file_path": rel,
                "role": role,
                "size_bytes": "MISSING",
                "modified_utc": "MISSING",
                "sha256": "MISSING",
            })
            continue
        st = p.stat()
        rows.append({
            "file_path": rel,
            "role": role,
            "size_bytes": st.st_size,
            "modified_utc": _dt.datetime.utcfromtimestamp(st.st_mtime)
                                       .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "sha256": _sha256(p),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_path", "role", "size_bytes",
                                           "modified_utc", "sha256"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)} ({len(rows)} rows)")
    if missing:
        print(f"WARNING: {len(missing)} missing file(s):", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
