"""Phase 1 — label-scale stress test driver.

Builds a 5-client pseudo-federation from `data/client_a_b/public_reorg_energy_15210.csv`
with synthetic affine + noise label transforms per source client, then evaluates
how well each candidate method generalises to the **target** client's native
label scale under K-fold CV.

Usage
-----
Smoke (1 seed, n_target=50, n_source=80, 2 rounds, 1 local epoch):
    python -m experiments.calibration_generalization.run_label_scale_stress --smoke

Full (5 seeds, n_target=100, n_source=200, 12 rounds, 5 local epochs):
    python -m experiments.calibration_generalization.run_label_scale_stress --full

Outputs land in results/calibration_generalization/label_scale_stress/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time
from datetime import datetime

import numpy as np
import torch

# Make sure project root is importable when invoked as a script.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.calibration_generalization.pseudo_federation import (
    build_label_scale_federation,
    label_scale_ratio,
)
from experiments.calibration_generalization.fed_train_minimal import (
    run_kfold_on_target,
)

OUT_DIR = PROJECT_ROOT / "results" / "calibration_generalization" / "label_scale_stress"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_config(args) -> dict:
    return {
        "experiment_id": "label_scale_stress",
        "phase": "phase1",
        "data_source": "data/client_a_b/public_reorg_energy_15210.csv",
        "n_target": args.n_target,
        "n_source": args.n_source,
        "n_source_clients": 4,
        "n_folds": args.n_folds,
        "seeds": args.seeds,
        "n_rounds": args.n_rounds,
        "n_local_epochs": args.n_local_epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "methods": list(args.methods),
        "kan_grid": 5,
        "encoder_out_dim": 256,
        "node_feat_dim": 11,
        "head_type": "kan",
        "device": str(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')),
        "bootstrap_seed_for_stats": 20260520,  # used by compute_pairwise_stats.py
        "smoke": args.smoke,
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="tiny smoke run (1 seed, small clients, 2 rounds)")
    p.add_argument("--full", action="store_true",
                   help="full Phase-1 run (5 seeds, n_target=100, n_source=200, 12 rounds)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="explicit seed list (overrides --smoke/--full presets)")
    p.add_argument("--n-target", type=int, default=None, dest="n_target")
    p.add_argument("--n-source", type=int, default=None, dest="n_source")
    p.add_argument("--n-folds", type=int, default=5, dest="n_folds")
    p.add_argument("--n-rounds", type=int, default=None, dest="n_rounds")
    p.add_argument("--n-local-epochs", type=int, default=None, dest="n_local_epochs")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--methods", nargs="+",
                   default=["local", "fedavg_no_cal", "fedper", "fedper_cal"])
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args(argv)

    # Resolve presets.
    if args.smoke and args.full:
        p.error("--smoke and --full are mutually exclusive")
    if args.smoke:
        args.seeds = args.seeds or [42]
        args.n_target = args.n_target or 50
        args.n_source = args.n_source or 80
        args.n_rounds = args.n_rounds or 2
        args.n_local_epochs = args.n_local_epochs or 1
        args.n_folds = 3
    else:
        args.seeds = args.seeds or [42, 123, 456, 789, 1000]
        args.n_target = args.n_target or 100
        args.n_source = args.n_source or 200
        args.n_rounds = args.n_rounds or 12
        args.n_local_epochs = args.n_local_epochs or 5

    device = torch.device(args.device) if args.device else (
        torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    config = _build_config(args)
    print(f"\n[Phase 1] {('SMOKE' if args.smoke else 'FULL')} run; "
          f"seeds={args.seeds}, methods={args.methods}, device={device}")
    print(f"n_target={args.n_target}, n_source={args.n_source}, "
          f"n_folds={args.n_folds}, n_rounds={args.n_rounds}, "
          f"n_local_epochs={args.n_local_epochs}")

    # Persist config.
    cfg_path = OUT_DIR / "config_label_scale_stress.json"
    with cfg_path.open("w") as fh:
        json.dump(config, fh, indent=2)
    print(f"wrote {cfg_path.relative_to(PROJECT_ROOT)}")

    # ---------- run ----------
    all_rows: list[dict] = []
    runtime_rows: list[dict] = []
    per_target_summary: dict = {}

    for seed in args.seeds:
        print(f"\n----- seed = {seed} -----")
        t_build_start = time.time()
        clients = build_label_scale_federation(
            n_target=args.n_target,
            n_source=args.n_source,
            n_source_clients=4,
            seed=seed,
        )
        t_build = time.time() - t_build_start
        print(f"built federation ({t_build:.1f} s); clients: "
              + ", ".join(f"{c.name}(n={len(c)})" for c in clients))

        # Record label-scale ratios (for the per-target summary).
        ratios = {c.name: label_scale_ratio(clients[0], c)
                  for c in clients[1:]}

        target_name = clients[0].name
        for method in args.methods:
            t_m = time.time()
            res = run_kfold_on_target(
                clients=clients,
                target_idx=0,
                method=method,
                n_folds=args.n_folds,
                n_rounds=args.n_rounds,
                n_local_epochs=args.n_local_epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                seed=seed,
                device=device,
            )
            elapsed = time.time() - t_m
            print(f"  [{method:14s}] MAE={res['mae_mean']:.4f}±{res['mae_std']:.4f} "
                  f"RMSE={res['rmse_mean']:.4f}±{res['rmse_std']:.4f} "
                  f"R²={res['r2_mean']:.3f}±{res['r2_std']:.3f}  ({elapsed:.0f}s)")

            # Per-fold rows
            for fold_d in res["per_fold"]:
                all_rows.append({
                    "experiment_id": "label_scale_stress",
                    "phase": "phase1",
                    "seed": seed,
                    "pseudo_task_id": target_name,
                    "method": method,
                    "target": target_name,
                    "fold": fold_d["fold"],
                    "n_train": args.n_target - fold_d["n_test"],
                    "n_test": fold_d["n_test"],
                    "label_transform": "identity",
                    "MAE": fold_d["mae"],
                    "RMSE": fold_d["rmse"],
                    "R2": fold_d["r2"],
                    "elapsed_sec": fold_d["elapsed_sec"],
                })

            # Save per-seed/per-method y_pred for later paired-stats
            preds_dir = OUT_DIR / "predictions"
            preds_dir.mkdir(exist_ok=True)
            with (preds_dir / f"{method}_seed{seed}.json").open("w") as fh:
                json.dump({
                    "method": method,
                    "seed": seed,
                    "target": target_name,
                    "y_true": res["y_true_all"],
                    "y_pred": res["y_pred_all"],
                    "mae_mean": res["mae_mean"],
                    "rmse_mean": res["rmse_mean"],
                    "r2_mean": res["r2_mean"],
                }, fh)

            runtime_rows.append({
                "seed": seed,
                "method": method,
                "elapsed_sec_total": elapsed,
                "elapsed_sec_per_fold_mean": elapsed / args.n_folds,
            })

        # Record scale ratios (one row per source × seed).
        for source_name, ratio in ratios.items():
            per_target_summary.setdefault(seed, {})[source_name] = ratio

    # ---------- summary tables ----------
    # summary_by_seed.csv: per (seed × method) aggregated metrics
    by_seed_rows = []
    for seed in args.seeds:
        for method in args.methods:
            sub = [r for r in all_rows if r["seed"] == seed and r["method"] == method]
            mae_vals = np.array([r["MAE"] for r in sub])
            rmse_vals = np.array([r["RMSE"] for r in sub])
            r2_vals = np.array([r["R2"] for r in sub])
            by_seed_rows.append({
                "seed": seed,
                "method": method,
                "MAE_mean_over_folds": float(mae_vals.mean()),
                "MAE_std_over_folds": float(mae_vals.std(ddof=1)) if len(mae_vals) > 1 else float("nan"),
                "RMSE_mean_over_folds": float(rmse_vals.mean()),
                "R2_mean_over_folds": float(r2_vals.mean()),
                "n_folds": len(sub),
            })

    # summary_by_target.csv: per (method) aggregated across seeds
    methods = list(args.methods)
    by_target_rows = []
    for method in methods:
        sub = [r for r in by_seed_rows if r["method"] == method]
        mae_vals = np.array([r["MAE_mean_over_folds"] for r in sub])
        rmse_vals = np.array([r["RMSE_mean_over_folds"] for r in sub])
        r2_vals = np.array([r["R2_mean_over_folds"] for r in sub])
        by_target_rows.append({
            "method": method,
            "MAE_mean_over_seeds": float(mae_vals.mean()),
            "MAE_sem_over_seeds": float(mae_vals.std(ddof=1) / np.sqrt(len(mae_vals))) if len(mae_vals) > 1 else float("nan"),
            "MAE_std_over_seeds": float(mae_vals.std(ddof=1)) if len(mae_vals) > 1 else float("nan"),
            "RMSE_mean_over_seeds": float(rmse_vals.mean()),
            "R2_mean_over_seeds": float(r2_vals.mean()),
            "n_seeds": len(sub),
        })

    # calibration_effect_by_scale_ratio.csv
    scale_ratio_rows = []
    for seed in args.seeds:
        for source_name, ratio in per_target_summary.get(seed, {}).items():
            scale_ratio_rows.append({
                "seed": seed,
                "source_client": source_name,
                "label_scale_ratio_vs_target": ratio,
            })

    # Write all CSVs
    def _write_csv(name, rows):
        path = OUT_DIR / name
        if not rows:
            return
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path.relative_to(PROJECT_ROOT)} ({len(rows)} rows)")

    _write_csv("per_fold_results.csv", all_rows)
    _write_csv("summary_by_seed.csv", by_seed_rows)
    _write_csv("summary_by_target.csv", by_target_rows)
    _write_csv("calibration_effect_by_scale_ratio.csv", scale_ratio_rows)
    _write_csv("runtime_log.csv", runtime_rows)

    # Write a markdown smoke log
    log_path = OUT_DIR / ("smoke_log.md" if args.smoke else "full_run_log.md")
    with log_path.open("w") as fh:
        fh.write(f"# Phase 1 — {'smoke' if args.smoke else 'full'} run log\n\n")
        fh.write(f"Started {config['timestamp_utc']}\n\n")
        fh.write(f"Config:\n```\n{json.dumps(config, indent=2)}\n```\n\n")
        fh.write(f"## Summary by method (averaged across {len(args.seeds)} seeds)\n\n")
        fh.write("| method | MAE (mean ± SEM over seeds) | RMSE | R² |\n")
        fh.write("|---|---|---|---|\n")
        for r in by_target_rows:
            sem_str = f"± {r['MAE_sem_over_seeds']:.4f}" if not np.isnan(r["MAE_sem_over_seeds"]) else "(n=1)"
            fh.write(f"| {r['method']} | {r['MAE_mean_over_seeds']:.4f} {sem_str} | "
                     f"{r['RMSE_mean_over_seeds']:.4f} | {r['R2_mean_over_seeds']:.3f} |\n")
        fh.write("\n## Runtime\n\n")
        total_elapsed = sum(r["elapsed_sec_total"] for r in runtime_rows)
        fh.write(f"Total: {total_elapsed:.1f} s = {total_elapsed/60:.2f} min\n")
    print(f"wrote {log_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
