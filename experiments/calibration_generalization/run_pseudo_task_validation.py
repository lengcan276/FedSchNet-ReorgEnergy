"""Phase 2 — pseudo-task multi-target validation driver.

For each of N pseudo-targets (label-quantile bins of QM9), train a small
federation with that target as the small private client and the other N-1
pseudo-clients as sources. Compare {Local, FedAvg-no-cal, FedPer,
FedPer+calibration} on each target.

The pseudo-clients here all use the identity label transform; the
heterogeneity is in label-quantile slice (mean shift across clients), not in
artificial scale ratio. This isolates the question of whether calibration
helps when sources have different label *means* but the same label
*regression task*.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.calibration_generalization.pseudo_federation import (
    build_pseudo_task_set,
)
from experiments.calibration_generalization.fed_train_minimal import (
    run_kfold_on_target,
)

OUT_DIR = PROJECT_ROOT / "results" / "calibration_generalization" / "pseudo_task_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_config(args) -> dict:
    return {
        "experiment_id": "pseudo_task_validation",
        "phase": "phase2",
        "data_source": "data/client_a_b/public_reorg_energy_15210.csv",
        "n_per_task": args.n_per_task,
        "n_tasks": args.n_tasks,
        "n_folds": args.n_folds,
        "seeds": args.seeds,
        "n_rounds": args.n_rounds,
        "n_local_epochs": args.n_local_epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "methods": list(args.methods),
        "bin_strategy": "label_quantile",
        "device": str(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')),
        "bootstrap_seed_for_stats": 20260520,
        "smoke": args.smoke,
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--n-per-task", type=int, default=None, dest="n_per_task")
    p.add_argument("--n-tasks", type=int, default=None, dest="n_tasks")
    p.add_argument("--n-folds", type=int, default=None, dest="n_folds")
    p.add_argument("--n-rounds", type=int, default=None, dest="n_rounds")
    p.add_argument("--n-local-epochs", type=int, default=None, dest="n_local_epochs")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--methods", nargs="+",
                   default=["local", "fedavg_no_cal", "fedper", "fedper_cal"])
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args(argv)

    if args.smoke and args.full:
        p.error("--smoke and --full are mutually exclusive")
    if args.smoke:
        args.seeds = args.seeds or [42]
        args.n_per_task = args.n_per_task or 30
        args.n_tasks = args.n_tasks or 4
        args.n_folds = args.n_folds or 3
        args.n_rounds = args.n_rounds or 2
        args.n_local_epochs = args.n_local_epochs or 1
    else:
        args.seeds = args.seeds or [42, 123, 456]   # 3 seeds for Phase 2 (cost balance)
        args.n_per_task = args.n_per_task or 50
        args.n_tasks = args.n_tasks or 10
        args.n_folds = args.n_folds or 3
        args.n_rounds = args.n_rounds or 8
        args.n_local_epochs = args.n_local_epochs or 3

    device = torch.device(args.device) if args.device else (
        torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    config = _build_config(args)
    print(f"\n[Phase 2] {('SMOKE' if args.smoke else 'FULL')} run; "
          f"n_tasks={args.n_tasks}, n_per_task={args.n_per_task}, "
          f"seeds={args.seeds}, methods={args.methods}, device={device}")

    cfg_path = OUT_DIR / "config_pseudo_task_validation.json"
    with cfg_path.open("w") as fh:
        json.dump(config, fh, indent=2)
    print(f"wrote {cfg_path.relative_to(PROJECT_ROOT)}")

    # ---------- definitions ----------
    # Per (seed, task) record the label mean / std (for the report).
    task_defs: list[dict] = []
    fold_rows: list[dict] = []
    runtime_rows: list[dict] = []

    for seed in args.seeds:
        print(f"\n----- seed = {seed} -----")
        clients = build_pseudo_task_set(
            n_per_task=args.n_per_task, n_tasks=args.n_tasks, seed=seed,
        )
        # Record task definitions for this seed (label stats).
        for i, c in enumerate(clients):
            task_defs.append({
                "seed": seed,
                "task_id": c.name,
                "n_per_task": len(c),
                "label_mean_eV": float(np.mean(c.transformed_y)),
                "label_std_eV": float(np.std(c.transformed_y)),
                "label_min_eV": float(np.min(c.transformed_y)),
                "label_max_eV": float(np.max(c.transformed_y)),
            })

        # For each pseudo target, run all four methods.
        for tgt_idx, tgt in enumerate(clients):
            print(f"\n  target {tgt.name} (label_mean={np.mean(tgt.transformed_y):.3f})")
            for method in args.methods:
                t0 = time.time()
                res = run_kfold_on_target(
                    clients=clients,
                    target_idx=tgt_idx,
                    method=method,
                    n_folds=args.n_folds,
                    n_rounds=args.n_rounds,
                    n_local_epochs=args.n_local_epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    seed=seed,
                    device=device,
                )
                elapsed = time.time() - t0
                print(f"    [{method:14s}] MAE={res['mae_mean']:.4f} "
                      f"R²={res['r2_mean']:.3f}  ({elapsed:.0f}s)")

                for fold_d in res["per_fold"]:
                    fold_rows.append({
                        "experiment_id": "pseudo_task_validation",
                        "phase": "phase2",
                        "seed": seed,
                        "pseudo_task_id": tgt.name,
                        "method": method,
                        "target": tgt.name,
                        "fold": fold_d["fold"],
                        "n_train": args.n_per_task - fold_d["n_test"],
                        "n_test": fold_d["n_test"],
                        "label_transform": "identity",
                        "MAE": fold_d["mae"],
                        "RMSE": fold_d["rmse"],
                        "R2": fold_d["r2"],
                        "elapsed_sec": fold_d["elapsed_sec"],
                    })

                # save per-target / per-seed / per-method predictions
                preds_dir = OUT_DIR / "predictions"
                preds_dir.mkdir(exist_ok=True)
                with (preds_dir / f"{tgt.name}_{method}_seed{seed}.json").open("w") as fh:
                    json.dump({
                        "task_id": tgt.name,
                        "method": method,
                        "seed": seed,
                        "y_true": res["y_true_all"],
                        "y_pred": res["y_pred_all"],
                        "mae_mean": res["mae_mean"],
                        "rmse_mean": res["rmse_mean"],
                        "r2_mean": res["r2_mean"],
                    }, fh)

                runtime_rows.append({
                    "seed": seed, "task_id": tgt.name, "method": method,
                    "elapsed_sec_total": elapsed,
                })

    # ---------- aggregated summaries ----------
    methods = list(args.methods)
    summary_by_seed = []
    for seed in args.seeds:
        for task in {r["target"] for r in fold_rows if r["seed"] == seed}:
            for method in methods:
                sub = [r for r in fold_rows
                       if r["seed"] == seed and r["target"] == task and r["method"] == method]
                if not sub:
                    continue
                mae = np.array([r["MAE"] for r in sub])
                summary_by_seed.append({
                    "seed": seed, "task_id": task, "method": method,
                    "MAE_mean_over_folds": float(mae.mean()),
                    "MAE_std_over_folds": float(mae.std(ddof=1)) if len(mae) > 1 else float("nan"),
                    "RMSE_mean_over_folds": float(np.mean([r["RMSE"] for r in sub])),
                    "R2_mean_over_folds": float(np.mean([r["R2"] for r in sub])),
                    "n_folds": len(sub),
                })

    summary_by_task = []
    for task in sorted({r["task_id"] for r in summary_by_seed}):
        for method in methods:
            sub = [r for r in summary_by_seed
                   if r["task_id"] == task and r["method"] == method]
            if not sub:
                continue
            mae = np.array([r["MAE_mean_over_folds"] for r in sub])
            summary_by_task.append({
                "task_id": task, "method": method,
                "MAE_mean_over_seeds": float(mae.mean()),
                "MAE_sem_over_seeds": float(mae.std(ddof=1) / np.sqrt(len(mae))) if len(mae) > 1 else float("nan"),
                "MAE_std_over_seeds": float(mae.std(ddof=1)) if len(mae) > 1 else float("nan"),
                "R2_mean_over_seeds": float(np.mean([r["R2_mean_over_folds"] for r in sub])),
                "n_seeds": len(sub),
            })

    # Improvement distribution: ΔMAE(fedper_cal - fedper) per (task, seed)
    improvement_rows = []
    for seed in args.seeds:
        for task in sorted({r["task_id"] for r in summary_by_seed if r["seed"] == seed}):
            fp = next((r for r in summary_by_seed
                       if r["seed"] == seed and r["task_id"] == task
                       and r["method"] == "fedper"), None)
            fpc = next((r for r in summary_by_seed
                        if r["seed"] == seed and r["task_id"] == task
                        and r["method"] == "fedper_cal"), None)
            if fp and fpc:
                d = fpc["MAE_mean_over_folds"] - fp["MAE_mean_over_folds"]
                improvement_rows.append({
                    "seed": seed, "task_id": task,
                    "MAE_fedper": fp["MAE_mean_over_folds"],
                    "MAE_fedper_cal": fpc["MAE_mean_over_folds"],
                    "delta_MAE_cal_minus_fedper": d,
                    "improved": d < 0,
                })

    def _write_csv(name, rows):
        path = OUT_DIR / name
        if not rows:
            return
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path.relative_to(PROJECT_ROOT)} ({len(rows)} rows)")

    _write_csv("pseudo_task_definitions.csv", task_defs)
    _write_csv("per_fold_results.csv", fold_rows)
    _write_csv("summary_by_seed.csv", summary_by_seed)
    _write_csv("summary_by_task.csv", summary_by_task)
    _write_csv("improvement_distribution.csv", improvement_rows)
    _write_csv("runtime_log.csv", runtime_rows)

    # Smoke / full log
    log_path = OUT_DIR / ("smoke_log.md" if args.smoke else "full_run_log.md")
    with log_path.open("w") as fh:
        fh.write(f"# Phase 2 — {'smoke' if args.smoke else 'full'} run log\n\n")
        fh.write(f"Config:\n```\n{json.dumps(config, indent=2)}\n```\n\n")
        fh.write("## Per-method mean MAE over all pseudo-targets and seeds\n\n")
        fh.write("| method | MAE | n |\n|---|---|---|\n")
        for method in methods:
            sub = [r for r in fold_rows if r["method"] == method]
            mae = np.mean([r["MAE"] for r in sub])
            fh.write(f"| {method} | {mae:.4f} | {len(sub)} |\n")
        if improvement_rows:
            n_improved = sum(1 for r in improvement_rows if r["improved"])
            fh.write(f"\n## Calibration vs FedPer improvement rate: "
                     f"{n_improved}/{len(improvement_rows)} (target × seed)\n")
        total_elapsed = sum(r["elapsed_sec_total"] for r in runtime_rows)
        fh.write(f"\n## Runtime\nTotal: {total_elapsed:.0f} s ({total_elapsed/60:.1f} min)\n")
    print(f"wrote {log_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
