"""Paper figures and tables for the PC2-FedReorg manuscript.

Read-only with respect to ``src/``, ``experiments/``, ``results/`` and training
code.  All outputs land in ``paper/figures`` and ``paper/tables``.

Run from project root:
    python paper/make_figures.py

Font policy
-----------
Spec required Times New Roman.  Neither GPU box nor CPU box have it installed
(``fc-match`` silently substitutes Liberation Serif).  Per user authorisation
on 2026-05-11, this script falls back to Liberation Serif with strong
TNR-compatible visuals (same x-height, similar metrics).  The script HARD-STOPS
if Liberation Serif is also missing.
"""

from __future__ import annotations

import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.stats import wilcoxon

import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D


# ============================================================================
# Font configuration
# ============================================================================
PRIMARY_FONT = "Liberation Serif"   # TNR metric-compatible (user-authorised)
_font_names = {f.name for f in font_manager.fontManager.ttflist}
if PRIMARY_FONT not in _font_names:
    raise RuntimeError(
        f"{PRIMARY_FONT} is not installed. Spec forbids further fallback. "
        f"Available serif fonts: "
        f"{sorted(n for n in _font_names if 'serif' in n.lower() or 'roman' in n.lower())}"
    )

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = [PRIMARY_FONT]
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["axes.linewidth"] = 0.8
mpl.rcParams["xtick.major.width"] = 0.8
mpl.rcParams["ytick.major.width"] = 0.8


# ============================================================================
# Paths
# ============================================================================
ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
FIG_DIR = PAPER_DIR / "figures"
TBL_DIR = PAPER_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)
FAIL_LOG = FIG_DIR / "molecule_draw_failures.txt"

AB_CSV = ROOT / "data" / "client_a_b" / "public_reorg_energy_15210.csv"
C_HOLE_CSV = ROOT.parent / "logs" / "hole_reorg_molecular.csv"
C_TRIPLET_CSV = ROOT.parent / "logs" / "reorganization_energy_summary.csv"
D_CSV = ROOT.parent / "logs" / "atahan_reorg_5876.csv"

T_TRANSFER = ROOT / "results" / "preverify" / "T_transferability.json"
T_CHEMISTRY = ROOT / "results" / "preverify" / "T_chemistry.json"
PC2_MAIN_CSV = ROOT / "results" / "pc2_batch1_multiseed_summary.csv"
PC2_MAIN_JSON = ROOT / "results" / "pc2_batch1_multiseed_predictions.json"
PC2_ABL_CSV = ROOT / "results" / "pc2_phase5_ablation_summary.csv"
PC2_ABL_JSON = ROOT / "results" / "pc2_phase5_ablation_predictions.json"
PC2_STATS_CSV = ROOT / "results" / "pc2_phase5_stats.csv"


# ============================================================================
# Colour palette (JCIM-friendly, colour-blind safe)
# ============================================================================
COL_A = "#3B7DB3"   # blue
COL_B = "#6FAE5C"   # green
COL_CHOLE = "#D97825"   # orange
COL_CTRIP = "#A0408E"   # purple
COL_D = "#D03B43"   # red
NODE_COLORS = {
    "A": COL_A, "B": COL_B,
    "C-hole": COL_CHOLE, "C-triplet": COL_CTRIP,
    "D": COL_D,
}
METHOD_COLORS = {
    "E60": "#888888",  # local
    "E61": "#3B7DB3",  # FedAvg
    "E62": "#6FAE5C",  # FedProx
    "E63": "#D97825",  # FedPer
    "E66": "#A0408E",  # PC²
    "E70": "#D03B43",  # PC² no-cal
    "E68": "#C4A02D",
    "E69": "#6A6A6A",
    "E65": "#3F8F7F",
}


# ============================================================================
# Drawing helper: SMILES -> PIL image
# ============================================================================
def smiles_to_image(smiles: str, size: int = 360) -> Image.Image | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
        opts = drawer.drawOptions()
        opts.padding = 0.10
        opts.bondLineWidth = 1.4
        opts.fixedBondLength = 22
        opts.baseFontSize = 0.55
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception as e:  # pragma: no cover
        _log_failure(smiles, repr(e))
        return None


def _log_failure(smiles: str, reason: str) -> None:
    with FAIL_LOG.open("a") as f:
        f.write(f"{smiles}\t{reason}\n")


# ============================================================================
# Deterministic representative-molecule selection
# ============================================================================
def select_representative(df: pd.DataFrame, smiles_col: str, label_col: str
                          ) -> tuple[str, float]:
    """Return (smiles, label) for the molecule whose label is closest to the
    median.  Falls through to next-closest if the picked SMILES fails to
    render."""
    sub = df[[smiles_col, label_col]].dropna().copy()
    median = float(sub[label_col].median())
    sub["dist"] = (sub[label_col] - median).abs()
    sub = sub.sort_values(["dist", label_col]).reset_index(drop=True)
    for _, row in sub.iterrows():
        smi = row[smiles_col]
        if Chem.MolFromSmiles(smi) is None:
            _log_failure(smi, "MolFromSmiles None")
            continue
        if smiles_to_image(smi) is None:
            continue
        return smi, float(row[label_col])
    raise RuntimeError(f"No renderable representative in dataframe; col={smiles_col}")


# ============================================================================
# Data loading (read-only)
# ============================================================================
def load_node_dataframes() -> dict[str, pd.DataFrame]:
    """Reproduce the data-utils split deterministically without invoking PyG.
    Returns dataframes for A, B, C-hole, C-triplet, D, each with ['smiles', 'label']."""
    out: dict[str, pd.DataFrame] = {}

    ab = pd.read_csv(AB_CSV)
    arom = []
    for smi in ab["smiles"]:
        m = Chem.MolFromSmiles(smi)
        arom.append(rdMolDescriptors.CalcNumAromaticRings(m) if m is not None else -1)
    ab = ab.copy()
    ab["arom"] = arom
    ab_valid = ab[ab["arom"] >= 0]
    out["A"] = ab_valid[ab_valid["arom"] == 0][["smiles", "reorg_energy_eV"]]\
        .rename(columns={"reorg_energy_eV": "label"}).reset_index(drop=True)
    out["B"] = ab_valid[ab_valid["arom"] >= 1][["smiles", "reorg_energy_eV"]]\
        .rename(columns={"reorg_energy_eV": "label"}).reset_index(drop=True)

    hole = pd.read_csv(C_HOLE_CSV).dropna(subset=["smiles", "lambda_hole_eV"])
    out["C-hole"] = hole[["smiles", "lambda_hole_eV"]]\
        .rename(columns={"lambda_hole_eV": "label"}).reset_index(drop=True)

    trip = pd.read_csv(C_TRIPLET_CSV)
    trip_v = trip.dropna(subset=["lambda_T_total_eV"]).reset_index(drop=True)
    smap = dict(zip(hole["molecule"], hole["smiles"]))
    trip_v = trip_v.copy()
    trip_v["smiles"] = trip_v["molecule"].map(smap)
    trip_v = trip_v.dropna(subset=["smiles"]).reset_index(drop=True)
    out["C-triplet"] = trip_v[["smiles", "lambda_T_total_eV", "molecule"]]\
        .rename(columns={"lambda_T_total_eV": "label"}).reset_index(drop=True)

    d = pd.read_csv(D_CSV).dropna(subset=["smiles", "reorg_eV"])
    out["D"] = d[["smiles", "reorg_eV"]]\
        .rename(columns={"reorg_eV": "label"}).reset_index(drop=True)

    return out


def node_metadata(node_dfs: dict[str, pd.DataFrame], t_meta: dict) -> dict[str, dict]:
    """Per-node summary stats for Figure 1 cards and Table 1."""
    info = {}
    descr = {
        "A": ("QM9 public", "cation reorg λ"),
        "B": ("QM9 public", "cation reorg λ"),
        "C-hole": ("private TADF", "hole reorg λ"),
        "C-triplet": ("private TADF", "triplet reorg λ"),
        "D": ("Atahan 2019", "hole reorg λ"),
    }
    for k, df in node_dfs.items():
        labels = df["label"].astype(float).values
        proto = t_meta["protocol_meta"][k]
        info[k] = {
            "n": int(len(df)),
            "mean": float(labels.mean()),
            "std": float(labels.std(ddof=1)) if len(labels) > 1 else 0.0,
            "min": float(labels.min()),
            "max": float(labels.max()),
            "source": descr[k][0],
            "quantity": descr[k][1],
            "functional": proto.get("functional", "n/a"),
            "basis": proto.get("basis", "n/a"),
            "charge_state": proto.get("charge_state", "n/a"),
            "geometry": proto.get("geometry", "n/a"),
        }
    return info


# ============================================================================
# Save helper: PDF + SVG + PNG@600dpi
# ============================================================================
def save_fig(fig: plt.Figure, stem: str) -> None:
    base = FIG_DIR / stem
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"  wrote {base.with_suffix('.pdf').name} + .svg + .png (600 dpi)")


# ============================================================================
# Figure 1: task-client heterogeneity
# ============================================================================
def fig1_task_clients(node_dfs, info, reps) -> None:
    print("[Fig 1] task-client heterogeneity")
    nodes = ["A", "B", "C-hole", "C-triplet", "D"]
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[3.4, 1.7],
                           width_ratios=[1, 1, 1, 1, 1, 1.05],
                           hspace=0.30, wspace=0.32, left=0.04, right=0.985,
                           top=0.94, bottom=0.06)

    for i, k in enumerate(nodes):
        ax = fig.add_subplot(gs[0, i])
        img = smiles_to_image(reps[k][0], size=420)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(NODE_COLORS[k]); s.set_linewidth(2.0)
        title = f"{k}"
        ax.set_title(title, color=NODE_COLORS[k], fontsize=15, fontweight="bold",
                      pad=4)

        meta = info[k]
        cap = (f"{meta['source']}\n"
                f"{meta['quantity']}\n"
                f"n = {meta['n']}\n"
                f"λ̄ = {meta['mean']:.3f} ± {meta['std']:.3f} eV\n"
                f"range [{meta['min']:.3f}, {meta['max']:.3f}]")
        ax.text(0.5, -0.04, cap, transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, linespacing=1.35)

    # Right column: label-scale comparison bar
    ax_bar = fig.add_subplot(gs[0, 5])
    means = [info[k]["mean"] for k in nodes]
    stds = [info[k]["std"] for k in nodes]
    ypos = np.arange(len(nodes))[::-1]
    ax_bar.barh(ypos, means, xerr=stds, color=[NODE_COLORS[k] for k in nodes],
                 edgecolor="black", linewidth=0.6, capsize=3,
                 error_kw=dict(elinewidth=0.8))
    ax_bar.set_yticks(ypos); ax_bar.set_yticklabels(nodes, fontsize=10)
    ax_bar.set_xlabel("Mean label λ (eV)", fontsize=10)
    ax_bar.set_title("Label-scale\nheterogeneity", fontsize=11, pad=4)
    ax_bar.tick_params(labelsize=9)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.set_xlim(0, max(m + s for m, s in zip(means, stds)) * 1.15)

    # Lower row: distribution strip
    ax_dist = fig.add_subplot(gs[1, :])
    pos = []
    for i, k in enumerate(nodes):
        labels = node_dfs[k]["label"].values
        x = labels
        y = np.full_like(x, i, dtype=float) + np.random.RandomState(0).uniform(-0.18, 0.18, size=len(x))
        ax_dist.scatter(x, y, s=4 if k == "D" else 14, alpha=0.45 if k == "D" else 0.7,
                         color=NODE_COLORS[k], edgecolors="none")
        pos.append(i)
    ax_dist.set_yticks(pos); ax_dist.set_yticklabels(nodes, fontsize=10)
    ax_dist.set_xlabel("Reorganization energy λ (eV)", fontsize=11)
    ax_dist.set_title("Per-task label distributions on a common axis "
                       "(all five tasks are reorganization-energy related, "
                       "but quantity, scale, protocol and chemical space differ)",
                       fontsize=11, pad=6)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    ax_dist.tick_params(labelsize=9)
    ax_dist.set_xlim(0, 3.6)

    fig.text(0.005, 0.992, "(a)", fontsize=13, fontweight="bold")
    fig.text(0.005, 0.31, "(b)", fontsize=13, fontweight="bold")
    save_fig(fig, "Fig1_task_client_molecules")


# ============================================================================
# Figure 2: PC2 federated architecture (double-column compact, 7.2 inch)
# ============================================================================
def fig2_architecture(reps, info) -> None:
    """Graphical-method figure: PC²-FedReorg's per-key aggregation routing.

    Shows ONE generic task-client lane (Mol → Encoder → Adapter → Head →
    Calibration → λ̂) with above-stage labels indicating per-key sharing rules.
    A compact federation-server callout summarises the four routing rules.
    The bottom band displays the z-space training / eV-space inference formula.

    Designed as a single-glance method figure for double-column layout.
    """
    print("[Fig 2] PC² method (single-lane graphical-method, simplified)")

    W, H = 7.2, 4.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # ---- Colour key (shared = blue family; private = red family) ----
    C_SHARED = "#3F7CB4"     # encoder / adapter -- shared via T-gates
    C_HEAD   = "#B03060"     # head -- private (FedPer-excluded)
    C_CAL    = "#7A5300"     # calibration -- never aggregated

    # ---- Title ----
    ax.text(W/2, H - 0.20,
            "PC²-FedReorg: per-key aggregation with task-specific calibration",
            ha="center", va="center", fontsize=11.5, fontweight="bold")

    # ============ Federation server (compact, right-hand inset) ============
    srv_x0, srv_x1 = W - 2.45, W - 0.15
    srv_y0, srv_y1 = H - 0.85, H - 0.40
    ax.add_patch(FancyBboxPatch((srv_x0, srv_y0), srv_x1 - srv_x0, srv_y1 - srv_y0,
                                boxstyle="round,pad=0.03",
                                facecolor="#1F2A44", edgecolor="black", linewidth=0.8))
    ax.text((srv_x0+srv_x1)/2, (srv_y0+srv_y1)/2,
            "Federation server  ·  per-key routing",
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="white")

    # ============ Single generic lane ============
    lane_y = 1.80
    box_h = 0.55
    x_mol, x_enc, x_adp, x_head, x_cal, x_pred = 0.55, 1.55, 2.55, 3.55, 4.55, 5.85
    box_w = 0.78

    # Lane label
    ax.text(x_mol - 0.42, lane_y, "Client c",
            ha="right", va="center", fontsize=10, fontweight="bold", color="#222")

    # Molecule (representative — pick a generic one from C-triplet)
    rep_key = "C-triplet"
    img = smiles_to_image(reps[rep_key][0], size=320)
    ext = [x_mol - 0.42, x_mol + 0.42, lane_y - 0.32, lane_y + 0.32]
    ax.imshow(img, extent=ext, aspect="auto", zorder=3)

    # Boxes
    def stage_box(xc, label, color, hatch=""):
        ax.add_patch(FancyBboxPatch((xc - box_w/2, lane_y - box_h/2), box_w, box_h,
                                     boxstyle="round,pad=0.03",
                                     facecolor=color, alpha=0.85,
                                     edgecolor="black", linewidth=0.8,
                                     hatch=hatch))
        ax.text(xc, lane_y, label, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="white")

    stage_box(x_enc,  "Encoder",  C_SHARED)
    stage_box(x_adp,  "Adapter",  C_SHARED)
    stage_box(x_head, "Head",     C_HEAD)
    stage_box(x_cal,  "Cal\n(μ,σ)", C_CAL)
    # Prediction
    ax.add_patch(FancyBboxPatch((x_pred - box_w/2, lane_y - box_h/2),
                                 box_w, box_h,
                                 boxstyle="round,pad=0.03",
                                 facecolor="white", edgecolor="black", linewidth=1.0))
    ax.text(x_pred, lane_y, r"$\hat\lambda_c$ (eV)",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # Arrows
    for x0, x1 in [(x_mol + 0.42, x_enc - box_w/2),
                   (x_enc + box_w/2, x_adp - box_w/2),
                   (x_adp + box_w/2, x_head - box_w/2),
                   (x_head + box_w/2, x_cal - box_w/2),
                   (x_cal + box_w/2, x_pred - box_w/2)]:
        ax.add_patch(FancyArrowPatch((x0, lane_y), (x1, lane_y),
                                     arrowstyle="-|>", mutation_scale=10,
                                     color="#444", lw=1.2))

    # ============ Per-stage routing rule labels (above each box) ============
    y_rule = lane_y + box_h/2 + 0.35
    rules = [
        (x_enc,  r"share via $T_{\mathrm{repr}}$",    C_SHARED),
        (x_adp,  r"share via $T_{\mathrm{adapter}}$", C_SHARED),
        (x_head, r"private (FedPer)" + "\n" + r"+ $T_{\mathrm{head}}$ for same-source pairs", C_HEAD),
        (x_cal,  "never aggregated\n(LOOCV-fold-private)", C_CAL),
    ]
    for xc, txt, c in rules:
        ax.text(xc, y_rule, txt, ha="center", va="bottom",
                fontsize=8.3, color=c, fontweight="bold", linespacing=1.2)

    # ============ Server-to-lane curly bracket (visual link) ============
    # Light dashed line from server bottom to encoder/adapter boxes
    for xc in (x_enc, x_adp):
        ax.plot([xc, xc + (srv_x0 + srv_x1)/2 - xc],
                [y_rule + 0.45, srv_y0 - 0.02],
                color="#888", lw=0.6, ls=":")

    # ============ Formula band (bottom) ============
    ax.add_patch(FancyBboxPatch((0.30, 0.30), W - 0.60, 0.80,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#F4F1E8", edgecolor=C_CAL,
                                 linewidth=1.0))
    ax.text(W/2, 0.92,
            "Per-task-client calibration  —  installed from each LOOCV fold's training labels",
            ha="center", va="center", fontsize=8.7,
            fontweight="bold", color=C_CAL)
    ax.text(W/2, 0.55,
            r"Training (z-space):  $z = (\lambda - \mu_c)\,/\,\sigma_c$"
            "        "
            r"Inference (eV-space):  $\hat\lambda = \sigma_c \cdot \hat z + \mu_c$",
            ha="center", va="center", fontsize=10.5)

    # ============ Small legend (top-left corner) ============
    leg_x, leg_y = 0.30, H - 0.55
    for i, (lbl, c) in enumerate([("shared", C_SHARED),
                                    ("private",  C_HEAD),
                                    ("never aggregated", C_CAL)]):
        ax.add_patch(Rectangle((leg_x + i*1.15, leg_y), 0.18, 0.10,
                                facecolor=c, edgecolor="none"))
        ax.text(leg_x + i*1.15 + 0.22, leg_y + 0.05, lbl,
                ha="left", va="center", fontsize=8.0, color="#333")

    save_fig(fig, "Fig2_pc2_method")


# ============================================================================
# Figure 3: transferability matrices
# ============================================================================
def fig3_transferability() -> None:
    print("[Fig 3] transferability heatmaps")
    T = json.loads(T_TRANSFER.read_text())
    nodes = T["node_keys"]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6))
    titles = [("T_repr", "encoder gate (chemistry-driven)"),
               ("T_adapter", "adapter gate (geom. mean)"),
               ("T_head", "head gate (protocol-conservative)")]
    keys = ["T_repr", "T_adapter", "T_head"]

    for ax, key, (title, sub) in zip(axes, keys, titles):
        mat = np.array([[T[key][i][j] for j in nodes] for i in nodes])
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(nodes))); ax.set_yticks(range(len(nodes)))
        ax.set_xticklabels(nodes, fontsize=10, rotation=30, ha="right")
        ax.set_yticklabels(nodes, fontsize=10)
        ax.set_xlabel("target j", fontsize=10.5)
        ax.set_ylabel("source i", fontsize=10.5)
        ax.set_title(f"{title}\n{sub}", fontsize=11, pad=6)
        for i, src in enumerate(nodes):
            for j, tgt in enumerate(nodes):
                v = mat[i, j]
                tc = "white" if v < 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=8.5, color=tc)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    fig.suptitle(
        "Auditable transferability matrices — used as compatibility metadata "
        "and safety guard, not as the main performance driver",
        fontsize=12, y=1.02
    )
    fig.tight_layout()

    # T_head call-outs
    ax_head = axes[2]
    nset = {n: i for i, n in enumerate(nodes)}
    callouts = [("D", "C-hole", "= 0  (V1 FAIL)"),
                 ("D", "C-triplet", "= 0  (quantity)"),
                 ("C-hole", "C-triplet", "= 0  (quantity)"),
                 ("C-triplet", "C-hole", "= 0  (quantity)")]
    for src, tgt, note in callouts:
        i = nset[src]; j = nset[tgt]
        ax_head.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9,
                                      fill=False, edgecolor="#FF0066", linewidth=1.6))
    save_fig(fig, "Fig3_transferability_matrices")


# ============================================================================
# Figure 4: main performance
# ============================================================================
def fig4_main_performance() -> None:
    print("[Fig 4] main performance C-hole / C-triplet")
    df = pd.read_csv(PC2_MAIN_CSV)
    stats = pd.read_csv(PC2_STATS_CSV)

    methods = ["E60", "E61", "E62", "E63", "E66"]
    labels = ["E60 Local", "E61 FedAvg", "E62 FedProx", "E63 FedPer", "E66 PC²"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))

    for ax, (target, title) in zip(axes,
                                     [("C-hole", "Client C — hole reorganization energy  (n = 53, LOOCV)"),
                                      ("C-triplet", "Client C — triplet reorganization energy  (n = 49, LOOCV)")]):
        means = []
        sems = []
        for m in methods:
            sub = df[(df["exp_id"] == m) & (df["target"] == target)]
            mae_vals = sub["MAE_mean"].astype(float).values
            means.append(float(mae_vals.mean()))
            sems.append(float(mae_vals.std(ddof=1) / np.sqrt(len(mae_vals))) if len(mae_vals) > 1 else 0.0)

        x = np.arange(len(methods))
        bars = ax.bar(x, means, yerr=sems, capsize=4,
                       color=[METHOD_COLORS[m] for m in methods],
                       edgecolor="black", linewidth=0.7,
                       error_kw=dict(elinewidth=0.9))
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5, rotation=10)
        ax.set_ylabel("MAE (eV)", fontsize=11)
        ax.set_title(title, fontsize=11.5, pad=6)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_ylim(0, max(means) * 1.32)

        # Significance brackets vs E66
        target_key = "loocv_c_hole" if target == "C-hole" else "loocv_c_triplet"
        sub = stats[stats["target"] == target_key].set_index("compared_to")

        y_top = max(means) * 1.05
        step = max(means) * 0.07
        e66_idx = methods.index("E66")
        for k, other in enumerate(["E61", "E62", "E63"]):
            if other not in sub.index:
                continue
            p = float(sub.loc[other, "wilcoxon_p"])
            other_idx = methods.index(other)
            yb = y_top + step * (k + 1)
            ax.plot([other_idx, other_idx, e66_idx, e66_idx],
                     [yb - step*0.25, yb, yb, yb - step*0.25],
                     color="#333", lw=0.9)
            note = "n.s." if p >= 0.05 else f"nominal p = {p:.3f}"
            ax.text((other_idx + e66_idx) / 2, yb + step*0.10, note,
                     ha="center", va="bottom", fontsize=8.6)

        # Overall headline
        if target == "C-hole":
            ax.text(0.02, 0.96,
                     "All paired tests vs PC² are n.s.\n(narrow MAE band 0.37–0.41 eV; small-n data ceiling)",
                     transform=ax.transAxes, ha="left", va="top", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="#FFF5E1", edgecolor="#B98A30",
                               linewidth=0.7))
        else:
            ax.text(0.02, 0.96,
                     "PC² > FedAvg / FedProx / FedPer\n"
                     "(paired Wilcoxon nominal p < 0.05;\n"
                     "unadjusted, would not survive Bonferroni — §4.3)",
                     transform=ax.transAxes, ha="left", va="top", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="#E5F3DF", edgecolor="#3F8F4F",
                               linewidth=0.7))

    fig.tight_layout()
    save_fig(fig, "Fig4_main_performance")


# ============================================================================
# Figure 5: ablation forest plot
# ============================================================================
def fig5_ablation_forest() -> None:
    """Calibration-mechanism figure (rewritten after E71 lands).

    Three panels:
      (a) C-triplet MAE bar plot for E63 FedPer, E70 PC² no-cal, E66 PC² full,
          E71 FedPer + calibration only.
      (b) Forest plot of paired ΔMAE on C-triplet for the three E71 contrasts
          plus the four planned PC² ablations (so the full ablation evidence
          stays visible).
      (c) Interpretation box stating the mechanism conclusion in plain text.

    Data sources (read-only):
      results/e71_summary.csv               (E71 across-seed MAE)
      results/e71_vs_e66_stats.csv          (E71 paired stats vs E66; CSV form)
      results/pc2_batch1_multiseed_summary.csv  (E63, E66)
      results/pc2_phase5_ablation_summary.csv   (E70)
      results/pc2_phase5_stats.csv          (existing PC² ablation paired stats)
      results/e71_vs_e63_stats.md / e71_vs_e70_stats.md  (E71 vs FedPer / no-cal,
                                                          parsed numerically below)
    """
    print("[Fig 5] calibration-mechanism figure (E71 + four PC² ablations)")

    # ------------------------------------------------------------------
    # Helper: across-seed mean MAE for (exp, target)
    # ------------------------------------------------------------------
    def _across_seed_mae(summary_df: pd.DataFrame, exp: str, target: str
                         ) -> tuple[float, float]:
        sub = summary_df[(summary_df["exp_id"] == exp)
                         & (summary_df["target"] == target)]
        if sub.empty:
            raise RuntimeError(f"No rows for {exp} / {target} in summary")
        m = sub["MAE_mean"].astype(float)
        return float(m.mean()), float(m.std(ddof=1) / np.sqrt(len(m)))

    e71_csv  = ROOT / "results" / "e71_summary.csv"
    e71_main = ROOT / "results" / "e71_vs_e66_stats.csv"
    df_main  = pd.read_csv(PC2_MAIN_CSV)
    df_abl   = pd.read_csv(PC2_ABL_CSV)
    df_e71   = pd.read_csv(e71_csv)
    e71stats = pd.read_csv(e71_main)
    pc2stats = pd.read_csv(PC2_STATS_CSV)

    # ------------------------------------------------------------------
    # Panel (a): C-triplet MAE bar plot, four methods
    # ------------------------------------------------------------------
    methods_a = [
        ("E63", "FedPer\n(no calibration)",        df_main, "#888888"),
        ("E70", "PC²\nno calibration",             df_abl,  "#D03B43"),
        ("E66", "PC²-FedReorg\n(full)",            df_main, "#A0408E"),
        ("E71", "FedPer +\ncalibration only",      df_e71,  "#2E8B57"),
    ]
    mae_means_t, mae_sems_t = [], []
    mae_means_h, mae_sems_h = [], []
    for exp, _, df, _ in methods_a:
        m, s = _across_seed_mae(df, exp, "C-triplet")
        mae_means_t.append(m); mae_sems_t.append(s)
        m, s = _across_seed_mae(df, exp, "C-hole")
        mae_means_h.append(m); mae_sems_h.append(s)

    # ------------------------------------------------------------------
    # Panel (b): forest plot of paired ΔMAE on C-triplet
    #   Three rows for E71 contrasts (E71 - X)
    #   Four rows for PC² ablations (PC² - X) preserved from prior Figure 5
    # ------------------------------------------------------------------
    def _e71_row(other: str, target: str = "loocv_c_triplet") -> dict:
        row = e71stats[(e71stats["target"] == target)
                       & (e71stats["compared_to"] == other)].iloc[0]
        return dict(diff=float(row["mae_diff"]),
                    lo=float(row["mae_diff_ci_low"]),
                    hi=float(row["mae_diff_ci_high"]),
                    p=float(row["wilcoxon_p"]))

    def _pc2_row(other: str, target: str = "loocv_c_triplet") -> dict:
        row = pc2stats[(pc2stats["target"] == target)
                       & (pc2stats["compared_to"] == other)].iloc[0]
        return dict(diff=float(row["mae_diff"]),
                    lo=float(row["mae_diff_ci_low"]),
                    hi=float(row["mae_diff_ci_high"]),
                    p=float(row["wilcoxon_p"]))

    # E71-vs-E63 / E71-vs-E70 are only in MD form in v1; recompute on the
    # fly from per-molecule predictions to keep the figure data-driven.
    def _e71_vs_arbitrary(other_exp: str, target: str = "loocv_c_triplet") -> dict:
        e71_preds = json.loads((ROOT / "results" / "e71_predictions.json").read_text())
        if other_exp in ("E66",):
            other_preds = json.loads(PC2_MAIN_JSON.read_text())
        elif other_exp in ("E70",):
            other_preds = json.loads(PC2_ABL_JSON.read_text())
        elif other_exp in ("E63", "E60", "E61", "E62"):
            other_preds = json.loads(PC2_MAIN_JSON.read_text())
        else:
            raise ValueError(f"Unknown other_exp={other_exp}")

        def med_err(blobs):
            seeds = sorted(blobs.keys())
            errs = []
            for s in seeds:
                blob = blobs[s].get(target)
                if blob is None:
                    continue
                yt = np.asarray(blob["y_true"], dtype=float)
                yp = np.asarray(blob["y_pred"], dtype=float)
                errs.append(np.abs(yp - yt))
            return np.median(np.vstack(errs), axis=0)

        err_e71 = med_err(e71_preds["E71"])
        err_oth = med_err(other_preds[other_exp])
        diff = float(err_e71.mean() - err_oth.mean())
        rng = np.random.default_rng(0)
        n = len(err_e71)
        boots = np.array([err_e71[rng.integers(0, n, n)].mean()
                          - err_oth[rng.integers(0, n, n)].mean()
                          for _ in range(5000)])
        # bootstrap on PAIRED resampling: same indices on both arrays
        boots = []
        for _ in range(5000):
            idx = rng.integers(0, n, n)
            boots.append(err_e71[idx].mean() - err_oth[idx].mean())
        boots = np.asarray(boots)
        try:
            from scipy.stats import wilcoxon
            _, p = wilcoxon(err_e71, err_oth, zero_method="wilcox",
                            alternative="two-sided")
        except Exception:
            p = float("nan")
        return dict(diff=diff,
                    lo=float(np.percentile(boots, 2.5)),
                    hi=float(np.percentile(boots, 97.5)),
                    p=float(p))

    contrasts_b = [
        # E71 contrasts: ΔMAE = MAE(E71) − MAE(other); negative => E71 better
        ("E71 vs PC²-FedReorg (E66)",            _e71_vs_arbitrary("E66"), "#2E8B57"),
        ("E71 vs PC² no-calibration (E70)",      _e71_vs_arbitrary("E70"), "#2E8B57"),
        ("E71 vs FedPer (E63)",                  _e71_vs_arbitrary("E63"), "#2E8B57"),
        # PC² ablations (for parity with previous Figure 5): ΔMAE = MAE(PC²) − MAE(other)
        ("PC² vs PC² no-calibration (E70)",      _pc2_row("E70"),          "#C03030"),
        ("PC² vs uniform gate (E69)",            _pc2_row("E69"),          "#666666"),
        ("PC² vs no-C-cross repr (E68)",         _pc2_row("E68"),          "#666666"),
        ("PC² vs D→C pretrain (E65)",            _pc2_row("E65"),          "#666666"),
    ]

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(13.6, 6.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 0.85], wspace=0.32)

    # Panel a: bar plot
    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(methods_a))
    width = 0.36
    ax_a.bar(x - width/2, mae_means_t,
             yerr=mae_sems_t, width=width, color=[c for *_, c in methods_a],
             edgecolor="black", lw=0.7, capsize=3, label="C-triplet (n=49)")
    ax_a.bar(x + width/2, mae_means_h,
             yerr=mae_sems_h, width=width, color=[c for *_, c in methods_a],
             edgecolor="black", lw=0.7, capsize=3, alpha=0.45,
             label="C-hole (n=53)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([lbl for _, lbl, _, _ in methods_a], fontsize=9)
    ax_a.set_ylabel("MAE (eV; mean ± SEM over 5 seeds)", fontsize=10)
    ax_a.set_title("(a)  C-triplet / C-hole MAE under four configurations",
                   fontsize=10.5, pad=6, loc="left")
    ax_a.spines["top"].set_visible(False); ax_a.spines["right"].set_visible(False)
    ax_a.legend(loc="upper right", fontsize=8.5, frameon=False)
    ax_a.set_ylim(0, max(mae_means_t) * 1.18)

    # Panel b: forest plot
    ax_b = fig.add_subplot(gs[0, 1])
    y_positions = list(range(len(contrasts_b)))[::-1]
    for y, (lbl, st, color) in zip(y_positions, contrasts_b):
        diff, lo, hi, p = st["diff"], st["lo"], st["hi"], st["p"]
        sig = (not np.isnan(p)) and p < 0.05
        # Negative diff = first method better; sign convention noted in xlabel
        ax_b.errorbar(diff, y, xerr=[[diff - lo], [hi - diff]], fmt="o",
                      color=color, ecolor=color, capsize=4, ms=7, lw=1.4,
                      markerfacecolor=color)
        ptxt = ("nominal p = {:.3f}".format(p) if sig else "n.s. (p = {:.2f})".format(p))
        ax_b.text(hi + 0.012, y, f"{lbl}   {ptxt}",
                  va="center", ha="left", fontsize=8.5)
    ax_b.axvline(0, color="black", lw=0.8, linestyle="--")
    ax_b.set_yticks([])
    ax_b.set_xlabel("Paired ΔMAE on C-triplet,  eV\n"
                    "Top 3 rows:  MAE(E71) − MAE(other);  rows 4–7:  MAE(PC²) − MAE(other);\n"
                    "negative = first method has lower MAE",
                    fontsize=9.5)
    ax_b.set_title("(b)  Paired effect-size forest plot (E71 contrasts + PC² ablations)",
                   fontsize=10.5, pad=6, loc="left")
    ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
    ax_b.set_xlim(-0.32, 0.32)

    # Panel c: interpretation box
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.axis("off")
    ax_c.set_title("(c)  Mechanism summary", fontsize=10.5, pad=6, loc="left")
    box_text = (
        "• E71 ≈ E66  (n.s.; p = 0.929)\n"
        "      FedPer + calibration\n"
        "      reproduces PC²-FedReorg.\n"
        "\n"
        "• E71 <  E63  (nominal p = 0.028)\n"
        "      Calibration alone improves\n"
        "      over plain FedPer.\n"
        "\n"
        "• E71 <  E70  (nominal p = 0.021)\n"
        "      Removing calibration is\n"
        "      more harmful than removing\n"
        "      adapter + gate.\n"
        "\n"
        "─────────────────────────\n"
        "Gate is not required for the\n"
        "observed C-triplet gain in this\n"
        "federation; retained as auditable\n"
        "compatibility metadata only.\n"
        "\n"
        "C-hole remains at a small-sample\n"
        "data ceiling (all p ≥ 0.42)."
    )
    ax_c.text(0.02, 0.98, box_text, transform=ax_c.transAxes,
              fontsize=8.7, va="top", ha="left", family="monospace",
              bbox=dict(facecolor="#F6F2E7", edgecolor="#A8A095", lw=0.8,
                        boxstyle="round,pad=0.5"))

    fig.suptitle("Calibration under personalized heads explains the C-triplet gain",
                 fontsize=11.5, y=1.01)
    fig.text(0.01, -0.02,
             "All p-values are paired Wilcoxon (5-seed median per-molecule abs error), "
             "unadjusted nominal; Bonferroni threshold for the three E71 contrasts is α/3 ≈ 0.017 "
             "(see §4.4).",
             fontsize=8.0, style="italic", color="#555555")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    save_fig(fig, "Fig5_ablation_forest")


# ============================================================================
# Figure 6: representative molecule-level error analysis
# ============================================================================
def fig6_molecule_cases(node_dfs) -> None:
    print("[Fig 6] per-molecule error cases (C-triplet, PC² vs FedAvg)")
    preds = json.loads(PC2_MAIN_JSON.read_text())

    def per_mol_abs_err(exp: str) -> np.ndarray:
        seeds = sorted(preds[exp].keys())
        errs = []
        for s in seeds:
            blob = preds[exp][s]["loocv_c_triplet"]
            yt = np.asarray(blob["y_true"], dtype=float)
            yp = np.asarray(blob["y_pred"], dtype=float)
            errs.append(np.abs(yp - yt))
        return np.median(np.vstack(errs), axis=0), yt

    err_pc2, ytrue = per_mol_abs_err("E66")
    err_fedavg, _ = per_mol_abs_err("E61")
    diff = err_pc2 - err_fedavg
    smiles_list = node_dfs["C-triplet"]["smiles"].tolist()
    assert len(smiles_list) == len(diff), (len(smiles_list), len(diff))

    order = np.argsort(diff)
    improved_idx = list(order[:3])           # most negative diff = PC² much better
    degraded_idx = list(order[-3:][::-1])    # most positive diff = PC² worse
    cases = [(i, "improved") for i in improved_idx] + [(i, "degraded") for i in degraded_idx]

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.6))
    for ax, (idx, kind) in zip(axes.flat, cases):
        smi = smiles_list[idx]
        img = smiles_to_image(smi, size=360)
        if img is None:
            # try next neighbour
            for alt in (idx - 1, idx + 1):
                if 0 <= alt < len(smiles_list):
                    img = smiles_to_image(smiles_list[alt], size=360)
                    if img is not None:
                        idx = alt; smi = smiles_list[alt]
                        break
        if img is not None:
            ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        col = "#3F8F4F" if kind == "improved" else "#C03030"
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(2.0)
        tag = "PC² improved" if kind == "improved" else "PC² degraded"
        ax.set_title(tag, color=col, fontsize=12, fontweight="bold")
        cap = (f"true λ = {ytrue[idx]:.3f} eV\n"
                f"|err| FedAvg = {err_fedavg[idx]:.3f}\n"
                f"|err| PC²    = {err_pc2[idx]:.3f}\n"
                f"Δ|err|       = {diff[idx]:+.3f}")
        ax.text(0.5, -0.03, cap, transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, linespacing=1.35,
                family="serif")

    fig.suptitle("C-triplet per-molecule error analysis: "
                  "top-3 PC²-improved (left) and bottom-3 PC²-degraded (right) "
                  "vs. FedAvg baseline",
                  fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # Demoted to SI per pre-submission cleanup (2026-05-13):
    # per-molecule ranking is qualitative and not a basis for statistical claims.
    save_fig(fig, "FigS1_per_molecule_cases")


# ============================================================================
# Tables 1-4
# ============================================================================
def _write_table(stem: str, rows: list[dict], header: list[str],
                  md_align: list[str] | None = None,
                  md_title: str = "") -> None:
    path_csv = TBL_DIR / f"{stem}.csv"
    path_md = TBL_DIR / f"{stem}.md"
    with path_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})
    md_lines = []
    if md_title:
        md_lines.append(f"# {md_title}\n")
    md_lines.append("| " + " | ".join(header) + " |")
    align = md_align or ["---"] * len(header)
    md_lines.append("| " + " | ".join(align) + " |")
    for r in rows:
        md_lines.append("| " + " | ".join(str(r.get(k, "")) for k in header) + " |")
    path_md.write_text("\n".join(md_lines) + "\n")
    print(f"  wrote {path_csv.name} + {path_md.name}")


def table1_task_clients(info) -> None:
    print("[Table 1] task-client dataset summary")
    nodes = ["A", "B", "C-hole", "C-triplet", "D"]
    header = ["task_client", "source", "quantity", "n", "lambda_mean_eV",
               "lambda_std_eV", "lambda_min_eV", "lambda_max_eV",
               "functional", "basis", "charge_state", "geometry",
               "protocol_note"]
    # For A and B (same public QM9 source), auxiliary protocol fields are
    # not separately curated -- the same-source shortcut of the T_head
    # clauses (Section 2.4) fires before per-field unknown checks. We mark
    # those cells as "-- (same-source)" rather than "unknown" to avoid the
    # misleading implication that A/B have missing protocol metadata.
    SAME_SOURCE = "— (same-source)"
    AB_NOTE = ("Protocol note: same public cation-reorg source protocol. "
               "Head-sharing note: A<->B compatible by same-source shortcut "
               "(Section 2.4).")
    notes = {
        "A": AB_NOTE,
        "B": AB_NOTE,
        "C-hole": "full Nelsen four-point treatment",
        "C-triplet": "full Nelsen four-point treatment",
        "D": ("functional/basis reported; geometry, charge-state and "
              "conformer policy not stated in source metadata"),
    }
    rows = []
    for k in nodes:
        m = info[k]
        if k in ("A", "B"):
            fn = bs = cs = ge = SAME_SOURCE
        else:
            fn, bs, cs, ge = m["functional"], m["basis"], m["charge_state"], m["geometry"]
        rows.append({
            "task_client": k, "source": m["source"], "quantity": m["quantity"],
            "n": m["n"],
            "lambda_mean_eV": f"{m['mean']:.4f}",
            "lambda_std_eV": f"{m['std']:.4f}",
            "lambda_min_eV": f"{m['min']:.4f}",
            "lambda_max_eV": f"{m['max']:.4f}",
            "functional": fn, "basis": bs,
            "charge_state": cs, "geometry": ge,
            "protocol_note": notes[k],
        })
    _write_table("table1_task_clients", rows, header,
                  md_title="Task-client dataset summary")


def table2_main_performance() -> None:
    print("[Table 2] main performance summary")
    df = pd.read_csv(PC2_MAIN_CSV)
    methods = ["E60", "E61", "E62", "E63", "E66"]
    method_lbl = {"E60": "E60 Local-only", "E61": "E61 FedAvg",
                   "E62": "E62 FedProx", "E63": "E63 FedPer",
                   "E66": "E66 PC²-FedReorg"}
    header = ["method", "target", "n_seeds", "MAE_mean", "MAE_sem",
               "RMSE_mean", "RMSE_sem", "R2_mean", "R2_sem"]
    rows = []
    for m in methods:
        for t in ["A_5fold", "B_5fold", "C-hole", "C-triplet"]:
            sub = df[(df["exp_id"] == m) & (df["target"] == t)]
            if len(sub) == 0:
                continue
            def stat(col):
                v = sub[col].astype(float).values
                return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else (float(v.mean()), 0.0)
            mae_m, mae_s = stat("MAE_mean")
            rms_m, rms_s = stat("RMSE_mean")
            r2_m, r2_s = stat("R2_mean")
            rows.append({
                "method": method_lbl[m], "target": t, "n_seeds": len(sub),
                "MAE_mean": f"{mae_m:.4f}", "MAE_sem": f"{mae_s:.4f}",
                "RMSE_mean": f"{rms_m:.4f}", "RMSE_sem": f"{rms_s:.4f}",
                "R2_mean": f"{r2_m:+.4f}", "R2_sem": f"{r2_s:.4f}",
            })
    _write_table("table2_main_performance", rows, header,
                  md_title="Main performance across A 5-fold, B 5-fold, C-hole LOOCV, C-triplet LOOCV")


def table3_paired_tests() -> None:
    print("[Table 3] paired statistical tests")
    stats = pd.read_csv(PC2_STATS_CSV)
    header = ["target", "reference", "compared_to", "n_mols",
               "mae_ref", "mae_other", "mae_diff",
               "mae_diff_ci_low", "mae_diff_ci_high",
               "ref_wins", "ref_loses", "wilcoxon_p", "significance"]
    rows = []
    for _, r in stats.iterrows():
        p = float(r["wilcoxon_p"])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        rows.append({
            "target": r["target"], "reference": r["reference"],
            "compared_to": r["compared_to"], "n_mols": int(r["n_mols"]),
            "mae_ref": f"{float(r['mae_ref']):.4f}",
            "mae_other": f"{float(r['mae_other']):.4f}",
            "mae_diff": f"{float(r['mae_diff']):+.4f}",
            "mae_diff_ci_low": f"{float(r['mae_diff_ci_low']):+.4f}",
            "mae_diff_ci_high": f"{float(r['mae_diff_ci_high']):+.4f}",
            "ref_wins": int(r["ref_wins"]),
            "ref_loses": int(r["ref_loses"]),
            "wilcoxon_p": f"{p:.4f}",
            "significance": sig,
        })
    _write_table("table3_paired_tests", rows, header,
                  md_title="Paired Wilcoxon signed-rank tests with bootstrap 95% CI")


def table4_ablation() -> None:
    print("[Table 4] ablation summary")
    main = pd.read_csv(PC2_MAIN_CSV)
    abl = pd.read_csv(PC2_ABL_CSV)
    all_df = pd.concat([main, abl], ignore_index=True)
    exps = {"E66": "E66 PC²-FedReorg (reference)",
             "E65": "E65 D→C supervised pretrain",
             "E68": "E68 PC² − T_repr C-cross removed",
             "E69": "E69 PC² − uniform FedAvg gate",
             "E70": "E70 PC² − no calibration"}
    header = ["experiment", "target", "n_seeds", "MAE_mean", "MAE_sem"]
    rows = []
    for exp, lbl in exps.items():
        for t in ["C-hole", "C-triplet"]:
            sub = all_df[(all_df["exp_id"] == exp) & (all_df["target"] == t)]
            if len(sub) == 0:
                continue
            mae_vals = sub["MAE_mean"].astype(float).values
            mae_m = float(mae_vals.mean())
            mae_s = float(mae_vals.std(ddof=1) / np.sqrt(len(mae_vals))) if len(mae_vals) > 1 else 0.0
            rows.append({"experiment": lbl, "target": t,
                          "n_seeds": len(sub),
                          "MAE_mean": f"{mae_m:.4f}",
                          "MAE_sem": f"{mae_s:.4f}"})
    _write_table("table4_ablation", rows, header,
                  md_title="Ablation summary (mean ± SEM across 5 seeds)")


# ============================================================================
# Main
# ============================================================================
def main() -> int:
    # Reset failure log
    FAIL_LOG.write_text("")  # empty file

    print("=" * 70)
    print(f"font = {PRIMARY_FONT}  (Times New Roman not available; "
           "user-authorised Liberation Serif fallback)")
    print("=" * 70)

    node_dfs = load_node_dataframes()
    T_meta = json.loads(T_TRANSFER.read_text())
    info = node_metadata(node_dfs, T_meta)

    # representative molecules (deterministic: median λ)
    reps = {}
    for k in ["A", "B", "C-hole", "C-triplet", "D"]:
        smi, lab = select_representative(node_dfs[k], "smiles", "label")
        reps[k] = (smi, lab)
        print(f"rep[{k}] = label {lab:.4f} eV   {smi[:60]}")

    # Figures
    fig1_task_clients(node_dfs, info, reps)
    fig2_architecture(reps, info)
    fig3_transferability()
    fig4_main_performance()
    fig5_ablation_forest()
    fig6_molecule_cases(node_dfs)

    # Tables
    table1_task_clients(info)
    table2_main_performance()
    table3_paired_tests()
    table4_ablation()

    # Cleanup empty failure log
    if FAIL_LOG.exists() and FAIL_LOG.stat().st_size == 0:
        FAIL_LOG.unlink()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
