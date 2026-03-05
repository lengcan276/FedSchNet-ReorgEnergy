"""
Case Study: Per-molecule analysis of Client D (TADF) predictions.
Compares 2D GIN (E32/E12) vs 3D SchNet (E51/E14) LOOCV results.

Outputs:
  results/tables/case_study_molecules.csv
  results/figures/fig15_case_study.pdf/png
  results/figures/fig16_representative_molecules.pdf/png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, Draw, rdMolDescriptors

RDLogger.logger().setLevel(RDLogger.ERROR)

# ============ Paths ============
PROJECT_ROOT = Path(__file__).resolve().parent
TABLES_DIR = PROJECT_ROOT / 'results' / 'tables'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OLD_RESULTS = PROJECT_ROOT.parent / 'federated_final' / 'results'
DATA_DIR = PROJECT_ROOT / 'data'

# ============ Style ============
COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
C_GIN = COLORS[0]   # blue
C_SCH = COLORS[1]   # green
C_BAD = COLORS[2]   # red
STROKE = [PathEffects.withStroke(linewidth=1.5, foreground='white')]

available_fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
SERIF = None
for c in ['Times New Roman', 'STIXGeneral', 'DejaVu Serif']:
    if c in available_fonts:
        SERIF = c; break
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [SERIF] if SERIF else ['DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'axes.linewidth': 0.8, 'axes.grid': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})


def savefig(fig, name):
    for fmt in ['pdf', 'png']:
        fig.savefig(FIGURES_DIR / f'{name}.{fmt}', format=fmt,
                    dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {name}.pdf + {name}.png")


# ============ Step 1: Load LOOCV predictions ============
print("=" * 60)
print("Case Study: Client D per-molecule analysis")
print("=" * 60)

# TADF SMILES (Client D in paper = client_c directory, the TADF lab data)
df_tadf = pd.read_csv(DATA_DIR / 'client_c' / 'hole_reorg_molecular.csv').dropna(subset=['smiles'])
smiles_list = df_tadf['smiles'].tolist()
names_list = df_tadf['molecule'].tolist()
y_true = df_tadf['lambda_hole_eV'].values

# E32 = GIN 4-client FedBN (paper ID: E12)
with open(OLD_RESULTS / 'batch6_4client_full.json') as f:
    d6 = json.load(f)
gin_data = d6['E32_FedBN_4Client']['loocv_c_hole']
y_pred_gin = np.array(gin_data['y_pred'])

# E51 = SchNet FedAvg 4-client (paper ID: E14)
with open(OLD_RESULTS / 'batch8_schnet_full.json') as f:
    d8 = json.load(f)
sch_data = d8['E51_FedAvg_SchNet']['loocv_c_hole']
y_pred_schnet = np.array(sch_data['y_pred'])

n = len(y_true)
print(f"  Loaded {n} molecules with GIN + SchNet predictions")

# ============ Step 2: Compute molecular descriptors ============
print("  Computing molecular descriptors...")
n_rotatable = []
n_aromatic = []
mol_weights = []
n_heavy = []

for smi in smiles_list:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        n_rotatable.append(Lipinski.NumRotatableBonds(mol))
        n_aromatic.append(rdMolDescriptors.CalcNumAromaticRings(mol))
        mol_weights.append(Descriptors.MolWt(mol))
        n_heavy.append(mol.GetNumHeavyAtoms())
    else:
        n_rotatable.append(0)
        n_aromatic.append(0)
        mol_weights.append(0)
        n_heavy.append(0)

# ============ Step 3: Build full table ============
error_gin = np.abs(y_true - y_pred_gin)
error_schnet = np.abs(y_true - y_pred_schnet)

df_all = pd.DataFrame({
    'molecule': names_list,
    'smiles': smiles_list,
    'y_true': np.round(y_true, 4),
    'y_pred_gin': np.round(y_pred_gin, 4),
    'y_pred_schnet': np.round(y_pred_schnet, 4),
    'error_gin': np.round(error_gin, 4),
    'error_schnet': np.round(error_schnet, 4),
    'n_rotatable': n_rotatable,
    'n_aromatic': n_aromatic,
    'mol_weight': np.round(mol_weights, 1),
    'n_heavy_atoms': n_heavy,
})

# ============ Step 4: Select 8 representative molecules ============
sorted_by_sch_err = df_all.sort_values('error_schnet').reset_index(drop=True)

best_3 = sorted_by_sch_err.head(3)
worst_3 = sorted_by_sch_err.tail(3)
mid_start = len(sorted_by_sch_err) // 2 - 1
mid_2 = sorted_by_sch_err.iloc[mid_start:mid_start + 2]

reps = pd.concat([best_3, mid_2, worst_3], ignore_index=True)
reps['category'] = (['best'] * 3 + ['medium'] * 2 + ['worst'] * 3)

print(f"\n  8 representative molecules selected:")
print(f"    Best 3:   error_schnet = {best_3['error_schnet'].values}")
print(f"    Medium 2: error_schnet = {mid_2['error_schnet'].values}")
print(f"    Worst 3:  error_schnet = {worst_3['error_schnet'].values}")

# Save full table
df_all.to_csv(TABLES_DIR / 'case_study_molecules.csv', index=False)
print(f"\n  Saved: case_study_molecules.csv ({len(df_all)} rows)")

# Also save representative subset
reps.to_csv(TABLES_DIR / 'case_study_representatives.csv', index=False)

# ============ Step 5: Fig15 — Case study visualization ============
print("\nFig15: Case study (bar + scatter)...")

# Sort reps by y_true for the bar chart
reps_sorted = reps.sort_values('y_true').reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

# ---- (a) 8 molecules: predicted vs true ----
ax = axes[0]
x = np.arange(len(reps_sorted))
bw = 0.3

ax.bar(x - bw / 2, reps_sorted['y_pred_gin'], bw,
       color=C_GIN, edgecolor='white', linewidth=0.8,
       alpha=0.85, label='2D GIN (E12)')
ax.bar(x + bw / 2, reps_sorted['y_pred_schnet'], bw,
       color=C_SCH, edgecolor='white', linewidth=0.8,
       alpha=0.85, label='3D SchNet (E14)')

# Red diamond markers for DFT true values
ax.scatter(x, reps_sorted['y_true'], marker='D', c=C_BAD, s=40,
           zorder=10, edgecolors='white', linewidths=0.3,
           label='DFT (true)')

# Short molecule labels on x-axis
short_names = []
for _, row in reps_sorted.iterrows():
    nm = row['molecule']
    if len(nm) > 8:
        nm = nm[:7] + '.'
    short_names.append(nm)

ax.set_xticks(x)
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('$\\lambda_{hole}$ (eV)')
ax.legend(fontsize=7, loc='upper left', frameon=True, framealpha=0.95)

t = ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top')
t.set_path_effects(STROKE)

# ---- (b) All 53 molecules: error vs n_rotatable_bonds ----
ax = axes[1]

ax.scatter(df_all['n_rotatable'], df_all['error_gin'],
           c=[C_GIN], s=30, alpha=0.6, edgecolors='white', linewidths=0.3,
           label='2D GIN', zorder=3)
ax.scatter(df_all['n_rotatable'], df_all['error_schnet'],
           c=[C_SCH], s=30, alpha=0.6, edgecolors='white', linewidths=0.3,
           label='3D SchNet', zorder=3)

# Trend lines (linear fit) in matching colors
for vals, color in [(df_all['error_gin'], C_GIN),
                     (df_all['error_schnet'], C_SCH)]:
    xr = df_all['n_rotatable'].values
    yr = vals.values
    if len(set(xr)) > 1:
        z = np.polyfit(xr, yr, 1)
        p = np.poly1d(z)
        x_line = np.linspace(xr.min(), xr.max(), 50)
        ax.plot(x_line, p(x_line), color=color, linestyle='--',
                linewidth=1.0, alpha=0.7)

ax.set_xlabel('Number of Rotatable Bonds')
ax.set_ylabel('Absolute Error (eV)')
ax.legend(fontsize=7, loc='upper right', frameon=True, framealpha=0.95)

t = ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=10,
            fontweight='bold', va='top')
t.set_path_effects(STROKE)

fig.tight_layout()
savefig(fig, 'fig15_case_study')

# ============ Step 6: Fig16 — 2D ball-and-stick grid ============
print("\nFig16: Representative molecules — 2D ball-and-stick (2x4)...")

from rdkit.Chem import AllChem
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Patch as LegPatch

# Atom color scheme (publication D-π-A style)
ATOM_COLORS = {
    6:  '#4CAF50',   # C  - green
    7:  '#2196F3',   # N  - blue
    8:  '#F44336',   # O  - red
    16: '#FFEB3B',   # S  - yellow
    15: '#FF9800',   # P  - orange
    9:  '#8BC34A',   # F  - light green
    17: '#4CAF50',   # Cl - green
    5:  '#E91E63',   # B  - pink
    35: '#A62929',   # Br - brown
}
ATOM_RADIUS = {6: 0.35, 7: 0.35, 8: 0.35, 16: 0.40, 15: 0.38,
               9: 0.30, 17: 0.36, 5: 0.32, 35: 0.38}
LABEL_ATOMS = {7: 'N', 8: 'O', 16: 'S', 15: 'P', 9: 'F', 5: 'B', 17: 'Cl', 35: 'Br'}

border_colors = {
    'best':   C_SCH,      # green
    'medium': COLORS[4],   # gold
    'worst':  C_BAD,       # red
}


def draw_ball_stick_2d(ax, smiles, title, error_gin, error_schnet, border_color):
    """Draw publication-quality 2D ball-and-stick model with shaded spheres."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        ax.axis('off')
        return

    # Use RDKit 2D coordinate layout (no 3D embedding)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()

    # Heavy atom indices only
    heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    if not heavy_idx:
        ax.axis('off')
        return

    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                        for i in heavy_idx])

    # Normalize to [-1,1] with margin
    if len(coords) > 1:
        center = coords.mean(axis=0)
        coords -= center
        scale = np.abs(coords).max()
        if scale > 0:
            coords /= (scale * 1.2)

    # Smaller radii for 2D layout to avoid overlap
    R_MAP = {6: 0.12, 7: 0.13, 8: 0.13, 16: 0.14, 15: 0.13, 9: 0.11,
             17: 0.13, 5: 0.12, 35: 0.14}

    # --- Draw bonds (below atoms) ---
    for bond in mol.GetBonds():
        bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if bi not in heavy_idx or bj not in heavy_idx:
            continue
        ii = heavy_idx.index(bi)
        jj = heavy_idx.index(bj)
        bt = bond.GetBondTypeAsDouble()

        ax.plot([coords[ii, 0], coords[jj, 0]],
                [coords[ii, 1], coords[jj, 1]],
                color='#424242', linewidth=2.5, zorder=1,
                solid_capstyle='round')

        # Double/triple bond: offset parallel line
        if bt >= 2:
            dx = coords[jj, 0] - coords[ii, 0]
            dy = coords[jj, 1] - coords[ii, 1]
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                off = 0.045
                nx, ny = -dy / norm * off, dx / norm * off
                ax.plot([coords[ii, 0] + nx, coords[jj, 0] + nx],
                        [coords[ii, 1] + ny, coords[jj, 1] + ny],
                        color='#757575', linewidth=1.5, zorder=1)
        if bt >= 3:
            dx = coords[jj, 0] - coords[ii, 0]
            dy = coords[jj, 1] - coords[ii, 1]
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                off = 0.045
                nx, ny = -dy / norm * off, dx / norm * off
                ax.plot([coords[ii, 0] - nx, coords[jj, 0] - nx],
                        [coords[ii, 1] - ny, coords[jj, 1] - ny],
                        color='#757575', linewidth=1.5, zorder=1)

    # --- Draw atoms (above bonds) ---
    for ii, atom_idx in enumerate(heavy_idx):
        atom = mol.GetAtomWithIdx(atom_idx)
        anum = atom.GetAtomicNum()
        color = ATOM_COLORS.get(anum, '#9E9E9E')
        r = R_MAP.get(anum, 0.12)
        cx, cy = coords[ii]

        # Main sphere
        sphere = Circle((cx, cy), r,
                         facecolor=color, edgecolor='#333333',
                         linewidth=0.5, zorder=3)
        ax.add_patch(sphere)

        # Highlight (upper-left white dot for 3D shading effect)
        hl = Circle((cx - r * 0.3, cy + r * 0.3),
                     r * 0.25,
                     facecolor='white', edgecolor='none',
                     alpha=0.4, zorder=4)
        ax.add_patch(hl)

        # Element label (C omitted, others white bold)
        if anum in LABEL_ATOMS:
            ax.text(cx, cy, LABEL_ATOMS[anum],
                    ha='center', va='center', fontsize=8,
                    fontweight='bold', color='white', zorder=5)

    # Axes
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Colored border rectangle
    rect = plt.Rectangle((-1.15, -1.15), 2.3, 2.3,
                          fill=False, edgecolor=border_color,
                          linewidth=3, zorder=10)
    ax.add_patch(rect)

    # Title with prediction errors
    nm = title if len(title) <= 14 else title[:13] + '.'
    ax.set_title(nm, fontsize=10, pad=18)
    # Error line below title at 9pt
    ax.text(0.5, 1.02,
            r"$\epsilon_{\mathrm{GIN}}$" + f"={error_gin:.2f}  "
            r"$\epsilon_{\mathrm{SchNet}}$" + f"={error_schnet:.2f}",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=9)


# --- Build 2×4 grid ---
reps_for_grid = reps.copy()
fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
plt.subplots_adjust(hspace=0.35, wspace=0.15, bottom=0.10)

for i, (_, row) in enumerate(reps_for_grid.iterrows()):
    ax = axes.flat[i]
    bc = border_colors[row['category']]
    draw_ball_stick_2d(ax, row['smiles'], row['molecule'],
                       row['error_gin'], row['error_schnet'], bc)

# --- Bottom legends ---
# Category legend
cat_legend = [
    LegPatch(facecolor='white', edgecolor=C_SCH, linewidth=2.5,
             label='Best predicted'),
    LegPatch(facecolor='white', edgecolor=COLORS[4], linewidth=2.5,
             label='Medium'),
    LegPatch(facecolor='white', edgecolor=C_BAD, linewidth=2.5,
             label='Worst predicted'),
]
leg1 = fig.legend(handles=cat_legend, loc='lower center', ncol=3,
                  fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.025))

# Atom color legend
atom_legend = [
    LegPatch(facecolor=ATOM_COLORS[6],  edgecolor='#333', linewidth=0.5, label='C'),
    LegPatch(facecolor=ATOM_COLORS[7],  edgecolor='#333', linewidth=0.5, label='N'),
    LegPatch(facecolor=ATOM_COLORS[8],  edgecolor='#333', linewidth=0.5, label='O'),
    LegPatch(facecolor=ATOM_COLORS[16], edgecolor='#333', linewidth=0.5, label='S'),
    LegPatch(facecolor=ATOM_COLORS[9],  edgecolor='#333', linewidth=0.5, label='F'),
]
fig.legend(handles=atom_legend, loc='lower center', ncol=5,
           fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.005))

savefig(fig, 'fig16_representative_molecules')

# ============ Summary statistics ============
print("\n" + "=" * 60)
print("Summary statistics (all 53 molecules):")
print(f"  GIN  MAE = {error_gin.mean():.4f} eV,  R² = {1 - np.sum((y_true - y_pred_gin)**2) / np.sum((y_true - y_true.mean())**2):.4f}")
print(f"  SchNet MAE = {error_schnet.mean():.4f} eV,  R² = {1 - np.sum((y_true - y_pred_schnet)**2) / np.sum((y_true - y_true.mean())**2):.4f}")
print(f"\n  SchNet better on {np.sum(error_schnet < error_gin)}/{n} molecules ({100*np.sum(error_schnet < error_gin)/n:.1f}%)")

# Correlation with rotatable bonds
from scipy import stats
r_gin, p_gin = stats.pearsonr(n_rotatable, error_gin)
r_sch, p_sch = stats.pearsonr(n_rotatable, error_schnet)
print(f"\n  Rotatable bonds vs error correlation:")
print(f"    GIN:    r={r_gin:.3f}, p={p_gin:.4f}")
print(f"    SchNet: r={r_sch:.3f}, p={p_sch:.4f}")
print("=" * 60)
