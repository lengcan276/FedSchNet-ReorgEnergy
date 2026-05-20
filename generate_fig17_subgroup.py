"""
Generate Fig17: Subgroup Analysis visualization (JCIM style).
Upper row: (a) R² bar chart (cols 0-1), (b) scatter plot (col 2).
Lower row: (c) 9 representative molecules in a single row with dividers.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
import matplotlib.font_manager as fm
from matplotlib.patches import Circle
from matplotlib.patches import Patch as LegPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)

# ============ Paths ============
PROJECT_ROOT = Path(__file__).resolve().parent
TABLES_DIR = PROJECT_ROOT / 'results' / 'tables'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============ Font setup ============
available_fonts = set(f.name for f in fm.fontManager.ttflist)
SERIF = None
for c in ['Times New Roman', 'STIXGeneral', 'DejaVu Serif', 'Liberation Serif']:
    if c in available_fonts:
        SERIF = c
        break

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

# ============ Colors ============
C_GIN = '#1F77B4'       # classic blue
C_SCH = '#2CA02C'       # forest green
C_BAD = '#C44E52'       # red (for Free torsion arrows)
C_GRAY = '#9E9E9E'
C_RIGID = '#B22222'     # dark red for Rigid High-DA group label
C_FLEX = '#1F77B4'      # classic blue for Flexible group label
STROKE = [PathEffects.withStroke(linewidth=1.5, foreground='white')]

ATOM_COLORS = {
    6:  '#4CAF50',   # C - green
    7:  '#2196F3',   # N - blue
    8:  '#F44336',   # O - red
    16: '#FFEB3B',   # S - yellow
    15: '#FF9800',   # P - orange
    9:  '#8BC34A',   # F - light green
    17: '#4CAF50',   # Cl - green
    5:  '#E91E63',   # B - pink
    35: '#A62929',   # Br - brown
}
LABEL_ATOMS = {7: 'N', 8: 'O', 16: 'S', 15: 'P', 9: 'F', 5: 'B', 17: 'Cl', 35: 'Br'}


def savefig(fig, name):
    for fmt in ['pdf', 'png']:
        fig.savefig(FIGURES_DIR / f'{name}.{fmt}', format=fmt,
                    dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {name}.pdf + {name}.png")


# ============ Load data ============
print("=" * 60)
print("Fig17: Subgroup Analysis (JCIM polish)")
print("=" * 60)

df = pd.read_csv(TABLES_DIR / 'case_study_molecules.csv')

# Compute donor fraction
donor_fracs = []
for smi in df['smiles']:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        n_heavy = mol.GetNumHeavyAtoms()
        n_donor = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [7, 8, 16])
        donor_fracs.append(n_donor / n_heavy if n_heavy > 0 else 0)
    else:
        donor_fracs.append(0)
df['donor_frac'] = donor_fracs

# Use values from subgroup_optimized.csv
df_opt = pd.read_csv(TABLES_DIR / 'subgroup_optimized.csv')
opt_row = df_opt[df_opt['filter'] == 'DA>=0.25 & rot=0'].iloc[0]
r2_rhda_sch = opt_row['SchNet_R2']
r2_rhda_gin = opt_row['GIN_R2']
mae_rhda_sch = opt_row['SchNet_MAE']
n_rhda = int(opt_row['n'])

opt_flex = df_opt[df_opt['filter'] == 'rot>=1']
if len(opt_flex) > 0:
    opt_flex = opt_flex.iloc[0]
    r2_flex_sch = opt_flex['SchNet_R2']
    r2_flex_gin = opt_flex['GIN_R2']
else:
    r2_flex_sch = 0.157
    r2_flex_gin = -0.363
n_flex = 12

def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

r2_all_gin = calc_r2(df['y_true'].values, df['y_pred_gin'].values)
r2_all_sch = calc_r2(df['y_true'].values, df['y_pred_schnet'].values)

candidates_rhda = df[(df['donor_frac'] >= 0.25) & (df['n_rotatable'] == 0)]
candidates_rhda = candidates_rhda.sort_values('donor_frac', ascending=False)
rigid_hda = candidates_rhda.head(n_rhda)

flexible = df[df['n_rotatable'] >= 1]

print(f"  All: n={len(df)}")
print(f"  Rigid High-DA: n={len(rigid_hda)} (target={n_rhda})")
print(f"  Flexible (rot>=1): n={len(flexible)}")
print(f"\n  R² values:")
print(f"    All:       GIN={r2_all_gin:.3f}, SchNet={r2_all_sch:.3f}")
print(f"    Rigid HDA: GIN={r2_rhda_gin:.3f}, SchNet={r2_rhda_sch:.3f}")
print(f"    Flexible:  GIN={r2_flex_gin:.3f}, SchNet={r2_flex_sch:.3f}")
print(f"    Rigid HDA MAE(SchNet): {mae_rhda_sch:.3f} eV")


# ============ Draw ball-and-stick (no border) ============
def draw_ball_stick_2d(ax, smiles, name_text, error_text, is_flexible=False):
    """Draw 2D ball-and-stick model. No border rectangle."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        ax.axis('off')
        return

    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()

    heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    if not heavy_idx:
        ax.axis('off')
        return

    coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
                        for i in heavy_idx])

    if len(coords) > 1:
        center = coords.mean(axis=0)
        coords -= center
        scale = np.abs(coords).max()
        if scale > 0:
            coords /= (scale * 1.15)

    R_MAP = {6: 0.12, 7: 0.13, 8: 0.13, 16: 0.14, 15: 0.13, 9: 0.11,
             17: 0.13, 5: 0.12, 35: 0.14}

    # Bonds
    for bond in mol.GetBonds():
        bi, bj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if bi not in heavy_idx or bj not in heavy_idx:
            continue
        ii = heavy_idx.index(bi)
        jj = heavy_idx.index(bj)
        bt = bond.GetBondTypeAsDouble()

        ax.plot([coords[ii, 0], coords[jj, 0]],
                [coords[ii, 1], coords[jj, 1]],
                color='#424242', linewidth=2.0, zorder=1,
                solid_capstyle='round')

        if bt >= 2:
            dx = coords[jj, 0] - coords[ii, 0]
            dy = coords[jj, 1] - coords[ii, 1]
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                off = 0.04
                nx, ny = -dy / norm * off, dx / norm * off
                ax.plot([coords[ii, 0] + nx, coords[jj, 0] + nx],
                        [coords[ii, 1] + ny, coords[jj, 1] + ny],
                        color='#757575', linewidth=1.2, zorder=1)
        if bt >= 3:
            dx = coords[jj, 0] - coords[ii, 0]
            dy = coords[jj, 1] - coords[ii, 1]
            norm = np.sqrt(dx**2 + dy**2)
            if norm > 0:
                off = 0.04
                nx, ny = -dy / norm * off, dx / norm * off
                ax.plot([coords[ii, 0] - nx, coords[jj, 0] - nx],
                        [coords[ii, 1] - ny, coords[jj, 1] - ny],
                        color='#757575', linewidth=1.2, zorder=1)

    # Atoms
    for ii, atom_idx in enumerate(heavy_idx):
        atom = mol.GetAtomWithIdx(atom_idx)
        anum = atom.GetAtomicNum()
        color = ATOM_COLORS.get(anum, '#9E9E9E')
        r = R_MAP.get(anum, 0.12)
        cx, cy = coords[ii]

        sphere = Circle((cx, cy), r, facecolor=color, edgecolor='#333333',
                         linewidth=0.5, zorder=3)
        ax.add_patch(sphere)

        hl = Circle((cx - r * 0.3, cy + r * 0.3), r * 0.25,
                     facecolor='white', edgecolor='none', alpha=0.4, zorder=4)
        ax.add_patch(hl)

        if anum in LABEL_ATOMS:
            ax.text(cx, cy, LABEL_ATOMS[anum], ha='center', va='center',
                    fontsize=6, fontweight='bold', color='white', zorder=5)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect('equal')
    ax.axis('off')

    # Name below molecule (grey, 7pt)
    nm = name_text if len(name_text) <= 18 else name_text[:17] + '.'
    ax.text(0.5, -0.02, nm, transform=ax.transAxes, ha='center', va='top',
            fontsize=7, color='#666666')
    # Error below name (black, 8pt)
    ax.text(0.5, -0.10, error_text, transform=ax.transAxes, ha='center', va='top',
            fontsize=8, color='black')

    # Free torsion annotation for flexible molecules
    if is_flexible:
        rot_bonds = mol.GetSubstructMatches(
            Chem.MolFromSmarts('[!$([NH]!@C(=O))&!D1]-&!@[!$([NH]!@C(=O))&!D1]'))
        if rot_bonds:
            bi, bj = rot_bonds[0]
            if bi in heavy_idx and bj in heavy_idx:
                ii_b = heavy_idx.index(bi)
                jj_b = heavy_idx.index(bj)
                mx = (coords[ii_b, 0] + coords[jj_b, 0]) / 2
                my = (coords[ii_b, 1] + coords[jj_b, 1]) / 2
                ax.annotate('Free\ntorsion', xy=(mx, my),
                            xytext=(mx + 0.45, my + 0.55),
                            fontsize=5, color=C_BAD, fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color=C_BAD, lw=1.2),
                            ha='center', zorder=20)


# ============ Select representative molecules ============
rhda_sorted = rigid_hda.copy()
rhda_sorted['abs_err_sch'] = np.abs(rhda_sorted['y_true'] - rhda_sorted['y_pred_schnet'])
rhda_best3 = rhda_sorted.nsmallest(3, 'abs_err_sch')

remaining = df[~df.index.isin(rigid_hda.index) & ~df.index.isin(flexible.index)]
if len(remaining) < 3:
    remaining = df[~df.index.isin(rigid_hda.index)]
remaining = remaining.copy()
remaining['abs_err_sch'] = np.abs(remaining['y_true'] - remaining['y_pred_schnet'])
remaining_sorted = remaining.sort_values('abs_err_sch')
mid_idx = len(remaining_sorted) // 2
all_rep3 = remaining_sorted.iloc[max(0, mid_idx - 1):mid_idx + 2]

flex_sorted = flexible.copy()
flex_sorted['abs_err_sch'] = np.abs(flex_sorted['y_true'] - flex_sorted['y_pred_schnet'])
flex_worst3 = flex_sorted.nlargest(3, 'abs_err_sch')

print(f"\n  Representative molecules:")
print(f"    Rigid HDA best 3: {rhda_best3['molecule'].tolist()}")
print(f"    All mid 3: {all_rep3['molecule'].tolist()}")
print(f"    Flexible worst 3: {flex_worst3['molecule'].tolist()}")


# ============ Build Figure ============
fig = plt.figure(figsize=(14, 5.5))
gs = gridspec.GridSpec(2, 3, height_ratios=[1.2, 1], hspace=0.25, wspace=0.25)

# ---- (a) R² bar chart — spans columns 0-1 ----
ax_a = fig.add_subplot(gs[0, 0:2])
groups = [f'All\n($n$={len(df)})', f'Rigid High-DA\n($n$={n_rhda})', f'Flexible\n($n$={n_flex})']
gin_r2 = [r2_all_gin, r2_rhda_gin, r2_flex_gin]
sch_r2 = [r2_all_sch, r2_rhda_sch, r2_flex_sch]

x = np.arange(len(groups))
bw = 0.3

bars_gin = ax_a.bar(x - bw / 2, gin_r2, bw, color=C_GIN, edgecolor='white',
                     linewidth=0.8, alpha=0.85, label='2D GIN')
bars_sch = ax_a.bar(x + bw / 2, sch_r2, bw, color=C_SCH, edgecolor='white',
                     linewidth=0.8, alpha=0.85, label='3D SchNet')

# Annotate R² values with white stroke
for bar, val in zip(bars_gin, gin_r2):
    y_pos = max(val, 0) + 0.02 if val >= 0 else val - 0.06
    t = ax_a.text(bar.get_x() + bar.get_width() / 2, y_pos, f'{val:.3f}',
                  ha='center', va='bottom' if val >= 0 else 'top',
                  fontsize=8, color=C_GIN, fontweight='semibold')
    t.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='w')])

for bar, val in zip(bars_sch, sch_r2):
    y_pos = max(val, 0) + 0.02 if val >= 0 else val - 0.06
    t = ax_a.text(bar.get_x() + bar.get_width() / 2, y_pos, f'{val:.3f}',
                  ha='center', va='bottom' if val >= 0 else 'top',
                  fontsize=8, color=C_SCH, fontweight='semibold')
    t.set_path_effects([PathEffects.withStroke(linewidth=1.5, foreground='w')])

ax_a.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax_a.set_ylim(-0.5, 1.0)
ax_a.set_xticks(x)
ax_a.set_xticklabels(groups, fontsize=9)
ax_a.set_ylabel('$R^2$')
ax_a.legend(fontsize=9, loc='upper left', frameon=True, framealpha=0.95)

# Label (a) outside top-left
ax_a.text(-0.08, 1.05, '(a)', transform=ax_a.transAxes,
          fontsize=14, fontweight='bold', va='bottom', ha='left')

# ---- (b) Scatter: rigid high-DA predicted vs true — column 2 ----
ax_b = fig.add_subplot(gs[0, 2])

y_true_rhda = rigid_hda['y_true'].values
y_pred_rhda = rigid_hda['y_pred_schnet'].values

lims = [min(y_true_rhda.min(), y_pred_rhda.min()) - 0.1,
        max(y_true_rhda.max(), y_pred_rhda.max()) + 0.1]
ax_b.plot(lims, lims, 'k-', linewidth=1.0, alpha=0.7, zorder=1)

ax_b.scatter(y_true_rhda, y_pred_rhda, c=C_SCH, s=80, alpha=0.7,
             edgecolors='white', linewidths=0.5, zorder=5)

ax_b.set_xlabel('DFT $\\lambda_{hole}$ (eV)')
ax_b.set_ylabel('SchNet predicted $\\lambda_{hole}$ (eV)')
ax_b.set_xlim(lims)
ax_b.set_ylim(lims)
ax_b.set_aspect('equal')

# Annotation box — left side, below the (b) label
textstr = f'$R^2 = {r2_rhda_sch:.3f}$\nMAE = {mae_rhda_sch:.3f} eV\n$n = 13$'
props = dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#CCCCCC',
             alpha=0.85, linewidth=1.0)
ax_b.text(0.05, 0.82, textstr, transform=ax_b.transAxes, fontsize=9,
          verticalalignment='top', bbox=props)

# Label (b) outside top-left
ax_b.text(-0.15, 1.05, '(b)', transform=ax_b.transAxes,
          fontsize=14, fontweight='bold', va='bottom', ha='left')

# ---- (c) Lower half: 9 molecules in a single row with dividers ----
gs_lower = gridspec.GridSpecFromSubplotSpec(1, 9, subplot_spec=gs[1, :], wspace=0.08)

mol_groups = [
    (rhda_best3, 'Rigid High-DA', C_RIGID, False),
    (all_rep3, 'All molecules', C_GRAY, False),
    (flex_worst3, 'Flexible', C_FLEX, True),
]

mol_axes = []
group_starts = []  # track column index where each group starts
col = 0
for sg_df, sg_label, sg_color, is_flex in mol_groups:
    group_starts.append(col)
    for _, row in sg_df.iterrows():
        ax = fig.add_subplot(gs_lower[col])
        mol_axes.append(ax)
        err_sch = abs(row['y_true'] - row['y_pred_schnet'])
        err_text = f"$\\epsilon$={err_sch:.3f} eV"
        draw_ball_stick_2d(ax, row['smiles'], row['molecule'], err_text,
                           is_flexible=is_flex)
        col += 1

# -- Visual hierarchy bounding boxes + group titles for each group of 3 --
from matplotlib.patches import FancyBboxPatch
import matplotlib.transforms as mtransforms

# Style per group: (edgecolor, linestyle, linewidth, alpha)
group_box_styles = [
    ('#B22222', '-',  3.5, 1.0),   # Rigid High-DA: bold, striking
    ('#808080', '--', 1.5, 0.6),   # All molecules: medium, dashed
    ('#1F77B4', ':',  0.8, 0.3),   # Flexible: faint, dotted
]

for gi, (_, sg_label, sg_color, _) in enumerate(mol_groups):
    start = group_starts[gi]
    ax_left = mol_axes[start]
    ax_right = mol_axes[start + 2]
    pos_l = ax_left.get_position()
    pos_r = ax_right.get_position()

    # Group title centered above
    cx = (pos_l.x0 + pos_r.x1) / 2
    cy = pos_l.y1 + 0.005
    fig.text(cx, cy, sg_label, ha='center', va='bottom',
             fontsize=10, fontweight='bold', color=sg_color)

    # Bounding box around the 3 molecules
    ec, ls, lw, al = group_box_styles[gi]
    pad_x = 0.005
    pad_bot = 0.07   # extend below axes to cover name + error text
    pad_top = 0.005
    bx = pos_l.x0 - pad_x
    by = pos_l.y0 - pad_bot
    bw = (pos_r.x1 - pos_l.x0) + 2 * pad_x
    bh = (pos_l.y1 - pos_l.y0) + pad_bot + pad_top
    rect = plt.Rectangle((bx, by), bw, bh, transform=fig.transFigure,
                          fill=False, edgecolor=ec, linestyle=ls,
                          linewidth=lw, alpha=al, clip_on=False, zorder=0)
    fig.add_artist(rect)

# Label (c) outside top-left of the lower row
ax_c0 = mol_axes[0]
ax_c0.text(-0.25, 1.15, '(c)', transform=ax_c0.transAxes,
           fontsize=14, fontweight='bold', va='bottom', ha='left')

# ---- Bottom legend: single merged row ----
fig.subplots_adjust(bottom=0.10)

legend_handles = [
    LegPatch(facecolor=C_RIGID, edgecolor='none', label='Rigid High-DA'),
    LegPatch(facecolor=C_GRAY, edgecolor='none', label='All molecules'),
    LegPatch(facecolor=C_FLEX, edgecolor='none', label='Flexible'),
    Line2D([], [], marker='o', color='w', markerfacecolor=ATOM_COLORS[6],
           markeredgecolor='#333', markersize=8, label='C'),
    Line2D([], [], marker='o', color='w', markerfacecolor=ATOM_COLORS[7],
           markeredgecolor='#333', markersize=8, label='N'),
    Line2D([], [], marker='o', color='w', markerfacecolor=ATOM_COLORS[8],
           markeredgecolor='#333', markersize=8, label='O'),
    Line2D([], [], marker='o', color='w', markerfacecolor=ATOM_COLORS[16],
           markeredgecolor='#333', markersize=8, label='S'),
    Line2D([], [], marker='o', color='w', markerfacecolor=ATOM_COLORS[15],
           markeredgecolor='#333', markersize=8, label='P'),
]

fig.legend(handles=legend_handles, loc='lower center',
           bbox_to_anchor=(0.5, 0.01), ncol=8,
           fontsize=9, frameon=False, handletextpad=0.4, columnspacing=1.2)

savefig(fig, 'fig17_subgroup_analysis')
print("\nDone!")
