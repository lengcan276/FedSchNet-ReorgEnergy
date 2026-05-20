"""
Generate Fig 7: Physics-ML Fusion overview.
(a) 2D molecular graph  (b) 3D ball-stick  (c) Representation bottleneck barh
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Arc
import matplotlib.patheffects as pe
from io import BytesIO
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor

# ============================================================
SMILES = 'COC1=CC=C(OC)C1=C1C(C#N)=C1C#N'
OUTDIR = 'results/figures'

C_LOCAL = '#C62828'
C_FED2D = '#1565C0'
C_FED3D = '#2E7D32'
C_RIGID = '#DAA520'
C_RIGID_EDGE = '#B8860B'

ATOM_COLORS = {
    6:  '#4CAF50', 7:  '#1565C0', 8:  '#C62828',
    16: '#F9A825', 17: '#00897B', 9:  '#FF8F00', 1: '#BDBDBD',
}
ATOM_RADIUS = {6: 0.030, 7: 0.028, 8: 0.027, 16: 0.033, 17: 0.032, 9: 0.025, 1: 0.018}
BOND_WIDTH = 2.5

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
})


def generate_2d_image(smiles, size=(450, 320)):
    """Generate 2D depiction via RDKit Cairo drawer."""
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.5
    opts.padding = 0.12
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    img = Image.open(BytesIO(drawer.GetDrawingText())).convert('RGBA')
    return np.array(img)


def draw_ball_stick(ax, smiles):
    """Draw 2D-layout ball-and-stick with clear atom spheres."""
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()

    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    heavy_ids = {a.GetIdx() for a in heavy}

    # Raw coords for heavy atoms
    raw = {a.GetIdx(): (conf.GetAtomPosition(a.GetIdx()).x,
                        conf.GetAtomPosition(a.GetIdx()).y) for a in heavy}
    xs = [v[0] for v in raw.values()]
    ys = [v[1] for v in raw.values()]
    cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
    span = max(max(xs)-min(xs), max(ys)-min(ys))
    if span == 0:
        span = 1

    def norm(x, y):
        return 0.5 + (x - cx) / span * 0.80, 0.5 + (y - cy) / span * 0.80

    # Bonds
    rotatable_bonds = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i not in heavy_ids or j not in heavy_ids:
            continue
        x1, y1 = norm(*raw[i])
        x2, y2 = norm(*raw[j])
        bt = bond.GetBondTypeAsDouble()
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if bt == 2 and length > 0:
            nx_, ny_ = -dy/length*0.012, dx/length*0.012
            ax.plot([x1+nx_, x2+nx_], [y1+ny_, y2+ny_], 'k-', lw=BOND_WIDTH, zorder=1)
            ax.plot([x1-nx_, x2-nx_], [y1-ny_, y2-ny_], 'k-', lw=BOND_WIDTH, zorder=1)
        elif bt == 3 and length > 0:
            nx_, ny_ = -dy/length*0.015, dx/length*0.015
            ax.plot([x1, x2], [y1, y2], 'k-', lw=BOND_WIDTH, zorder=1)
            ax.plot([x1+nx_, x2+nx_], [y1+ny_, y2+ny_], 'k-', lw=BOND_WIDTH*0.7, zorder=1)
            ax.plot([x1-nx_, x2-nx_], [y1-ny_, y2-ny_], 'k-', lw=BOND_WIDTH*0.7, zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2], 'k-', lw=BOND_WIDTH, zorder=1)

        # Detect rotatable single bonds (non-ring, between heavy atoms)
        if bond.GetBondTypeAsDouble() == 1.0 and not bond.IsInRing():
            a1 = mol.GetAtomWithIdx(i)
            a2 = mol.GetAtomWithIdx(j)
            # Prefer inter-ring or ring-to-non-ring bonds
            if a1.IsInRing() or a2.IsInRing():
                rotatable_bonds.append((i, j))

    # Atoms
    for atom in heavy:
        idx = atom.GetIdx()
        anum = atom.GetAtomicNum()
        x, y = norm(*raw[idx])
        color = ATOM_COLORS.get(anum, '#9E9E9E')
        r = ATOM_RADIUS.get(anum, 0.027)
        circle = plt.Circle((x, y), r, color=color, ec='black', lw=0.8, zorder=3)
        ax.add_patch(circle)
        if anum != 6:
            ax.text(x, y, atom.GetSymbol(), ha='center', va='center',
                    fontsize=6, fontweight='bold', color='white', zorder=4)

    # "Free torsion" on first rotatable bond
    if rotatable_bonds:
        i, j = rotatable_bonds[0]
        x1, y1 = norm(*raw[i])
        x2, y2 = norm(*raw[j])
        mx, my = (x1+x2)/2, (y1+y2)/2
        angle = np.degrees(np.arctan2(y2-y1, x2-x1))
        arc = Arc((mx, my), 0.12, 0.12, angle=angle,
                  theta1=-80, theta2=80, color='#C62828', lw=2.5, zorder=5)
        ax.add_patch(arc)
        # Place label above or below, whichever has more room
        ly = my + 0.08 if my < 0.5 else my - 0.08
        ax.text(mx, ly, 'Free torsion', ha='center', va='center',
                fontsize=8, fontweight='bold', color='#C62828', zorder=5,
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

    ax.set_xlim(0.02, 0.98)
    ax.set_ylim(0.02, 0.98)
    ax.set_aspect('equal')
    ax.axis('off')


def draw_barh(ax):
    """Horizontal bar chart: 4-bar representation bottleneck story."""
    labels = [
        'Local 2D GIN\n(E5, n=53)',
        'Fed. 2D GIN\n(E8, n=53)',
        'Fed. 3D SchNet\n(E14, n=53)',
        'Fed. 3D SchNet\n(Rigid D\u2013A, n=13)',
    ]
    values = [-0.677, 0.028, 0.181, 0.852]
    colors = [C_LOCAL, C_FED2D, C_FED3D, C_RIGID]
    edge_colors = ['black', 'black', 'black', C_RIGID_EDGE]
    edge_widths = [0.8, 0.8, 0.8, 1.5]

    y_pos = np.arange(len(labels))

    # Draw bars individually for per-bar edge control
    for i in range(len(values)):
        ax.barh(y_pos[i], values[i], height=0.50, color=colors[i],
                edgecolor=edge_colors[i], linewidth=edge_widths[i], zorder=3)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.0, zorder=2)

    # R² value labels
    # Bar 0: white text inside the wide red bar
    ax.text(-0.34, 0, 'R²=\u22120.677', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')
    # Bar 1: outside right
    ax.text(0.028 + 0.03, 1, 'R²=+0.028', ha='left', va='center',
            fontsize=9, fontweight='bold', color=C_FED2D)
    # Bar 2: outside right
    ax.text(0.181 + 0.03, 2, 'R²=+0.181', ha='left', va='center',
            fontsize=9, fontweight='bold', color=C_FED3D)
    # Bar 3: outside right
    ax.text(0.852 + 0.03, 3, 'R²=+0.852', ha='left', va='center',
            fontsize=9, fontweight='bold', color=C_RIGID_EDGE)

    # --- Gain annotations with arc arrows ---
    # 1→2: Federation gain
    ax.annotate('', xy=(-0.02, 0.72), xytext=(-0.55, 0.28),
                arrowprops=dict(arrowstyle='->', color=C_FED2D, lw=1.2,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(-0.30, 0.50, 'Federation\ngain', ha='center', va='center',
            fontsize=8, color=C_FED2D, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))

    # 2→3: 3D representation gain
    ax.annotate('', xy=(0.15, 1.72), xytext=(0.028, 1.28),
                arrowprops=dict(arrowstyle='->', color=C_FED3D, lw=1.2,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(0.10, 1.50, '3D repr.\ngain', ha='center', va='center',
            fontsize=8, color=C_FED3D, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))

    # 3→4: Conformational fidelity gain
    ax.annotate('', xy=(0.70, 2.72), xytext=(0.181, 2.28),
                arrowprops=dict(arrowstyle='->', color=C_RIGID_EDGE, lw=1.2,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(0.44, 2.50, 'Conformational\nfidelity gain', ha='center', va='center',
            fontsize=8, color=C_RIGID_EDGE, fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('R² (Client D, λ_hole)', fontsize=9)
    ax.set_xlim(-0.80, 1.05)
    ax.set_ylim(-0.5, 3.5)
    ax.grid(axis='x', alpha=0.25, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ============================================================
# Main
# ============================================================
fig = plt.figure(figsize=(8.5, 5.5))
gs = GridSpec(2, 2, width_ratios=[0.78, 1.22], height_ratios=[1, 1],
              wspace=0.50, hspace=0.40)

# (a) 3D ball-stick — top left (aligns with 3D SchNet bars on right)
ax_a = fig.add_subplot(gs[0, 0])
draw_ball_stick(ax_a, SMILES)
ax_a.set_title('(a) 3D Conformer', fontsize=11, fontweight='bold', pad=4, loc='left')
ax_a.text(0.5, -0.04, 'Torsional degrees of freedom visible',
          transform=ax_a.transAxes, ha='center', va='top', fontsize=8, color='#333333')

# (b) 2D molecule — bottom left (aligns with 2D GIN bars on right)
ax_b = fig.add_subplot(gs[1, 0])
img_2d = generate_2d_image(SMILES, size=(450, 320))
ax_b.imshow(img_2d)
ax_b.axis('off')
ax_b.set_title('(b) 2D Molecular Graph', fontsize=11, fontweight='bold', pad=4, loc='left')
ax_b.text(0.5, -0.04, 'Topological connectivity only \u2014 no spatial information',
          transform=ax_b.transAxes, ha='center', va='top', fontsize=8, color='#333333')

# (c) Bar chart
ax_c = fig.add_subplot(gs[:, 1])
draw_barh(ax_c)
ax_c.set_title('(c) Representation Bottleneck', fontsize=11, fontweight='bold', pad=6, loc='left')

plt.savefig(f'{OUTDIR}/fig7_physics_ml_fusion.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f'{OUTDIR}/fig7_physics_ml_fusion.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print('Saved fig7_physics_ml_fusion.pdf/.png')
plt.close()
