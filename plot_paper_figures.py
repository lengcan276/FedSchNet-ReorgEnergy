"""
Generate all paper figures in JCIM style.
Outputs PDF (vector) + PNG (300 dpi) to results/figures/.

Figure list (9 figures):
  fig2_dataset_distribution — seaborn histplot + violin
  fig4_tsne                 — t-SNE with 2D molecule insets
  fig5_kan_vs_mlp           — KAN vs MLP bar chart
  fig6_strategies_ssl       — Combined: federation strategies + SSL methods
  fig7_physics_ml_fusion    — Physics mechanism + ML performance
  fig8_waterfall            — Client D federated gain
  fig9_ablation_combined    — 2x2 ablation subplots
  fig13_gin_vs_schnet       — GIN vs SchNet comparison
  fig14_scatter             — Predicted vs true scatter
"""

import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Patch, ConnectionPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

# ============ Paths ============
PROJECT_ROOT = Path(__file__).resolve().parent
TABLES_DIR = PROJECT_ROOT / 'results' / 'tables'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OLD_RESULTS_DIR = PROJECT_ROOT.parent / 'federated_final' / 'results'
DATA_DIR = PROJECT_ROOT / 'data'

# ============ Font setup ============
available_fonts = set(f.name for f in fm.fontManager.ttflist)
SERIF_FONT = None
for candidate in ['Times New Roman', 'STIXGeneral', 'DejaVu Serif',
                   'Liberation Serif', 'FreeSerif']:
    if candidate in available_fonts:
        SERIF_FONT = candidate
        break

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [SERIF_FONT] if SERIF_FONT else ['DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'lines.linewidth': 1.5,
})
print(f"Using serif font: {SERIF_FONT or 'DejaVu Serif (default)'}")

# ============ Color scheme ============
COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974']
V_BLUE = COLORS[0]
V_GREEN = COLORS[1]
V_RED = COLORS[2]
V_PURPLE = COLORS[3]
V_GOLD = COLORS[4]

C_A = V_BLUE
C_B = V_GREEN
C_C = V_GOLD
C_D = V_RED

# Thin white stroke for external (floating) annotations
STROKE = [PathEffects.withStroke(linewidth=1.5, foreground='white')]


# ============ Helpers ============

def savefig(fig, name):
    for fmt in ['pdf', 'png']:
        fig.savefig(FIGURES_DIR / f'{name}.{fmt}', format=fmt,
                    dpi=300 if fmt == 'png' else None,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {name}.pdf + {name}.png")


def parse_ms(val_str):
    if val_str is None or (isinstance(val_str, float) and np.isnan(val_str)):
        return None, None
    s = str(val_str).strip()
    if not s:
        return None, None
    for sep in ['±', '+/-']:
        if sep in s:
            parts = s.split(sep)
            return float(parts[0]), float(parts[1])
    try:
        return float(s), 0.0
    except ValueError:
        return None, None


def load_summary():
    return pd.read_csv(TABLES_DIR / 'all_experiments_summary.csv')


def load_json(fn):
    p = OLD_RESULTS_DIR / fn
    if not p.exists():
        print(f"  WARNING: {p} not found")
        return {}
    with open(p) as f:
        return json.load(f)


def annotate_float(ax, x, y, text, fontsize=8, color='#333',
                   offset_pts=(0, 8)):
    """External floating annotation with thin white stroke."""
    txt = ax.annotate(text, xy=(x, y), xytext=offset_pts,
                      textcoords='offset points', ha='center', va='bottom',
                      fontsize=fontsize, fontweight='bold', color=color,
                      zorder=10)
    txt.set_path_effects(STROKE)
    return txt


# ============ Fig 2: Dataset distribution ============

def plot_fig2_dataset():
    print("\nFig2: Dataset distribution...")

    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.logger().setLevel(RDLogger.ERROR)

    df_pub = pd.read_csv(DATA_DIR / 'client_a_b' / 'public_reorg_energy_15210.csv')
    df_tadf = pd.read_csv(DATA_DIR / 'client_c' / 'hole_reorg_molecular.csv').dropna(subset=['smiles'])
    df_osc = pd.read_csv(DATA_DIR / 'client_d' / 'atahan_reorg_5876.csv').dropna(subset=['smiles'])

    arom = []
    for smi in df_pub['smiles']:
        mol = Chem.MolFromSmiles(smi)
        arom.append(rdMolDescriptors.CalcNumAromaticRings(mol) if mol else -1)
    df_pub = df_pub.copy()
    df_pub['n_arom'] = arom
    df_pub = df_pub[df_pub['n_arom'] >= 0]

    vals_a = df_pub[df_pub['n_arom'] == 0]['reorg_energy_eV'].values
    vals_b = df_pub[df_pub['n_arom'] >= 1]['reorg_energy_eV'].values
    vals_c = df_osc['reorg_eV'].values
    vals_d = df_tadf['lambda_hole_eV'].values

    df_long = pd.concat([
        pd.DataFrame({'energy': vals_a, 'client': f'Client A (n={len(vals_a)})'}),
        pd.DataFrame({'energy': vals_b, 'client': f'Client B (n={len(vals_b)})'}),
        pd.DataFrame({'energy': vals_c, 'client': f'Client C (n={len(vals_c)})'}),
    ], ignore_index=True)

    df_violin = pd.concat([
        pd.DataFrame({'energy': vals_a, 'client': 'A'}),
        pd.DataFrame({'energy': vals_b, 'client': 'B'}),
        pd.DataFrame({'energy': vals_c, 'client': 'C'}),
        pd.DataFrame({'energy': vals_d, 'client': 'D'}),
    ], ignore_index=True)

    palette_hist = {
        f'Client A (n={len(vals_a)})': C_A,
        f'Client B (n={len(vals_b)})': C_B,
        f'Client C (n={len(vals_c)})': C_C,
    }
    palette_violin = {'A': C_A, 'B': C_B, 'C': C_C, 'D': C_D}

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))

    ax = axes[0]
    sns.histplot(data=df_long, x='energy', hue='client', palette=palette_hist,
                 element='step', fill=True, alpha=0.45, stat='density',
                 common_norm=False, bins=50, ax=ax, linewidth=1.0)
    if ax.get_legend():
        ax.get_legend().remove()

    sns.rugplot(data=pd.DataFrame({'energy': vals_d}), x='energy',
                color=C_D, height=0.06, ax=ax, linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Reorganization Energy (eV)')
    ax.set_ylabel('Density')

    legend_elements = [
        Patch(facecolor=C_A, alpha=0.5, label=f'Client A (n={len(vals_a)})'),
        Patch(facecolor=C_B, alpha=0.5, label=f'Client B (n={len(vals_b)})'),
        Patch(facecolor=C_C, alpha=0.5, label=f'Client C (n={len(vals_c)})'),
        Patch(facecolor=C_D, alpha=0.7, label=f'Client D (n={len(vals_d)})'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='upper right',
              frameon=True, framealpha=0.95)

    t = ax.text(0.03, 0.95, '(a)', transform=ax.transAxes, fontsize=10,
                fontweight='bold', va='top')
    t.set_path_effects(STROKE)

    ax = axes[1]
    sns.violinplot(data=df_violin, x='client', y='energy', hue='client',
                   palette=palette_violin, legend=False,
                   inner=None, linewidth=0.8, cut=0, density_norm='width',
                   ax=ax, saturation=0.9)

    ax.set_xlabel('')
    ax.set_ylabel('Reorganization Energy (eV)')
    t = ax.text(0.03, 0.95, '(b)', transform=ax.transAxes, fontsize=10,
                fontweight='bold', va='top')
    t.set_path_effects(STROKE)

    fig.tight_layout()
    savefig(fig, 'fig2_dataset_distribution')


# ============ Fig 4: t-SNE with molecular structure insets ============

def plot_fig4_tsne():
    print("\nFig4: t-SNE with molecule insets...")

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw
    RDLogger.logger().setLevel(RDLogger.ERROR)

    cache_path = TABLES_DIR / 'tsne_coordinates.csv'
    if cache_path.exists():
        df_check = pd.read_csv(cache_path)
        if 'Client D' not in df_check['client'].values:
            cache_path.unlink()

    if cache_path.exists():
        df_tsne = pd.read_csv(cache_path)
    else:
        from rdkit.Chem import AllChem, rdMolDescriptors
        from sklearn.manifold import TSNE

        df_pub = pd.read_csv(DATA_DIR / 'client_a_b' / 'public_reorg_energy_15210.csv')
        df_hole = pd.read_csv(DATA_DIR / 'client_c' / 'hole_reorg_molecular.csv').dropna(subset=['smiles'])
        df_osc = pd.read_csv(DATA_DIR / 'client_d' / 'atahan_reorg_5876.csv').dropna(subset=['smiles'])

        arom = []
        for smi in df_pub['smiles']:
            mol = Chem.MolFromSmiles(smi)
            arom.append(rdMolDescriptors.CalcNumAromaticRings(mol) if mol else -1)
        df_pub = df_pub.copy()
        df_pub['n_arom'] = arom
        df_pub = df_pub[df_pub['n_arom'] >= 0]

        np.random.seed(42)
        n_s = 500
        sm_a = df_pub[df_pub['n_arom'] == 0]['smiles'].sample(n=n_s, random_state=42).tolist()
        sm_b = df_pub[df_pub['n_arom'] >= 1]['smiles'].sample(n=n_s, random_state=42).tolist()
        sm_d = df_hole['smiles'].tolist()
        sm_c = df_osc['smiles'].sample(n=n_s, random_state=42).tolist()

        all_smi = sm_a + sm_b + sm_c + sm_d
        labels = (['Client A'] * len(sm_a) + ['Client B'] * len(sm_b) +
                  ['Client C'] * len(sm_c) + ['Client D'] * len(sm_d))

        fps, vlabels, vsmi = [], [], []
        for smi, lab in zip(all_smi, labels):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(np.array(fp))
            vlabels.append(lab)
            vsmi.append(smi)

        X = np.stack(fps)
        print(f"  Computing t-SNE on {X.shape[0]} molecules...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        coords = tsne.fit_transform(X)
        df_tsne = pd.DataFrame({'tsne_1': coords[:, 0], 'tsne_2': coords[:, 1],
                                'client': vlabels, 'smiles': vsmi})
        df_tsne.to_csv(cache_path, index=False)

    from scipy.spatial import ConvexHull

    # High-saturation palette for t-SNE
    tsne_colors = {
        'Client A': '#2166AC',  # deep blue
        'Client B': '#1B9E77',  # deep green
        'Client C': '#D95F02',  # deep orange
        'Client D': '#E31A1C',  # bright red
    }
    tsne_markers = {
        'Client A': 'o', 'Client B': 's', 'Client C': 'D', 'Client D': '*',
    }
    tsne_sizes = {
        'Client A': 25, 'Client B': 25, 'Client C': 25, 'Client D': 200,
    }

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.subplots_adjust(left=0.20, right=0.80, top=0.82, bottom=0.18)

    # --- Convex hull outlines (dashed, no fill) ---
    for cl in ['Client A', 'Client B', 'Client C', 'Client D']:
        sub = df_tsne[df_tsne['client'] == cl]
        pts = sub[['tsne_1', 'tsne_2']].values
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                verts = np.append(hull.vertices, hull.vertices[0])  # close loop
                ax.plot(pts[verts, 0], pts[verts, 1],
                        color=tsne_colors[cl], linestyle='--',
                        linewidth=1.0, alpha=0.5, zorder=0)
            except Exception:
                pass

    # --- Scatter points ---
    for cl in ['Client A', 'Client B', 'Client C', 'Client D']:
        sub = df_tsne[df_tsne['client'] == cl]
        zo = 5 if cl == 'Client D' else 2
        alph = 0.9 if cl == 'Client D' else 0.55
        ax.scatter(sub['tsne_1'], sub['tsne_2'],
                   c=[tsne_colors[cl]], marker=tsne_markers[cl],
                   s=tsne_sizes[cl], alpha=alph,
                   label=f'Client {cl[-1]} (n={len(sub)})',
                   zorder=zo, edgecolors='white',
                   linewidths=0.8 if cl == 'Client D' else 0.3)

    # Keep spines (border) but remove ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # --- Molecule insets: 2 per client, well-separated on 4 edges ---
    offset_map = {
        'Client A': [(-300, 80), (-300, -80)],    # far left
        'Client B': [(-80, 280), (80, 280)],       # far top
        'Client C': [(300, 80), (300, -80)],       # far right
        'Client D': [(-80, -280), (80, -280)],     # far bottom
    }
    # Short label for annotation next to each molecule image
    client_short = {
        'Client A': 'A', 'Client B': 'B', 'Client C': 'C', 'Client D': 'D',
    }

    for cl_name in ['Client A', 'Client B', 'Client C', 'Client D']:
        sub = df_tsne[df_tsne['client'] == cl_name].reset_index(drop=True)
        coords_arr = sub[['tsne_1', 'tsne_2']].values
        smiles_arr = sub['smiles'].values

        if cl_name == 'Client D':
            pick_indices = [0, min(25, len(sub) - 1)]
        else:
            center = coords_arr.mean(axis=0)
            dists = np.linalg.norm(coords_arr - center, axis=1)
            idx_center = np.argmin(dists)
            idx_mid = np.argsort(dists)[len(dists) // 2]
            pick_indices = [idx_center, idx_mid]

        for mol_i, idx in enumerate(pick_indices):
            smi = smiles_arr[idx]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            img = Draw.MolToImage(mol, size=(300, 300))
            imagebox = OffsetImage(np.array(img), zoom=0.6)
            data_xy = (coords_arr[idx, 0], coords_arr[idx, 1])
            xybox = offset_map[cl_name][mol_i]
            cl_color = tsne_colors[cl_name]

            ab = AnnotationBbox(
                imagebox, data_xy,
                xybox=xybox,
                xycoords='data',
                boxcoords='offset points',
                arrowprops=dict(arrowstyle='-', color=cl_color,
                                lw=0.8, linestyle='--'),
                bboxprops=dict(edgecolor='none', facecolor='white',
                               alpha=0.95),
                pad=0, zorder=1)
            ax.add_artist(ab)

            # Small client label next to molecule image
            # Offset the text slightly from the image center
            lbl = client_short[cl_name]
            # Place text above the molecule box
            txt_offset_x = xybox[0]
            txt_offset_y = xybox[1] + 70  # above image
            txt = ax.annotate(lbl, xy=data_xy,
                              xytext=(txt_offset_x, txt_offset_y),
                              textcoords='offset points',
                              ha='center', va='center',
                              fontsize=7, fontweight='bold',
                              color=cl_color, zorder=10)
            txt.set_path_effects(STROKE)

    # Legend in upper-right, no border
    ax.legend(loc='upper right', frameon=False, fontsize=10,
              markerscale=1.5, labelspacing=0.8)

    savefig(fig, 'fig4_tsne')


# ============ Fig 5: KAN vs MLP ============

def plot_fig5_kan_vs_mlp():
    print("\nFig5: KAN vs MLP...")

    df = load_summary()
    r2_mlp, std_mlp = parse_ms(df[df['exp_id'] == 'E1'].iloc[0]['R2_A'])
    r2_kan, std_kan = parse_ms(df[df['exp_id'] == 'E7'].iloc[0]['R2_A'])

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    x = [0, 1]
    h = [r2_mlp, r2_kan]
    e = [std_mlp, std_kan]
    bar_colors = [V_BLUE, V_RED]
    labels = ['GIN + MLP\n(E1)', 'GIN + KAN\n(E4)']

    for i in range(2):
        ax.bar(x[i], h[i], yerr=e[i], width=0.5, color=bar_colors[i],
               edgecolor='white', linewidth=0.8, capsize=3,
               error_kw={'linewidth': 1.0, 'color': '#333'})

    pct = (r2_kan - r2_mlp) / r2_mlp * 100
    ann = ax.annotate(f'+{pct:.1f}%',
                      xy=(1, h[1] + e[1] + 0.02),
                      xytext=(0.5, h[1] + e[1] + 0.10),
                      fontsize=10, ha='center', fontweight='bold', color=V_GREEN,
                      arrowprops=dict(arrowstyle='->', color=V_GREEN, lw=1.5))
    ann.set_path_effects(STROKE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('$R^2$ (Client A, 5-fold CV)')
    ax.set_ylim(0, 0.95)

    fig.tight_layout()
    savefig(fig, 'fig5_kan_vs_mlp')


# ============ Fig 6: Combined strategies + SSL ============

def plot_fig6_combined():
    print("\nFig6: Combined federation strategies + SSL methods...")

    df = load_summary()
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    clients = ['Client A', 'Client B', 'Client D']
    client_colors = [C_A, C_B, C_D]
    bw = 0.20

    ax = axes[0]
    strat_labels = ['Local', 'FedAvg\n(E6)', 'FedPer\n(E7)', 'FedPer+\nFedBN (E8)']
    n_strats = len(strat_labels)

    e_la = df[df['exp_id'] == 'E7'].iloc[0]
    e_ld = df[df['exp_id'] == 'E9'].iloc[0]
    e_fa = df[df['exp_id'] == 'E10'].iloc[0]
    e_fp = df[df['exp_id'] == 'E14'].iloc[0]
    e_fb = df[df['exp_id'] == 'E15'].iloc[0]

    def get_abd(row, local_only=None):
        ra, sa = parse_ms(row.get('R2_A'))
        rb, sb = parse_ms(row.get('R2_B'))
        rd, sd = parse_ms(row.get('R2_Dh'))
        vals = [ra, rb, rd]
        errs = [sa or 0, sb or 0, sd or 0]
        if local_only == 'A':
            vals[1] = np.nan; vals[2] = np.nan
        elif local_only == 'D':
            vals[0] = np.nan; vals[1] = np.nan
        return vals, errs

    va_local_a, ea_local_a = get_abd(e_la, local_only='A')
    va_local_d, ea_local_d = get_abd(e_ld, local_only='D')
    local_vals = [va_local_a[0], np.nan, va_local_d[2]]
    local_errs = [ea_local_a[0], 0, ea_local_d[2]]

    strat_data = {
        'Local': (local_vals, local_errs),
        'FedAvg': get_abd(e_fa),
        'FedPer': get_abd(e_fp),
        'FedPer+FedBN': get_abd(e_fb),
    }
    strat_keys = ['Local', 'FedAvg', 'FedPer', 'FedPer+FedBN']

    x = np.arange(n_strats)
    for j, (cl, col) in enumerate(zip(clients, client_colors)):
        vals = [strat_data[s][0][j] for s in strat_keys]
        errs = [strat_data[s][1][j] for s in strat_keys]
        valid_x, valid_v, valid_e = [], [], []
        for k in range(n_strats):
            if vals[k] is not None and not np.isnan(vals[k]):
                valid_x.append(x[k])
                valid_v.append(vals[k])
                valid_e.append(errs[k])
        offset = (j - len(clients) / 2 + 0.5) * bw
        ax.bar(np.array(valid_x) + offset, valid_v, bw, yerr=valid_e,
               color=col, edgecolor='white', linewidth=0.8, alpha=0.85,
               capsize=2, error_kw={'linewidth': 0.8}, label=cl)

    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=8)
    ax.set_ylabel('$R^2$')
    ax.set_ylim(-0.85, 1.1)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    t = ax.text(0.02, 0.95, '(a)', transform=ax.transAxes, fontsize=10,
                fontweight='bold', va='top')
    t.set_path_effects(STROKE)

    ax = axes[1]
    method_labels = ['AtomMask\n(E9)', 'EdgePred\n(E10)', 'GraphCL\n(E11)']
    old_ids = ['E16', 'E17', 'E18']
    col_keys = ['R2_A', 'R2_B', 'R2_Dh']

    x = np.arange(len(method_labels))
    for j, (cl, col_key, col) in enumerate(zip(clients, col_keys, client_colors)):
        vals, errs = [], []
        for eid in old_ids:
            row = df[df['exp_id'] == eid]
            if row.empty:
                vals.append(0); errs.append(0); continue
            v, s = parse_ms(row.iloc[0][col_key])
            vals.append(v if v is not None else 0)
            errs.append(s if s is not None else 0)
        offset = (j - len(clients) / 2 + 0.5) * bw
        ax.bar(x + offset, vals, bw, yerr=errs, color=col, edgecolor='white',
               linewidth=0.8, alpha=0.85, capsize=2,
               error_kw={'linewidth': 0.8}, label=cl)

    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=8)
    ax.set_ylabel('$R^2$')
    ax.set_ylim(-0.4, 1.1)
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    t = ax.text(0.02, 0.95, '(b)', transform=ax.transAxes, fontsize=10,
                fontweight='bold', va='top')
    t.set_path_effects(STROKE)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.97),
               ncol=3, fontsize=9, frameon=True, framealpha=0.95,
               handletextpad=0.3, columnspacing=1.0)
    savefig(fig, 'fig6_strategies_ssl')


# ============ Fig 7: Physics mechanism + ML performance fusion ============

def plot_fig7_physics_ml():
    print("\nFig7: Physics-ML fusion figure...")

    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    from PIL import Image
    RDLogger.logger().setLevel(RDLogger.ERROR)

    # Pick the largest TADF molecule (by MW) from Client D
    df_tadf = pd.read_csv(DATA_DIR / 'client_c' / 'hole_reorg_molecular.csv').dropna(subset=['smiles'])
    best_idx, best_mw = 0, 0
    for i, smi in enumerate(df_tadf['smiles']):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mw = sum(a.GetMass() for a in mol.GetAtoms())
            if mw > best_mw:
                best_mw = mw
                best_idx = i
    selected_smiles = df_tadf['smiles'].iloc[best_idx]
    selected_name = df_tadf['molecule'].iloc[best_idx]
    print(f"  Selected TADF molecule: {selected_name} ({selected_smiles}), MW={best_mw:.1f}")

    fig = plt.figure(figsize=(7, 4))
    gs = gridspec.GridSpec(2, 2, width_ratios=[55, 45], hspace=0.35, wspace=0.35)

    # ---- (a) 2D structure ----
    ax_2d = fig.add_subplot(gs[0, 0])
    mol_2d = Chem.MolFromSmiles(selected_smiles)
    if mol_2d:
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol_2d)
        drawer.FinishDrawing()
        png_data = drawer.GetDrawingText()
        img_2d = Image.open(io.BytesIO(png_data))
        ax_2d.imshow(img_2d)
    ax_2d.axis('off')
    t = ax_2d.text(0.02, 0.98, '(a) 2D Molecular Graph', transform=ax_2d.transAxes,
                   fontsize=9, fontweight='bold', va='top')
    t.set_path_effects(STROKE)
    ax_2d.text(0.02, -0.02, 'Topological connectivity only', transform=ax_2d.transAxes,
               fontsize=8, color='gray', va='top')

    # ---- (b) 3D ball-and-stick ----
    ax_3d = fig.add_subplot(gs[1, 0], projection='3d')
    mol_3d = Chem.AddHs(Chem.MolFromSmiles(selected_smiles))
    embedded = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDGv3())
    if embedded == 0:
        AllChem.MMFFOptimizeMolecule(mol_3d)
    conf = mol_3d.GetConformer()
    pos = conf.GetPositions()

    # Atom colors
    atom_colors = {6: '#808080', 7: '#3050F8', 8: '#FF0D0D',
                   16: '#FFFF30', 1: '#CCCCCC', 5: '#FFB5B5',
                   9: '#90E050', 17: '#1FF01F', 35: '#A62929'}

    # Draw bonds first
    rotatable_smarts = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    rot_matches = mol_3d.GetSubstructMatches(rotatable_smarts) if rotatable_smarts else []

    # Find the most central rotatable bond
    center_pos = pos.mean(axis=0)
    best_rot_bond = None
    best_rot_dist = float('inf')
    for match in rot_matches:
        mid = (pos[match[0]] + pos[match[1]]) / 2
        d = np.linalg.norm(mid - center_pos)
        if d < best_rot_dist:
            best_rot_dist = d
            best_rot_bond = match

    for bond in mol_3d.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        p1, p2 = pos[i], pos[j]
        is_rot = best_rot_bond and set([i, j]) == set(best_rot_bond)
        lw = 3.0 if is_rot else 1.5
        col = '#C44E52' if is_rot else '#505050'
        ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                   color=col, linewidth=lw, zorder=1)

    # Draw atoms
    for i, atom in enumerate(mol_3d.GetAtoms()):
        anum = atom.GetAtomicNum()
        sz = 30 if anum == 1 else 120
        col = atom_colors.get(anum, '#808080')
        ax_3d.scatter(pos[i, 0], pos[i, 1], pos[i, 2],
                      c=[col], s=sz, edgecolors='white', linewidths=0.3, zorder=2)

    # Label the rotatable bond
    if best_rot_bond:
        mid = (pos[best_rot_bond[0]] + pos[best_rot_bond[1]]) / 2
        ax_3d.text(mid[0] + 0.5, mid[1] + 0.5, mid[2] + 0.5,
                   'Free torsion', fontsize=8, color='#C44E52', fontweight='bold')

    ax_3d.view_init(elev=20, azim=45)
    ax_3d.set_axis_off()
    # Remove background panes
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor('none')
    ax_3d.yaxis.pane.set_edgecolor('none')
    ax_3d.zaxis.pane.set_edgecolor('none')

    t = ax_3d.text2D(0.02, 0.98, '(b) 3D Conformer', transform=ax_3d.transAxes,
                     fontsize=9, fontweight='bold', va='top')
    t.set_path_effects(STROKE)
    ax_3d.text2D(0.02, 0.02, 'Torsional degrees of freedom visible',
                 transform=ax_3d.transAxes, fontsize=8, color='gray', va='bottom')

    # ---- (c) Horizontal bar: representation upgrade path ----
    ax_c = fig.add_subplot(gs[:, 1])
    methods = ['Local 2D GIN\n(E5)', 'Federated 2D GIN\n(E8)', 'Federated 3D SchNet\n(E14)']
    r2_values = [-0.677, 0.028, 0.181]
    bar_colors = [V_RED, V_BLUE, V_GREEN]

    bars = ax_c.barh(range(len(methods)), r2_values, height=0.6,
                     color=bar_colors, edgecolor='white', linewidth=0.8, alpha=0.85)
    ax_c.set_yticks(range(len(methods)))
    ax_c.set_yticklabels(methods, fontsize=8)
    ax_c.set_xlabel('$R^2$ (Client D, hole $\\lambda$)')
    ax_c.axvline(x=0, color='black', linewidth=0.8, linestyle='--')

    # Federation gain arrow (between bar 0 and bar 1)
    ax_c.annotate('', xy=(r2_values[1], 1), xytext=(r2_values[0], 0),
                  arrowprops=dict(arrowstyle='<->', color=V_BLUE, lw=1.5))
    mid_x = (r2_values[0] + r2_values[1]) / 2
    txt = ax_c.text(mid_x, 0.5, 'Federation\ngain', fontsize=7,
                    ha='center', va='center', color=V_BLUE, fontweight='bold')
    txt.set_path_effects(STROKE)

    # 3D repr gain arrow (between bar 1 and bar 2)
    ax_c.annotate('', xy=(r2_values[2], 2), xytext=(r2_values[1], 1),
                  arrowprops=dict(arrowstyle='<->', color=V_GREEN, lw=1.5))
    mid_x2 = (r2_values[1] + r2_values[2]) / 2
    txt2 = ax_c.text(mid_x2, 1.5, '3D repr.\ngain', fontsize=7,
                     ha='center', va='center', color=V_GREEN, fontweight='bold')
    txt2.set_path_effects(STROKE)

    ax_c.set_xlim(-0.85, 0.35)
    t = ax_c.text(0.02, 0.98, '(c) Representation Bottleneck', transform=ax_c.transAxes,
                  fontsize=9, fontweight='bold', va='top')
    t.set_path_effects(STROKE)

    # ---- Connection arrow from (b) to (c) ----
    con = ConnectionPatch(
        xyA=(1.0, 0.5), xyB=(0.0, 0.85),
        coordsA=ax_3d.transAxes, coordsB=ax_c.transAxes,
        arrowstyle='->', color='gray', linestyle='--',
        linewidth=1.2, mutation_scale=15)
    fig.add_artist(con)

    fig.text(0.52, 0.18, '3D coordinates resolve\ntorsional blindness',
             fontsize=8, fontstyle='italic', color='gray', ha='center')

    savefig(fig, 'fig7_physics_ml_fusion')


# ============ Fig 8: Waterfall ============

def plot_fig8_waterfall():
    print("\nFig8: Client D waterfall...")

    stages = ['Local\n(E5)', 'FedAvg\n(E6)', 'FedPer\n(E7)',
              'FedPer+FedBN\n(E8)', 'SchNet\nFedAvg (E14)']
    r2 = [-0.6766, -0.5071, -0.1075, 0.0279, 0.1806]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    n = len(stages)
    x = np.arange(n)

    bar_colors = [V_GREEN if v >= 0 else V_RED for v in r2]
    ax.bar(x, r2, width=0.6, color=bar_colors, edgecolor='white',
           linewidth=0.8, alpha=0.85)

    for i in range(n - 1):
        bar_top = r2[i]
        ax.plot([x[i] + 0.3, x[i + 1] - 0.3], [bar_top, bar_top],
                color='grey', linestyle='--', linewidth=1.0, zorder=1)
        ax.plot([x[i + 1] - 0.3, x[i + 1] - 0.3], [bar_top, r2[i + 1]],
                color='grey', linestyle='--', linewidth=1.0, zorder=1)

        delta = r2[i + 1] - r2[i]
        sign = '+' if delta >= 0 else ''
        col = V_GREEN if delta > 0 else V_RED
        label_y = max(r2[i], r2[i + 1]) + 0.06
        txt = ax.text((x[i] + x[i + 1]) / 2, label_y,
                      f'{sign}{delta:.3f}', ha='center', fontsize=8,
                      color=col, fontweight='bold', fontstyle='italic')
        txt.set_path_effects(STROKE)

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel('$R^2$ (Client D, hole $\\lambda$, LOOCV)')
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='-')
    ax.set_ylim(-0.85, 0.48)

    fig.tight_layout()
    savefig(fig, 'fig8_waterfall')


# ============ Fig 9: Combined ablation 2x2 ============

def plot_fig9_ablation():
    print("\nFig9: Combined ablation (2x2)...")

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

    plot_colors = [V_BLUE, V_RED, V_GREEN, V_BLUE]
    markers = ['o', 's', 'D', '^']
    panel_labels = ['(a)', '(b)', '(c)', '(d)']

    def curve_plot(ax, xdata, ydata, color, marker, xlabel,
                   x_labels=None, panel='(a)'):
        ax.plot(xdata, ydata, marker=marker, linestyle='-', color=color,
                markersize=7, linewidth=2,
                markeredgecolor='white', markeredgewidth=1.0, zorder=5)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('$R^2$')

        for idx, (xv, yv) in enumerate(zip(xdata, ydata)):
            lbl = x_labels[idx] if x_labels else str(xv)
            if idx % 2 == 0:
                off_pts = (0, 10)
            else:
                off_pts = (0, -14)
            txt = ax.annotate(lbl, xy=(xv, yv), xytext=off_pts,
                              textcoords='offset points', ha='center',
                              va='bottom' if idx % 2 == 0 else 'top',
                              fontsize=7, fontweight='bold', color='#333',
                              zorder=10)
            txt.set_path_effects(STROKE)

        t = ax.text(0.04, 0.96, panel, transform=ax.transAxes, fontsize=12,
                    fontweight='bold', va='top')
        t.set_path_effects(STROKE)

        y_range = max(ydata) - min(ydata) if max(ydata) != min(ydata) else 0.1
        ymin, ymax = min(ydata), max(ydata)
        pad = max(y_range * 0.4, 0.12)
        ax.set_ylim(ymin - pad * 0.5, ymax + pad)

    df = pd.read_csv(TABLES_DIR / 'ablation_kan_grid.csv')
    grids = df['grid'].values
    curve_plot(axes[0, 0], grids, df['R2'].values,
               plot_colors[0], markers[0], 'KAN Grid Size',
               x_labels=[f'g={int(g)}' for g in grids], panel=panel_labels[0])
    axes[0, 0].set_xticks(grids)

    df = pd.read_csv(TABLES_DIR / 'ablation_ssl_rounds.csv')
    rounds = df['ssl_rounds'].values
    curve_plot(axes[0, 1], rounds, df['R2'].values,
               plot_colors[1], markers[1], 'SSL Pretraining Rounds',
               x_labels=[f'{int(r)}' for r in rounds], panel=panel_labels[1])
    axes[0, 1].set_xticks(rounds)

    df = pd.read_csv(TABLES_DIR / 'ablation_data_size.csv')
    pct = (df['ratio'] * 100).values
    curve_plot(axes[1, 0], pct, df['R2'].values,
               plot_colors[2], markers[2], 'Client D Data Fraction (%)',
               x_labels=[f'n={int(n)}' for n in df['n_molecules'].values],
               panel=panel_labels[2])
    axes[1, 0].set_xticks(pct)
    axes[1, 0].set_xticklabels([f'{int(p)}%' for p in pct])

    df = pd.read_csv(TABLES_DIR / 'ablation_comm_rounds.csv')
    cr = df['n_rounds'].values
    curve_plot(axes[1, 1], cr, df['R2'].values,
               plot_colors[3], markers[3], 'Communication Rounds',
               x_labels=[f'{int(r)}' for r in cr], panel=panel_labels[3])
    axes[1, 1].set_xticks(cr)

    fig.tight_layout()
    savefig(fig, 'fig9_ablation_combined')


# ============ Fig 13: GIN vs SchNet ============

def plot_fig13_gin_vs_schnet():
    print("\nFig13: GIN vs SchNet...")

    df = load_summary()
    e32 = df[df['exp_id'] == 'E32'].iloc[0]
    e51 = df[df['exp_id'] == 'E51'].iloc[0]

    clients = ['Client A', 'Client B', 'Client C', 'Client D']
    cols = ['R2_A', 'R2_B', 'R2_C', 'R2_Dh']

    r2g, sg, r2s, ss_ = [], [], [], []
    for col in cols:
        v, s = parse_ms(e32[col])
        r2g.append(v if v is not None else 0.0)
        sg.append(s if s is not None else 0.0)
        v, s = parse_ms(e51[col])
        r2s.append(v if v is not None else 0.0)
        ss_.append(s if s is not None else 0.0)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(clients))
    bw = 0.3

    ax.bar(x - bw / 2, r2g, bw, yerr=sg, label='2D GIN (E12)', color=V_BLUE,
           edgecolor='white', linewidth=0.8, alpha=0.85, capsize=3,
           error_kw={'linewidth': 0.8})
    ax.bar(x + bw / 2, r2s, bw, yerr=ss_, label='3D SchNet (E14)', color=V_GREEN,
           edgecolor='white', linewidth=0.8, alpha=0.85, capsize=3,
           error_kw={'linewidth': 0.8})

    delta_d = r2s[3] - r2g[3]
    y_top = max(r2s[3] + ss_[3], r2g[3] + sg[3]) + 0.06
    ann = ax.annotate(f'$\\Delta R^2 = +{delta_d:.3f}$',
                      xy=(3 + bw / 2, r2s[3] + ss_[3] + 0.03),
                      xytext=(3 + bw / 2, y_top + 0.10),
                      fontsize=9, fontweight='bold', color=V_RED, ha='center',
                      arrowprops=dict(arrowstyle='->', color=V_RED, lw=1.5))
    ann.set_path_effects(STROKE)

    ax.set_xticks(x)
    ax.set_xticklabels(clients)
    ax.set_ylabel('$R^2$')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, fontsize=9)
    ax.set_ylim(-0.2, 1.15)

    fig.tight_layout()
    savefig(fig, 'fig13_gin_vs_schnet')


# ============ Fig 14: Scatter ============

def plot_fig14_scatter():
    print("\nFig14: SchNet scatter plot...")

    results = load_json('batch8_schnet_full.json')
    if 'E51_FedAvg_SchNet' not in results:
        print("  SKIPPED: no data"); return

    loocv = results['E51_FedAvg_SchNet'].get('loocv_c_hole', {})
    yt = loocv.get('y_true')
    yp = loocv.get('y_pred')
    if yt is None or yp is None:
        print("  SKIPPED: no y_true/y_pred"); return

    yt, yp = np.array(yt), np.array(yp)
    print(f"  Client D hole lambda: n={len(yt)}, range=[{yt.min():.4f}, {yt.max():.4f}] eV")

    mae = np.mean(np.abs(yt - yp))
    r2 = 1 - np.sum((yt - yp) ** 2) / (np.sum((yt - yt.mean()) ** 2) + 1e-12)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    ax.scatter(yt, yp, c=[V_BLUE], s=80, alpha=0.7,
               edgecolors='white', linewidths=0.8, zorder=5)

    vmin = min(yt.min(), yp.min()) - 0.15
    vmax = max(yt.max(), yp.max()) + 0.15
    ax.plot([vmin, vmax], [vmin, vmax], color='black', linestyle='--',
            linewidth=0.8, alpha=0.6, zorder=1)

    ax.set_xlabel('True $\\lambda_{hole}$ (eV)')
    ax.set_ylabel('Predicted $\\lambda_{hole}$ (eV)')
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')

    textstr = f'$R^2 = {r2:.3f}$\nMAE = {mae:.3f} eV\nn = {len(yt)}'
    props = dict(boxstyle='round,pad=0.4', facecolor='white',
                 edgecolor='gray', alpha=0.85)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            va='top', bbox=props)

    ax.set_title('SchNet FedAvg (E14): Client D LOOCV', fontsize=10)
    fig.tight_layout()
    savefig(fig, 'fig14_scatter')


# ============ Main ============

def main():
    print("=" * 60)
    print("Generating paper figures (JCIM style, SCI submission)")
    print(f"Output: {FIGURES_DIR}")
    print("=" * 60)

    plot_fig2_dataset()
    plot_fig4_tsne()
    plot_fig5_kan_vs_mlp()
    plot_fig6_combined()
    plot_fig7_physics_ml()
    plot_fig8_waterfall()
    plot_fig9_ablation()
    plot_fig13_gin_vs_schnet()
    plot_fig14_scatter()

    print("\n" + "=" * 60)
    print("Generated files:")
    print("=" * 60)
    total = 0
    for f in sorted(FIGURES_DIR.iterdir()):
        sz = f.stat().st_size
        total += sz
        print(f"  {f.name:40s}  {sz / 1024:.1f} KB")
    print(f"  {'TOTAL':40s}  {total / 1024:.1f} KB")
    print("=" * 60)


if __name__ == '__main__':
    main()
