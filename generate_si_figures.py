"""
Generate SI figures: ablation panels, centralized comparison, per-molecule errors.
All data read from actual CSV files.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLOR = '#4C72B0'
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})


def read_csv(filename):
    path = RESULTS_DIR / filename
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


# ================================================================
#  Figure S1: 2x2 Ablation Details
# ================================================================
def fig_s1_ablation():
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    fig.subplots_adjust(hspace=0.38, wspace=0.35)

    # (a) KAN grid
    ax = axes[0, 0]
    data = read_csv('ablation_kan_grid.csv')
    x = [int(r['grid']) for r in data]
    y = [float(r['R2']) for r in data]
    ax.plot(x, y, 'o-', color=COLOR, markersize=6, linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    for xi, yi in zip(x, y):
        offset = 0.015 if yi < 0 else -0.02
        ax.annotate(f'{yi:.3f}', (xi, yi), textcoords='offset points',
                    xytext=(0, -14 if yi > -0.1 else 8), fontsize=7, ha='center')
    ax.set_xlabel('Grid Size')
    ax.set_ylabel('R² (Client D hole)')
    ax.set_xticks(x)
    ax.text(0.05, 0.95, '(a)', transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top')

    # (b) SSL rounds
    ax = axes[0, 1]
    data = read_csv('ablation_ssl_rounds.csv')
    x = [int(r['ssl_rounds']) for r in data]
    y = [float(r['R2']) for r in data]
    ax.plot(x, y, 'o-', color=COLOR, markersize=6, linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    for xi, yi in zip(x, y):
        yt = -14 if yi > min(y) + 0.02 else 8
        ax.annotate(f'{yi:.3f}', (xi, yi), textcoords='offset points',
                    xytext=(0, yt), fontsize=7, ha='center')
    ax.set_xlabel('SSL Pre-training Rounds')
    ax.set_ylabel('R² (Client D hole)')
    ax.set_xticks(x)
    ax.text(0.05, 0.95, '(b)', transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top')

    # (c) Data fraction
    ax = axes[1, 0]
    data = read_csv('ablation_data_size.csv')
    x = [float(r['ratio']) * 100 for r in data]
    y = [float(r['R2']) for r in data]
    ax.plot(x, y, 'o-', color=COLOR, markersize=6, linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    for xi, yi in zip(x, y):
        yt = -14 if yi > -1.0 else 8
        ax.annotate(f'{yi:.3f}', (xi, yi), textcoords='offset points',
                    xytext=(0, yt), fontsize=7, ha='center')
    ax.set_xlabel('Data Fraction (%)')
    ax.set_ylabel('R² (Client D hole)')
    ax.set_xticks(x)
    ax.text(0.05, 0.95, '(c)', transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top')

    # (d) Communication rounds
    ax = axes[1, 1]
    data = read_csv('ablation_comm_rounds.csv')
    x = [int(r['n_rounds']) for r in data]
    y = [float(r['R2']) for r in data]
    ax.plot(x, y, 'o-', color=COLOR, markersize=6, linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    for xi, yi in zip(x, y):
        yt = -14 if yi > min(y) + 0.02 else 8
        ax.annotate(f'{yi:.3f}', (xi, yi), textcoords='offset points',
                    xytext=(0, yt), fontsize=7, ha='center')
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('R² (Client D hole)')
    ax.set_xticks(x)
    ax.text(0.05, 0.95, '(d)', transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='top')

    for row in axes:
        for ax in row:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for ext in ['pdf', 'png']:
        fig.savefig(FIGURES_DIR / f'figS1_ablation_details.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  figS1_ablation_details saved')


# ================================================================
#  Figure S5: Centralized vs Federated vs Local
# ================================================================
def fig_s5_centralized():
    # Read data
    cent = read_csv('centralized_upper_bound.csv')
    exp = read_csv('all_experiments_summary.csv')

    # Build lookup
    exp_dict = {r['exp_id']: r for r in exp}

    # Local SchNet+KAN = E50, Federated SchNet+KAN = E52 (FedPer), Centralized from cent
    local_hole = float(exp_dict['E50']['R2_Dh'])
    fed_hole = float(exp_dict['E52']['R2_Dh'])
    cent_hole = float(cent[0]['R2'])  # hole row

    local_trip = float(exp_dict['E50']['R2_Dt'])
    fed_trip = float(exp_dict['E52']['R2_Dt'])
    cent_trip = float(cent[1]['R2'])  # triplet row

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    fig.subplots_adjust(wspace=0.5)

    methods = ['Local', 'Federated\n(FedPer)', 'Centralized']
    colors = ['#999999', '#2ca02c', '#d62728']

    # Hole
    vals_h = [local_hole, fed_hole, cent_hole]
    bars = ax1.barh(methods, vals_h, color=colors, height=0.5, edgecolor='white')
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    for bar, v in zip(bars, vals_h):
        xpos = v + 0.03 if v >= 0 else v - 0.03
        ha = 'left' if v >= 0 else 'right'
        ax1.text(xpos, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
                 va='center', ha=ha, fontsize=9)
    ax1.set_xlabel('R²')
    ax1.set_title('Client D $\\lambda_{hole}$', fontsize=11)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(min(vals_h) - 0.15, max(vals_h) + 0.15)

    # Triplet
    vals_t = [local_trip, fed_trip, cent_trip]
    bars = ax2.barh(methods, vals_t, color=colors, height=0.5, edgecolor='white')
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    for bar, v in zip(bars, vals_t):
        xpos = v + 0.03 if v >= 0 else v - 0.03
        ha = 'left' if v >= 0 else 'right'
        ax2.text(xpos, bar.get_y() + bar.get_height()/2, f'{v:.3f}',
                 va='center', ha=ha, fontsize=9)
    ax2.set_xlabel('R²')
    ax2.set_title('Client D $\\lambda_{triplet}$', fontsize=11)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(min(vals_t) - 0.15, max(vals_t) + 0.15)

    for ext in ['pdf', 'png']:
        fig.savefig(FIGURES_DIR / f'figS5_centralized_comparison.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  figS5_centralized_comparison saved')


# ================================================================
#  Figure S6: Per-molecule errors
# ================================================================
def fig_s6_per_molecule():
    data = read_csv('case_study_molecules.csv')

    molecules = [r['molecule'] for r in data]
    y_true = np.array([float(r['y_true']) for r in data])
    err_gin = np.array([float(r['error_gin']) for r in data])
    err_schnet = np.array([float(r['error_schnet']) for r in data])

    # Sort by SchNet error ascending
    order = np.argsort(err_schnet)
    molecules_s = [molecules[i] for i in order]
    err_gin_s = err_gin[order]
    err_schnet_s = err_schnet[order]

    n = len(molecules_s)
    x = np.arange(n)

    mae_schnet = np.mean(err_schnet)
    schnet_wins = np.sum(err_schnet < err_gin)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), gridspec_kw={'height_ratios': [1, 1]})
    fig.subplots_adjust(hspace=0.35)

    # (a) SchNet error bar chart with color gradient
    norm = Normalize(vmin=err_schnet_s.min(), vmax=err_schnet_s.max())
    cmap = plt.cm.RdYlGn_r
    colors = [cmap(norm(v)) for v in err_schnet_s]

    ax1.bar(x, err_schnet_s, color=colors, width=0.8, edgecolor='none')
    ax1.axhline(mae_schnet, color='gray', linestyle='--', linewidth=1,
                label=f'MAE = {mae_schnet:.3f} eV')

    # Label best 3 and worst 3
    for idx in list(range(3)) + list(range(n-3, n)):
        name = molecules_s[idx]
        if len(name) > 18:
            name = name[:16] + '..'
        yoff = 0.03
        ax1.text(idx, err_schnet_s[idx] + yoff, name, fontsize=5,
                 rotation=50, ha='left', va='bottom')

    ax1.set_ylabel('|Error| (eV)')
    ax1.set_xlabel('Molecule index (sorted by SchNet error)')
    ax1.legend(fontsize=8, loc='center right')
    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_title('(a) SchNet absolute errors (sorted)', loc='left',
                  fontsize=10, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # (b) GIN vs SchNet comparison
    ax2.fill_between(x, 0, np.maximum(err_gin_s, err_schnet_s),
                     where=err_schnet_s <= err_gin_s,
                     alpha=0.15, color='green', label=None)
    ax2.fill_between(x, 0, np.maximum(err_gin_s, err_schnet_s),
                     where=err_schnet_s > err_gin_s,
                     alpha=0.15, color='blue', label=None)

    ax2.plot(x, err_gin_s, '-', color='#4C72B0', linewidth=1.2, alpha=0.8, label='GIN')
    ax2.plot(x, err_schnet_s, '-', color='#2ca02c', linewidth=1.2, alpha=0.8, label='SchNet')

    ax2.set_ylabel('|Error| (eV)')
    ax2.set_xlabel('Molecule index (sorted by SchNet error)')
    ax2.set_xlim(-0.5, n - 0.5)

    pct = schnet_wins / n * 100
    ax2.text(0.98, 0.95, f'SchNet wins: {schnet_wins}/{n} ({pct:.1f}%)',
             transform=ax2.transAxes, fontsize=9, ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))

    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_title('(b) GIN vs. SchNet per-molecule errors', loc='left',
                  fontsize=10, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    for ext in ['pdf', 'png']:
        fig.savefig(FIGURES_DIR / f'figS6_per_molecule_errors.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  figS6_per_molecule_errors saved')


# ================================================================
#  MAIN
# ================================================================
if __name__ == '__main__':
    print('Generating SI figures...')
    fig_s1_ablation()
    fig_s5_centralized()
    fig_s6_per_molecule()
    print('All SI figures generated.')
