"""Pre-verification for Client D -> Client C transfer learning.

Three CPU-only checks that must pass before committing GPU time to a D->C
supervised transfer experiment:

  V1  Label-protocol & distribution comparability (D vs C-hole / C-triplet)
  V2  Classical-ML strong baseline on Client C (Morgan2048 + RF / XGB / GP / KNN)
  V3  Chemical-space overlap C <-> D (Tanimoto + Murcko scaffolds + t-SNE highlight)

Outputs land in results/preverify/. Does not modify any existing results.

Usage
-----
    python experiments/preverify.py --v1
    python experiments/preverify.py --v2
    python experiments/preverify.py --v3
    python experiments/preverify.py --all      # default if no flag given
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.data_utils import (  # noqa: E402
    load_client_c_hole, load_client_c_triplet, load_client_d,
    load_public_data, split_public_by_aromaticity,
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

OUT_DIR = PROJECT_ROOT / 'results' / 'preverify'
TSNE_COORDS = PROJECT_ROOT / 'results' / 'tables' / 'tsne_coordinates.csv'

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
SEEDS = [42, 123, 456, 789, 1000]

# Atahan-Evrenk & Atalay, J. Phys. Chem. A 2019, 123, 7855-7863.
# Confirmed from the paper abstract via WebFetch (2026-05-08): B3LYP / 6-31G*.
# Vertical-vs-adiabatic, charge state, and conformer treatment are not stated
# in the abstract; treat as unknown.
ATAHAN_PROTOCOL = {
    'source': 'Atahan-Evrenk & Atalay, J. Phys. Chem. A 2019, 123, 7855-7863',
    'functional': 'B3LYP',
    'basis': '6-31G*',
    'library_size_paper': 5631,
    'library_size_project': 5876,
    'geometry': 'not stated in abstract',
    'charge_state': 'not stated in abstract (likely neutral->cation, hole)',
    'conformers': 'not stated in abstract',
}

# Client C protocol from data/client_c/reorganization_energy_summary.csv columns
# (`neutral_method` = RwB97XD, `cation_method` / `triplet_method` = UwB97XD,
#  `IP_adiabatic_eV`, `lambda1_S0_eV` / `lambda2_T1_eV` -> 4-point).
CLIENT_C_PROTOCOL = {
    'source': 'data/client_c/reorganization_energy_summary.csv columns',
    'functional': 'RwB97XD (neutral) / UwB97XD (cation, triplet)',
    'basis': 'see Gaussian logs (not in CSV)',
    'geometry': 'adiabatic (4-point Nelsen)',
    'charge_state': 'neutral->cation (hole-lambda); neutral->T1 (triplet-lambda)',
    'conformers': 'one per molecule (ground-state geometry)',
}


# ---------- shared helpers ----------

def _smiles_to_fp(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)


def _featurize(smiles_list):
    """Return (X[N,2048] uint8, fps list, valid mask) — invalid SMILES dropped."""
    fps, X, valid = [], [], []
    for smi in smiles_list:
        fp = _smiles_to_fp(smi)
        if fp is None:
            valid.append(False)
            continue
        valid.append(True)
        fps.append(fp)
        arr = np.zeros((MORGAN_BITS,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X.append(arr)
    return np.asarray(X), fps, np.asarray(valid)


def _tanimoto_dist_matrix(fps):
    n = len(fps)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        dist[i] = 1.0 - np.asarray(sims)
    np.fill_diagonal(dist, 0.0)
    return dist


def _stats(y):
    y = np.asarray(y, dtype=float)
    return {
        'n': int(len(y)),
        'mean': float(np.mean(y)),
        'std': float(np.std(y)),
        'min': float(np.min(y)),
        'p5': float(np.percentile(y, 5)),
        'median': float(np.median(y)),
        'p95': float(np.percentile(y, 95)),
        'max': float(np.max(y)),
        'skew': float(sp_stats.skew(y)),
    }


# ---------- V1 ----------

def cmd_v1():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_d = load_client_d()
    df_ch = load_client_c_hole()
    df_ct = load_client_c_triplet()

    y_d = df_d['reorg_eV'].to_numpy(dtype=float)
    y_ch = df_ch['lambda_hole_eV'].to_numpy(dtype=float)
    y_ct = df_ct['lambda_T_total_eV'].to_numpy(dtype=float)

    s_d, s_ch, s_ct = _stats(y_d), _stats(y_ch), _stats(y_ct)
    ks_d_ch = sp_stats.ks_2samp(y_d, y_ch)
    ks_d_ct = sp_stats.ks_2samp(y_d, y_ct)

    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=120)
    ax.hist(y_d, bins=60, density=True, alpha=0.5,
            label=f'Client D / Atahan (n={len(y_d)})', color='steelblue')
    ax.hist(y_ch, bins=20, density=True, alpha=0.5,
            label=f'Client C hole-$\\lambda$ (n={len(y_ch)})', color='crimson')
    ax.hist(y_ct, bins=20, density=True, alpha=0.5,
            label=f'Client C triplet-$\\lambda$ (n={len(y_ct)})', color='darkorange')
    ax.set_xlabel('Reorganization energy (eV)')
    ax.set_ylabel('Density')
    ax.set_title('Label distribution: Client D vs Client C')
    ax.legend()
    fig.tight_layout()
    fig_path = OUT_DIR / 'V1_label_distributions.png'
    fig.savefig(fig_path)
    plt.close(fig)

    # Verdict — evidence for D's protocol is the paper *abstract* only, so
    # geometry / charge-state / conformer / 4-point details are UNKNOWN. This
    # caps the achievable verdict at WARN/INCOMPLETE; PASS requires full-methods
    # access. FAIL is still reachable on empirical grounds (disjoint distributions).
    same_functional_family = False  # B3LYP (pure hybrid) vs wB97XD (range-separated hybrid)
    evidence_complete = False        # only the Atahan-Evrenk 2019 abstract was fetched
    ranges_overlap_dh = not (s_d['p95'] < s_ch['p5'] or s_ch['p95'] < s_d['p5'])

    if ks_d_ch.pvalue < 1e-3 and not ranges_overlap_dh:
        verdict = 'FAIL'
        reason = (
            'D vs C-hole: KS p<1e-3 AND p5/p95 ranges disjoint, AND functionals '
            'differ (B3LYP/6-31G* vs wB97XD). Naive supervised transfer would '
            'inherit a systematic label bias. Pivot to MAML / label-aware '
            'multi-task instead.'
        )
    elif not evidence_complete or not same_functional_family:
        verdict = 'WARN/INCOMPLETE'
        reason = (
            'Functionals differ (D: B3LYP/6-31G*, C: wB97XD), AND Atahan '
            'protocol evidence here is abstract-only — geometry, charge state, '
            'conformer treatment, and 4-point vs vertical method remain UNKNOWN. '
            'Apply explicit label re-scaling in any later transfer plan; access '
            'the full methods section before strengthening this verdict.'
        )
    else:
        # Unreachable while evidence_complete=False; kept for symmetry / future use.
        verdict = 'PASS'
        reason = 'Protocols broadly comparable.'

    md = []
    md.append('# V1 — Label protocol & distribution comparability\n')
    md.append('## Protocol comparison\n')
    md.append('| Aspect | Client D (Atahan-Evrenk 2019) | Client C |')
    md.append('|---|---|---|')
    md.append(f"| Functional | {ATAHAN_PROTOCOL['functional']} | {CLIENT_C_PROTOCOL['functional']} |")
    md.append(f"| Basis | {ATAHAN_PROTOCOL['basis']} | {CLIENT_C_PROTOCOL['basis']} |")
    md.append(f"| Geometry | {ATAHAN_PROTOCOL['geometry']} | {CLIENT_C_PROTOCOL['geometry']} |")
    md.append(f"| Charge state | {ATAHAN_PROTOCOL['charge_state']} | {CLIENT_C_PROTOCOL['charge_state']} |")
    md.append(f"| Conformers | {ATAHAN_PROTOCOL['conformers']} | {CLIENT_C_PROTOCOL['conformers']} |")
    md.append(
        f"| Library size | {ATAHAN_PROTOCOL['library_size_paper']} (paper) / "
        f"{ATAHAN_PROTOCOL['library_size_project']} (project) | "
        '53 (hole) / 49 (triplet) |'
    )
    md.append(f"| Source | {ATAHAN_PROTOCOL['source']} | {CLIENT_C_PROTOCOL['source']} |")
    md.append('')
    md.append(
        '> **Evidence-source caveat.** Atahan protocol evidence source = '
        'abstract/PubMed only; geometry / charge-state / conformer / 4-point '
        'details remain UNKNOWN unless the full methods section is accessed. '
        'V1 therefore cannot confirm protocol identity from this evidence alone — '
        'the achievable verdict is capped at **WARN/INCOMPLETE**, never PASS, '
        'until the full paper is read.\n'
    )
    md.append('## Label distribution stats (eV)\n')
    md.append('| Stat | D | C-hole | C-triplet |')
    md.append('|---|---:|---:|---:|')
    for k in ['n', 'mean', 'std', 'min', 'p5', 'median', 'p95', 'max', 'skew']:
        if k == 'n':
            md.append(f'| {k} | {s_d[k]} | {s_ch[k]} | {s_ct[k]} |')
        else:
            md.append(f'| {k} | {s_d[k]:.4f} | {s_ch[k]:.4f} | {s_ct[k]:.4f} |')
    md.append('')
    md.append('## KS two-sample tests\n')
    md.append(f'- D vs C-hole:    KS={ks_d_ch.statistic:.3f}, p={ks_d_ch.pvalue:.3e}')
    md.append(f'- D vs C-triplet: KS={ks_d_ct.statistic:.3f}, p={ks_d_ct.pvalue:.3e}')
    md.append('')
    md.append('## Discussion\n')
    md.append(
        '- B3LYP (pure hybrid, ~20% HF exchange) systematically underestimates '
        'reorganization energies for extended π systems compared with range-separated '
        'hybrids such as wB97X-D, which more accurately describe charge localization.'
    )
    md.append(
        '- Different basis-set quality (6-31G* vs the larger basis typical for wB97XD) '
        'adds a smaller but non-negligible systematic offset.'
    )
    md.append(
        '- Geometry and conformer treatment for D are not stated in the Atahan-Evrenk '
        '2019 abstract; full-paper inspection is required before claiming protocol identity.'
    )
    md.append(
        '- The dataset size differs (paper: 5631; project: 5876). Confirm provenance '
        'of the local CSV — it may be an extended set whose protocol has not been audited.'
    )
    md.append('')
    md.append(f'**Verdict: {verdict}**\n')
    md.append(f'Reason: {reason}\n')

    md_path = OUT_DIR / 'V1_label_protocol.md'
    md_path.write_text('\n'.join(md))
    print(f'[V1] verdict={verdict}')
    print(f'[V1] -> {md_path}')
    print(f'[V1] -> {fig_path}')


# ---------- V2 ----------

def _build_models(seed):
    models = {
        'RF': RandomForestRegressor(
            n_estimators=500, random_state=seed, n_jobs=-1
        ),
        'GP': GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(),
            random_state=seed, normalize_y=True, alpha=1e-6,
        ),
        'KNN': KNeighborsRegressor(
            n_neighbors=5, metric='precomputed', weights='distance'
        ),
    }
    if HAS_XGB:
        # Fixed conservative hyperparameters; no early stopping. LOOCV has no
        # held-out validation fold without leaking the test sample, so we never
        # pass eval_set / early_stopping_rounds.
        models['XGB'] = XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=-1, verbosity=0,
        )
    return models


def _loocv_predict(model, X, y, dist=None):
    n = len(y)
    y_hat = np.zeros(n, dtype=float)
    for tr, te in LeaveOneOut().split(np.arange(n)):
        if dist is not None:
            X_tr = dist[np.ix_(tr, tr)]
            X_te = dist[np.ix_(te, tr)]
            model.fit(X_tr, y[tr])
            y_hat[te] = model.predict(X_te)
        else:
            model.fit(X[tr], y[tr])
            y_hat[te] = model.predict(X[te])
    return y_hat


def _eval_target(name, smiles, y_raw, agg_rows, pred_rows):
    print(f'[V2] featurizing {name} (n={len(smiles)}) ...')
    X, fps, valid = _featurize(smiles)
    smiles_kept = [s for s, v in zip(smiles, valid) if v]
    y = np.asarray(y_raw, dtype=float)[valid]
    if len(y) != len(smiles):
        print(f'[V2]   {len(smiles) - len(y)} SMILES dropped (invalid)')
    dist = _tanimoto_dist_matrix(fps)

    for seed in SEEDS:
        for model_name, model in _build_models(seed).items():
            t0 = time.time()
            y_hat = _loocv_predict(model, X, y, dist=dist if model_name == 'KNN' else None)
            mae = float(mean_absolute_error(y, y_hat))
            rmse = float(np.sqrt(mean_squared_error(y, y_hat)))
            r2 = float(r2_score(y, y_hat))
            elapsed = time.time() - t0
            agg_rows.append({
                'model': model_name, 'target': name, 'seed': seed,
                'mae': mae, 'rmse': rmse, 'r2': r2, 'sec': elapsed,
            })
            for fold_idx, (smi, yt, yp) in enumerate(zip(smiles_kept, y, y_hat)):
                pred_rows.append({
                    'target': name, 'model': model_name, 'seed': seed,
                    'fold': fold_idx, 'smiles': smi,
                    'y_true': float(yt), 'y_pred': float(yp),
                    'abs_error': float(abs(yt - yp)),
                })
            print(f'[V2]   {model_name:>3s} seed={seed:>4d}  '
                  f'MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:+.3f}  ({elapsed:.1f}s)')


def cmd_v2():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not HAS_XGB:
        print('[V2] WARNING: xgboost not installed — running RF / GP / KNN only.')
        print('[V2]          install with: pip install xgboost')

    df_ch = load_client_c_hole()
    df_ct = load_client_c_triplet()

    agg_rows, pred_rows = [], []
    _eval_target('hole', df_ch['smiles'].tolist(), df_ch['lambda_hole_eV'].tolist(),
                 agg_rows, pred_rows)
    _eval_target('triplet', df_ct['smiles'].tolist(), df_ct['lambda_T_total_eV'].tolist(),
                 agg_rows, pred_rows)

    df_out = pd.DataFrame(agg_rows)
    csv_path = OUT_DIR / 'V2_classical_baselines.csv'
    df_out.to_csv(csv_path, index=False)

    df_pred = pd.DataFrame(pred_rows)
    pred_path = OUT_DIR / 'V2_predictions.csv'
    df_pred.to_csv(pred_path, index=False)

    md = ['# V2 — Classical-ML baselines on Client C\n']
    md.append('5-seed × LOOCV. Morgan radius=2, nBits=2048.')
    md.append('')
    if not HAS_XGB:
        md.append('> Note: xgboost not installed — XGB row absent from results below.\n')

    best_per_target = {}
    for tgt in ['hole', 'triplet']:
        sub = df_out[df_out.target == tgt]
        n_show = len(df_ch) if tgt == 'hole' else len(df_ct)
        md.append(f'## {tgt} (n={n_show})\n')
        md.append('| model | MAE (mean ± std) | RMSE (mean ± std) | R² (mean ± std) |')
        md.append('|---|---|---|---|')
        for m in sub.model.unique():
            sm = sub[sub.model == m]
            md.append(
                f'| {m} | {sm.mae.mean():.4f} ± {sm.mae.std():.4f} '
                f'| {sm.rmse.mean():.4f} ± {sm.rmse.std():.4f} '
                f'| {sm.r2.mean():+.3f} ± {sm.r2.std():.3f} |'
            )
        md.append('')
        if len(sub):
            best_per_target[tgt] = float(sub.groupby('model').r2.mean().max())

    md.append('## Verdict\n')
    bh = best_per_target.get('hole')
    bt = best_per_target.get('triplet')
    md.append(f'- Best classical R² on C-hole: **{bh:+.3f}** (vs E52 SchNet+KAN+FedPer: 0.139)' if bh is not None else '- C-hole: no result')
    md.append(f'- Best classical R² on C-triplet: **{bt:+.3f}**' if bt is not None else '- C-triplet: no result')
    md.append('')
    if bh is not None:
        if bh > 0.30:
            md.append(
                '**Architecture-replacement priority.** Classical Morgan-FP regressor '
                'beats current SchNet+KAN; reframe Client C local model as a '
                'fingerprint-based regressor before any federation tweaks.'
            )
        elif bh > 0.10:
            md.append(
                '**SchNet+KAN is in the right ballpark.** Transfer experiments worth '
                'running; current architectural choice is not obviously misallocated.'
            )
        else:
            md.append(
                '**Possible data ceiling.** All non-deep methods fail too; investigate '
                'label noise / DFT protocol heterogeneity / conformer dependence on '
                'Client C before more model work.'
            )
    md_path = OUT_DIR / 'V2_summary.md'
    md_path.write_text('\n'.join(md))

    print(f'[V2] -> {csv_path}')
    print(f'[V2] -> {pred_path}')
    print(f'[V2] -> {md_path}')
    if bh is not None:
        print(f'[V2] best R²: hole={bh:+.3f}  triplet={bt:+.3f}')


# ---------- V3 ----------

def _scaffolds_of(smiles_iter):
    out = set()
    for smi in smiles_iter:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            if scaf is not None and scaf.GetNumAtoms() > 0:
                out.add(Chem.MolToSmiles(scaf))
        except Exception:
            continue
    return out


def cmd_v3():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_d = load_client_d()
    df_ch = load_client_c_hole()
    df_ct = load_client_c_triplet()

    print('[V3] featurizing D ...')
    _, fps_d, valid_d = _featurize(df_d['smiles'].tolist())
    smiles_d = [s for s, v in zip(df_d['smiles'].tolist(), valid_d) if v]

    print('[V3] featurizing C-hole ...')
    _, fps_ch, valid_ch = _featurize(df_ch['smiles'].tolist())
    smiles_ch = [s for s, v in zip(df_ch['smiles'].tolist(), valid_ch) if v]

    print('[V3] featurizing C-triplet ...')
    _, fps_ct, valid_ct = _featurize(df_ct['smiles'].tolist())
    smiles_ct = [s for s, v in zip(df_ct['smiles'].tolist(), valid_ct) if v]

    rows = []
    for tag, fps_c, smis_c in [('hole', fps_ch, smiles_ch), ('triplet', fps_ct, smiles_ct)]:
        for fp_c, smi_c in zip(fps_c, smis_c):
            sims = DataStructs.BulkTanimotoSimilarity(fp_c, fps_d)
            i = int(np.argmax(sims))
            rows.append({
                'smiles_c': smi_c, 'target': tag,
                'max_tanimoto': float(sims[i]),
                'nearest_smiles_d': smiles_d[i],
            })
    df_v3 = pd.DataFrame(rows)
    csv_path = OUT_DIR / 'V3_chem_space.csv'
    df_v3.to_csv(csv_path, index=False)

    print('[V3] computing Murcko scaffolds ...')
    scaf_d = _scaffolds_of(smiles_d)
    scaf_ch = _scaffolds_of(smiles_ch)
    scaf_ct = _scaffolds_of(smiles_ct)
    overlap_h = len(scaf_ch & scaf_d) / max(1, len(scaf_ch))
    overlap_t = len(scaf_ct & scaf_d) / max(1, len(scaf_ct))

    tsne_made = False
    fig_path = OUT_DIR / 'V3_tsne_highlight.png'
    if TSNE_COORDS.exists():
        df_tsne = pd.read_csv(TSNE_COORDS)
        # Schema confirmed: tsne_1, tsne_2, client, smiles
        fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=120)
        layers = [
            ('A', 'lightgrey', 0.30, 8),
            ('B', 'silver', 0.30, 8),
            ('D', 'steelblue', 0.20, 6),
            ('C', 'crimson', 0.95, 60),
        ]
        for client_lbl, color, alpha, size in layers:
            mask = df_tsne['client'].astype(str).str.upper() == client_lbl
            n_pts = int(mask.sum())
            ax.scatter(
                df_tsne.loc[mask, 'tsne_1'], df_tsne.loc[mask, 'tsne_2'],
                c=color, alpha=alpha, s=size,
                label=f'Client {client_lbl} (n={n_pts})',
                edgecolors='none' if client_lbl != 'C' else 'black',
                linewidths=0.0 if client_lbl != 'C' else 0.4,
            )
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.set_title('Chemical space (Morgan2048 + t-SNE) — Client C highlighted')
        ax.legend(loc='best', framealpha=0.9)
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        tsne_made = True

    md = ['# V3 — Chemical-space overlap C ↔ D\n']
    md.append('## Per-C-molecule nearest neighbour in D (Tanimoto, Morgan2048)\n')
    md.append('| target | n | median maxT | IQR | frac > 0.5 | frac > 0.7 |')
    md.append('|---|---:|---:|---:|---:|---:|')
    medians = {}
    for tag in ['hole', 'triplet']:
        sub = df_v3[df_v3.target == tag]
        if len(sub) == 0:
            continue
        m = sub.max_tanimoto
        medians[tag] = float(m.median())
        md.append(
            f'| {tag} | {len(sub)} | {m.median():.3f} '
            f'| [{m.quantile(0.25):.3f}, {m.quantile(0.75):.3f}] '
            f'| {(m > 0.5).mean():.2f} | {(m > 0.7).mean():.2f} |'
        )
    md.append('')
    md.append('## Murcko scaffold overlap\n')
    md.append(f'- C-hole unique scaffolds: {len(scaf_ch)}; '
              f'shared with D: {len(scaf_ch & scaf_d)} ({overlap_h * 100:.1f}%)')
    md.append(f'- C-triplet unique scaffolds: {len(scaf_ct)}; '
              f'shared with D: {len(scaf_ct & scaf_d)} ({overlap_t * 100:.1f}%)')
    md.append(f'- D unique scaffolds: {len(scaf_d)}')
    md.append('')

    med_h = medians.get('hole', 0.0)
    md.append('## Verdict (hole-λ basis)\n')
    if med_h >= 0.50 and overlap_h >= 0.30:
        v = ('STRONG OVERLAP — D→C transfer is reasonable to expect positive. '
             'Proceed with the supervised-pretrain plan.')
    elif med_h >= 0.30 or overlap_h >= 0.10:
        v = ('MODERATE OVERLAP — expect modest transfer benefit (≤0.05 R²). '
             'Any later transfer experiment must include a negative-transfer '
             'ablation (D-pretrain vs random-init).')
    else:
        v = ('LOW OVERLAP — D→C is functionally OOD transfer. Deprioritize, '
             'escalate to meta-learning / domain-adversarial methods.')
    md.append(v)
    md.append('')
    if not tsne_made:
        md.append('> NOTE: results/tables/tsne_coordinates.csv not found — '
                  't-SNE highlight figure skipped. Run plot_paper_figures.py first '
                  'to populate it, then re-run --v3.')

    md_path = OUT_DIR / 'V3_summary.md'
    md_path.write_text('\n'.join(md))
    print(f'[V3] -> {csv_path}')
    print(f'[V3] -> {md_path}')
    if tsne_made:
        print(f'[V3] -> {fig_path}')
    print(f'[V3] median maxT (hole) = {med_h:.3f}; scaffold overlap (hole) = {overlap_h * 100:.1f}%')


# ---------- V3 all-pairs: 5×5 task-client transferability (Phase 0 of PC²-FedReorg) ----------

# Five task-client nodes for the PC²-FedReorg transferability matrix.
# C-hole and C-triplet are deliberately separate nodes (different quantities ->
# different T_head, label distributions, and (mu, sigma) calibration), even
# though they correspond to the same physical lab and largely the same molecules.
TASK_CLIENT_KEYS = ['A', 'B', 'C-hole', 'C-triplet', 'D']


def _load_task_client_smiles() -> dict:
    """Return {node_key: list[smiles]} for the 5 task-client nodes."""
    print('[V3-allpairs] Loading public data and splitting A/B by aromaticity ...')
    df_pub = load_public_data()
    client_a_df, client_b_df = split_public_by_aromaticity(df_pub)

    print('[V3-allpairs] Loading C-hole / C-triplet / D ...')
    df_ch = load_client_c_hole()
    df_ct = load_client_c_triplet()
    df_d = load_client_d()

    return {
        'A': client_a_df['smiles'].tolist(),
        'B': client_b_df['smiles'].tolist(),
        'C-hole': df_ch['smiles'].tolist(),
        'C-triplet': df_ct['smiles'].tolist(),
        'D': df_d['smiles'].tolist(),
    }


def cmd_v3_allpairs():
    """Compute K[i,j] and S[i,j] for all 20 ordered off-diagonal pairs in the
    5-node task-client matrix {A, B, C-hole, C-triplet, D}, plus diagonals = 1.

    K[i,j] = mean over i-mols of max-Tanimoto similarity to any j-mol.
    S[i,j] = |Scaf(i) ∩ Scaf(j)| / |Scaf(i)|  (asymmetric, source-normalized).

    Output: results/preverify/T_chemistry.json
    """
    import json as _json

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    smiles_per_node = _load_task_client_smiles()

    fps_per_node = {}
    scaf_per_node = {}
    n_kept = {}
    for node in TASK_CLIENT_KEYS:
        smis = smiles_per_node[node]
        print(f'[V3-allpairs] Featurizing {node} (n_input={len(smis)}) ...')
        _, fps, valid = _featurize(smis)
        fps_per_node[node] = fps
        n_kept[node] = int(len(fps))
        smis_kept = [s for s, v in zip(smis, valid) if v]
        print(f'[V3-allpairs]   computing Murcko scaffolds for {node} ...')
        scaf_per_node[node] = _scaffolds_of(smis_kept)
        print(f'[V3-allpairs]   {node}: kept {n_kept[node]}, '
              f'{len(scaf_per_node[node])} unique scaffolds')

    # K[i][j] = mean over i-mols of max Tanimoto to j-mols. Diagonal = 1.0.
    print('[V3-allpairs] Computing K (Tanimoto) for 20 ordered off-diag pairs ...')
    K = {i: {} for i in TASK_CLIENT_KEYS}
    for i in TASK_CLIENT_KEYS:
        for j in TASK_CLIENT_KEYS:
            if i == j:
                K[i][j] = 1.0
                continue
            t0 = time.time()
            fps_i = fps_per_node[i]
            fps_j = fps_per_node[j]
            max_sims = np.empty(len(fps_i), dtype=np.float64)
            for k, fp in enumerate(fps_i):
                sims = DataStructs.BulkTanimotoSimilarity(fp, fps_j)
                max_sims[k] = float(np.max(sims))
            K[i][j] = float(max_sims.mean())
            print(f'[V3-allpairs]   K[{i:>9s} -> {j:<9s}] = {K[i][j]:.4f}  '
                  f'({n_kept[i]} x {n_kept[j]}, {time.time() - t0:.1f}s)')

    # S[i][j] = |Scaf(i) ∩ Scaf(j)| / |Scaf(i)|. Diagonal = 1.0 (when scaffolds nonempty).
    print('[V3-allpairs] Computing S (Murcko scaffold overlap) ...')
    S = {i: {} for i in TASK_CLIENT_KEYS}
    for i in TASK_CLIENT_KEYS:
        for j in TASK_CLIENT_KEYS:
            scaf_i = scaf_per_node[i]
            scaf_j = scaf_per_node[j]
            if len(scaf_i) == 0:
                S[i][j] = 0.0
                continue
            S[i][j] = float(len(scaf_i & scaf_j) / len(scaf_i))

    out = {
        'description': (
            'V3-allpairs chemistry-overlap stats for the 5-node task-client '
            'transferability matrix used by PC²-FedReorg. K[i,j] = mean over '
            'i-mols of max-Tanimoto similarity to any j-mol (Morgan radius=2, '
            f'nBits={MORGAN_BITS}). S[i,j] = |Scaf(i) ∩ Scaf(j)| / |Scaf(i)| '
            '(Murcko scaffolds, asymmetric, source-normalized). Diagonals = 1.0.'
        ),
        'node_keys': TASK_CLIENT_KEYS,
        'n_kept': n_kept,
        'n_unique_scaffolds': {k: len(scaf_per_node[k]) for k in TASK_CLIENT_KEYS},
        'morgan_radius': MORGAN_RADIUS,
        'morgan_bits': MORGAN_BITS,
        'K': K,
        'S': S,
    }

    json_path = OUT_DIR / 'T_chemistry.json'
    json_path.write_text(_json.dumps(out, indent=2))

    md = ['# V3-allpairs — 5×5 task-client chemistry overlap (Phase 0)\n']
    md.append('Inputs to PC²-FedReorg `T_repr[i,j] = max(eps, w_K * K[i,j] + w_S * S[i,j])`.')
    md.append('Diagonals are 1.0 by definition.')
    md.append('')
    md.append(f'## Node sizes\n')
    md.append('| node | n (kept) | unique scaffolds |')
    md.append('|---|---:|---:|')
    for k in TASK_CLIENT_KEYS:
        md.append(f'| {k} | {n_kept[k]} | {len(scaf_per_node[k])} |')
    md.append('')

    def _matrix_md(label, M):
        lines = [f'## {label} matrix (rows = source i, cols = target j)\n']
        header = '| i \\\\ j | ' + ' | '.join(TASK_CLIENT_KEYS) + ' |'
        sep = '|---|' + '---:|' * len(TASK_CLIENT_KEYS)
        lines.append(header)
        lines.append(sep)
        for i in TASK_CLIENT_KEYS:
            cells = [f'{M[i][j]:.4f}' for j in TASK_CLIENT_KEYS]
            lines.append(f'| {i} | ' + ' | '.join(cells) + ' |')
        lines.append('')
        return lines

    md.extend(_matrix_md('K (Tanimoto similarity, source-mean of max)', K))
    md.extend(_matrix_md('S (Murcko scaffold overlap, source-normalized)', S))

    md.append('## T_repr preview (defaults: w_K=0.6, w_S=0.4, floor=1e-3)\n')
    md.append('| i \\\\ j | ' + ' | '.join(TASK_CLIENT_KEYS) + ' |')
    md.append('|---|' + '---:|' * len(TASK_CLIENT_KEYS))
    for i in TASK_CLIENT_KEYS:
        cells = []
        for j in TASK_CLIENT_KEYS:
            if i == j:
                cells.append('1.0000')
            else:
                t = max(1e-3, 0.6 * K[i][j] + 0.4 * S[i][j])
                cells.append(f'{t:.4f}')
        md.append(f'| {i} | ' + ' | '.join(cells) + ' |')
    md.append('')

    md.append('## Sanity checks (locked algorithmic invariants)\n')
    md.append('Independent of the K/S numbers above, these T_head invariants must hold:')
    md.append('- T_head[D, C-hole] = 0 (clauses 2 + 4: D protocol unknown + V1=FAIL hard rule)')
    md.append('- T_head[D, C-triplet] = 0 (clause 1: quantity mismatch)')
    md.append('- T_head[C-hole, C-triplet] = 0 (clause 1)')
    md.append('- T_head[C-triplet, C-hole] = 0 (clause 1)')
    md.append('- T_head[A, C-triplet] = 0 (clause 1)')
    md.append('- T_head[B, C-triplet] = 0 (clause 1)')
    md.append('')

    md_path = OUT_DIR / 'T_chemistry.md'
    md_path.write_text('\n'.join(md))

    print(f'[V3-allpairs] -> {json_path}')
    print(f'[V3-allpairs] -> {md_path}')


# ---------- entry ----------

def main():
    p = argparse.ArgumentParser(description='Pre-verification for D->C transfer.')
    p.add_argument('--v1', action='store_true', help='label-protocol & distribution check')
    p.add_argument('--v2', action='store_true', help='classical-ML baselines on Client C')
    p.add_argument('--v3', action='store_true', help='chemical-space overlap C<->D')
    p.add_argument('--v3-allpairs', action='store_true',
                   help='5x5 task-client K/S matrix for PC²-FedReorg (Phase 0)')
    p.add_argument('--all', action='store_true', help='run V1 + V2 + V3 (default if no flag given; --v3-allpairs is opt-in)')
    args = p.parse_args()

    run_all = args.all or not (args.v1 or args.v2 or args.v3 or args.v3_allpairs)
    if run_all or args.v1:
        cmd_v1()
    if run_all or args.v2:
        cmd_v2()
    if run_all or args.v3:
        cmd_v3()
    if args.v3_allpairs:
        cmd_v3_allpairs()


if __name__ == '__main__':
    main()
