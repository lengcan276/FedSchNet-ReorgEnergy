"""
Data preparation module.
- Load public dataset and Client C TADF data
- SMILES to PyG molecular graph conversion
- Client A/B/C data splitting
"""

import os
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, rdPartialCharges
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

# Suppress RDKit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings('ignore')

# ============ Path configuration ============
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PUBLIC_CSV = DATA_DIR / 'client_a_b' / 'public_reorg_energy_15210.csv'
HOLE_REORG_CSV = DATA_DIR / 'client_c' / 'hole_reorg_molecular.csv'
TRIPLET_REORG_CSV = DATA_DIR / 'client_c' / 'reorganization_energy_summary.csv'
ATAHAN_CSV = DATA_DIR / 'client_d' / 'atahan_reorg_5876.csv'

# ============ Atom type mapping ============
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'Other']
ATOM_TYPE_MAP = {a: i for i, a in enumerate(ATOM_TYPES)}
NODE_FEAT_DIM = 11   # 8 (atom type) + 1 (degree) + 1 (formal charge) + 1 (Gasteiger charge)
EDGE_FEAT_DIM = 4    # single/double/triple/aromatic bond
GLOBAL_DESC_DIM = 6  # MolWt, TPSA, LogP, NumAromaticRings, MaxAbsPartialCharge, MolMR
PHYSICS_FEAT_DIM = 4 # HOMO, LUMO, gap, dipole
PHYS_FEATS_DIM = 5   # NumRotatableBonds, MaxPartialCharge, MinPartialCharge, TPSA, FractionCSP3
CONF3D_FEATS_DIM = 6 # n_rotatable_bonds, conf_rmsd_mean, conf_rmsd_max, energy_spread, max_dihedral_range, asphericity


def smiles_to_graph(smiles: str, y: Optional[float] = None) -> Optional[Data]:
    """Convert SMILES to a PyG Data object.

    Node features (11-dim):
      - Atom type one-hot: C,N,O,S,F,Cl,Br,Other -> 8-dim
      - Degree -> 1-dim (normalized to 0-1, divided by 4)
      - Formal charge -> 1-dim
      - Gasteiger charge -> 1-dim

    Edge features (4-dim):
      - Bond type one-hot: single, double, triple, aromatic
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Compute Gasteiger charges
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        pass

    # === Node features ===
    node_feats = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        # Atom type one-hot (8-dim)
        atom_type_idx = ATOM_TYPE_MAP.get(symbol, ATOM_TYPE_MAP['Other'])
        one_hot = [0.0] * 8
        one_hot[atom_type_idx] = 1.0

        # Degree (normalized)
        degree = atom.GetDegree() / 4.0

        # Formal charge
        formal_charge = float(atom.GetFormalCharge())

        # Gasteiger charge
        gasteiger = atom.GetDoubleProp('_GasteigerCharge') if atom.HasProp('_GasteigerCharge') else 0.0
        if np.isnan(gasteiger) or np.isinf(gasteiger):
            gasteiger = 0.0

        node_feats.append(one_hot + [degree, formal_charge, gasteiger])

    x = torch.tensor(node_feats, dtype=torch.float)

    # === Edge features and edge index ===
    edge_index = []
    edge_attr = []
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: [1, 0, 0, 0],
        Chem.rdchem.BondType.DOUBLE: [0, 1, 0, 0],
        Chem.rdchem.BondType.TRIPLE: [0, 0, 1, 0],
        Chem.rdchem.BondType.AROMATIC: [0, 0, 0, 1],
    }

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = bond_type_map.get(bond.GetBondType(), [0, 0, 0, 0])
        # Bidirectional edges
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([bt, bt])

    if len(edge_index) == 0:
        # Single-atom molecule
        edge_index_tensor = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.zeros((0, EDGE_FEAT_DIM), dtype=torch.float)
    else:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)

    # === Global descriptors (6-dim) ===
    mol_no_h = Chem.RemoveHs(mol)
    global_desc = _compute_global_descriptors(mol_no_h)

    # === 5-dim physical descriptors ===
    phys_desc_5d = _compute_phys_descriptors_5d(mol_no_h)

    data = Data(
        x=x,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr_tensor,
        global_desc=torch.tensor(global_desc, dtype=torch.float).unsqueeze(0),
        phys_feats=torch.tensor(phys_desc_5d, dtype=torch.float).unsqueeze(0),
        smiles=smiles,
    )

    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)

    return data


def _compute_global_descriptors(mol) -> list:
    """Compute 6-dim global molecular descriptors: MolWt, TPSA, LogP, NumAromaticRings, MaxAbsPartialCharge, MolMR."""
    try:
        mol_wt = Descriptors.MolWt(mol)
    except Exception:
        mol_wt = 0.0
    try:
        tpsa = Descriptors.TPSA(mol)
    except Exception:
        tpsa = 0.0
    try:
        logp = Descriptors.MolLogP(mol)
    except Exception:
        logp = 0.0
    try:
        n_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    except Exception:
        n_aromatic = 0
    try:
        max_charge = Descriptors.MaxAbsPartialCharge(mol)
        if max_charge is None or np.isnan(max_charge):
            max_charge = 0.0
    except Exception:
        max_charge = 0.0
    try:
        mol_mr = Descriptors.MolMR(mol)
    except Exception:
        mol_mr = 0.0

    return [mol_wt, tpsa, logp, float(n_aromatic), max_charge, mol_mr]


def _compute_phys_descriptors_5d(mol) -> list:
    """Compute 5-dim graph-level physical descriptors.

    1. NumRotatableBonds (flexibility)
    2. MaxPartialCharge (charge concentration, Gasteiger)
    3. MinPartialCharge (charge concentration, Gasteiger)
    4. TPSA (topological polar surface area)
    5. FractionCSP3 (sp3-hybridized carbon fraction, spatial non-planarity)
    """
    try:
        n_rotatable = float(Descriptors.NumRotatableBonds(mol))
    except Exception:
        n_rotatable = 0.0

    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            if atom.HasProp('_GasteigerCharge'):
                c = atom.GetDoubleProp('_GasteigerCharge')
                if not (np.isnan(c) or np.isinf(c)):
                    charges.append(c)
        max_partial = max(charges) if charges else 0.0
        min_partial = min(charges) if charges else 0.0
    except Exception:
        max_partial = 0.0
        min_partial = 0.0

    try:
        tpsa = Descriptors.TPSA(mol)
    except Exception:
        tpsa = 0.0

    try:
        frac_csp3 = Descriptors.FractionCSP3(mol)
    except Exception:
        frac_csp3 = 0.0

    return [n_rotatable, max_partial, min_partial, tpsa, frac_csp3]


# ============ Data loading functions ============

def load_public_data() -> pd.DataFrame:
    """Load public dataset public_reorg_energy_15210.csv."""
    df = pd.read_csv(PUBLIC_CSV)
    print(f"[Public data] Loaded {len(df)} records")
    print(f"  Columns: {list(df.columns)}")
    print(f"  reorg_energy_eV: mean={df['reorg_energy_eV'].mean():.4f}, "
          f"std={df['reorg_energy_eV'].std():.4f}, "
          f"range=[{df['reorg_energy_eV'].min():.4f}, {df['reorg_energy_eV'].max():.4f}]")
    return df


def split_public_by_aromaticity(df: pd.DataFrame) -> tuple:
    """Split public data by aromatic ring count: Client A (NumAromaticRings==0), Client B (>=1)."""
    aromatic_counts = []
    for smiles in tqdm(df['smiles'], desc='Computing aromatic ring counts'):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            aromatic_counts.append(rdMolDescriptors.CalcNumAromaticRings(mol))
        else:
            aromatic_counts.append(-1)  # Invalid SMILES

    df = df.copy()
    df['num_aromatic_rings'] = aromatic_counts

    # Remove invalid SMILES
    valid_df = df[df['num_aromatic_rings'] >= 0]
    client_a_df = valid_df[valid_df['num_aromatic_rings'] == 0].reset_index(drop=True)
    client_b_df = valid_df[valid_df['num_aromatic_rings'] >= 1].reset_index(drop=True)

    print(f"\n[Client split]")
    print(f"  Client A (non-aromatic): {len(client_a_df)} molecules")
    print(f"    lambda: mean={client_a_df['reorg_energy_eV'].mean():.4f}, "
          f"std={client_a_df['reorg_energy_eV'].std():.4f}")
    print(f"  Client B (aromatic):     {len(client_b_df)} molecules")
    print(f"    lambda: mean={client_b_df['reorg_energy_eV'].mean():.4f}, "
          f"std={client_b_df['reorg_energy_eV'].std():.4f}")

    invalid_count = len(df) - len(valid_df)
    if invalid_count > 0:
        print(f"  Invalid SMILES skipped: {invalid_count}")

    return client_a_df, client_b_df


def load_client_c_hole() -> pd.DataFrame:
    """Load Client C hole reorganization energy data (molecular level, 53 molecules)."""
    df = pd.read_csv(HOLE_REORG_CSV)
    # Drop empty rows
    df = df.dropna(subset=['smiles']).reset_index(drop=True)
    print(f"\n[Client C - hole lambda] Loaded {len(df)} molecules")
    print(f"  lambda_hole_eV: mean={df['lambda_hole_eV'].mean():.4f}, "
          f"std={df['lambda_hole_eV'].std():.4f}, "
          f"range=[{df['lambda_hole_eV'].min():.4f}, {df['lambda_hole_eV'].max():.4f}]")
    return df


def load_client_c_triplet() -> pd.DataFrame:
    """Load Client C triplet reorganization energy data (49 molecules with valid lambda_T)."""
    df = pd.read_csv(TRIPLET_REORG_CSV)
    # Keep only rows with valid lambda_T_total_eV
    df_valid = df.dropna(subset=['lambda_T_total_eV']).reset_index(drop=True)
    print(f"\n[Client C - triplet lambda] Loaded {len(df_valid)}/{len(df)} molecules (valid lambda_T)")
    print(f"  lambda_T_total_eV: mean={df_valid['lambda_T_total_eV'].mean():.4f}, "
          f"std={df_valid['lambda_T_total_eV'].std():.4f}, "
          f"range=[{df_valid['lambda_T_total_eV'].min():.4f}, {df_valid['lambda_T_total_eV'].max():.4f}]")

    # Get SMILES from hole_reorg_molecular.csv by merging on molecule column
    hole_df = pd.read_csv(HOLE_REORG_CSV).dropna(subset=['smiles'])
    smiles_map = dict(zip(hole_df['molecule'], hole_df['smiles']))
    df_valid = df_valid.copy()
    df_valid['smiles'] = df_valid['molecule'].map(smiles_map)

    missing_smiles = df_valid['smiles'].isna().sum()
    if missing_smiles > 0:
        print(f"  Warning: {missing_smiles} molecules have no SMILES, will be skipped")
        df_valid = df_valid.dropna(subset=['smiles']).reset_index(drop=True)

    return df_valid


def load_client_d() -> pd.DataFrame:
    """Load Client D data (Atahan-Evrenk 2019, 5876 conjugated organic semiconductors, hole lambda)."""
    df = pd.read_csv(ATAHAN_CSV)
    df = df.dropna(subset=['smiles', 'reorg_eV']).reset_index(drop=True)
    print(f"\n[Client D - Atahan] Loaded {len(df)} molecules")
    print(f"  reorg_eV: mean={df['reorg_eV'].mean():.4f}, "
          f"std={df['reorg_eV'].std():.4f}, "
          f"range=[{df['reorg_eV'].min():.4f}, {df['reorg_eV'].max():.4f}]")
    return df


# ============ Batch graph construction ============

def build_graph_dataset(
    smiles_list: list,
    y_list: list,
    desc: str = 'Building graphs',
    physics_feats: Optional[dict] = None,
    mol_names: Optional[list] = None,
) -> list:
    """Batch convert SMILES to a list of PyG Data objects.

    Args:
        smiles_list: List of SMILES strings.
        y_list: List of target labels.
        desc: Description for tqdm progress bar.
        physics_feats: {mol_name: [HOMO, LUMO, gap, dipole]} physics feature dict.
        mol_names: List of molecule names (aligned with smiles_list) for looking up physics_feats.

    Returns:
        List of PyG Data objects.
    """
    data_list = []
    n_fail = 0

    for i, (smi, y_val) in enumerate(tqdm(zip(smiles_list, y_list), total=len(smiles_list), desc=desc)):
        data = smiles_to_graph(smi, y=y_val)
        if data is None:
            n_fail += 1
            continue

        # Attach physics features if available
        if physics_feats is not None and mol_names is not None:
            mol_name = mol_names[i]
            pf = physics_feats.get(mol_name, [0.0, 0.0, 0.0, 0.0])
            data.physics_feat = torch.tensor(pf, dtype=torch.float).unsqueeze(0)

        data_list.append(data)

    if n_fail > 0:
        print(f"  Warning: {n_fail}/{len(smiles_list)} SMILES failed to build graphs")

    print(f"  Successfully built {len(data_list)} molecular graphs")
    return data_list


# ============ Node-level physics feature augmentation (E21) ============

def augment_nodes_with_physics(data_list: list, pad_zeros: bool = False) -> list:
    """Broadcast physics features (HOMO/LUMO/gap/dipole) to each node and concatenate.

    Client C molecules have physics_feat -> broadcast [1, 4] -> [N, 4] and concat to node features.
    Client A/B molecules (pad_zeros=True) -> pad with zeros.

    Node features: [N, NODE_FEAT_DIM] -> [N, NODE_FEAT_DIM + PHYSICS_FEAT_DIM]
    """
    augmented = []
    for data in data_list:
        d = data.clone()
        n_nodes = d.x.size(0)
        if not pad_zeros and hasattr(d, 'physics_feat') and d.physics_feat is not None:
            pf = d.physics_feat.squeeze(0)  # [PHYSICS_FEAT_DIM]
            pf_broadcast = pf.unsqueeze(0).expand(n_nodes, -1)  # [N, PHYSICS_FEAT_DIM]
            d.x = torch.cat([d.x, pf_broadcast], dim=-1)
        else:
            zeros = torch.zeros(n_nodes, PHYSICS_FEAT_DIM, dtype=d.x.dtype)
            d.x = torch.cat([d.x, zeros], dim=-1)
        augmented.append(d)
    return augmented


# ============ Main function: prepare all client data ============

def prepare_all_clients(target_type: str = 'hole', include_d: bool = False,
                        normalize_y: bool = False) -> dict:
    """Prepare data for all clients.

    Args:
        target_type: 'hole' or 'triplet', determines which label Client C uses.
            - 'hole': Client C uses hole_reorg_molecular.csv (53 molecules, lambda_hole_eV)
            - 'triplet': Client C uses reorganization_energy_summary.csv (49 molecules, lambda_T_total_eV)
        include_d: Whether to include Client D (Atahan-Evrenk 2019, 5876 molecules).
        normalize_y: Whether to apply per-client y-label standardization (per-client StandardScaler).

    Returns:
        dict: {
            'client_a': list of Data,
            'client_b': list of Data,
            'client_c': list of Data,
            'client_d': list of Data (only when include_d=True),
            'physics_feats': dict,
            'target_type': str,
            'norm_stats': dict (only when normalize_y=True, {client_name: (y_mean, y_std)}),
        }
    """
    print("=" * 60)
    print(f"Phase 1: Data preparation (target_type={target_type}, include_d={include_d}, normalize_y={normalize_y})")
    print("=" * 60)

    # 1. Load public data and split
    public_df = load_public_data()
    client_a_df, client_b_df = split_public_by_aromaticity(public_df)

    # 2. Build Client A graphs
    print("\n[Building Client A molecular graphs]")
    client_a_graphs = build_graph_dataset(
        client_a_df['smiles'].tolist(),
        client_a_df['reorg_energy_eV'].tolist(),
        desc='Client A'
    )

    # 3. Build Client B graphs
    print("\n[Building Client B molecular graphs]")
    client_b_graphs = build_graph_dataset(
        client_b_df['smiles'].tolist(),
        client_b_df['reorg_energy_eV'].tolist(),
        desc='Client B'
    )

    # 4. Load Client C data
    if target_type == 'hole':
        c_df = load_client_c_hole()
        c_smiles = c_df['smiles'].tolist()
        c_y = c_df['lambda_hole_eV'].tolist()
        c_mol_names = c_df['molecule'].tolist()
    else:
        c_df = load_client_c_triplet()
        c_smiles = c_df['smiles'].tolist()
        c_y = c_df['lambda_T_total_eV'].tolist()
        c_mol_names = c_df['molecule'].tolist()

    # 5. Physics features (DFT extraction removed; kept as empty dict)
    physics_feats = {}

    # 6. Build Client C graphs
    print("\n[Building Client C molecular graphs]")
    client_c_graphs = build_graph_dataset(
        c_smiles, c_y, desc='Client C',
        physics_feats=physics_feats, mol_names=c_mol_names
    )

    # 7. Build Client D graphs (if requested)
    client_d_graphs = []
    if include_d:
        d_df = load_client_d()
        print("\n[Building Client D molecular graphs]")
        client_d_graphs = build_graph_dataset(
            d_df['smiles'].tolist(),
            d_df['reorg_eV'].tolist(),
            desc='Client D'
        )

    # Summary statistics
    print("\n" + "=" * 60)
    print("Data preparation complete - summary statistics")
    print("=" * 60)
    print(f"  Client A: {len(client_a_graphs)} molecular graphs (non-aromatic, public data)")
    print(f"  Client B: {len(client_b_graphs)} molecular graphs (aromatic, public data)")
    print(f"  Client C: {len(client_c_graphs)} molecular graphs (TADF, {target_type} lambda)")
    if include_d:
        print(f"  Client D: {len(client_d_graphs)} molecular graphs (Atahan, hole lambda)")

    # Validate data
    _validate_graphs(client_a_graphs, 'Client A')
    _validate_graphs(client_b_graphs, 'Client B')
    _validate_graphs(client_c_graphs, 'Client C')
    if include_d and client_d_graphs:
        _validate_graphs(client_d_graphs, 'Client D')

    result = {
        'client_a': client_a_graphs,
        'client_b': client_b_graphs,
        'client_c': client_c_graphs,
        'physics_feats': physics_feats,
        'target_type': target_type,
    }
    if include_d:
        result['client_d'] = client_d_graphs

    # Per-client label normalization
    if normalize_y:
        norm_stats = normalize_client_labels(result)
        result['norm_stats'] = norm_stats

    return result


def normalize_client_labels(result: dict) -> dict:
    """Per-client label standardization: y = (y - mean) / std.

    Computes y_mean and y_std for client_a/b/c/d separately and normalizes in-place.
    Returns norm_stats: {client_name: (y_mean, y_std)}.
    """
    norm_stats = {}
    client_keys = [k for k in ['client_a', 'client_b', 'client_c', 'client_d']
                   if k in result and result[k]]

    print("\n[Per-client label standardization]")
    for key in client_keys:
        data_list = result[key]
        ys = torch.cat([d.y for d in data_list])
        y_mean = ys.mean().item()
        y_std = ys.std().item()
        y_std = max(y_std, 1e-6)  # prevent division by zero

        # Normalize in-place
        for d in data_list:
            d.y = (d.y - y_mean) / y_std

        norm_stats[key] = (y_mean, y_std)
        print(f"  {key}: y_mean={y_mean:.4f}, y_std={y_std:.4f}")

    return norm_stats


def _validate_graphs(data_list: list, name: str):
    """Validate a list of molecular graphs."""
    if not data_list:
        print(f"  [Validate] {name}: empty list!")
        return

    none_count = sum(1 for d in data_list if d is None)
    if none_count > 0:
        print(f"  [Validate] {name}: {none_count} None graphs!")

    # Check feature dimensions
    sample = data_list[0]
    print(f"  [Validate] {name}: node_feat_dim={sample.x.shape[1]}, "
          f"edge_feat_dim={sample.edge_attr.shape[1] if sample.edge_attr.shape[0] > 0 else 'N/A'}, "
          f"global_desc_dim={sample.global_desc.shape[1]}")

    # y-value statistics
    y_vals = torch.cat([d.y for d in data_list]).numpy()
    print(f"  [Validate] {name}: y mean={y_vals.mean():.4f}, std={y_vals.std():.4f}, "
          f"range=[{y_vals.min():.4f}, {y_vals.max():.4f}]")


# ============ 3D conformer feature computation ============

def _compute_3d_conformer_features(smiles: str, n_confs: int = 10) -> list:
    """Compute 6-dim 3D conformer features (ETKDG multi-conformer generation).

    1. n_rotatable_bonds - number of rotatable bonds
    2. conf_rmsd_mean - mean RMSD between conformers
    3. conf_rmsd_max - maximum RMSD between conformers
    4. energy_spread_kcal - MMFF energy range (kcal/mol)
    5. max_dihedral_range - maximum dihedral angle variation range
    6. asphericity - asphericity of the first conformer

    On failure: return [0.0]*6
    """
    from rdkit.Chem import AllChem, Descriptors3D, rdMolTransforms

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0] * 6
        mol = Chem.AddHs(mol)

        # 1. n_rotatable_bonds
        n_rotatable = float(rdMolDescriptors.CalcNumRotatableBonds(mol))

        # Generate multiple conformers
        params = AllChem.ETKDG()
        params.randomSeed = 42
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
        if len(conf_ids) == 0:
            # Try single conformer
            if AllChem.EmbedMolecule(mol, params) == -1:
                return [n_rotatable, 0.0, 0.0, 0.0, 0.0, 0.0]
            conf_ids = [0]

        # MMFF optimization and energy extraction
        mmff_results = AllChem.MMFFOptimizeMoleculeConfs(mol)
        energies = []
        for res in mmff_results:
            if res[0] == 0:  # converged
                energies.append(res[1])
            elif res[0] == 1:  # not converged but has energy
                energies.append(res[1])

        # 2-3. RMSD (between conformers)
        conf_rmsd_mean = 0.0
        conf_rmsd_max = 0.0
        conf_id_list = list(conf_ids)
        if len(conf_id_list) >= 2:
            rmsds = []
            for ii in range(len(conf_id_list)):
                for jj in range(ii + 1, len(conf_id_list)):
                    try:
                        rmsd_val = AllChem.GetConformerRMS(mol, conf_id_list[ii], conf_id_list[jj])
                        rmsds.append(rmsd_val)
                    except Exception:
                        pass
            if rmsds:
                conf_rmsd_mean = float(np.mean(rmsds))
                conf_rmsd_max = float(np.max(rmsds))

        # 4. energy_spread_kcal
        energy_spread = 0.0
        if len(energies) >= 2:
            energy_spread = float(max(energies) - min(energies))

        # 5. max_dihedral_range - maximum rotatable bond dihedral angle variation
        max_dihedral_range = 0.0
        if len(conf_id_list) >= 2:
            rotatable_bonds = mol.GetSubstructMatches(Chem.MolFromSmarts('[!$([NH]!@C(=O))&!D1]-&!@[!$([NH]!@C(=O))&!D1]'))
            dihedral_ranges = []
            for bond_match in rotatable_bonds:
                a_idx, b_idx = bond_match
                a_atom = mol.GetAtomWithIdx(a_idx)
                b_atom = mol.GetAtomWithIdx(b_idx)
                a_neighbors = [n.GetIdx() for n in a_atom.GetNeighbors() if n.GetIdx() != b_idx]
                b_neighbors = [n.GetIdx() for n in b_atom.GetNeighbors() if n.GetIdx() != a_idx]
                if not a_neighbors or not b_neighbors:
                    continue
                i_idx = a_neighbors[0]
                l_idx = b_neighbors[0]
                angles = []
                for cid in conf_id_list:
                    try:
                        conf = mol.GetConformer(cid)
                        angle = rdMolTransforms.GetDihedralDeg(conf, i_idx, a_idx, b_idx, l_idx)
                        angles.append(angle)
                    except Exception:
                        pass
                if len(angles) >= 2:
                    dihedral_ranges.append(max(angles) - min(angles))
            if dihedral_ranges:
                max_dihedral_range = float(max(dihedral_ranges))

        # 6. asphericity (first conformer)
        asphericity = 0.0
        try:
            asphericity = float(Descriptors3D.Asphericity(mol, confId=conf_id_list[0]))
        except Exception:
            pass

        return [n_rotatable, conf_rmsd_mean, conf_rmsd_max,
                energy_spread, max_dihedral_range, asphericity]

    except Exception:
        return [0.0] * 6


def compute_3d_features_for_all(smiles_list: list) -> dict:
    """Compute 3D conformer features for all SMILES with pickle caching.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        dict: {smiles: [6 floats]}
    """
    import pickle

    cache_path = DATA_DIR / 'conformer_features_cache.pkl'

    # Load cache
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            print(f"  [3D feature cache] Loaded {len(cache)} records")
        except Exception:
            cache = {}

    # Find SMILES that need computation
    unique_smiles = list(set(smiles_list))
    to_compute = [s for s in unique_smiles if s not in cache]
    print(f"  [3D features] Total {len(unique_smiles)} unique SMILES, "
          f"cache hits {len(unique_smiles) - len(to_compute)}, "
          f"to compute {len(to_compute)}")

    # Compute missing entries
    if to_compute:
        for smi in tqdm(to_compute, desc='Computing 3D features'):
            cache[smi] = _compute_3d_conformer_features(smi)

        # Save cache
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"  [3D feature cache] Saved {len(cache)} records -> {cache_path}")

    return cache


def attach_3d_features(data_list: list, features_dict: dict) -> None:
    """Concatenate 6D 3D conformer features to existing phys_feats (in-place modification).

    If data.phys_feats already exists (5D) -> concatenate to 11D.
    If data.phys_feats does not exist -> set to 6D.
    """
    n_attached = 0
    for data in data_list:
        smi = data.smiles if hasattr(data, 'smiles') else None
        if smi is None or smi not in features_dict:
            feats_3d = torch.tensor([[0.0] * CONF3D_FEATS_DIM], dtype=torch.float)
        else:
            feats_3d = torch.tensor([features_dict[smi]], dtype=torch.float)  # [1, 6]

        if hasattr(data, 'phys_feats') and data.phys_feats is not None:
            # Concatenate to existing phys_feats -> [1, 5+6=11]
            data.phys_feats = torch.cat([data.phys_feats, feats_3d], dim=-1)
        else:
            data.phys_feats = feats_3d
        n_attached += 1

    print(f"  [attach_3d_features] Attached 3D features to {n_attached} molecules, "
          f"phys_feats dim: {data_list[0].phys_feats.shape[-1] if data_list else '?'}")


def print_3d_feature_stats(data_list: list, client_name: str):
    """Print 3D feature statistics (mean/std per feature)."""
    feat_names = ['n_rotatable', 'rmsd_mean', 'rmsd_max',
                  'energy_spread', 'max_dihedral', 'asphericity']
    if not data_list:
        return

    all_feats = []
    for d in data_list:
        if hasattr(d, 'phys_feats') and d.phys_feats is not None:
            pf = d.phys_feats.squeeze(0).numpy()
            # Take last 6 dims (3D features)
            if len(pf) > 6:
                pf = pf[-6:]
            all_feats.append(pf)

    if not all_feats:
        return

    feats_np = np.stack(all_feats)
    print(f"\n  [{client_name}] 3D feature statistics ({len(all_feats)} molecules):")
    for i, name in enumerate(feat_names):
        if i < feats_np.shape[1]:
            print(f"    {name:20s}: mean={feats_np[:, i].mean():.4f}, std={feats_np[:, i].std():.4f}")


# ============ 3D Data for SchNet ============

def smiles_to_3d_data(smiles: str, y: Optional[float] = None) -> Optional[Data]:
    """Convert SMILES to 3D PyG Data (for SchNet).

    SMILES -> AddHs -> EmbedMolecule(ETKDGv3) -> MMFFOptimize
    Returns Data(z=atomic_numbers, pos=coordinates, y=y, smiles=smiles).
    Does not need edge_index (SchNet uses radius_graph for dynamic generation).
    """
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Try embedding 3D coordinates up to 3 times
    embedded = False
    for attempt in range(3):
        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = 42 + attempt
            result = AllChem.EmbedMolecule(mol, params)
            if result == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                except Exception:
                    pass  # Use initial conformer if optimization fails
                embedded = True
                break
        except Exception:
            continue

    if not embedded:
        return None

    # Extract atomic numbers and coordinates
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    z = torch.tensor([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(n_atoms)],
                     dtype=torch.long)
    pos = torch.tensor(
        [[conf.GetAtomPosition(i).x,
          conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z] for i in range(n_atoms)],
        dtype=torch.float
    )

    data = Data(z=z, pos=pos, smiles=smiles)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)

    return data


def prepare_all_clients_3d(target_type: str = 'hole', include_d: bool = False,
                           normalize_y: bool = False) -> dict:
    """Prepare 3D data for all clients (for SchNet).

    Same client split logic as prepare_all_clients, but uses smiles_to_3d_data()
    to generate Data(z, pos, y) instead of Data(x, edge_index, edge_attr, ...).

    With pickle cache: data/conformer_3d_cache.pkl
    """
    import pickle

    print("=" * 60)
    print(f"3D data preparation (target_type={target_type}, include_d={include_d})")
    print("=" * 60)

    cache_path = DATA_DIR / 'conformer_3d_cache.pkl'
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            print(f"  [3D cache] Loaded {len(cache)} records")
        except Exception:
            cache = {}

    def get_3d_data(smiles, y_val):
        """Get 3D data from cache or compute it."""
        cache_key = smiles
        if cache_key in cache:
            z, pos = cache[cache_key]
            data = Data(z=z.clone(), pos=pos.clone(), smiles=smiles)
            if y_val is not None:
                data.y = torch.tensor([y_val], dtype=torch.float)
            return data
        else:
            data = smiles_to_3d_data(smiles, y=y_val)
            if data is not None:
                cache[cache_key] = (data.z.clone(), data.pos.clone())
            return data

    def build_3d_dataset(smiles_list, y_list, desc='Building 3D'):
        data_list = []
        n_fail = 0
        for smi, y_val in tqdm(zip(smiles_list, y_list), total=len(smiles_list), desc=desc):
            data = get_3d_data(smi, y_val)
            if data is None:
                n_fail += 1
            else:
                data_list.append(data)
        if n_fail > 0:
            print(f"  Warning: {n_fail}/{len(smiles_list)} SMILES failed 3D embedding")
        print(f"  Successfully built {len(data_list)} 3D molecular graphs")
        return data_list

    # 1. Load public data and split
    public_df = load_public_data()
    client_a_df, client_b_df = split_public_by_aromaticity(public_df)

    # 2. Build Client A 3D graphs
    print("\n[Building Client A 3D molecular graphs]")
    client_a_graphs = build_3d_dataset(
        client_a_df['smiles'].tolist(),
        client_a_df['reorg_energy_eV'].tolist(),
        desc='Client A 3D'
    )

    # 3. Build Client B 3D graphs
    print("\n[Building Client B 3D molecular graphs]")
    client_b_graphs = build_3d_dataset(
        client_b_df['smiles'].tolist(),
        client_b_df['reorg_energy_eV'].tolist(),
        desc='Client B 3D'
    )

    # 4. Load Client C data
    if target_type == 'hole':
        c_df = load_client_c_hole()
        c_smiles = c_df['smiles'].tolist()
        c_y = c_df['lambda_hole_eV'].tolist()
    else:
        c_df = load_client_c_triplet()
        c_smiles = c_df['smiles'].tolist()
        c_y = c_df['lambda_T_total_eV'].tolist()

    print("\n[Building Client C 3D molecular graphs]")
    client_c_graphs = build_3d_dataset(c_smiles, c_y, desc='Client C 3D')

    # 5. Client D (if requested)
    client_d_graphs = []
    if include_d:
        d_df = load_client_d()
        print("\n[Building Client D 3D molecular graphs]")
        client_d_graphs = build_3d_dataset(
            d_df['smiles'].tolist(),
            d_df['reorg_eV'].tolist(),
            desc='Client D 3D'
        )

    # Save cache
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"\n  [3D cache] Saved {len(cache)} records -> {cache_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"3D data preparation complete")
    print(f"  Client A: {len(client_a_graphs)} 3D molecular graphs")
    print(f"  Client B: {len(client_b_graphs)} 3D molecular graphs")
    print(f"  Client C: {len(client_c_graphs)} 3D molecular graphs")
    if include_d:
        print(f"  Client D: {len(client_d_graphs)} 3D molecular graphs")

    # Print atom count statistics
    for name, graphs in [('A', client_a_graphs), ('B', client_b_graphs),
                          ('C', client_c_graphs), ('D', client_d_graphs)]:
        if graphs:
            n_atoms = [d.z.size(0) for d in graphs]
            print(f"  Client {name}: atom count mean={np.mean(n_atoms):.1f}, "
                  f"max={max(n_atoms)}, min={min(n_atoms)}")
    print("=" * 60)

    result = {
        'client_a': client_a_graphs,
        'client_b': client_b_graphs,
        'client_c': client_c_graphs,
        'target_type': target_type,
    }
    if include_d:
        result['client_d'] = client_d_graphs

    # Per-client label normalization
    if normalize_y:
        norm_stats = normalize_client_labels(result)
        result['norm_stats'] = norm_stats

    return result


# ============ Physical feature StandardScaler normalization ============

def normalize_phys_feats(data_lists: list) -> None:
    """StandardScaler normalization for phys_feats (in-place modification).

    Fits scaler on the training set and normalizes all data.
    data_lists: [client_a_graphs, client_b_graphs, client_c_graphs, ...]
    """
    from sklearn.preprocessing import StandardScaler

    all_feats = []
    for data_list in data_lists:
        for data in data_list:
            if hasattr(data, 'phys_feats') and data.phys_feats is not None:
                all_feats.append(data.phys_feats.squeeze(0).numpy())

    if not all_feats:
        print("  [normalize_phys_feats] No phys_feats data found, skipping")
        return

    all_feats_np = np.stack(all_feats)
    scaler = StandardScaler()
    scaler.fit(all_feats_np)

    print(f"  [normalize_phys_feats] Fitted scaler: {len(all_feats)} samples, "
          f"mean={scaler.mean_}, scale={scaler.scale_}")

    # Transform in-place
    for data_list in data_lists:
        for data in data_list:
            if hasattr(data, 'phys_feats') and data.phys_feats is not None:
                feat = data.phys_feats.squeeze(0).numpy().reshape(1, -1)
                feat_scaled = scaler.transform(feat)
                data.phys_feats = torch.tensor(feat_scaled, dtype=torch.float)
