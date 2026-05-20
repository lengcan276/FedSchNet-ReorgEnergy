"""Build pseudo-federations from `data/client_a_b/public_reorg_energy_15210.csv`.

A *pseudo-federation* is a set of synthetic clients, each holding a disjoint
subset of QM9-derived molecules and (optionally) an affine label transform.
The transforms create controlled label-scale heterogeneity that mirrors the
real-world cross-protocol heterogeneity in the main-paper federation, while
keeping the molecular substrate identical so any observed calibration effect
cannot be explained by chemistry drift.

This module does **not** touch `src/`, `experiments/`, or any existing
`results/*` file; it only consumes `src/data_utils.smiles_to_graph` and
`src/data_utils.load_public_data` via Python import.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

# Lazy torch import so importing this module does not require GPU.
def _torch_data_module():
    import torch  # noqa: F401
    from src.data_utils import smiles_to_graph  # noqa: F401
    from src.data_utils import load_public_data  # noqa: F401
    return smiles_to_graph, load_public_data


@dataclass
class LabelTransform:
    """A reproducible affine + noise label transform y' = scale*y + shift + N(0, noise_std).

    `noise_std` is in raw label units (eV), applied AFTER scale+shift. All
    transform parameters are recorded verbatim in the per-run config JSON so
    the synthetic-heterogeneity setting can be reproduced.
    """
    name: str
    scale: float = 1.0
    shift: float = 0.0
    noise_std: float = 0.0

    def apply(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = self.scale * y + self.shift
        if self.noise_std > 0.0:
            out = out + rng.normal(0.0, self.noise_std, size=out.shape)
        return out.astype(np.float32)

    def to_dict(self) -> dict:
        return {"name": self.name, "scale": self.scale, "shift": self.shift,
                "noise_std": self.noise_std}


@dataclass
class PseudoClient:
    """A single pseudo-federation client.

    `data_list` is a list of PyG `Data` objects (graph + label `y` after
    transform). `is_target` flags the small private client whose
    generalization performance is the primary outcome.
    """
    name: str
    transform: LabelTransform
    smiles: list[str]
    raw_y: np.ndarray
    transformed_y: np.ndarray
    data_list: list = field(default_factory=list)
    is_target: bool = False

    def __len__(self) -> int:
        return len(self.data_list)


def _stable_hash(*items) -> int:
    """Deterministic 32-bit hash of the arguments for use as an RNG seed."""
    h = hashlib.sha256("|".join(str(x) for x in items).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def build_label_scale_federation(
    n_target: int = 100,
    n_source: int = 200,
    n_source_clients: int = 4,
    seed: int = 42,
    public_csv: Optional[str] = None,
) -> list[PseudoClient]:
    """Build the Phase-1 label-scale stress federation.

    Layout (default):
        target  S1: n_target molecules, transform y → y         (identity)
        source  S2: n_source molecules, transform y → 2 y + 0.5
        source  S3: n_source molecules, transform y → 0.5 y - 0.2
        source  S4: n_source molecules, transform y → 3 y + 1.0
        source  S5: n_source molecules, transform y → y + N(0, 0.05) (label noise)

    Molecule partitions are DISJOINT, drawn from `public_reorg_energy_15210.csv`.
    Seed governs both the random partition and any noise injection.
    """
    smiles_to_graph, load_public_data = _torch_data_module()

    df = load_public_data() if public_csv is None else pd.read_csv(public_csv)
    df = df.dropna(subset=["smiles", "reorg_energy_eV"]).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    n_total_needed = n_target + n_source * n_source_clients
    if n_total_needed > len(df):
        raise ValueError(
            f"Need {n_total_needed} disjoint molecules but only {len(df)} available"
        )
    perm = rng.permutation(len(df))[:n_total_needed]

    transforms = [
        LabelTransform("S1_target",   scale=1.0, shift=0.0,  noise_std=0.0),
        LabelTransform("S2_scale2",   scale=2.0, shift=0.5,  noise_std=0.0),
        LabelTransform("S3_shrink",   scale=0.5, shift=-0.2, noise_std=0.0),
        LabelTransform("S4_scale3",   scale=3.0, shift=1.0,  noise_std=0.0),
        LabelTransform("S5_noise",    scale=1.0, shift=0.0,  noise_std=0.05),
    ]
    if n_source_clients != 4:
        # Truncate / extend if a different topology is requested.
        transforms = transforms[: 1 + n_source_clients]

    clients: list[PseudoClient] = []
    cursor = 0
    for i, tr in enumerate(transforms):
        n = n_target if i == 0 else n_source
        idx = perm[cursor: cursor + n]
        cursor += n
        rows = df.iloc[idx].reset_index(drop=True)
        smiles_list = rows["smiles"].astype(str).tolist()
        y_raw = rows["reorg_energy_eV"].astype(np.float32).to_numpy()
        # Use a derived RNG for noise application so the transform itself is
        # reproducible independently of the molecule-selection RNG.
        noise_rng = np.random.default_rng(_stable_hash(seed, tr.name))
        y_t = tr.apply(y_raw, noise_rng)

        data_list = []
        n_fail = 0
        for smi, y in zip(smiles_list, y_t):
            d = smiles_to_graph(smi, y=float(y))
            if d is None:
                n_fail += 1
                continue
            data_list.append(d)
        if n_fail > 0:
            print(f"[pseudo_federation] {tr.name}: {n_fail} smiles_to_graph failures (skipped)")
        clients.append(PseudoClient(
            name=tr.name, transform=tr, smiles=smiles_list,
            raw_y=y_raw, transformed_y=y_t, data_list=data_list,
            is_target=(i == 0),
        ))
    return clients


def build_pseudo_task_set(
    n_per_task: int = 50,
    n_tasks: int = 10,
    seed: int = 42,
    public_csv: Optional[str] = None,
    bin_strategy: str = "label_quantile",
) -> list[PseudoClient]:
    """Build the Phase-2 pseudo-task multi-target set.

    Returns a list of pseudo-clients, each acting as a *pseudo target* with
    its own n_per_task molecules sampled to occupy a particular slice of the
    label distribution (quantile bins) so the targets differ in label mean
    but share chemistry substrate. Per-task label transforms are mild
    (scale=1.0, shift=0.0); the heterogeneity in this phase is in the
    *target* MAE scale, not in the synthetic transform.
    """
    smiles_to_graph, load_public_data = _torch_data_module()
    df = load_public_data() if public_csv is None else pd.read_csv(public_csv)
    df = df.dropna(subset=["smiles", "reorg_energy_eV"]).reset_index(drop=True)
    df = df.sort_values("reorg_energy_eV").reset_index(drop=True)
    n = len(df)

    rng = np.random.default_rng(seed)
    clients: list[PseudoClient] = []
    if bin_strategy == "label_quantile":
        # Cut the label range into n_tasks ordered slices and sample
        # n_per_task molecules from each slice (without replacement).
        edges = np.linspace(0, n, n_tasks + 1, dtype=int)
        for i in range(n_tasks):
            slab_idx = np.arange(edges[i], edges[i + 1])
            chosen = rng.choice(slab_idx, size=min(n_per_task, len(slab_idx)),
                                replace=False)
            rows = df.iloc[chosen].reset_index(drop=True)
            smiles_list = rows["smiles"].astype(str).tolist()
            y_raw = rows["reorg_energy_eV"].astype(np.float32).to_numpy()
            data_list = [smiles_to_graph(s, y=float(yv))
                         for s, yv in zip(smiles_list, y_raw)]
            data_list = [d for d in data_list if d is not None]
            tr = LabelTransform(f"T{i:02d}_quantile{i}", scale=1.0,
                                shift=0.0, noise_std=0.0)
            clients.append(PseudoClient(
                name=tr.name, transform=tr, smiles=smiles_list,
                raw_y=y_raw, transformed_y=y_raw, data_list=data_list,
                is_target=True,
            ))
    else:
        raise NotImplementedError(f"bin_strategy={bin_strategy!r} not implemented")
    return clients


def label_scale_ratio(target: PseudoClient, source: PseudoClient) -> float:
    """Quantitative label-scale mismatch metric: |σ_source / σ_target|.

    Used to plot calibration's effect size against heterogeneity strength.
    """
    s_t = float(np.std(target.transformed_y))
    s_s = float(np.std(source.transformed_y))
    if s_t == 0:
        return float("inf") if s_s != 0 else 1.0
    return s_s / s_t
