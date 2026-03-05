"""
Self-supervised pretraining.
- AtomMask: Mask nodes -> predict atom type + Gasteiger charge
- EdgePred: Drop edges -> predict edge existence
- GraphCL:  Graph contrastive learning (InfoNCE)
- Federated SSL training loop
"""

import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

try:
    from .data_utils import NODE_FEAT_DIM, GLOBAL_DESC_DIM
    from .models import GINEncoder
except ImportError:
    from data_utils import NODE_FEAT_DIM, GLOBAL_DESC_DIM
    from models import GINEncoder

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============ SSL Pretraining Heads ============

class AtomMaskHead(nn.Module):
    """AtomMask pretraining head: predict masked node atom type and Gasteiger charge."""

    def __init__(self, hidden_dim: int = 300, num_atom_types: int = 8):
        super().__init__()
        # Input is the node representation from the last GIN layer (hidden_dim)
        self.atom_type_pred = nn.Linear(hidden_dim, num_atom_types)
        self.charge_pred = nn.Linear(hidden_dim, 1)

    def forward(self, node_repr):
        """node_repr: [N_masked, hidden_dim]"""
        atom_logits = self.atom_type_pred(node_repr)  # [N_masked, 8]
        charge = self.charge_pred(node_repr)            # [N_masked, 1]
        return atom_logits, charge.squeeze(-1)


class EdgePredHead(nn.Module):
    """EdgePred pretraining head: predict whether an edge exists."""

    def __init__(self, hidden_dim: int = 300):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_repr, edge_pairs):
        """
        node_repr: [N, hidden_dim]
        edge_pairs: [M, 2] node pair indices
        """
        src = node_repr[edge_pairs[:, 0]]  # [M, hidden_dim]
        dst = node_repr[edge_pairs[:, 1]]  # [M, hidden_dim]
        pair_repr = torch.cat([src, dst], dim=-1)  # [M, hidden_dim*2]
        return self.pred(pair_repr).squeeze(-1)  # [M]


# ============ SSL Encoder (no global pooling, returns node-level representations) ============

class SSLEncoder(nn.Module):
    """SSL-specific Encoder: contains GINEncoder + SSL pretraining head.

    Shares the same architecture as the GINEncoder in ReorgEnergyModel,
    but additionally provides node-level output for SSL tasks.
    """

    def __init__(self, in_dim: int = NODE_FEAT_DIM, hidden_dim: int = 300,
                 out_dim: int = 256, num_layers: int = 5, dropout: float = 0.1):
        super().__init__()
        self.encoder = GINEncoder(
            in_dim=in_dim, hidden_dim=hidden_dim,
            out_dim=out_dim, num_layers=num_layers, dropout=dropout
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward_node(self, x, edge_index, batch):
        """Return node representations from the last GIN layer [N, hidden_dim]."""
        h = x
        for i in range(self.encoder.num_layers):
            h = self.encoder.convs[i](h, edge_index)
            h = self.encoder.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.encoder.dropout, training=self.training)
        return h  # [N, hidden_dim]

    def forward_graph(self, x, edge_index, batch):
        """Return graph-level representations [B, out_dim]."""
        return self.encoder(x, edge_index, batch)

    def get_encoder_state_dict(self):
        """Get encoder state_dict, can be directly loaded into ReorgEnergyModel's encoder."""
        return self.encoder.state_dict()


# ============ AtomMask SSL ============

def atom_mask_augment(data: Data, mask_ratio: float = 0.15):
    """Mask node features and return augmented data + original labels.

    Mask 15% of nodes: set features to zero.
    Labels: atom type (argmax of first 8 dims) + Gasteiger charge (dim 11, index=10)
    """
    data = data.clone()
    n_nodes = data.x.size(0)
    n_mask = max(1, int(n_nodes * mask_ratio))

    # Randomly select nodes to mask
    perm = torch.randperm(n_nodes)
    mask_idx = perm[:n_mask]

    # Save original labels
    atom_types = data.x[mask_idx, :8].argmax(dim=-1)  # [n_mask]
    charges = data.x[mask_idx, 10].clone()  # [n_mask] Gasteiger charge

    # Mask node features (set to zero)
    data.x[mask_idx] = 0.0

    return data, mask_idx, atom_types, charges


class AtomMaskSSL:
    """AtomMask self-supervised learning."""

    def __init__(self, ssl_encoder: SSLEncoder, hidden_dim: int = 300,
                 device: torch.device = torch.device('cpu')):
        self.ssl_encoder = ssl_encoder.to(device)
        self.head = AtomMaskHead(hidden_dim=hidden_dim).to(device)
        self.device = device
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def compute_loss(self, data: Data):
        """Compute AtomMask loss: CE(atom_type) + 0.1 * MSE(charge)"""
        data_aug, mask_idx, atom_types, charges = atom_mask_augment(data.clone())
        data_aug = data_aug.to(self.device)
        mask_idx = mask_idx.to(self.device)
        atom_types = atom_types.to(self.device)
        charges = charges.to(self.device)

        # Forward pass to get node representations
        node_repr = self.ssl_encoder.forward_node(
            data_aug.x, data_aug.edge_index, data_aug.batch
        )

        # Get representations of masked nodes
        masked_repr = node_repr[mask_idx]  # [n_mask, hidden_dim]

        # Predict
        atom_logits, charge_pred = self.head(masked_repr)

        # Loss
        loss_ce = self.ce_loss(atom_logits, atom_types)
        loss_mse = self.mse_loss(charge_pred, charges)
        loss = loss_ce + 0.1 * loss_mse

        return loss

    def parameters(self):
        return list(self.ssl_encoder.parameters()) + list(self.head.parameters())


# ============ EdgePred SSL ============

def edge_pred_augment(data: Data, drop_ratio: float = 0.20):
    """Drop edges and generate positive/negative samples.

    Positive samples: dropped edges
    Negative samples: equal number of non-existing edges (randomly sampled)
    """
    data = data.clone()
    n_nodes = data.x.size(0)
    edge_index = data.edge_index  # [2, E]
    n_edges = edge_index.size(1)

    # Edges are stored bidirectionally, process as pairs
    n_pairs = n_edges // 2
    n_drop = max(1, int(n_pairs * drop_ratio))

    # Select edge pairs to drop
    perm = torch.randperm(n_pairs)
    drop_pair_idx = perm[:n_drop]

    # Bidirectional edge indices: pair i corresponds to edge 2*i and 2*i+1
    drop_edge_idx = []
    for pi in drop_pair_idx:
        drop_edge_idx.extend([pi.item() * 2, pi.item() * 2 + 1])
    drop_edge_idx = torch.tensor(drop_edge_idx, dtype=torch.long)

    # Positive samples (dropped edges, take one direction only)
    pos_edges = edge_index[:, drop_pair_idx * 2].t()  # [n_drop, 2]

    # Remove from edge_index
    keep_mask = torch.ones(n_edges, dtype=torch.bool)
    keep_mask[drop_edge_idx] = False
    data.edge_index = edge_index[:, keep_mask]
    if data.edge_attr is not None and data.edge_attr.size(0) == n_edges:
        data.edge_attr = data.edge_attr[keep_mask]

    # Negative samples: randomly sample non-existing edges
    existing_edges = set()
    for i in range(edge_index.size(1)):
        existing_edges.add((edge_index[0, i].item(), edge_index[1, i].item()))

    neg_edges = []
    max_attempts = n_drop * 10
    attempts = 0
    while len(neg_edges) < n_drop and attempts < max_attempts:
        u = torch.randint(0, n_nodes, (1,)).item()
        v = torch.randint(0, n_nodes, (1,)).item()
        if u != v and (u, v) not in existing_edges:
            neg_edges.append([u, v])
            existing_edges.add((u, v))
        attempts += 1

    # If not enough negative samples, truncate positive samples to match
    n_neg = len(neg_edges)
    if n_neg < n_drop:
        pos_edges = pos_edges[:n_neg]
    neg_edges = torch.tensor(neg_edges, dtype=torch.long) if neg_edges else torch.zeros((0, 2), dtype=torch.long)

    return data, pos_edges, neg_edges


class EdgePredSSL:
    """EdgePred self-supervised learning."""

    def __init__(self, ssl_encoder: SSLEncoder, hidden_dim: int = 300,
                 device: torch.device = torch.device('cpu')):
        self.ssl_encoder = ssl_encoder.to(device)
        self.head = EdgePredHead(hidden_dim=hidden_dim).to(device)
        self.device = device

    def compute_loss(self, data: Data):
        """Compute EdgePred BCE loss."""
        data_aug, pos_edges, neg_edges = edge_pred_augment(data.clone())

        if pos_edges.size(0) == 0 or neg_edges.size(0) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        data_aug = data_aug.to(self.device)
        pos_edges = pos_edges.to(self.device)
        neg_edges = neg_edges.to(self.device)

        # Forward pass
        node_repr = self.ssl_encoder.forward_node(
            data_aug.x, data_aug.edge_index, data_aug.batch
        )

        # Predict for positive and negative samples
        all_edges = torch.cat([pos_edges, neg_edges], dim=0)  # [2*n, 2]
        pred = self.head(node_repr, all_edges)  # [2*n]

        # Labels
        labels = torch.cat([
            torch.ones(pos_edges.size(0), device=self.device),
            torch.zeros(neg_edges.size(0), device=self.device),
        ])

        loss = F.binary_cross_entropy_with_logits(pred, labels)
        return loss

    def parameters(self):
        return list(self.ssl_encoder.parameters()) + list(self.head.parameters())


# ============ GraphCL SSL ============

def graph_cl_augment_mask(data: Data, mask_ratio: float = 0.20):
    """Augmentation 1: mask 20% of node features."""
    data = data.clone()
    n_nodes = data.x.size(0)
    n_mask = max(1, int(n_nodes * mask_ratio))
    perm = torch.randperm(n_nodes)
    mask_idx = perm[:n_mask]
    data.x[mask_idx] = 0.0
    return data


def graph_cl_augment_edge_drop(data: Data, drop_ratio: float = 0.20):
    """Augmentation 2: drop 20% of edges."""
    data = data.clone()
    n_edges = data.edge_index.size(1)
    n_drop = max(1, int(n_edges * drop_ratio))
    perm = torch.randperm(n_edges)
    keep_idx = perm[n_drop:]
    data.edge_index = data.edge_index[:, keep_idx]
    if data.edge_attr is not None and data.edge_attr.size(0) == n_edges:
        data.edge_attr = data.edge_attr[keep_idx]
    return data


class GraphCLSSL:
    """GraphCL contrastive learning."""

    def __init__(self, ssl_encoder: SSLEncoder, out_dim: int = 256,
                 temperature: float = 0.1,
                 device: torch.device = torch.device('cpu')):
        self.ssl_encoder = ssl_encoder.to(device)
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 128),
        ).to(device)
        self.temperature = temperature
        self.device = device

    def compute_loss(self, data: Data):
        """Compute InfoNCE loss.

        Augmentation 1: mask 20% of node features
        Augmentation 2: drop 20% of edges
        """
        # Two augmentations
        data1 = graph_cl_augment_mask(data.clone()).to(self.device)
        data2 = graph_cl_augment_edge_drop(data.clone()).to(self.device)

        # Get graph representations
        z1 = self.ssl_encoder.forward_graph(data1.x, data1.edge_index, data1.batch)
        z2 = self.ssl_encoder.forward_graph(data2.x, data2.edge_index, data2.batch)

        # Project
        z1 = self.projector(z1)  # [B, 128]
        z2 = self.projector(z2)  # [B, 128]

        # L2 normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        batch_size = z1.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # InfoNCE
        # Concatenate all representations
        z = torch.cat([z1, z2], dim=0)  # [2B, 128]
        sim = torch.mm(z, z.t()) / self.temperature  # [2B, 2B]

        # Construct labels: positive pairs (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size),
        ]).to(self.device)

        # Remove self-similarity diagonal
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=self.device)
        sim = sim.masked_fill(~mask, float('-inf'))

        loss = F.cross_entropy(sim, labels)
        return loss

    def parameters(self):
        return list(self.ssl_encoder.parameters()) + list(self.projector.parameters())


# ============ Federated SSL Training ============

def fedavg_ssl_aggregate(client_encoders: list, weights: list, exclude_bn: bool = True):
    """FedAvg aggregation of SSL encoder parameters (excluding BN layers).

    Args:
        client_encoders: list of SSLEncoder for each client
        weights: aggregation weight list
        exclude_bn: whether to exclude BatchNorm parameters

    Returns:
        Aggregated state_dict
    """
    state_dicts = [enc.encoder.state_dict() for enc in client_encoders]
    avg_dict = {}

    for key in state_dicts[0]:
        # Exclude BN layers
        if exclude_bn and ('bn' in key or 'norm' in key):
            continue

        avg_dict[key] = sum(
            w * sd[key].float() for w, sd in zip(weights, state_dicts)
        )

    return avg_dict


def distribute_ssl_params(client_encoders: list, avg_dict: dict):
    """Distribute aggregated parameters to each client."""
    for enc in client_encoders:
        local_dict = enc.encoder.state_dict()
        for key in avg_dict:
            local_dict[key] = avg_dict[key].clone()
        enc.encoder.load_state_dict(local_dict)


def train_ssl_one_epoch(ssl_task, dataloader, optimizer, device):
    """Train one epoch of the SSL task."""
    ssl_task.ssl_encoder.train()
    total_loss = 0.0
    n_batches = 0

    for batch_data in dataloader:
        batch_data = batch_data.to(device)
        optimizer.zero_grad()
        loss = ssl_task.compute_loss(batch_data)
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ssl_task.parameters(), max_norm=5.0)
            optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _get_ssl_device_map(client_names: list,
                        device_a: torch.device,
                        device_b: torch.device) -> dict:
    """GPU assignment for SSL training: A,D -> device_a; B,C -> device_b."""
    device_map = {}
    for name in client_names:
        if name in ('client_a', 'client_d'):
            device_map[name] = device_a
        else:
            device_map[name] = device_b
    return device_map


def federated_ssl_pretrain(
    client_data: dict,
    ssl_method: str = 'atom_mask',
    n_rounds: int = 20,
    n_local_epochs: int = 5,
    lr: float = 1e-3,
    batch_size_ab: int = 512,
    hidden_dim: int = 300,
    out_dim: int = 256,
    num_layers: int = 5,
    device_a: torch.device = torch.device('cuda:0'),
    device_b: torch.device = torch.device('cuda:1'),
    save_path: str = None,
):
    """Federated self-supervised pretraining (supports N clients).

    Args:
        client_data: {'client_a': [...], 'client_b': [...], 'client_c': [...], optionally 'client_d': [...]}
        ssl_method: 'atom_mask', 'edge_pred', 'graph_cl'
        n_rounds: number of federated communication rounds
        n_local_epochs: number of local training epochs per round
        lr: learning rate
        batch_size_ab: batch size for Client A/B/D
        device_a: GPU 0
        device_b: GPU 1

    Returns:
        pretrained encoder state_dict, loss history
    """
    # Identify clients participating in SSL (exclude non-graph data keys)
    client_keys = [k for k in client_data if k.startswith('client_') and
                   k not in ('client_c_conformers',) and client_data[k]]

    print(f"\n{'='*60}")
    print(f"Federated SSL pretraining: method={ssl_method}, rounds={n_rounds}, "
          f"local_epochs={n_local_epochs}, clients={client_keys}")
    print(f"{'='*60}")

    # GPU assignment
    device_map = _get_ssl_device_map(client_keys, device_a, device_b)

    # DataLoaders
    loaders = {}
    for name in client_keys:
        data_list = client_data[name]
        if name in ('client_c',):
            bs = min(len(data_list), 64)
        else:
            bs = batch_size_ab
        loaders[name] = DataLoader(data_list, batch_size=bs,
                                   shuffle=True, drop_last=False)

    # Create SSLEncoder for each client
    def make_encoder():
        return SSLEncoder(in_dim=NODE_FEAT_DIM, hidden_dim=hidden_dim,
                          out_dim=out_dim, num_layers=num_layers)

    encoders = {}
    first_name = client_keys[0]
    encoders[first_name] = make_encoder()
    init_dict = encoders[first_name].encoder.state_dict()
    for name in client_keys[1:]:
        encoders[name] = make_encoder()
        encoders[name].encoder.load_state_dict(copy.deepcopy(init_dict))

    # Create SSL tasks
    def make_ssl_task(encoder, device):
        if ssl_method == 'atom_mask':
            return AtomMaskSSL(encoder, hidden_dim=hidden_dim, device=device)
        elif ssl_method == 'edge_pred':
            return EdgePredSSL(encoder, hidden_dim=hidden_dim, device=device)
        elif ssl_method == 'graph_cl':
            return GraphCLSSL(encoder, out_dim=out_dim, device=device)
        else:
            raise ValueError(f"Unknown SSL method: {ssl_method}")

    ssl_tasks = {name: make_ssl_task(encoders[name], device_map[name])
                 for name in client_keys}

    # Optimizers
    optimizers = {name: torch.optim.Adam(ssl_tasks[name].parameters(), lr=lr)
                  for name in client_keys}

    # Aggregation weights (weighted by data volume)
    counts = {name: len(client_data[name]) for name in client_keys}
    total = sum(counts.values())
    weights_list = [counts[name] / total for name in client_keys]
    weight_str = ', '.join(f"{name.replace('client_', '').upper()}={w:.3f}"
                           for name, w in zip(client_keys, weights_list))
    print(f"Aggregation weights: {weight_str}")

    # Training loop
    loss_history = {name: [] for name in client_keys}
    loss_history['round'] = []
    start_time = time.time()

    for rnd in range(1, n_rounds + 1):
        # Local training
        loss_sums = {name: 0.0 for name in client_keys}
        for ep in range(n_local_epochs):
            for name in client_keys:
                loss_sums[name] += train_ssl_one_epoch(
                    ssl_tasks[name], loaders[name],
                    optimizers[name], device_map[name])

        avg_losses = {name: loss_sums[name] / n_local_epochs for name in client_keys}

        # FedAvg aggregate encoder parameters (excluding BN)
        encoders_cpu = [copy.deepcopy(encoders[name]).cpu() for name in client_keys]
        avg_dict = fedavg_ssl_aggregate(encoders_cpu, weights_list, exclude_bn=True)

        # Distribute aggregated parameters
        distribute_ssl_params([encoders[name] for name in client_keys], avg_dict)

        # Record
        loss_history['round'].append(rnd)
        for name in client_keys:
            loss_history[name].append(avg_losses[name])

        elapsed = time.time() - start_time
        loss_str = '  '.join(f"{name.replace('client_', '').upper()}={avg_losses[name]:.4f}"
                             for name in client_keys)
        print(f"  Round {rnd:2d}/{n_rounds} | Loss {loss_str} | Time={elapsed:.0f}s")

    # Save pretrained encoder
    if save_path is None:
        save_path = str(CHECKPOINT_DIR / f'pretrained_encoder_{ssl_method}.pth')

    # Save using the first client's aggregated encoder
    final_state = encoders[first_name].get_encoder_state_dict()
    torch.save({
        'encoder_state_dict': final_state,
        'ssl_method': ssl_method,
        'n_rounds': n_rounds,
        'loss_history': loss_history,
    }, save_path)
    print(f"\nPretrained encoder saved: {save_path}")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.1f}s")

    return final_state, loss_history
