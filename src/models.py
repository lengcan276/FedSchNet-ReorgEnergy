"""
Phase 2: Model Definitions
- GIN Encoder (shared layers)
- KAN Head / MLP Head / Physics-KAN Head (private layers)
- Complete model: Encoder + Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool, global_max_pool
from efficient_kan import KANLinear

try:
    from .data_utils import NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_DESC_DIM, PHYSICS_FEAT_DIM, PHYS_FEATS_DIM
except ImportError:
    from data_utils import NODE_FEAT_DIM, EDGE_FEAT_DIM, GLOBAL_DESC_DIM, PHYSICS_FEAT_DIM, PHYS_FEATS_DIM


# ============ GIN Encoder ============

class GINEncoder(nn.Module):
    """5-layer GIN Encoder (federated shared layers)

    GINConv x 5, hidden=300, BatchNorm, ReLU
    JumpingKnowledge: concat all layer outputs
    Global pooling: mean + max -> concat
    Projection: Linear -> 256-dim
    """

    def __init__(self, in_dim: int = NODE_FEAT_DIM, hidden_dim: int = 300,
                 out_dim: int = 256, num_layers: int = 5, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # GIN convolution layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_channels = in_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # JumpingKnowledge: concat all layers
        self.jk = JumpingKnowledge(mode='cat', channels=hidden_dim, num_layers=num_layers)

        # Projection layer: after JK concat = hidden_dim * num_layers
        # Global pooling: mean + max -> concat -> hidden_dim * num_layers * 2
        jk_dim = hidden_dim * num_layers
        pool_dim = jk_dim * 2  # mean + max
        self.project = nn.Linear(pool_dim, out_dim)

    def forward(self, x, edge_index, batch):
        """
        Args:
            x: [N, in_dim] node features
            edge_index: [2, E] edge indices
            batch: [N] batch assignment

        Returns:
            graph_repr: [B, out_dim] graph-level representation
        """
        layer_outputs = []
        h = x

        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            layer_outputs.append(h)

        # JumpingKnowledge
        h = self.jk(layer_outputs)  # [N, hidden_dim * num_layers]

        # Global pooling: mean + max
        h_mean = global_mean_pool(h, batch)  # [B, hidden_dim * num_layers]
        h_max = global_max_pool(h, batch)    # [B, hidden_dim * num_layers]
        h_graph = torch.cat([h_mean, h_max], dim=-1)  # [B, hidden_dim * num_layers * 2]

        # Projection
        out = self.project(h_graph)  # [B, out_dim]
        return out


# ============ Regression Heads ============

class KANHead(nn.Module):
    """KAN regression head (private layer, core innovation)

    KANLinear(256, 64, grid=5, spline_order=3)
    KANLinear(64, 32, grid=5, spline_order=3)
    KANLinear(32, 1, grid=5, spline_order=3)
    """

    def __init__(self, in_dim: int = 256, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.kan1 = KANLinear(in_dim, 64, grid_size=grid_size, spline_order=spline_order)
        self.kan2 = KANLinear(64, 32, grid_size=grid_size, spline_order=spline_order)
        self.kan3 = KANLinear(32, 1, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        x = self.kan1(x)
        x = self.kan2(x)
        x = self.kan3(x)
        return x


class MLPHead(nn.Module):
    """MLP regression head (baseline comparison)

    Linear(256,128) -> ReLU -> Dropout(0.3)
    Linear(128,64) -> ReLU -> Dropout(0.3)
    Linear(64,1)
    """

    def __init__(self, in_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.mlp(x)


class PhysicsKANHead(nn.Module):
    """Physics-KAN regression head (Client C specific)

    Input: GNN output (256) + global descriptors (6) + physics features (4) = 266-dim
    KANLinear(266, 64) -> KANLinear(64, 1)
    """

    def __init__(self, gnn_dim: int = 256, global_dim: int = GLOBAL_DESC_DIM,
                 physics_dim: int = PHYSICS_FEAT_DIM, grid_size: int = 5,
                 spline_order: int = 3):
        super().__init__()
        total_dim = gnn_dim + global_dim + physics_dim  # 256 + 6 + 4 = 266
        self.kan1 = KANLinear(total_dim, 64, grid_size=grid_size, spline_order=spline_order)
        self.kan2 = KANLinear(64, 1, grid_size=grid_size, spline_order=spline_order)

    def forward(self, x):
        """x: [B, gnn_dim + global_dim + physics_dim]"""
        x = self.kan1(x)
        x = self.kan2(x)
        return x


# ============ PC²-FedReorg modules (Phase 2) ============

class AdapterLayer(nn.Module):
    """Residual bottleneck adapter: ``h_out = h + Adapter(h)``.

    Two-layer module mapping ``hidden_dim -> bottleneck_dim -> hidden_dim``.
    The final projection is initialized to zero, so the adapter starts exactly
    as identity. This guarantees that flipping ``use_adapter=False -> True``
    does not perturb predictions at initialization, and avoids destabilizing
    small-data fits (Client C-hole / C-triplet have only 49-53 molecules).

    The I/O dimension is preserved -- downstream heads keep their existing
    ``in_dim`` and existing checkpoints remain loadable with strict=False.
    """

    def __init__(self, hidden_dim: int = 256, bottleneck_dim: int = 128,
                 adapter_type: str = 'mlp', kan_grid: int = 5,
                 kan_spline_order: int = 3):
        super().__init__()
        self.adapter_type = adapter_type
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        if adapter_type == 'mlp':
            self.fc1 = nn.Linear(hidden_dim, bottleneck_dim)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(bottleneck_dim, hidden_dim)
            # Zero-init output projection so the adapter is identity at init.
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        elif adapter_type == 'kan':
            self.fc1 = KANLinear(hidden_dim, bottleneck_dim,
                                 grid_size=kan_grid, spline_order=kan_spline_order)
            self.act = nn.Identity()
            self.fc2 = KANLinear(bottleneck_dim, hidden_dim,
                                 grid_size=kan_grid, spline_order=kan_spline_order)
            # Zero-init last KAN layer's weights for identity at init.
            with torch.no_grad():
                for attr in ('base_weight', 'spline_weight', 'spline_scaler'):
                    if hasattr(self.fc2, attr):
                        param = getattr(self.fc2, attr)
                        if isinstance(param, torch.Tensor):
                            param.zero_()
        else:
            raise ValueError(f'unknown adapter_type {adapter_type!r}; '
                             f"expected 'mlp' or 'kan'")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.fc2(self.act(self.fc1(h)))


class CalibrationHead(nn.Module):
    """Per-task-client label de-standardization (frozen buffers, no params).

    Forward (inference / metric reporting):
        ``physical = sigma * z_pred + mu``
    Standardize (training target prep):
        ``z_target = (y - mu) / sigma``

    Both ``mu`` and ``sigma`` are non-trainable buffers (registered with
    ``register_buffer``); they participate in ``state_dict`` for checkpoint
    portability but never accumulate gradients.

    Critical invariants -- the model author / federated aggregator must enforce:

    1. **Never aggregate** across clients. Any state_dict key containing
       ``'calibration'`` must be filtered out by the federated aggregator
       (added to ``src/federated.py:_get_exclude_keys`` in Phase 3).
       Until that change lands, do NOT use ``use_calibration=True`` together
       with ``get_aggregation_fn('fedavg' | 'fedper' | 'fedbn')``.

    2. **Per task-client storage**. Each of {A, B, C-hole, C-triplet, D} keeps
       its own ``(mu, sigma)``. Sharing a single CalibrationHead across nodes
       defeats the purpose.

    3. **Per-fold update for LOOCV**. For C-hole / C-triplet, ``set_stats``
       MUST be called per LOOCV fold using only the train-split labels.
       The training loop is responsible; this module simply stores whatever
       last value was installed.
    """

    SIGMA_FLOOR = 1e-8

    def __init__(self, mu_init: float = 0.0, sigma_init: float = 1.0):
        super().__init__()
        self.register_buffer('mu', torch.tensor(float(mu_init)))
        self.register_buffer(
            'sigma', torch.tensor(max(float(sigma_init), self.SIGMA_FLOOR))
        )

    def set_stats(self, mu: float, sigma: float):
        """Install train-fold statistics. Must be called per LOOCV fold for
        Client C-hole / C-triplet to avoid leakage."""
        self.mu.fill_(float(mu))
        self.sigma.fill_(max(float(sigma), self.SIGMA_FLOOR))

    def standardize(self, y: torch.Tensor) -> torch.Tensor:
        """Convert physical labels (eV) to standardized z. Used for loss."""
        return (y - self.mu) / self.sigma

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """De-standardize predictions: lambda_pred = sigma * z + mu."""
        return self.sigma * z + self.mu


# ============ Complete Model ============

class ReorgEnergyModel(nn.Module):
    """Complete model: GIN Encoder + (optional) Adapter + Regression Head + (optional) Calibration.

    Args:
        head_type: 'kan', 'mlp', 'physics_kan'
        use_global_desc: whether to concatenate global descriptors to graph representation
        use_physics_feat: whether to use physics features (Client C only)
        kan_grid: KAN grid size
        phys_feats_dim: extra physics feature dim concatenated to graph repr
        use_adapter: insert a residual bottleneck adapter between encoder and head.
            Identity at initialization (zero-init output projection); does not
            change ``state_dict`` keys when False (backward-compatible with
            E1-E59 checkpoints).
        adapter_type: 'mlp' (default; recommended for small data) or 'kan'.
        adapter_bottleneck: bottleneck inner dim of the residual adapter (default 128).
        use_calibration: install a CalibrationHead with frozen ``(mu, sigma)``
            buffers for per-task-client label de-standardization.
            See :class:`CalibrationHead` for the no-aggregation contract.
    """

    def __init__(self, head_type: str = 'kan', in_dim: int = NODE_FEAT_DIM,
                 hidden_dim: int = 300, encoder_out_dim: int = 256,
                 num_layers: int = 5, dropout: float = 0.1,
                 use_global_desc: bool = True, use_physics_feat: bool = False,
                 kan_grid: int = 5, phys_feats_dim: int = 0,
                 use_adapter: bool = False, adapter_type: str = 'mlp',
                 adapter_bottleneck: int = 128,
                 use_calibration: bool = False):
        super().__init__()
        self.head_type = head_type
        self.use_global_desc = use_global_desc
        self.use_physics_feat = use_physics_feat
        self.phys_feats_dim = phys_feats_dim
        self.use_adapter = use_adapter
        self.use_calibration = use_calibration

        # Encoder
        self.encoder = GINEncoder(
            in_dim=in_dim, hidden_dim=hidden_dim,
            out_dim=encoder_out_dim, num_layers=num_layers,
            dropout=dropout
        )

        # Optional residual adapter (only created when use_adapter=True so that
        # state_dict has no 'adapter.*' keys in backward-compatible mode).
        if use_adapter:
            self.adapter = AdapterLayer(
                hidden_dim=encoder_out_dim,
                bottleneck_dim=adapter_bottleneck,
                adapter_type=adapter_type,
                kan_grid=kan_grid,
            )

        # Head input dimension (adapter is residual -> dim unchanged)
        if head_type == 'physics_kan':
            self.head = PhysicsKANHead(
                gnn_dim=encoder_out_dim, grid_size=kan_grid
            )
        elif head_type == 'kan':
            head_in = encoder_out_dim
            if use_global_desc:
                head_in += GLOBAL_DESC_DIM
            head_in += phys_feats_dim
            self.head = KANHead(in_dim=head_in, grid_size=kan_grid)
        else:  # mlp
            head_in = encoder_out_dim
            if use_global_desc:
                head_in += GLOBAL_DESC_DIM
            head_in += phys_feats_dim
            self.head = MLPHead(in_dim=head_in, dropout=0.3)

        # Optional calibration head (frozen buffers; never aggregated).
        if use_calibration:
            self.calibration = CalibrationHead()

    def forward(self, data, return_z: bool = False, return_physical: bool = False):
        """
        Args:
            data: PyG Batch object containing x, edge_index, batch, global_desc, [physics_feat]
            return_z: if True, returns the head output in standardized z-space
                (used to compute training loss against standardized targets).
            return_physical: if True, returns the prediction in physical eV-space
                (calibration applied if enabled).

        Returns:
            By default, a single tensor of shape [B, 1]:
              - if use_calibration=True: physical eV-space prediction
              - if use_calibration=False: head output (already eV-space since
                labels are not standardized).
            If both return_z and return_physical are True, returns a tuple
            ``(z, physical)``.
        """
        # Encoder
        graph_repr = self.encoder(data.x, data.edge_index, data.batch)  # [B, 256]

        # Optional residual adapter (identity at init when use_adapter=True)
        if self.use_adapter:
            graph_repr = self.adapter(graph_repr)  # [B, 256], dim unchanged

        # Feature concatenation
        if self.head_type == 'physics_kan':
            features = [graph_repr, data.global_desc]
            if hasattr(data, 'physics_feat') and data.physics_feat is not None:
                features.append(data.physics_feat)
            else:
                features.append(torch.zeros(graph_repr.size(0), PHYSICS_FEAT_DIM,
                                            device=graph_repr.device))
            h = torch.cat(features, dim=-1)
        else:
            features = [graph_repr]
            if self.use_global_desc:
                features.append(data.global_desc)
            if self.phys_feats_dim > 0:
                if hasattr(data, 'phys_feats') and data.phys_feats is not None:
                    features.append(data.phys_feats)
                else:
                    features.append(torch.zeros(graph_repr.size(0), self.phys_feats_dim,
                                                device=graph_repr.device))
            h = torch.cat(features, dim=-1)

        z_pred = self.head(h)  # [B, 1] in standardized space when use_calibration=True

        # Output routing
        if return_z and return_physical:
            phys = self.calibration(z_pred) if self.use_calibration else z_pred
            return z_pred, phys
        if return_z:
            return z_pred
        if return_physical:
            return self.calibration(z_pred) if self.use_calibration else z_pred
        # Default (backward-compatible)
        if self.use_calibration:
            return self.calibration(z_pred)
        return z_pred

    def get_encoder_params(self):
        """Return encoder parameters (used for federated aggregation)"""
        return self.encoder.parameters()

    def get_head_params(self):
        """Return head parameters (private layers)"""
        return self.head.parameters()


# ============ SchNet 3D Encoder ============

class SchNet3DEncoder(nn.Module):
    """SchNet 3D encoder -- operates on atomic numbers (z) and coordinates (pos).

    Replaces the final output MLP so that forward() returns the full
    hidden_channels-dimensional graph embedding instead of a scalar.
    """

    def __init__(self, hidden_channels: int = 256, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 7.5):
        super().__init__()
        from torch_geometric.nn import SchNet

        self.schnet = SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )
        # Replace output MLP to get full hidden_channels as features
        self.schnet.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.schnet.lin2 = nn.Identity()
        self.out_dim = hidden_channels

    def forward(self, z, pos, batch):
        """
        Args:
            z: [N] atomic numbers (long)
            pos: [N, 3] coordinates (float)
            batch: [N] batch assignment

        Returns:
            graph_repr: [B, hidden_channels]
        """
        return self.schnet(z, pos, batch)  # [B, hidden_channels]


class SchNetKANModel(nn.Module):
    """Complete model: SchNet 3D Encoder + KAN/MLP Head

    forward(data) interface matches ReorgEnergyModel.forward(data),
    so existing train_one_epoch and evaluate work unchanged.
    """

    def __init__(self, hidden_channels: int = 256, num_filters: int = 128,
                 num_interactions: int = 6, cutoff: float = 7.5,
                 head_type: str = 'kan', kan_grid: int = 5):
        super().__init__()
        self.encoder = SchNet3DEncoder(
            hidden_channels=hidden_channels, num_filters=num_filters,
            num_interactions=num_interactions, cutoff=cutoff,
        )
        if head_type == 'kan':
            self.head = KANHead(hidden_channels, grid_size=kan_grid)
        else:
            self.head = MLPHead(hidden_channels)

    def forward(self, data):
        h = self.encoder(data.z, data.pos, data.batch)
        return self.head(h)


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    return {
        'total': total,
        'trainable': trainable,
        'encoder': encoder_params,
        'head': head_params,
    }
