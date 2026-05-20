"""
Hyperparameter configuration.
"""

# ============ Model ============
GIN_HIDDEN = 300
GIN_LAYERS = 5
ENCODER_OUT_DIM = 256
GIN_DROPOUT = 0.1
KAN_GRID = 5
KAN_SPLINE_ORDER = 3
MLP_DROPOUT = 0.3

# ============ Training ============
LR_PRETRAIN = 1e-3
LR_FINETUNE = 1e-4
BATCH_SIZE_AB = 512       # Client A/B (A30 24GB)
PATIENCE = 15
N_EPOCHS_LOCAL = 200      # Max epochs for local training
N_EPOCHS_FED_FINETUNE = 100  # Epochs for post-federation fine-tuning

# ============ Federation ============
N_FED_ROUNDS = 50         # Federation communication rounds
N_LOCAL_EPOCHS = 5        # Local epochs per round
FEDPROX_MU = 0.01

# ============ SSL ============
N_SSL_ROUNDS = 20         # SSL pretraining federation rounds
N_SSL_LOCAL_EPOCHS = 5
ATOM_MASK_RATIO = 0.15
EDGE_DROP_RATIO = 0.20
GRAPHCL_TEMP = 0.1

# ============ Evaluation ============
N_FOLDS_AB = 5            # K-fold CV for Client A/B
# Client C: LOOCV (49 or 53 molecules)


# ============ PC²-FedReorg (Phase 4) ============
# Defaults are OFF so existing E1-E59 experiments are completely unchanged.
# Only E66 (PC²-FedReorg) enables these via run_all.py's EXPERIMENTS dict.

USE_ADAPTER = False                     # Insert AdapterLayer between encoder and head
ADAPTER_TYPE = 'mlp'                    # 'mlp' (default; recommended for small data) or 'kan'
ADAPTER_BOTTLENECK = 128                # Bottleneck inner dim of the residual adapter
USE_CALIBRATION = False                 # Per-task-client (mu, sigma) standardization
USE_PC2_AGGREGATION = False             # Use experiments/pc2_fedreorg:pc2_fed_dispatch
TRANSFERABILITY_PATH = 'results/preverify/T_transferability.json'

# T_repr / T_head / T_adapter are produced by Phase 1 (src/transferability.py).
# Phase 4 only consumes the JSON; do not regenerate it inside training runs.
