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
