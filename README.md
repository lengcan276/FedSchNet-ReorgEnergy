# FedSchNet-ReorgEnergy

Personalized federated self-supervised learning with GIN/SchNet encoders and KAN regression heads for molecular reorganization energy prediction.

## Method Overview

This framework simulates cross-institutional collaboration using federated learning to predict reorganization energies (lambda) of organic molecules, with a focus on boosting performance for a small private TADF (Thermally Activated Delayed Fluorescence) dataset via knowledge transfer from larger public databases.

**Key components:**
- **Encoders:** GIN (2D molecular graph) and SchNet (3D coordinates)
- **Regression heads:** KAN (Kolmogorov-Arnold Networks) vs MLP baseline
- **Federation strategies:** FedAvg, FedPer, FedBN, FedProx
- **Self-supervised pretraining:** AtomMask, EdgePred, GraphCL
- **Physics augmentation:** 5D/11D molecular descriptors, HOMO/LUMO node features

## Datasets

| Client | Source | Molecules | Target | Description |
|--------|--------|-----------|--------|-------------|
| A | QM9 (public) | ~6,000 | Cation reorg. energy | Non-aromatic molecules |
| B | QM9 (public) | ~9,000 | Cation reorg. energy | Aromatic molecules |
| C | TADF (private) | 49-53 | Hole/triplet reorg. energy | Small lab dataset |
| D | Atahan-Evrenk 2019 | 5,876 | Hole reorg. energy | Conjugated organic semiconductors |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run local baselines (Batch 1)
python experiments/run_all.py --batch1

# Run core federated experiments (Batch 2)
python experiments/run_all.py --batch2

# Run all experiments
python experiments/run_all.py --all

# Run specific experiment
python experiments/run_all.py --exp E14

# Run ablation studies
python experiments/ablation.py --all

# List available experiments
python experiments/run_all.py --help
```

## Project Structure

```
FedSchNet-ReorgEnergy/
├── README.md
├── requirements.txt
├── data/
│   ├── client_a_b/public_reorg_energy_15210.csv
│   ├── client_c/{hole_reorg_molecular.csv, reorganization_energy_summary.csv}
│   └── client_d/atahan_reorg_5876.csv
├── src/
│   ├── data_utils.py       # Data loading, SMILES-to-graph, client splitting
│   ├── models.py           # GIN/SchNet encoders, KAN/MLP/Physics-KAN heads
│   ├── federated.py        # Federation strategies (FedAvg, FedPer, FedBN, FedProx)
│   ├── ssl_pretrain.py     # Self-supervised pretraining (AtomMask, EdgePred, GraphCL)
│   └── train_eval.py       # Training loops, evaluation, cross-validation
├── experiments/
│   ├── configs.py           # Hyperparameter configuration
│   ├── run_all.py           # Main experiment runner
│   └── ablation.py          # Ablation studies
├── results/
│   ├── tables/
│   │   ├── all_experiments_summary.csv
│   │   ├── ablation_kan_grid.csv
│   │   ├── ablation_ssl_rounds.csv
│   │   ├── ablation_data_size.csv
│   │   └── ablation_comm_rounds.csv
│   └── figures/
└── notebooks/
```

## Key Results

### Client C (TADF, hole lambda) - LOOCV

| Experiment | Encoder | Strategy | MAE (eV) | R² |
|------------|---------|----------|----------|----|
| E9 Local | GIN+KAN | None | 0.4103 | -0.677 |
| E15 FedBN | GIN+KAN | FedPer+FedBN | 0.3583 | 0.028 |
| E52 FedPer | SchNet+KAN | FedPer | 0.2877 | 0.139 |
| E51 FedAvg | SchNet+KAN | FedAvg | 0.3035 | 0.181 |

### Clients A/B (QM9 public, 5-fold CV)

| Experiment | MAE_A (eV) | R²_A | MAE_B (eV) | R²_B |
|------------|-----------|------|-----------|------|
| E14 FedPer GIN+KAN | 0.1425 | 0.777 | 0.0780 | 0.865 |
| E15 FedBN GIN+KAN | 0.1505 | 0.750 | 0.0808 | 0.854 |

## Hardware

- 2x NVIDIA A30 (24GB each)
- GPU 0: Client A/D training
- GPU 1: Client B/C training
- Federation aggregation on CPU

## Citation

```
TBD
```
