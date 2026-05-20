# PC²-FedReorg transferability matrices (Phase 1)

Computed from `T_chemistry.json` (Phase 0) and `PROTOCOL_META`
via `src/transferability.py:compute_T_matrices`. **No post-hoc tuning** — 
parameters are locked at w_K=0.6, w_S=0.4, alpha_W=5.0, gamma_L=1.0, cutoff=0.01.

## Protocol metadata

| node | quantity | source_id | functional | basis | geometry | charge_state | conformers |
|---|---|---|---|---|---|---|---|
| A | reorg_energy_general | qm9_public_reorg_15210 | unknown | unknown | unknown | unknown | unknown |
| B | reorg_energy_general | qm9_public_reorg_15210 | unknown | unknown | unknown | unknown | unknown |
| C-hole | hole_lambda | client_c_dft | wB97XD | unknown_in_csv | adiabatic_4point_nelsen | neutral_to_cation | one_per_molecule |
| C-triplet | triplet_lambda | client_c_dft | wB97XD | unknown_in_csv | adiabatic_4point_nelsen | neutral_to_T1 | one_per_molecule |
| D | hole_lambda | atahan_2019 | B3LYP | 6-31G* | unknown | unknown | unknown |

## T_repr (rows = source i, cols = target j)

| i \\ j | A | B | C-hole | C-triplet | D |
|---|---:|---:|---:|---:|---:|
| A | 1.0000 | 0.1473 | 0.0695 | 0.0690 | 0.0673 |
| B | 0.1556 | 1.0000 | 0.0854 | 0.0842 | 0.1201 |
| C-hole | 0.1461 | 0.1435 | 1.0000 | 0.9831 | 0.0744 |
| C-triplet | 0.1451 | 0.1439 | 1.0000 | 1.0000 | 0.0745 |
| D | 0.1132 | 0.1893 | 0.0546 | 0.0546 | 1.0000 |

## T_head (rows = source i, cols = target j)

| i \\ j | A | B | C-hole | C-triplet | D |
|---|---:|---:|---:|---:|---:|
| A | 1.0000 | 0.3321 | 0.0000 | 0.0000 | 0.0000 |
| B | 0.3321 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| C-hole | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| C-triplet | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| D | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

## T_adapter (rows = source i, cols = target j)

| i \\ j | A | B | C-hole | C-triplet | D |
|---|---:|---:|---:|---:|---:|
| A | 1.0000 | 0.2211 | 0.0010 | 0.0010 | 0.0010 |
| B | 0.2273 | 1.0000 | 0.0010 | 0.0010 | 0.0010 |
| C-hole | 0.0010 | 0.0010 | 1.0000 | 0.0010 | 0.0010 |
| C-triplet | 0.0010 | 0.0010 | 0.0010 | 1.0000 | 0.0010 |
| D | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 1.0000 |

## Why each off-diagonal T_head is 0

| src | dst | T_head | reason |
|---|---|---:|---|
| A | B | 0.3321 | (compatible; T_head from L_ij) |
| A | C-hole | 0.0000 | clause 1: quantity mismatch |
| A | C-triplet | 0.0000 | clause 1: quantity mismatch |
| A | D | 0.0000 | clause 1: quantity mismatch |
| B | A | 0.3321 | (compatible; T_head from L_ij) |
| B | C-hole | 0.0000 | clause 1: quantity mismatch |
| B | C-triplet | 0.0000 | clause 1: quantity mismatch |
| B | D | 0.0000 | clause 1: quantity mismatch |
| C-hole | A | 0.0000 | clause 1: quantity mismatch |
| C-hole | B | 0.0000 | clause 1: quantity mismatch |
| C-hole | C-triplet | 0.0000 | clause 1: quantity mismatch |
| C-hole | D | 0.0000 | clause 3: functional differs (wB97XD vs B3LYP) |
| C-triplet | A | 0.0000 | clause 1: quantity mismatch |
| C-triplet | B | 0.0000 | clause 1: quantity mismatch |
| C-triplet | C-hole | 0.0000 | clause 1: quantity mismatch |
| C-triplet | D | 0.0000 | clause 1: quantity mismatch |
| D | A | 0.0000 | clause 1: quantity mismatch |
| D | B | 0.0000 | clause 1: quantity mismatch |
| D | C-hole | 0.0000 | clause 4: V1=FAIL hard rule (D -> C) |
| D | C-triplet | 0.0000 | clause 1: quantity mismatch |

## Locked invariants (must hold; verified in tests/test_transferability.py)

- T_head[D, C-hole] == 0  (clause 4: V1=FAIL hard rule)
- T_head[D, C-triplet] == 0  (clause 1: quantity mismatch)
- T_head[C-hole, C-triplet] == 0  (clause 1)
- T_head[C-triplet, C-hole] == 0  (clause 1)
- T_repr[C-hole, C-triplet] > 0.95
- T_repr[D, C-hole] < 0.10
- T_repr[B, D] > 0.05  AND  T_head[B, D] == 0  (chemistry-share OK; head-share NOT OK)
