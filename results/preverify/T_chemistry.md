# V3-allpairs — 5×5 task-client chemistry overlap (Phase 0)

Inputs to PC²-FedReorg `T_repr[i,j] = max(eps, w_K * K[i,j] + w_S * S[i,j])`.
Diagonals are 1.0 by definition.

## Node sizes

| node | n (kept) | unique scaffolds |
|---|---:|---:|
| A | 6020 | 631 |
| B | 9190 | 680 |
| C-hole | 53 | 4 |
| C-triplet | 49 | 4 |
| D | 5876 | 5873 |

## K (Tanimoto similarity, source-mean of max) matrix (rows = source i, cols = target j)

| i \\ j | A | B | C-hole | C-triplet | D |
|---|---:|---:|---:|---:|---:|
| A | 1.0000 | 0.2455 | 0.1158 | 0.1150 | 0.1122 |
| B | 0.2593 | 1.0000 | 0.1423 | 0.1404 | 0.1883 |
| C-hole | 0.2435 | 0.2392 | 1.0000 | 0.9718 | 0.1241 |
| C-triplet | 0.2418 | 0.2399 | 1.0000 | 1.0000 | 0.1241 |
| D | 0.1887 | 0.3141 | 0.0910 | 0.0909 | 1.0000 |

## S (Murcko scaffold overlap, source-normalized) matrix (rows = source i, cols = target j)

| i \\ j | A | B | C-hole | C-triplet | D |
|---|---:|---:|---:|---:|---:|
| A | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| B | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0176 |
| C-hole | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| C-triplet | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| D | 0.0000 | 0.0020 | 0.0000 | 0.0000 | 1.0000 |

## T_repr preview (defaults: w_K=0.6, w_S=0.4, floor=1e-3)

| i \\ j | A | B | C-hole | C-triplet | D |
|---|---:|---:|---:|---:|---:|
| A | 1.0000 | 0.1473 | 0.0695 | 0.0690 | 0.0673 |
| B | 0.1556 | 1.0000 | 0.0854 | 0.0842 | 0.1201 |
| C-hole | 0.1461 | 0.1435 | 1.0000 | 0.9831 | 0.0744 |
| C-triplet | 0.1451 | 0.1439 | 1.0000 | 1.0000 | 0.0745 |
| D | 0.1132 | 0.1893 | 0.0546 | 0.0546 | 1.0000 |

## Sanity checks (locked algorithmic invariants)

Independent of the K/S numbers above, these T_head invariants must hold:
- T_head[D, C-hole] = 0 (clauses 2 + 4: D protocol unknown + V1=FAIL hard rule)
- T_head[D, C-triplet] = 0 (clause 1: quantity mismatch)
- T_head[C-hole, C-triplet] = 0 (clause 1)
- T_head[C-triplet, C-hole] = 0 (clause 1)
- T_head[A, C-triplet] = 0 (clause 1)
- T_head[B, C-triplet] = 0 (clause 1)
