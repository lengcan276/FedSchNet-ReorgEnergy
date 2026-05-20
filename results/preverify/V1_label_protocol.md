# V1 — Label protocol & distribution comparability

## Protocol comparison

| Aspect | Client D (Atahan-Evrenk 2019) | Client C |
|---|---|---|
| Functional | B3LYP | RwB97XD (neutral) / UwB97XD (cation, triplet) |
| Basis | 6-31G* | see Gaussian logs (not in CSV) |
| Geometry | not stated in abstract | adiabatic (4-point Nelsen) |
| Charge state | not stated in abstract (likely neutral->cation, hole) | neutral->cation (hole-lambda); neutral->T1 (triplet-lambda) |
| Conformers | not stated in abstract | one per molecule (ground-state geometry) |
| Library size | 5631 (paper) / 5876 (project) | 53 (hole) / 49 (triplet) |
| Source | Atahan-Evrenk & Atalay, J. Phys. Chem. A 2019, 123, 7855-7863 | data/client_c/reorganization_energy_summary.csv columns |

> **Evidence-source caveat.** Atahan protocol evidence source = abstract/PubMed only; geometry / charge-state / conformer / 4-point details remain UNKNOWN unless the full methods section is accessed. V1 therefore cannot confirm protocol identity from this evidence alone — the achievable verdict is capped at **WARN/INCOMPLETE**, never PASS, until the full paper is read.

## Label distribution stats (eV)

| Stat | D | C-hole | C-triplet |
|---|---:|---:|---:|
| n | 5876 | 53 | 49 |
| mean | 0.2581 | 1.2154 | 2.6382 |
| std | 0.0483 | 0.4961 | 0.7453 |
| min | 0.0765 | 0.4493 | 1.0466 |
| p5 | 0.1621 | 0.6720 | 1.1819 |
| median | 0.2635 | 1.0572 | 2.7470 |
| p95 | 0.3171 | 2.2439 | 3.5591 |
| max | 0.8697 | 3.0041 | 4.2728 |
| skew | 0.8469 | 1.4335 | -0.4377 |

## KS two-sample tests

- D vs C-hole:    KS=0.998, p=1.137e-116
- D vs C-triplet: KS=1.000, p=2.041e-122

## Discussion

- B3LYP (pure hybrid, ~20% HF exchange) systematically underestimates reorganization energies for extended π systems compared with range-separated hybrids such as wB97X-D, which more accurately describe charge localization.
- Different basis-set quality (6-31G* vs the larger basis typical for wB97XD) adds a smaller but non-negligible systematic offset.
- Geometry and conformer treatment for D are not stated in the Atahan-Evrenk 2019 abstract; full-paper inspection is required before claiming protocol identity.
- The dataset size differs (paper: 5631; project: 5876). Confirm provenance of the local CSV — it may be an extended set whose protocol has not been audited.

**Verdict: FAIL**

Reason: D vs C-hole: KS p<1e-3 AND p5/p95 ranges disjoint, AND functionals differ (B3LYP/6-31G* vs wB97XD). Naive supervised transfer would inherit a systematic label bias. Pivot to MAML / label-aware multi-task instead.
