# MorphoEval report — audit `2026-10-22-zhang-sorting-morphogenesis-audit-v2`

Rubric **0.1.0** · checkpoint `967dc2c3` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Zhang, Goldstein & Levin (2025) Autonomous Cell-View Sorting Morphogenesis, 3-Ablation Platonic Space Audit & Hoel Causal Emergence 2.0 (`sorting-morphogenesis/validate_kymograph_and_paper.py`) | 95 | 92 | 93 | 95 | 96 | 95 | 94 | 94.3 | 93.1 | 94.9 | 94.0 | **94.0** | faithful |

## Findings and remediation

### Zhang, Goldstein & Levin (2025) Autonomous Cell-View Sorting Morphogenesis, 3-Ablation Platonic Space Audit & Hoel Causal Emergence 2.0  —  overall 94.0 (faithful)

Four-language bit-exact replication and formal Lean 4 verification of Zhang, Goldstein & Levin (2025), paired with a 3-ablation mechanistic evaluation of Levin's (2025/2026) Latent Platonic Space claims and Erik Hoel's (2013, 2026) Causal Emergence 1.0 & 2.0 on S_3..S_6.

- **Model fidelity to source: 95** — Audited line-by-line against Zhangtaining/cell_research@1fd2bd59 ([U1]-[U6]), reproducing Bubble local swap pinning, Insertion prefix reset across frozen defects (L68-82), Selection ideal_position pointer reset vs keep modes and .pyc passive-freeze bytecode rule, exact discrepancy-run Delayed Gratification segmentation, and all 3 execution schedulers across Python 3, C++20, and Node.js.  
  evidence: `sorting-morphogenesis/sorting_morphogenesis.py` L65–290; `sorting-morphogenesis/verify_multilang_ports.py` L1–180
- **Morphological realism / specimen match: 92** — Reproduces all space-time kymograph defect-pinning vs. defect-tunneling regimes, Delayed Gratification trajectories, and all 14 published quantitative benchmarks across Figs. 4A-B, 5B-C, 7A-C, 8A-E, and 9-10.  
  evidence: `sorting-morphogenesis/validate_kymograph_and_paper.py` L1–220; `docs/media/zhang_goldstein_levin_kymograph_validation.png` L0–0
- **Parameter expressivity: 93** — Sweeps algotypes (B, I, S), frozen defect types (movable/passive vs immovable/stuck, f=0..3), Ablation 1 (with vs without Insertion L74-76 FREEZE reset), Ablation 2 (I(v) vs E(v) on S_3..S_6), and Ablation 3 (unique keys vs duplicate key multiplicity C = 1..40 and < vs <= tie breaking).  
  evidence: `sorting-morphogenesis/audit_platonic_space_claims.py` L1–240; `sorting-morphogenesis/quantify_emergence_ce2.py` L1–260
- **Morphometric validation: 95** — Validates 14/14 quantitative paper benchmarks plus exact ablation and Causal Emergence 2.0 metrics: Insertion error jump 4.90 -> 21.95 (4.5x), Bubble Lyapunov uphill steps 0/192 (Delta I = -1) vs E(v) 47 uphill ticks, Hoel CE 2.0 Delta CP = +0.1191 on I(v) vs -0.2525 on E(v), and C=2 anti-clustering 0.477 < 0.50 vs C=10 clustering 0.659.  
  evidence: `sorting-morphogenesis/figures/platonic_claims_ablation_metrics.json` L1–38; `sorting-morphogenesis/figures/quantifying_emergence_ce2_metrics.json` L1–310
- **Source and data traceability: 96** — Pinned to Zhang, Goldstein & Levin (2025) DOI 10.1177/10597123241269740, upstream commit 1fd2bd5921c1f6b423a71f691d5189106a8a1020, Levin (2026) DOI 10.3390/philosophies11050161, and Hoel (2013, 2026) DOIs with a 4-tier claim-boundary catalog.  
  evidence: `sorting-morphogenesis/catalog.json` L1–110; `sorting-morphogenesis/PLATONIC_SPACE_CLAIMS_EVALUATION.md` L1–130
- **Code-to-model traceability: 95** — Every algorithmic rule [U1]-[U6] and mathematical claim is linked to Python 3, C++20, Node.js, and 6 machine-checked Lean 4 theorems in morphoeval/lean/loops/Loop_zhang_sorting_morphogenesis.lean (0 sorry).  
  evidence: `morphoeval/lean/loops/Loop_zhang_sorting_morphogenesis.lean` L1–263; `sorting-morphogenesis/cpp/zhang_sorting_morphogenesis.cpp` L1–180
- **Reproducibility package: 94** — Deterministic seed-locked execution across Python 3, C++20, Node.js, and Lean 4 emitting CSV, JSON, multi-panel PNG figures, and animated GIFs.  
  evidence: `docs/media/zhang_levin_platonic_claims_audit.png` L0–0; `docs/media/zhang_hoel_quantifying_emergence_ce2.png` L0–0

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Zhang, Goldstein & Levin (2025) Autonomous Cell-View Sorting Morphogenesis, 3-Ablation Platonic Space Audit & Hoel Causal Emergence 2.0

| measurement | value | reference | method |
|---|---:|---|---|
| Ablation 1 — Insertion Sort Final Error WITH L74-76 FREEZE Prefix Reset (n=60, f=3) | 4.9 monotonicity_error | 4.9 | Mean final adjacent monotonicity error E(v) with upstream InsertionSortCell.py L74-76 `if status == FREEZE: prev = -1` active |
| Ablation 1 — Insertion Sort Final Error WITHOUT L74-76 FREEZE Prefix Reset (n=60, f=3) | 21.95 monotonicity_error | 4.9 | Mean final adjacent monotonicity error E(v) when the 3-line `if status == FREEZE: prev = -1` handler is ablated *(4.5x error jump (4.90 -> 21.95) when explicit FREEZE prefix reset is removed)* |
| Ablation 2 — Bubble Sort Uphill Steps on True Coxeter Inversion Potential I(v) (192 Swaps) | 0 uphill_steps | 0 | Count of swap steps where pairwise inversions I(v) = #{(i<j): v_i > v_j} increases (Delta I > 0) |
| Ablation 2 — Bubble Sort Uphill Ticks on Coarse-Grained Adjacent Error E(v) (192 Swaps) | 47 uphill_steps | 47 | Count of swap steps where adjacent monotonicity error E(v) = #{i: v_i > v_{i+1}} ticks upward by +1 |
| Ablation 2 — Cells Crossing Immovable Frozen Barrier in Bubble & Insertion Sort | 0 cells | 0 | Exact count of cells crossing from left of immovable defect d to right of d under adjacent swaps |
| Hoel Causal Emergence 2.0 (Patterns 2026) Delta CP on Coxeter Inversions I(v) (S_6, N=720) | 0.1191 Delta_CP | 0.1191 | Intensive Causal Primitive gain CP(I) - CP(micro) on exact 720x720 S_6 permutation Markov chain *(Macro determinism on I(v) = 1.0000 (CP = 0.9687 vs micro CP = 0.8496))* |
| Hoel Causal Emergence 2.0 (Patterns 2026) Delta CP on Neighbor Error E(v) (S_6, N=720) | -0.2525 Delta_CP | -0.2525 | Intensive Causal Primitive change CP(E) - CP(micro) when projecting S_6 onto adjacent error E(v) *(Determinism drops from 0.8687 to 0.7209 (negative causal emergence))* |
| Ablation 3 — Unique-Key Chimera (B+S, 0 Ties, Fig. 8A) Peak Algotype Aggregation | 0.624 fraction | 0.64 | Peak same-algotype neighbor fraction A(t) on strictly unique keys 1..80 (zero duplicate keys) |
| Ablation 3 — Duplicate-Key C=2 Copies per Key Anti-Clustering (S B | S B) | 0.4767 fraction | 0.5 | Final same-algotype neighbor fraction A_final at C=2 copies per key (n=80, 66.2% Selection in left slot) |
| Ablation 3 — Duplicate-Key C=10 Copies per Key Homotypic Clique Clustering (SSSSS | BBBBB) | 0.6592 fraction | 0.66 | Final same-algotype neighbor fraction A_final at C=10 copies per key (n=80, 75.0% Selection in left half) |
| Decisive Comparison D vs. D + P_dualist Cell-Swap Trajectory Difference (Py / C++20 / JS / Lean 4) | 0 cell_mismatches | 0 | Bit-exact kymograph cell mismatch count across Python 3, C++20, and Node.js SplitMix64 ports |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.5.4
- scipy: 1.16.3
- matplotlib: 3.10.7

Four-language bit-exact replication (Python 3, C++20, Node.js, Lean 4) of Zhang, Goldstein & Levin (2025), paired with a 3-ablation mechanistic audit of the Latent Platonic Space claims in Levin (2025/2026) and an exact S_3..S_6 Markov-chain Causal Emergence 1.0 & 2.0 (Hoel 2013, 2026) resolution.
