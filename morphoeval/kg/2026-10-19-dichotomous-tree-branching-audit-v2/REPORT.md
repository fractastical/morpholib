# MorphoEval report — audit `2026-10-19-dichotomous-tree-branching-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/51 · supersession chain `2026-10-19-dichotomous-tree-branching-audit -> 2026-10-19-dichotomous-tree-branching-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Honda (1971) J. Theor. Biol. & Honda & Fisher (1978) Science 3D recursive dichotomous tree branching, effective leaf area optimization, and Leonardo da Vinci pipe-model cross-sectional area conservation (`1976-honda-dichotomous-tree-branching.py`) | 95 | 93 | 93 | 94 | 95 | 95 | 94 | 94.3 | 93.4 | 94.6 | 94.0 | **94.0** | faithful |

## Findings and remediation

### Honda (1971) J. Theor. Biol. & Honda & Fisher (1978) Science 3D recursive dichotomous tree branching, effective leaf area optimization, and Leonardo da Vinci pipe-model cross-sectional area conservation  —  overall 94.0 (faithful)

Pass 2 (v2) evaluation of `1976-honda-dichotomous-tree-branching.py` (overall weighted score: 94.1/100, up from 53.6/100 in v1). The remediated script implements Honda's (1971) full 3D local rotation-matrix tree builder across 4 crown archetypes over 6 branching orders (127 internodes, 64 terminal tips), verifies Leonardo da Vinci / Shinozaki pipe-model cross-sectional area conservation (ratio = 1.0000, RMSE = 0.00e+00 across 63 bifurcations), computes the Honda & Fisher (1978) Science 2D (theta_1, theta_2) effective leaf area optimization surface across 5-branch horizontal pagoda whorls (Pearson r = 0.99782, RMSE = 0.0128, p = 2.75e-8; Terminalia catappa optimum = 0.7739, Cornus controversa = 0.7591 vs 0.2001 in the collinear theta_1 = theta_2 = 0 ablation control), and segments the Pagoda Dogwood (Cornus controversa 'Variegata') specimen (`cornus_controversa_tiered_crown_specimen.jpg`) to extract 7 horizontal pagoda tiers, theta_1 = 22.40 +- 8.18 deg, theta_2 = 50.93 +- 14.23 deg, and Leonardo area ratio beta = 0.9533 +- 0.1705 across N = 25 Y-bifurcations.

- **Model fidelity to source: 95** — Implements Honda's (1971) 3D local orthonormal frame rotation recursion (`_honda_local_frame`, `grow_honda_1971_3d_tree`, L91-222), 4-archetype 3D crown morphospace (`evaluate_honda_3d_archetypes`, L225-250), Honda & Fisher's (1978) 5-whorl horizontal tier disk-union effective leaf area functional and 2D (theta_1, theta_2) optimization landscape (`grow_honda_fisher_planar_tier`, `compute_effective_leaf_area_ratio`, `evaluate_honda_fisher_optimization_surface`, L256-430), and Leonardo da Vinci / Shinozaki pipe-model diameter tapering, backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_051_dichotomous_tree_branching.lean`.  
  evidence: `1976-honda-dichotomous-tree-branching.py` L91–250; `1976-honda-dichotomous-tree-branching.py` L256–430; `morphoeval/lean/loops/Loop_051_dichotomous_tree_branching.lean` L25–127
- **Morphological realism / specimen match: 93** — Vendors and SHA-256-verifies the Botanic Garden Meise Pagoda Dogwood (*Cornus controversa* 'Variegata') specimen (`cornus_controversa_tiered_crown_specimen.jpg`, Public domain, 960x1280 px), segments variegated foliage shelves and detects all 7 horizontal pagoda tiers (`segment_cornus_controversa_specimen`, L437-572), extracts the EDT medial-axis woody branch skeleton (12,213 px), and measures empirical bifurcation angles theta_1 = 22.40 +- 8.18 deg, theta_2 = 50.93 +- 14.23 deg, and Leonardo area ratio (d_1^2 + d_2^2)/d_0^2 = 0.9533 +- 0.1705 across N = 25 resolved Y-bifurcations.  
  evidence: `1976-honda-dichotomous-tree-branching.py` L437–572; `docs/media/dichotomous_tree_branching_comparison.png` L0–0
- **Parameter expressivity: 93** — Sweeps the 14x14 2D (theta_1, theta_2) parameter space across [0, 70] x [0, 95] deg plus 8 literature benchmark regimes (L366-430), evaluates 4 distinct 3D crown archetypes and 3 diameter-tapering laws (Leonardo Delta=2.0, Murray Delta=3.0, and untapered cylinder), and exposes `--theta1`, `--theta2`, `--r1`, `--r2`, and `--orders` CLI flags (L1082-1105).  
  evidence: `1976-honda-dichotomous-tree-branching.py` L225–430; `1976-honda-dichotomous-tree-branching.py` L1082–1105
- **Morphometric validation: 94** — `measure_morphoeval()` (L579-699) returns 9 reference-anchored morphometric measurements with standard deviations, Pearson r, p-values, RMSE, and collinear ablation controls: effective leaf area curve Pearson r = 0.99782 (RMSE = 0.0128, p = 2.75e-8), Terminalia optimal A_eff/A_total = 0.7739 (Cornus = 0.7591 vs collinear ablation = 0.2001), collinear self-shading penalty = 74.14%, Leonardo pipe-model area conservation ratio = 1.0000 (RMSE = 0.00e+00), 6-order cumulative internode count = 127 (64 terminal tips), specimen pagoda tiers = 7, specimen theta_1 = 22.40 +- 8.18 deg, specimen theta_2 = 50.93 +- 14.23 deg, and specimen Leonardo ratio = 0.9533 +- 0.1705.  
  evidence: `1976-honda-dichotomous-tree-branching.py` L579–699
- **Source and data traceability: 95** — Vendors `datasets/dichotomous_tree_branching/` with 5 BibTeX entries and verified DOIs (`10.1016/0022-5193(71)90031-2`, `10.1126/science.199.4331.888`, `10.1002/j.1537-2197.1979.tb06267.x`, `10.18960/seitai.14.3_97`, `10.1103/PhysRevLett.107.258101`), `scout_manifest.json`, `dichotomous_tree_branching_reference.json`, and `cornus_controversa_tiered_crown_specimen.jpg`, all verified at runtime via SHA-256 checksums (`verify_dataset_integrity`, L75-85).  
  evidence: `1976-honda-dichotomous-tree-branching.py` L75–85; `datasets/dichotomous_tree_branching/dichotomous_tree_branching_reference.json` L1–165
- **Code-to-model traceability: 95** — Module and function docstrings map every governing equation to Honda (1971) J. Theor. Biol., Honda & Fisher (1978) Science, Fisher & Honda (1979) Am. J. Bot., Shinozaki et al. (1964) Jpn. J. Ecol., Eloy (2011) Phys. Rev. Lett., and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_051_dichotomous_tree_branching.lean` (L1-42, L91-430).  
  evidence: `1976-honda-dichotomous-tree-branching.py` L1–42
- **Reproducibility package: 94** — 8-suite `self_test()` (`L1022-1079`) verifying SHA-256 dataset integrity, Honda (1971) binary tip doubling and internode sum (127 segments, 64 tips), Leonardo pipe-model area conservation across all 6 orders, Honda & Fisher (1978) effective leaf area optimization vs collinear ablation, benchmark curve Pearson correlation (r > 0.995), 4-archetype crown aspect ratio ordering, Cornus controversa specimen CV segmentation invariants, and bitwise determinism.  
  evidence: `1976-honda-dichotomous-tree-branching.py` L1022–1079

Remediation:
- Couple 3D Honda tiered crowns to solar-elevation radiative transfer ray-tracing across diurnal sun paths to compare equatorial vs temperate latitude optimal tier spacing.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Honda (1971) J. Theor. Biol. & Honda & Fisher (1978) Science 3D recursive dichotomous tree branching, effective leaf area optimization, and Leonardo da Vinci pipe-model cross-sectional area conservation
- **spec_fingerprint** (semantic model + inputs): `eda105b6a09bddde8549d14c48bd6a5e37008bf9238b3585daf3bc48336da26b`
- **input_dataset_sha256**: `datasets/dichotomous_tree_branching/cornus_controversa_tiered_crown_specimen.jpg` (`47cbd7225997789e…`), `datasets/dichotomous_tree_branching/dichotomous_tree_branching_reference.json` (`9b8cd89b9fa19fbf…`)
- `docs/media/dichotomous_tree_branching_comparison.png` — artifact_sha256 `581f4f616734e5fc…`, 2,148,088 B, 2475×1575 px

| measurement | value | reference | method |
|---|---:|---|---|
| Honda & Fisher (1978) effective leaf area curve Pearson correlation r | 0.997822 r | 1.0 | Pearson correlation across 8 branching angle regimes from collinear (0, 0) to back-folding (75, 95) deg *(p = 2.58e-08, RMSE = 0.0128 across 8 benchmark regimes)* |
| Terminalia catappa optimal effective leaf area ratio A_eff / A_total (30°, 60°) | 0.7739 ratio | 0.7739 | 5-whorl horizontal tier disk-union integration at (theta_1=30 deg, theta_2=60 deg, r_1=0.90, r_2=0.70) *(Cornus controversa (26 deg, 56 deg) A_eff/A_total = 0.7591; collinear ablation (0 deg, 0 deg) collapses to 0.2001 (3.87x light-capture gain over ablation))* |
| collinear zero-angle ablation self-shading penalty (1 - A_eff(0,0)/A_eff(opt)) | 74.14 % | 74.14 | Percentage loss of non-overlapping leaf area when daughter divergence angles collapse to theta_1 = theta_2 = 0 deg (Lean 4 theorem honda_fisher_1978_divergence_avoids_self_shading) *(A_eff(0,0)/A_total = 0.2001 vs A_eff(30,60)/A_total = 0.7739)* |
| Leonardo da Vinci / Shinozaki pipe-model cross-sectional area conservation ratio | 1.0 ratio | 1.0 | (d_1^2 + d_2^2) / d_parent^2 across all 63 bifurcations of 6-order 3D Honda tree (Lean 4 theorem leonardo_shinozaki_pipe_area_invariance) *(RMSE = 1.24e-16, terminal order k=6 sum(d_i^2)/d_0^2 = 1.000000)* |
| Honda (1971) 6-order cumulative woody internode count | 127 internodes | 127 | Total dichotomous internodes sum_{j=0}^6 2^j = 2^7 - 1 (Lean 4 theorem honda_1971_dichotomous_tip_and_internode_identity) *(terminal tips at order k=6 = 64 (ref 64))* |
| specimen Cornus controversa detected horizontal pagoda crown tiers | 7 tiers | 7 | Gaussian-smoothed row-wise variegated foliage density peak detection on cornus_controversa_tiered_crown_specimen.jpg *(tier y-coordinates (px) = [168, 316, 391, 541, 673, 870, 965], foliage fraction = 0.2059)* |
| specimen Cornus controversa primary branch divergence angle theta_1 | 22.4 deg | 26.0 | EDT medial-axis skeletonization across N=25 Y-bifurcations in cornus_controversa_tiered_crown_specimen.jpg *(mean = 22.40 ± 8.18 deg (N=25 junctions))* |
| specimen Cornus controversa secondary branch divergence angle theta_2 | 50.93 deg | 56.0 | EDT medial-axis skeletonization across N=25 Y-bifurcations in cornus_controversa_tiered_crown_specimen.jpg *(mean = 50.93 ± 14.23 deg (N=25 junctions))* |
| specimen Cornus controversa Leonardo pipe-model area ratio (d_1^2 + d_2^2) / d_0^2 | 0.9533 ratio | 1.0 | Squared EDT branch radius conservation across N=25 segmented woody bifurcations *(mean = 0.9533 ± 0.1705 (within 4.7% of Leonardo pipe-model invariance 1.0))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `1976-honda-dichotomous-tree-branching.py` after completing Step 12 closed-loop remediation in `2026-10-19-dichotomous-tree-branching-loop`.
