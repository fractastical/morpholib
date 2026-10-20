# MorphoEval report — audit `2026-10-19-pavement-cell-lobing-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/39 · supersession chain `2026-10-19-pavement-cell-lobing-audit -> 2026-10-19-pavement-cell-lobing-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sapala et al. (2018) & Fu et al. (2005) ROP2-ROP6 interdigitated puzzle-shaped pavement cell lobing and Largest Empty Circle (LEC) turgor stress scaling (`2017-runions-pavement-cell-lobing.py`) | 94 | 91 | 92 | 94 | 95 | 93 | 93 | 93.3 | 92.1 | 93.7 | 93.0 | **93.0** | faithful |

## Findings and remediation

### Sapala et al. (2018) & Fu et al. (2005) ROP2-ROP6 interdigitated puzzle-shaped pavement cell lobing and Largest Empty Circle (LEC) turgor stress scaling  —  overall 93.0 (faithful)

Pass 2 (v2) evaluation of `2017-runions-pavement-cell-lobing.py` (overall weighted score: 93.1/100, up from 53.6/100 in v1). The remediated script computes exact Euclidean Distance Transform Largest Empty Circle radii R_LEC(A) and outer periclinal wall stresses across cell area A in [250, 4500] um^2 (WT R_LEC plateau 11.576 um vs ref 11.4 um; scaling r = 0.9859, p = 2.97e-4, RMSE = 0.314 um; 3.27x stress reduction vs unlobed isotropic cells); integrates a 196-node coupled ROP2-RIC4 / ROP6-RIC1-KTN1 cortical reaction-diffusion PDE system across shared anticlinal walls (intracortical r = -0.9899, trans-wall complementarity r = 0.9998, wavelength 24.50 um vs ref 24.5 um); validates ktn1/ric1 and rop2 rop4 mutant lobeyness collapse (ktn1 L = 1.1107 vs ref 1.11, WT S = 0.6735 vs ref 0.68); and segments 10 interlocking puzzle cells on the SHA-256-locked Arabidopsis leaf epidermis SEM micrograph (`arabidopsis_pavement_cell_specimen.jpg`, mean S = 0.720).

- **Model fidelity to source: 94** — Implements exact 2D Euclidean Distance Transform Largest Empty Circle (`compute_exact_lec_from_polygon`, L137-186), ConvexHull solidity & lobeyness (`compute_lobeyness_and_solidity`, L114-134), Sapala et al. (2018) spring-remodeled puzzle cell area scaling across WT, isotropic, ktn1/ric1, rop2 rop4 RNAi, and monocot file regimes (`simulate_puzzle_cell_area_scaling`, L257-357), and a 196-node coupled ROP2–ROP6 anticlinal boundary reaction-diffusion PDE solver (`simulate_rop2_rop6_cortical_rd`, L363-478), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_039_pavement_cell_lobing.lean`.  
  evidence: `2017-runions-pavement-cell-lobing.py` L137–357; `2017-runions-pavement-cell-lobing.py` L363–478; `morphoeval/lean/loops/Loop_039_pavement_cell_lobing.lean` L15–117
- **Morphological realism / specimen match: 91** — Vendors and SHA-256-verifies the CC BY-SA 3.0 Arabidopsis thaliana leaf epidermis SEM micrograph (`arabidopsis_pavement_cell_specimen.jpg`, 1280x960), applies DoG anticlinal wall groove enhancement and zero-compactness watershed segmentation (`segment_arabidopsis_pavement_specimen`, L484-575) to extract 10 interlocking puzzle-shaped pavement cells, and measures their empirical solidity (mean S = 0.720 ± 0.041, min S = 0.626), lobeyness (mean L = 1.506), and inscribed R_LEC circles.  
  evidence: `2017-runions-pavement-cell-lobing.py` L484–575; `docs/media/pavement_cell_lobing_comparison.png` L0–0
- **Parameter expressivity: 92** — Sweeps 6 developmental cell area stages A in [250, 4500] um^2 across 5 morphological / genetic regimes (WT Col-0, unlobed isotropic, ktn1/ric1 mutant, rop2 rop4 RNAi, and monocot parallel-file epidermis, L257-357), supports ROP2-ROP6 cross-inhibition ablation in the PDE solver (L404-410), and exposes `--turgor-P`, `--wall-h`, `--neck-R0`, `--gamma-cross`, `--alpha-wall`, and `--seed` CLI flags (L846-885).  
  evidence: `2017-runions-pavement-cell-lobing.py` L257–357; `2017-runions-pavement-cell-lobing.py` L846–885
- **Morphometric validation: 94** — `measure_morphoeval()` (L724-802) reports 9 reference-anchored morphometric measurements with uncertainty, p-values, and negative controls: mature WT R_LEC = 11.576 um (ref 11.4 um), R_LEC(A) trajectory Pearson r = 0.9859 (p = 2.97e-4, RMSE = 0.314 um), isotropic-to-puzzle stress ratio 3.270 (ref 3.32), ROP2-ROP6 intracortical anticorrelation r = -0.9899 (ref -0.92, p = 1.64e-166), trans-wall complementarity r = 0.9998 (ref 0.94), lobe wavelength 24.50 um (ref 24.5 um), mature WT solidity S = 0.6735 (ref 0.68), ktn1 mutant lobeyness collapse L = 1.1107 (ref 1.11), and segmented SEM specimen solidity S = 0.7200 (ref 0.68).  
  evidence: `2017-runions-pavement-cell-lobing.py` L724–802
- **Source and data traceability: 95** — Vendors `datasets/pavement_cell_lobing/` with 5 primary BibTeX entries and verified DOIs (`10.7554/eLife.32794`, `10.1016/j.cell.2004.12.026`, `10.1038/s41477-019-0500-2`, `10.1016/j.cub.2013.01.022`, `10.7554/eLife.01967`), `scout_manifest.json`, `pavement_cell_lobing_reference.json`, and `arabidopsis_pavement_cell_specimen.jpg`, all verified at runtime via SHA-256 checksums (`verify_dataset_integrity`, L79-93).  
  evidence: `2017-runions-pavement-cell-lobing.py` L79–93; `datasets/pavement_cell_lobing/pavement_cell_lobing_reference.json` L1–165
- **Code-to-model traceability: 93** — Module and function docstrings map every governing equation to Sapala et al. (2018) eLife Figs. 1-5 & Box 1, Fu et al. (2005) Cell Figs. 1-7, Lin et al. (2013) Curr. Biol., Bidhendi et al. (2019) Nat. Plants, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_039_pavement_cell_lobing.lean` (L1-51).  
  evidence: `2017-runions-pavement-cell-lobing.py` L1–51
- **Reproducibility package: 93** — 8-point `run_self_test()` suite (`L805-843`) verifying SHA-256 dataset integrity, Sapala et al. (2018) LEC plateau & 3.27x stress suppression, ROP2-ROP6 cortical PDE anticorrelation & exact intercellular area conservation, ktn1 & rop2 rop4 mutant lobeyness collapse, monocot narrow file R_LEC bound, SEM specimen segmentation, 9-metric `measure_morphoeval()` contract, and bitwise determinism.  
  evidence: `2017-runions-pavement-cell-lobing.py` L805–843

Remediation:
- Couple the 2D spring-remodeling pavement cell polygon network to a 3D pressurized periclinal shell finite-element solver to resolve out-of-plane anticlinal wall buckling variations with depth.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Sapala et al. (2018) & Fu et al. (2005) ROP2-ROP6 interdigitated puzzle-shaped pavement cell lobing and Largest Empty Circle (LEC) turgor stress scaling
- **spec_fingerprint** (semantic model + inputs): `972460a5749d7d30868f5de9c61bcfd278fa143f89d629090a75b7aa4ebdc287`
- **input_dataset_sha256**: `datasets/pavement_cell_lobing/arabidopsis_pavement_cell_specimen.jpg` (`4b7c1062f6ebfd6e…`), `datasets/pavement_cell_lobing/paper_citations.bib` (`e5d0ea8ec26365c9…`), `datasets/pavement_cell_lobing/pavement_cell_lobing_reference.json` (`de4e6daab5eae4a0…`), `datasets/pavement_cell_lobing/scout_manifest.json` (`72c917ec10ac1a55…`)
- `docs/media/pavement_cell_lobing_comparison.png` — artifact_sha256 `8df82fae6327068a…`, 961,496 B, 2548×1568 px

| measurement | value | reference | method |
|---|---:|---|---|
| wt_mature_lec_radius_at_4500um2 | 11.576 um | 11.4 | Exact 2D Euclidean Distance Transform inscribed circle on spring-remodeled WT puzzle polygon (RMSE across 6 area stages = 0.314 um) |
| wt_lec_area_scaling_pearson_r | 0.9859 r | 1.0 | Pearson correlation across 6 developmental cell area stages (r = 0.9859, p = 2.97e-04, RMSE = 0.314 um) |
| isotropic_vs_puzzle_stress_reduction_ratio | 3.27 ratio | 3.32 | Ratio sigma_iso(4500) / sigma_WT(4500) = 37.85 MPa / 11.58 MPa at P = 0.60 MPa, h = 0.30 um (Lean 4 Law 1 verified) |
| rop2_rop6_intracortical_anticorrelation_r | -0.9899 r | -0.92 | 196-node periodic anticlinal boundary reaction-diffusion PDE integration (r = -0.9899, p = 1.64e-166) |
| rop2_rop6_trans_wall_complementarity_r | 0.9998 r | 0.94 | Cross-wall Pearson correlation between U_A(s) and V_B(s) (r = 0.9998, p = 0.00e+00, area conservation error = 0.00e+00 um^2) |
| anticlinal_lobe_wavelength_um | 24.5 um | 24.5 | Dominant Fourier mode k = 8 along L = 196.0 um shared anticlinal wall |
| wt_puzzle_cell_solidity_S | 0.6735 dimensionless | 0.68 | ConvexHull area ratio on mature WT puzzle polygons (lobeyness L = 1.432 vs ref 1.62) |
| ktn1_mutant_lobeyness_collapse_L | 1.1107 dimensionless | 1.11 | Negative perturbation control k_rop -> 0.11 (mutant solidity S = 0.886 vs ref 0.90; rop2 rop4 RNAi L = 1.060 vs ref 1.07) |
| specimen_segmented_pavement_solidity_S | 0.72 dimensionless | 0.68 | DoG + zero-compactness watershed segmentation of N = 10 puzzle cells on 1280x960 SEM micrograph (mean S = 0.720 ± 0.041, min S = 0.626, calibrated R_LEC = 11.40 um) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2017-runions-pavement-cell-lobing.py` after completing Step 12 closed-loop remediation in `2026-10-19-pavement-cell-lobing-loop`.
