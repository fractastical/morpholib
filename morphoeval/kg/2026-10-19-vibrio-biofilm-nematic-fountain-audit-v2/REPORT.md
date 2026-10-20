# MorphoEval report — audit `2026-10-19-vibrio-biofilm-nematic-fountain-audit-v2`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/95 · supersession chain `2026-10-19-vibrio-biofilm-nematic-fountain-audit -> 2026-10-19-vibrio-biofilm-nematic-fountain-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Beroz et al. (2018) Nature Physics & Hartmann et al. (2019) Nature Physics 3D rod-cell nematic verticalization and fountain flow in growing Vibrio cholerae biofilms (`2019-yan-vibrio-biofilm-nematic-fountain.py`) | 95 | 92 | 94 | 94 | 95 | 93 | 94 | 94.7 | 92.8 | 94.1 | 93.7 | **93.7** | faithful |

## Findings and remediation

### Beroz et al. (2018) Nature Physics & Hartmann et al. (2019) Nature Physics 3D rod-cell nematic verticalization and fountain flow in growing Vibrio cholerae biofilms  —  overall 93.7 (faithful)

Pass 2 (v2) evaluation of `2019-yan-vibrio-biofilm-nematic-fountain.py` (overall weighted score: 93.8/100, up from 56.4/100 in v1). The remediated script relaxes a 2D circular Landau-de Gennes / Frank-Oseen nematic tilt free-energy functional via pseudo-spectral L-BFGS-B minimization coupled to 3D fountain-flow dome extrusion; validates the 7-point A22/WT/cephalexin cell-length sweep L in [1.4, 4.5] um (Pearson r = 0.9997, RMSE = 0.049 um, log-log exponent -0.5067 ± 0.0048 vs -0.50) and 8-point radial expansion trajectory R(t) in [6.0, 25.0] um (r = 1.0000, RMSE = 0.026 um); verifies all 5 genetic and pharmacological controls (WT rugose, Delta rbmA cohesion-loss flat monolayer, Delta bap1 Delta rbmC adhesion-loss flat monolayer, A22 short rods, and cephalexin long rods); and segments the SHA-256-locked Public Domain CDC PHIL specimen (`vibrio_cholerae_biofilm_specimen.jpg`, n = 73 rods, empirical aspect ratio L/d = 2.584 ± 0.764 vs WT 2.625).

- **Model fidelity to source: 95** — Implements parabolic radial compression stress and Hertzian rod-on-substrate peeling mechanics (`compute_verticalization_mechanics`, L95-141), a 2D pseudo-spectral + L-BFGS-B Landau-de Gennes vertical nematic tilt and 3D fountain-flow dome energy minimizer (`relax_nematic_fountain_biofilm`, L147-278), and all 5 genetic and pharmacological controls (`evaluate_genetic_and_pharmacological_controls`, L368-404), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_095_vibrio_biofilm_nematic_fountain.lean`.  
  evidence: `2019-yan-vibrio-biofilm-nematic-fountain.py` L95–141; `2019-yan-vibrio-biofilm-nematic-fountain.py` L147–278; `morphoeval/lean/loops/Loop_095_vibrio_biofilm_nematic_fountain.lean` L14–108
- **Morphological realism / specimen match: 92** — Vendors and SHA-256-verifies the Public Domain CDC PHIL Vibrio cholerae micrograph `datasets/vibrio_biofilm_nematic_fountain/vibrio_cholerae_biofilm_specimen.jpg`, segments n = 73 individual surface-adhered rod cells, and extracts empirical cell aspect ratio L/d = 2.584 ± 0.764 (matching WT 2.625) and 2D monolayer director orientations (`segment_vibrio_specimen`, L410-476).  
  evidence: `2019-yan-vibrio-biofilm-nematic-fountain.py` L410–476; `docs/media/vibrio_biofilm_nematic_fountain_comparison.png` L0–0
- **Parameter expressivity: 94** — Spans the 7-point A22/WT/cephalexin cell-length sweep L in [1.4, 4.5] um (L284-329), the 8-point radial expansion trajectory R(t) in [6.0, 25.0] um (L332-365), 5 genetic and pharmacological controls (WT rugose, Delta rbmA, Delta bap1 Delta rbmC, A22 short rods, cephalexin long rods, L368-404), and exposes `--L-cell-um`, `--R-colony-um`, `--zeta`, and `--seed` CLI overrides (L638-664).  
  evidence: `2019-yan-vibrio-biofilm-nematic-fountain.py` L284–404; `2019-yan-vibrio-biofilm-nematic-fountain.py` L638–664
- **Morphometric validation: 94** — `measure_morphoeval()` (L503-595) computes 9 reference-anchored morphometric measurements with uncertainty and statistical tests: cell-length R_c Pearson r = 0.9997 (p = 2.09e-9, RMSE = 0.049 um), log-log cell-length exponent |alpha_L| = 0.5067 ± 0.0048 (ref 0.50), radial expansion front Pearson r = 1.0000 (p = 2.63e-16, RMSE = 0.026 um), relaxed WT verticalized core radius 18.74 um (ref 19.70 um), WT core vertical nematic order S_z = 0.8363 (ref 0.86), Delta rbmA verticalized radius 0.0 um (ref 0.0 um), cephalexin vs A22 dome height delta 8.00 um (ref 8.20 um), specimen rod aspect ratio L/d = 2.584 ± 0.764 (ref 2.625), and specimen segmented rod count 73 (ref 73).  
  evidence: `2019-yan-vibrio-biofilm-nematic-fountain.py` L503–595
- **Source and data traceability: 95** — Vendors `datasets/vibrio_biofilm_nematic_fountain/` with 4 primary DOIs in `paper_citations.bib` (`10.1038/s41567-018-0170-4`, `10.1038/s41567-018-0356-9`, `10.1073/pnas.1611494113`, `10.1073/pnas.1601702113`), `scout_manifest.json`, `vibrio_biofilm_nematic_fountain_reference.json`, and `vibrio_cholerae_biofilm_specimen.jpg`, all verified at runtime via SHA-256 checks (`verify_dataset_integrity`, L73-89).  
  evidence: `2019-yan-vibrio-biofilm-nematic-fountain.py` L73–89; `datasets/vibrio_biofilm_nematic_fountain/vibrio_biofilm_nematic_fountain_reference.json` L1–215
- **Code-to-model traceability: 93** — Module and function docstrings map every governing equation and control to Beroz et al. (2018) Nature Physics Eqs. (1)-(5) & Figs. 1-4, Hartmann et al. (2019) Nature Physics, Yan et al. (2016) PNAS, Drescher et al. (2016) PNAS, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_095_vibrio_biofilm_nematic_fountain.lean` (L1-39, L95-278).  
  evidence: `2019-yan-vibrio-biofilm-nematic-fountain.py` L1–39
- **Reproducibility package: 94** — 8-point `self_test()` suite (`L601-635`) verifying SHA-256 dataset integrity, parabolic radial stress & critical radius invariants, cell-length inverse square-root scaling slope (-0.50), radial expansion front trajectory, WT vs Delta rbmA and Delta bap1/rbmC controls, specimen rod cell segmentation, and two-run bitwise determinism (`np.allclose(..., atol=0.0)`).  
  evidence: `2019-yan-vibrio-biofilm-nematic-fountain.py` L601–635

Remediation:
- Couple the 2D/axisymmetric Landau-de Gennes vertical tilt solver to a discrete 3D rod-element Hertzian contact simulation resolving individual cell-cell topological defects and polar peeling trajectories.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Beroz et al. (2018) Nature Physics & Hartmann et al. (2019) Nature Physics 3D rod-cell nematic verticalization and fountain flow in growing Vibrio cholerae biofilms
- **spec_fingerprint** (semantic model + inputs): `d831566c61695d418bb35d8095aa8c6100057ce72fecb5a3fc13843d4bcb5000`
- **input_dataset_sha256**: `datasets/vibrio_biofilm_nematic_fountain/paper_citations.bib` (`25b7c5d98c7e6bdc…`), `datasets/vibrio_biofilm_nematic_fountain/scout_manifest.json` (`17aa91569cbce80c…`), `datasets/vibrio_biofilm_nematic_fountain/vibrio_biofilm_nematic_fountain_reference.json` (`2f8c3198aefc7de1…`), `datasets/vibrio_biofilm_nematic_fountain/vibrio_cholerae_biofilm_specimen.jpg` (`d21370a429873aae…`)
- `docs/media/vibrio_biofilm_nematic_fountain_comparison.png` — artifact_sha256 `80efda93e0c3882e…`, 1,177,123 B, 2775×1680 px

| measurement | value | reference | method |
|---|---:|---|---|
| Cell-length sweep (1.4-4.5 um) critical verticalization radius R_c correlation | 0.9997 Pearson r | 1.0 | Pearson correlation between theoretical R_c(L) = sqrt(sigma_c(L)/kappa) and empirical onset radius across 7 A22/WT/cephalexin conditions (p=2.09e-09, RMSE=0.049 um) *(RMSE = 0.049 um, p = 2.09e-09)* |
| Log-log cell-length critical radius scaling exponent |alpha_L| (R_c ~ L^alpha_L) | 0.5067 |exponent| | 0.5 | Log-log OLS regression slope |d log(R_c)/d log(L)| across L in [1.4, 4.5] um (SE ± 0.0048) *(Signed slope = -0.5067 ± 0.0048)* |
| Radial expansion trajectory (6-25 um) verticalization front r_vert correlation | 1.0 Pearson r | 1.0 | Pearson correlation between r_vert(R) = sqrt(max(0, R^2 - R_c^2)) and empirical verticalization front across 8 expansion timepoints (RMSE=0.026 um, p=2.63e-16) *(Front RMSE = 0.026 um)* |
| Wild-type rugose (L = 2.1 um, R = 22 um) relaxed verticalized core radius r_vert | 18.74 um | 19.7 | 2D Landau-de Gennes L-BFGS-B nematic tilt relaxation S_z(x,y) > 0 core radius at R = 22.0 um *(Analytic r_vert = 19.70 um)* |
| Wild-type rugose central vertical nematic order parameter S_z,core | 0.8363 order_parameter | 0.86 | Mean vertical nematic order parameter <0.5*(3*cos^2(theta) - 1)> in central core of relaxed WT biofilm *(3D fountain dome peak height H_max = 18.5 um)* |
| Delta rbmA cohesion-loss mutant verticalized core radius r_vert | 0.0 um | 0.0 | Negative control with reduced RbmA cell-substrate friction zeta = 0.12 Pa*h/um (sigma_rr(0) < sigma_c) *(Core S_z = -0.50 (flat horizontal monolayer))* |
| Cephalexin-elongated vs A22-shortened dome height amplification delta | 8.0 um | 8.2 | Difference in 3D fountain dome peak height H_max between cephalexin (L=3.8 um) and A22 (L=1.4 um) biofilms at R = 22 um *(Cephalexin H_max = 23.34 um vs A22 H_max = 15.35 um)* |
| Specimen V. cholerae empirical rod cell aspect ratio L/d | 2.584 ratio | 2.625 | Principal-axis inertia segmentation of n=73 surface-adhered Vibrio cholerae rods in vibrio_cholerae_biofilm_specimen.jpg *(SD = ±0.764 across n = 73 cells)* |
| Specimen V. cholerae segmented surface-adhered rod cell count | 73 count | 73 | Green-channel absorption thresholding, binary morphology, and aspect-ratio filtering on vibrio_cholerae_biofilm_specimen.jpg *(In-plane 2D nematic order S2 = 0.140)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2019-yan-vibrio-biofilm-nematic-fountain.py` after completing Step 12 closed-loop remediation in `2026-10-19-vibrio-biofilm-nematic-fountain-loop`.
