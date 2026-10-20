# MorphoEval report — audit `2026-10-19-vibrio-biofilm-nematic-fountain-audit`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/95

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Beroz et al. (2018) Nature Physics & Hartmann et al. (2019) Nature Physics 3D rod-cell nematic verticalization and fountain flow in growing Vibrio cholerae biofilms (`2019-yan-vibrio-biofilm-nematic-fountain.py`) | 58 | 35 | 50 | 52 | 74 | 68 | 64 | 55.3 | 41.4 | 68.4 | 53.7 | **53.7** | partially faithful |

## Findings and remediation

### Beroz et al. (2018) Nature Physics & Hartmann et al. (2019) Nature Physics 3D rod-cell nematic verticalization and fountain flow in growing Vibrio cholerae biofilms  —  overall 53.7 (partially faithful)

Pass 1 (v1) baseline evaluation of `2019-yan-vibrio-biofilm-nematic-fountain.py` (overall weighted score: 56.4/100). The v1 script implements the parabolic radial compressive stress profile sigma_rr(r) = kappa*(R^2 - r^2), the inverse cell-length peeling buckling threshold sigma_c(L) = sigma_c0*(L_0/L), and the verticalization front radius r_vert(R, L) = sqrt(R^2 - R_c(L)^2) across the 7-point A22/cephalexin cell-length sweep, but uses a closed-form radial tanh ansatz instead of relaxing a 2D/axisymmetric Landau-de Gennes nematic director & fountain-flow energy functional, omits the 8-point radial expansion trajectory and Delta rbmA / Delta bap1 Delta rbmC adhesion-loss flat-monolayer controls, and does not segment the vendored Vibrio cholerae micrograph (`vibrio_cholerae_biofilm_specimen.jpg`).

- **Model fidelity to source: 58** — Computes parabolic radial stress coefficient kappa = zeta*g/(4*h), inverse cell-length buckling stress sigma_c(L), critical radius R_c(L), and verticalization front r_vert (L34-53), backed by 3 Lean 4 theorems in `Loop_095_vibrio_biofilm_nematic_fountain.lean`. However, `generate_parametric_verticalization_field` (L56-75) uses a hardcoded radial tanh ansatz rather than relaxing a nematic director tilt & fountain-flow energy functional, and lacks Delta rbmA and Delta bap1 Delta rbmC mutant controls.  
  evidence: `2019-yan-vibrio-biofilm-nematic-fountain.py` L34–53; `2019-yan-vibrio-biofilm-nematic-fountain.py` L56–75
- **Morphological realism / specimen match: 35** — Although `datasets/vibrio_biofilm_nematic_fountain/vibrio_cholerae_biofilm_specimen.jpg` is vendored on disk, the v1 script never loads, segments, or extracts rod cell count, major/minor axes, aspect ratio L/d, or 2D nematic orientation from the biological specimen image.  
  evidence: `2019-yan-vibrio-biofilm-nematic-fountain.py` L78–91; `docs/media/vibrio_biofilm_nematic_fountain_comparison.png` L0–0
- **Parameter expressivity: 50** — Evaluates 7 cell lengths L in [1.4, 4.5] um (L78-91), but omits the 8-point radial expansion trajectory R(t) in [6.0, 25.0] um, omits the 5 genetic and pharmacological controls (WT, Delta rbmA, Delta bap1 Delta rbmC, A22 short rods, cephalexin long rods), and exposes no CLI parameter overrides.  
  evidence: `2019-yan-vibrio-biofilm-nematic-fountain.py` L78–91
- **Morphometric validation: 52** — `measure_morphoeval()` (L146-181) reports only 4 analytic numbers (WT R_c, WT r_vert, A22 R_c, cephalexin R_c) without Pearson r, p-value, RMSE, log-log inverse square-root scaling slope, radial expansion front correlation, Delta rbmA negative-control verticalization delta, or specimen rod aspect ratio L/d.  
  evidence: `2019-yan-vibrio-biofilm-nematic-fountain.py` L146–181
- **Source and data traceability: 74** — Vendors `datasets/vibrio_biofilm_nematic_fountain/` with 4 primary DOIs in `paper_citations.bib` (`10.1038/s41567-018-0170-4`, `10.1038/s41567-018-0356-9`, `10.1073/pnas.1611494113`, `10.1073/pnas.1601702113`), `scout_manifest.json`, and `vibrio_biofilm_nematic_fountain_reference.json`, but lacks runtime SHA-256 verification in `2019-yan-vibrio-biofilm-nematic-fountain.py`.  
  evidence: `datasets/vibrio_biofilm_nematic_fountain/vibrio_biofilm_nematic_fountain_reference.json` L1–215
- **Code-to-model traceability: 68** — Links module docstrings to Beroz et al. (2018) Nature Physics, Hartmann et al. (2019) Nature Physics, and the 3 Lean 4 theorems in `Loop_095_vibrio_biofilm_nematic_fountain.lean` (L1-11), but lacks trace links for variational nematic tilt relaxation, 3D fountain-flow dome extrusion, and Delta rbmA adhesion-loss controls.  
  evidence: `2019-yan-vibrio-biofilm-nematic-fountain.py` L1–11
- **Reproducibility package: 64** — Provides `--self-test` (`L184-189`) checking cell-length sweep monotonicity, but lacks SHA-256 asset verification, two-run bitwise determinism checks on an energy minimizer, and mutant/ablation assertions.  
  evidence: `2019-yan-vibrio-biofilm-nematic-fountain.py` L184–189

Remediation:
- Replace the closed-form radial tanh ansatz with a 2D/axisymmetric Landau-de Gennes nematic tilt L-BFGS-B free-energy minimizer coupled to 3D fountain-flow dome extrusion H(x,y).
- Implement both the 7-point A22/WT/cephalexin cell-length sweep L in [1.4, 4.5] um (verifying inverse square-root critical radius scaling R_c ~ L^-0.5) and the 8-point radial expansion trajectory R(t) in [6.0, 25.0] um (verifying r_vert(t) = sqrt(R(t)^2 - R_c^2)).
- Implement all 5 genetic and pharmacological controls: wild-type rugose, Delta rbmA cell-cell cohesion mutant, Delta bap1 Delta rbmC substrate-adhesion mutant, A22 short rods, and cephalexin long rods.
- Segment the vendored Public Domain Vibrio cholerae micrograph (`vibrio_cholerae_biofilm_specimen.jpg`) to extract individual rod cells, empirical aspect ratio L/d, and 2D nematic orientation order, and enforce runtime SHA-256 verification.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Beroz et al. (2018) Nature Physics & Hartmann et al. (2019) Nature Physics 3D rod-cell nematic verticalization and fountain flow in growing Vibrio cholerae biofilms
- **spec_fingerprint** (semantic model + inputs): `d831566c61695d418bb35d8095aa8c6100057ce72fecb5a3fc13843d4bcb5000`
- **input_dataset_sha256**: `datasets/vibrio_biofilm_nematic_fountain/paper_citations.bib` (`25b7c5d98c7e6bdc…`), `datasets/vibrio_biofilm_nematic_fountain/scout_manifest.json` (`17aa91569cbce80c…`), `datasets/vibrio_biofilm_nematic_fountain/vibrio_biofilm_nematic_fountain_reference.json` (`2f8c3198aefc7de1…`), `datasets/vibrio_biofilm_nematic_fountain/vibrio_cholerae_biofilm_specimen.jpg` (`d21370a429873aae…`)
- `docs/media/vibrio_biofilm_nematic_fountain_comparison.png` — artifact_sha256 `0bb2af64dcd69827…`, 153,325 B, 1680×1330 px

| measurement | value | reference | method |
|---|---:|---|---|
| Wild-type (L = 2.1 um) critical verticalization radius R_c | 9.8 um | 9.8 | Analytic critical verticalization radius R_c = sqrt(sigma_c / kappa) |
| Wild-type (L = 2.1 um, R = 22 um) verticalized core radius r_vert | 19.7 um | 19.7 | Analytic verticalization front radius sqrt(R^2 - R_c^2) at R = 22.0 um |
| A22 short-rod (L = 1.4 um) critical verticalization radius R_c | 12.0 um | 12.1 | Critical colony radius for MreB-inhibited short rods (L = 1.4 um) |
| Cephalexin long-rod (L = 3.8 um) critical verticalization radius R_c | 7.28 um | 7.3 | Critical colony radius for division-inhibited elongated rods (L = 3.8 um) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2019-yan-vibrio-biofilm-nematic-fountain.py` prior to Step 12 closed-loop remediation in `2026-10-19-vibrio-biofilm-nematic-fountain-loop`.
