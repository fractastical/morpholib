# MorphoEval report — audit `2026-10-19-venus-flytrap-snap-buckling-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/66

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Forterre et al. (2005) Nature Doubly-Curved Shallow Shell Elastic Energy & Saddle-Node Cusp Catastrophe in Dionaea muscipula (`2005-forterre-venus-flytrap-snap-buckling.py`) | 58 | 46 | 52 | 55 | 68 | 60 | 62 | 56.0 | 49.4 | 63.5 | 55.6 | **55.6** | partially faithful |

## Findings and remediation

### Forterre et al. (2005) Nature Doubly-Curved Shallow Shell Elastic Energy & Saddle-Node Cusp Catastrophe in Dionaea muscipula  —  overall 55.6 (partially faithful)

Pass 1 (v1) baseline evaluation of `2005-forterre-venus-flytrap-snap-buckling.py` (overall weighted score: 56.4/100). The baseline evaluates the closed-form zero-bias pitchfork equilibrium and 1D symmetric energy curve, but lacks full 2D coupled (kappa_x(t), kappa_y(t)) overdamped poroelastic + inertial snap dynamics across alpha in [0.2, 5.0], omits the three-stage closure kinematics (slow initial phase -> rapid 100 ms snap-buckling -> slow poroelastic damping), omits the cusp catastrophe phase diagram in (alpha, kappa_x0) and surgical curvature-relief strip cutting ablation, and does not segment the vendored Dionaea muscipula specimen image.

- **Model fidelity to source: 58** — Evaluates critical coupling alpha_c and zero-bias analytical equilibrium (`critical_coupling_alpha`, `analytical_equilibrium_curvature`, L29-38), but does not integrate the coupled 2D (kappa_x(t), kappa_y(t)) poroelastic + inertial equations of motion or minimize the 2D shallow-shell energy U(kappa_x, kappa_y).  
  evidence: `2005-forterre-venus-flytrap-snap-buckling.py` L29–49
- **Morphological realism / specimen match: 46** — Does not load or segment the vendored Dionaea muscipula trap lobe photograph `datasets/venus_flytrap_snap_buckling/dionaea_muscipula_trap_lobe_specimen.jpg` and renders only a 2-panel baseline plot.  
  evidence: `2005-forterre-venus-flytrap-snap-buckling.py` L52–85
- **Parameter expressivity: 52** — Evaluates static analytical curves without sweeping coupled 2D snap dynamics across alpha in [0.2, 5.0], surgical curvature-relief strip ablation, or CLI parameter overrides.  
  evidence: `2005-forterre-venus-flytrap-snap-buckling.py` L52–85
- **Morphometric validation: 55** — `measure_morphoeval()` (L109-133) returns only 2 static baseline checks without saddle-node fold thresholds, three-stage closure trajectory correlations, surgical ablation checks, or specimen segmentation metrics.  
  evidence: `2005-forterre-venus-flytrap-snap-buckling.py` L109–133
- **Source and data traceability: 68** — Vendors `datasets/venus_flytrap_snap_buckling/` with 4 DOIs in `paper_citations.bib`, `scout_manifest.json`, reference JSON, and Dionaea muscipula specimen JPEG, but the baseline script does not enforce runtime SHA-256 verification.  
  evidence: `datasets/venus_flytrap_snap_buckling/paper_citations.bib` L1–43
- **Code-to-model traceability: 60** — Cites Forterre et al. (2005) Nature in module docstring, but lacks function-level equation mappings to the 3 Lean 4 theorems in `Loop_066_venus_flytrap_snap_buckling.lean`.  
  evidence: `2005-forterre-venus-flytrap-snap-buckling.py` L1–10
- **Reproducibility package: 62** — Provides `--self-test` (`L136-141`), but checks only two static values without SHA-256 dataset verification, surgical ablation negative controls, or bitwise determinism checks.  
  evidence: `2005-forterre-venus-flytrap-snap-buckling.py` L136–141

Remediation:
- Implement full coupled 2D (kappa_x(t), kappa_y(t)) overdamped poroelastic + inertial snap dynamics and L-BFGS-B energy minimization of U(kappa_x, kappa_y) across alpha in [0.2, 5.0], reproducing the three-stage closure kinematics (slow initial convex phase -> rapid 100 ms elastic snap-buckling -> slow poroelastic damping).
- Implement the cusp catastrophe phase diagram in (alpha, kappa_x0) with analytical and numerical saddle-node fold boundaries kappa_x0_fold(alpha), developmental leaf-size scaling alpha = L^4*kappa_init^2/h^2 across juvenile/subadult/mature traps, and surgical midrib/curvature-relief strip cutting ablation (L_eff = L/3 -> alpha < alpha_c) eliminating snap.
- Implement computer-vision segmentation of `datasets/venus_flytrap_snap_buckling/dionaea_muscipula_trap_lobe_specimen.jpg`, isolating the bilobed anthocyanin-red inner trap contour, central midrib hinge, bilateral lobe symmetry, and marginal cilia teeth.
- Enforce runtime SHA-256 verification of all 4 dataset files, expand `measure_morphoeval()` to 9 quantitative reference-anchored metrics, and upgrade the figure to a 6-panel synthesis layout.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Forterre et al. (2005) Nature Doubly-Curved Shallow Shell Elastic Energy & Saddle-Node Cusp Catastrophe in Dionaea muscipula
- **spec_fingerprint** (semantic model + inputs): `09b1066fde1900adefad4e5592cdbc34803c6f1439dd9667ec1014d5137d4133`
- **input_dataset_sha256**: `datasets/venus_flytrap_snap_buckling/dionaea_muscipula_trap_lobe_specimen.jpg` (`f2738290d66e4418…`), `datasets/venus_flytrap_snap_buckling/paper_citations.bib` (`f6ac2d52597b043c…`), `datasets/venus_flytrap_snap_buckling/scout_manifest.json` (`572531a0986c798e…`), `datasets/venus_flytrap_snap_buckling/venus_flytrap_snap_buckling_reference.json` (`593d9449e870fcf2…`)
- `docs/media/venus_flytrap_snap_buckling_comparison.png` — artifact_sha256 `044fe607a4b7f632…`, 116,318 B, 1610×672 px

| measurement | value | reference | method |
|---|---:|---|---|
| Forterre et al. (2005) critical cusp coupling threshold alpha_c (nu=0.28) | 0.8 dimensionless | 0.8 | Analytical cusp catastrophe threshold alpha_c = (1 + nu) / (c_stretch * k_init^2) *(Baseline Pass 1 static threshold check)* |
| Zero-bias bistable equilibrium curvature kappa*(alpha=3.0) / kappa_init | 0.8563 kappa_over_kinit | 0.8563 | Closed-form pitchfork equilibrium sqrt(1 - alpha_c / alpha) *(Baseline Pass 1 static curvature check)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2005-forterre-venus-flytrap-snap-buckling.py` prior to Step 12 remediation.
