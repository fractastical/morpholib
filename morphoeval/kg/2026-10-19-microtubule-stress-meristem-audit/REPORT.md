# MorphoEval report — audit `2026-10-19-microtubule-stress-meristem-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/36

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bozorg et al. (2014), Sampathkumar et al. (2014), Hamant et al. (2008) & Uyttewaal et al. (2012) shoot apical meristem pressurized shell stress tensor, stress-vs-strain CMT alignment, and katanin organ boundary morphogenesis (`2008-hamant-microtubule-stress-meristem.py`) | 54 | 44 | 52 | 53 | 68 | 58 | 56 | 53.3 | 47.4 | 60.5 | 53.1 | **53.1** | partially faithful |

## Findings and remediation

### Bozorg et al. (2014), Sampathkumar et al. (2014), Hamant et al. (2008) & Uyttewaal et al. (2012) shoot apical meristem pressurized shell stress tensor, stress-vs-strain CMT alignment, and katanin organ boundary morphogenesis  —  overall 53.1 (partially faithful)

Pass 1 (v1) baseline evaluation of `2008-hamant-microtubule-stress-meristem.py` (weighted score: 54.0/100). The Pass 1 baseline script evaluates only algebraic Young-Laplace sphere and cylinder formulas (`compute_baseline_shell_stresses`, L25-40) without a 2D/3D finite-element or vertex shell turgor stress + anisotropic viscoplastic Lockhart growth solver across all 4 canonical SAM zones (Central Dome K > 0, Peripheral Flank, Organ-Meristem Saddle Boundary K < 0, and Cylindrical Stem), without the Bozorg et al. (2014) stress-vs-strain directional cue discrimination experiment, without laser cell-ablation ring reorientation or lateral compression clamp perturbations, without `bot1`/`ktn1` katanin or Oryzalin microtubule depolymerization controls, and without structure-tensor / cell-wall segmentation of the vendored *Arabidopsis thaliana* shoot apical meristem micrograph.

- **Model fidelity to source: 54** — Computes only closed-form algebraic sphere and cylinder Young-Laplace stress ratios (`compute_baseline_shell_stresses`, L25-40), omitting the negative-Gaussian-curvature organ-meristem saddle boundary (K = kappa_1 * kappa_2 < 0), the Bozorg et al. (2014) stress-vs-strain feedback comparison, anisotropic viscoplastic Lockhart cell-wall growth, laser single-cell ablation ring stress reorientation, and katanin (`bot1`/`ktn1`) / Oryzalin feedback gain perturbations.  
  evidence: `2008-hamant-microtubule-stress-meristem.py` L25–40
- **Morphological realism / specimen match: 44** — Vendors `datasets/microtubule_stress_meristem/arabidopsis_meristem_boundary_specimen.jpg` on disk, but Pass 1 does not load, segment, or compute structure-tensor cell-wall nematic orientation and boundary crease curvature from the histological specimen.  
  evidence: `2008-hamant-microtubule-stress-meristem.py` L71–106
- **Parameter expressivity: 52** — Evaluates only a single static parameter point (P = 0.50 MPa, R = 40 um, h = 0.25 um) without sweeping across the 4 SAM curvature zones, 5 experimental perturbation regimes, or exposing CLI overrides.  
  evidence: `2008-hamant-microtubule-stress-meristem.py` L25–40
- **Morphometric validation: 53** — Exposes only 3 unreferenced algebraic numbers in `measure_morphoeval()` (L43-68) without reference values, uncertainty bounds, Pearson correlations, p-values, stress-vs-strain cue discrimination metrics, or empirical specimen measurements.  
  evidence: `2008-hamant-microtubule-stress-meristem.py` L43–68
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `microtubule_stress_meristem_reference.json` in `datasets/microtubule_stress_meristem/`, but the Pass 1 script does not verify SHA-256 dataset checksums or load reference benchmarks at runtime.  
  evidence: `datasets/microtubule_stress_meristem/microtubule_stress_meristem_reference.json` L1–165
- **Code-to-model traceability: 58** — Contains only a brief top-level docstring without mapping functions to Bozorg et al. (2014), Sampathkumar et al. (2014), Hamant et al. (2008), Uyttewaal et al. (2012), or the 3 Lean 4 theorems in `Loop_036_microtubule_stress_meristem.lean`.  
  evidence: `2008-hamant-microtubule-stress-meristem.py` L1–10
- **Reproducibility package: 56** — Supports `--verify-only` with a single assertion on the 2:1 cylinder stress ratio (L109-116), but lacks a comprehensive self-test suite covering SHA-256 integrity, Lean 4 algebraic laws, stress-vs-strain cue discrimination, ablation ring reorientation, and specimen CV segmentation.  
  evidence: `2008-hamant-microtubule-stress-meristem.py` L109–116

Remediation:
- Upgrade `2008-hamant-microtubule-stress-meristem.py` to solve the 2D/3D Finite-Element / Vertex Shell Turgor Stress + Anisotropic Viscoplastic Lockhart Growth equations across all 4 canonical SAM zones (Central Dome K>0, Peripheral Flank, Organ-Meristem Saddle Boundary K<0, and Cylindrical Stem sigma_theta/sigma_z = 2).
- Implement Bozorg et al. (2014) stress-vs-strain cue comparison, laser single-cell ablation ring circumferential reorientation, lateral compression clamp perturbation, and `bot1`/`ktn1` + Oryzalin microtubule depolymerization ablation controls.
- Add structure-tensor & cell-wall segmentation on `datasets/microtubule_stress_meristem/arabidopsis_meristem_boundary_specimen.jpg` and render a 6-panel publication figure.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bozorg et al. (2014), Sampathkumar et al. (2014), Hamant et al. (2008) & Uyttewaal et al. (2012) shoot apical meristem pressurized shell stress tensor, stress-vs-strain CMT alignment, and katanin organ boundary morphogenesis
- **spec_fingerprint** (semantic model + inputs): `58c04b2b1d6826f3891b148876d3e11ab4e8c9ee2c8ea4cca35459f9bbf26dca`
- **input_dataset_sha256**: `datasets/microtubule_stress_meristem/arabidopsis_meristem_boundary_specimen.jpg` (`3a8b54b71f7c780c…`), `datasets/microtubule_stress_meristem/microtubule_stress_meristem_reference.json` (`3e04a6dded528819…`), `datasets/microtubule_stress_meristem/paper_citations.bib` (`9355fa7c7b784918…`), `datasets/microtubule_stress_meristem/scout_manifest.json` (`e3cd62cc5263f317…`)
- `docs/media/microtubule_stress_meristem_comparison.png` — artifact_sha256 `4afaa1f945e5e725…`, 47,656 B, 1080×480 px

| measurement | value | reference | method |
|---|---:|---|---|
| spherical dome isotropic stress sigma_0 | 40.0 MPa | — | Young-Laplace sphere P*R/(2*h) |
| cylindrical stem hoop stress sigma_theta | 80.0 MPa | — | Young-Laplace cylinder P*R/h |
| cylindrical stem hoop-to-axial stress ratio | 2.0 ratio | — | sigma_theta / sigma_z on cylinder |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2008-hamant-microtubule-stress-meristem.py` prior to Step 12 closed-loop remediation.
