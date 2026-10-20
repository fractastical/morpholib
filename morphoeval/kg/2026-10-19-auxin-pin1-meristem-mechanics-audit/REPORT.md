# MorphoEval report — audit `2026-10-19-auxin-pin1-meristem-mechanics-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/33

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Hamant et al. (2008) & Heisler et al. (2010) pressurized SAM shell stress tensor, cortical microtubule alignment, and PIN1-auxin phyllotactic mechanics (`2008-hamant-meristem-microtubule-stress.py`) | 56 | 46 | 52 | 52 | 68 | 58 | 54 | 54.7 | 48.2 | 59.7 | 53.6 | **53.6** | partially faithful |

## Findings and remediation

### Hamant et al. (2008) & Heisler et al. (2010) pressurized SAM shell stress tensor, cortical microtubule alignment, and PIN1-auxin phyllotactic mechanics  —  overall 53.6 (partially faithful)

Pass 1 (v1) baseline evaluation of `2008-hamant-meristem-microtubule-stress.py` (weighted score: 54.1/100). The Pass 1 script evaluates only analytical sphere vs cylinder Laplace-Young ratios without cellular L1 meristem topology, saddle-crease primordium boundary stress tensors, dynamical CMT + PIN1 co-alignment ODEs, Kirsch laser ablation or lateral compression perturbations, or specimen image segmentation.

- **Model fidelity to source: 56** — Computes basic analytical spherical dome (PR/2t) and cylindrical flank (PR/t) Laplace-Young stresses (L23-41), but omits negative-Gaussian-curvature saddle crease stress at primordium boundaries, dynamical cortical microtubule (CMT) and PIN1 polarity co-alignment ODEs on a cellular L1 dome, anisotropic Lockhart cell-wall yielding, and Kirsch laser-ablation stress redistribution.  
  evidence: `2008-hamant-meristem-microtubule-stress.py` L23–41
- **Morphological realism / specimen match: 46** — Vendors `datasets/auxin_pin1_meristem_mechanics/arabidopsis_sam_meristem_specimen.jpg` on disk, but Pass 1 does not load, segment, or compare the histological SAM dome and saddle boundary crease against the simulated shell geometry.  
  evidence: `2008-hamant-meristem-microtubule-stress.py` L63–96
- **Parameter expressivity: 52** — Evaluates a single static (P, R, t) parameter set without sweeping the 4 SAM curvature zones, laser ablation, lateral compression, oryzalin microtubule depolymerization, or CLI parameter overrides.  
  evidence: `2008-hamant-meristem-microtubule-stress.py` L23–41
- **Morphometric validation: 52** — Exposes only 2 trivial analytical ratios in `measure_morphoeval()` without ODE simulation error metrics, statistical p-values, ablation reorientation fractions, phyllotactic divergence angles, or specimen segmentation morphometrics.  
  evidence: `2008-hamant-meristem-microtubule-stress.py` L44–60
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `auxin_pin1_meristem_mechanics_reference.json` with verified DOIs, but Pass 1 does not load or SHA-256-verify the dataset at runtime.  
  evidence: `datasets/auxin_pin1_meristem_mechanics/auxin_pin1_meristem_mechanics_reference.json` L1–130
- **Code-to-model traceability: 58** — Header cites Hamant et al. (2008) and Heisler et al. (2010), but lacks explicit equation-to-code-line and Lean 4 theorem mappings.  
  evidence: `2008-hamant-meristem-microtubule-stress.py` L1–20
- **Reproducibility package: 54** — Runs deterministically to generate PNG and GIF outputs, but lacks a `--self-test` suite, SHA-256 integrity checks, and negative perturbation controls.  
  evidence: `2008-hamant-meristem-microtubule-stress.py` L63–100

Remediation:
- Implement full doubly-curved pressurized shell principal stress solver across all 4 SAM zones (spherical dome, peripheral flank, cylindrical stem, and negative-Gaussian-curvature primordium saddle crease) plus anisotropic Lockhart cell-wall yielding.
- Implement coupled cellular L1 dome ODE solver for cortical microtubule (CMT) director alignment along maximal principal tensile stress e_1 and up-the-gradient + stress-aligned PIN1 auxin transport generating golden-angle (137.5 deg) phyllotactic primordia.
- Implement Kirsch circular laser-ablation wound stress redistribution, lateral micro-vise compression, and oryzalin CMT-depolymerization negative control.
- Segment the vendored Arabidopsis SAM histological specimen (`arabidopsis_sam_meristem_specimen.jpg`) to extract the apical dome curvature profile, saddle boundary crease, and cellular centroids, with full SHA-256 verification and `--self-test` suite.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Hamant et al. (2008) & Heisler et al. (2010) pressurized SAM shell stress tensor, cortical microtubule alignment, and PIN1-auxin phyllotactic mechanics
- **spec_fingerprint** (semantic model + inputs): `e6283030cad9fe71483e6b95b0b71d85936ee41d7e76429d3b5bb4c3138dcc69`
- **input_dataset_sha256**: `datasets/auxin_pin1_meristem_mechanics/arabidopsis_sam_meristem_specimen.jpg` (`3a8b54b71f7c780c…`), `datasets/auxin_pin1_meristem_mechanics/auxin_pin1_meristem_mechanics_reference.json` (`db56b93840c38568…`), `datasets/auxin_pin1_meristem_mechanics/paper_citations.bib` (`f9f54880f2e74e09…`), `datasets/auxin_pin1_meristem_mechanics/scout_manifest.json` (`1c17f6d448f3c423…`)
- `docs/media/auxin_pin1_meristem_mechanics_comparison.png` — artifact_sha256 `8ca9f0c12dd0e2a7…`, 61,118 B, 1200×540 px

| measurement | value | reference | method |
|---|---:|---|---|
| cylindrical_flank_hoop_to_axial_stress_ratio | 2.0 ratio | 2.0 | Analytical thin-shell Laplace-Young ratio sigma_theta / sigma_z on cylindrical stem flank |
| spherical_dome_stress_anisotropy | 0.0 dimensionless | 0.0 | Analytical thin-shell stress anisotropy (sigma_1 - sigma_2)/(sigma_1 + sigma_2) on spherical dome |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2008-hamant-meristem-microtubule-stress.py` prior to Step 12 closed-loop remediation.
