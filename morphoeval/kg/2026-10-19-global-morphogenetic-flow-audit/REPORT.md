# MorphoEval report — audit `2026-10-19-global-morphogenetic-flow-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Streichan et al. (2018), Mayer et al. (2010) & Lefebvre et al. (2023) Drosophila global morphogenetic flow & active nematic myosin tensor mechanics (`2018-streichan-global-morphogenetic-flow.py`) | 60 | 52 | 56 | 55 | 66 | 60 | 61 | 58.7 | 53.1 | 62.4 | 57.6 | **57.6** | partially faithful |

## Findings and remediation

### Streichan et al. (2018), Mayer et al. (2010) & Lefebvre et al. (2023) Drosophila global morphogenetic flow & active nematic myosin tensor mechanics  —  overall 57.6 (partially faithful)

Pass 1 baseline assessment of `2018-streichan-global-morphogenetic-flow.py` (58.7/100). The initial script verifies the trace-free active nematic stress tensor identity and compares basic wild-type vs isotropic-only flow profiles, but uses an analytic surrogate rather than an explicit 2D spectral/finite-difference compressible Stokes-Brinkman PDE solver `eta*nabla^2 v + eta_b*nabla(nabla.v) - (eta/ell_H^2)*v = -nabla.sigma^a`, lacks all 6 Streichan et al. (2018) wild-type and mutant regimes (`wildtype_gbe_anisotropic_plus_pmg`, `anisotropic_dv_cables_only`, `twist_snail_ventral_furrow_mutant`, `isotropic_myosin_only_no_nematic_q`, `eve_pair_rule_mutant`, and `bcd_nos_tsl_maternal_triple_mutant`), lacks the 2D `(ell_H x eta_b/eta)` parameter morphospace, and does not yet perform automated computer-vision segmentation of embryo contour, pair-rule stripe spacing, and dorsal-posterior invagination geometry on `drosophila_germband_extension_specimen.jpg`.

- **Model fidelity to source: 60** — Implements trace-free active nematic stress tensor construction `Q_xx + Q_yy = 0` (L67-86), but Pass 1 uses analytic velocity profiles rather than solving the 2D compressible Stokes-Brinkman PDE in Fourier/spectral space and integrating Lagrangian germband extension trajectories.  
  evidence: `2018-streichan-global-morphogenetic-flow.py` L67–86; `2018-streichan-global-morphogenetic-flow.py` L89–118
- **Morphological realism / specimen match: 52** — Vendors `datasets/global_morphogenetic_flow/drosophila_germband_extension_specimen.jpg` (CC BY 4.0 whole-embryo Drosophila confocal micrograph from IDR0151), but Pass 1 only displays the raw image without automated CV segmentation of embryo AP/DV aspect ratio, pair-rule stripe periodicity, or posterior midgut invagination fold.  
  evidence: `2018-streichan-global-morphogenetic-flow.py` L49–60
- **Parameter expressivity: 56** — Evaluates 3 regimes in Pass 1, requiring expansion to all 6 Streichan et al. (2018) wild-type and mutant regimes plus a 2D morphospace sweep over hydrodynamic screening length `ell_H = sqrt(eta/xi)` and bulk-to-shear viscosity ratio `eta_b/eta`.  
  evidence: `2018-streichan-global-morphogenetic-flow.py` L89–118
- **Morphometric validation: 55** — Provides 3 baseline measurements in `measure_morphoeval()`, requiring expansion to 7 quantitative morphometric benchmarks spanning 6-regime velocity and flow correlation benchmarks plus confocal specimen morphometrics.  
  evidence: `2018-streichan-global-morphogenetic-flow.py` L121–160
- **Source and data traceability: 66** — Vendors `scout_manifest.json`, `global_morphogenetic_flow_reference.json`, `paper_citations.bib`, and `drosophila_germband_extension_specimen.jpg` with SHA-256 verification linking Streichan et al. (2018), Mayer et al. (2010), Lefebvre et al. (2023), and Brauns et al. (2024).  
  evidence: `2018-streichan-global-morphogenetic-flow.py` L35–60
- **Code-to-model traceability: 60** — Links module docstring to `Loop_048_global_morphogenetic_flow.lean`, requiring complete function-level mappings to Streichan et al. (2018) Figs. 1-5 and Lefebvre et al. (2023) Figs. 1-4.  
  evidence: `2018-streichan-global-morphogenetic-flow.py` L1–20
- **Reproducibility package: 61** — Provides `--self-test` and deterministic execution, requiring expansion to verify exact spectral Stokes-Brinkman Green's function inversion, all 6 Streichan regimes, bitwise determinism, and specimen CV segmentation.  
  evidence: `2018-streichan-global-morphogenetic-flow.py` L196–206

Remediation:
- Implement a full 2D spectral/finite-difference compressible Stokes-Brinkman PDE solver `eta*nabla^2 v + eta_b*nabla(nabla.v) - (eta/ell_H^2)*v = -nabla.sigma^a` on the cartographic Drosophila blastoderm domain with Lagrangian germband extension marker advection.
- Evaluate all 6 Streichan et al. (2018) regimes (`wildtype_gbe_anisotropic_plus_pmg`, `anisotropic_dv_cables_only`, `twist_snail_ventral_furrow_mutant`, `isotropic_myosin_only_no_nematic_q`, `eve_pair_rule_mutant`, `bcd_nos_tsl_maternal_triple_mutant`) and a 15x15 2D `(ell_H x eta_b/eta)` parameter morphospace.
- Add automated CV segmentation of the prolate spheroid embryo contour, 8 pair-rule trunk stripes, DV cable gradient anisotropy, and posterior midgut invagination indentation on `datasets/global_morphogenetic_flow/drosophila_germband_extension_specimen.jpg`.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Streichan et al. (2018), Mayer et al. (2010) & Lefebvre et al. (2023) Drosophila global morphogenetic flow & active nematic myosin tensor mechanics
- **spec_fingerprint** (semantic model + inputs): `c6f91e927b85c9e6c96461d53e4ae9c082d9c0ad9d74ed650b6b2f21fb08b782`
- **input_dataset_sha256**: `datasets/global_morphogenetic_flow/drosophila_germband_extension_specimen.jpg` (`2d7fad8f57edcadf…`), `datasets/global_morphogenetic_flow/global_morphogenetic_flow_reference.json` (`160ead86ff9c447b…`), `datasets/global_morphogenetic_flow/scout_manifest.json` (`990f3ea83019f3d1…`)
- `docs/media/global_morphogenetic_flow_comparison.png` — artifact_sha256 `e49de2ffd5171569…`, 337,692 B, 1820×585 px

| measurement | value | reference | method |
|---|---:|---|---|
| wild-type global morphogenetic flow spatial correlation r | 0.924 correlation_r | 0.924 | Baseline Pass 1 active nematic myosin flow correlation against Streichan et al. (2018) *(WT peak posterior GBE speed = 6.40 um/min)* |
| isotropic-only myosin assumption peak AP velocity | 2.15 um/min | 2.15 | Baseline Pass 1 isotropic myosin test without DV nematic tensor Q_ij *(Without anisotropic DV cables, global GBE flow drops to 2.15 um/min)* |
| bcd nos tsl maternal triple mutant peak AP velocity | 0.0 um/min | 0.0 | Baseline Pass 1 maternal AP-null triple mutant control (Q_ij = 0, m_pmg = 0) *(Complete collapse of posterior-directed germband extension flow)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2018-streichan-global-morphogenetic-flow.py` in `2026-10-19-global-morphogenetic-flow-loop` prior to Pass 2 closed-loop remediation.
