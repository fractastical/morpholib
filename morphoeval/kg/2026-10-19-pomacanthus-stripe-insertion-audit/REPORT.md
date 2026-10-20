# MorphoEval report — audit `2026-10-19-pomacanthus-stripe-insertion-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Kondo & Asai (1995) Pomacanthus reaction-diffusion stripe insertion & Y-branch unzipping (`1995-kondo-pomacanthus-stripe-insertion.py`) | 56 | 49 | 52 | 53 | 64 | 58 | 56 | 54.7 | 50.5 | 59.2 | 54.4 | **54.4** | partially faithful |

## Findings and remediation

### Kondo & Asai (1995) Pomacanthus reaction-diffusion stripe insertion & Y-branch unzipping  —  overall 54.4 (partially faithful)

Pass 1 baseline assessment of `1995-kondo-pomacanthus-stripe-insertion.py` (54.7/100). The initial script derives the analytical equilibrium (A*, I*) = (3.0, 2.5), critical diffusivity ratio d_c = 10.2596, and intrinsic wavelength lambda_c = 3.4739 for the Kondo & Asai (1995) Fig. 1 piecewise-linear reaction-diffusion model, but approximates the growing-domain kymograph kinematically rather than integrating the full nonlinear piecewise-linear saturating Gierer-Meinhardt PDE dA/dt = F(A,I) - g_A A + D_A Lap(A), dI/dt = G(A,I) - g_I I + D_I Lap(I) across the 3 -> 6 -> 12 -> 24 stripe insertion cascade in P. semicirculatus, lacks 2D Y-junction branch unzipping simulation in P. imperator, and lacks automated computer-vision segmentation of the vendored Pomacanthus semicirculatus juvenile specimen (`pomacanthus_semicirculatus_specimen.jpg`).

- **Model fidelity to source: 56** — Computes the exact interior steady state (A*, I*) = (3.0, 2.5), critical diffusivity ratio d_c = 10.2596, and intrinsic wavelength lambda_c = 3.4739 (L32-69), but Pass 1 uses a kinematic cosine step approximation (`kinematic_stripe_kymograph`, L72-80) rather than solving the nonlinear piecewise-linear saturating Gierer-Meinhardt PDE on an expanding 1D domain and simulating 2D Y-branch horizontal unzipping.  
  evidence: `1995-kondo-pomacanthus-stripe-insertion.py` L32–69; `1995-kondo-pomacanthus-stripe-insertion.py` L72–80
- **Morphological realism / specimen match: 49** — Displays `datasets/pomacanthus_stripe_insertion/pomacanthus_semicirculatus_specimen.jpg` (CC BY-SA 4.0 juvenile Semicircle angelfish), but Pass 1 lacks automated computer-vision segmentation of primary white vs newly inserted secondary mid-trough stripes and inter-stripe spacing regularity on the flank ROI.  
  evidence: `1995-kondo-pomacanthus-stripe-insertion.py` L108–118
- **Parameter expressivity: 52** — Evaluates the canonical Fig. 1 parameter set (D_A = 0.007, D_I = 0.10), but lacks a 5-regime ontogenetic body-length doubling & subcritical-diffusion control sweep (s = 1, 2, 4, 8 -> N = 3, 6, 12, 24 stripes plus D_I = 0.050 -> N = 0 stripes) and 2D P. imperator Y-branch unzipping comparison.  
  evidence: `1995-kondo-pomacanthus-stripe-insertion.py` L72–80
- **Morphometric validation: 53** — Reports 2 analytical quantities in `measure_morphoeval()` (`d_c` and `lambda_c`), but lacks simulated PDE stripe wavelength accuracy, 5-regime stripe-insertion cascade RMSE, 2D Y-branch unzipping velocity/displacement metrics, and empirical specimen inter-stripe wavelength and alternating amplitude ratio.  
  evidence: `1995-kondo-pomacanthus-stripe-insertion.py` L83–105
- **Source and data traceability: 64** — Vendors `scout_manifest.json`, `pomacanthus_stripe_insertion_reference.json`, `paper_citations.bib`, and `pomacanthus_semicirculatus_specimen.jpg`, but runtime SHA-256 dataset verification (`verify_dataset_integrity`) is not yet enforced in the script.  
  evidence: `datasets/pomacanthus_stripe_insertion/pomacanthus_stripe_insertion_reference.json` L1–150
- **Code-to-model traceability: 58** — Maps linear stability parameters to Kondo & Asai (1995) Fig. 1, but needs explicit section/equation docstrings linking every solver routine to the 3 Lean 4 theorems in `Loop_004_pomacanthus_stripe_insertion.lean`.  
  evidence: `1995-kondo-pomacanthus-stripe-insertion.py` L1–12
- **Reproducibility package: 56** — Provides `--self-test`, but only checks `(A*, I*) = (3.0, 2.5)` and lacks SHA-256 dataset verification, bitwise PDE solver determinism, subcritical negative control assertion, 2D Y-branch unzipping verification, and specimen CV segmentation checks.  
  evidence: `1995-kondo-pomacanthus-stripe-insertion.py` L128–133

Remediation:
- Replace the kinematic kymograph with a full IMEX / finite-difference solver for the Kondo & Asai (1995) piecewise-linear saturating Gierer-Meinhardt reaction-diffusion PDE dA/dt = min(F_max, max(0, c_1 A + c_2 I + c_3)) - g_A A + D_A Lap(A), dI/dt = min(G_max, max(0, c_4 A + c_5)) - g_I I + D_I Lap(I) across the 5-regime P. semicirculatus doubling cascade (s = 1, 2, 4, 8 -> 3, 6, 12, 24 stripes plus subcritical control D_I = 0.050 -> 0 stripes) and continuous growing-domain kymograph.
- Implement a 2D Kondo-Asai reaction-diffusion solver modeling Pomacanthus imperator Y-junction branch point horizontal unzipping when dorso-ventral flank spacing widens relative to boundary stripe spacing.
- Add computer-vision segmentation of `datasets/pomacanthus_stripe_insertion/pomacanthus_semicirculatus_specimen.jpg` to detect the curved body stripes along the anterior-posterior flank transect, quantify primary vs newly inserted secondary mid-trough stripe prominences and inter-stripe wavelength in mm, and enforce runtime SHA-256 verification.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Kondo & Asai (1995) Pomacanthus reaction-diffusion stripe insertion & Y-branch unzipping
- **spec_fingerprint** (semantic model + inputs): `e4c6766f0513e073e4470f6a811662276c20929359a8102829c89d29daf6842b`
- **input_dataset_sha256**: `datasets/pomacanthus_stripe_insertion/pomacanthus_semicirculatus_specimen.jpg` (`5564f850c1905fb6…`), `datasets/pomacanthus_stripe_insertion/pomacanthus_stripe_insertion_reference.json` (`4e059d555df1a3d7…`), `datasets/pomacanthus_stripe_insertion/scout_manifest.json` (`bf877731da64b97b…`)
- `docs/media/pomacanthus_stripe_insertion_comparison.png` — artifact_sha256 `35275dea0eb0c598…`, 390,103 B, 1200×504 px

| measurement | value | reference | method |
|---|---:|---|---|
| Kondo-Asai (1995) critical diffusivity ratio d_c = (D_I / D_A)_c | 10.2596 dimensionless | 10.2596 | Analytical Turing bifurcation threshold for Kondo & Asai (1995) Fig. 1 Jacobian |
| Kondo-Asai (1995) intrinsic Turing wavelength lambda_c | 3.4739 grid units | 3.4739 | 2*pi / k_c from linearized Kondo-Asai reaction-diffusion dispersion relation |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1995-kondo-pomacanthus-stripe-insertion.py` in `2026-10-19-pomacanthus-stripe-insertion-loop` prior to Pass 2 closed-loop remediation.
