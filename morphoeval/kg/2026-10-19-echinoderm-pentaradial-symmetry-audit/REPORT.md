# MorphoEval report — audit `2026-10-19-echinoderm-pentaradial-symmetry-audit`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/75

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Morris (2012), Mooi & David (1998) & Meinhardt (2008) echinoderm bilateral-to-pentaradial hydrocoel ring symmetry breaking & Lovén's Law (`2005-meinhardt-echinoderm-pentaradial-symmetry.py`) | 54 | 46 | 49 | 50 | 62 | 55 | 53 | 52.3 | 47.5 | 56.5 | 51.7 | **51.7** | partially faithful |

## Findings and remediation

### Morris (2012), Mooi & David (1998) & Meinhardt (2008) echinoderm bilateral-to-pentaradial hydrocoel ring symmetry breaking & Lovén's Law  —  overall 51.7 (partially faithful)

Pass 1 (v1) baseline evaluation of `2005-meinhardt-echinoderm-pentaradial-symmetry.py` (consensus score 52.7/100). Evaluates an algebraic dispersion curve on a static circle to pick m* = 5 (72 deg), but omits a genuine Gierer-Meinhardt reaction-diffusion PDE simulation on a closing hydrocoel crescent-to-ring domain, omits Lovén's Law (1874) asymmetric initiation sequence (II, IV, I, III, V) and biserial peristomial plate enlargement (ba-ab-ba-ab-ab), omits Duboc et al. (2005) right-sided Nodal knockdown twin-rudiment (2x5 = 10 podia) perturbation, lacks polar Fourier C_5 harmonic segmentation of the vendored Tripneustes gratilla echinoid test specimen, and lacks runtime SHA-256 dataset verification.

- **Model fidelity to source: 54** — Evaluates a static algebraic dispersion curve sigma(m) and analytic cosine ring envelope to illustrate m* = 5, but omits time-stepping the Gierer-Meinhardt activator-inhibitor PDE during hydrocoel crescent-to-ring closure, omits Lovén's Law (II, IV, I, III, V) sequential initiation, and omits Duboc et al. (2005) Nodal left-right twin-rudiment (2x5) symmetry breaking.  
  evidence: `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L33–82
- **Morphological realism / specimen match: 46** — Displays the vendored Tripneustes gratilla regular echinoid test specimen in Panel D without performing polar Fourier C_5 harmonic analysis or detecting the 5 ambulacral ray angles and 72.0 deg inter-ray spacings.  
  evidence: `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L106–108
- **Parameter expressivity: 49** — Evaluates 4 ring radii in the GIF loop, but lacks a full mode-selection staircase across 2*pi*R_ring/lambda_c spanning m in {3, 4, 5, 6, 7}, lacks Nodal LOF/GOF perturbation sweeps, and provides no CLI parameter flags.  
  evidence: `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L118–134
- **Morphometric validation: 50** — Exposes only 2 scalar metrics in `measure_morphoeval()` without testing the hydrocoel ring mode staircase, Lovén's Law ray initiation rank correlation, Duboc Nodal LOF 2x5 twin-rudiment podia count, or specimen C_5 Fourier power.  
  evidence: `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L137–159
- **Source and data traceability: 62** — Vendors `datasets/echinoderm_pentaradial_symmetry/` with 5 BibTeX references and the CC BY-SA 4.0 Tripneustes gratilla test micrograph, but does not verify SHA-256 digests at runtime.  
  evidence: `datasets/echinoderm_pentaradial_symmetry/echinoderm_pentaradial_symmetry_reference.json` L1–125
- **Code-to-model traceability: 55** — Header docstring cites Morris (2012), Mooi & David (1998), and Meinhardt (2008), but does not map functions to the 3 machine-checked Lean 4 theorems in `Loop_075_echinoderm_pentaradial_symmetry.lean`.  
  evidence: `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L1–11
- **Reproducibility package: 53** — Provides `--self-test` and `--skip-gif`, but `self_test()` checks only 3 basic assertions and lacks bitwise determinism and SHA-256 integrity checks.  
  evidence: `2005-meinhardt-echinoderm-pentaradial-symmetry.py` L162–168

Remediation:
- Implement a time-stepped Gierer-Meinhardt activator-inhibitor reaction-diffusion PDE on a closing hydrocoel crescent-to-ring domain, demonstrating spontaneous m^* = 5 (Delta_theta = 72 deg) lobe formation and sequential anterior-to-posterior initiation matching Lovén's Law (II, IV, I, III, V) and the asymmetric biserial peristomial plate law (Ib, IIb, IIIa, IVa, Va).
- Implement the full hydrocoel circumference-to-wavelength mode selection staircase m^* in {3, 4, 5, 6, 7} vs 2*pi*R_ring/lambda_c alongside a homogeneous sub-threshold negative control (D_h = D_a, zero spatial Turing instability).
- Implement Duboc et al. (2005) right-sided Nodal/BMP2-4 left-right symmetry breaking and its perturbations: wild-type single left rudiment (1x5 = 5 podia), Nodal morpholino knockdown bilateral twin rudiments (2x5 = 10 podia), and ectopic bilateral Nodal overexpression (0 podia).
- Implement automated polar Fourier C_5 harmonic segmentation (`segment_echinoid_test_specimen`) on `echinoderm_pentaradial_test_specimen.jpg` to detect the 5 ambulacral rays (72.0 +/- 1.4 deg spacing, C_5 fundamental power fraction 0.9255), plus runtime SHA-256 dataset verification, 9 morphometrics in `measure_morphoeval()`, and an 8-check `self_test()` suite.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Morris (2012), Mooi & David (1998) & Meinhardt (2008) echinoderm bilateral-to-pentaradial hydrocoel ring symmetry breaking & Lovén's Law
- **spec_fingerprint** (semantic model + inputs): `f94f62955d7ff6974c8a237ced577144db975e6e7e74430aca24de11b763608f`
- **input_dataset_sha256**: `datasets/echinoderm_pentaradial_symmetry/echinoderm_pentaradial_symmetry_reference.json` (`c516e298727883cf…`), `datasets/echinoderm_pentaradial_symmetry/echinoderm_pentaradial_test_specimen.jpg` (`953b2f7e38029627…`), `datasets/echinoderm_pentaradial_symmetry/scout_manifest.json` (`0b333ebe2a2ed524…`)
- `docs/media/echinoderm_pentaradial_symmetry_comparison.png` — artifact_sha256 `28b98789b0b62803…`, 659,701 B, 1331×1177 px

| measurement | value | reference | method |
|---|---:|---|---|
| Selected hydrocoel ring primary podia mode m^* at R_ring=75 um | 5.0 lobes | 5.0 | Argmax of circular-ring Turing dispersion relation sigma(m; R_ring=75 um) |
| Pentaradial inter-ambulacral angular spacing Delta_theta | 72.0 deg | 72.0 | 360 / m^* for canonical hydrocoel ring radius R_ring=75 um |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2005-meinhardt-echinoderm-pentaradial-symmetry.py` prior to Step 12 closed-loop remediation.
