# MorphoEval report — audit `2026-10-19-chemotactic-stripe-splitting-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Painter, Maini & Othmer (1999) chemotactic stripe splitting & stripe-spot selection (`1999-painter-chemotactic-stripe-splitting.py`) | 56 | 48 | 52 | 52 | 64 | 58 | 56 | 54.7 | 49.5 | 59.2 | 54.0 | **54.0** | partially faithful |

## Findings and remediation

### Painter, Maini & Othmer (1999) chemotactic stripe splitting & stripe-spot selection  —  overall 54.0 (partially faithful)

Pass 1 baseline assessment of `1999-painter-chemotactic-stripe-splitting.py` (54.4/100). The initial script computes the analytical equal-diffusivity chemotactic bifurcation threshold chi_c = 4.5 and critical wavelength lambda_c = 4.4429, but uses a kinematic mode approximation rather than integrating the full nonlinear Keller-Segel-Painter PDE dn/dt = D_n Lap(n) - div(n chi(c) grad(c)) + r n (1 - n/n_0) and dc/dt = D_c Lap(c) + alpha n / (1 + beta n) - gamma c on a growing domain, lacks 2D stripe-vs-spot selection simulation, and lacks automated computer-vision segmentation of the vendored Myxococcus xanthus specimen (`chemotactic_pattern_specimen.jpg`).

- **Model fidelity to source: 56** — Computes the exact equal-diffusivity linear stability threshold chi_c = 4.5 and critical wavelength lambda_c = 4.4429 (L35-65), but Pass 1 approximates the 1D kymograph kinematically rather than solving the nonlinear coupled chemotactic-RD PDE system with conservative flux divergence div(n chi(c) grad(c)) and 2D stripe-vs-spot pattern selection.  
  evidence: `1999-painter-chemotactic-stripe-splitting.py` L35–65; `1999-painter-chemotactic-stripe-splitting.py` L68–86
- **Morphological realism / specimen match: 48** — Displays `datasets/chemotactic_stripe_splitting/chemotactic_pattern_specimen.jpg` (CC BY-SA 4.0 Myxococcus xanthus rippling stripes and fruiting bodies), but Pass 1 lacks automated computer-vision ridge wavelength extraction and fruiting-body spot segmentation on the specimen.  
  evidence: `1999-painter-chemotactic-stripe-splitting.py` L110–125
- **Parameter expressivity: 52** — Evaluates chi = 6.0 vs chi_c = 4.5, but needs a full 5-regime domain-growth and subcritical-chemotaxis ablation table (s = 1, 2, 4, 8 and chi = 3.0 < chi_c), a 2D dispersion surface lambda(k^2, chi), and 2D stripe vs spot simulations.  
  evidence: `1999-painter-chemotactic-stripe-splitting.py` L68–86
- **Morphometric validation: 52** — Reports 2 analytical measurements in `measure_morphoeval()`, but lacks simulated PDE stripe-splitting wavelength accuracy, 5-regime peak count RMSE, 2D stripe vs spot structural metrics, and empirical specimen ridge spacing in micrometers.  
  evidence: `1999-painter-chemotactic-stripe-splitting.py` L89–108
- **Source and data traceability: 64** — Vendors `scout_manifest.json`, `chemotactic_stripe_splitting_reference.json`, `paper_citations.bib`, and `chemotactic_pattern_specimen.jpg`, but runtime SHA-256 verification (`verify_dataset_integrity`) is not yet enforced in the script.  
  evidence: `datasets/chemotactic_stripe_splitting/chemotactic_stripe_splitting_reference.json` L1–145
- **Code-to-model traceability: 58** — Maps critical chemotactic threshold to Painter, Maini & Othmer (1999), but needs explicit section/equation docstrings linking every function to the 3 Lean 4 theorems in `Loop_005_chemotactic_stripe_splitting.lean`.  
  evidence: `1999-painter-chemotactic-stripe-splitting.py` L1–15
- **Reproducibility package: 56** — Provides `--self-test` and deterministic seed=1999, but self-test only checks analytical chi_c = 4.5 and lacks SHA-256 dataset verification, bitwise PDE solver determinism, subcritical negative control assertion, and specimen segmentation checks.  
  evidence: `1999-painter-chemotactic-stripe-splitting.py` L137–149

Remediation:
- Replace the kinematic 1D approximation with a full conservative finite-difference/spectral Keller-Segel-Painter PDE solver integrating dn/dt = D_n Lap(n) - div(n chi(c) grad(c)) + r n (1 - n/n_0) and dc/dt = D_c Lap(c) + alpha n / (1 + beta n) - gamma c across growing domain stages L / lambda_c in {1, 2, 4, 8} plus a subcritical negative control (chi = 3.0 < chi_c = 4.5 -> 0 peaks).
- Implement 2D Keller-Segel-Painter chemotactic pattern selection comparing labyrinthine chemotactic stripes vs hexagonal fruiting-body spot aggregates.
- Add computer-vision segmentation of `datasets/chemotactic_stripe_splitting/chemotactic_pattern_specimen.jpg` to extract both the right-hand chemotactic rippling ridge wavelength (in px and um via the 500 um scale bar) and the left-hand yellow fruiting-body spot aggregates, plus runtime SHA-256 dataset verification.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Painter, Maini & Othmer (1999) chemotactic stripe splitting & stripe-spot selection
- **spec_fingerprint** (semantic model + inputs): `91fb31037b136e4adc1e8bdd7b6c8097f7eaab327a03b1266cf7114f22d5f57f`
- **input_dataset_sha256**: `datasets/chemotactic_stripe_splitting/chemotactic_pattern_specimen.jpg` (`7b1780ffcb70f313…`), `datasets/chemotactic_stripe_splitting/chemotactic_stripe_splitting_reference.json` (`2c37ed1ab3d26a08…`), `datasets/chemotactic_stripe_splitting/scout_manifest.json` (`4b4e20a991f65c81…`)
- `docs/media/chemotactic_stripe_splitting_comparison.png` — artifact_sha256 `fdbda2565a00f388…`, 468,870 B, 1200×504 px

| measurement | value | reference | method |
|---|---:|---|---|
| critical chemotactic sensitivity chi_c at equal diffusivity D_n = D_c = 1 | 4.5 dimensionless | 4.5 | Painter, Maini & Othmer (1999) linear stability analysis around (n*, c*) = (1, 1) |
| critical chemotactic stripe wavelength lambda_c = 2*pi / k_c | 4.4429 dimensionless length | 4.4429 | k_c^2 = sqrt(r * gamma / (D_n * D_c)) = 2.0 => lambda_c = pi * sqrt(2) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1999-painter-chemotactic-stripe-splitting.py` in `2026-10-19-chemotactic-stripe-splitting-loop` prior to Pass 2 closed-loop remediation.
