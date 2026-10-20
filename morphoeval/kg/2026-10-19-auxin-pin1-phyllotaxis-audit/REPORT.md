# MorphoEval report — audit `2026-10-19-auxin-pin1-phyllotaxis-audit`

Rubric **0.1.0** · checkpoint `0cb0132` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/32

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Jönsson et al. (2006) & Smith et al. (2006) up-the-gradient PIN1 polarization and polar auxin transport model for shoot apical meristem phyllotaxis (`2006-jonsson-auxin-pin1-meristem.py`) | 56 | 45 | 51 | 52 | 68 | 58 | 54 | 54.3 | 47.6 | 59.7 | 53.3 | **53.3** | partially faithful |

## Findings and remediation

### Jönsson et al. (2006) & Smith et al. (2006) up-the-gradient PIN1 polarization and polar auxin transport model for shoot apical meristem phyllotaxis  —  overall 53.3 (partially faithful)

Pass 1 (v1) baseline evaluation of `2006-jonsson-auxin-pin1-meristem.py` (weighted score: 53.8/100). The Pass 1 baseline script implements only a 32-cell 1D periodic ring without a 2D cellular L1 meristem dome topology, radial growth advection and primordial auxin depletion sinks, dispersion-relation eigenvalue verification, transitions across the 4 phyllotactic regimes (spiral 137.5 deg, distichous 180 deg, decussate 90 deg, tricussate 120 deg), NPA/pin1 pharmacological block and exogenous IAA microdroplet rescue experiments, or segmentation of the vendored Aeonium tabuliforme spiral rosette specimen.

- **Model fidelity to source: 56** — Implements a minimal 1D periodic ring with Boltzmann PIN1 weights exp(c_1 a_j) and active + diffusive auxin transport (L22-50), but omits the 2D cellular L1 shoot apical meristem dome with central zone (CZ) vs peripheral organogenic ring (PZ), radial tissue advection and primordial vascular sink depletion, linearized Turing-like dispersion analysis, and 4-mode phyllotactic transitions.  
  evidence: `2006-jonsson-auxin-pin1-meristem.py` L22–50
- **Morphological realism / specimen match: 45** — Vendors `datasets/auxin_pin1_phyllotaxis/rosette_phyllotaxis_specimen.jpg` on disk, but Pass 1 neither loads nor segments the Aeonium tabuliforme spiral rosette to compare leaf primordia centroids, divergence angles, or parastichy windings against the simulation.  
  evidence: `2006-jonsson-auxin-pin1-meristem.py` L75–102
- **Parameter expressivity: 51** — Runs a single hardcoded parameter tuple without sweeping the meristem ring-to-spacing ratio Gamma across spiral (137.5 deg), distichous (180 deg), decussate (90 deg), and tricussate (120 deg) phyllotactic modes, or testing NPA polar-transport block and IAA paste rescue.  
  evidence: `2006-jonsson-auxin-pin1-meristem.py` L22–31
- **Morphometric validation: 52** — Exposes only 2 static parameter ratios in `measure_morphoeval()` (L53-72) without divergence angle statistics, parastichy counts, PIN1 flux conservation checks, NPA/IAA perturbation metrics, or specimen segmentation validation.  
  evidence: `2006-jonsson-auxin-pin1-meristem.py` L53–72
- **Source and data traceability: 68** — Vendors `paper_citations.bib` and `auxin_pin1_phyllotaxis_reference.json` with verified DOIs, but Pass 1 does not load or SHA-256-verify the dataset at runtime.  
  evidence: `datasets/auxin_pin1_phyllotaxis/auxin_pin1_phyllotaxis_reference.json` L1–120
- **Code-to-model traceability: 58** — Module header cites Jönsson et al. (2006) and Smith et al. (2006), but lacks explicit equation-to-code-line and Lean 4 theorem mappings.  
  evidence: `2006-jonsson-auxin-pin1-meristem.py` L1–15
- **Reproducibility package: 54** — Generates PNG and GIF outputs deterministically, but lacks a `--self-test` verification suite, SHA-256 checksum guards, and perturbation controls.  
  evidence: `2006-jonsson-auxin-pin1-meristem.py` L75–106

Remediation:
- Implement a 2D cellular L1 shoot apical meristem (SAM) dome + peripheral organogenic ring simulator with up-the-gradient Boltzmann PIN1 allocation P_ij = P_i exp(c_1 a_j) / sum_k exp(c_1 a_k), active + diffusive auxin transport, radial growth advection, and vascular sink depletion.
- Implement quantitative transitions across all 4 phyllotactic regimes of Smith et al. (2006) & Jönsson et al. (2006): Spiral Fibonacci (137.5 deg), Distichous alternate (180 deg), Decussate opposite pairs (90 deg), and Tricussate whorled triplets (120 deg), plus linear Turing-like instability dispersion analysis.
- Implement the Reinhardt et al. (2003) / Smith et al. (2006) perturbation experiments: NPA / pin1 polar transport block (T = 0 -> bare pin-formed apex with 0 primordia) and localized exogenous IAA lanolin microdroplet rescue (inducing a single primordium at phi_paste = 60 deg).
- Segment the vendored Aeonium tabuliforme spiral rosette specimen (`rosette_phyllotaxis_specimen.jpg`) to extract leaf primordia centroids, radial phyllotactic scaling, and Fibonacci parastichy spirals, with full SHA-256 verification and an 8-check `--self-test` suite.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Jönsson et al. (2006) & Smith et al. (2006) up-the-gradient PIN1 polarization and polar auxin transport model for shoot apical meristem phyllotaxis
- **spec_fingerprint** (semantic model + inputs): `722bfcfebddd9b7a82c1bd7b9d04a028378c8556fe894a24ae273b6224593820`
- **input_dataset_sha256**: `datasets/auxin_pin1_phyllotaxis/auxin_pin1_phyllotaxis_reference.json` (`96d4b539eaaf1649…`), `datasets/auxin_pin1_phyllotaxis/paper_citations.bib` (`9fe96650b32df64b…`), `datasets/auxin_pin1_phyllotaxis/rosette_phyllotaxis_specimen.jpg` (`ec50801b52dabc46…`), `datasets/auxin_pin1_phyllotaxis/scout_manifest.json` (`2e474ee5544f04f4…`)
- `docs/media/auxin_pin1_phyllotaxis_comparison.png` — artifact_sha256 `6b600014785b8ea3…`, 55,606 B, 1200×504 px

| measurement | value | reference | method |
|---|---:|---|---|
| homogeneous_auxin_steady_state | 2.0 a.u. | 2.0 | Analytical ratio rho_0 / mu |
| pin1_feedback_product_c1_a0 | 3.7 dimensionless | 3.7 | 1D ring baseline feedback parameter product |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2006-jonsson-auxin-pin1-meristem.py` prior to Step 12 closed-loop remediation.
