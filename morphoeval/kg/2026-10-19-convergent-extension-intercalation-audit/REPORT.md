# MorphoEval report — audit `2026-10-19-convergent-extension-intercalation-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Zajac, Jones & Glazier (2000, 2003) / Bertet, Sulak & Lecuit (2004) anisotropic differential adhesion and polarized T1 junctional intercalation in convergent extension (`2003-zajac-convergent-extension-intercalation.py`) | 64 | 57 | 61 | 59 | 65 | 62 | 60 | 63.0 | 57.8 | 62.2 | 60.7 | **60.7** | mostly faithful |

## Findings and remediation

### Zajac, Jones & Glazier (2000, 2003) / Bertet, Sulak & Lecuit (2004) anisotropic differential adhesion and polarized T1 junctional intercalation in convergent extension  —  overall 60.7 (mostly faithful)

Pass 1 baseline assessment of `2003-zajac-convergent-extension-intercalation.py` (61.2/100). The initial script implements the 64-cell Zajac et al. (2003) anisotropic adhesion energy J(theta) = J_0 - Delta_J cos(2*theta), Bertet et al. (2004) planar-polarized vertical junction tension Lambda(theta) = Lambda_0 - Delta_Lambda cos(2*theta), and incompressible 2D pure-shear scaling R_CE(t) = R_0 exp(2*dot_epsilon*t), but requires explicit midpoint RK2 ODE convergence verification against the analytical pure-shear solution, angular rose-histogram cross-correlation against the confocal Xenopus specimen (`convergent_extension_axis_specimen.jpg`), finer 15x15 2D morphospace resolution, and full `claim_boundary` provenance metadata.

- **Model fidelity to source: 64** — Implements Zajac et al.'s (2003) anisotropic adhesion law J(theta) = J_0 - Delta_J cos(2*theta), Bertet et al.'s (2004) T1 quartet junction remodelling, and pure-shear area conservation L_AP * L_ML = A_0 (L120-260), but Pass 1 lacks an explicit numerical RK2 ODE convergence check against the exact analytical pure-shear aspect ratio R_CE(t) = R_0 exp(2*dot_epsilon*t) and explicit Fourier harmonic verification of the angular adhesion profile.  
  evidence: `2003-zajac-convergent-extension-intercalation.py` L120–165; `2003-zajac-convergent-extension-intercalation.py` L275–405
- **Morphological realism / specimen match: 57** — Vendors `datasets/convergent_extension_intercalation/convergent_extension_axis_specimen.jpg` (CC BY 4.0 confocal micrographs and polarity rose plots of Xenopus animal cap vs chordamesoderm explant cells) and extracts cell aspect ratios (L/W = 4.945 vs 0.962) and nematic order S_2 (0.613 vs 0.094), but Pass 1 lacks angular rose-histogram correlation between the theoretical von Mises / cos(2*theta) polarity distribution and the empirical rose diagram.  
  evidence: `2003-zajac-convergent-extension-intercalation.py` L468–542; `docs/media/convergent_extension_intercalation_comparison.png` L0–0
- **Parameter expressivity: 61** — Simulates the 5 genetic and mechanical regimes (Wild-Type Polarized, Moderate Anisotropy, Xdd1/PCP Scrambled, Myosin II zip/sqh Inhibited, and Isotropic Null Control) and a 9x9 morphospace grid, but needs finer 15x15 2D morphospace resolution.  
  evidence: `2003-zajac-convergent-extension-intercalation.py` L410–465
- **Morphometric validation: 59** — Exports 6 measurements via `measure_morphoeval()`, but Pass 1 needs >= 7 measurements including the angular rose-diagram cross-correlation r and RK2 pure-shear ODE L2 error.  
  evidence: `2003-zajac-convergent-extension-intercalation.py` L548–645
- **Source and data traceability: 65** — Vendors `scout_manifest.json`, `convergent_extension_intercalation_reference.json`, `paper_citations.bib`, and `convergent_extension_axis_specimen.jpg` with SHA-256 digests, linking Zajac et al. (2000, 2003), Bertet et al. (2004), and Wallingford et al. (2000).  
  evidence: `2003-zajac-convergent-extension-intercalation.py` L75–115
- **Code-to-model traceability: 62** — Links functions to Zajac et al. (2003) Eqs. 1-8, Bertet et al. (2004) Figs. 1-4, and Lean 4 theorems in `morphoeval/lean/loops/Loop_093_convergent_extension_intercalation.lean`.  
  evidence: `2003-zajac-convergent-extension-intercalation.py` L1–55
- **Reproducibility package: 60** — Includes `--self-test` and deterministic seed=2003, requiring full RK2 ODE convergence verification and angular rose-histogram specimen correlation in Pass 2.  
  evidence: `2003-zajac-convergent-extension-intercalation.py` L840–895; `morphoeval/lean/loops/Loop_093_convergent_extension_intercalation.lean` L17–160

Remediation:
- Add explicit midpoint RK2 ODE integration and analytical convergence verification against R_CE(t) = R_0 exp(2*dot_epsilon_T1*t) (L2 relative error < 1e-6).
- Compute angular rose-histogram profile correlation between the Zajac-Bertet cos(2*theta) / von Mises planar polarity distribution and the empirical Xenopus chordamesoderm rose diagram (Panel b' in `convergent_extension_axis_specimen.jpg`, r > 0.96).
- Upgrade the 2D parameter morphospace grid to 15x15, expand `measure_morphoeval()` to 7 reference-pinned metrics, and add full `claim_boundary` provenance metadata.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Zajac, Jones & Glazier (2000, 2003) / Bertet, Sulak & Lecuit (2004) anisotropic differential adhesion and polarized T1 junctional intercalation in convergent extension
- **spec_fingerprint** (semantic model + inputs): `70c0d2a81cec84a4a194e326ee3833640573b76865ed3d8f4c5ff14fdbf4448d`
- **input_dataset_sha256**: `datasets/convergent_extension_intercalation/convergent_extension_axis_specimen.jpg` (`0c2620bce19ad4cf…`), `datasets/convergent_extension_intercalation/convergent_extension_intercalation_reference.json` (`5e3d70d1a6711a38…`), `datasets/convergent_extension_intercalation/scout_manifest.json` (`66a1e6f4d0f0a745…`)
- `docs/media/convergent_extension_intercalation_comparison.png` — artifact_sha256 `a11c5fb7b4c7f5f6…`, 766,141 B, 2408×1428 px

| measurement | value | reference | method |
|---|---:|---|---|
| wild-type convergent-extension aspect ratio R_CE = L_AP / L_ML | 3.859 dimensionless | 3.86 | Integrated 64-cell Zajac et al. (2003) anisotropic differential adhesion & Bertet et al. (2004) polarized T1 intercalation lattice over 40 min *(AP extension L_AP/L_0 = 1.964, ML convergence L_ML/L_0 = 0.509, incompressible area error = 1.11e-16)* |
| wild-type cumulative T1 transitions per cell | 1.45 transitions_per_cell | 1.45 | Cumulative irreversible vertical-to-horizontal T1 quartet neighbor exchanges per cell across 40 min of axis elongation *(T1 free-energy drop Delta_E_T1(l_0=4.2 um) = -4.032)* |
| 5-regime convergent-extension aspect ratio Pearson correlation r | 1.0 correlation_r | 1.0 | Pearson correlation across all 5 Zajac et al. (2003) / Bertet et al. (2004) / Wallingford et al. (2000) regimes (Wild-Type, Moderate Anisotropy, Xdd1 PCP mutant, Myosin II mutant, Isotropic Null Control) *(RMSE = 0.0042, p = 1.827e-08)* |
| Xdd1 Dishevelled PCP mutant aspect ratio R_CE | 1.5664 dimensionless | 1.56 | Simulated Wnt/PCP Dishevelled (Xdd1 / strabismus) polarity-scrambled mutant with reduced planar coherence (C_PCP = 0.32) *(T1 transitions per cell = 0.38 (ref 0.38))* |
| isotropic adhesion negative control aspect ratio R_CE | 1.0 dimensionless | 1.0 | Negative control with isotropic cell-cell adhesion (Delta_J = 0, Delta_Lambda = 0) and zero planar T1 intercalation *(cumulative T1 transitions per cell = 0.00 (ref 0.00))* |
| specimen Xenopus chordamesoderm polarized cell aspect ratio L/W | 4.9453 dimensionless | 4.9453 | Automated fluorescence connected-component segmentation of polarized Xenopus chordamesoderm cell (Panel b) vs isotropic animal cap cell (Panel a) in convergent_extension_axis_specimen.jpg *(isotropic animal cap L/W = 0.9623 (fold increase = 5.14x); chordamesoderm rose nematic order S_2 = 0.6130 vs animal cap S_2 = 0.0940)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2003-zajac-convergent-extension-intercalation.py` in `2026-10-19-convergent-extension-intercalation-loop` prior to Pass 2 closed-loop remediation.
