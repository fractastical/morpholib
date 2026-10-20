# MorphoEval report — audit `2026-10-19-auxin-pin1-phyllotaxis-audit-v2`

Rubric **0.1.0** · checkpoint `0cb0132` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/32 · supersession chain `2026-10-19-auxin-pin1-phyllotaxis-audit -> 2026-10-19-auxin-pin1-phyllotaxis-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Jönsson et al. (2006) & Smith et al. (2006) up-the-gradient PIN1 polarization and polar auxin transport model for shoot apical meristem phyllotaxis (`2006-jonsson-auxin-pin1-meristem.py`) | 94 | 91 | 91 | 93 | 95 | 92 | 93 | 93.0 | 91.8 | 93.4 | 92.6 | **92.6** | faithful |

## Findings and remediation

### Jönsson et al. (2006) & Smith et al. (2006) up-the-gradient PIN1 polarization and polar auxin transport model for shoot apical meristem phyllotaxis  —  overall 92.6 (faithful)

Pass 2 (v2) evaluation of `2006-jonsson-auxin-pin1-meristem.py` (overall weighted score: 92.8/100, up from 53.8/100 in v1). The remediated script implements the linearized Turing-like up-the-gradient dispersion relation and Boltzmann PIN1 allocation law (critical threshold c_1*a_0 = 1.75, characteristic spacing lambda_c = 5.196 cells / 36.37 um); integrates a 180-cell 2D SAM L1 Auxin-PIN1 ODE model reproducing golden-angle phyllotaxis (137.39 ± 1.24 deg vs ref 137.51 deg) and PIN1 convergence polarity index 0.8527 (ref 0.84); reproduces transitions across all 4 phyllotactic regimes (Spiral 137.5 deg, Distichous 179.5 deg, Decussate 89.9 deg, Tricussate 120.1 deg; Pearson r = 1.0000, p = 2.23e-5); validates Reinhardt et al. (2003) NPA/pin1-1 polar transport block (contrast 1.0000, 0 primordia) and localized IAA lanolin paste rescue at 60.0 deg (1 primordium, 0.00 deg error); and segments 410 spiral rosette leaves on the SHA-256-locked Aeonium tabuliforme specimen (`rosette_phyllotaxis_specimen.jpg`, NN packing regularity 1 - CV = 0.7737 vs ref 0.75).

- **Model fidelity to source: 94** — Implements Boltzmann up-the-gradient PIN1 membrane allocation and linearized Turing-like dispersion eigenvalues (`evaluate_up_gradient_dispersion_and_allocation`, L102-182), a 180-cell 2D L1 shoot apical meristem coupled Auxin-PIN1 ODE integrator (`simulate_2d_sam_auxin_pin1_phyllotaxis`, L188-348), 4-regime phyllotactic mode transitions across Gamma = R_PZ / lambda_0 (`simulate_four_phyllotactic_modes`, L354-441), and NPA/pin1 block + exogenous IAA microdroplet rescue (`simulate_npa_and_iaa_rescue_experiments`, L447-531), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_032_auxin_pin1_phyllotaxis.lean`.  
  evidence: `2006-jonsson-auxin-pin1-meristem.py` L102–348; `2006-jonsson-auxin-pin1-meristem.py` L354–531; `morphoeval/lean/loops/Loop_032_auxin_pin1_phyllotaxis.lean` L14–110
- **Morphological realism / specimen match: 91** — Vendors and SHA-256-verifies the CC BY-SA 3.0 top-down spiral rosette specimen of Aeonium tabuliforme (`rosette_phyllotaxis_specimen.jpg`), segments the circular rosette disc and SAM center via excess-green chromaticity (ExG = 2G - R - B), detects 410 imbricated leaf tips via Difference-of-Gaussians local maxima across all 12 azimuthal sectors (coverage = 1.0000), and verifies leaf packing regularity 1 - CV = 0.7737 (`segment_rosette_phyllotaxis_specimen`, L537-609).  
  evidence: `2006-jonsson-auxin-pin1-meristem.py` L537–609; `docs/media/auxin_pin1_phyllotaxis_comparison.png` L0–0
- **Parameter expressivity: 91** — Evaluates 4 phyllotactic mode regimes spanning Gamma in [0.85, 2.75] and R_PZ in [20.5, 66.5] um (L354-441), 3 pharmacological/genetic perturbation regimes (Wild-Type, 10 uM NPA / pin1-1 block, and NPA + 1 mM IAA paste rescue, L447-531), and exposes `--T-polar`, `--D-diff`, `--c1-gain`, `--rho0`, `--mu`, and `--seed` CLI overrides (L870-909).  
  evidence: `2006-jonsson-auxin-pin1-meristem.py` L354–441; `2006-jonsson-auxin-pin1-meristem.py` L870–909
- **Morphometric validation: 93** — `measure_morphoeval()` (L748-825) computes 9 reference-anchored morphometric measurements with uncertainty, p-values, and negative controls, all 9 within 3.2% of reference benchmarks: spiral divergence angle 137.39 ± 1.24 deg (ref 137.51), characteristic spacing wavelength 5.196 cells (ref 5.2), critical PIN1 feedback threshold c_1*a_0 = 1.7500 (ref 1.75), PIN1 convergence polarity index 0.8527 (ref 0.84), 4-mode divergence angle Pearson r = 1.0000 (p = 2.23e-5, RMSE = 0.24 deg), NPA polar transport block contrast 1.0000 (ref 1.0, 0 primordia), exogenous IAA paste rescue peak angle 60.00 deg (ref 60.0 deg, 0.00 deg error), rosette azimuthal sector coverage 1.0000 (ref 1.0), and leaf spacing regularity 1 - CV = 0.7737 (ref 0.75).  
  evidence: `2006-jonsson-auxin-pin1-meristem.py` L748–825
- **Source and data traceability: 95** — Vendors `datasets/auxin_pin1_phyllotaxis/` with 5 primary BibTeX entries and verified DOIs (`10.1073/pnas.0509839103`, `10.1073/pnas.0510457103`, `10.1038/nature02081`, `10.1016/j.cub.2005.09.052`, `10.1073/pnas.0510430103`), `scout_manifest.json`, `auxin_pin1_phyllotaxis_reference.json`, and `rosette_phyllotaxis_specimen.jpg`, all verified at runtime via SHA-256 checksums (`verify_dataset_integrity`, L73-93).  
  evidence: `2006-jonsson-auxin-pin1-meristem.py` L73–93; `datasets/auxin_pin1_phyllotaxis/auxin_pin1_phyllotaxis_reference.json` L1–135
- **Code-to-model traceability: 92** — Module and function docstrings map every equation to Jönsson et al. (2006) PNAS Eqs. 1-4 & Figs. 2-4, Smith et al. (2006) PNAS Eqs. 1-3 & Figs. 3-5, Reinhardt et al. (2003) Nature Figs. 1-2, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_032_auxin_pin1_phyllotaxis.lean` (L1-36, L102-230).  
  evidence: `2006-jonsson-auxin-pin1-meristem.py` L1–36
- **Reproducibility package: 93** — 8-point `run_self_test()` suite (`L828-868`) verifying SHA-256 dataset integrity, Boltzmann PIN1 pool conservation (< 1e-14) and up-the-gradient monotonicity, Turing-like transport instability threshold and NPA dispersion collapse, 2D SAM intercellular flux conservation (< 1e-11), 4 phyllotactic mode regimes, NPA block and IAA paste rescue, rosette specimen segmentation, and bitwise determinism.  
  evidence: `2006-jonsson-auxin-pin1-meristem.py` L828–868

Remediation:
- Couple the 2D L1 epidermal dome Auxin-PIN1 simulator to explicit sub-cellular PIN1 exocytosis/endocytosis trafficking kinetics and 3D subepidermal canalization of provascular strands.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Jönsson et al. (2006) & Smith et al. (2006) up-the-gradient PIN1 polarization and polar auxin transport model for shoot apical meristem phyllotaxis
- **spec_fingerprint** (semantic model + inputs): `722bfcfebddd9b7a82c1bd7b9d04a028378c8556fe894a24ae273b6224593820`
- **input_dataset_sha256**: `datasets/auxin_pin1_phyllotaxis/auxin_pin1_phyllotaxis_reference.json` (`96d4b539eaaf1649…`), `datasets/auxin_pin1_phyllotaxis/paper_citations.bib` (`9fe96650b32df64b…`), `datasets/auxin_pin1_phyllotaxis/rosette_phyllotaxis_specimen.jpg` (`ec50801b52dabc46…`), `datasets/auxin_pin1_phyllotaxis/scout_manifest.json` (`2e474ee5544f04f4…`)
- `docs/media/auxin_pin1_phyllotaxis_comparison.png` — artifact_sha256 `2483a59dbe5c9790…`, 1,612,274 B, 2548×1596 px

| measurement | value | reference | method |
|---|---:|---|---|
| spiral_phyllotaxis_divergence_angle_mean | 137.39 deg | 137.51 | 180-cell 2D SAM L1 Auxin-PIN1 ODE integration across P1..P8 (mean = 137.39 ± 1.24 deg SD) |
| characteristic_primordium_spacing_wavelength | 5.196 cells | 5.2 | Linearized dispersion eigenvalue peak k_max*l_0 = 1.2093 rad/cell (sigma_max = 0.2017 h^-1, lambda_c = 36.37 um) |
| critical_pin1_feedback_threshold_c1_a0 | 1.75 dimensionless | 1.75 | Exact linear stability threshold where D_eff_scaled = -2.3400 < 0 (Lean 4 Law 3 verified) |
| pin1_primordium_convergence_polarity_index | 0.8527 index | 0.84 | Mean cosine alignment of cellular PIN1 polarity vectors toward peripheral ring auxin maxima (PIN1 pool conservation err = 4.44e-16, flux err = 1.69e-14) |
| four_mode_divergence_angle_pearson_r | 1.0 r | 1.0 | Pearson correlation across all 4 phyllotactic Gamma regimes (r = 1.0000, p = 2.23e-05, RMSE = 0.24 deg) |
| npa_polar_transport_block_contrast | 1.0 ratio | 1.0 | Negative perturbation control T = 0.0 h^-1 (primordia count = 0 vs WT 5, WT contrast = 2.28) |
| exogenous_iaa_paste_rescue_peak_angle | 60.0 deg | 60.0 | Exogenous IAA source at phi_paste = 60.0 deg on NPA/pin1 ring (angular localization error = 0.00 deg, rescued primordia = 1, contrast = 2.73) |
| specimen_rosette_azimuthal_sector_coverage | 1.0 fraction | 1.0 | ExG + DoG segmentation of N = 410 imbricated rosette leaves around SAM center (651.5, 563.1) px |
| specimen_rosette_leaf_spacing_regularity_1_minus_cv | 0.7737 index | 0.75 | Nearest-neighbor Euclidean distance distribution across 410 segmented rosette leaves (mean = 23.4 px, CV = 0.2263) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2006-jonsson-auxin-pin1-meristem.py` after completing Step 12 closed-loop remediation in `2026-10-19-auxin-pin1-phyllotaxis-loop`.
