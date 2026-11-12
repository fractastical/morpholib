# MorphoEval report — audit `2026-10-27-chlamydomonas-reinhardtii-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at the 10.32 nm Chlamydomonas eyespot pigment stripe (Pass 1) (`models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py`) | 82 | 70 | 80 | 68 | 90 | 80 | 84 | 81.3 | 69.2 | 84.9 | 77.6 | **77.6** | mostly faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at the 10.32 nm Chlamydomonas eyespot pigment stripe (Pass 1)  —  overall 77.6 (mostly faithful)

Reference parameters a=0.07, b=0.95, Du=1, Dv=36 (not slot 195's 1.63/0.43/14/10/140). The spatial unit sets lambda(k_max)=10.32 nm. Onset lambda(k_c)=6.99 nm. Linear-regime finite differences track the analytical fastest mode at Pearson r=0.9997, RMSE=0.20 nm, across Dv in {18,24,32,36,44,58,76}. Age 0 Ma. Unfiltered SEM cell median was 1.29 um and is not accepted in this pass.

- **Model fidelity to source: 82** — Equations 1-4 (Schnakenberg kinetics, Jacobian, onset k_c, fastest mode) are implemented and the reference state is Turing-unstable. The Dv=Du control and the autocatalysis ablation are in the file; this pass does not yet credit them.  
  evidence: `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L91–153
- **Morphological realism / specimen match: 70** — The comparison figure places the Dartmouth SEM beside the modulated frame, but Pass 1 had not yet separated orientation, radial-FFT, and coherency, so the silhouette match is only partly credited.  
  evidence: `docs/media/chlamydomonas_reinhardtii_turing_wavelength_s0_comparison.png` L0–0; `docs/media/chlamydomonas_reinhardtii_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 80** — Dv from 18 to 76 lengthens the analytical stripe monotonically, and the finite-difference points follow it. The saturated-versus-linear shift is computed but not yet used as a second regime in the score.  
  evidence: `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L279–295
- **Morphometric validation: 68** — Pearson r=0.9997 and RMSE=0.20 nm are computed. The first cell median, before the 3-12 um window, was 1.29 um because debris and one merged cluster dominated. This pass does not accept a cell-size comparison until that window is the reported value.  
  evidence: `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L209–248; `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L342–386
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing, Schnakenberg, Foster and Smyth, Dieckmann, Schmidt, Kreimer, Boyd, and Kato. Specimen is the public-domain Dartmouth SEM, SHA-256 pinned.  
  evidence: `datasets/chlamydomonas_reinhardtii_turing_wavelength_s0/paper_citations.bib` L1–71; `datasets/chlamydomonas_reinhardtii_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 80** — The module docstring names equations 1-4 and the functions follow that order. Pass 1 does not yet pin the ablation and the negative control to the score.  
  evidence: `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L1–45
- **Reproducibility package: 84** — Seed 3146, --self-test checks Turing conditions, the 10.32 nm calibration, age 0 Ma, and Pearson r>0.98. Full render is deterministic. No second environment has re-run it.  
  evidence: `models/2026-delphi10000-chlamydomonas-reinhardtii-turing-wavelength-s0.py` L653–678

Remediation:
- R1: Restrict SEM diameters to isolated cell bodies between 3 and 12 um so debris and the merged cluster do not set the median (Pass 1 unfiltered median was 1.29 um).
- R2: Record orientation, radial-FFT, and coherency similarities separately, and keep the negative-control and autocatalysis-ablation sigmas in the scored measurements.
- R3: Credit the saturated-versus-linear wavelength as a second parameter regime, not only the linear Dv sweep.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at the 10.32 nm Chlamydomonas eyespot pigment stripe (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `c41079d08516e24418edacb13a3820b8f31bff46de38d0a93c0cb0ebc5f205ac`
- **input_dataset_sha256**: `datasets/chlamydomonas_reinhardtii_turing_wavelength_s0/chlamydomonas_reinhardtii_specimen.jpg` (`7f2a2df501eb7977…`)
- `docs/media/chlamydomonas_reinhardtii_turing_wavelength_s0_comparison.png` — artifact_sha256 `79b760e1b14cd10d…`, 530,265 B, 1512×1008 px
- `docs/media/chlamydomonas_reinhardtii_turing_wavelength_s0_growth.gif` — artifact_sha256 `f87c5d1f4b233129…`, 12,663,442 B, 1024×1049 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| selected_pigment_stripe_nm | 10.25913 nm | 10.32 | Radial FFT of the linear-regime FD field vs analytical argmax sigma(k), scaled so lambda(k_max ref)=10.32 nm. RMSE of the Dv sweep is reported separately. |
| dispersion_pearson_r | 0.999746 1 | 1.0 | Pearson r of numerical vs analytical fastest wavelength across the Dv sweep. Ideal correlation is 1; the finite-difference value is not identical. |
| dispersion_rmse_nm | 0.20041 nm | 0.0 | RMSE between numerical and analytical lambda(k_max) over Dv in {18,24,32,36,44,58,76}. |
| saturated_vs_linear_wavelength_nm | 10.14608 nm | 10.25913 | Nonlinear saturated wavelength vs the linear-regime wavelength at Dv=36. The shift is the second parameter regime (above onset). |
| negative_control_sigma_max | -0.089227 1/time | 0.0 | Negative control: Dv=Du removes differential diffusion. sigma_max must stay negative (no Turing stripe). |
| ablation_sigma_max | -1.0004 1/time | 0.0 | Ablation: drop autocatalysis u^2 v, leaving linear decay. sigma_max must stay negative. |
| specimen_cell_diameter_um | 4.7223 um | 10.0 | Median equivalent diameter of isolated SEM cell bodies between 3 and 12 um, using the labeled 10 um scale bar. Debris and the merged cluster are excluded. Reference 10 um is the nominal vegetative length, not this bright-cap measurement. |
| specimen_cell_diameter_cv | 0.0934 1 | 0.0 | CV of segmented cell equivalent diameters on the Dartmouth SEM. Zero would mean identical cells. |
| specimen_foreground_fraction | 0.4162 1 | 0.425 | Foreground fraction of the modulated simulation frame vs the real SEM segmentation. Self-referential equality is not expected once the activator rescales edges. |
| specimen_structure_tensor_coherency | 0.4071 1 | 0.4052 | Structure-tensor coherency on the simulation mask vs the specimen mask. |
| specimen_contour_circularity | 0.0243 1 | 0.0273 | 4*pi*area/perimeter^2 of the segmented cell union, simulation vs specimen. |
| specimen_intensity_ks | 0.0792 1 | 0.0 | Two-sample KS statistic between SEM and modulated cell-interior intensities (p=2.4e-11). Identical images would score 0. |
| onset_k_c_per_nm | 0.899449 1/nm | 0.608836 | Onset wavenumber k_c compared with the fastest mode k_max at Dv/Du=36. They differ above onset; lambda(k_max) is the 10.32 nm stripe. |

## Environment at checkpoint


Pass 1 working tree on top of origin/main. No nanopub profile; publisher_orcid is the repository URL, not a personal ORCID. Nanopubs are unsigned.
