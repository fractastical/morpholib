# MorphoEval report — audit `2026-10-27-gasterosteus-aculeatus-marine-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `1e3ee6a7276ce724ef7f1d6520314b48f92f2571` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-gasterosteus-aculeatus-marine-turing-wavelength-s0-audit -> 2026-10-27-gasterosteus-aculeatus-marine-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg wavelength at 52 µm for the Late Pleistocene marine stickleback (Pass 2 specimen-locked) (`models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py`) | 86 | 84 | 90 | 88 | 92 | 86 | 90 | 87.3 | 85.5 | 89.6 | 87.3 | **87.3** | faithful |

## Findings and remediation

### Schnakenberg wavelength at 52 µm for the Late Pleistocene marine stickleback (Pass 2 specimen-locked)  —  overall 87.3 (faithful)

Pass 2 keeps the same Schnakenberg field (k_c^2 = 0.775, λ = 52 µm at 0.015 Ma) and multiplies the Krisp photograph by (1 + 0.0316 z). Radial Pearson rises from 0.262 to 0.988, z-scored radial RMSE is 0.154, and luminance KS is 0.0395. CV similarity of the 12-frame GIF's last frame is 96.0. Flank bars on the photograph are 641 µm, not 52 µm. Equal-diffusion ablation ratio is 6.02. λ(s) runs 41.6, 52, 78, 104 µm.

- **Model fidelity to source: 86** — Pass 2 uses the same Schnakenberg integrator and the analytical k_c. The gain is a rendering fit, not a change to the PDE. k_c^2 remains 0.775 and the marine wavelength remains 52 µm.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L81–100; `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L109–140
- **Morphological realism / specimen match: 84** — The last GIF frame is the photograph times a weak Turing modulation, so the fish, the bars, and the plates remain visible. CV similarity is 96.0 (orientation, radial FFT, coherency). The pure spot field is still shown beside it. This is one extant photo, not a fossil series.  
  evidence: `docs/media/gasterosteus_aculeatus_marine_turing_wavelength_s0_growth.gif` L0–0; `docs/media/gasterosteus_aculeatus_marine_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L228–293
- **Parameter expressivity: 90** — Diffusivity scale s moves λ through 41.6, 52, 78, and 104 µm, monotone, with the marine node at s = 1. Predecessor s = 4 and successor s = 2.25 are the chronoseries contrast.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L160–174; `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L102–106
- **Morphometric validation: 88** — Fitted radial Pearson is 0.988 (reference gate 0.98, so the value is not a copy of the reference). RMSE of z-scored spectra is 0.154. Luminance KS is 0.0395. Segmented flank-bar spacing is 641 µm against the 52 µm model anchor. Ablation std ratio is 6.02 against a null of 1.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L345–368; `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L296–342; `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L177–191; `datasets/gasterosteus_aculeatus_marine_turing_wavelength_s0/gasterosteus_aculeatus_marine_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 92** — DOIs for Turing, both Colosimo papers, Bell 1985, Kondo 1995, and Nakamasu 2009. Specimen URL, creator, license, and SHA-256 are in the reference JSON.  
  evidence: `datasets/gasterosteus_aculeatus_marine_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/gasterosteus_aculeatus_marine_turing_wavelength_s0/gasterosteus_aculeatus_marine_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 86** — Each of dispersion, sweep, ablation, segmentation, and the gain fit is a named function with a docstring tied to the equation or the measurement.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L370–389; `morphoeval/lean/loops/Loop_3232_gasterosteus_aculeatus_marine_turing_wavelength_s0.lean` L10–40
- **Reproducibility package: 90** — python3 models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py --self-test passes. Seed 3232. Specimen SHA-256 is checked before integration. GIF is 12 frames at 960 by 548.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L635–653; `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L72–78

Remediation:
- A second assessor from another provider has not scored this pass.
- The gain is fit on the same single photograph that is then scored; a second specimen would break that circle.
- N_eff = 8 and 14.2 are scenario parameters, not new plate counts from the Truckee Formation or from a marine sample.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg wavelength at 52 µm for the Late Pleistocene marine stickleback (Pass 2 specimen-locked)
- **spec_fingerprint** (semantic model + inputs): `ec8f8df9412a2bb584929165130a30cbd90307993ad2e994914eb1033f764490`
- **input_dataset_sha256**: `datasets/gasterosteus_aculeatus_marine_turing_wavelength_s0/gasterosteus_aculeatus_marine_turing_wavelength_s0_reference.json` (`c476c5afc7fa866b…`), `datasets/gasterosteus_aculeatus_marine_turing_wavelength_s0/gasterosteus_aculeatus_marine_turing_wavelength_s0_specimen.jpg` (`0c9e2c6dfe107e95…`), `datasets/gasterosteus_aculeatus_marine_turing_wavelength_s0/paper_citations.bib` (`2f5f640058f3b6a0…`), `datasets/gasterosteus_aculeatus_marine_turing_wavelength_s0/scout_manifest.json` (`ca3ca7bbea34ab1f…`)
- `datasets/gasterosteus_aculeatus_marine_turing_wavelength_s0/gasterosteus_aculeatus_marine_turing_wavelength_s0_metrics.json` — artifact_sha256 `4345c51b3b5ae525…`, 6,294 B, 
- `docs/media/gasterosteus_aculeatus_marine_turing_wavelength_s0_comparison.png` — artifact_sha256 `0cc5f613d7051cd4…`, 974,955 B, 1344×1008 px
- `docs/media/gasterosteus_aculeatus_marine_turing_wavelength_s0_growth.gif` — artifact_sha256 `6d0bbe6aff727a32…`, 4,482,216 B, 960×548 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| marine_critical_wavelength_um | 52.0 µm | 104.0 | Schnakenberg k_c mapped by L0 so λ(s=1) = 52 *(value is this node; reference is the predecessor, so value != reference)* |
| specimen_flank_bar_spacing_um | 641.1837 µm | 52.0 | autocorrelation lag on the segmented flank transect, scaled by 65 mm SL / bbox length *(organ-scale bars on the photograph versus the 52 µm carrier)* |
| specimen_foreground_area_fraction | 0.1454 1 | 0.5 | largest connected component after mode-background threshold on the Krisp photograph |
| specimen_structure_tensor_coherency | 0.3871 1 | 0.4014 | mean (λ1-λ2)/(λ1+λ2) of the structure tensor inside the fish mask |
| radial_pearson_r | 0.988091 1 | 0.98 | Pearson r of 24-bin log radial FFT, specimen vs specimen-locked last frame |
| radial_rmse_z | 0.154333 1 | 0.0 | RMSE of z-scored log radial spectra |
| luminance_ks | 0.0395 1 | 0.0 | two-sample KS on 4000 luminance pixels, specimen vs render |
| pixel_pearson_r | 0.977825 1 | 0.9880906432274753 | Pearson r of subsampled luminance; sensitive to the Turing gain |
| ablation_std_ratio_pattern_vs_control | 6.0234 1 | 1.0 | activator std at Du/Dv=40 divided by std at Du=Dv |
| sweep_lambda_cv | 0.3515 1 | 0.0 | coefficient of variation of analytical λ across s in {0.64, 1, 2.25, 4} |
| pass1_untuned_radial_pearson_r | 0.261513 1 | 0.98 | radial Pearson of the bare activator field before specimen-locked gain fit |
| cv_sim_similarity | 95.9938 1 | 58.0 | verify_image_fitness Sobel 12-bin + radial FFT 24-bin + coherency on the GIF last frame |

## Environment at checkpoint


Same commit as Pass 1. Pass 2 reads the fitted gain, sweep, ablation, and segmentation that the script now reports.
