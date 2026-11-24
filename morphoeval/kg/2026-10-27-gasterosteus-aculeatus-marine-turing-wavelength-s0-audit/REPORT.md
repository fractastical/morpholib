# MorphoEval report — audit `2026-10-27-gasterosteus-aculeatus-marine-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `1e3ee6a7276ce724ef7f1d6520314b48f92f2571` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg wavelength at 52 µm for the Late Pleistocene marine stickleback (Pass 1 bare field) (`models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py`) | 84 | 52 | 74 | 64 | 90 | 82 | 84 | 80.7 | 56.5 | 85.5 | 72.4 | **72.4** | mostly faithful |

## Findings and remediation

### Schnakenberg wavelength at 52 µm for the Late Pleistocene marine stickleback (Pass 1 bare field)  —  overall 72.4 (mostly faithful)

Pass 1 reports the bare Schnakenberg activator (k_c^2 = 0.775, λ = 52 µm, age 0.015 Ma). Radial Pearson against the Holger Krisp CC BY 4.0 photograph is 0.262 and the z-scored radial RMSE is 1.215. The photograph's flank bars measure about 641 µm, which is not the 52 µm carrier.

- **Model fidelity to source: 84** — The script integrates Schnakenberg kinetics with implicit diffusion and the analytical k_c. Parameters are chosen to hit 52 µm; they are not a published stickleback rate table.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L81–100; `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L109–140
- **Morphological realism / specimen match: 52** — The growth frames show a spot field, and the comparison figure places it beside the fish, but the Pass-1 bare field is not the fish. Computer-vision similarity of that bare field is 80.5, while radial Pearson is only 0.262.  
  evidence: `docs/media/gasterosteus_aculeatus_marine_turing_wavelength_s0_comparison.png` L0–0; `docs/media/gasterosteus_aculeatus_marine_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 74** — λ(s) = 52 * sqrt(s) is monotone on s in {0.64, 1, 2.25, 4}, but Pass 1 did not yet use that sweep as the reported specimen fit.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L160–174
- **Morphometric validation: 64** — Flank-bar spacing on the photograph is 641 µm against the 52 µm anchor (values differ). The Pass-1 radial Pearson of 0.262 does not clear 0.98, so the goodness-of-fit is not yet a pass.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L551–601; `datasets/gasterosteus_aculeatus_marine_turing_wavelength_s0/gasterosteus_aculeatus_marine_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Turing, Colosimo 2004/2005, Bell 1985, Kondo 1995, and Nakamasu 2009 are cited with DOIs. The specimen is Holger Krisp, CC BY 4.0, SHA-256 pinned.  
  evidence: `datasets/gasterosteus_aculeatus_marine_turing_wavelength_s0/paper_citations.bib` L1–21
- **Code-to-model traceability: 82** — dispersion() and integrate() are commented against the Turing formulae. The plate-count map is labeled as a reconstruction.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L1–28
- **Reproducibility package: 84** — Seed 3232, --self-test, and pinned specimen bytes. Pass 1 self-test correctly rejects the bare-field Pearson.  
  evidence: `models/2026-delphi10000-gasterosteus-aculeatus-marine-turing-wavelength-s0.py` L635–653

Remediation:
- Goodness-of-fit: bare-field radial Pearson is 0.262. Fit a specimen-locked gain and recompute Pearson, RMSE, and KS.
- Real specimen CV segmentation: report foreground fraction, structure-tensor coherency, and flank-bar spacing, and keep bar spacing unequal to the 52 µm reference.
- Parameter-regime sweep: require λ(s) to increase monotonically for s in {0.64, 1, 2.25, 4}.
- Negative control: equal diffusion (Du = Dv) must stay outside the Turing band and the pattern/control activator-std ratio must exceed 3.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg wavelength at 52 µm for the Late Pleistocene marine stickleback (Pass 1 bare field)
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


Pass 1 scores the bare activator field. Pass 2 locks that field to the specimen.
