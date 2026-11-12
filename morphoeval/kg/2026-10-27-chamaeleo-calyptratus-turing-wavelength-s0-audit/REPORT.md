# MorphoEval report — audit `2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Holocene Chamaeleo calyptratus Turing wavelength at 260.4 µm (Pass 1) (`models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py`) | 86 | 70 | 84 | 86 | 88 | 82 | 84 | 85.3 | 76.0 | 84.8 | 81.4 | **81.4** | faithful |

## Findings and remediation

### Holocene Chamaeleo calyptratus Turing wavelength at 260.4 µm (Pass 1)  —  overall 81.4 (faithful)

Schnakenberg kinetics on a periodic flank patch, diffusivities solved so the most-unstable wavelength is 260.4 µm, with Dv_y/Dv_x = 0.42 so the unstable modes are anteroposterior. Self-test: analytical λ = 260.55 µm, saturated field λ = 254.3 µm, stripe anisotropy 17.3, sweep Pearson r = 0.999, dispersion Pearson r = 0.998, equal-diffusion ablation ratio 1.9e-4. The GIF opens on the noisy field and ends on the H. Zell photograph; the simulated bars are not visible in the last frame.

- **Model fidelity to source: 86** — Equations (1)-(4) are the Schnakenberg system and the Turing dispersion, integrated by spectral IMEX. The dorsoventral diffusion ratio is an addition that the cited papers do not contain.  
  evidence: `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L94–163; `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L166–243
- **Morphological realism / specimen match: 70** — The comparison figure places the H. Zell animal next to a green-gold vertical-bar field. The bars are the right family of pattern, but they are a periodic patch, not the animal's outline. GIF frame 12 is the photograph; the simulated bars have dropped out.  
  evidence: `docs/media/chamaeleo_calyptratus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/chamaeleo_calyptratus_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 84** — Scaling both diffusivities by s in {0.70, 0.85, 1.00, 1.15, 1.30, 1.50} moves the measured wavenumber with the predicted 1/sqrt(s) schedule (r = 0.999). The anisotropy switch is exercised by the ablation but not by a second rendered morph.  
  evidence: `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L287–307
- **Morphometric validation: 86** — Dispersion Pearson r = 0.998 (RMSE 0.028), sweep Pearson r = 0.999 (RMSE of k = 0.048), KS distance 0.45 between specimen and field orientation histograms, ablation amplitude ratio 1.9e-4. The 260.4 µm target is not a measurement on this photograph.  
  evidence: `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L245–285; `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L462–530; `datasets/chamaeleo_calyptratus_turing_wavelength_s0/chamaeleo_calyptratus_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 88** — Crossref-checked DOIs for Turing, Schnakenberg, Murray, Ligon & McGraw, and Teyssier. Specimen is H. Zell, CC BY-SA 3.0, SHA-256 pinned. The slot scale is identified as assigned rather than measured.  
  evidence: `datasets/chamaeleo_calyptratus_turing_wavelength_s0/paper_citations.bib` L1–48; `datasets/chamaeleo_calyptratus_turing_wavelength_s0/chamaeleo_calyptratus_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 82** — The module docstring names equations (1)-(4) and the functions follow that order. There is no line-by-line map onto Murray's section numbers.  
  evidence: `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L1–55
- **Reproducibility package: 84** — Seed 3144, --self-test covering the dispersion, the sweep, the ablation and the mask, and a documented invocation. Dependencies are the repo scientific stack. One assessor.  
  evidence: `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L696–722; `datasets/chamaeleo_calyptratus_turing_wavelength_s0/README.md` L1–40

Remediation:
- {'id': 'R1', 'item': 'GIF frame 12 is the unmodified photograph, so the selected bars are not visible at the end of the animation. Composite the saturated field onto the flank and keep the enclosure.'}
- {'id': 'R2', 'item': 'Saturation mask still includes rock above the dorsum and a sand sliver. Restrict the orientation metric to a flank ROI.'}
- {'id': 'R3', 'item': 'The GIF caption uses a lambda glyph the default bitmap font cannot draw.'}
- {'id': 'R4', 'item': 'Single assessor. Left open; a second provider was not available in this run.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Holocene Chamaeleo calyptratus Turing wavelength at 260.4 µm (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `6475b1431772fac61c5a57dfa5cde945b3400d8a6722139a1e4cfa44c65c4959`
- `datasets/chamaeleo_calyptratus_turing_wavelength_s0/chamaeleo_calyptratus_turing_wavelength_s0_metrics.json` — artifact_sha256 `c9572e151ea2e2b9…`, 4,008 B, 
- `docs/media/chamaeleo_calyptratus_turing_wavelength_s0_comparison.png` — artifact_sha256 `44f97b41f6edc702…`, 1,124,181 B, 1344×1008 px
- `docs/media/chamaeleo_calyptratus_turing_wavelength_s0_growth.gif` — artifact_sha256 `8764f1e54ba23cf2…`, 3,446,754 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytical Turing wavelength lambda_c | 260.5497 µm | 260.4 | argmax_k sigma(k) of the Schnakenberg Jacobian, scaled so the design target is 260.4 µm *(Teyssier et al. 2015 measured ~100 nm guanine spacing in Furcifer pardalis, a different species and scale)* |
| simulated flank dominant wavelength | 254.3026 µm | 260.5497 | power-weighted spectral peak of the saturated vertical-bar field on a 5 by 4 wavelength patch *(nonlinear and anisotropic shift relative to the linear peak; value is not forced equal to the reference)* |
| Pearson r of measured versus analytical wavenumber across the diffusion sweep | 0.999005 1 | 1.0 | six diffusivity scales s in {0.70, 0.85, 1.00, 1.15, 1.30, 1.50} on a 12×8 wavelength patch; Pearson r of k_meas vs k_pred *(RMSE of k is reported separately; r is not copied from the reference)* |
| RMSE of sweep wavenumber | 0.047677 1/length | 0.0 | sqrt mean square of (k_meas - k_pred) on the six-point diffusivity sweep |
| Pearson r of linear dispersion sigma(k) | 0.997794 1 | 1.0 | eigenvector-seeded IMEX growth versus the closed-form larger eigenvalue of J - k^2 D *(RMSE(sigma)=0.02752 over n=12 modes)* |
| KS distance of gradient-orientation histograms | 0.453336 1 | 0.0 | two-sample Kolmogorov-Smirnov on axial Sobel orientations, specimen flank mask versus simulated bars *(KS p=1.79e-321. Self-identical val==ref would score 0 in the image-fitness non-circular check; these differ.)* |
| specimen flank structure-tensor coherency | 0.4683 1 | 0.99 | Saturation segmentation of the H. Zell CC BY-SA photograph (pale sand and dark cave rejected), then mean structure-tensor coherency on the animal mask |
| specimen flank dominant gradient orientation | 87.073 deg | 79.721 | magnitude-weighted mean Sobel angle on the segmented animal (0 = horizontal gradient, i.e. vertical bars) |
| ablation amplitude ratio (Dv forced equal to Du) | 0.000187 1 | 1.0 | negative control: isotropic equal diffusivities cannot satisfy the Turing condition; std(u) final / std(u) initial *(ratio << 1 confirms the pattern is diffusion-driven rather than a reaction oscillation)* |
| vertical-bar anisotropy gx/gy | 17.3491 1 | 1.0 | mean square ∂u/∂x divided by mean square ∂u/∂y on the saturated field *(Dv_y = 0.42 Dv_x drops dorsoventral modes below threshold so bars run dorsoventrally)* |
| wavelength coefficient of variation across the sweep | 0.13128 1 | 0.141611 | sample CV of the six measured wavelengths |
| scale separation versus the 80 µm predecessor cutoff | 3.255 1 | 1.0 | λ_c / 80 µm. The 80 µm figure is this slot's sub-wavelength cutoff, not a caliper reading *(Holocene chronoseries is 1/1 at 0 Ma; the predecessor/successor contrast is spatial, not a species replacement. Successor envelope is 10 λ_c = 2604 µm)* |

## Environment at checkpoint


Pass 1 working tree. New slot files are not in this commit. publisher_orcid is the placeholder 0000-0000-0000-0000 because ~/.nanopub/profile.yml is absent; the nanopubs are unsigned and the placeholder is not an identity claim.
