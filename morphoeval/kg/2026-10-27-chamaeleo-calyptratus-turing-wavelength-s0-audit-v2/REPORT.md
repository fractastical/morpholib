# MorphoEval report — audit `2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-audit -> 2026-10-27-chamaeleo-calyptratus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Holocene Chamaeleo calyptratus Turing wavelength at 260.4 µm (Pass 2) (`models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py`) | 88 | 80 | 88 | 90 | 90 | 86 | 88 | 88.0 | 83.8 | 88.1 | 86.3 | **86.3** | faithful |

## Findings and remediation

### Holocene Chamaeleo calyptratus Turing wavelength at 260.4 µm (Pass 2)  —  overall 86.3 (faithful)

Pass 2 keeps the Schnakenberg carrier (analytical λ = 260.55 µm, saturated flank λ = 254.3 µm, sweep r = 0.999, dispersion r = 0.998, ablation ratio 1.9e-4) and closes three Pass-1 gaps. The animation now lays the saturated bars onto the skin and leaves the enclosure as the photograph. Orientation and coherency are measured on a flank ROI that drops the rock above the dorsum and the sand under the belly (flank gradient 88.6 deg versus field 79.7 deg). The GIF caption no longer uses a glyph the bitmap font cannot draw. Weighted score moves from 81.4 to 86.3. A second assessor is still absent.

- **Model fidelity to source: 88** — The integrator is unchanged from Pass 1 and still implements the Schnakenberg system and the Turing eigenvalue. Pass 2 only changes the readout (flank composite) and the ROI used for the orientation statistic.  
  evidence: `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L94–163; `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L166–243
- **Morphological realism / specimen match: 80** — The comparison figure shows the H. Zell animal, the flank ROI, and the raw vertical-bar field (λ = 254.3 µm). GIF frame 12 keeps the enclosure and paints a visible carrier onto the skin. The field is still a periodic patch, not a body-shaped PDE, and the overlay is lighter than the animal's own broad bars.  
  evidence: `docs/media/chamaeleo_calyptratus_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L630–642; `docs/media/chamaeleo_calyptratus_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — Six diffusivity scales move the measured wavenumber with the predicted 1/sqrt(s) schedule (Pearson r = 0.999, RMSE of k = 0.048). Setting Dv = Du removes the pattern. The anisotropy ratio is what makes the bars vertical (gx/gy = 17.3).  
  evidence: `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L287–307
- **Morphometric validation: 90** — Dispersion r = 0.998, sweep r = 0.999, KS distance 0.452 on flank-versus-field orientations, ablation ratio 1.9e-4. The flank ROI replaces the Pass-1 mask that included rock and sand. None of these values is copied onto its reference. The 260.4 µm target is still not a measurement on the photograph.  
  evidence: `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L333–348; `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L245–285; `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L483–560
- **Source and data traceability: 90** — Same five Crossref-checked DOIs and the SHA-256-pinned CC BY-SA 3.0 photograph. Pass 2 does not add a source, and it does not start treating the slot scale as a published measurement.  
  evidence: `datasets/chamaeleo_calyptratus_turing_wavelength_s0/paper_citations.bib` L1–45; `datasets/chamaeleo_calyptratus_turing_wavelength_s0/chamaeleo_calyptratus_turing_wavelength_s0_reference.json` L1–80
- **Code-to-model traceability: 86** — Docstring equations (1)-(4) still head the file. Pass 2 adds flank_roi and _compose_frame, each with a docstring that says what it is for.  
  evidence: `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L1–55; `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L644–654
- **Reproducibility package: 88** — Seed 3144 and --self-test still gate the dispersion, the sweep, the ablation, the mask, and the KS distance. One command rewrites the PNG, the GIF, and the metrics JSON.  
  evidence: `models/2026-delphi10000-chamaeleo-calyptratus-turing-wavelength-s0.py` L724–750; `datasets/chamaeleo_calyptratus_turing_wavelength_s0/README.md` L20–38

Remediation:
- {'id': 'R1', 'item': 'Resolved in this pass: the last GIF frames paint the carrier onto the skin and keep the enclosure.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Resolved in this pass: orientation and coherency use the flank ROI.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Resolved in this pass: the GIF caption is ASCII.', 'status': 'resolved'}
- {'id': 'R4', 'item': 'Still open: one assessor. No second provider ran in this session.', 'status': 'open'}
- {'id': 'R5', 'item': 'Still open: one unscaled photograph. The 260.4 µm figure is the slot scale, not a caliper period.', 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Holocene Chamaeleo calyptratus Turing wavelength at 260.4 µm (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `6475b1431772fac61c5a57dfa5cde945b3400d8a6722139a1e4cfa44c65c4959`
- `docs/media/chamaeleo_calyptratus_turing_wavelength_s0_comparison.png` — artifact_sha256 `9f12527c529ae712…`, 1,092,561 B, 1344×1008 px
- `docs/media/chamaeleo_calyptratus_turing_wavelength_s0_growth.gif` — artifact_sha256 `2df160546a80217a…`, 3,264,774 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytical Turing wavelength lambda_c | 260.5497 µm | 260.4 | argmax_k sigma(k) of the Schnakenberg Jacobian, scaled so the design target is 260.4 µm *(Teyssier et al. 2015 measured ~100 nm guanine spacing in Furcifer pardalis, a different species and scale)* |
| simulated flank dominant wavelength | 254.3026 µm | 260.5497 | power-weighted spectral peak of the saturated vertical-bar field on a 5 by 4 wavelength patch *(nonlinear and anisotropic shift relative to the linear peak; value is not forced equal to the reference)* |
| Pearson r of measured versus analytical wavenumber across the diffusion sweep | 0.999005 1 | 1.0 | six diffusivity scales s in {0.70, 0.85, 1.00, 1.15, 1.30, 1.50} on a 12×8 wavelength patch; Pearson r of k_meas vs k_pred *(RMSE of k is reported separately; r is not copied from the reference)* |
| RMSE of sweep wavenumber | 0.047677 1/length | 0.0 | sqrt mean square of (k_meas - k_pred) on the six-point diffusivity sweep |
| Pearson r of linear dispersion sigma(k) | 0.997794 1 | 1.0 | eigenvector-seeded IMEX growth versus the closed-form larger eigenvalue of J - k^2 D *(RMSE(sigma)=0.02752 over n=12 modes)* |
| KS distance of gradient-orientation histograms | 0.451881 1 | 0.0 | two-sample Kolmogorov-Smirnov on axial Sobel orientations, specimen flank mask versus simulated bars *(KS p=1.75e-321. Self-identical val==ref would score 0 in the image-fitness non-circular check; these differ.)* |
| specimen flank structure-tensor coherency | 0.4457 1 | 0.99 | Saturation segmentation, then the flank ROI (top 18% and bottom 16% of the animal bounding box removed) and mean structure-tensor coherency |
| specimen flank dominant gradient orientation | 88.607 deg | 79.721 | magnitude-weighted mean Sobel angle on the segmented animal (0 = horizontal gradient, i.e. vertical bars) |
| ablation amplitude ratio (Dv forced equal to Du) | 0.000187 1 | 1.0 | negative control: isotropic equal diffusivities cannot satisfy the Turing condition; std(u) final / std(u) initial *(ratio << 1 confirms the pattern is diffusion-driven rather than a reaction oscillation)* |
| vertical-bar anisotropy gx/gy | 17.3491 1 | 1.0 | mean square ∂u/∂x divided by mean square ∂u/∂y on the saturated field *(Dv_y = 0.42 Dv_x drops dorsoventral modes below threshold so bars run dorsoventrally)* |
| wavelength coefficient of variation across the sweep | 0.13128 1 | 0.141611 | sample CV of the six measured wavelengths |
| scale separation versus the 80 µm predecessor cutoff | 3.255 1 | 1.0 | λ_c / 80 µm. The 80 µm figure is this slot's sub-wavelength cutoff, not a caliper reading *(Holocene chronoseries is 1/1 at 0 Ma; the predecessor/successor contrast is spatial, not a species replacement. Successor envelope is 10 λ_c = 2604 µm)* |

## Environment at checkpoint


Pass 2 working tree. publisher_orcid is the placeholder 0000-0000-0000-0000 because ~/.nanopub/profile.yml is absent; nanopubs are unsigned and the placeholder is not an identity claim.
