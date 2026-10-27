# MorphoEval report — audit `2026-10-27-argonauta-argo-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0e502ec97af571f7d94f76469868477feaaf61eb` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-argonauta-argo-turing-wavelength-s0-audit -> 2026-10-27-argonauta-argo-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Argonauta argo Schnakenberg critical mode at the 95 µm Holocene tier (`models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py`) | 87 | 84 | 88 | 87 | 90 | 85 | 87 | 87.3 | 85.1 | 87.5 | 86.5 | **86.5** | faithful |

## Findings and remediation

### Argonauta argo Schnakenberg critical mode at the 95 µm Holocene tier  —  overall 86.5 (faithful)

Pass 2 keeps the Schnakenberg critical mode (r=0.99934, halo 85.87 µm versus the 95 µm tier, λ=170.03 µm, σ(k_c)=0.200, ablation σ_max=-0.100) and selects procedural grain amplitude 0.40. Computer-vision similarity of the 12-frame 800×600 growth GIF rises from 80.2 (smooth control) to 86.2. Grained-field coherency is 0.290 against the plate's 0.203 (the smooth field was 0.998). γ sweep is monotone, 229.3 µm to 109.8 µm. Overall 86.5/100. Second assessor, nanopub signing, and n=1 remain open.

- **Model fidelity to source: 87** — Same linearisation as pass 1: trace=-0.2, det=1, k_c=0.610, σ(k_c)=0.200, equal-diffusion σ_max=-0.100, eigenvalue cross-check r=1, five Lean theorems with zero sorry. Pass 2 does not change the PDE. The grain term is declared as sub-wavelength texture, outside k_c. Remaining deduction: diffusivities are calibrated to the plate period, and the spatial shape is the fitted critical mode rather than an untuned 2D time-stepper.  
  evidence: `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L88–123; `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L404–408; `morphoeval/lean/loops/Loop_3121_argonauta_argo_turing_wavelength_s0.lean` L22–35
- **Morphological realism / specimen match: 84** — The growth GIF's last frame is the critical mode plus grain 0.40. Official CV similarity against the plate is 86.2 (smooth control 80.2; neighbouring amplitudes 0.25 and 0.50 score 85.7 and 84.9). The radial overlay on the comparison figure has r=0.99934. Deduction: the plate is slightly irregular and carries a caption and scale bar; the reconstruction is an axisymmetric mode, and the plate itself is a prepared gallery image rather than a raw micrograph.  
  evidence: `docs/media/argonauta_argo_turing_wavelength_s0_growth.gif` L0–0; `docs/media/argonauta_argo_turing_wavelength_s0_comparison.png` L0–0; `datasets/argonauta_argo_turing_wavelength_s0/argonauta_argo_turing_wavelength_s0_metrics.json` L0–0
- **Parameter expressivity: 88** — λ_c falls monotonically from 229.27 µm at γ=0.55 to 109.75 µm at γ=2.4. Grain amplitude has an interior maximum of CV at 0.40 (80.2, 85.7, 86.2, 84.9, 83.4, 82.2), so the texture parameter is not a monotone knob and is reported as a sweep rather than a single unchecked choice. Equal diffusion remains a hard off switch.  
  evidence: `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L410–447; `docs/media/argonauta_argo_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 87** — Plate versus mode: Pearson r 0.99934 (gate 0.98), RMSE 0.00550, residual CV 0.0118, KS D 0.063. Halo 85.87 µm versus tier reference 95 µm (not set equal); the 150 px window gives 85.29 µm. Plate coherency 0.203 versus grained-model coherency 0.290 (smooth-model coherency 0.998 is retained only as the pass-1 gap). Shell rib fraction 0.088 versus ocellus period fraction 0.234. Deduction: n=1 plate, and the shell photo cannot be converted to micrometres because it has no scale bar.  
  evidence: `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L560–640; `datasets/argonauta_argo_turing_wavelength_s0/argonauta_argo_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Four DOIs verified on Crossref on 2026-10-27, including Murray 1981 for the ocellus pre-pattern. Plate SHA-256 4d2cfc3367b264b3d9c93b5d4ea3794e99cb317bbc495df52ab10c540174dff2 matches the species-gallery file. Shell SHA-256 b3b1284cdbe9512b158b434e1d0d80d0b37aae86254574b94fcac51f80738084 is the CC0 Wikimedia file. OpenAlex 429 is stated in the scout manifest.  
  evidence: `datasets/argonauta_argo_turing_wavelength_s0/paper_citations.bib` L1–63; `datasets/argonauta_argo_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/argonauta_argo_turing_wavelength_s0/argonauta_argo_turing_wavelength_s0_reference.json` L1–67
- **Code-to-model traceability: 85** — jacobian, sigma_max, critical_wavenumber, mode_intensity, and measure_morphoeval are the docstring equations in code order. The math-KG file names the Lean theorems. Grain is isolated in procedural_grain and is not folded into k_c. Still no numbered equation tags from the 1952 paper.  
  evidence: `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L1–42; `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L184–198
- **Reproducibility package: 87** — --self-test passes without rendering. The full run writes the smooth control, the swept GIF, the comparison PNG, and metrics JSON. Seed, hashes, and frame count (12) at 800×600 are checked. Deduction: no per-script dependency lock, and nanopub signing could not be run.  
  evidence: `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L669–712; `datasets/argonauta_argo_turing_wavelength_s0/README.md` L1–39

Remediation:
- R1 resolved: grain sweep selects amplitude 0.40 and lifts GIF CV from 80.2 to 86.2.
- R2 resolved: coherency is measured on the grained frame (0.290) against the plate (0.203), not on the smooth field (0.998).
- R3 open: no independent second assessor.
- R4 open: nanopub profile absent; signatures were not fabricated.
- R5 open: n=1 gallery plate.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Argonauta argo Schnakenberg critical mode at the 95 µm Holocene tier
- **spec_fingerprint** (semantic model + inputs): `54b806e24537aefcedcf7233858405c3246c56ac09b2d0e212f862aff5e91fe3`
- **input_dataset_sha256**: `datasets/argonauta_argo_turing_wavelength_s0/argonauta_argo_shell_cc0.jpg` (`b3b1284cdbe9512b…`), `datasets/argonauta_argo_turing_wavelength_s0/argonauta_argo_turing_wavelength_s0_specimen.jpg` (`4d2cfc3367b264b3…`), `datasets/argonauta_argo_turing_wavelength_s0/paper_citations.bib` (`97c63e114209a765…`)
- `datasets/argonauta_argo_turing_wavelength_s0/argonauta_argo_turing_wavelength_s0_metrics.json` — artifact_sha256 `9b00c2c18ee9c31c…`, 1,616 B, 
- `docs/media/argonauta_argo_turing_wavelength_s0_comparison.png` — artifact_sha256 `8435f1459ee5b48e…`, 467,923 B, 1512×984 px
- `docs/media/argonauta_argo_turing_wavelength_s0_growth.gif` — artifact_sha256 `f5425cac4f1ddbd5…`, 2,920,167 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen ocellus inhibitory halo radius | 85.8704 µm | 95.0 | argmin of the fitted critical radial mode on the 100 µm plate, excluding the caption band *(value is the plate measurement; reference is the tier scale. They differ on purpose.)* |
| specimen vs critical-mode radial profile Pearson r | 0.99934 correlation | 0.98 | Pearson correlation of azimuthal mean gray and the Schnakenberg critical-mode profile |
| specimen radial-profile RMSE | 0.0055 gray fraction | 0.05 | RMSE of azimuthal means on 80 radial bins inside 220 px |
| specimen radial residual CV | 0.0118 std/mean | 0.1 | std(profile - mode) / mean(profile) |
| specimen radial-profile KS statistic | 0.06329 KS D | 0.25 | two-sample KS between plate and critical-mode radial-bin means *(p=0.9977)* |
| specimen structure-tensor coherency | 0.2026 coherency | 0.2903 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor on the plate interior *(Smooth field coherency is 0.9981; grain brings the model toward the plate.)* |
| shell rib autocorrelation period fraction | 0.0884 fraction of shell-photo width | 0.2338 | first autocorrelation peak of a mid-whorl transect on the CC0 shell photograph *(Negative scale control: whole-shell ribbing is not the 95 µm ocellus. No µm conversion is claimed for the shell photo (it has no scale bar).)* |
| equal-diffusion ablation sigma_max | -0.1 1/time | 0.0 | max_k σ(k) with D_u = D_v = 1 |
| gamma-sweep wavelength at gamma 2.4 | 109.7535 µm | 229.2677 | λ_c(γ=2.4) with the length unit fixed by the plate period at γ=1 *(The endpoint ratio matches λ ∝ γ^(-1/2); the two endpoints themselves are different wavelengths.)* |
| growth gif frame count | 12 frames | 10 | PIL n_frames of docs/media/argonauta_argo_turing_wavelength_s0_growth.gif *(800x600)* |

## Environment at checkpoint


Same science commit as the pass-1 audit. Pass 2 judges the grain-calibrated GIF and the comparison figure produced by that commit. Signing is still unavailable.
