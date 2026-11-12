# MorphoEval report — audit `2026-10-27-argonauta-argo-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0e502ec97af571f7d94f76469868477feaaf61eb` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Argonauta argo Schnakenberg critical mode at the 95 µm Holocene tier (`models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py`) | 84 | 74 | 86 | 80 | 88 | 82 | 84 | 84.7 | 76.2 | 84.8 | 81.3 | **81.3** | faithful |

## Findings and remediation

### Argonauta argo Schnakenberg critical mode at the 95 µm Holocene tier  —  overall 81.3 (faithful)

Pass 1 judges the smooth critical-mode animation (12 frames, 800×600). Schnakenberg parameters a=0.10, b=0.90, D_u=1, D_v=18 give trace=-0.2, det=1, k_c=0.610, σ(k_c)=0.200, and equal-diffusion σ_max=-0.100. The plate's radial profile matches the mode at Pearson r=0.99934 (RMSE 0.00550). Measured inhibitory halo is 85.87 µm against the 95 µm tier label. Smooth-control image CV against the plate is 80.2. Overall 81.3/100. Grain amplitude, grained coherency, and a second assessor are still open.

- **Model fidelity to source: 84** — The Jacobian, quadratic dispersion, critical wavenumber, and equal-diffusion ablation match the Schnakenberg linearisation, and the matrix-eigenvalue cross-check has Pearson r = 1. Five Lean theorems encode this slot's scaled integers (trace -20, det 10000, Turing margin 1340² > 720000). Deduction: D_u and D_v are chosen so k_c sits on the measured period; they are not published Argonauta diffusivities, and the radial profile is a fitted critical-mode shape rather than a long free 2D integration.  
  evidence: `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L88–123; `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L261–296; `morphoeval/lean/loops/Loop_3121_argonauta_argo_turing_wavelength_s0.lean` L10–43
- **Morphological realism / specimen match: 74** — The smooth control is recognisably one brown ocellus, and the radial means overlie the plate (r=0.99934). Computer-vision similarity of that GIF to the plate is 80.2. The field is perfectly axisymmetric and has none of the plate's grain or caption, so it is not a close pixel-level match. This score is for the smooth GIF only.  
  evidence: `datasets/argonauta_argo_turing_wavelength_s0/argonauta_argo_turing_wavelength_s0_smooth_control.gif` L0–0; `datasets/argonauta_argo_turing_wavelength_s0/argonauta_argo_turing_wavelength_s0_metrics.json` L0–0
- **Parameter expressivity: 86** — Raising γ from 0.55 to 2.4 shortens λ_c from 229.3 µm to 109.8 µm, monotone and consistent with λ ∝ γ^(-1/2). Equal diffusion (D_u = D_v) drops σ_max from +0.200 to -0.100. The grain-amplitude sweep is computed in the same run but is not what this pass judges.  
  evidence: `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L261–296; `datasets/argonauta_argo_turing_wavelength_s0/argonauta_argo_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 80** — Halo 85.87 µm versus tier 95 µm, Pearson r 0.99934, RMSE 0.00550, residual CV 0.0118, KS D 0.063, and a second radial window (halo 85.29 µm, r 0.99957). Shell-photo rib fraction 0.088 versus ocellus period fraction 0.234. Deduction: structure-tensor coherency of the smooth field is 0.998 against the plate's 0.203, so the segmentation comparison is not yet on the delivered texture.  
  evidence: `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L298–351; `datasets/argonauta_argo_turing_wavelength_s0/argonauta_argo_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 88** — Turing 1952, Gierer & Meinhardt 1972, and Finn & Norman 2010 were confirmed on Crossref. The plate and the CC0 shell (Di Turner, iNaturalist 15508745) are SHA-256 pinned and checked at runtime. OpenAlex was rate-limited (HTTP 429); that gap is recorded in the scout manifest rather than filled with invented hits.  
  evidence: `datasets/argonauta_argo_turing_wavelength_s0/paper_citations.bib` L1–63; `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L74–86
- **Code-to-model traceability: 82** — The module docstring states the PDE, the steady state, k_c, and the claim boundary, and each of those is a named function. Lean declarations name the same scaled integers. Deduction: there is no line-by-line map onto numbered equations of Turing 1952.  
  evidence: `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L1–42; `morphoeval/math_kg_loops/delphi10000-argonauta-argo-turing-wavelength-s0.yaml` L1–54
- **Reproducibility package: 84** — Seed 3121, --self-test, and pinned input hashes. The self-test passed (r=0.99934, ablation σ_max=-0.100, age 0 Ma, size 9.48e-05 m). Deduction: dependency versions are the environment's, not a per-script lock, and the pass-1 GIF is the smooth control rather than the grain-calibrated deliverable.  
  evidence: `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L669–700; `models/2026-delphi10000-argonauta-argo-turing-wavelength-s0.py` L56–67

Remediation:
- R1: Sweep procedural grain amplitude and keep the value that raises CV similarity of the growth GIF; the smooth control scores 80.2.
- R2: Recompute structure-tensor coherency on the grained frame. The smooth field's coherency (0.998) is not comparable to the plate (0.203).
- R3: Independent second assessor. This round is builder-scored.
- R4: Nanopublication signing. ~/.nanopub/profile.yml is not present; do not invent a signature.
- R5: The pattern target is one gallery plate (n=1), not a population of Argonauta ocelli.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Argonauta argo Schnakenberg critical mode at the 95 µm Holocene tier
- **spec_fingerprint** (semantic model + inputs): `54b806e24537aefcedcf7233858405c3246c56ac09b2d0e212f862aff5e91fe3`
- **input_dataset_sha256**: `datasets/argonauta_argo_turing_wavelength_s0/argonauta_argo_shell_cc0.jpg` (`b3b1284cdbe9512b…`), `datasets/argonauta_argo_turing_wavelength_s0/argonauta_argo_turing_wavelength_s0_specimen.jpg` (`4d2cfc3367b264b3…`), `datasets/argonauta_argo_turing_wavelength_s0/paper_citations.bib` (`97c63e114209a765…`)
- `datasets/argonauta_argo_turing_wavelength_s0/argonauta_argo_turing_wavelength_s0_metrics.json` — artifact_sha256 `9b00c2c18ee9c31c…`, 1,616 B, 
- `datasets/argonauta_argo_turing_wavelength_s0/argonauta_argo_turing_wavelength_s0_smooth_control.gif` — artifact_sha256 `23ce19963dbc42f1…`, 658,416 B, 800×600 px, 12 frames

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


Slot 3121 science commit. Python 3 with numpy, scipy, Pillow, matplotlib. OpenAlex returned HTTP 429; Crossref DOIs were checked on 2026-10-27. ~/.nanopub/profile.yml is absent, so this audit is not signed.
