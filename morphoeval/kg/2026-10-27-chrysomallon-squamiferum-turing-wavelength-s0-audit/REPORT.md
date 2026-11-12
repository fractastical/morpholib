# MorphoEval report — audit `2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `6f01b4c5e4c9fa53e8bf2c8825be8b7ccb7fba3c` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 2.0 nm on Chrysomallon squamiferum (Pass 1) (`models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py`) | 74 | 76 | 46 | 58 | 90 | 82 | 84 | 64.7 | 69.2 | 85.5 | 72.7 | **72.7** | mostly faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 2.0 nm on Chrysomallon squamiferum (Pass 1)  —  overall 72.7 (mostly faithful)

Pass 1 integrates one Schnakenberg field (a=0.062, b=1.084, D_v/D_u=36, seed 3147) with diffusivities chosen so the onset wavenumber is 2π/2.0 nm. Closed-form σ(k) matches numpy eigenvalues at r=1.0. The PDE radial peak is 2.52 nm, not 2.0 nm. The paratype photograph is the anatomical canvas. No parameter sweep, no equal-diffusion ablation, and no KS comparison against the specimen were run.

- **Model fidelity to source: 74** — Schnakenberg kinetics and the Turing onset formula are implemented and the integer identity matches Loop_3147. Above onset, that formula is not the fastest mode, and Pass 1 still reports λ=2.0 nm from the formula while the PDE peak is 2.52 nm.  
  evidence: `models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py` L105–132; `models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py` L350–407
- **Morphological realism / specimen match: 76** — The frame is the real CC BY 4.0 paratype (MNHN-IM-2000-30072, Marie Hennion), so the animal is the animal. Pass 1 modulation of the sclerite mask is weak, and there is no 2 nm micrograph to match.  
  evidence: `datasets/chrysomallon_squamiferum_turing_wavelength_s0/chrysomallon_squamiferum_specimen.jpg` L0–0; `docs/media/chrysomallon_squamiferum_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 46** — Only λ=2.0 nm is integrated. γ, a, and b are constants. Nothing in the Pass 1 path shows that wavelength moves when the diffusivity scale moves.  
  evidence: `models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py` L350–370
- **Morphometric validation: 58** — Eigenvalue Pearson r=1.0 compares two evaluations of the same Jacobian. The PDE wavelength 2.520 nm versus the 2.0 nm assignment is a 26% miss. Foreground fraction 0.394 versus prior 0.55 is a real mask measurement, not a fit.  
  evidence: `datasets/chrysomallon_squamiferum_turing_wavelength_s0/pass1_metrics.json` L0–0
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing, Schnakenberg, Warén, Yao, Chen, and Sun; Skinner 1964 cited by journal URL; specimen SHA-256, catalog number, and CC BY 4.0 attribution are pinned.  
  evidence: `datasets/chrysomallon_squamiferum_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/chrysomallon_squamiferum_turing_wavelength_s0/chrysomallon_squamiferum_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 82** — Dispersion and Jacobian are commented against Turing 1952 and Schnakenberg 1979. The Lean integer identity is the same thousandths (f_u=892, g_v=-1313, ratio 36).  
  evidence: `models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py` L1–40; `morphoeval/lean/loops/Loop_3147_chrysomallon_squamiferum_turing_wavelength_s0.lean` L40–42
- **Reproducibility package: 84** — Deterministic seed 3147, pinned specimen hash, and a passing --self-test. Pass 1 self-test accepts the tautological eigenvalue correlation and does not check the PDE wavelength.  
  evidence: `models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py` L66–81; `datasets/chrysomallon_squamiferum_turing_wavelength_s0/README.md` L24–36

Remediation:
- {'id': 'R1', 'item': 'Sweep target wavelength (diffusivity scale) and report Pearson r and RMSE of PDE radial-FFT peaks against the targets.'}
- {'id': 'R2', 'item': 'Run an equal-diffusion ablation (D_u=D_v) and report the amplitude ratio against the patterned field.'}
- {'id': 'R3', 'item': 'Segment the paratype and compare it to the render with structure-tensor coherency, circularity, and a KS test.'}
- {'id': 'R4', 'item': 'Retune diffusivities so the maximum of σ(k), not the onset formula, sits on 2.0 nm, and require the PDE peak within 8%.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 2.0 nm on Chrysomallon squamiferum (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `e0865ee451716d4a3fa5bd6b6983a8391e5da7f8dc195fece7fe187c823b6b31`
- `datasets/chrysomallon_squamiferum_turing_wavelength_s0/chrysomallon_squamiferum_specimen.jpg` — artifact_sha256 `0312a80d873d68d2…`, 235,221 B, 1600×1111 px
- `datasets/chrysomallon_squamiferum_turing_wavelength_s0/pass1_metrics.json` — artifact_sha256 `1c8aa31f5a81fb73…`, 2,356 B, 
- `docs/media/chrysomallon_squamiferum_turing_wavelength_s0_comparison.png` — artifact_sha256 `f78b2d8abd16f668…`, 505,249 B, 1248×960 px

| measurement | value | reference | method |
|---|---:|---|---|
| specimen_foreground_fraction | 0.3936 1 | 0.55 | gray<0.78 mask on paratype MNHN-IM-2000-30072 *(non-circular: measured area fraction differs from the prior)* |
| specimen_contour_circularity | 0.1183 1 | 1.0 | 4*pi*area/perimeter^2 on the sclerite-and-shell mask *(non-circular specimen shape metric)* |
| specimen_structure_tensor_coherency | 0.3097 1 | 0.3455 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor *(KS intensity statistic 0.2475 (p=0); orientation KS 0.0265)* |
| pde_versus_target_wavelength_pearson_r | 0.996439 1 | 0.98 | Pearson r of 6 target wavelengths vs PDE radial-FFT peaks after fastest-mode calibration *(RMSE 0.1823 nm; residual CV 0.0365; sweep [1.35, 1.7, 2.0, 2.35, 2.8, 3.3] -> [1.229, 1.559, 1.935, 2.211, 2.62, 2.97])* |
| calibrated_wavelength_nm | 1.9347 nm | 2.0 | dominant radial-FFT wavelength of the 2.0 nm Schnakenberg run *(equal-diffusion ablation std ratio 6.546e-12)* |
| equal_diffusion_ablation_ratio | 0.0 1 | 1.0 | std(u | D_u=D_v) / std(u | D_v=36 D_u) at the calibrated 2.0 nm scale *(negative control; trace f_u+g_v < 0 so equal diffusion is linearly stable)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 baseline before fastest-mode calibration, sweep, ablation, and specimen KS.
