# MorphoEval report — audit `2026-10-27-astyanax-mexicanus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Turing wavenumber at 666 nm in the Astyanax cave morph (`models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py`) | 82 | 74 | 58 | 76 | 88 | 80 | 84 | 74.0 | 74.8 | 84.3 | 77.4 | **77.4** | mostly faithful |

## Findings and remediation

### Turing wavenumber at 666 nm in the Astyanax cave morph  —  overall 77.4 (mostly faithful)

Pass 1. Schnakenberg reaction-diffusion on a periodic 72 by 112 patch, with Du:Dv = 2:20 and gamma = 1, calibrated so the fastest linear mode has wavelength 6.656e-7 m (666 nm nominal, Holocene cave morph, 0 Ma). Forward-Euler integration of the linear Fourier-mode ODE matches the closed-form growth rate at Pearson r = 0.99997 (RMSE 0.00455, 11 modes). The nonlinear patch selects 660.7 nm. The cave-morph voucher (CC BY-SA 4.0, Seb Morič) is masked and used as a visualization carrier; the photograph cannot resolve 666 nm. sigma(k=0) = -0.1, so the homogeneous state is stable. No diffusion-ratio sweep, no Du = Dv ablation in the reported metrics, and no flank KS comparison yet.

- **Model fidelity to source: 82** — The script integrates Schnakenberg kinetics with a Fourier pseudo-spectral step and compares growth rates to the eigenvalue of gamma*J - k^2 D. Parameters are chosen so the fastest mode sits on the card wavelength; they are not fitted to a cavefish time series. Eye and neuromast pathways are excluded from the state vector, which matches the claim boundary.  
  evidence: `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L85–96; `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L164–188
- **Morphological realism / specimen match: 74** — The GIF is a 12-frame, 960x522 animation on the real cave-morph photograph. At overlay beta 0.42 the patch reads as flank mottling, and the fish silhouette is the voucher. The match is carried by that silhouette. A 666 nm mode is below the optical resolution of the photograph, so the panel cannot be a literal specimen of the wavelength.  
  evidence: `docs/media/astyanax_mexicanus_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L258–310
- **Parameter expressivity: 58** — Du, Dv, gamma and beta are single values. The fastest-mode calibration shows that the spatial unit controls the reported wavelength, but pass 1 does not sweep the diffusion ratio or show that lambda moves monotonically. Expressivity is stated, not exercised.  
  evidence: `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L49–73
- **Morphometric validation: 76** — Eleven Fourier modes give Pearson r = 0.99997 and RMSE = 0.00455 against the analytical growth rate. The nonlinear radial peak is 660.7 nm against the 665.6 nm linear calibration (about 0.7 percent low). sigma(0) = -0.1. The voucher melanin-dark fraction is 0.002, against an Oca2-null reference of 0. No KS test and no ablation metric are in the pass-1 table.  
  evidence: `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L111–162; `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L353–410
- **Source and data traceability: 88** — Crossref-checked DOIs for Turing, Schnakenberg, Murray, Jeffery, Yamamoto, Protas, Gross, McCauley and Yoshizawa. The voucher is Seb Morič's CC BY-SA 4.0 Commons photograph, SHA-256 pinned. OpenAlex supplied Bilandžija et al. 2013 as an additional oca2 paper. The 666 nm length itself has no paper measurement.  
  evidence: `datasets/astyanax_mexicanus_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/astyanax_mexicanus_turing_wavelength_s0/README.md` L1–48
- **Code-to-model traceability: 80** — dispersion_eigenvalue, the pseudo-spectral step, and the Lean theorems (trace, determinant, drive, discriminant, equal-diffusivity failure, wavenumber bracket, 36-37x sibling ratio) point at the same integer Jacobian. The composite helper still contains an unused resampled buffer.  
  evidence: `morphoeval/lean/loops/Loop_3125_astyanax_mexicanus_turing_wavelength_s0.lean` L21–41; `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L105–108
- **Reproducibility package: 84** — python3 models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py --self-test passes. Seed 3125, specimen SHA-256 checked before integration, outputs written to fixed paths. Dependencies are the repo scientific stack (numpy, scipy, Pillow, matplotlib). No second machine was used.  
  evidence: `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L77–82; `models/2026-delphi10000-astyanax-mexicanus-turing-wavelength-s0.py` L451–461

Remediation:
- {'id': 'R1', 'item': 'Sweep the diffusion ratio Dv/Du and show that the selected wavelength moves monotonically. Pass 1 renders a single (Du, Dv, gamma).'}
- {'id': 'R2', 'item': 'Add the Du = Dv ablation and report that the maximum growth rate is negative, matching the Lean equal-diffusivity theorem.'}
- {'id': 'R3', 'item': 'Segment the voucher flank and compare its orientation histogram to the simulated patch with a KS statistic. The pass-1 melanin fraction is a threshold count, not a distributional test.'}
- {'id': 'R4', 'item': 'Drop the unused resampled buffer in compose_frame and keep the signed-field composite as the only render path.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Turing wavenumber at 666 nm in the Astyanax cave morph
- **spec_fingerprint** (semantic model + inputs): `567daaa26ea32b6e0db869c7c7df6b1cf14d5ac54c7e7b02f34d577ba1dab69a`
- **input_dataset_sha256**: `datasets/astyanax_mexicanus_turing_wavelength_s0/astyanax_mexicanus_specimen.jpg` (`b7b2d7744b19592f…`), `datasets/astyanax_mexicanus_turing_wavelength_s0/paper_citations.bib` (`a4f57d067c37118a…`)
- `docs/media/astyanax_mexicanus_turing_wavelength_s0_comparison.png` — artifact_sha256 `6b4715eaa25378a9…`, 380,391 B, 1512×528 px
- `docs/media/astyanax_mexicanus_turing_wavelength_s0_growth.gif` — artifact_sha256 `198ad1aa6a86bd6d…`, 2,534,246 B, 960×522 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion_growth_pearson_r | 0.9999681838003501 1 | 1.0 | Pearson r between closed-form Turing growth rates and forward-Euler Fourier-mode slopes |
| dispersion_growth_rmse | 0.004547985598148426 1/time | 0.0 | RMSE of Euler growth rates against analytical sigma(k), dt=0.1 |
| specimen_stripe_wavelength_nm | 660.6743783241391 nm | 24180.0 | radial FFT peak of the nonlinear patch, converted with the linear-mode calibration meters/px |
| specimen_melanin_dark_fraction | 0.002041123521088599 1 | 0.0 | fraction of body-mask pixels with max channel < 0.22 and low saturation on the CC voucher |
| homogeneous_mode_growth | -0.09999999999999998 1/time | -0.2 | sigma(k=0) of the Schnakenberg Jacobian; must be negative (stable without diffusion) |

## Environment at checkpoint


Pass-1 files are new on this branch and are not in the parent commit. Loci are checked against the working tree; blob URLs at the parent commit do not yet contain them.
