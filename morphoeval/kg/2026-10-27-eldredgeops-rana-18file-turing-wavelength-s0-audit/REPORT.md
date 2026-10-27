# MorphoEval report — audit `2026-10-27-eldredgeops-rana-18file-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `332efe5138233530509bf80322afc73c77f98255` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Turing k_c at 492 nm for the 18-file Eldredgeops eye (Cazenovian, 387 Ma) (`models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py`) | 88 | 76 | 64 | 78 | 90 | 84 | 86 | 80.0 | 76.8 | 86.8 | 80.7 | **80.7** | faithful |

## Findings and remediation

### Turing k_c at 492 nm for the 18-file Eldredgeops eye (Cazenovian, 387 Ma)  —  overall 80.7 (faithful)

Pass 1 integrates the Schnakenberg system so the linear critical wavelength is exactly 492 nm (4.915e-07 m) at 387 Ma. RK4 growth rates on the Turing band match the leading eigenvalue (Pearson r = 1, RMSE ~ 3e-12) because the spatial operator is spectral and the initial condition is the eigenvector. The saturated 2D field selects a nonlinear FFT peak at 574 nm. Equal diffusivities give max Re(sigma) = -0.812. The 12-frame 720x540 GIF ends on the CC BY-SA Commons eye photograph. The 18-to-17 file step changes wavelengths per file by 18/17 if the dorsoventral arc is conserved; it does not retune k_c. Gamma sweep, ablation field, Otsu mask, and KS test are implemented but not executed in this pass.

- **Model fidelity to source: 88** — The script implements Schnakenberg kinetics, the steady-state Jacobian, the four Turing inequalities, and k_c^2 = (Dv f_u + Du g_v) / (2 Du Dv), then rescales space so lambda_c equals the assigned 492 nm. The 2D step is a Fourier reaction-diffusion split of those same terms. It is not a fit to a measured Devonian diffusivity.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L70–88; `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L188–233
- **Morphological realism / specimen match: 76** — The comparison figure places the CC BY-SA Eldredgeops eye beside the saturated field, and the GIF's last frame is that photograph at 720x540. Earlier frames are the growing spot field, which is not a lens-by-lens drawing of the eye. Score reflects both the real photograph and that gap.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L273–344; `docs/media/eldredgeops_rana_18file_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 64** — Equal diffusivities are shown to stabilise the homogeneous state (max Re(sigma) = -0.812), so the diffusivity contrast does control instability. The gamma and Dv/Du sweeps exist as functions but are not run or drawn in the Pass-1 figure, so monotonic control is not yet visible.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L38–48; `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L172–180
- **Morphometric validation: 78** — Dispersion Pearson r = 1.0 with RMSE 3.15e-12 on real-eigenvalue modes (spectral mode plus RK4). Nonlinear FFT wavelength is 574 nm against the linear reference 492 nm. Specimen structure-tensor coherency is 0.479 against 0.746 on the simulated field. KS, Otsu foreground, and multi-seed CV are not in the Pass-1 measurement set.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L130–162; `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L235–247
- **Source and data traceability: 90** — Turing 1952, Schnakenberg 1979, Eldredge 1972 (AMNH handle 2246/1095 and DOI 10.5531/sd.paleo.7), and Clarkson et al. 2006 are cited with resolvable identifiers. The eye photograph is CC BY-SA 4.0, artist RealGatba, SHA-256 pinned. The 492 nm scale is labelled as a campaign assignment.  
  evidence: `datasets/eldredgeops_rana_18file_turing_wavelength_s0/eldredgeops_rana_18file_turing_wavelength_s0_reference.json` L11–36; `datasets/eldredgeops_rana_18file_turing_wavelength_s0/paper_citations.bib` L1–48
- **Code-to-model traceability: 84** — jacobian_raw, diffusivities, dispersion_matrix, rk4_growth, and simulate_field each map to one displayed equation. Lean theorems in MathOfBiology.Loop_3215 pin wavelength positivity, the file-count step, the age order, and Dv > Du. No second assessor re-scored this pass.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L91–128; `morphoeval/lean/loops/Loop_3215_eldredgeops_rana_18file_turing_wavelength_s0.lean` L19–45
- **Reproducibility package: 86** — --self-test checks r > 0.98, RMSE, lambda_c = 1 model unit, negative ablation growth, the 18/17 ratio, GIF frame count and resolution, and the specimen SHA-256. Seed 3215 is fixed. Dependencies are the repo scientific stack (numpy, scipy, pillow, matplotlib).  
  evidence: `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L438–468; `datasets/eldredgeops_rana_18file_turing_wavelength_s0/README.md` L1–40

Remediation:
- Parameter-regime sweep: gamma and Dv/Du sweeps are defined but not executed or drawn on the Pass-1 figure.
- Negative control: equal-diffusion max sigma is analytic only; the ablation field is not simulated in the Pass-1 render.
- Specimen segmentation: whole-frame coherency is reported, but Otsu foreground, radial-spectrum KS, and multi-seed wavelength are not in the Pass-1 outputs.
- No independent second assessor scored this pass.
- Nanopublications cannot be signed: ~/.nanopub/profile.yml is absent.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Turing k_c at 492 nm for the 18-file Eldredgeops eye (Cazenovian, 387 Ma)
- **spec_fingerprint** (semantic model + inputs): `77967ca675ff28a543407cf6030276194492fe8aeb2e1d000397c86f88e1f90a`
- **input_dataset_sha256**: `datasets/eldredgeops_rana_18file_turing_wavelength_s0/eldredgeops_rana_18file_turing_wavelength_s0_reference.json` (`8ef91c6ae2a52449…`), `datasets/eldredgeops_rana_18file_turing_wavelength_s0/eldredgeops_rana_18file_turing_wavelength_s0_specimen.jpg` (`3a83d6e639450366…`)
- `docs/media/eldredgeops_rana_18file_turing_wavelength_s0_comparison.png` — artifact_sha256 `12bd2e9acf0b5629…`, 406,017 B, 1344×1008 px
- `docs/media/eldredgeops_rana_18file_turing_wavelength_s0_growth.gif` — artifact_sha256 `3a460c467ba2181d…`, 3,281,645 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| RK4 vs eigenvalue dispersion Pearson r | 1.0 Pearson r | 1.0 | corrcoef of RK4 Fourier-mode growth rates against the leading real eigenvalue of J - k^2 D *(RMSE = 0.000000 on n = 16 real-eigenvalue modes)* |
| nonlinear FFT wavelength of the saturated u field | 574.0 nm | 492.0 | radial peak of |FFT(u)|^2 times 492 nm per model unit *(Saturated spots shift off the linear k_c; the two numbers are not forced equal)* |
| specimen eye structure-tensor coherency | 0.4788 coherency | 0.7456 | mean (λ1-λ2)/(λ1+λ2) on Sobel structure tensor of the Commons eye photograph *(Photograph versus simulation; a self-copy would have been rejected)* |
| equal-diffusion ablation max real growth rate | -0.811874 1/time | 0.0 | max Re(σ(k)) with Du = Dv on k/k_c in [0.2, 3] *(Negative value: removing the diffusivity contrast kills the Turing band)* |
| conserved-arc wavelengths-per-file ratio versus 17-file successor | 1.058824 ratio | 1.0 | n_files(387 Ma) / n_files(384 Ma) = 18/17 at fixed 492 nm *(The 492 nm wavenumber is held fixed; file loss changes wavelengths per file)* |

## Environment at checkpoint


Pass 1 working tree. ORCID is the vendored ProbKnow template publisher because ~/.nanopub/profile.yml is absent; nanopubs are unsigned and are not a publication by that ORCID holder.
