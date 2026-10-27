# MorphoEval report — audit `2026-10-27-eldredgeops-rana-18file-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `6c9f67254bb352d26d4034b290d6f535a474fdb2` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-eldredgeops-rana-18file-turing-wavelength-s0-audit -> 2026-10-27-eldredgeops-rana-18file-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Turing k_c at 492 nm for the 18-file Eldredgeops eye (Pass 2) (`models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py`) | 90 | 84 | 88 | 90 | 90 | 86 | 90 | 89.3 | 86.2 | 88.9 | 88.0 | **88.0** | faithful |

## Findings and remediation

### Turing k_c at 492 nm for the 18-file Eldredgeops eye (Pass 2)  —  overall 88.0 (faithful)

Pass 2 runs the gamma sweep (lambda falls monotonically from 696 nm at gamma 1.2 to 341 nm at gamma 5.0, with the slot value 492 nm at gamma 2.4), the Dv/Du ratio sweep, and a simulated Du=Dv field whose standard deviation is 3.1e-5 against 0.353 for the Turing field. Otsu segmentation of the Commons eye gives foreground fraction 0.537. The KS distance between specimen and simulation radial spectra is 0.635 (reference 0). Three seeds share the nonlinear FFT bin at 574 nm (sample CV 0 on this grid). Dispersion Pearson r remains 1.0 with RMSE 3.15e-12. Image-fitness CV similarity of the 720x540 12-frame GIF against the eye photograph is 96.0.

- **Model fidelity to source: 90** — Unchanged Schnakenberg Jacobian and k_c formula, now checked by a gamma sweep that recovers lambda proportional to gamma^(-1/2) and by a simulated equal-diffusion field that stays flat. Still not a measured Devonian diffusivity.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L70–88; `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L510–537
- **Morphological realism / specimen match: 84** — The Pass-2 figure shows the Commons eye, the Turing field, and the flat ablation field. Otsu foreground fraction on the photograph is 0.537. The spot lattice is not a lens-resolved copy of the eye, so the score stays below the low 90s. The GIF still ends on the photograph (CV 96.0).  
  evidence: `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L565–580; `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L618–668; `docs/media/eldredgeops_rana_18file_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — Raising gamma from 1.2 to 5.0 shortens lambda_c monotonically (696, 568, 492, 426, 381, 341 nm). Factors of the slot Dv/Du ratio move the band from stable to unstable. The Du=Dv field std is 3.1e-5 against 0.353 for the patterned run.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L510–521; `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L673–698
- **Morphometric validation: 90** — Dispersion Pearson r = 1.0, RMSE 3.15e-12. Nonlinear wavelength 574 nm versus linear 492 nm. Specimen coherency 0.479 versus simulation 0.746. KS distance of radial spectra 0.635 versus a null of 0. Otsu foreground 0.537 versus a half-frame null of 0.5. Three seeds agree at 574 nm. These references are not copies of the measured values.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L130–162; `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L596–616
- **Source and data traceability: 90** — Same resolvable citations and SHA-256-pinned CC BY-SA 4.0 photograph (RealGatba, File:Eldredgeops eye.jpg) as Pass 1. The 492 nm and 387 Ma assignments are stated in the reference JSON and README.  
  evidence: `datasets/eldredgeops_rana_18file_turing_wavelength_s0/eldredgeops_rana_18file_turing_wavelength_s0_reference.json` L11–36; `datasets/eldredgeops_rana_18file_turing_wavelength_s0/README.md` L1–40
- **Code-to-model traceability: 86** — Pass-2 functions gamma_sweep, ratio_sweep, specimen_segmentation, ks_statistic, and augment_pass2 are separate from the Pass-1 PDE core and are named in the comparison figure. Lean theorems are unchanged. Still a single assessor.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L673–700; `morphoeval/lean/loops/Loop_3215_eldredgeops_rana_18file_turing_wavelength_s0.lean` L19–45
- **Reproducibility package: 90** — --pass2 --self-test re-checks Pearson r > 0.98, the negative analytic ablation, monotonic gamma, ablation-field std below 15 percent of the Turing std, a non-zero KS distance, GIF frames and resolution, and the specimen SHA-256. Seed 3215 is fixed; two extra seeds are 3216 and 3217.  
  evidence: `models/2026-delphi10000-eldredgeops-rana-18file-turing-wavelength-s0.py` L441–465

Remediation:
- The Commons eye is not a stratigraphically pinned Cazenovian 18-file voucher.
- Pearson r = 1 is the spectral mode plus RK4 on its eigenvector, not a pixel registration of the eye. The nonlinear peak sits at 574 nm.
- Three seeds share one radial FFT bin, so the wavelength CV on this 96^2 grid is 0.
- No second assessor. Nanopubs are unsigned: ~/.nanopub/profile.yml is absent and the nanopub package is not installed.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Turing k_c at 492 nm for the 18-file Eldredgeops eye (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `77967ca675ff28a543407cf6030276194492fe8aeb2e1d000397c86f88e1f90a`
- **input_dataset_sha256**: `datasets/eldredgeops_rana_18file_turing_wavelength_s0/eldredgeops_rana_18file_turing_wavelength_s0_reference.json` (`8ef91c6ae2a52449…`), `datasets/eldredgeops_rana_18file_turing_wavelength_s0/eldredgeops_rana_18file_turing_wavelength_s0_specimen.jpg` (`3a83d6e639450366…`)
- `docs/media/eldredgeops_rana_18file_turing_wavelength_s0_comparison.png` — artifact_sha256 `3bd1a330bd5a1ec1…`, 430,251 B, 1620×984 px
- `docs/media/eldredgeops_rana_18file_turing_wavelength_s0_growth.gif` — artifact_sha256 `3a460c467ba2181d…`, 3,281,645 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| RK4 vs eigenvalue dispersion Pearson r | 1.0 Pearson r | 1.0 | corrcoef of RK4 Fourier-mode growth rates against the leading real eigenvalue of J - k^2 D *(RMSE = 0.000000 on n = 16 real-eigenvalue modes)* |
| nonlinear FFT wavelength of the saturated u field | 574.0 nm | 492.0 | radial peak of |FFT(u)|^2 times 492 nm per model unit *(Saturated spots shift off the linear k_c; the two numbers are not forced equal)* |
| specimen eye structure-tensor coherency | 0.4788 coherency | 0.7456 | mean (λ1-λ2)/(λ1+λ2) on Sobel structure tensor of the Commons eye photograph *(Photograph versus simulation; a self-copy would have been rejected)* |
| equal-diffusion ablation max real growth rate | -0.811874 1/time | 0.0 | max Re(σ(k)) with Du = Dv on k/k_c in [0.2, 3] *(Negative value: removing the diffusivity contrast kills the Turing band)* |
| conserved-arc wavelengths-per-file ratio versus 17-file successor | 1.058824 ratio | 1.0 | n_files(387 Ma) / n_files(384 Ma) = 18/17 at fixed 492 nm *(The 492 nm wavenumber is held fixed; file loss changes wavelengths per file)* |
| gamma sweep endpoint wavelength | 340.8676 nm | 492.0 | λ(γ=5) = 492 * sqrt(2.4/5) with Du, Dv fixed at the slot rescaling *(monotonic_decreasing=True)* |
| equal-diffusion ablation field standard deviation | 3.1e-05 concentration | 0.352923 | std(u) after 160 Fourier-split steps with Du = Dv *(control vs patterned field; not a self-comparison)* |
| specimen eye Otsu foreground fraction | 0.5369 area fraction | 0.5 | fraction of pixels darker than Otsu threshold on the Commons eye *(threshold=0.523)* |
| KS distance of specimen vs simulation radial spectra | 0.6348 KS | 0.0 | max |ΔCDF| between normalised radial FFT profiles *(Non-zero: the spot lattice is not a copy of the eye photograph)* |
| three-seed mean nonlinear wavelength | 574.0 nm | 492.0 | mean FFT-peak wavelength at seeds 3215, 3216 and 3217 *(per-seed nm = [574.0, 574.0, 574.0]; sample CV = 0.000000 (shared radial bin))* |

## Environment at checkpoint


Pass 2. Unsigned nanopubs: no ~/.nanopub/profile.yml. The publisher ORCID is the vendored ProbKnow template URI, not a claim of authorship by that person.
