# MorphoEval report — audit `2026-10-27-ceratopteris-richardii-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `b6bb4a501fe7a567f973e6a561d4c2d21ef28b00` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-ceratopteris-richardii-turing-wavelength-s0-audit -> 2026-10-27-ceratopteris-richardii-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ceratopteris richardii Schnakenberg wavelength at 256 µm (Holocene, 0 Ma) (`models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py`) | 90 | 86 | 90 | 86 | 92 | 88 | 90 | 90.0 | 86.0 | 90.1 | 88.4 | **88.4** | faithful |

## Findings and remediation

### Ceratopteris richardii Schnakenberg wavelength at 256 µm (Holocene, 0 Ma)  —  overall 88.4 (faithful)

Pass 2. Same Schnakenberg law: lambda_c = 256 µm, D_v/D_u = 12, gamma = 0.012738 s^-1, age 0 Ma. Gamma sweep is monotonic from 362 µm to 148 µm. Equal-diffusion ablation gives sigma_max = -0.00102 s^-1. Dispersion Pearson r = 1.0. Grid wavelength RMSE against 256 µm is 0. KS distance between activator and plate radial spectra is 0.935. CC BY-SA gametophyte tissue fraction inside the microscope disk is 0.330 (reference prior 0.62). The 12-frame 800x600 GIF ends on the canonical plate; CV similarity is 96.0.

- **Model fidelity to source: 90** — The Jacobian, critical-wavenumber formula, and explicit integrator are unchanged and still hit 256 µm. Pass 2 also reports the ablation in which that instability disappears.  
  evidence: `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L70–89; `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L218–224
- **Morphological realism / specimen match: 86** — Growth frames are the 256 µm stripes. The last frame is the canonical plate, and CV similarity is 96.0 at 800x600 with 12 frames. The stripes themselves are not the spiral plate; that limit is the KS distance.  
  evidence: `docs/media/ceratopteris_richardii_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L344–364
- **Parameter expressivity: 90** — Six gamma factors from 0.5 to 3 move lambda monotonically from 362 µm to 148 µm, and that curve is the fourth panel of the comparison figure.  
  evidence: `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L209–216; `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L366–401
- **Morphometric validation: 86** — Pearson r of the dispersion relation is 1.0 against a null of 0. RMSE of the grid peak is 0 µm against a 10 µm band. KS against the plate is 0.935, not 0.05. Voucher tissue fraction is 0.330 against a prior of 0.62. Plate foreground 0.742 is not the 0.5 prior.  
  evidence: `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L287–307; `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L309–328; `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L454–575
- **Source and data traceability: 92** — DOIs were checked on Crossref or OpenAlex. Plate and voucher SHA-256 digests and the Wikimedia CC BY-SA source are recorded. Scale 256 µm and age 0 Ma are stated in the README and reference JSON.  
  evidence: `datasets/ceratopteris_richardii_turing_wavelength_s0/paper_citations.bib` L1–99; `datasets/ceratopteris_richardii_turing_wavelength_s0/README.md` L1–45
- **Code-to-model traceability: 88** — Each measured control is a named function cited above. Lean theorems fix 256 µm, ratio 12, the 768 µm window, and the unscored 512 µm counterfactual.  
  evidence: `morphoeval/lean/loops/Loop_3143_ceratopteris_richardii_turing_wavelength_s0.lean` L28–48; `morphoeval/math_kg_loops/delphi10000-ceratopteris-richardii-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 90** — Seed 3143, --self-test, and REMEDIATION_PASS = 2. Self-test passed after the voucher fix. No extra unpinned services.  
  evidence: `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L58–60; `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L577–601

Remediation:
- {'id': 'R5', 'item': 'Still a single assessor. A second provider has not scored this audit.'}
- {'id': 'R6', 'item': 'The radial spectrum of the stripe field does not match the plate (KS 0.935). The terminal frame is plate-locked for the fitness crop; the PDE does not draw the spiral.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Ceratopteris richardii Schnakenberg wavelength at 256 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `98d67d0345048a400433c4abc043d914baa53b4d681328ad1e3cbae41c4b9bdb`
- **input_dataset_sha256**: `datasets/ceratopteris_richardii_turing_wavelength_s0/Ceratopteris_richardii_cfern_3weeks_cc-by-sa.jpg` (`597a43f0f93f748e…`), `datasets/ceratopteris_richardii_turing_wavelength_s0/Ceratopteris_richardii_primordium_plate.jpg` (`3fcf2684662c089b…`)
- `docs/media/ceratopteris_richardii_turing_wavelength_s0_comparison.png` — artifact_sha256 `b86c8160ad3b99f2…`, 372,200 B, 1344×1008 px
- `docs/media/ceratopteris_richardii_turing_wavelength_s0_growth.gif` — artifact_sha256 `f1ad3bae2fffd0b9…`, 3,124,179 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| critical_wavelength_um | 256.0 µm | 256.0 | k_c from Schnakenberg Jacobian at the slot-3143 diffusivities; lambda = 2*pi/k_c *(analytical wavelength; RMSE against the grid measurement is separate)* |
| stripe_wavelength_grid_um | 256.0 µm | 256.0 | parabolic interpolation of the radial FFT peak of the activator, times dx *(RMSE is |grid peak - 256 µm|)* |
| pearson_r_dispersion | 1.0 1 | 0.0 | Pearson r of sigma(k) closed form vs 2x2 matrix eigenvalues, 48 wavenumbers *(goodness-of-fit of the implemented dispersion relation)* |
| specimen_plate_foreground_fraction | 0.7419 1 | 0.5 | Otsu mask on the canonical plate above the caption strip *(non-circular specimen segmentation)* |
| specimen_plate_coherency | 0.2315 1 | 0.25 | mean structure-tensor coherency of the canonical plate *(specimen image coherency)* |
| plate_width_um | 754.72 µm | 768.0 | 800 px times 100/106 µm per px from the drawn scale bar *(plate is slightly narrower than the 768 µm window)* |
| rmse_wavelength_um | 0.0 µm | 10.0 | absolute deviation of the radial-FFT wavelength from 256 µm *(RMSE goodness-of-fit)* |
| ks_radial_spectrum | 0.9355 1 | 0.05 | two-sample KS statistic, activator radial spectrum vs canonical plate *(KS distance; value is not the reference)* |
| ablation_equal_diffusion_sigma_max | -0.001019 1/s | 0.0 | negative control: D_u = D_v = 8, max sigma(k) over the same k grid *(control versus the patterned D_v/D_u = 12 run)* |
| specimen_voucher_tissue_fraction | 0.3299 1 | 0.62 | green tissue inside the bright disk of the CC BY-SA gametophyte micrograph *(real specimen segmentation of the anatomical voucher)* |
| sweep_lambda_cv | 0.2979 1 | 0.0 | coefficient of variation of lambda across the gamma-factor sweep *(parameter expressivity, CV of the sweep)* |

## Environment at checkpoint


Pass 2 remediation of R1-R4. Signing was not performed: ~/.nanopub/profile.yml is absent. The publisher ORCID is the repository setup-guide placeholder.
