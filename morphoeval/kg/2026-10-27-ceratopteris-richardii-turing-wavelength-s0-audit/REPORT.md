# MorphoEval report — audit `2026-10-27-ceratopteris-richardii-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `8bc1f20b8ecb6382629bf389fb84a0d39e22d0ca` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ceratopteris richardii Schnakenberg wavelength at 256 µm (Holocene, 0 Ma) (`models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py`) | 88 | 62 | 74 | 76 | 90 | 86 | 88 | 83.3 | 67.2 | 88.1 | 78.3 | **78.3** | mostly faithful |

## Findings and remediation

### Ceratopteris richardii Schnakenberg wavelength at 256 µm (Holocene, 0 Ma)  —  overall 78.3 (mostly faithful)

Pass 1. Analytical lambda_c = 256 µm at D_u = 8, D_v = 96 (ratio 12), gamma = 0.012738 s^-1. Dispersion Pearson r = 1.0 against matrix eigenvalues. The integrated field is seeded on the critical mode and its Fourier peak is 256 µm. The 12-frame GIF is that stripe field (800x600). Computer-vision similarity to the canonical plate is 65.9, short of the image-fitness gate. Plate Otsu foreground fraction is 0.742 against a 0.5 prior. Sweep, ablation, voucher segmentation and terminal-plate lock are implemented but not exercised (REMEDIATION_PASS = 1).

- **Model fidelity to source: 88** — Schnakenberg steady state, Jacobian, and k_c formula are implemented and gamma is solved for 256 µm. The field is seeded on the critical mode rather than grown from unstructured noise, so spontaneous mode selection is not what the figure shows.  
  evidence: `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L63–89; `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L166–189
- **Morphological realism / specimen match: 62** — The comparison figure places the canonical plate beside green stripes. The GIF never becomes the plate. CV similarity of the last frame to the gallery plate is 65.9.  
  evidence: `docs/media/ceratopteris_richardii_turing_wavelength_s0_comparison.png` L0–0; `docs/media/ceratopteris_richardii_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 74** — self_test checks that lambda falls as gamma rises, and that equal diffusion is stable. Those sweeps are not in the rendered figure or the Pass-1 measurement list because REMEDIATION_PASS is 1.  
  evidence: `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L209–224; `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L571–594
- **Morphometric validation: 76** — Dispersion Pearson r is 1.0 against a null of 0. Plate foreground fraction 0.742 and coherency 0.231 are measured, not copied from the reference priors. KS, RMSE, and the gametophyte voucher are not yet in the measurement list.  
  evidence: `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L151–158; `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L263–285
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing, Schnakenberg, Gierer-Meinhardt, Geng 2022, Bartz 2018 and Hickok 1995. Plate SHA-256 and the CC BY-SA voucher URL are in the reference JSON.  
  evidence: `datasets/ceratopteris_richardii_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/ceratopteris_richardii_turing_wavelength_s0/ceratopteris_richardii_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 86** — Module docstring and function docstrings name the Schnakenberg terms and the 256 µm constraint. Lean theorems pin the same integers (256, ratio 12, window 768).  
  evidence: `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L1–28; `morphoeval/lean/loops/Loop_3143_ceratopteris_richardii_turing_wavelength_s0.lean` L28–46
- **Reproducibility package: 88** — Seed 3143, --self-test, and deterministic explicit integration. Pass 1 self-test passed. Dependencies are the repo scientific stack.  
  evidence: `models/2026-delphi10000-ceratopteris-richardii-turing-wavelength-s0.py` L571–599; `datasets/ceratopteris_richardii_turing_wavelength_s0/README.md` L1–40

Remediation:
- {'id': 'R1', 'item': 'Lock the GIF terminal frame to the canonical plate so orientation, radial FFT and coherency are scored against that plate (Pass-1 CV 65.9, image fitness below 70).'}
- {'id': 'R2', 'item': 'Render the gamma sweep and emit it as a measurement. Self-test already checks monotonicity.'}
- {'id': 'R3', 'item': 'Report the equal-diffusion ablation sigma_max as a negative control measurement.'}
- {'id': 'R4', 'item': 'Segment the CC BY-SA gametophyte voucher and add KS plus RMSE goodness-of-fit against the plate.'}
- {'id': 'R5', 'item': 'Single assessor only. No second provider scored this audit.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Ceratopteris richardii Schnakenberg wavelength at 256 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `98d67d0345048a400433c4abc043d914baa53b4d681328ad1e3cbae41c4b9bdb`
- **input_dataset_sha256**: `datasets/ceratopteris_richardii_turing_wavelength_s0/Ceratopteris_richardii_cfern_3weeks_cc-by-sa.jpg` (`597a43f0f93f748e…`), `datasets/ceratopteris_richardii_turing_wavelength_s0/Ceratopteris_richardii_primordium_plate.jpg` (`3fcf2684662c089b…`)
- `docs/media/ceratopteris_richardii_turing_wavelength_s0_comparison.png` — artifact_sha256 `b784a5d262e3f76d…`, 369,299 B, 1344×1008 px
- `docs/media/ceratopteris_richardii_turing_wavelength_s0_growth.gif` — artifact_sha256 `9af095f205ddbfa2…`, 2,463,913 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| critical_wavelength_um | 256.0 µm | 256.0 | k_c from Schnakenberg Jacobian at the slot-3143 diffusivities; lambda = 2*pi/k_c *(analytical wavelength; RMSE against the grid measurement is separate)* |
| stripe_wavelength_grid_um | 256.0 µm | 256.0 | parabolic interpolation of the radial FFT peak of the activator, times dx *(RMSE is |grid peak - 256 µm|)* |
| pearson_r_dispersion | 1.0 1 | 0.0 | Pearson r of sigma(k) closed form vs 2x2 matrix eigenvalues, 48 wavenumbers *(goodness-of-fit of the implemented dispersion relation)* |
| specimen_plate_foreground_fraction | 0.7419 1 | 0.5 | Otsu mask on the canonical plate above the caption strip *(non-circular specimen segmentation)* |
| specimen_plate_coherency | 0.2315 1 | 0.25 | mean structure-tensor coherency of the canonical plate *(specimen image coherency)* |
| plate_width_um | 754.72 µm | 768.0 | 800 px times 100/106 µm per px from the drawn scale bar *(plate is slightly narrower than the 768 µm window)* |

## Environment at checkpoint


Pass 1. REMEDIATION_PASS = 1. No nanopub profile on this runner; ORCID is the repository setup-guide placeholder and the nanopubs are unsigned.
