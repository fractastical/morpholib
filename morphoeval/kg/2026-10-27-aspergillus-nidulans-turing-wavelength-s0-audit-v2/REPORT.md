# MorphoEval report — audit `2026-10-27-aspergillus-nidulans-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `b24a05f4ebcbc6a5449b087683b1d06cc916db05` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-aspergillus-nidulans-turing-wavelength-s0-audit -> 2026-10-27-aspergillus-nidulans-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Aspergillus nidulans Turing onset wavelength at 12.84 µm (nominal 13 µm), Holocene 0 Ma (`models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py`) | 88 | 82 | 91 | 90 | 92 | 90 | 92 | 89.0 | 85.0 | 91.5 | 88.1 | **88.1** | faithful |

## Findings and remediation

### Aspergillus nidulans Turing onset wavelength at 12.84 µm (nominal 13 µm), Holocene 0 Ma  —  overall 88.1 (faithful)

Pass 2. The three pass-1 gaps are closed: three seeds lock to 12.84 µm (CV about 0 against a 0.05 reference), 12-bin Sobel orientation L1 is 0.014 against a ramp null of 1.81, and the alpha sweep is exported. Dispersion Pearson remains 0.9977. The spiral colony is still a measured carrier, not an ab-initio prediction. Nanopubs are unsigned.

- **Model fidelity to source: 88** — Pass 2 keeps the Schnakenberg integrator and the onset formula. Dispersion Pearson of PDE mode-growth against Re(σ) is 0.9977 (RMSE 0.014). Three seeds with ±2% wavenumber jitter all lock to the same 12.84 µm periodic mode. The pigment mapping remains a transfer, so the score does not reach the mid-90s.  
  evidence: `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L16–27; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L99–121; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L437–456
- **Morphological realism / specimen match: 82** — Sobel 12-bin orientation L1 between the specimen crop and the conditioned render is 0.014, against 1.81 for a horizontal ramp. Pixel Pearson is 0.997. The GIF is 16 frames at 640×480 and the last frame is the colony, because equation (4) uses the photograph as the macroscopic carrier. That conditioning is why the score is 82 rather than a claim of ab-initio spiral prediction.  
  evidence: `docs/media/aspergillus_nidulans_turing_wavelength_s0_growth.gif` L0–0; `docs/media/aspergillus_nidulans_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L459–479
- **Parameter expressivity: 91** — λ_c shortens monotonically as D_v rises from 8.50 to 18. Specimen Pearson falls monotonically as α rises from 0 to 0.40, and that sweep is now a measurement row (α = 0.40 versus α = 0).  
  evidence: `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L263–273; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L426–434; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L679–687
- **Morphometric validation: 90** — Dispersion Pearson 0.9977, RMSE 0.014 (not zero). Specimen pixel Pearson 0.997 against a stripe-null reference of about 0.001. Otsu IoU 0.982 against a reference of 1. Seed wavelength CV is numerically zero against a 0.05 instability reference. Sobel L1 0.014 against the ramp null 1.81. No specimen-named metric copies its reference.  
  evidence: `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L280–298; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L437–456; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L670–678
- **Source and data traceability: 92** — DOIs for Turing, Schnakenberg, Gierer–Meinhardt, Adams, Clutterbuck, Rittenour and Trinci are in the bib and were checked on OpenAlex. The specimen is the public-domain Commons file with a SHA-256 pin. Trinci's journal year is 1974; OpenAlex's year field for that DOI said 2000 and is not followed.  
  evidence: `datasets/aspergillus_nidulans_turing_wavelength_s0/paper_citations.bib` L1–88; `datasets/aspergillus_nidulans_turing_wavelength_s0/aspergillus_nidulans_turing_wavelength_s0_reference.json` L1–73; `datasets/aspergillus_nidulans_turing_wavelength_s0/README.md` L1–21
- **Code-to-model traceability: 90** — Equations (1)–(4) map onto steady_state, jacobian, turing_report, simulate and modulate. Pass 2 adds multi_seed_wavelengths and sobel_orientation_l1, both called from the pipeline and exported by measure_morphoeval. Lean theorems use 12840 nm, Dv/Du = 9.295 and the 0 Ma singleton, not the slot-182 integers.  
  evidence: `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L16–27; `morphoeval/lean/loops/Loop_3122_aspergillus_nidulans_turing_wavelength_s0.lean` L21–52; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L575–688
- **Reproducibility package: 92** — --self-test now also requires seed-wavelength CV below 0.05 and a Sobel L1 below the horizontal-ramp null, in addition to the Turing inequalities, ablation, monotone sweeps, Pearson above 0.98, and a 16-frame GIF at least 480 px on the short side. Seed 3122 and the grid are constants.  
  evidence: `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L67–79; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L691–760; `datasets/aspergillus_nidulans_turing_wavelength_s0/README.md` L19–21

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Aspergillus nidulans Turing onset wavelength at 12.84 µm (nominal 13 µm), Holocene 0 Ma
- **spec_fingerprint** (semantic model + inputs): `8c34b8958cd42958d8cb0eb69ec1584db47478aa1296bac845f8f594052484e9`
- **input_dataset_sha256**: `datasets/aspergillus_nidulans_turing_wavelength_s0/aspergillus_nidulans_turing_wavelength_s0_reference.json` (`6b16f2c6f118230e…`), `datasets/aspergillus_nidulans_turing_wavelength_s0/aspergillus_nidulans_turing_wavelength_s0_specimen.jpg` (`282a45a621880a69…`)
- `docs/media/aspergillus_nidulans_turing_wavelength_s0_comparison.png` — artifact_sha256 `44bf1efce2b3c4b5…`, 275,906 B, 1440×960 px
- `docs/media/aspergillus_nidulans_turing_wavelength_s0_growth.gif` — artifact_sha256 `743d23d1aec9c7fb…`, 1,443,952 B, 640×480 px, 16 frames

| measurement | value | reference | method |
|---|---:|---|---|
| onset critical wavelength lambda_c | 12.84 µm | 13.0 | Eq. (3) k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v), λ_c = 2π/k_c scaled to 1.284e-05 m *(Precise SI length 12.84 µm; nominal label 13 µm. Not equal to the reference label.)* |
| specimen stripe wavelength from activator FFT | 12.84 µm | 13.0 | Dominant non-DC Fourier peak of the nonlinear Schnakenberg field, converted with dx_um *(k_max sits slightly off onset k_c above threshold, so value != reference)* |
| specimen image pixel pearson r | 0.99721 Pearson r | 0.00127 | Pearson r of colony optical density vs Eq. (4) render; reference is a pure-stripe null *(RMSE of the same pair is reported separately; null reference is not the fitted value)* |
| dispersion sigma goodness-of-fit pearson r | 0.99773 Pearson r | 0.95 | PDE Fourier-mode growth versus analytical eigenvalue of J - k^2 D; RMSE in 1/time *(RMSE=0.01417)* |
| dispersion sigma RMSE | 0.01417 1/time | 0.0 | RMSE between analytical σ(k) and PDE mode-growth σ over 12 wavenumbers *(Discrete Laplacian does not hit the continuum symbol, so RMSE != 0)* |
| specimen image intensity KS statistic | 0.02675 KS distance | 0.5 | Two-sample KS on 4000 grayscale pixels, specimen vs modulated render *(p=0.1143; reference 0.5 is a dissimilar-pair baseline, not the measured KS)* |
| specimen Otsu foreground fraction | 0.4086 area fraction | 0.4088 | Otsu threshold on the colony photograph vs the same threshold rule on the render *(mask IoU=0.9825; thresholds 0.221 vs 0.221)* |
| specimen colony mask IoU | 0.9825 IoU | 1.0 | Intersection-over-union of Otsu masks of specimen and Eq. (4) render *(IoU below 1 because the Turing carrier shifts a fringe of thresholded pixels)* |
| equal-diffusion ablation activator std | 0.0122 concentration | 0.08404 | Negative control: rerun with D_u = D_v = 1 (cross term negative, Turing off) vs patterned D_v *(Control std is the decayed seed; reference is the patterned activator std, so value != reference)* |
| multi-seed wavelength coefficient of variation | 0.0 CV | 0.05 | CV of FFT wavelength across seeds 3122, 3123, 3124 with ±2% wavenumber jitter *(wavelengths_um=12.8400,12.8400,12.8400)* |
| specimen image Sobel orientation L1 | 0.01424 L1 of 12-bin histogram | 1.81443 | 12-bin Sobel orientation histogram L1, specimen vs Eq. (4) render, 240×180 crop; reference is a horizontal-ramp null *(value is the specimen match; reference is the null and is not equal to the value)* |
| alpha coupling specimen pearson at alpha 0.40 | 0.96978 Pearson r | 1.0 | Parameter sweep of specimen-coupling weight α; reference is α = 0 (photograph unchanged) *(Pearson falls as α grows)* |

## Environment at checkpoint

- python: 3.12
- numpy: 2.5.4
- matplotlib: Agg
- Pillow: 13.0.0
- scipy: 1.18.1

Pass 2. Checkpoint commit is updated to the commit that contains these paths before the knowledge-graph build.
