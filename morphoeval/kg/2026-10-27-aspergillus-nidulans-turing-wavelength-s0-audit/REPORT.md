# MorphoEval report — audit `2026-10-27-aspergillus-nidulans-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Aspergillus nidulans Turing onset wavelength at 12.84 µm (nominal 13 µm), Holocene 0 Ma (`models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py`) | 86 | 76 | 86 | 84 | 90 | 86 | 88 | 86.0 | 79.0 | 88.1 | 83.8 | **83.8** | faithful |

## Findings and remediation

### Aspergillus nidulans Turing onset wavelength at 12.84 µm (nominal 13 µm), Holocene 0 Ma  —  overall 83.8 (faithful)

Pass 1. Schnakenberg onset wavelength is fixed at 1.284e-05 m (12.84 µm, nominal 13 µm) for the Holocene singleton at 0 Ma. Dispersion Pearson is 0.9977. The colony GIF is specimen-conditioned, so specimen-match is 76. Sobel orientation, multi-seed CV, and the alpha measurement row are still open.

- **Model fidelity to source: 86** — Schnakenberg kinetics, the steady state, the Jacobian and the onset formula k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) are implemented directly, and the PDE mode-growth Pearson against Re(σ) is 0.998. The pigment identification and the 12.84 µm length are a transfer onto A. nidulans, so this is not a line-by-line reproduction of a fungus paper.  
  evidence: `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L16–27; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L89–121; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L154–195
- **Morphological realism / specimen match: 76** — The comparison figure places the public-domain colony beside the Otsu mask, the conditioned render, the activator stripes and the flat ablation. The GIF last frame follows the colony because equation (4) multiplies the photograph by a weak carrier (α = 0.12), which is why computer-vision similarity is high. The spiral itself is measured, not generated, so the rubric score stays below the visual similarity number.  
  evidence: `docs/media/aspergillus_nidulans_turing_wavelength_s0_comparison.png` L0–0; `docs/media/aspergillus_nidulans_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L337–378
- **Parameter expressivity: 86** — Raising D_v shortens λ_c monotonically across 8.50–18, which is the wavenumber control the law claims. Raising α lowers pixel Pearson against the specimen monotonically. The α table is checked in the self-test and is not yet a measurement-sidecar row.  
  evidence: `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L263–273; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L426–435
- **Morphometric validation: 84** — Onset λ_c is 12.84 µm against the nominal 13 µm label. Dispersion Pearson is 0.998 with RMSE 0.014. Specimen pixel Pearson is 0.997 against a pure-stripe null near 0. Otsu IoU is 0.982, not 1. Equal-diffusion activator std is lower than the patterned std. Intensity KS is not the 12-bin Sobel histogram, and the wavelength was not repeated across seeds.  
  evidence: `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L280–298; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L312–330; `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L239–253
- **Source and data traceability: 90** — Seed DOIs were resolved on OpenAlex after the keyword query returned unrelated hits. The colony file is the Commons public-domain macrograph, SHA-256 pinned in the reference JSON. Slot 182 is a different simulation id and is not reused.  
  evidence: `datasets/aspergillus_nidulans_turing_wavelength_s0/paper_citations.bib` L1–88; `datasets/aspergillus_nidulans_turing_wavelength_s0/aspergillus_nidulans_turing_wavelength_s0_reference.json` L1–55; `datasets/aspergillus_nidulans_turing_wavelength_s0/scout_manifest.json` L1–20
- **Code-to-model traceability: 86** — Docstring equations (1)–(4) match steady_state, jacobian, turing_report, simulate and modulate. The Lean file records the 12840 nm wavelength, the 9.295 diffusion ratio and the 0 Ma singleton series. It does not encode the real-eigenvalue formula, so code-traceability is not complete.  
  evidence: `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L16–27; `morphoeval/lean/loops/Loop_3122_aspergillus_nidulans_turing_wavelength_s0.lean` L14–52; `morphoeval/math_kg_loops/delphi10000-aspergillus-nidulans-turing-wavelength-s0.yaml` L16–55
- **Reproducibility package: 88** — Seed 3122, grid 84, D_v 9.295 and α 0.12 are module constants. --self-test renders the PNG and the 16-frame GIF and checks the Turing inequalities, the ablation, the monotone sweep, Pearson > 0.98, frame count and resolution. Dependencies are the repo scientific stack.  
  evidence: `models/2026-delphi10000-aspergillus-nidulans-turing-wavelength-s0.py` L67–79; `datasets/aspergillus_nidulans_turing_wavelength_s0/README.md` L1–21; `docs/media/aspergillus_nidulans_turing_wavelength_s0_growth.gif` L0–0

Remediation:
- Run the nonlinear integrator from at least three seeds and report the coefficient of variation of the measured wavelength.
- Compare 12-bin Sobel orientation histograms of the specimen and the GIF last frame; the pass-1 KS is on grayscale intensity only.
- Export the α-coupling sweep into measure_morphoeval so parameter expressivity is in the measurement sidecar.

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

## Environment at checkpoint

- python: 3.12
- numpy: 2.5.4
- matplotlib: Agg
- Pillow: 13.0.0
- scipy: 1.18.1

Pass 1 checkpoint is replaced with the commit that contains these paths before the knowledge-graph build.
