# MorphoEval report — audit `2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength on Cardamine hirsuta at 154 µm (Pass 1) (`models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py`) | 82 | 76 | 78 | 74 | 90 | 82 | 86 | 80.7 | 75.2 | 86.3 | 80.2 | **80.2** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength on Cardamine hirsuta at 154 µm (Pass 1)  —  overall 80.2 (faithful)

Holocene (0 Ma) detail-scale Schnakenberg model with slot coefficients a=0.10, b=0.90, D_u=0.04, D_v=0.42. Classical lambda_c is pinned at 154.1 µm (k_c^2 = 185/21). Self-test dispersion Pearson r = 0.9999, RMSE = 0.00156, equal-diffusion sigma_max = -0.102, and a monotonic rise in selected wavelength from 164 µm to 217 µm as D_v/D_u goes from 10.5 to 40. Nonlinear spacing on two seeds is 173.4 µm, 12% longer than the classical law. The 12-frame 720x540 GIF scores CV similarity 96.0 against the public-domain plant. Pass 1 does not yet treat those sidecar measurements as the scored record.

- **Model fidelity to source: 82** — The script implements Schnakenberg kinetics, the Jacobian at (u0, v0) = (1, 0.9), and k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v). Trace is negative and the squared Turing inequality holds. The pigment-ocellus reading of u is a reconstruction, so this is not a line-by-line reproduction of a Cardamine pigment paper.  
  evidence: `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L93–118; `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L144–163; `morphoeval/lean/loops/Loop_3140_cardamine_hirsuta_turing_wavelength_s0.lean` L13–28
- **Morphological realism / specimen match: 76** — The GIF is the public-domain plant, with a mild ocellus modulation inside the leaf mask, and the comparison figure places that plant beside the field. Computer-vision similarity of the last frame is 96.0, but the photograph cannot show a 154 µm pattern, and Pass 1 does not yet score the segmentation metrics.  
  evidence: `docs/media/cardamine_hirsuta_turing_wavelength_s0_comparison.png` L0–0; `docs/media/cardamine_hirsuta_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L312–336
- **Parameter expressivity: 78** — Five inhibitor diffusivities are swept and the selected wavelength increases from 163.9 µm to 217.4 µm. Pass 1 records the curve but leaves the monotonicity flag for the measurement sidecar (R1).  
  evidence: `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L226–239; `docs/media/cardamine_hirsuta_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 74** — Dispersion Pearson r is 0.9999 and RMSE is 0.00156 on nine wavenumbers. Nonlinear spacing is 173.4 µm against the 154.1 µm law, which is reported rather than forced to match. Segmentation, KS, and the ablation standard deviation are implemented and are not yet the scored sidecar (R2–R4).  
  evidence: `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L183–217; `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L165–181; `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L692–710
- **Source and data traceability: 90** — Crossref-checked DOIs, a SHA-256-pinned public-domain Commons photograph, and the scout manifest are in the dataset directory. The 154.1 µm scale and 0 Ma age are stated in the reference JSON and the README.  
  evidence: `datasets/cardamine_hirsuta_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/cardamine_hirsuta_turing_wavelength_s0/cardamine_hirsuta_turing_wavelength_s0_reference.json` L1–40; `datasets/cardamine_hirsuta_turing_wavelength_s0/README.md` L1–20
- **Code-to-model traceability: 82** — Equation comments point at the wavenumber law, the Neumann integrator, and the dispersion fit. The Lean file proves the integer forms of those identities, including the Holocene terminal condition 1 - 1 = 0.  
  evidence: `morphoeval/math_kg_loops/delphi10000-cardamine-hirsuta-turing-wavelength-s0.yaml` L1–40; `morphoeval/lean/loops/Loop_3140_cardamine_hirsuta_turing_wavelength_s0.lean` L32–45
- **Reproducibility package: 86** — --self-test checks the hashes, the dispersion correlation, the ablation sign, sweep monotonicity, GIF length, and resolution. Seeds 3140 and 3141 are fixed. Dependencies are the repo scientific stack (numpy, scipy, pillow, matplotlib).  
  evidence: `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L52–76; `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L666–712

Remediation:
- {'id': 'R1', 'item': 'Score the five-point D_v sweep monotonicity from the measurement sidecar, not only from the figure.'}
- {'id': 'R2', 'item': 'Score the equal-diffusion ablation (sigma_max < 0 and a falling field standard deviation) as a negative control.'}
- {'id': 'R3', 'item': 'Record specimen segmentation (foreground fraction, aspect, circularity, coherency) with references that are not copies of the measured values.'}
- {'id': 'R4', 'item': 'Record dispersion Pearson r, RMSE, and the specimen-versus-render orientation KS statistic.'}
- {'id': 'R5', 'item': 'Independent second assessor. This Pass 1 score is a single-agent reading.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=2.5]** `parameter_sweep` → `parameter-expressivity`: Fingerprint the D_v sweep in the measurement sidecar.
- **[OPEN · P=3.0]** `morphometric_validation` → `morphometric-validation`: Fingerprint ablation, segmentation, Pearson r, RMSE, and KS.
- **[OPEN · P=0.5]** `independent_assessor` → `specimen-match`: A second assessor has not scored this loop.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength on Cardamine hirsuta at 154 µm (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `619c6e2e8c89814d4927f6c66381065778340adfaf1d52680367bc6b98ea29d7`
- **input_dataset_sha256**: `datasets/cardamine_hirsuta_turing_wavelength_s0/README.md` (`03de545ad27ccbf4…`), `datasets/cardamine_hirsuta_turing_wavelength_s0/cardamine_hirsuta_specimen.jpg` (`1c268fecb3660b31…`), `datasets/cardamine_hirsuta_turing_wavelength_s0/cardamine_hirsuta_turing_wavelength_s0_reference.json` (`c696a1319c853366…`), `datasets/cardamine_hirsuta_turing_wavelength_s0/paper_citations.bib` (`bfcb6bdfc46bcf19…`), `datasets/cardamine_hirsuta_turing_wavelength_s0/scout_manifest.json` (`79d1ae5d5b27c95e…`)
- `docs/media/cardamine_hirsuta_turing_wavelength_s0_comparison.png` — artifact_sha256 `42f28ae17a71284a…`, 344,427 B, 1440×860 px
- `docs/media/cardamine_hirsuta_turing_wavelength_s0_growth.gif` — artifact_sha256 `911ab38123183c13…`, 775,887 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen leaf foreground area fraction | 0.2825 fraction | 0.5 | Non-white mask on the public-domain Cardamine hirsuta photograph after a 1-pixel opening. |
| specimen rosette bounding-box aspect ratio | 0.6048 width/height | 1.0 | Bounding box of the segmented plant silhouette. |
| specimen contour circularity | 0.0926 4*pi*A/P^2 | 1.0 | Perimeter from the erosion outline of the specimen mask. |
| specimen structure-tensor coherency | 0.2984 coherency | 0.2 | Mean structure-tensor coherency of the specimen luminance. |
| nonlinear ocellus spacing on the detail field | 173.362 µm | 154.1 | Mean first autocorrelation peak of activator u over seeds 3140 and 3141, times dx = 154.1/16 µm. *(per-seed µm = [173.36249999999998, 173.36249999999998]; CV = 0.0000; pattern std = 0.2515)* |
| dispersion-band pearson r | 0.999896 Pearson r | 0.98 | Pearson correlation of analytic σ(k) and a linear-window fit on nine wavenumbers around k_c. *(RMSE = 0.001564)* |
| dispersion-band RMSE | 0.001564 growth-rate units | 0.05 | Root mean square error between analytic and numerically fitted σ(k). |
| equal-diffusion ablation sigma_max | -0.1016 1/time | 0.0 | Maximum analytic growth rate on k in [0.2, 6] with D_u = D_v, plus a 1500-step control integration. *(ablation std 0.0201 -> 0.0002; patterned sigma(k_c) = 0.0703)* |
| diffusion-ratio sweep selected wavelength at D_v/D_u = 40 | 217.435 µm | 154.1 | Argmax of sigma(k) at D_v = 1.60, converted with the operating-point length unit. *(monotonic_increasing=True)* |
| specimen versus render orientation KS statistic | 0.0156 KS D | 0.0 | Two-sample KS on Sobel orientations of the specimen crop and the GIF last frame. *(p = 0.9213; radial FFT Pearson r = 0.9948; CV similarity = 96.0)* |
| specimen image radial-FFT pearson r | 0.9948 Pearson r | 0.9 | Pearson r of the 22-bin log radial FFT used by the image-fitness verifier. |

## Environment at checkpoint


Base commit when the loop was scored. Slot 3140 files are new in the working tree on top of this commit, so blob URLs at this commit do not yet contain them. Evidence is pinned by path and line in the loop commit.
