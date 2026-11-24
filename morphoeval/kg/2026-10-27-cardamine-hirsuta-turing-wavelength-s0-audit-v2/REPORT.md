# MorphoEval report — audit `2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit -> 2026-10-27-cardamine-hirsuta-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength on Cardamine hirsuta at 154 µm (Pass 2) (`models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py`) | 88 | 86 | 90 | 88 | 92 | 86 | 90 | 88.7 | 86.8 | 89.6 | 88.2 | **88.2** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength on Cardamine hirsuta at 154 µm (Pass 2)  —  overall 88.2 (faithful)

Pass 2 keeps the same Schnakenberg point (a=0.10, b=0.90, D_u=0.04, D_v=0.42, lambda_c=154.1 µm, age 0 Ma, chronoseries 1/1). Measured record: dispersion Pearson r = 0.999896, RMSE = 0.001564, nonlinear spacing 173.362 µm on both seeds, ablation sigma_max = -0.1016 with the control standard deviation falling, sweep wavelengths 163.9, 173.6, 188.0, 201.0, 217.4 µm (monotonic), specimen foreground fraction 0.2825, bounding-box aspect 0.6048, contour circularity 0.0926, orientation KS 0.0156, radial-FFT Pearson r = 0.9948, GIF 12 frames at 720x540, CV similarity 96.0. The 12% nonlinear shift versus 154.1 µm stays in the record.

- **Model fidelity to source: 88** — Pass 2 keeps the implemented Jacobian, wavenumber law, Neumann integrator, and the integer Lean checks. The ablation panel shows the equal-diffusion field staying flat on the same color scale as the patterned field, which is the negative control the kinetics predict. The pigment label remains a reconstruction, so the score stays below 95.  
  evidence: `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L103–134; `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L241–266; `morphoeval/lean/loops/Loop_3140_cardamine_hirsuta_turing_wavelength_s0.lean` L13–28
- **Morphological realism / specimen match: 86** — Last-frame CV similarity against the Commons photograph is 96.0 (orientation, radial FFT, coherency, occupancy), with radial-FFT Pearson r = 0.9948. The comparison figure shows the same plant. The score stays under 90 because the ocellus texture is a model modulation, not a micrograph at 154 µm.  
  evidence: `docs/media/cardamine_hirsuta_turing_wavelength_s0_growth.gif` L0–0; `docs/media/cardamine_hirsuta_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L338–399
- **Parameter expressivity: 90** — Selected wavelength increases at every step of D_v/D_u in {10.5, 13.75, 20, 27.5, 40}: 163.9, 173.6, 188.0, 201.0, 217.4 µm. The self-test asserts that sequence is strictly increasing. The equal-diffusion point is a separate switch that kills the pattern rather than a point on that curve.  
  evidence: `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L219–239; `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L704–704
- **Morphometric validation: 88** — Dispersion-band Pearson r = 0.999896 against reference gate 0.98, RMSE = 0.001564 against a 0.05 tolerance, nonlinear spacing 173.362 µm against the 154.1 µm law (values differ on purpose), ablation sigma_max = -0.1016 against 0, and the specimen mask metrics differ from their priors. Orientation KS is about 0.016, not zero. n = 1 photograph and two seeds.  
  evidence: `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L551–664; `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L268–293; `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L699–710
- **Source and data traceability: 92** — DOIs were checked on Crossref while scouting. The specimen file is the Commons public-domain JPEG, SHA-256 1c268fecb3660b310d99f688a13a679c347aabc4e2522eace472f8c985bf8510, distinct from the species-gallery copy. Scale 154 µm and age 0 Ma are in the reference JSON, the README, and the figure title.  
  evidence: `datasets/cardamine_hirsuta_turing_wavelength_s0/scout_manifest.json` L1–30; `datasets/cardamine_hirsuta_turing_wavelength_s0/paper_citations.bib` L1–70; `datasets/cardamine_hirsuta_turing_wavelength_s0/cardamine_hirsuta_turing_wavelength_s0_reference.json` L8–30
- **Code-to-model traceability: 86** — Each of k_c, the instability inequality, and the eight-wavelength Holocene cutoff has a Lean theorem and a Python function. The math-kg YAML names those declarations. Pass 2 does not add a second formalization.  
  evidence: `morphoeval/math_kg_loops/delphi10000-cardamine-hirsuta-turing-wavelength-s0.yaml` L24–55; `morphoeval/lean/loops/Loop_3140_cardamine_hirsuta_turing_wavelength_s0.lean` L32–46
- **Reproducibility package: 90** — Deterministic seeds, pinned input hashes, and a self-test that failed closed on the numerical gates before this audit was scored. Re-running --self-test regenerates the PNG and the GIF.  
  evidence: `models/2026-delphi10000-cardamine-hirsuta-turing-wavelength-s0.py` L78–84; `datasets/cardamine_hirsuta_turing_wavelength_s0/README.md` L14–22

Remediation:
- {'id': 'R1', 'item': 'D_v sweep monotonicity is now an asserted self-test and a measurement row.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Equal-diffusion ablation sigma_max = -0.1016 is scored, and the flat field is in the comparison figure.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Specimen foreground fraction, aspect, circularity, and coherency are measurement rows with priors that differ from the measured values.', 'status': 'resolved'}
- {'id': 'R4', 'item': 'Pearson r = 0.999896, RMSE = 0.001564, and the orientation KS statistic are measurement rows.', 'status': 'resolved'}
- {'id': 'R5', 'item': 'Independent second assessor is still absent. Pass 2 remains a single-agent score.', 'status': 'open'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=0.5]** `independent_assessor` → `specimen-match`: A second assessor has not scored this loop.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength on Cardamine hirsuta at 154 µm (Pass 2)
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
