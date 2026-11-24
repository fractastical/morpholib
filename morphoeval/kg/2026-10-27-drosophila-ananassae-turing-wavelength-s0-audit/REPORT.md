# MorphoEval report — audit `2026-10-27-drosophila-ananassae-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `b527a03f163994a07bd008d5b308a54ec56094a4` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila ananassae Turing wavenumber at 2.1 µm (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py`) | 82 | 80 | 46 | 58 | 86 | 80 | 84 | 70.0 | 71.8 | 83.6 | 74.8 | **74.8** | mostly faithful |

## Findings and remediation

### Drosophila ananassae Turing wavenumber at 2.1 µm (Holocene, 0 Ma)  —  overall 74.8 (mostly faithful)

Pass 1 integrates Schnakenberg kinetics (a=0.11, b=1.15, Dv/Du=19.7) on a 128×96 periodic grid. The fastest linear mode is placed at 2.106 µm by the choice of dx. The saturated FFT wavelength is 2.030 µm. The growth GIF is 12 frames at 960×640 and modulates the public-domain female plate. There is no parameter sweep, no integrated ablation, no specimen-versus-render segmentation, and no Pearson/RMSE/KS fit.

- **Model fidelity to source: 82** — The script integrates the Schnakenberg PDE and selects k_c as argmax Re σ(k), with Du fixed by the onset formula at the design wavenumber. Parameters belong to this 2.1 µm slot. The equal-diffusion check is only on the linear symbol, not on an integrated field.  
  evidence: `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L78–123; `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L151–186
- **Morphological realism / specimen match: 80** — The comparison figure places the public-domain female plate beside the activator and beside the plate modulated by that field. The 12-frame GIF is 960×640. Computer-vision similarity of the last frame to the plate is 94.3. The plate is a 225×112 illustration, so the match is to that plate, not to a 2.1 µm micrograph.  
  evidence: `docs/media/drosophila_ananassae_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_ananassae_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L251–289
- **Parameter expressivity: 46** — Only the nominal (a, b, ratio) point is integrated. The dispersion plot shows σ(k) at that point. Ratio, production rates, and domain scale are not swept, so monotonic control of wavelength is not demonstrated.  
  evidence: `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L40–52; `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L465–470
- **Morphometric validation: 58** — The saturated FFT wavelength is 2.030 µm against the analytic 2.106 µm, so the two numbers are not copies of each other. Plate circularity is 0.136 against a circle null of 1, and aspect ratio is 2.02. Pearson r, RMSE, and KS across a sweep are not computed.  
  evidence: `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L188–231; `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L359–431
- **Source and data traceability: 86** — Eight Crossref-checked DOIs are in the bib and the reference JSON. The plate URL, public-domain license, and SHA-256 are pinned, and verify_dataset_integrity checks them. The OpenAlex query for this slot returned no hits; that is recorded in the scout manifest rather than filled with an unrelated paper.  
  evidence: `datasets/drosophila_ananassae_turing_wavelength_s0/paper_citations.bib` L1–84; `datasets/drosophila_ananassae_turing_wavelength_s0/drosophila_ananassae_turing_wavelength_s0_reference.json` L77–86; `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L66–76
- **Code-to-model traceability: 80** — Jacobian, onset k, fastest mode, and the chronospecies λ² diffusivity scaling are each a named function. The Lean file states the same nanometre wavelengths and the milli-unit h-condition. Comments name Turing 1952 and Schnakenberg 1979; they do not pin equation numbers inside those papers.  
  evidence: `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L1–28; `morphoeval/lean/loops/Loop_3176_drosophila_ananassae_turing_wavelength_s0.lean` L24–40
- **Reproducibility package: 84** — Seed 3176, pinned dataset hashes, and --self-test (wavelength identity, positive growth, chronospecies ratios, plate aspect). The run is deterministic. Lean was not executed because no lean binary is installed.  
  evidence: `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L434–471; `datasets/drosophila_ananassae_turing_wavelength_s0/README.md` L1–29

Remediation:
- Add a parameter-regime sweep (diffusion ratio and a kinetic scale factor) and record how wavelength moves.
- Integrate the equal-diffusion control and a linear-kinetics ablation; do not stop at the sign of σ(k).
- Segment the real plate and the GIF last frame and compare foreground fraction, circularity, aspect, and coherency.
- Report Pearson r, RMSE, and a KS statistic for the wavelength law, with value unequal to the reference.
- Kernel-check the Lean file when a Lean 4 binary is available.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila ananassae Turing wavenumber at 2.1 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `cb48fb2dc3693d0cac7caeab627f7cf256ca041c37d25c1fbbe642b593156101`
- **input_dataset_sha256**: `datasets/drosophila_ananassae_turing_wavelength_s0/drosophila_ananassae_specimen.png` (`e3f07821c201b2b5…`), `datasets/drosophila_ananassae_turing_wavelength_s0/drosophila_ananassae_turing_wavelength_s0_reference.json` (`2768c7dcb983c315…`)
- `docs/media/drosophila_ananassae_turing_wavelength_s0_comparison.png` — artifact_sha256 `4a7d8f8ac9bfbd85…`, 555,209 B, 1200×800 px
- `docs/media/drosophila_ananassae_turing_wavelength_s0_growth.gif` — artifact_sha256 `87c0dd9a019df28b…`, 3,444,176 B, 960×640 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| fft_wavelength_m | 2.0301e-06 m | 2.106e-06 | dominant non-DC Fourier mode of the saturated activator, times dx |
| specimen_contour_circularity | 0.136 1 | 1.0 | 4πA/P² on the border-contrast mask of the Wikimedia plate |
| specimen_aspect_ratio | 2.018 1 | 1.0 | bounding-box width/height of the plate foreground |
| wavelength_ratio_americana_over_s0 | 65.8594 1 | 1.0 | 139 µm predecessor / 2.106 µm detail wavelength |
| growth_gif_frame_count | 12 frames | 6 | PIL n_frames of the rendered GIF |
| growth_gif_min_resolution_px | 640 px | 340 | min(width, height) of the GIF |

## Environment at checkpoint


Pass 1 sources only. Lean binary was not on PATH, so the four theorems were not kernel-executed. publisher_orcid is the creator URI already used on this repository's MorphoEval nanopubs. ~/.nanopub/profile.yml is absent, so these nanopubs are not signed.
