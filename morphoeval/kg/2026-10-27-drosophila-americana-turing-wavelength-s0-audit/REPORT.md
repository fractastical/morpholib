# MorphoEval report — audit `2026-10-27-drosophila-americana-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila americana detail-scale Turing wavelength (2.117 µm, 0 Ma) (`models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py`) | 88 | 76 | 82 | 84 | 90 | 85 | 86 | 86.0 | 79.0 | 87.1 | 83.5 | **83.5** | faithful |

## Findings and remediation

### Drosophila americana detail-scale Turing wavelength (2.117 µm, 0 Ma)  —  overall 83.5 (faithful)

Pass 1. Schnakenberg activator-inhibitor at diffusion ratio r = 10, with the Turing onset wavenumber k_c and the maximum-growth wavelength k_sigma calibrated so the expressed scale is 2.117 µm (2.1 µm, log10 = -5.674) at 0 Ma. D_u sweep Pearson r = 0.989 against lambda(k_sigma), RMSE 0.41 px. Equal-diffusion ablation amplitude ratio is ~0. Detail-scale wavelengths are 1.958 µm (affinis), 2.117 µm (americana), 1.900 µm (ananassae), buffered by body-length exponent 0.35 relative to the 96/139/85 µm primordium siblings. Species-card CV similarity of the growth GIF is 79.0 (12 frames, 640×520). Orientation KS on 12 histogram bins is 0.5 and is not yet a pixel-sample test.

- **Model fidelity to source: 88** — onset_kc2 and sigma_plus implement the published critical-wavenumber formula and the larger Jacobian eigenvalue. Kinetics are the cubic activator-inhibitor form. r = 10 is a reconstruction chosen so k_sigma stays near k_c, not a value tabulated by Turing or Wittkopp.  
  evidence: `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L80–134; `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L147–203
- **Morphological realism / specimen match: 76** — The growth GIF last frame scores CV similarity 79.0 against the species card (12 frames, 640×520, frame difference 0.12). The card is a spiral pigment image, not an ocellar micrograph, and the simulation is a stripe field, so the match is textural rather than anatomical. The GIF caption uses λ and µm, which the default bitmap font does not draw. Orientation KS on 12 bins is 0.5.  
  evidence: `docs/media/drosophila_americana_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_americana_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L397–425
- **Parameter expressivity: 82** — Five D_u scales move the measured wavelength with Pearson r = 0.989. Equal diffusion collapses the amplitude. The chronoseries is a D_u rescaling at fixed r. There is not yet a near-onset bracket showing k_sigma approach k_c, and (a, b) are not swept.  
  evidence: `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L64–65; `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L264–317
- **Morphometric validation: 84** — Pearson r = 0.989 and RMSE = 0.41 px against lambda(k_sigma). Ablation amplitude ratio is about 0. The detail-scale americana/affinis ratio is below the primordium-scale sibling ratio. The orientation KS is reported but is computed on histogram bins rather than pixel samples, so it is a weak distributional test. n = 1 species card.  
  evidence: `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L205–229; `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L354–381
- **Source and data traceability: 90** — Crossref-verified DOIs for Turing, Gierer & Meinhardt, Murray, Wittkopp 2009, Clark 2007, and O'Grady & DeSalle 2018. The species card SHA-256 is pinned. The catalog Cell DOI is recorded as a 404 rather than cited. Commons had no D. americana photograph.  
  evidence: `datasets/drosophila_americana_turing_wavelength_s0/paper_citations.bib` L1–69; `datasets/drosophila_americana_turing_wavelength_s0/drosophila_americana_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 85** — k_c, the Jacobian, and the chronoseries formula are named in the module docstring and in the Lean proofs (six theorems, no sorry). Self-test checks the Lean nanometre integers 2117, 1958, and 1900. The Fourier scheme is commented as matching the dispersion operator.  
  evidence: `morphoeval/lean/loops/Loop_3175_drosophila_americana_turing_wavelength_s0.lean` L16–48; `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L1–28
- **Reproducibility package: 86** — Seed 3175, --self-test, and a deterministic Fourier step. Self-test passed (r = 0.989, ablation ratio ~ 0). Dependencies are the repo scientific stack (numpy, scipy, pillow, matplotlib). No second-assessor rerun and no pinned container.  
  evidence: `models/2026-delphi10000-drosophila-americana-turing-wavelength-s0.py` L614–642

Remediation:
- R1: GIF caption uses λ and µm, which the default PIL font does not draw. Replace with ASCII.
- R2: Orientation KS is ks_2samp on 12 histogram bins (statistic 0.5). Recompute on pixel orientation samples and report that distance.
- R3: Add a near-onset run (r closer to r_c ≈ 8.57) showing the fractional gap (lambda_sigma - lambda_c) / lambda_c shrinks relative to r = 10.
- R4: Single species card and a single assessor. The card is not an ocellar micrograph; do not treat a higher score as anatomical validation.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila americana detail-scale Turing wavelength (2.117 µm, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `2f4962a2f9aad429899235fd1af99851f464c94cb46349fdccbc4240f10a01bb`
- **input_dataset_sha256**: `datasets/drosophila_americana_turing_wavelength_s0/Drosophila_americana_specimen.jpg` (`c14927be1f313896…`), `datasets/drosophila_americana_turing_wavelength_s0/drosophila_americana_turing_wavelength_s0_reference.json` (`54ba0c292c786355…`), `datasets/drosophila_americana_turing_wavelength_s0/paper_citations.bib` (`1770ea1c999fdf81…`)
- `docs/media/drosophila_americana_turing_wavelength_s0_comparison.png` — artifact_sha256 `2de831d22be834c6…`, 409,578 B, 1248×984 px
- `docs/media/drosophila_americana_turing_wavelength_s0_growth.gif` — artifact_sha256 `9c01fec6a77bcd33…`, 2,104,603 B, 640×520 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| D. americana expressed stripe wavelength | 2.0751 µm | 2.117 | radial FFT peak on the D_u-sweep grid at scale 1, converted with the ruler on which lambda(k_sigma) = 2.117 µm *(value is the FFT measurement; reference is the assigned scale, not a copy of the measurement)* |
| analytical versus simulated wavelength Pearson r | 0.9892 Pearson r | 0.98 | pearsonr of FFT wavelengths against lambda(k_sigma) across the D_u sweep |
| wavelength RMSE against maximum-growth prediction | 0.4056 px | 1.0299 | RMSE of measured radial-FFT wavelength versus sigma-peak wavelength over the D_u sweep *(KS/CV companion: orientation KS is reported as its own measurement)* |
| specimen pigment-segment fraction | 0.1795 area fraction | 0.8753 | Otsu threshold on the species-card grayscale; fraction darker than the threshold |
| specimen versus simulation orientation KS statistic | 0.5 KS distance | 0.0 | ks_2samp on 12-bin Sobel orientation histograms of the species card and the simulated field |
| equal-diffusion ablation amplitude ratio | 0.0 std ratio | 1.0 | final activator std at r = 1 divided by final std at r = 10, same D_u and seed family *(negative control; discriminant at r = 1 is negative)* |
| detail-scale versus primordium-scale americana/affinis wavelength ratio | 1.0812 ratio | 1.4457 | lambda_ame/lambda_aff at beta = 0.35 body-length buffering, versus sibling primordium wavelengths |

## Environment at checkpoint


Pass 1 working tree. New slot files are not in commit 0c76d15d. publisher_orcid is the dct:creator already used on this repo's MorphoEval nanopubs; this pass does not sign.
