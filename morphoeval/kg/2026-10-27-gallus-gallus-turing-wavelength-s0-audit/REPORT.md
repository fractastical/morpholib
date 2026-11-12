# MorphoEval report — audit `2026-10-27-gallus-gallus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `8be940bbc4e23235f2785d66e06db9da83a65aee` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Gallus gallus Holocene Turing pigment-ocellus wavelength, 319.2 µm (pass 1) (`models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py`) | 76 | 74 | 48 | 62 | 84 | 80 | 82 | 66.7 | 69.5 | 82.1 | 72.4 | **72.4** | mostly faithful |

## Findings and remediation

### Gallus gallus Holocene Turing pigment-ocellus wavelength, 319.2 µm (pass 1)  —  overall 72.4 (mostly faithful)

Pass 1 solves the Schnakenberg onset wavenumber so λ_c = 319.2 µm exactly at age 0 Ma (Gallus chronoseries 1/1, no predecessor or successor). A single radial harmonic plus a quadratic trend matches the gallery ocellus plate at Pearson r = 0.985 against a trend-only null of lower correlation. The painted scale-bar reader is wrong on this pass (it spans stray bright pixels and reports ~10 µm). There is no Dv sweep, no integrated equal-diffusion ablation, and no computer-vision segmentation of the plate.

- **Model fidelity to source: 76** — The onset formula k_c^2 = (Dv fu + Du gv) / (2 Du Dv) is implemented and γ is solved so λ_c is 319.2 µm. Pass 1 does not integrate the PDE, so the selected wavelength is analytical rather than emergent from a time-stepper.  
  evidence: `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` L62–104; `morphoeval/lean/loops/Loop_3228_gallus_gallus_turing_wavelength_s0.lean` L47–51
- **Morphological realism / specimen match: 74** — The 800×600, 12-frame GIF is a centered ocellus in the plate's purple grain and scores CV similarity 90.5 against the gallery plate. The match is a radial fit to that same plate, so it is not an independent feather.  
  evidence: `docs/media/gallus_gallus_turing_wavelength_s0_growth.gif` L0–0; `docs/media/gallus_gallus_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 48** — a, b, Du, Dv and the solved γ are defined, and the equal-diffusion margin is zero, but pass 1 never sweeps a parameter across a range or shows the resulting change in λ_c or pattern amplitude.  
  evidence: `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` L52–82
- **Morphometric validation: 62** — In-sample Pearson r = 0.985 versus a trend-only null, and the model wavelength 319.2 µm is reported against the (incorrectly read) painted bar. No held-out split, no KS statistic, and no CV segmentation of the plate. The scale-bar width is wrong.  
  evidence: `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` L143–171; `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` L362–414; `datasets/gallus_gallus_turing_wavelength_s0/gallus_gallus_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 84** — Turing 1952, Prum & Williamson 2002, Harris et al. 2005, and Mou et al. 2011 are cited with DOIs. The ocellus plate SHA-256 and the CC BY 4.0 junglefowl voucher are in the reference JSON. The plate JPEG itself matches a datasets/*/*_specimen.jpg gitignore and is not in commit 8be940bb.  
  evidence: `datasets/gallus_gallus_turing_wavelength_s0/paper_citations.bib` L1–43; `datasets/gallus_gallus_turing_wavelength_s0/gallus_gallus_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 80** — gamma_for_wavelength and critical_wavelength_um are the onset formula. Lean theorems pin the onset gap, the singleton chronoseries, and the six-wavelength patch. The radial renderer is a weakly nonlinear reduction, not a line-by-line transcription of Prum & Williamson's growing-feather geometry.  
  evidence: `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` L84–104; `morphoeval/math_kg_loops/delphi10000-gallus-gallus-turing-wavelength-s0.yaml` L20–60
- **Reproducibility package: 82** — --self-test checks λ_c, age 0 Ma, the singleton series, R > 1, and the equal-diffusion margin. Seed 3228 fixes the grain. The known-good command is in the dataset README.  
  evidence: `models/2026-delphi10000-gallus-gallus-turing-wavelength-s0.py` L416–431; `datasets/gallus_gallus_turing_wavelength_s0/README.md` L28–34

Remediation:
- {'id': 'R1', 'item': 'Sweep inhibitor diffusivity Dv at fixed γ and show λ_c(Dv) and the Turing margin R(Dv), including the stable side of onset.'}
- {'id': 'R2', 'item': 'Integrate the Schnakenberg PDE: the critical mode must grow at Dv = 16 and decay under the equal-diffusion ablation Dv = Du.'}
- {'id': 'R3', 'item': 'Segment the real ocellus plate (foreground fraction, structure-tensor coherency, circularity) and compare those measurements to the rendered frame.'}
- {'id': 'R4', 'item': 'Replace the single in-sample harmonic with a held-out multi-harmonic goodness-of-fit (Pearson r, RMSE, KS) and fix the painted scale-bar measurement.'}
- {'id': 'R5', 'item': 'Force-track gallus_gallus_ocellus_specimen.jpg so the SHA-256 pin is in the repository despite the specimen-jpg gitignore.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Gallus gallus Holocene Turing pigment-ocellus wavelength, 319.2 µm (pass 1)
- **spec_fingerprint** (semantic model + inputs): `d79e485299e3508a55f35819cf41b353933badb10f650ee2ade00ae23887b314`
- **input_dataset_sha256**: `datasets/gallus_gallus_turing_wavelength_s0/gallus_gallus_ocellus_specimen.jpg` (`7e7d58db6bb564eb…`), `datasets/gallus_gallus_turing_wavelength_s0/gallus_gallus_turing_wavelength_s0_reference.json` (`5589d95bf0b106f7…`)
- `datasets/gallus_gallus_turing_wavelength_s0/gallus_gallus_turing_wavelength_s0_metrics.json` — artifact_sha256 `4b42bdb2c4a5b1a9…`, 724 B, 
- `docs/media/gallus_gallus_turing_wavelength_s0_comparison.png` — artifact_sha256 `a9d7854ae24967e5…`, 436,094 B, 1500×504 px
- `docs/media/gallus_gallus_turing_wavelength_s0_growth.gif` — artifact_sha256 `851a1b8a33379170…`, 5,544,731 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing onset wavelength lambda_c | 319.2 µm | 10.1955 | lambda_c = 2π / k_c with k_c^2 = (Dv fu + Du gv) / (2 Du Dv), γ solved for 319.2 µm *(value is the Holocene model wavelength; reference is the inconsistent painted bar)* |
| specimen radial profile Pearson r (single harmonic) | 0.985 correlation | 0.7551 | least squares of exp(-r/400)·cos(k r + φ) plus quadratic trend against azimuthal mean *(Pass 1 in-sample r. Held-out multi-harmonic fit, RMSE detail, and KS are pass-2 remediations.)* |
| specimen radial profile RMSE | 5.6025 gray levels | 32.446 | root mean square residual of the single-harmonic reconstruction *(single plate, n = 1)* |
| Holocene age | 0.0 Ma | 124.6 | Gallus chronoseries 1/1 assigned horizon *(predecessor and successor chronospecies are absent, so Δage inside the series is undefined rather than 124.6)* |

## Environment at checkpoint

- python: 3.12.3
- note: numpy, scipy, pillow, matplotlib, pyyaml, rdflib installed for this run

Pass 1. Onset wavenumber and single-harmonic plate comparison only.
