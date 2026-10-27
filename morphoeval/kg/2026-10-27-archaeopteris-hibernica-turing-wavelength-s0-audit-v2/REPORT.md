# MorphoEval report — audit `2026-10-27-archaeopteris-hibernica-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-archaeopteris-hibernica-turing-wavelength-s0-audit -> 2026-10-27-archaeopteris-hibernica-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Archaeopteris hibernica Famennian epidermal Turing wavelength (31 µm, 370 Ma) (`models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py`) | 90 | 86 | 88 | 90 | 90 | 86 | 90 | 89.3 | 87.5 | 88.9 | 88.5 | **88.5** | faithful |

## Findings and remediation

### Archaeopteris hibernica Famennian epidermal Turing wavelength (31 µm, 370 Ma)  —  overall 88.5 (faithful)

Pass 2 closes the three Pass 1 gaps. Equal-diffusion amplitude ratio is 0.246 after t=140 (was 0.669 at t=24). The seeded pattern is 31.0 µm and the unseeded pattern at the same gamma is 26.57 µm, a finite-amplitude shift of 4.43 µm, now printed on the comparison figure and stored in metrics.json. Dispersion Pearson r = 0.9865, RMSE = 0.0375. Specimen orientation KS = 0.25, radial-spectrum Pearson r = 0.994, radial RMSE = 0.257. GIF CV similarity remains 96.0 at 630x720, 12 frames. gamma sweep stays monotone from 62 µm down to 15.5 µm analytical.

- **Model fidelity to source: 90** — Schnakenberg kinetics, pseudospectral diffusion, and k_c = argmax sigma(k) are implemented and distinguished from the textbook band center. The equal-diffusion control is now integrated long enough for the slowest stable mode to fall to an amplitude ratio of 0.246, and it does not satisfy the Turing inequalities.  
  evidence: `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L87–122; `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L291–307
- **Morphological realism / specimen match: 86** — The NMNH frond and the 31 µm ocellus field are shown together. The 12-frame GIF scores CV similarity 96 against that photograph. Orientation KS is 0.25 and the radial spectra correlate at 0.994. The cellular panel is not a copy of the slab, and the slab does not resolve epidermal cells.  
  evidence: `docs/media/archaeopteris_hibernica_turing_wavelength_s0_comparison.png` L0–0; `docs/media/archaeopteris_hibernica_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — gamma from 0.25x to 4x shortens analytical wavelength 62 to 15.5 µm and noise-selected wavelength 62 to 13.3 µm, both monotone. The Famennian operating point is reported twice: seeded 31.0 µm and unseeded 26.57 µm, so the seed dependence is measured rather than hidden.  
  evidence: `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L310–352; `datasets/archaeopteris_hibernica_turing_wavelength_s0/metrics.json` L0–0
- **Morphometric validation: 90** — Dispersion Pearson r = 0.9865 and RMSE = 0.0375 on Fourier modes 3-13. Specimen segmentation gives coherency 0.671 versus a 0.5 baseline, foreground fraction 0.621 versus 0.4, orientation KS 0.25 versus a 0.05 threshold, and radial RMSE 0.257 versus 0. Radial Pearson of the frond crops is 0.994. None of these value/reference pairs are copies of each other.  
  evidence: `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L232–289; `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L444–468; `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L618–670
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing, Schnakenberg, Meyer-Berthaud, Boyce and Knoll, Beck, and Rothwell. The specimen is the public-domain NMNH photograph, SHA-256 pinned in the reference JSON.  
  evidence: `datasets/archaeopteris_hibernica_turing_wavelength_s0/paper_citations.bib` L1–63; `datasets/archaeopteris_hibernica_turing_wavelength_s0/archaeopteris_hibernica_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 86** — Each modelling step is its own function, and the Lean module proves 6*31 = 186, 21 < 31 < 66, and Dv/Du = 24 with no sorry. The module states that gamma was solved for argmax sigma, not for the band-center formula.  
  evidence: `morphoeval/lean/loops/Loop_3119_archaeopteris_hibernica_turing_wavelength_s0.lean` L14–42; `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L87–112
- **Reproducibility package: 90** — Seed 3119. --self-test now also requires the ablation ratio below 0.35 and a nonzero finite-amplitude shift. Specimen SHA-256 is checked before any run. Invocation is in the dataset README.  
  evidence: `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L728–761; `datasets/archaeopteris_hibernica_turing_wavelength_s0/README.md` L1–40

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Archaeopteris hibernica Famennian epidermal Turing wavelength (31 µm, 370 Ma)
- **spec_fingerprint** (semantic model + inputs): `a4e175a99a098be3c6247b8689ef9fb8b86a807901aee40d5798aca5032374bc`
- **input_dataset_sha256**: `datasets/archaeopteris_hibernica_turing_wavelength_s0/Archaeopteris_hibernica_NMNH.jpg` (`20c480a93bc91bf3…`), `datasets/archaeopteris_hibernica_turing_wavelength_s0/archaeopteris_hibernica_turing_wavelength_s0_reference.json` (`655a409a6f13bf84…`), `datasets/archaeopteris_hibernica_turing_wavelength_s0/paper_citations.bib` (`4912199010e7d045…`)
- `datasets/archaeopteris_hibernica_turing_wavelength_s0/metrics.json` — artifact_sha256 `a231c95f4803295c…`, 2,187 B, 
- `docs/media/archaeopteris_hibernica_turing_wavelength_s0_comparison.png` — artifact_sha256 `75313c23bb6bc7bf…`, 473,788 B, 1512×1008 px
- `docs/media/archaeopteris_hibernica_turing_wavelength_s0_growth.gif` — artifact_sha256 `d4f606fd4e0047da…`, 3,794,246 B, 630×720 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| stripe_wavelength_um | 31.0 um | 31.0 | dominant Fourier ring of the steady activator on the 186 um periodic square *(numerical mode versus the assigned cellular wavelength)* |
| dispersion_pearson_r | 0.9865 1 | 1.0 | Pearson r of measured early-time Fourier growth rates versus linear sigma(k); RMSE reported separately *(value is the fit, reference is the ideal correlation and is not copied from the value)* |
| dispersion_rmse | 0.0375 1/time | 0.0 | RMSE between measured and theoretical growth rates over Fourier modes 3-13 (Turing band) |
| finite_amplitude_wavelength_shift_um | 4.4286 um | 0.0 | seeded dominant wavelength minus the unseeded noise-selected wavelength at the same Famennian gamma *(positive means the seeded 31 um mode is longer than the unseeded pattern)* |
| ablation_amplitude_ratio | 0.2463 1 | 1.0 | negative control: std(u-u0) final/initial with Du=Dv; ratio below 1 means the perturbation decayed *(equal-diffusion control versus the Turing regime)* |
| specimen_leaf_coherency | 0.6707 1 | 0.5 | structure-tensor coherency of the NMNH frond photograph after border-contrast segmentation *(measured on the real specimen image)* |
| specimen_segmentation_foreground_fraction | 0.621 1 | 0.4 | foreground fraction of the museum slab versus its border median |
| specimen_image_orientation_ks | 0.25 1 | 0.05 | KS statistic between 12-bin Sobel orientation histograms of the specimen crop and the GIF last frame *(specimen versus simulation orientation spectra)* |
| specimen_radial_spectrum_rmse | 0.2573 1 | 0.0 | RMSE of log radial FFT profiles, specimen crop versus GIF last frame |
| predecessor_cortex_wavelength_um | 66.0473 um | 31.0 | lambda(h)=31*(h/12.4)^0.62 at Tetraxylopteris cortical thickness 42 um *(reconstructed predecessor, not a direct fossil measurement)* |
| successor_frond_wavelength_um | 21.3597 um | 31.0 | same thickness law at Elkinsia lamina thickness 6.8 um |

## Environment at checkpoint

- python: 3.12
- numpy: 2.5.4
- scipy: 1.18.1

Pass 2. Ablation integrated to t=140 and the seeded-versus-unseeded wavelength shift is now a recorded metric. Parent commit does not yet contain these paths.
