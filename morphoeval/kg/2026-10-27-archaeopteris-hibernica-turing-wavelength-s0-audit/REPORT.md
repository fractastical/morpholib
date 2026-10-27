# MorphoEval report — audit `2026-10-27-archaeopteris-hibernica-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Archaeopteris hibernica Famennian epidermal Turing wavelength (31 µm, 370 Ma) (`models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py`) | 86 | 82 | 80 | 84 | 90 | 84 | 86 | 84.0 | 82.8 | 86.8 | 84.3 | **84.3** | faithful |

## Findings and remediation

### Archaeopteris hibernica Famennian epidermal Turing wavelength (31 µm, 370 Ma)  —  overall 84.3 (faithful)

Pass 1. Schnakenberg kinetics on a periodic 186 µm square. gamma is solved so argmax sigma(k) is 31 µm. Seeded steady state locks that mode (numerical wavelength 31.0 µm). Dispersion Pearson r = 0.9865, RMSE = 0.0375. Equal-diffusion control does not form a pattern (amplitude ratio 0.669) but the decay is shallow. An unseeded run at the same gamma selects 26.6 µm, one Fourier bin off. Gamma sweep from 0.25x to 4x shortens both the analytical and the noise-selected wavelength. The growth GIF is the NMNH frond veiled by the activator (CV similarity 96, 12 frames, 630x720).

- **Model fidelity to source: 86** — The script integrates Schnakenberg kinetics with a pseudospectral step and defines k_c as argmax sigma(k), kept distinct from the textbook band center. gamma is solved for a 31 µm fastest mode. The equal-diffusion control is only a partial decay (ratio 0.669), so the negative control is not yet sharp.  
  evidence: `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L87–122; `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L136–144
- **Morphological realism / specimen match: 82** — The comparison figure places the NMNH frond beside a 31 µm ocellus lattice, and the GIF last frame scores CV similarity 96 against that photograph. The cellular panel is a spot field, not a drawing of the slab, and the photograph cannot show 31 µm structure.  
  evidence: `docs/media/archaeopteris_hibernica_turing_wavelength_s0_comparison.png` L0–0; `docs/media/archaeopteris_hibernica_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 80** — Raising gamma from 0.25x to 4x shortens analytical wavelength from 62 µm to 15.5 µm and the noise-selected wavelength from 62 µm to 13.3 µm. At the Famennian gamma the unseeded run selects 26.6 µm rather than the seeded 31 µm mode, so the operating point is still seed-dependent.  
  evidence: `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L310–352; `datasets/archaeopteris_hibernica_turing_wavelength_s0/metrics.json` L0–0
- **Morphometric validation: 84** — Linear dispersion matches measured growth rates at Pearson r = 0.9865 and RMSE = 0.0375. Specimen segmentation reports coherency 0.671, foreground fraction 0.621, orientation KS 0.25, and radial-spectrum RMSE 0.257 against the GIF. The chronoseries thicknesses are reconstructions, and the ablation ratio is not yet a clear collapse.  
  evidence: `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L232–289; `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L444–468; `datasets/archaeopteris_hibernica_turing_wavelength_s0/metrics.json` L0–0
- **Source and data traceability: 90** — Six Crossref-checked DOIs, a public-domain NMNH photograph with SHA-256 pin, and Wikimedia source URL are in the dataset README and reference JSON. OpenAlex returned no hits for the exact scout query; Crossref was used instead.  
  evidence: `datasets/archaeopteris_hibernica_turing_wavelength_s0/paper_citations.bib` L1–63; `datasets/archaeopteris_hibernica_turing_wavelength_s0/archaeopteris_hibernica_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — Jacobian, dispersion, sweep, ablation, and the thickness law are separate functions. Lean theorems pin 6*31 = 186, the 21 < 31 < 66 chronoseries, and Dv/Du = 24. The band-center formula is not the quantity gamma was solved for, and the module says so.  
  evidence: `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L77–84; `morphoeval/lean/loops/Loop_3119_archaeopteris_hibernica_turing_wavelength_s0.lean` L14–42
- **Reproducibility package: 86** — Seed 3119, --self-test checks r > 0.98, the 31 µm mode, ablation decay, monotone sweep, and the chronoseries order. Specimen bytes are SHA-pinned. Pass 1 has not yet driven the ablation ratio down or recorded the finite-amplitude shift as its own metric.  
  evidence: `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L538–546; `models/2026-delphi10000-archaeopteris-hibernica-turing-wavelength-s0.py` L728–761

Remediation:
- Drive the equal-diffusion ablation ratio below 0.3 with a longer, still-stable control, and record that ratio.
- Quantify the finite-amplitude shift between the seeded 31.0 µm mode and the unseeded 26.6 µm mode in metrics.json and on the comparison figure.
- Keep the specimen KS, radial Pearson, and radial RMSE beside the dispersion goodness-of-fit so the frond comparison is not only the CV scalar.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Archaeopteris hibernica Famennian epidermal Turing wavelength (31 µm, 370 Ma)
- **spec_fingerprint** (semantic model + inputs): `a4e175a99a098be3c6247b8689ef9fb8b86a807901aee40d5798aca5032374bc`
- **input_dataset_sha256**: `datasets/archaeopteris_hibernica_turing_wavelength_s0/Archaeopteris_hibernica_NMNH.jpg` (`20c480a93bc91bf3…`), `datasets/archaeopteris_hibernica_turing_wavelength_s0/archaeopteris_hibernica_turing_wavelength_s0_reference.json` (`655a409a6f13bf84…`), `datasets/archaeopteris_hibernica_turing_wavelength_s0/paper_citations.bib` (`4912199010e7d045…`)
- `datasets/archaeopteris_hibernica_turing_wavelength_s0/metrics.json` — artifact_sha256 `38aa39b2be743e72…`, 2,093 B, 
- `docs/media/archaeopteris_hibernica_turing_wavelength_s0_comparison.png` — artifact_sha256 `a62d6e1c484adc8f…`, 465,507 B, 1512×1008 px
- `docs/media/archaeopteris_hibernica_turing_wavelength_s0_growth.gif` — artifact_sha256 `d4f606fd4e0047da…`, 3,794,246 B, 630×720 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| stripe_wavelength_um | 31.0 um | 31.0 | dominant Fourier ring of the steady activator on the 186 um periodic square *(numerical mode versus the assigned cellular wavelength)* |
| dispersion_pearson_r | 0.9865 1 | 1.0 | Pearson r of measured early-time Fourier growth rates versus linear sigma(k); RMSE reported separately *(value is the fit, reference is the ideal correlation and is not copied from the value)* |
| dispersion_rmse | 0.0375 1/time | 0.0 | RMSE between measured and theoretical growth rates over Fourier modes 3-13 (Turing band) |
| ablation_amplitude_ratio | 0.6689 1 | 1.0 | negative control: std(u-u0) final/initial with Du=Dv; ratio below 1 means the perturbation decayed *(equal-diffusion control versus the Turing regime)* |
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

Pass 1 working tree on slot 3119. New paths are not in this parent commit.
