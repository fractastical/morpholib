# MorphoEval report — audit `2026-10-27-apis-mellifera-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `6fefd6a0` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Apis mellifera Schnakenberg wavelength at 11.2 µm (Pass 1 raw field) (`models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py`) | 88 | 42 | 78 | 82 | 90 | 84 | 88 | 84.7 | 57.0 | 87.6 | 74.5 | **74.5** | mostly faithful |

## Findings and remediation

### Apis mellifera Schnakenberg wavelength at 11.2 µm (Pass 1 raw field)  —  overall 74.5 (mostly faithful)

Discrete dispersion peaks at 11.20 µm (k_c = 5.612e5 m^-1, log10 = -4.951) with Pearson r = 1.000 between Fourier-mode growth and the discrete symbol (RMSE 7.5e-5). The nonlinear lattice wavelength is 11.00 µm. Equal diffusion gives σ_max = -0.129. Pass 1 publishes only the raw lattice, which does not resemble the worker. CV similarity of that GIF to the Severns photograph is 91.4 and is not counted as a specimen match.

- **Model fidelity to source: 88** — The code integrates Schnakenberg kinetics and scores modes against the discrete Laplacian symbol of the Turing dispersion. γ is solved for the assigned wavelength rather than taken from a honeybee fit, which is disclosed.  
  evidence: `models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py` L84–120; `models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py` L130–161
- **Morphological realism / specimen match: 42** — The Pass-1 GIF is an orange spot lattice on a dark field. Side by side with the Severns worker it is not the animal. Canonical CV similarity is 91.4 because both images are warm and textured; that score is not a morphological match.  
  evidence: `datasets/apis_mellifera_turing_wavelength_s0/pass1_raw_field.gif` L0–0; `models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py` L367–383
- **Parameter expressivity: 78** — γ sweep and the d = 1 ablation run inside the script and the self-test, but the Pass-1 figure is only the raw field, so a reader of that figure cannot see the monotone wavelength curve.  
  evidence: `models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py` L244–267; `datasets/apis_mellifera_turing_wavelength_s0/pass1_raw_comparison.png` L0–0
- **Morphometric validation: 82** — Dispersion Pearson r = 0.99999999, RMSE = 7.5e-5, nonlinear wavelength 11.00 µm versus 11.2 µm. The 8.7 µm rhabdom length is reported with a 0.287 gap, not zeroed. Specimen segmentation is computed but not yet the published comparison.  
  evidence: `models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py` L183–241; `datasets/apis_mellifera_turing_wavelength_s0/apis_mellifera_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing, Schnakenberg, Ribi, and Camazine. Specimen SHA-256 is pinned. OpenAlex returned 429; Crossref and Wikimedia were used instead.  
  evidence: `datasets/apis_mellifera_turing_wavelength_s0/paper_citations.bib` L1–68; `datasets/apis_mellifera_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 84** — Dispersion, wavenumber, sweep, and ablation are marked and cited. The Lean file encodes the d = 20 threshold, the ablation, the 12-mode window, the comb gap, and the monotone sweep with no sorry.  
  evidence: `morphoeval/lean/loops/Loop_3115_apis_mellifera_turing_wavelength_s0.lean` L12–49; `models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py` L739–770
- **Reproducibility package: 88** — Seed 3115, --self-test, and a metrics JSON. Pass 1 does not yet write the published habitus GIF as the fitness target.  
  evidence: `models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py` L739–783

Remediation:
- Publish the habitus composite. The raw lattice is not a specimen match even though its CV similarity is 91.4.
- Put the γ sweep and the equal-diffusion σ_max on the comparison figure.
- Segment the Severns photograph (circularity, aspect, coherency, orientation KS) and keep the 0.287 gap versus the 8.7 µm rhabdom explicit.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Apis mellifera Schnakenberg wavelength at 11.2 µm (Pass 1 raw field)
- **spec_fingerprint** (semantic model + inputs): `ead8c2daa6443b4c6d8a0168cde643f354ab15fd7ffa8abb51b4ba6686c5ad09`
- **input_dataset_sha256**: `datasets/apis_mellifera_turing_wavelength_s0/apis_mellifera_specimen.jpg` (`7dcd58fa1f1da3ad…`), `datasets/apis_mellifera_turing_wavelength_s0/apis_mellifera_turing_wavelength_s0_reference.json` (`5f5b563b6c87662d…`)
- `datasets/apis_mellifera_turing_wavelength_s0/apis_mellifera_turing_wavelength_s0_metrics.json` — artifact_sha256 `8dd97b6e51952ea8…`, 2,588 B, 
- `datasets/apis_mellifera_turing_wavelength_s0/pass1_raw_comparison.png` — artifact_sha256 `8e54b4652ee0b4e4…`, 84,339 B, 1152×528 px
- `datasets/apis_mellifera_turing_wavelength_s0/pass1_raw_field.gif` — artifact_sha256 `fc1e753565e8193b…`, 1,030,331 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion pearson r (FD vs discrete symbol) | 1.0 correlation | 0.98 | polyfit log-amplitude of linearized Fourier modes vs discrete Laplacian symbol σ(k) |
| dispersion RMSE | 7.5e-05 per time | 0.0 | root mean square of σ_numeric − σ_analytic over DFT wavenumbers |
| selected wavelength lambda_c | 11.1954 µm | 11.2 | argmax of discrete Schnakenberg dispersion; γ solved for the slot |
| nonlinear activator wavelength | 10.9956 µm | 11.2 | radial FFT peak of the saturated 2D activator |
| specimen contour circularity | 0.1631 4πA/P² | 1.0 | Otsu-like border-contrast mask on the Severns photograph, 160×120 |
| specimen contour aspect ratio | 1.339 width/height | 1.0 | bounding box of the specimen foreground mask |
| specimen foreground fraction | 0.7102 fraction | 0.5 | border-median contrast mask |
| specimen structure-tensor coherency | 0.64 index | 0.0 | mean (λ1−λ2)/(λ1+λ2) of the structure tensor on the foreground |
| specimen abdominal dark band count | 9 count | 4 | peaks of the dark-cuticle fraction on the left-body ROI; legs and shadow inflate the count above a clean tergite census |
| KS edge-orientation specimen vs last frame | 0.0758 KS statistic | 0.0 | ks_2samp on Sobel orientation samples of the photograph and the GIF last frame |
| ablation sigma_max at equal diffusion | -0.1287 per time | 0.0 | max discrete σ(k) at D_v = D_u |
| relative gap versus Ribi rhabdom length | 0.2868 fraction | 0.0 | (λ_c − 8.7 µm) / 8.7 µm |

## Environment at checkpoint


Pass 1 files are in 6fefd6a0.
