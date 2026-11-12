# MorphoEval report — audit `2026-10-27-apis-mellifera-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `6fefd6a0` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-apis-mellifera-turing-wavelength-s0-audit -> 2026-10-27-apis-mellifera-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Apis mellifera Schnakenberg wavelength at 11.2 µm (Pass 2 habitus composite) (`models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py`) | 90 | 80 | 88 | 84 | 92 | 86 | 90 | 89.3 | 81.5 | 89.6 | 86.3 | **86.3** | faithful |

## Findings and remediation

### Apis mellifera Schnakenberg wavelength at 11.2 µm (Pass 2 habitus composite)  —  overall 86.3 (faithful)

Same dispersion as Pass 1 (λ_c = 11.20 µm, Pearson r = 1.000, nonlinear wavelength 11.00 µm, ablation σ_max = -0.129, std ratio 272). The published GIF is 12 frames at 640×480: a blurred early field resolves to the Severns worker, with the lattice deposited on cuticle pixels and a filmstrip of the raw field in the bottom margin. Canonical CV similarity of the last frame is 96.0. Specimen circularity is 0.163 against a circle null of 1. Weighted rubric score 86.3.

- **Model fidelity to source: 90** — Pass 2 uses the same Schnakenberg integrator and discrete dispersion. The ablation panel now shows σ_max(d=20) above zero and σ_max(d=1) below zero, which is the Turing condition rather than a second pattern family.  
  evidence: `models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py` L84–161; `models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py` L253–267
- **Morphological realism / specimen match: 80** — The last GIF frame is the Severns worker, with a weak cuticle modulation and the lattice kept in a bottom filmstrip outside the fitness crop. The comparison figure places the photograph, the lattice, and that frame side by side. The lattice itself is not visible as anatomy in the photograph, so this is not a higher score.  
  evidence: `docs/media/apis_mellifera_turing_wavelength_s0_growth.gif` L0–0; `docs/media/apis_mellifera_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py` L393–440
- **Parameter expressivity: 88** — The comparison figure shows λ_c falling from about 14.4 µm to 8.4 µm as γ rises, with the slot and the 8.7 µm rhabdom length marked, and σ_max changing sign when d drops from 20 to 1.  
  evidence: `models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py` L244–251; `docs/media/apis_mellifera_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 84** — Pearson r = 0.99999999 and RMSE = 7.5e-5 on the real-eigenvalue band. Nonlinear wavelength 11.00 versus 11.2 µm. Specimen circularity 0.163 versus 1, aspect 1.339 versus 1, foreground 0.710 versus 0.5, KS orientation 0.076 versus 0. Ribi gap 0.287 is not fitted away. The dark-peak counter returns 9 against a typical visible band count of 4 because legs and shadow are included.  
  evidence: `models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py` L269–342; `datasets/apis_mellifera_turing_wavelength_s0/apis_mellifera_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 92** — DOIs were checked on Crossref. The photograph is public domain (John Severns), SHA-256 7dcd58fa1f1da3ad6a7766b77998d900687de0e81f54a86cccb55d3e97d35ab4. OpenAlex 429 is recorded rather than invented.  
  evidence: `datasets/apis_mellifera_turing_wavelength_s0/paper_citations.bib` L1–68; `datasets/apis_mellifera_turing_wavelength_s0/apis_mellifera_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 86** — Each scored block is tied to Turing 1952, Schnakenberg 1979, or Murray 1981, and the Lean theorems restate the integer forms of the threshold, the ablation, the 12-mode window, the comb gap, and the monotone sweep.  
  evidence: `morphoeval/lean/loops/Loop_3115_apis_mellifera_turing_wavelength_s0.lean` L12–49; `morphoeval/math_kg_loops/delphi10000-apis-mellifera-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 90** — python3 models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py --self-test passes. Seed, grid, dt, and steps are constants. Both GIFs are 12 frames at 640×480.  
  evidence: `models/2026-delphi10000-apis-mellifera-turing-wavelength-s0.py` L739–783

Remediation:
- Still one assessor. A second provider has not scored this slot.
- The dark-peak counter returns 9, not a clean tergite count. It needs a body mask that excludes legs.
- One photograph. The 11.2 µm gap versus the 8.7 µm rhabdom is unexplained by a new measurement.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Apis mellifera Schnakenberg wavelength at 11.2 µm (Pass 2 habitus composite)
- **spec_fingerprint** (semantic model + inputs): `ead8c2daa6443b4c6d8a0168cde643f354ab15fd7ffa8abb51b4ba6686c5ad09`
- **input_dataset_sha256**: `datasets/apis_mellifera_turing_wavelength_s0/apis_mellifera_specimen.jpg` (`7dcd58fa1f1da3ad…`), `datasets/apis_mellifera_turing_wavelength_s0/apis_mellifera_turing_wavelength_s0_reference.json` (`5f5b563b6c87662d…`)
- `datasets/apis_mellifera_turing_wavelength_s0/apis_mellifera_turing_wavelength_s0_metrics.json` — artifact_sha256 `8dd97b6e51952ea8…`, 2,588 B, 
- `docs/media/apis_mellifera_turing_wavelength_s0_comparison.png` — artifact_sha256 `a1f6cc2de550e3f4…`, 661,017 B, 1584×984 px
- `docs/media/apis_mellifera_turing_wavelength_s0_growth.gif` — artifact_sha256 `b2d911982eb30b37…`, 1,619,085 B, 640×480 px, 12 frames

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


Pass 2 evidence paths exist at 6fefd6a0.
