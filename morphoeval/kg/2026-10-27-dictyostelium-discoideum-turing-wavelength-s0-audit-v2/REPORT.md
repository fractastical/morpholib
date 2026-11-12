# MorphoEval report — audit `2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit -> 2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dictyostelium discoideum Schnakenberg Turing wavelength at 2.304 µm (Holocene, 0 Ma) (`models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py`) | 90 | 76 | 88 | 92 | 92 | 86 | 90 | 89.3 | 82.0 | 89.6 | 86.5 | **86.5** | faithful |

## Findings and remediation

### Dictyostelium discoideum Schnakenberg Turing wavelength at 2.304 µm (Holocene, 0 Ma)  —  overall 86.5 (faithful)

Pass 2 keeps the same Schnakenberg integrator. The discrete-Laplacian peak is 2.303721 µm against the catalog coordinate 2.304 µm. On 23 unstable Fourier modes, measured growth versus linear theory has Pearson r = 0.9999999973, RMSE = 3.668e-05 /s, and KS = 0.0870. Raising δ = Dv/Du from 14 to 64, with Du fixed, moves λ from 2.053 µm to 2.847 µm and is monotonic; δ <= 12 has σ_max < 0. Equal diffusion drops var(u) by about 3.8e-05 of its initial value while δ=24 raises var(u) by about 247×. The sorocarp mask has foreground fraction 0.181, circularity 0.096, coherency 0.580, none equal to the reference placeholders. GIF CV similarity against that photograph is 83.2 (12 frames, 960×720). The photograph remains a fruiting body, not a 2.3 µm cortical frame.

- **Model fidelity to source: 90** — Schnakenberg kinetics, periodic five-point Laplacian, and the unstable-eigenvector initial condition reproduce σ(k) with Pearson r = 0.9999999973 on 23 modes (RMSE 3.67e-05 /s). Ten points are withheld because the biological cortical waves are excitable (Taniguchi 2013), and Schnakenberg is the reconstructed carrier of k_c rather than that phase model.  
  evidence: `models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py` L84–123; `models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py` L166–233; `datasets/dictyostelium_discoideum_turing_wavelength_s0/dictyostelium_discoideum_turing_wavelength_s0_metrics.json` L0–0
- **Morphological realism / specimen match: 76** — Side by side, the public-domain sorocarp and the cortical spot field are different objects, and the border-contrast mask is a coarse 160×120 segmentation (circularity 0.096), not a hand trace of two stalks. The GIF last frame still shares Sobel, radial-FFT, and coherency structure with the photograph at cv_sim_similarity 83.2, which is a texture match rather than anatomical identity. The score credits that measured resemblance and the labeled scale gap.  
  evidence: `docs/media/dictyostelium_discoideum_turing_wavelength_s0_comparison.png` L0–0; `docs/media/dictyostelium_discoideum_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py` L259–316
- **Parameter expressivity: 88** — With Du fixed, δ from 14 to 64 raises λ_peak monotonically from 2.053 µm to 2.847 µm and raises σ_max from 0.029 to 0.374 /s. δ = 1, 8, and 12 stay stable (σ_max < 0). The reference δ=24 sits on that curve at 2.304 µm. Domain length is not swept visually; the integer mode count (5 on the cell, 0 at 0.20 µm) is analytic.  
  evidence: `models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py` L235–257; `docs/media/dictyostelium_discoideum_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 92** — Dispersion Pearson r = 0.9999999973 against a reference of 1 (not equal), RMSE = 3.668e-05 /s against a 0.05 screening tolerance, KS = 0.087. Specimen foreground 0.181, circularity 0.096, and coherency 0.580 are measured on the photograph and stored against different reference placeholders. Spot count 32 is not the lattice expectation 25. Orientation KS between photograph and GIF is 0.059. Pixel identity of the sorocarp with the spot field is not claimed.  
  evidence: `models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py` L548–653; `datasets/dictyostelium_discoideum_turing_wavelength_s0/dictyostelium_discoideum_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 92** — Primary DOIs were resolved on Crossref or OpenAlex, including Bretschneider 2009 and Sawai 2005 added for the cortical-wave and aggregation-scale boundaries. Specimen SHA-256 e2d2b739211bda975bc175167debf554af4991ce21bc8b1106d1053e6bc95a0b, public domain, Bruno in Columbus, 2008.  
  evidence: `datasets/dictyostelium_discoideum_turing_wavelength_s0/paper_citations.bib` L1–117; `datasets/dictyostelium_discoideum_turing_wavelength_s0/README.md` L1–57
- **Code-to-model traceability: 86** — Docstring, sigma_discrete, mode_growth_table, and the Lean theorems (diffusion gate, N=5, N=0 at 200 nm, 1 mm successor gap) name the same integers. The Lean kernel was not run. Turing 1952 is not pinned to a page facsimile.  
  evidence: `morphoeval/lean/loops/Loop_3168_dictyostelium_discoideum_turing_wavelength_s0.lean` L16–42; `morphoeval/math_kg_loops/delphi10000-dictyostelium-discoideum-turing-wavelength-s0.yaml` L1–67
- **Reproducibility package: 90** — python3 script --self-test and a full run regenerate the PNG, the 12-frame GIF, and the metrics JSON from seed 3168. Dependencies are numpy, scipy, pillow, matplotlib. No second machine and no Lean binary in this environment.  
  evidence: `models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py` L655–703; `datasets/dictyostelium_discoideum_turing_wavelength_s0/README.md` L42–53

Remediation:
- R-assessor: still a single builder-assessor. No second provider scored Pass 2.
- R-scale: 2.304 µm remains the catalog coordinate. A cortical micrograph at that scale was not vendored; the public-domain image is a sorocarp.
- R-lean: the four theorems use decide on closed Int goals and contain no sorry, but lean was not installed, so the kernel did not check them.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Dictyostelium discoideum Schnakenberg Turing wavelength at 2.304 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `960dc004eb4baeba25b2cb4a7d5ced2a0307317641afcb7da587f46f715075d8`
- **input_dataset_sha256**: `datasets/dictyostelium_discoideum_turing_wavelength_s0/dictyostelium_discoideum_turing_wavelength_s0_reference.json` (`b4d3f1e6feedaad3…`), `datasets/dictyostelium_discoideum_turing_wavelength_s0/dictyostelium_discoideum_turing_wavelength_s0_specimen.jpg` (`e2d2b739211bda97…`)
- `datasets/dictyostelium_discoideum_turing_wavelength_s0/dictyostelium_discoideum_turing_wavelength_s0_metrics.json` — artifact_sha256 `2df3e28ead71868e…`, 7,403 B, 
- `docs/media/dictyostelium_discoideum_turing_wavelength_s0_comparison.png` — artifact_sha256 `bac77ec104f165d2…`, 292,818 B, 1584×984 px
- `docs/media/dictyostelium_discoideum_turing_wavelength_s0_growth.gif` — artifact_sha256 `b332e96863e53865…`, 5,497,667 B, 960×720 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| simulated stripe wavelength | 2.3037213180107243 µm | 2.304 | argmax of discrete-Laplacian Schnakenberg growth rate σ(k) *(absolute error 0.000279 µm)* |
| dispersion-relation Pearson r | 0.9999999972994091 1 | 1.0 | Pearson r of integrated Fourier-mode growth versus discrete-Laplacian σ(k) on the unstable band *(RMSE 3.668159e-05 1/s over 23 modes; KS statistic 0.0870)* |
| dispersion RMSE | 3.668158916207434e-05 1/s | 0.05 | root mean square residual of σ_measured against σ_theory |
| dispersion KS statistic | 0.08695652173913049 1 | 0.2 | two-sample KS between theoretical and measured growth-rate samples *(p=1)* |
| specimen contour circularity | 0.09609050225669176 1 | 0.75 | 4πA/P² on the border-contrast mask of the sorocarp photograph *(value != reference: fruiting-body silhouette versus cortical spot)* |
| specimen foreground fraction | 0.18145833333333333 1 | 0.45 | mean of the 160×120 border-contrast mask |
| specimen structure-tensor coherency | 0.5803873198291329 1 | 0.25 | mean (λ1-λ2)/(λ1+λ2) on the specimen mask |
| specimen image orientation KS | 0.05918209876543212 1 | 0.05 | KS statistic of Sobel orientations, specimen photograph versus GIF last frame |
| equal-diffusion ablation variance ratio | 3.823912608393148e-05 1 | 246.55114018806117 | var(u) final / var(u) initial with Du = Dv *(negative control versus the Turing regime)* |
| cortical spot count | 32 1 | 25 | connected components of u above the 0.72 quantile |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- pillow: 13.0.0

Same working tree as Pass 1. Pass 2 re-scores the sweep, ablation, segmentation, and dispersion table that the script now reports in the metrics JSON.
