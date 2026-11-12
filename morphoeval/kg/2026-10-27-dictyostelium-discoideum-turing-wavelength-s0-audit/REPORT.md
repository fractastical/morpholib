# MorphoEval report — audit `2026-10-27-dictyostelium-discoideum-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dictyostelium discoideum Schnakenberg Turing wavelength at 2.304 µm (Holocene, 0 Ma) (`models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py`) | 84 | 62 | 58 | 64 | 90 | 78 | 82 | 75.3 | 62.8 | 83.6 | 72.8 | **72.8** | mostly faithful |

## Findings and remediation

### Dictyostelium discoideum Schnakenberg Turing wavelength at 2.304 µm (Holocene, 0 Ma)  —  overall 72.8 (mostly faithful)

Pass 1 integrates Schnakenberg kinetics on an 11.52 µm periodic cell so the discrete-Laplacian growth rate peaks at 2.304 µm (measured 2.3037 µm). The GIF is a 12-frame, 960×720 activator field. The public-domain sorocarp is the species photograph, not a cortical micrograph. Diffusion-ratio sweep, equal-diffusion ablation, specimen segmentation, and dispersion goodness-of-fit are left for the v2 re-score.

- **Model fidelity to source: 84** — The script implements the Schnakenberg PDE, the five-point periodic Laplacian, and σ(k) from the discrete eigenvalue. Pass 1 credits that reference regime. It does not yet credit the dispersion-table residual as a fidelity score.  
  evidence: `models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py` L84–123; `models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py` L125–164
- **Morphological realism / specimen match: 62** — The comparison figure places the public-domain sorocarp beside a cortical spot field. They are different magnifications. Pass 1 does not yet use the measured CV similarity as a specimen-match score.  
  evidence: `docs/media/dictyostelium_discoideum_turing_wavelength_s0_comparison.png` L0–0; `docs/media/dictyostelium_discoideum_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 58** — Pass 1 judges only the reference point δ=24, λ=2.304 µm, five modes per cell. The diffusion-ratio sweep is the Pass 2 remediation.  
  evidence: `models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py` L49–66
- **Morphometric validation: 64** — Pass 1 records that the analytical peak is 2.3037 µm against the 2.304 µm coordinate. Pearson r, RMSE, KS, and the specimen mask statistics are the Pass 2 remediation.  
  evidence: `models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py` L109–123
- **Source and data traceability: 90** — Turing 1952, Schnakenberg 1979, Taniguchi 2013, and Höfer 1995 were confirmed on Crossref or OpenAlex. The sorocarp file is the Wikimedia public-domain JPEG by Bruno in Columbus, SHA-256 pinned in the reference JSON.  
  evidence: `datasets/dictyostelium_discoideum_turing_wavelength_s0/paper_citations.bib` L1–117; `datasets/dictyostelium_discoideum_turing_wavelength_s0/dictyostelium_discoideum_turing_wavelength_s0_reference.json` L102–114
- **Code-to-model traceability: 78** — The module docstring states the PDE, k_c, the 2.304 µm scale, age 0 Ma, and the claim boundary. Pass 1 does not yet pin the mode-growth table to the Lean statements.  
  evidence: `models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py` L1–30; `morphoeval/lean/loops/Loop_3168_dictyostelium_discoideum_turing_wavelength_s0.lean` L16–42
- **Reproducibility package: 82** — Seed 3168, --self-test, and pinned Du, Dv. Pass 1 self-test checks the 1% wavelength band, five modes, and the equal-diffusion sign. Full metrics JSON is the Pass 2 package.  
  evidence: `models/2026-delphi10000-dictyostelium-discoideum-turing-wavelength-s0.py` L655–703

Remediation:
- R-sweep: score the δ = Dv/Du sweep (onset and the monotonic rise of λ once σ_max > 0) under parameter-expressivity.
- R-ablation: score the equal-diffusion negative control (variance collapse versus the δ=24 run).
- R-cv: enter the real specimen border-contrast segmentation (foreground, circularity, coherency) as morphometrics with references that are not equal to the measured values.
- R-gof: report Pearson r, RMSE, and KS for the dispersion relation, plus the specimen-versus-GIF orientation KS.
- R-assessor: a second assessor did not score this pass.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Dictyostelium discoideum Schnakenberg Turing wavelength at 2.304 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `960dc004eb4baeba25b2cb4a7d5ced2a0307317641afcb7da587f46f715075d8`
- **input_dataset_sha256**: `datasets/dictyostelium_discoideum_turing_wavelength_s0/dictyostelium_discoideum_turing_wavelength_s0_reference.json` (`b4d3f1e6feedaad3…`), `datasets/dictyostelium_discoideum_turing_wavelength_s0/dictyostelium_discoideum_turing_wavelength_s0_specimen.jpg` (`e2d2b739211bda97…`)
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

Working tree for loop 3168. Lean kernel was not on PATH.
