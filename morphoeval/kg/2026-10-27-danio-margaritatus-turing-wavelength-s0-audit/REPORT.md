# MorphoEval report — audit `2026-10-27-danio-margaritatus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `23fb4b106f961ebb8b80abe7209257c902f834ad` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 13.39 µm in Danio margaritatus (Holocene, 0 Ma) (`models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py`) | 76 | 72 | 74 | 68 | 86 | 76 | 82 | 75.3 | 70.5 | 81.7 | 75.3 | **75.3** | mostly faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 13.39 µm in Danio margaritatus (Holocene, 0 Ma)  —  overall 75.3 (mostly faithful)

Pass 1. Linear lambda_c is 13.39 µm by construction and the 1D PDE growth-rate series matches the eigenvalue (Pearson r = 0.99999995, RMSE = 5.5e-4 /s). Danio albolineatus is stable at this rung (sigma_max = -0.080 /s). Danio nigrofasciatus selects 17.80 µm. The adult photograph's pearl nearest-neighbour gap is 510 µm, so the elementary mode is not resolved. First-pass gaps: the diffusion-ratio sweep had to be checked at the anchor, the pearl hull had to be kept on the blue flank, circularity had to be a polygon quotient rather than a clipped raster, and the comparison figure had to separate the nonlinear FFT peak (17.22 µm) from the linear law.

- **Model fidelity to source: 76** — Schnakenberg kinetics, Jacobian, and k_c formula are implemented and the 1D growth test matches the eigenvalue. Pass 1 still docks the nonlinear FFT peak (17.22 µm) sitting well above the linear 13.39 µm law, and gamma = 1/s is a time unit rather than a measured rate.  
  evidence: `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L74–110; `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L231–291
- **Morphological realism / specimen match: 72** — The CC BY-SA lateral photograph is shown beside a flank hull. Pass 1 hull selection still needed a blue-window gate so wood orange was not treated as pearls, and the GIF flank is a convex polygon rather than the body outline.  
  evidence: `docs/media/danio_margaritatus_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L335–396
- **Parameter expressivity: 74** — Three chronospecies and a reaction knockout are present. Pass 1 requires the delta sweep to be monotone and to recover exactly 13.39 µm at delta = 40 before this dimension is credited in full.  
  evidence: `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L302–318; `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L784–805
- **Morphometric validation: 68** — Dispersion RMSE and Pearson r are real. Specimen NN spacing (510 µm), circularity, and a KS distance are computed, but the spatial holdout was not yet the flank-axis chord model, and the bootstrap interval was not on the figure.  
  evidence: `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L688–782
- **Source and data traceability: 86** — DOIs for Turing, Nakamasu, Patterson, Volkening, Conway, and Quigley are in the bib file. The specimen is CC BY-SA 3.0 (Cisamarc) and SHA-256 pinned. OpenAlex scout record is in the dataset.  
  evidence: `datasets/danio_margaritatus_turing_wavelength_s0/paper_citations.bib` L1–89; `datasets/danio_margaritatus_turing_wavelength_s0/danio_margaritatus_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 76** — The module docstring and the sweep comment restate k_c^2. Pass 1 does not yet pin every render choice to a paper section.  
  evidence: `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L1–40; `morphoeval/lean/loops/Loop_3162_danio_margaritatus_turing_wavelength_s0.lean` L24–52
- **Reproducibility package: 82** — Seed 3162, --self-test, and a pinned specimen digest. Pass 1 figure did not yet carry the bootstrap interval that the metrics JSON already stores.  
  evidence: `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L784–805

Remediation:
- {'id': 'R1', 'item': 'Keep the pearl hull on the blue flank (drop wood and plants) and report polygon circularity instead of a raster quotient clipped at 1.'}
- {'id': 'R2', 'item': 'Diffusion-ratio sweep at fixed D_u must decrease lambda_c and equal 13.39 µm at delta = 40.'}
- {'id': 'R3', 'item': 'Print the bootstrap 95% CI and the nonlinear FFT peak on the comparison figure, separate from the linear 13.39 µm law.'}
- {'id': 'R4', 'item': 'Replace the axis-aligned ellipse occupancy model with the flank principal-axis chord model and report the odd-bin Pearson without forcing it above 0.98.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 13.39 µm in Danio margaritatus (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `b77145afbb1daaa36679993eab12015d9069adc4fe8d9a9b382f23a8172ce984`
- **input_dataset_sha256**: `datasets/danio_margaritatus_turing_wavelength_s0/danio_margaritatus_specimen.jpg` (`48a9005afc001e30…`), `datasets/danio_margaritatus_turing_wavelength_s0/danio_margaritatus_turing_wavelength_s0_reference.json` (`ecf0d9be62697438…`)
- `docs/media/danio_margaritatus_turing_wavelength_s0_comparison.png` — artifact_sha256 `8eac595b6d5b6eb2…`, 630,986 B, 1512×984 px

| measurement | value | reference | method |
|---|---:|---|---|
| critical wavelength lambda_c | 13.39 µm | 17.8 | k_c^2 = gamma (D_v f_u + D_u g_v) / (2 D_u D_v); D_u solved so lambda_c = 13.39 µm *(predecessor albolineatus has no real k_c at delta=1.35)* |
| dispersion Pearson r analytic vs 1D PDE | 0.9999999531548494 Pearson r | 0.95 | cosine-mode growth on a periodic line versus the closed-form eigenvalue *(RMSE = 0.00055 1/s)* |
| dispersion RMSE | 0.0005526070525730378 1/s | 0.05 | RMSE between analytic sigma(k) and the linearized 1D PDE growth rate *(not zero: the explicit Euler step and the weak nonlinearity at eps=1e-4 leave a residual)* |
| adult pearl nearest-neighbour spacing | 510.1005699664499 µm | 13.39 | median centroid spacing of segmented orange pearls, scaled by 0.018 m / hull major axis *(CV = 0.399; bootstrap 95% CI in metrics JSON; elementary cells per gap = 38.10)* |
| pearl-field circularity | 0.7079191663342005 4*pi*A/P^2 | 1.0 | convex hull of inlier pearl centroids on the CC BY-SA photograph *(aspect = 1.776; foreground fraction = 0.341)* |
| KS statistic of median-scaled NN distances | 0.20730593607305936 KS D | 0.0 | two-sample KS between specimen NN/median(NN) and Schnakenberg spot NN/median(NN) *(p = 0.0984)* |
| reaction-knockout ablation amplitude ratio | 1.0243603329022087e-11 std(u_ablated)/std(u) | 1.0 | same seed and diffusion, u^2 v exchange removed *(equal-diffusion std = 0.0003; pattern std = 1.1409)* |
| specimen CDF Pearson versus ellipse marginal | 0.9770954977375305 Pearson r | 0.9 | cumulative distribution of pearls along the flank principal axis versus local chord length *(odd-bin Pearson = 0.770; CDF RMSE = 0.1198)* |

## Environment at checkpoint


Pass 1 score of the slot 3162 script. R1–R4 are the gaps closed before the v2 re-score.
