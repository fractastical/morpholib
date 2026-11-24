# MorphoEval report — audit `2026-10-27-danio-margaritatus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `23fb4b106f961ebb8b80abe7209257c902f834ad` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-danio-margaritatus-turing-wavelength-s0-audit -> 2026-10-27-danio-margaritatus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 13.39 µm in Danio margaritatus (Holocene, 0 Ma) (`models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py`) | 88 | 84 | 88 | 81 | 90 | 85 | 88 | 88.0 | 82.9 | 87.9 | 85.9 | **85.9** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 13.39 µm in Danio margaritatus (Holocene, 0 Ma)  —  overall 85.9 (faithful)

Pass 2. lambda_c = 13.39 µm (1.339e-05 m, log10 = -4.873) at geological age 0 Ma. Diffusion-ratio sweep is monotone and equals 13.39 µm at delta = 40. Albolineatus sigma_max = -0.0805 /s (stable). Nigrofasciatus lambda_c = 17.80 µm (unstable). Dispersion Pearson r = 0.99999995, RMSE = 5.53e-4 /s. Flank segmentation: 73 pearls, polygon circularity 0.708, aspect 1.78, median NN 510.1 µm (bootstrap 95% CI 471–536 µm). Nonlinear FFT peak of the cellular field is 17.22 µm and is plotted as a separate number. PCA-chord CDF Pearson is 0.977; odd-bin Pearson is 0.770 and is not claimed to clear 0.98. KS D on median-scaled nearest neighbours is 0.207. Reaction-knockout amplitude ratio is ~0. GIF is 720×540, 12 frames. CV similarity to the specimen is 93.3.

- **Model fidelity to source: 88** — The script integrates Schnakenberg kinetics with zero-flux boundaries, solves D_u so the linear law is exactly 1.339e-05 m, and checks that law with a periodic 1D growth series (r = 0.99999995). The nonlinear spot field's FFT peak is 17.22 µm; that shift is reported rather than overwritten. gamma = 1/s remains a chosen time unit, and the three-cell Nakamasu system is not the integrator.  
  evidence: `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L85–110; `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L168–211; `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L231–291
- **Morphological realism / specimen match: 84** — The CC BY-SA photograph, the blue-flank pearl hull (73 centroids, wood excluded), and the cellular spot field are on one figure. The animation paints the Schnakenberg field inside that hull on a low-pass of the photograph. The hull is a polygon, so fins and the snout outline are not reproduced. CV similarity of the last GIF frame to the photograph is 93.3.  
  evidence: `docs/media/danio_margaritatus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/danio_margaritatus_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L335–396
- **Parameter expressivity: 88** — With D_u fixed, lambda_c falls from 14.25 µm at delta = 8 to 13.30 µm at delta = 75 and is 13.39 µm at delta = 40. Albolineatus (delta 1.35) has no unstable band. Nigrofasciatus is a longer unstable wave. Reaction knockout and equal diffusion both collapse the pattern. The self-test refuses a non-monotone sweep.  
  evidence: `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L302–318; `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L599–620
- **Morphometric validation: 81** — Specimen NN median is 510.1 µm against the elementary reference 13.39 µm (they are not equal). Bootstrap 95% CI is 471–536 µm and a ±10% body-length sweep is stored. Polygon circularity is 0.708, not 1. KS D on median-scaled NN distances is 0.207. Dispersion RMSE is 5.53e-4 /s. The PCA-chord CDF Pearson is 0.977 and the odd-bin holdout is 0.770; the holdout is not rounded up to the dispersion correlation.  
  evidence: `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L406–463; `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L688–782; `docs/media/danio_margaritatus_turing_wavelength_s0_comparison.png` L0–0
- **Source and data traceability: 90** — Bib file carries Turing 1952, Nakamasu 2009, Patterson 2014, Volkening 2018, Conway 2008, Quigley 2004, Spiewak 2018, and Huang 2021 (albolineatus). Specimen SHA-256 is 48a9005afc001e30472b386bdcabc9db7b2ecd48224c68324755cb166de0181f, license CC BY-SA 3.0, creator Cisamarc. Scout manifest records that slot 3162 does not collide with the 612 µm primordium script.  
  evidence: `datasets/danio_margaritatus_turing_wavelength_s0/paper_citations.bib` L1–89; `datasets/danio_margaritatus_turing_wavelength_s0/scout_manifest.json` L1–30
- **Code-to-model traceability: 85** — Docstring, sweep comment, and GIF footer state k_c^2. Lean theorems state the threshold ordering, the longer successor wavelength, the 12-mode window, and that 510101 nm is 38 elementary cells with a remainder. Render thresholds are reconstruction choices and are labeled as such.  
  evidence: `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L1–40; `morphoeval/lean/loops/Loop_3162_danio_margaritatus_turing_wavelength_s0.lean` L24–52
- **Reproducibility package: 88** — python3 models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py --self-test checks r > 0.98, albolineatus stability, monotone sweep, ablation ratio < 0.35, GIF size, and NN distinct from 13.39 µm. Seed 3162. Specimen digest is pinned. Nanopub signing was not available (no ~/.nanopub/profile.yml).  
  evidence: `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L65–72; `models/2026-delphi10000-danio-margaritatus-turing-wavelength-s0.py` L784–805

Remediation:
- {'id': 'R1', 'item': 'Resolved. segment_pearls requires a local blue window and drops isolated centroids. Circularity is the shoelace quotient (0.708).', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Resolved. Sweep lambdas are 14.25, 13.87, 13.64, 13.48, 13.39, 13.34, 13.30 µm. Self-test asserts the decrease.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Resolved. Figure titles show NN 95% CI 471–536 µm and nonlinear FFT peak 17.22 µm beside linear 13.39 µm.', 'status': 'resolved'}
- {'id': 'R4', 'item': 'Resolved. Holdout uses the principal-axis chord. CDF r = 0.977, odd-bin r = 0.770, reported separately from the dispersion r.', 'status': 'resolved'}
- {'id': 'R5', 'item': 'Open. One photograph and a catalog length of 0.018 m. The ±10% length sweep moves NN from 574 to 701 µm and does not replace a measured scale bar.'}
- {'id': 'R6', 'item': "Open. Predecessor and successor Jacobians are reconstructions at this rung, not re-fits to those species' plates."}
- {'id': 'R7', 'item': 'Open. The animated flank is the convex hull, not a traced body outline.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 13.39 µm in Danio margaritatus (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `b77145afbb1daaa36679993eab12015d9069adc4fe8d9a9b382f23a8172ce984`
- **input_dataset_sha256**: `datasets/danio_margaritatus_turing_wavelength_s0/danio_margaritatus_specimen.jpg` (`48a9005afc001e30…`), `datasets/danio_margaritatus_turing_wavelength_s0/danio_margaritatus_turing_wavelength_s0_reference.json` (`ecf0d9be62697438…`)
- `docs/media/danio_margaritatus_turing_wavelength_s0_comparison.png` — artifact_sha256 `8eac595b6d5b6eb2…`, 630,986 B, 1512×984 px
- `docs/media/danio_margaritatus_turing_wavelength_s0_growth.gif` — artifact_sha256 `286ed51b632a837f…`, 1,242,941 B, 720×540 px, 12 frames

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


Pass 2 re-score after R1–R4. publisher_orcid is the creator already used on this repo's MorphoEval nanopubs. No ~/.nanopub/profile.yml, so these nanopubs are unsigned.
