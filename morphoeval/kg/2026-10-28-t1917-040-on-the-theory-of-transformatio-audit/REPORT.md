# MorphoEval report — audit `2026-10-28-t1917-040-on-the-theory-of-transformatio-audit`

Rubric **0.1.0** · checkpoint `7d216d80` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Dürer/Camper facial grids — whole-grid deformation test on human profile landmarks (Pass 1) (`thompson/claims/t1917-040_on-the-theory-of-transformations.py`) | 66 | 35 | 40 | 42 | 80 | 70 | 64 | 57.3 | 37.6 | 70.9 | 53.5 | **53.5** | partially faithful |

## Findings and remediation

### Thompson (1917) Dürer/Camper facial grids — whole-grid deformation test on human profile landmarks (Pass 1)  —  overall 53.5 (partially faithful)

Thompson's types (1) stretch, (3) shear and their affine closure are implemented exactly and mirrored by 6 Lean theorems. Nine CC/PD profile photographs (SHA-256 pinned) are segmented by a Lab background model and landmarked by the extremum sequence; 4 pass QC (Romy Schneider, Roberts, Odúnóku, Margaret of Hesse), 5 fail (gradient wall shadow, halftone/low contrast, faded lower face, textured background, lettering). On 12 ordered pairs the median landmark RMSE falls from 0.189 (similarity) to 0.103 (affine), median R_affine 0.58 (range 0.04-0.81). Convexity 8.5 deg (n=2, norm 12+/-4), nasolabial 93 deg (n=4, SD 39 — columella misdetected on at least two faces). Provisional verdict "mixed"; no noise floor, no CV, no controls.

- **Model fidelity to source: 66** — E1/E3/E4/E5 are coded exactly as Thompson states them (x1, c*y1; x = xi - eta cot w, y = eta cosec w; affine X' = AX + t; tan additivity) and the Lean module proves the group law, whole-grid property, facial-angle monotonicity, circle->ellipse, affine closure and invertibility with no placeholder. Deductions: the full 6-dof affine fitted to 8-9 nearly collinear landmarks is not Thompson's 2-parameter Dürer grid (stretch c + shear k) and is ill-conditioned — fitted anisotropy ranges 0.11-6.8 and rotation up to 69 deg, which no Dürer head exhibits; type (2) logarithmic stretch (E2) is not implemented.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L98–126; `morphoeval/lean/loops/Thompson1917_T1917_040.lean` L49–135; `datasets/thompson_1917/t1917-040/t1917_040_pairwise_fits.csv` L0–0; `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L166–184
- **Morphological realism / specimen match: 35** — Side by side, the accepted silhouettes follow the real profiles (nose, lips, chin) and the Dürer-grid panel reads like Thompson's Fig. 366. But 5 of 9 specimens fail segmentation or landmarking, the Romy Schneider "n" sits on the hair fringe rather than the nasion, lower-lip landmarks are missing on 3 of 4 accepted faces, and the GIF specimen's dense curve includes the hair line. The rendered result does not yet look like a clean anatomical landmark set.  
  evidence: `docs/media/t1917_040_on_the_theory_of_transformatio_comparison.png` L0–0; `datasets/thompson_1917/t1917-040/t1917_040_landmarks.csv` L0–0; `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L253–283
- **Parameter expressivity: 40** — The animation sweeps Thompson's omega 60->120 deg and shows the facial line tilting monotonically with k (E5), and the self-test proves the affine recovers an exact Dürer grid. But the analysis itself has no parameter sweep (c, k, eps), the fitted affine parameters are not interpretable because of the conditioning problem, and nothing demonstrates what range of (c, k) real faces occupy.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L632–684; `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L772–819; `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L499–536
- **Morphometric validation: 42** — Landmark RMSE in centroid-size units for similarity vs affine on every ordered pair, R_affine as the claim statistic, and convexity / nasolabial angles compared with Legan & Burstone norms are computed. Deductions: n = 4 specimens / 12 pairs; no landmark noise floor, so a residual of 0.10 cannot be judged; no leave-one-out CV for a 6-dof fit on 8 points; no negative or positive control on real data; nasolabial SD of 39 deg signals a columella misdetection; convexity computed on only 2 faces.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L196–214; `datasets/thompson_1917/t1917-040/t1917_040_metrics.json` L0–0; `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L711–720
- **Source and data traceability: 80** — IA scan identifier, leaf numbers and SHA-256 of 5 page images; 9 Commons files with title, page URL, licence, author and SHA-256 re-verified at runtime; PMID/DOI for the norms; the mismatching Dryad pointer is recorded with reason; page/figure correction of the claim inventory documented. Deductions: Dürer 1528 and Camper 1794 have no resolvable identifier; Commons derivatives (<= 960 px) are pinned, not the originals.  
  evidence: `datasets/thompson_1917/t1917-040/t1917_040_on_the_theory_of_transformatio_reference.json` L0–0; `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L225–231; `datasets/thompson_1917/t1917-040/scout_manifest.json` L0–0
- **Code-to-model traceability: 70** — Every transformation function names its equation id and Thompson page; the Lean theorems cite the Python functions they mirror; the KG yaml links law -> theorem -> code locus. Deductions: landmark-detection heuristics (prominence 2 % / 0.3 %, 2x rule for li) are not traceable to a published protocol beyond a method-family citation.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L1–60; `morphoeval/math_kg_loops/thompson1917-t1917-040.yaml` L0–0; `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L296–378
- **Reproducibility package: 64** — Deterministic (seed 1917, no stochastic step), --self-test (<1 s) and full run (~4 s) documented in the docstring, outputs written to fixed paths, every input digest verified. Deductions: per-specimen processing annotations (ROI, deskew, k_sigma) live in the reference JSON and were tuned by the builder; dependency versions are not pinned in the script; no CLI switch to vary thresholds.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L772–834; `datasets/thompson_1917/t1917-040/t1917_040_on_the_theory_of_transformatio_reference.json` L0–0

Remediation:
- {'id': 'R1', 'item': 'Independent second assessor (different provider) — this round is builder-scored'}
- {'id': 'R2', 'item': "Replace the unconstrained 6-dof affine by Thompson's Dürer sub-family (uniform scale + stretch c + shear k, 5 dof, fitted in the Procrustes frame) so the parameters are interpretable and the fit is conditioned on nearly collinear profile landmarks; keep the affine as an upper bound"}
- {'id': 'R3', 'item': 'Add dense arc-length semilandmarks of the profile curve between the top landmark and menton so the fit is not driven by 8 points'}
- {'id': 'R4', 'item': 'Noise floor: re-detect landmarks under image perturbation (rescale 0.85/1.15, blur 1 px) and report RMSE; LOO-CV residual for every pair'}
- {'id': 'R5', 'item': 'Controls on real data: positive (exact Dürer grid applied to a specimen is recovered), negative (scrambled correspondences), ablation (local nose-only displacement is NOT absorbed by the grid)'}
- {'id': 'R6', 'item': 'Statistical goodness of fit: Pearson r and RMSE of fitted k vs measured facial-line tan change (E5), KS test similarity vs affine residuals, CV of R across pairs'}
- {'id': 'R7', 'item': "Parameter-regime sweep over omega (50-130 deg) and c (0.67-1.5) applied to a real specimen with angle read-outs; digitise the obliquity of Thompson's Fig. 366 nets from the pinned scan (Hough transform)"}
- {'id': 'R8', 'item': 'Segmentation: rescue the gradient-background specimen (edge-assisted refinement) and fix the columella / nasion detection that produce a 39 deg nasolabial SD and a nasion on the hair fringe; pick the GIF specimen with a visible forehead'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Dürer/Camper facial grids — whole-grid deformation test on human profile landmarks (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `bf2c0ad03b62d78daf7290b9c4b6a8298b657d4806b6d9a98874151b2165a166`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-040/paper_citations.bib` (`1279402c7d3411df…`), `datasets/thompson_1917/t1917-040/scout_manifest.json` (`c738aa8c422f8f96…`), `datasets/thompson_1917/t1917-040/specimens/beverly_sills_profile.jpg` (`679a08ff1f459089…`), `datasets/thompson_1917/t1917-040/specimens/helen_keller_1904_profile.jpg` (`7fa2a48b647c3b09…`), `datasets/thompson_1917/t1917-040/specimens/kubra_gumusay_2023_profile.jpg` (`aaa06c6f5c336da7…`), `datasets/thompson_1917/t1917-040/specimens/margaret_of_hesse_profile.jpg` (`1b38e84fc0eb1594…`), `datasets/thompson_1917/t1917-040/specimens/nasta_rojc_profile.jpg` (`4b5ed492fb19ec93…`), `datasets/thompson_1917/t1917-040/specimens/nell_walden_1905_profile.png` (`b1f49c57c7aeefee…`), `datasets/thompson_1917/t1917-040/specimens/roberts_2013_profile.jpg` (`643f14a150df84e0…`), `datasets/thompson_1917/t1917-040/specimens/romy_schneider_profile.jpg` (`0857743562b65c3a…`), `datasets/thompson_1917/t1917-040/specimens/winifred_odunoku_profile.jpg` (`b48ece4b38e69468…`), `datasets/thompson_1917/t1917-040/t1917_040_on_the_theory_of_transformatio_reference.json` (`a4d2db858ac2cfd9…`), `datasets/thompson_1917/t1917-040/thompson_1917_pages/ongrowthform1917thom_leaf759_p740.jpg` (`9e583cdcbd5c7ced…`), `datasets/thompson_1917/t1917-040/thompson_1917_pages/ongrowthform1917thom_leaf761_p742.jpg` (`214db8a09bd3fdfe…`)
- `datasets/thompson_1917/t1917-040/scout_manifest.json` — artifact_sha256 `c738aa8c422f8f96…`, 24,023 B, 
- `datasets/thompson_1917/t1917-040/t1917_040_landmarks.csv` — artifact_sha256 `02906f25494a8c2f…`, 1,425 B, 
- `datasets/thompson_1917/t1917-040/t1917_040_metrics.json` — artifact_sha256 `6d98a5436f7fceea…`, 2,525 B, 
- `datasets/thompson_1917/t1917-040/t1917_040_on_the_theory_of_transformatio_reference.json` — artifact_sha256 `a4d2db858ac2cfd9…`, 22,001 B, 
- `datasets/thompson_1917/t1917-040/t1917_040_pairwise_fits.csv` — artifact_sha256 `5bd7764c20a22891…`, 1,830 B, 
- `docs/media/t1917_040_on_the_theory_of_transformatio_comparison.png` — artifact_sha256 `b4e443ab9874866b…`, 516,848 B, 1760×1100 px
- `morphoeval/math_kg_loops/thompson1917-t1917-040.yaml` — artifact_sha256 `4ef3b0c7f4761102…`, 8,877 B, 

| measurement | value | reference | method |
|---|---:|---|---|
| accepted specimens with 11 profile landmarks | 4 count | — | Lab background-model silhouette + extremum-sequence landmarks, QC gate |
| median landmark RMSE, similarity null | 0.1892 centroid-size units | — | Procrustes similarity fit per ordered pair (4 dof), rmse |
| median landmark RMSE, whole-grid affine (Dürer grid) | 0.103 centroid-size units | — | least-squares affine fit per ordered pair (6 dof) vs similarity null, rmse |
| median R_affine = 1 - SS(affine)/SS(similarity) | 0.5795 fraction | 1.0 | fraction of non-similarity shape difference absorbed by the whole-grid deformation (claim statistic) *(claim predicts ~1 (whole-grid); 0 = all difference is local)* |
| facial convexity angle G-Sn-Pg, mean over specimens | 8.47 deg | 12.0 | 180 - interior angle at Sn; std over specimens = 8.55 *(Legan & Burstone 1980 adult norm 12 +/- 4 deg)* |
| nasolabial angle Cm-Sn-Ls, mean over specimens | 93.03 deg | 102.0 | interior angle at Sn; std over specimens = 39.02 *(Legan & Burstone 1980 adult norm 102 +/- 8 deg)* |

## Environment at checkpoint


Branch cursor/t1917-040-durer-facial-grids-d049, commit 7d216d80 (Pass 1 script, dataset, Lean module, KG yaml). Environment: Python 3.12.3, numpy 1.26.4, scipy, scikit-image, Pillow, matplotlib (Agg); lean 4.35.0 via elan.
