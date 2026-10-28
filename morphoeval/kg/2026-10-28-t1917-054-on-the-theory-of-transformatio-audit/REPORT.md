# MorphoEval report — audit `2026-10-28-t1917-054-on-the-theory-of-transformatio-audit`

Rubric **0.1.0** · checkpoint `8b77a9d17` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Figs. 388-393: Archaeopteryx -> Apatornis pelvis by equidistant coordinate-grid interpolation (`thompson/claims/t1917-054_on-the-theory-of-transformations.py`) | 74 | 66 | 46 | 56 | 88 | 82 | 76 | 64.7 | 62.2 | 81.6 | 68.8 | **68.8** | mostly faithful |

## Findings and remediation

### Thompson (1917) Figs. 388-393: Archaeopteryx -> Apatornis pelvis by equidistant coordinate-grid interpolation  —  overall 68.8 (mostly faithful)

Pass 1 (v1). Segments the five wash silhouettes of Fig. 393 from the SHA-256-pinned Internet Archive page scan (Otsu 139, opening, five components of 63-80 kpx), finds the acetabulum and the four process tips automatically, places 4 x 24 arc-length semilandmarks, runs GPA, fits the Archaeopteryx -> Apatornis grid as a thin-plate spline (E4) with an affine-only control (E5), and generates Thompson's equidistant stages phi_t (E2/E6). Heilmann's drawn intermediates sit at t* = 0.263 / 0.529 / 0.824 (nominal 0.25 / 0.5 / 0.75; max residual 0.074); the interpolated prediction is closer to each drawn stage (Procrustes 0.134 / 0.121 / 0.093) than the nearest endpoint (0.214 / 0.312 / 0.145) and than the affine interpolation (0.162 / 0.199 / 0.293). Centroid sizes 2190 / 2171 / 2158 / 2309 / 2325 px are not monotone (the series is not drawn to the grid's scale). Part B ("biologically real") is untestable: no landmark data for intermediate taxa; Dryad k1v34cf is a variable mismatch. Verdict in result.json: mixed (A supported, B untestable). Overall 68.8/100.

- **Model fidelity to source: 74** — E2/E6 (equidistant interpolation), E4 (TPS), E5 (affine interpolation M_t) and E7 (t*) are implemented exactly as stated in the reference JSON and mirrored by 11 Lean theorems (endpoints/equidistance, similarity equivariance, no-fold, t* soundness, affine determinants). Deduction: Thompson's actual Fig. 389 grid is not digitised; phi is reconstructed from silhouette correspondences, and the no-fold condition (E1, det > 0) is proved in Lean but not checked numerically on the fitted TPS.  
  evidence: `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L307–366; `morphoeval/lean/loops/Thompson1917_T1917_054.lean` L24–180; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L409–425
- **Morphological realism / specimen match: 66** — The output is compared side by side with the source author's own figures: Figs. 388/389 crops, the segmented Fig. 393 with landmarks, and each predicted stage overlaid on Heilmann's drawn stage. The predicted outlines are visibly intermediate and distinct per stage. Deductions: the stage panels are cramped (grid padding dwarfs the form); the real Archaeopteryx fossil photograph is vendored but not analysed; Apatornis is represented by a synsacrum plate only.  
  evidence: `docs/media/t1917_054_on_the_theory_of_transformatio_comparison.png` L0–0; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L498–577; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L498–577
- **Parameter expressivity: 46** — t visibly and monotonically moves the form from Archaeopteryx to Apatornis (GIF, 33 frames) and the pubis angle rises 131 -> 162 deg along the drawn series. --n-semi and --lam exist on the CLI but no sweep is run or reported, and the TPS-vs-affine family switch is exercised only at the default setting.  
  evidence: `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L580–613; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L762–777
- **Morphometric validation: 56** — Quantitative morphometrics are computed on the output and compared with the source's claim: t* vs nominal stages, Procrustes distance of the prediction vs two controls (nearest endpoint, affine interpolation), descriptors (pubis/ischium angles, pre/post acetabular ratio, centroid size). Missing: correspondence-free statistics (IoU, boundary RMSE), Pearson r between predicted and drawn coordinates, any null distribution or permutation p-value, and uncertainty from the digitisation (semilandmark density).  
  evidence: `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L427–470; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L665–708; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L427–470
- **Source and data traceability: 88** — All seven inputs are SHA-256 pinned in the reference JSON and re-verified at runtime (fail closed); the IA item, leaf-to-page rule, Commons file pages, authors and licences are recorded; DOIs for Bookstein, Gower, Deeming & Mayr, Clarke are in the .bib; the pre-scouted Dryad dataset is recorded with its mismatch reason. Deduction: Thompson 1917 itself has no DOI (IA URL only) and the Dryad file could not be fetched.  
  evidence: `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L128–141; `datasets/thompson_1917/t1917-054/scout_manifest.json` L0–0; `datasets/thompson_1917/t1917-054/paper_citations.bib` L1–95
- **Code-to-model traceability: 82** — The module docstring maps E1-E7 to named functions, quotes the source passages, and names the Lean mirror; each function docstring cites its equation; the KG YAML pins lean_locus and code_locus per law. Deduction: code_locus entries in the KG YAML name functions rather than line ranges.  
  evidence: `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L1–80; `morphoeval/math_kg_loops/thompson1917-t1917-054.yaml` L1–120
- **Reproducibility package: 76** — Deterministic (no stochastic step; two-run equality asserted), seed constant declared, fail-closed input hashes, 10-check self-test in 1.4 s, full run 3.8 s, documented invocation, outputs to declared paths. Deductions: dependency versions pinned only via requirements.txt lower bounds; the comparison PNG is git-ignored by repo policy so a reviewer must re-render it.  
  evidence: `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L711–759; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L762–781

Remediation:
- {'id': 'R1', 'item': 'Independent second assessor (different provider) — this round is builder-scored'}
- {'id': 'R2', 'item': 'Parameter-regime sweep over n_semi in {8, 12, 16, 24, 32, 48} and lambda in {0, 1e-3, 1e-2, 1e-1}; report t* and distance ratios per setting'}
- {'id': 'R3', 'item': 'Negative/null controls: shuffled-stage permutation (all 3! pairings) and a random-correspondence null; report p-values'}
- {'id': 'R4', 'item': 'Correspondence-free goodness of fit: IoU of superimposed filled outlines and symmetric boundary RMSE; Pearson r and RMSE of predicted vs drawn coordinates; KS test of per-landmark residuals vs endpoint-control residuals'}
- {'id': 'R5', 'item': 'Jacobian no-fold check of the TPS grid at t = 0..1 (min det over the 9 x 8 grid)'}
- {'id': 'R6', 'item': 'Bone-vs-matrix colour segmentation of the Berlin Archaeopteryx specimen as a labelled, non-verdict CV pass; enlarge the predicted-vs-drawn panels'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=2.5]** `parameter_sweep` → `parameter-expressivity`: Sweep semilandmark density (8..48) and TPS lambda; report stability of t* and of the prediction/endpoint ratio
- **[OPEN · P=2.5]** `morphometric_validation` → `morphometric-validation`: Correspondence-free IoU and boundary RMSE, Pearson r of predicted vs drawn coordinates, shuffled-stage permutation null, KS test of residual distributions
- **[OPEN · P=2.5]** `specimen_comparison` → `specimen-match`: CV segmentation of the Berlin Archaeopteryx photograph (bone vs matrix) and a larger predicted-vs-drawn figure
- **[OPEN · P=2.5]** `model_equation` → `model-fidelity`: Numerical no-fold (Jacobian determinant) check of phi_t on the 9 x 8 grid at every stage, mirroring Lean interp_no_fold
- **[OPEN · P=2.5]** `independent_assessor` → `model-fidelity`: Independent second assessor from a different provider

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Figs. 388-393: Archaeopteryx -> Apatornis pelvis by equidistant coordinate-grid interpolation
- **spec_fingerprint** (semantic model + inputs): `970a80c4a196f6e5a31934d373b23b236d1d7d84f48c848cb068165ec1804809`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-054/apatornis_celer_holotype_synsacrum_marsh1880.jpg` (`28213a31e4c88718…`), `datasets/thompson_1917/t1917-054/archaeopteryx_lithographica_berlin_specimen_mfn.jpg` (`96341aed3d81b2b6…`), `datasets/thompson_1917/t1917-054/dryad_k1v34cf_deeming_mayr_2018_metadata.json` (`4f2b721ee3bec249…`), `datasets/thompson_1917/t1917-054/ongrowthform1917thom_p757_leaf0776_figs388-389.jpg` (`5214f00748cee71c…`), `datasets/thompson_1917/t1917-054/ongrowthform1917thom_p758_leaf0777_figs390-391.jpg` (`3b7d8054e6d7e995…`), `datasets/thompson_1917/t1917-054/ongrowthform1917thom_p759_leaf0778_fig392.jpg` (`17283e61fc562441…`), `datasets/thompson_1917/t1917-054/ongrowthform1917thom_p760_leaf0779_fig393.jpg` (`0e73dc9e37ed7e64…`), `datasets/thompson_1917/t1917-054/paper_citations.bib` (`6a29f62000543207…`), `datasets/thompson_1917/t1917-054/scout_manifest.json` (`6a8a3e51bdde2a4a…`), `datasets/thompson_1917/t1917-054/t1917_054_on_the_theory_of_transformatio_reference.json` (`92bc0f5512fe8f9d…`)
- `datasets/thompson_1917/t1917-054/scout_manifest.json` — artifact_sha256 `6a8a3e51bdde2a4a…`, 11,273 B, 
- `docs/media/t1917_054_on_the_theory_of_transformatio_comparison.png` — artifact_sha256 `88a7e3f72db88201…`, 737,024 B, 1980×1210 px

| measurement | value | reference | method |
|---|---:|---|---|
| best-fit stage t* of Heilmann's first intermediate (Fig. 391) | 0.2627 t (0 = Archaeopteryx, 1 = Apatornis) | 0.25 | E7 least-squares projection of the GPA-aligned drawn stage onto the Archaeopteryx->Apatornis segment |
| best-fit stage t* of the second intermediate (Fig. 392 top) | 0.5288 t | 0.5 | E7 projection |
| best-fit stage t* of the third intermediate (Fig. 392 bottom) | 0.8235 t | 0.75 | E7 projection |
| max equidistance residual |t* - k/4| over the three stages | 0.0735 t units | 0.0 | max over k of |t*_k - k/4| *(mean residual 0.0384; d(A,B) = 0.5827 so one stage = 0.1457 Procrustes units)* |
| mean full Procrustes distance, interpolated grid prediction vs drawn intermediate | 0.116 Procrustes distance | 0.2235 | E2/E4: TPS phi_t applied to the Archaeopteryx configuration, full Procrustes fit onto the drawn stage, mean over stages 1-3 *(affine-only interpolation control: 0.2181; ratio interpolated/endpoint = 0.519)* |
| Archaeopteryx -> Apatornis full Procrustes distance (Fig. 393 endpoints) | 0.5827 Procrustes distance | — | GPA of acetabulum + 4 x 24 semilandmarks, unit centroid size |
| pubis angle to ilium axis along the series (A, 1, 2, 3, B) | [131.3, 140.7, 143.2, 153.3, 162.2] deg | — | angle between acetabulum->pubis tip and anterior->posterior ilium tip vectors, per segmented form |
| centroid size of the five drawn forms | [2190.4, 2171.0, 2157.9, 2308.8, 2324.5] px (page scan) | — | sqrt of summed squared deviations from centroid of the landmark configuration *(not monotone: Heilmann's Fig. 393 series is not drawn to the interpolated grid's scale)* |

## Environment at checkpoint

- python: 3.12
- numpy: 1.26.4
- scipy: 1.x (pip, requirements.txt lower bound >=1.7)
- scikit-image: >=0.19
- matplotlib: >=3.3
- Pillow: >=8.0
- lean: 4.35.0 (elan stable)

Pass 1 of the 12-step loop 2026-10-28-t1917-054-on-the-theory-of-transformatio-loop (branch cursor/t1917-054-archaeopteryx-pelvis-01ab). Self-test 10/10 in 1.4 s; full run 3.8 s. The comparison PNG is regenerated by the script and is git-ignored by repo policy (docs/media/*.png); the GIF is tracked.
