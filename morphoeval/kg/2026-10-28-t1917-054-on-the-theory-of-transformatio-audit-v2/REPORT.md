# MorphoEval report — audit `2026-10-28-t1917-054-on-the-theory-of-transformatio-audit-v2`

Rubric **0.1.0** · checkpoint `7a7c0f5b6` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-28-t1917-054-on-the-theory-of-transformatio-audit -> 2026-10-28-t1917-054-on-the-theory-of-transformatio-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Figs. 388-393: Archaeopteryx -> Apatornis pelvis by equidistant coordinate-grid interpolation (`thompson/claims/t1917-054_on-the-theory-of-transformations.py`) | 86 | 78 | 80 | 86 | 88 | 86 | 82 | 84.0 | 81.0 | 85.1 | 83.1 | **83.1** | faithful |

## Findings and remediation

### Thompson (1917) Figs. 388-393: Archaeopteryx -> Apatornis pelvis by equidistant coordinate-grid interpolation  —  overall 83.1 (faithful)

Pass 2 (v2), overall 83.1/100 (up from 68.8 in Pass 1). The five Fig. 393 silhouettes are segmented and landmarked from the SHA-256-pinned IA scan; GPA; Thompson's equidistant interpolation phi_t = (1-t) id + t phi (E2/E6) is run with phi as (i) a dense semilandmark TPS, (ii) a 9-anchor coarse TPS grid and (iii) an affine map (E5), and scored against Heilmann's drawn intermediates with the nearest-endpoint null. Heilmann's stages sit at t* = 0.263 / 0.529 / 0.824 (nominal 0.25 / 0.5 / 0.75; max residual 0.074; sweep SD <= 0.003 over 24 settings). Dense interpolation: Procrustes 0.134 / 0.121 / 0.093 vs endpoint 0.214 / 0.312 / 0.145 and affine 0.162 / 0.199 / 0.293; IoU 0.572 / 0.599 / 0.683 vs endpoint 0.538 / 0.436 / 0.629; r = 0.991-0.996; KS p <= 5.5e-6; correspondence-shift null exact p = 0.0104 (z = -4.4). The dense TPS folds as a grid (min det -10.4 inside the form) whereas the 9-anchor coarse grid is fold-free (min det 0.23) and beats the endpoint on stages 1-2 only. Part B ("biologically real") remains untestable; Dryad k1v34cf is a labelled variable mismatch. Verdict in result.json: mixed (A supported, approximately equidistant; B untestable).

- **Model fidelity to source: 86** — E2/E6, E4, E5 and E7 are implemented as stated and mirrored by 11 Lean theorems; Pass 2 adds the numerical E1 check (min det D phi_t over the 9 x 8 grid and inside the form at 21 stages) that Lean states in 1-D, plus the coarse 9-anchor grid that realises Thompson's actual construction (few anchors, smooth grid, fold-free) beside the dense semilandmark spline. Remaining deduction: Fig. 389's drawn grid is still not digitised line by line, so phi is reconstructed from the silhouettes rather than read off the plate.  
  evidence: `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L322–376; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L483–514; `morphoeval/lean/loops/Thompson1917_T1917_054.lean` L24–180; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L582–600
- **Morphological realism / specimen match: 78** — Predicted stages are overlaid on Heilmann's drawn stages at form scale (15-panel figure: Figs. 388/389, segmented Fig. 393, three overlays with the fold-free grid, controls), and the correspondence-free IoU quantifies the visual match (0.57-0.68 vs 0.44-0.63 for the endpoint null). The real Berlin Archaeopteryx photograph is now analysed (bone/matrix mask) but, as the audit must say, it contributes nothing to the pelvis comparison because the pelvis is disarticulated; Apatornis is still a synsacrum plate. The dense prediction shows fold wiggles on the thin pubis, visible in the overlays.  
  evidence: `docs/media/t1917_054_on_the_theory_of_transformatio_comparison.png` L0–0; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L814–836; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L554–579
- **Parameter expressivity: 80** — t moves the form monotonically (GIF; pubis angle 131 -> 162 deg; t* strictly increasing); the sweep over 6 semilandmark densities x 4 lambdas is run and reported (t* SD <= 0.003; prediction/endpoint ratio 0.51 -> 0.91 as lambda grows, i.e. lambda visibly trades endpoint fidelity for smoothness); three transformation families (dense TPS, coarse 9-anchor TPS, affine) are compared at every stage. Deduction: the anchor set of the coarse grid (9) is fixed, not swept.  
  evidence: `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L717–755; `datasets/thompson_1917/t1917-054/result.json` L0–0; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L511–514
- **Morphometric validation: 86** — Quantitative morphometrics on the output are compared against the source's claim with explicit nulls and uncertainty: t* vs nominal with sweep SD; Procrustes, IoU, boundary RMSE, Pearson r and RMSE vs three controls; KS test of residual distributions (p <= 5.5e-6); 3! pairing permutation (nominal best, p = 1/6 floor); exact correspondence-shift null (95 shifts, p = 0.0104, z = -4.4). measure_morphoeval() exposes 11 measurements, 8 with reference values. Deductions: the permutation test has only 6 arrangements; no literature morphometrics of real intermediate taxa exist in the pinned data (part B).  
  evidence: `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L461–481; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L517–552; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L1038–1098; `datasets/thompson_1917/t1917-054/result.json` L0–0
- **Source and data traceability: 88** — Unchanged from Pass 1 and still strong: seven inputs SHA-256 pinned and re-verified at runtime (fail closed); IA item, leaf-to-page rule, Commons file pages with authors and licences, DOIs for all modern references, and the pre-scouted Dryad record with its mismatch reason. Deduction: Thompson 1917 has no DOI (IA URL only); the Dryad file itself could not be fetched (browser challenge), only its API metadata.  
  evidence: `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L138–151; `datasets/thompson_1917/t1917-054/scout_manifest.json` L0–0; `datasets/thompson_1917/t1917-054/paper_citations.bib` L1–95
- **Code-to-model traceability: 86** — Module docstring maps E1-E7 to functions, quotes the source passages, lists the Pass 2 controls and names the Lean mirror; function docstrings cite their equation or remediation item (R2-R6); the KG YAML pins lean_locus and line-range code_locus per law. Deduction: the Pass 2 statistics (IoU, KS, nulls) have no equation number in the reference JSON beyond E7.  
  evidence: `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L1–90; `morphoeval/math_kg_loops/thompson1917-t1917-054.yaml` L1–125
- **Reproducibility package: 82** — Deterministic (two-run equality asserted), fail-closed input hashes, 16-check self-test in 2.7 s covering the equations, the numerical Jacobian on known shear/scaling maps, IoU/RMSE sanity, segmentation areas, t* ordering, all three controls, both nulls, the no-fold check and the specimen pass; full run 28 s; CLI flags for n_semi, lambda, sweep, specimen, GIF. Deductions: dependency versions pinned only via requirements.txt lower bounds; the comparison PNG is git-ignored by repo policy so a reviewer must re-render it.  
  evidence: `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L1101–1175; `thompson/claims/t1917-054_on-the-theory-of-transformations.py` L1178–1200

Remediation:
- R1 (open): independent second assessor from a different provider — this round is builder-scored
- R7: digitise the grid lines of Figs. 388-390 directly and compare Thompson's drawn intermediate grids with the fitted coarse TPS grid
- R8: part B — assemble 2-D pelvis landmarks of real intermediate Mesozoic birds from open literature plates (e.g. Marsh 1880 Ichthyornis/Hesperornis, PD) and test the interpolated stages against them
- R9: sweep the coarse-grid anchor count (5, 9, 13, 17) to locate where a fold-free grid starts to reproduce the drawn stages as well as the dense correspondence

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=2.5]** `parameter_sweep` → `parameter-expressivity`: Sweep semilandmark density (8..48) and TPS lambda; report stability of t* and of the prediction/endpoint ratio
- **[RESOLVED · P=2.5]** `morphometric_validation` → `morphometric-validation`: Correspondence-free IoU and boundary RMSE, Pearson r, shuffled-stage permutation null, KS test
- **[RESOLVED · P=2.5]** `specimen_comparison` → `specimen-match`: CV segmentation of the Berlin Archaeopteryx photograph and a larger predicted-vs-drawn figure
- **[RESOLVED · P=2.5]** `model_equation` → `model-fidelity`: Numerical no-fold (Jacobian determinant) check of phi_t at every stage
- **[OPEN · P=2.5]** `independent_assessor` → `model-fidelity`: Independent second assessor from a different provider
- **[OPEN · P=2.5]** `specimen_comparison` → `morphometric-validation`: Part B: 2-D pelvis landmarks of phylogenetically intermediate taxa (Jeholornis, Sapeornis, Confuciusornis, Enantiornithes, Hesperornis, Ichthyornis) to test whether the interpolated stages are biologically real

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Figs. 388-393: Archaeopteryx -> Apatornis pelvis by equidistant coordinate-grid interpolation
- **spec_fingerprint** (semantic model + inputs): `2777b4737ea8977ede0d810eac8e3fa1010e0a04c769590e31a7cb66eaa76bf4`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-054/apatornis_celer_holotype_synsacrum_marsh1880.jpg` (`28213a31e4c88718…`), `datasets/thompson_1917/t1917-054/archaeopteryx_lithographica_berlin_specimen_mfn.jpg` (`96341aed3d81b2b6…`), `datasets/thompson_1917/t1917-054/dryad_k1v34cf_deeming_mayr_2018_metadata.json` (`4f2b721ee3bec249…`), `datasets/thompson_1917/t1917-054/ongrowthform1917thom_p757_leaf0776_figs388-389.jpg` (`5214f00748cee71c…`), `datasets/thompson_1917/t1917-054/ongrowthform1917thom_p758_leaf0777_figs390-391.jpg` (`3b7d8054e6d7e995…`), `datasets/thompson_1917/t1917-054/ongrowthform1917thom_p759_leaf0778_fig392.jpg` (`17283e61fc562441…`), `datasets/thompson_1917/t1917-054/ongrowthform1917thom_p760_leaf0779_fig393.jpg` (`0e73dc9e37ed7e64…`), `datasets/thompson_1917/t1917-054/paper_citations.bib` (`6a29f62000543207…`), `datasets/thompson_1917/t1917-054/scout_manifest.json` (`6a8a3e51bdde2a4a…`), `datasets/thompson_1917/t1917-054/t1917_054_on_the_theory_of_transformatio_reference.json` (`92bc0f5512fe8f9d…`)
- `datasets/thompson_1917/t1917-054/result.json` — artifact_sha256 `2a03419b94cc0dcb…`, 16,321 B, 
- `datasets/thompson_1917/t1917-054/scout_manifest.json` — artifact_sha256 `6a8a3e51bdde2a4a…`, 11,273 B, 
- `docs/media/t1917_054_on_the_theory_of_transformatio_comparison.png` — artifact_sha256 `a30bd5dd44fca703…`, 1,061,469 B, 2200×1350 px

| measurement | value | reference | method |
|---|---:|---|---|
| best-fit stage t* of Heilmann's first intermediate (Fig. 391) | 0.2627 t (0 = Archaeopteryx, 1 = Apatornis) | 0.25 | E7 least-squares projection of the GPA-aligned drawn stage onto the Archaeopteryx->Apatornis segment *(sweep over 24 settings: mean 0.260 ± SD 0.003 (CV 0.010))* |
| best-fit stage t* of the second intermediate (Fig. 392 top) | 0.5288 t | 0.5 | E7 projection *(sweep mean 0.527 ± SD 0.002 (CV 0.004))* |
| best-fit stage t* of the third intermediate (Fig. 392 bottom) | 0.8235 t | 0.75 | E7 projection *(sweep mean 0.824 ± SD 0.002 (CV 0.002))* |
| max equidistance residual |t* - k/4| over the three stages | 0.0735 t units | 0.0 | max over k of |t*_k - k/4| *(mean residual 0.0384; d(A,B) = 0.5827 so one stage = 0.1457 Procrustes units; pairing permutation p = 0.167)* |
| mean full Procrustes distance, dense interpolation vs drawn intermediate | 0.116 Procrustes distance | 0.2235 | E2/E4: TPS phi_t applied to the Archaeopteryx configuration, full Procrustes fit onto the drawn stage, mean over stages 1-3 *(ratio vs endpoint control 0.519 (sweep range 0.511-0.914); affine interpolation 0.2181; 9-anchor coarse grid 0.1938; correspondence-shift null p = 0.0104, z = -4.4)* |
| mean IoU of filled outlines, dense interpolation vs drawn intermediate | 0.6178 IoU (intersection/union) | 0.5343 | dense Archaeopteryx outline carried through phi_t, similarity-superimposed on the drawn stage, rasterised 400x400 *(affine 0.546; 9-anchor coarse grid 0.587; symmetric boundary RMSE/centroid size 0.0066 vs endpoint 0.0090)* |
| Pearson r, predicted vs drawn landmark coordinates (mean over stages 1-3) | 0.9931 r | 0.9722 | corrcoef of flattened Procrustes-aligned coordinates *(RMSE/centroid size 0.0118 vs endpoint 0.0227; KS (pred residuals < endpoint residuals) p = 5.5e-06, 1.1e-22, 3.0e-09)* |
| min det(D phi_t) inside the Archaeopteryx form, 9-anchor coarse grid, t in [0, 1] | 0.2341 det (1 = identity) | 0.0 | central-difference Jacobian of the TPS phi_t on reference-grid points inside the dense outline, min over 21 stages *(dense semilandmark TPS folds: min det -10.36 inside the form (-41.9 on the padded 9x8 grid))* |
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

Pass 2 of the 12-step loop 2026-10-28-t1917-054-on-the-theory-of-transformatio-loop (branch cursor/t1917-054-archaeopteryx-pelvis-01ab). Self-test 16/16 in 2.7 s; full run including the 24-setting sweep 28 s. The comparison PNG is regenerated by the script and is git-ignored by repo policy (docs/media/*.png); the GIF is tracked.
