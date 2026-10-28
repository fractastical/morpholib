# MorphoEval report — audit `2026-10-28-t1917-055-on-the-theory-of-transformatio-audit-v2`

Rubric **0.1.0** · checkpoint `530b3b50c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-28-t1917-055-on-the-theory-of-transformatio-audit -> 2026-10-28-t1917-055-on-the-theory-of-transformatio-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Figs. 394-400: rhinoceros and allied skulls by coordinate-grid transformation (Osborn) (`thompson/claims/t1917-055_on-the-theory-of-transformations.py`) | 85 | 77 | 79 | 84 | 88 | 85 | 81 | 83.0 | 79.6 | 84.4 | 82.1 | **82.1** | faithful |

## Findings and remediation

### Thompson (1917) Figs. 394-400: rhinoceros and allied skulls by coordinate-grid transformation (Osborn)  —  overall 82.1 (faithful)

Pass 2 (v2), overall 82.7/100 (up from 67.0 in Pass 1). Nine panels digitised from SHA-256-pinned IA crops; primary Hyrachyus (394) -> Aceratherium (395) with 40 matched grid nodes at defaults. TPS applied to the Hyrachyus boundary: Procrustes 0.370 vs identity 0.377 vs affine 0.360; IoU 0.570 (affine 0.587); Pearson r 0.976; chamfer/cs 0.015; shuffled-grid null worse than nominal. Parameter sweep: 36 primary-only settings; 27/27 with >=40 matches beat identity on Procrustes; affine beats TPS throughout those rows. min det D phi_t inside skull 0.259. Secondary 3/6 beat identity. Ceratotherium CV descriptive. Verdict mixed.

- **Model fidelity to source: 85** — E2/E6, E4, E5 and E7 implemented as stated; eight Lean theorems compile with 0 sorry; Pass 2 adds numerical E1 via jacobian_min_det/check_no_fold on the primary TPS (min det inside skull 0.259 over 21 stages). Remaining deductions: Fig. 394-395 grid ink not digitised line-by-line; affine map fits the primary outline slightly better than the TPS on Procrustes, so the strict "smooth non-affine grid" reading is not numerically dominant.  
  evidence: `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L184–227; `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L280–303; `morphoeval/lean/loops/Thompson1917_T1917_055.lean` L16–102; `datasets/thompson_1917/t1917-055/result.json` L24–40
- **Morphological realism / specimen match: 77** — Six-panel comparison figure: Fig. 394/395 with detected grids, primary overlay titled with Procrustes/IoU/r, secondary pair bar chart, four-bar control panel (TPS/identity/affine/shuffled), Ceratotherium panel with bone mask contour. IoU 0.570 quantifies visual overlap. Automatic matching and coarse rabbit/horse panels remain limitations; specimen photo is rhinoceros not Hyrachyus fossil.  
  evidence: `docs/media/t1917_055_on_the_theory_of_transformatio_comparison.png` L0–0; `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L628–697; `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L357–373
- **Parameter expressivity: 79** — Primary-only sweep over 36 settings (maxd x lambda x n_boundary) is executed and written to result.json; 27 rows have >=40 matches and all beat identity on Procrustes; lambda and maxd visibly trade match count vs fit. Growth GIF still shows t along the primary deformation. Deduction: affine beats TPS across swept >=40-match rows; secondary pairs fixed at defaults only.  
  evidence: `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L557–585; `datasets/thompson_1917/t1917-055/result.json` L52–55; `datasets/thompson_1917/t1917-055/result.json` L400–407
- **Morphometric validation: 84** — Primary pair reports Procrustes, IoU, Pearson r, RMSE/cs, chamfer/cs vs identity, affine, and shuffled-grid null; headline numbers pinned in result.json; secondary six pairs with the same compare_outlines block; sweep summarised in measure_morphoeval. Mixed verdict is explicit: TPS beats identity but not affine. Deductions: shuffled null is single-seed permutation not exhaustive; no KS test; part C3 rabbit/horse remain weak; no real 3D GM for part C4.  
  evidence: `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L264–277; `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L458–473; `datasets/thompson_1917/t1917-055/result.json` L23–40; `datasets/thompson_1917/t1917-055/result.json` L17–21
- **Source and data traceability: 88** — Unchanged strength from Pass 1: six inputs SHA-256 pinned and fail-closed at runtime; IA crops, reference JSON quotes, Commons Ceratotherium metadata, Osborn/Bookstein bibliography; portfolio mismatch documented in result.json. Deduction: Thompson 1917 has no DOI.  
  evidence: `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L111–122; `datasets/thompson_1917/t1917-055/t1917_055_on_the_theory_of_transformatio_reference.json` L13–52; `datasets/thompson_1917/t1917-055/scout_manifest.json` L0–0
- **Code-to-model traceability: 85** — Module docstring lists Pass 2 outputs (sweep, null, IoU, chamfer, Jacobian, specimen); analyze_pair and write_result wire all controls into result.json; math_kg_loops documents grid_matching and controls in model_spec. Deduction: extended fit statistics have no separate equation ids in E1-E7.  
  evidence: `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L36–46; `morphoeval/math_kg_loops/thompson1917-t1917-055.yaml` L1–35; `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L748–785
- **Reproducibility package: 81** — Deterministic (SEED=1917), fail-closed hashes, self-test through primary metrics and specimen bounds, CLI --no-sweep / --no-specimen / --no-gif, full pipeline writes comparison PNG, GIF, and result.json with 36 sweep rows. Deductions: requirements.txt lower bounds only; PNG git-ignored.  
  evidence: `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L811–855; `thompson/claims/t1917-055_on-the-theory-of-transformations.py` L858–878

Remediation:
- R1 (open): independent second assessor from a different provider — this round is builder-scored
- R7: hand-digitise grid line intersections on Figs. 394-395 and compare with automatic peak matching
- R8: sparse-anchor or conformal grid matching Thompson's occipital Fig. 396 panels to test whether non-affine phi is required once correspondences are fixed
- R9: assemble open 2-D skull landmarks for perissodactyl intermediates to test secondary pairs beyond diagram self-consistency

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=2.5]** `parameter_sweep` → `parameter-expressivity`: Primary-only sweep over maxd, lambda, n_boundary; stability vs identity and affine
- **[RESOLVED · P=2.5]** `morphometric_validation` → `morphometric-validation`: Shuffled-grid null; IoU, Pearson r, chamfer RMSE
- **[RESOLVED · P=2.5]** `specimen_comparison` → `specimen-match`: Ceratotherium bone-vs-matrix CV and expanded comparison figure
- **[RESOLVED · P=2.5]** `model_equation` → `model-fidelity`: Numerical Jacobian no-fold check on primary TPS
- **[OPEN · P=2.5]** `independent_assessor` → `model-fidelity`: Independent second assessor from a different provider
- **[OPEN · P=2.5]** `digitisation` → `specimen-match`: Hand-labelled grid correspondences or independent digitisation of Fig. 394-395 grid ink
- **[OPEN · P=2.5]** `morphometric_validation` → `morphometric-validation`: Resolve affine-vs-TPS ambiguity: Thompson-like sparse anchor grid or literature 2-D landmarks for Osborn skulls

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Figs. 394-400: rhinoceros and allied skulls by coordinate-grid transformation (Osborn)
- **spec_fingerprint** (semantic model + inputs): `1799c9e22b84705ed0af5fe44b5529c9ffab5f336a64446d6582253a48d2e8c1`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-055/ceratotherium_simum_skull_mnhn_1400.jpg` (`7ed8c3776d2ab62a…`), `datasets/thompson_1917/t1917-055/ongrowthform1917thom_p761_leaf0781_figs394-395.jpg` (`978a7d9039629c42…`), `datasets/thompson_1917/t1917-055/ongrowthform1917thom_p762_leaf0782_fig396.jpg` (`7a8a134814786620…`), `datasets/thompson_1917/t1917-055/ongrowthform1917thom_p763_leaf0783_figs397-398.jpg` (`229d3074b7e141bf…`), `datasets/thompson_1917/t1917-055/ongrowthform1917thom_p764_leaf0784_figs399-400.jpg` (`5dcb998220348ea5…`), `datasets/thompson_1917/t1917-055/t1917_055_on_the_theory_of_transformatio_reference.json` (`ee9eba19f703221d…`)
- `datasets/thompson_1917/t1917-055/scout_manifest.json` — artifact_sha256 `3a991bddcb8e4576…`, 3,922 B, 
- `docs/media/t1917_055_on_the_theory_of_transformatio_comparison.png` — artifact_sha256 `65d3271d99b1aa5c…`, 540,693 B, 1980×1100 px

| measurement | value | reference | method |
|---|---:|---|---|
| matched grid landmarks (Hyrachyus -> Aceratherium) | 40 count | — | normalized-bbox greedy NN, maxd=0.12 |
| Procrustes distance, TPS outline vs Aceratherium | 0.3702 Procrustes distance | — | TPS on matched grids applied to Hyrachyus boundary *(identity control 0.3766)* |
| IoU filled outlines, TPS vs Aceratherium | 0.5704 IoU | — | similarity-aligned dense boundaries, raster 400² |
| parameter sweep settings | 36 count | — | maxd x lam x n_boundary grid |

## Environment at checkpoint

- python: 3.12
- numpy: 1.26.4
- scipy: 1.x (pip, requirements.txt lower bound >=1.7)
- scikit-image: >=0.19
- matplotlib: >=3.3
- Pillow: >=8.0
- lean: 4.35.0 (elan stable)

Pass 2 of the 12-step loop 2026-10-28-t1917-055-on-the-theory-of-transformatio-loop. Self-test includes IoU/chamfer sanity, primary >=40 matches, TPS beats identity, Ceratotherium segmentation bounds; full run writes 36-row parameter_sweep to result.json. Comparison PNG git-ignored (docs/media/*.png).
