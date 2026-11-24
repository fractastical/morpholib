# MorphoEval report — audit `2026-10-31-t1917-056-on-the-theory-of-transformatio-audit`

Rubric **0.1.0** · checkpoint `local` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Eohippus→Equus equid skull coordinate deformation (`thompson/claims/t1917-056_on-the-theory-of-transformations.py`) | 70 | 38 | 44 | 50 | 82 | 74 | 64 | 61.3 | 42.5 | 72.7 | 57.2 | **57.2** | partially faithful |

## Findings and remediation

### Thompson (1917) Eohippus→Equus equid skull coordinate deformation  —  overall 57.2 (partially faithful)

Pass 1 baseline: 18-landmark digitisation; TPS phi_t vs Heilmann stages; Dryad equid GM identified but not fetched; --pass1 skips CV segmentation and parameter sweeps.

- **Model fidelity to source: 70** — Implements Thompson E2/E4/E6 stage interpolation and TPS warp on digitised endpoints; Lean Loop_245 present; Dryad 3D GM series not loaded.  
  evidence: `thompson/claims/t1917-056_on-the-theory-of-transformations.py` L118–180; `morphoeval/lean/loops/Thompson1917_T1917_056.lean` L1–75
- **Morphological realism / specimen match: 38** — Gallery photos pinned but Pass 1 figure omits CV silhouette overlay; zebra proxy not E. caballus.  
  evidence: `thompson/claims/t1917-056_on-the-theory-of-transformations.py` L280–310
- **Parameter expressivity: 44** — No TPS lambda sweep or equidistance grid in Pass 1 CLI path.  
  evidence: `thompson/claims/t1917-056_on-the-theory-of-transformations.py` L205–225
- **Morphometric validation: 50** — Stage RMSE and equidistance residuals computed; Pass 1 measure path lacks KS and shuffled ablation exports.  
  evidence: `thompson/claims/t1917-056_on-the-theory-of-transformations.py` L265–285
- **Source and data traceability: 82** — verify_pinned_assets checks landmarks JSON and specimen JPGs; scout documents Dryad fetch block.  
  evidence: `datasets/thompson_1917/t1917-056/scout_manifest.json` L12–28
- **Code-to-model traceability: 74** — Module docstring cites claim, Dryad boundary, and Lean/math_kg yaml link.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-056.yaml` L1–35
- **Reproducibility package: 64** — --self-test and --pass1 flags; deterministic measure_morphoeval; Dryad binary absent.  
  evidence: `thompson/claims/t1917-056_on-the-theory-of-transformations.py` L365–396

Remediation:
- Default full_validation: TPS lambda sweep, shuffled-label ablation, KS on pairwise distances.
- Add specimen CV segmentation panel and landmark-in-mask fraction on equid gallery photo.
- Document Dryad fetch failure vs species match in result.json (partial test boundary).
- Attempt IA page vendoring for Fig. 401-402 when archive.org accessible.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Eohippus→Equus equid skull coordinate deformation
- **spec_fingerprint** (semantic model + inputs): `885385f32cc021aefedec35731f59225d662ddf2cdd8959acb551369644bc14a`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-056/scout_manifest.json` (`cb62934529ec73bd…`), `datasets/thompson_1917/t1917-056/thompson_fig401_402_landmarks.json` (`e2d755701d463a2e…`)
- `docs/media/t1917_056_on_the_theory_of_transformatio_comparison.png` — artifact_sha256 `642c269d14e132d2…`, 151,288 B, 1800×1200 px

| measurement | value | reference | method |
|---|---:|---|---|
| rmse_tps_eo_to_eq | 1.542766483606174e-16 shape units | — | TPS fit Eohippus→Equus endpoints (Fig. 401) |
| median_stage_rmse | 0.009686452518830352 shape units | — | RMSE of phi_t(Eohippus) vs Heilmann drawn stages |
| median_equidist_residual | 0.0 t-units | — | |t* - k/4| minimising RMSE to drawn intermediate k |
| tps_rmse_homologous | 1.542766483606174e-16 shape units | — | TPS with homologous landmark labels |
| tps_rmse_shuffled_labels | 1.9157410903079397e-14 shape units | — | TPS ablation with shuffled Eohippus labels (seed 1917056) |
| specimen_segmentation_coverage_eo | 0.1835125 fraction | — | Otsu + largest CC on gallery Eohippus photo |
| specimen_segmentation_coverage_eq | 0.14611458333333333 fraction | — | Otsu + largest CC on gallery equid photo |
| landmark_in_mask_fraction | 0.2222222222222222 fraction | — | Equus landmarks vs equid segmentation mask |
| ks_statistic | 0.26349206349206344 D | — | KS: GPA endpoints vs TPS midpoint pairwise distances |
| ks_pvalue | 1.2433513319372065e-19 p | — | scipy.stats.ks_2samp |

## Environment at checkpoint

- python: 3.12
- numpy: 2.x
- scipy: 1.x
- matplotlib: 3.x

Pass 1 with --pass1: digitised Fig. 401-402 landmarks, similarity/affine/TPS; Dryad GM not vendored; no lambda sweep, shuffled ablation, KS, or specimen CV panel in Pass 1 run path.
