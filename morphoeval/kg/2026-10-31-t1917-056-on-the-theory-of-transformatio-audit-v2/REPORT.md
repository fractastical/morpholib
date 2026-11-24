# MorphoEval report — audit `2026-10-31-t1917-056-on-the-theory-of-transformatio-audit-v2`

Rubric **0.1.0** · checkpoint `local` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-31-t1917-056-on-the-theory-of-transformatio-audit -> 2026-10-31-t1917-056-on-the-theory-of-transformatio-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Eohippus→Equus skull transformation (Fig. 401-402) (`thompson/claims/t1917-056_on-the-theory-of-transformations.py`) | 84 | 58 | 76 | 78 | 86 | 80 | 78 | 81.3 | 65.5 | 81.2 | 75.0 | **75.0** | mostly faithful |

## Findings and remediation

### Thompson (1917) Eohippus→Equus skull transformation (Fig. 401-402)  —  overall 75.0 (mostly faithful)

Pass 2: Heilmann stage RMSE ~0.01 on TPS phi_t; equidistance residuals small; homologous TPS beats shuffled labels; KS + lambda sweep + dual specimen segmentation. Verdict mixed — diagram interpolation supported but Dryad fossil GM series not loaded (partial test).

- **Model fidelity to source: 84** — Full E2/E4/E6 pipeline with TPS lambda sweep and ablation; Lean Loop_245 linked in math_kg.  
  evidence: `thompson/claims/t1917-056_on-the-theory-of-transformations.py` L205–240
- **Morphological realism / specimen match: 58** — Otsu segmentation on both gallery specimens; landmark-in-mask fraction exported; zebra proxy.  
  evidence: `thompson/claims/t1917-056_on-the-theory-of-transformations.py` L225–245
- **Parameter expressivity: 76** — Nine-point lambda grid on primary TPS; equidistance t* search on intermediates.  
  evidence: `thompson/claims/t1917-056_on-the-theory-of-transformations.py` L214–222
- **Morphometric validation: 78** — measure_morphoeval exports RMSE, equidistance residual, KS, shuffled TPS, segmentation fractions.  
  evidence: `thompson/claims/t1917-056_on-the-theory-of-transformations.py` L265–285
- **Source and data traceability: 86** — SHA-256 on landmarks and photos; scout manifest records Dryad species match and fetch block.  
  evidence: `datasets/thompson_1917/t1917-056/t1917_056_on_the_theory_of_transformatio_reference.json` L38–45
- **Code-to-model traceability: 80** — Docstring, result.json verdict boundary, math_kg yaml.  
  evidence: `datasets/thompson_1917/t1917-056/result.json` L1–25
- **Reproducibility package: 78** — Deterministic self-test; --pass1 vs default paths; pinned assets enforced.  
  evidence: `thompson/claims/t1917-056_on-the-theory-of-transformations.py` L365–396

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Eohippus→Equus skull transformation (Fig. 401-402)
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
- Pillow: 11.x

Pass 2 default run: lambda sweep, shuffled ablation, KS, dual specimen segmentation. Dryad doi:10.5061/dryad.crjdfn3g3 still not vendored (HTTP 401/403).
