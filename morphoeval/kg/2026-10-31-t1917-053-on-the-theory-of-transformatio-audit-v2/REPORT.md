# MorphoEval report — audit `2026-10-31-t1917-053-on-the-theory-of-transformatio-audit-v2`

Rubric **0.1.0** · checkpoint `35d30f4cf` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-31-t1917-053-on-the-theory-of-transformatio-audit -> 2026-10-31-t1917-053-on-the-theory-of-transformatio-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Fig. 387: Dimorphodon -> Pteranodon skull coordinate transformation (`thompson/claims/t1917-053_on-the-theory-of-transformations.py`) | 72 | 68 | 70 | 74 | 88 | 80 | 78 | 71.3 | 70.2 | 81.9 | 74.1 | **74.1** | mostly faithful |

## Findings and remediation

### Thompson (1917) Fig. 387: Dimorphodon -> Pteranodon skull coordinate transformation  —  overall 74.1 (mostly faithful)

Pass 2: dense-outline Procrustes (128-pt resample), IoU/chamfer/r/RMSE, 24-setting sweep, shuffled-landmark null (p≈0.80), specimen Otsu CV on YPM1177 photo, Jacobian min det -15.3 (grid folds). TPS Procrustes 0.726 vs similarity 0.663 vs affine 0.625 at default n_semi=20; coarse n_semi=8 TPS beats similarity slightly. Verdict mixed (high r=0.977, IoU=0.51). Foth Dryad clade frame file blocked (401).

- **Model fidelity to source: 72** — Lean theorems + TPS pipeline; numerical Jacobian shows grid fold at t=1.  
  evidence: `morphoeval/lean/loops/Thompson1917_T1917_053.lean` L24–130; `thompson/claims/t1917-053_on-the-theory-of-transformations.py` L380–395
- **Morphological realism / specimen match: 68** — Fig. 387 CV segmentation + YPM1177 specimen Otsu CV (descriptive).  
  evidence: `thompson/claims/t1917-053_on-the-theory-of-transformations.py` L310–316; `datasets/thompson_1917/t1917-053/result.json` L20–35
- **Parameter expressivity: 70** — 24-setting sweep over n_semi and TPS lambda recorded in result.json.  
  evidence: `datasets/thompson_1917/t1917-053/result.json` L39–80
- **Morphometric validation: 74** — Dense Procrustes, IoU, r, shuffled-null, KS; null p=0.80 not significant.  
  evidence: `thompson/claims/t1917-053_on-the-theory-of-transformations.py` L250–310; `datasets/thompson_1917/t1917-053/result.json` L30–31
- **Source and data traceability: 88** — Pinned IA + Commons + Dryad metadata; Foth file blocked not mismatched.  
  evidence: `datasets/thompson_1917/t1917-053/scout_manifest.json` L1–50
- **Code-to-model traceability: 80** — measure_morphoeval + KG YAML law entries.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-053.yaml` L1–45
- **Reproducibility package: 78** — Deterministic self_test; full Pass 2 run ~8s.  
  evidence: `thompson/claims/t1917-053_on-the-theory-of-transformations.py` L490–553

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Fig. 387: Dimorphodon -> Pteranodon skull coordinate transformation
- **spec_fingerprint** (semantic model + inputs): `2f185f72ad54c895b5a22263bae433c97f4c9067a6ce1e7845bfc5bc0a745ff5`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-053/dryad_4nr4cf37_metadata.json` (`2b2c64fd9aacd5b1…`), `datasets/thompson_1917/t1917-053/nie1905_dimorphodon_pteranodon_comparison.png` (`18604cc23a3c7635…`), `datasets/thompson_1917/t1917-053/ongrowthform1917thom_p756_leaf0775_fig387.jpg` (`faf3c2cff141abad…`), `datasets/thompson_1917/t1917-053/pteranodon_longiceps_ypm1177_vendored.jpg` (`5f3272484ee1b584…`), `datasets/thompson_1917/t1917-053/t1917_053_on_the_theory_of_transformatio_reference.json` (`9e1babca8eff80e8…`)
- `docs/media/t1917_053_on_the_theory_of_transformatio_comparison.png` — artifact_sha256 `8826b70c2a275dab…`, 135,226 B, 2100×750 px

| measurement | value | reference | method |
|---|---:|---|---|
| procrustes_baseline_dim_pte | 0.4570418585781493 1 | — | GPA full Procrustes distance |
| tps_procrustes_dense | 0.726455500283346 1 | — | 128-pt resampled dense outline |
| similarity_procrustes_dense | 0.6633276105738255 1 | — | similarity control |
| affine_procrustes_dense | 0.625435008069873 1 | — | affine control |
| pearson_r_tps | 0.9770588292502028 1 | — | aligned dense landmarks |
| iou_tps | 0.5126508831403795 1 | — | filled outline IoU |
| shuffled_landmark_null_p | 0.8009950248756219 1 | — | 200 cyclic shifts |
| jacobian_min_det_t1 | -15.311841567221414 1 | — | min det D phi_t on 8x4 grid |

## Environment at checkpoint

