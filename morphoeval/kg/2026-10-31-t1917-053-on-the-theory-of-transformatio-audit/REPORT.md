# MorphoEval report — audit `2026-10-31-t1917-053-on-the-theory-of-transformatio-audit`

Rubric **0.1.0** · checkpoint `35d30f4cf` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Fig. 387: Dimorphodon -> Pteranodon skull coordinate transformation (`thompson/claims/t1917-053_on-the-theory-of-transformations.py`) | 62 | 58 | 40 | 48 | 86 | 78 | 74 | 54.7 | 54.2 | 79.1 | 61.8 | **61.8** | mostly faithful |

## Findings and remediation

### Thompson (1917) Fig. 387: Dimorphodon -> Pteranodon skull coordinate transformation  —  overall 61.8 (mostly faithful)

Pass 1 segments Fig. 387a/b skull outlines (Otsu), places orbit + 4 tip landmarks + semilandmarks, fits TPS Dimorphodon->Pteranodon. Dense-outline Procrustes reported; no null/sweep/CV yet.

- **Model fidelity to source: 62** — TPS/affine/interp_stage implemented; Lean mirror present; Thompson oblique grid not digitised; no Jacobian in Pass 1.  
  evidence: `thompson/claims/t1917-053_on-the-theory-of-transformations.py` L318–360; `morphoeval/lean/loops/Thompson1917_T1917_053.lean` L1–120; `thompson/claims/t1917-053_on-the-theory-of-transformations.py` L1–1
- **Morphological realism / specimen match: 58** — Comparison PNG from Fig. 387 segmentation; real Pteranodon photo not analysed in Pass 1.  
  evidence: `docs/media/t1917_053_on_the_theory_of_transformatio_comparison.png` L0–0; `thompson/claims/t1917-053_on-the-theory-of-transformations.py` L1–1
- **Parameter expressivity: 40** — GIF over phi_t; no parameter sweep in Pass 1.  
  evidence: `thompson/claims/t1917-053_on-the-theory-of-transformations.py` L400–430; `thompson/claims/t1917-053_on-the-theory-of-transformations.py` L520–540
- **Morphometric validation: 48** — Dense-outline Procrustes and controls; no null or KS in Pass 1.  
  evidence: `thompson/claims/t1917-053_on-the-theory-of-transformations.py` L250–290; `thompson/claims/t1917-053_on-the-theory-of-transformations.py` L1–1
- **Source and data traceability: 86** — SHA-256 pins, IA page, Dryad metadata, OpenAlex logged in scout manifest.  
  evidence: `datasets/thompson_1917/t1917-053/t1917_053_on_the_theory_of_transformatio_reference.json` L1–60
- **Code-to-model traceability: 78** — Docstring + math_kg_loops YAML link equations to Lean.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-053.yaml` L1–40
- **Reproducibility package: 74** — self_test, pinned hashes, --pass1 CLI; PNG git-ignored by policy.  
  evidence: `thompson/claims/t1917-053_on-the-theory-of-transformations.py` L490–510

Remediation:
- {'id': 'R1', 'item': 'Parameter sweep n_semi x lambda; shuffled-landmark null; KS on residuals'}
- {'id': 'R2', 'item': 'Specimen CV on pteranodon_longiceps_ypm1177_vendored.jpg; Jacobian no-fold on grid'}
- {'id': 'R3', 'item': 'Score dense outlines (landmark-exact TPS is tautological on anchors)'}

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

- python: 3.12
