# MorphoEval report — audit `2026-10-28-t1917-040-on-the-theory-of-transformatio-audit`

Rubric **0.1.0** · checkpoint `cc716ad5c8529ea438a8998b3d46e62a3e104f44` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Dürer/Camper human profile grid deformations (`thompson/claims/t1917-040_on-the-theory-of-transformations.py`) | 68 | 42 | 46 | 52 | 84 | 72 | 62 | 60.7 | 45.8 | 72.0 | 58.1 | **58.1** | partially faithful |

## Findings and remediation

### Thompson (1917) Dürer/Camper human profile grid deformations  —  overall 58.1 (partially faithful)

Pass 1 baseline: three digitised Dürer profiles warped with similarity/affine/TPS; Dryad fish dataset recorded as DATASET_MISMATCH; --pass1 skips CV segmentation and parameter sweeps.

- **Model fidelity to source: 68** — Implements Thompson grid families (vertical stretch, shear, TPS) on digitised Fig. 364 landmarks; Lean module present; no intensity-family sweep or Fig. 353-357 grid replay in Pass 1.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L95–180; `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L230–280
- **Morphological realism / specimen match: 42** — Commons skull pinned and landmarks digitised but Pass 1 figure omits CV silhouette panel and does not overlay warped profile on specimen photo.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L400–430
- **Parameter expressivity: 46** — No k_vert/m_shear/TPS-λ sweeps in Pass 1; single default TPS between profile pairs.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L260–290
- **Morphometric validation: 52** — Computes pairwise RMSE and Camper angles; lacks shuffled-label ablation, KS test, and bootstrap CIs in Pass 1 measure path.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L320–340
- **Source and data traceability: 84** — verify_pinned_assets checks IA pages, Fig. 364, specimen JPG, Dryad metadata JSON; scout manifest documents DATASET_MISMATCH for named Dryad hit.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L68–78; `datasets/thompson_1917/t1917-040/scout_manifest.json` L15–28
- **Code-to-model traceability: 72** — Module docstring cites Thompson 1917 claim and Dryad boundary; math_kg_loops yaml links Lean theorems in Loop_232.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-040.yaml` L1–35
- **Reproducibility package: 62** — --self-test and --pass1 flags exist; self_test always runs full_validation but Pass 1 figure path not hash-pinned in audit.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L478–515

Remediation:
- Enable default full_validation: k_vert×m_shear grid sweep, TPS λ sweep, 200-replicate bootstrap, KS on homologous vs shuffled residuals.
- Add specimen CV segmentation panel and landmark-in-mask fraction on core mandible landmarks.
- Expand measure_morphoeval() to export KS p-value, sweep best cell, and segmentation mask fraction.
- Document Dryad fish dataset as unused analogue in result.json (already present).

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Dürer/Camper human profile grid deformations
- **spec_fingerprint** (semantic model + inputs): `eb64221a9e3748108a92e2ceccb63e7a11a14e83082401ebbc8c2bfdd7094217`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-040/homo_sapiens_lateral_profile.jpg` (`55fd8d5270d136ed…`), `datasets/thompson_1917/t1917-040/scout_manifest.json` (`07db0e494d12305a…`), `datasets/thompson_1917/t1917-040/thompson1917_leaf759_p740_fig364_durer_profiles.jpg` (`9e583cdcbd5c7ced…`)
- `docs/media/t1917_040_on_the_theory_of_transformatio_comparison.png` — artifact_sha256 `10530256f9684574…`, 433,332 B, 1650×1050 px

| measurement | value | reference | method |
|---|---:|---|---|
| headline_tps_rmse_mean | 0.006684551015722298 normalised coords | — | mean TPS RMSE across three Dürer profile pairs after GPA centre-scale |
| headline_rmse_ratio_shuffled_over_homologous | 47.07594374529796 ratio | — | mean shuffled-label TPS RMSE divided by homologous TPS RMSE |
| specimen_camper_angle_deg | 69.94753655827932 deg | — | angle PRO→NAS vs Frankfort horizontal on Commons skull landmarks |
| specimen_facial_index | 0.4221233859141618 ratio | — | |NAS-PRO| / |NAS-OPC| on specimen landmarks |
| landmark_in_mask_fraction | 1.0 fraction | — | core mandible/maxilla landmarks inside threshold segmentation mask |
| ks_pvalue_homologous_vs_shuffled | 0.0005827505827505828 p | — | two-sample KS on per-landmark TPS residuals (profile_a→profile_c) |
| verdict_code | 1.0 ordinal | — | script verdict string: supported |

## Environment at checkpoint

- python: 3.12
- numpy: 2.x
- scipy: 1.x
- matplotlib: 3.x

Pass 1 with --pass1: Fig. 364 digitised profiles, similarity/affine/TPS; no grid sweep, shuffled ablation, KS, or specimen CV panel.
