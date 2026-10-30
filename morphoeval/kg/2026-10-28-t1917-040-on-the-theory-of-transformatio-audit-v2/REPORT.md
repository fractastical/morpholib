# MorphoEval report — audit `2026-10-28-t1917-040-on-the-theory-of-transformatio-audit-v2`

Rubric **0.1.0** · checkpoint `cc716ad5c8529ea438a8998b3d46e62a3e104f44` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-28-t1917-040-on-the-theory-of-transformatio-audit -> 2026-10-28-t1917-040-on-the-theory-of-transformatio-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Dürer/Camper human profile grid deformations (Pass 2) (`thompson/claims/t1917-040_on-the-theory-of-transformations.py`) | 82 | 58 | 74 | 80 | 86 | 76 | 78 | 79.3 | 66.2 | 80.1 | 74.3 | **74.3** | mostly faithful |

## Findings and remediation

### Thompson (1917) Dürer/Camper human profile grid deformations (Pass 2)  —  overall 74.3 (mostly faithful)

Full validation on digitised Fig. 364 profiles: homologous TPS RMSE ≪ label-shuffled null (ratio ≫ 1); Camper φ ordering preserved across profiles; specimen CV mask + core landmark in-mask fraction ≥ 0.8; KS rejects shuffled residual equality (p ≪ 0.05).

- **Model fidelity to source: 82** — Thompson stretch/shear/TPS on Dürer landmarks with 5×5 grid sweep and TPS λ ablation; Lean Loop_232 proves rotation invariance, vertical stretch, and shear y-preservation.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L260–310; `morphoeval/lean/loops/Thompson1917_T1917_040.lean` L1–55
- **Morphological realism / specimen match: 58** — Segments Commons rendered skull (not museum photo) and plots silhouette; warped profile not overlaid on specimen radiograph; upper calvaria landmarks off bone on this model.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L195–220; `datasets/thompson_1917/t1917-040/scout_manifest.json` L44–50
- **Parameter expressivity: 74** — Sweeps k_vert and m_shear on Thompson Fig. 353/357 ranges plus TPS regularisation λ grid.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L268–285
- **Morphometric validation: 80** — measure_morphoeval exports verdict, RMSE ratio, Camper angles, KS p-value, landmark-in-mask; self_test asserts shuffled RMSE ≥ homologous and ablation ordering.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L330–350; `datasets/thompson_1917/t1917-040/t1917_040_metrics.json` L1–40
- **Source and data traceability: 86** — IA Fig. 364/366 and Commons specimen SHA-256 pinned; Dryad metadata JSON pinned with download-blocker note; result.json states DATASET_MISMATCH.  
  evidence: `datasets/thompson_1917/t1917-040/t1917_040_on_the_theory_of_transformatio_reference.json` L30–45
- **Code-to-model traceability: 76** — Docstring maps E1-E5 equations; math_kg_loops yaml links Lean theorems to warp code paths.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-040.yaml` L10–35
- **Reproducibility package: 78** — --self-test verifies pins, determinism, ablation ordering, segmentation fraction; --pass1 retains reproducible baseline.  
  evidence: `thompson/claims/t1917-040_on-the-theory-of-transformations.py` L478–495

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Dürer/Camper human profile grid deformations (Pass 2)
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
- Pillow: 10.x

Pass 2 default: grid sweep, bootstrap, KS test, specimen CV silhouette; verdict supported on digitised Dürer profiles (TPS RMSE ≪ shuffled null); Dryad cichlid dataset not used.
