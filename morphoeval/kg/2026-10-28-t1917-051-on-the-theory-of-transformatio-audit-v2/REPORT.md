# MorphoEval report — audit `2026-10-28-t1917-051-on-the-theory-of-transformatio-audit-v2`

Rubric **0.1.0** · checkpoint `f1d96f32ba3f00517c53e9cd007634306cc8d4f4` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-28-t1917-051-on-the-theory-of-transformatio-audit -> 2026-10-28-t1917-051-on-the-theory-of-transformatio-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Fig. 383 crocodylian skull coordinate transformations (Pass 2) (`thompson/claims/t1917-051_on-the-theory-of-transformations.py`) | 80 | 58 | 72 | 78 | 86 | 74 | 76 | 77.3 | 65.5 | 78.8 | 73.0 | **73.0** | mostly faithful |

## Findings and remediation

### Thompson (1917) Fig. 383 crocodylian skull coordinate transformations (Pass 2)  —  overall 73.0 (mostly faithful)

Full validation: 78-landmark GPA species means; cross-species similarity RMSE exceeds within-species LOO noise floor by ~40% (headline ratio 1.40) → mixed Thompson verdict; TPS shuffled-label ablation RMSE > homologous; KS p≈0.001 between within- and warped-distance distributions; specimen mask covers ~24% of frame, ~35% of aligned landmarks fall inside mask.

- **Model fidelity to source: 80** — Thompson warp families plus Moebius c parameter sweep and TPS ablation; Lean 4 module pins Procrustes/similarity/affine laws; Thompson Fig. 383 IA page pinned but not digitised.  
  evidence: `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L292–320
- **Morphological realism / specimen match: 58** — Segments CC BY 2.0 alligator skull and reports landmark-in-mask fraction; comparison figure remains landmark-scatter without photo overlay or Thompson grid digitisation.  
  evidence: `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L230–246; `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L330–380
- **Parameter expressivity: 72** — Moebius c swept on [0.02, 0.28] with best-RMSE recorded; default full run exercises grid.  
  evidence: `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L296–303
- **Morphometric validation: 78** — Nine measure_morphoeval metrics including LOO noise floor, KS test, ablation, segmentation; documents Notosuchus gap explicitly in result.json.  
  evidence: `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L350–365; `datasets/thompson_1917/t1917-051/result.json` L1–30
- **Source and data traceability: 86** — SHA-256 pinned NTS and specimen; Dryad zip and scout manifest document taxon overlap.  
  evidence: `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L56–68
- **Code-to-model traceability: 74** — Header cites Thompson 1917 and Watanabe 2014; math_kg_loops yaml links Lean theorems.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-051.yaml` L10–35
- **Reproducibility package: 76** — self_test checks NTS shape, determinism, shuffled ablation ordering; --pass1 flag retains baseline path.  
  evidence: `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L463–478

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Fig. 383 crocodylian skull coordinate transformations (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `d83a1ab5f05d94feaf3f0f8e42e08bf2eef326fe6ea9418c9c49f97ae7d3bf89`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-051/WatanabeSlice_JEB_PreGPA.nts` (`4ebb3bf100ef27fe…`), `datasets/thompson_1917/t1917-051/scout_manifest.json` (`7bdead2ffdf7d38c…`)
- `docs/media/t1917_051_on_the_theory_of_transformatio_comparison.png` — artifact_sha256 `99352022cff55fe5…`, 105,148 B, 1800×1200 px

| measurement | value | reference | method |
|---|---:|---|---|
| cropor_loo_median_rmse | 0.005581773271044194 GPA units | — | median leave-one-out similarity RMSE on C. porosus specimens after GPA |
| cross_species_median_rmse_similarity | 0.00780408971751147 GPA units | — | median similarity RMSE across species-mean pairs |
| headline_rmse_ratio | 1.3981380716403664 ratio | — | cross-species median RMSE divided by within-species LOO noise floor |
| tps_rmse_homologous | 1.2103045684545723e-13 GPA units | — | TPS fit cropor→allmis with homologous landmarks |
| tps_rmse_shuffled_labels | 7.090487841036764e-12 GPA units | — | TPS ablation with shuffled source landmark labels (seed 1917) |
| specimen_segmentation_coverage | 0.24420565453334078 fraction | — | Otsu grayscale + largest connected component on Commons skull JPG |
| landmark_in_mask_fraction | 0.34615384615384615 fraction | — | Procrustes-aligned allmis mean landmarks vs segmentation mask |
| ks_statistic | 0.1853598014888338 D | — | KS test: within-species vs TPS-warped pairwise landmark distances |
| ks_pvalue | 0.0013358853821844362 p | — | two-sample KS p-value (scipy.stats.ks_2samp) |

## Environment at checkpoint

- python: 3.12
- numpy: 2.x
- scipy: 1.x
- matplotlib: 3.x
- Pillow: 10.x

Pass 2 default run (full_validation=True): Moebius c sweep, TPS label-shuffle ablation, Otsu specimen segmentation, KS distance test; result.json verdict mixed (rmse ratio ~1.40, Notosuchus absent from Dryad).
