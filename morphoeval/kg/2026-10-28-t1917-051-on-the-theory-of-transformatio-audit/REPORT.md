# MorphoEval report — audit `2026-10-28-t1917-051-on-the-theory-of-transformatio-audit`

Rubric **0.1.0** · checkpoint `f1d96f32ba3f00517c53e9cd007634306cc8d4f4` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Fig. 383 crocodylian skull coordinate transformations (`thompson/claims/t1917-051_on-the-theory-of-transformations.py`) | 72 | 44 | 50 | 58 | 82 | 70 | 64 | 64.7 | 49.2 | 71.6 | 60.6 | **60.6** | mostly faithful |

## Findings and remediation

### Thompson (1917) Fig. 383 crocodylian skull coordinate transformations  —  overall 60.6 (mostly faithful)

Pass 1 baseline: parses Watanabe & Slice (2014) NTS landmarks, GPA, species-mean lateral shapes for cropor/croacu/allmis; fits similarity, affine, Moebius, and TPS between pairs; headline cross-species RMSE / within-species LOO ratio ≈ 1.40 → mixed verdict; Notosuchus not testable on Dryad set.

- **Model fidelity to source: 72** — Implements Thompson-style similarity, affine, Moebius coaxial, and Bookstein TPS maps on 78 landmarks with pinned Dryad ingestion; omits Pass-2 Moebius c sweep and zero-reflection grid visualization against Thompson IA Fig. 383.  
  evidence: `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L175–220; `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L248–290
- **Morphological realism / specimen match: 44** — Vendored CC BY 2.0 alligator skull photograph is pinned but not segmented or overlaid in Pass 1 figures; Thompson 1917 Fig. 383 page fetched (IA leaf 772) but not digitised in panels.  
  evidence: `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L330–380
- **Parameter expressivity: 50** — No CLI or sweep over Moebius c or TPS bending in Pass 1; single default c=0.12.  
  evidence: `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L488–495
- **Morphometric validation: 58** — Computes pairwise RMSE and LOO noise floor; lacks shuffled-label ablation, KS test, and specimen landmark-in-mask fraction in Pass 1 measure_morphoeval path when --pass1 used.  
  evidence: `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L292–328
- **Source and data traceability: 82** — Runtime SHA-256 verification of NTS and specimen JPG; reference JSON and scout manifest cite Dryad DOI and OpenAlex paper.  
  evidence: `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L56–68
- **Code-to-model traceability: 70** — Module docstring cites Thompson 1917 and Watanabe 2014; warp functions map to rubric families but lack line-level Lean theorem references in docstrings.  
  evidence: `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L1–16
- **Reproducibility package: 64** — --self-test and --pass1 flags exist; self_test does not verify Pass-1 vs Pass-2 feature split or two-run figure hash stability.  
  evidence: `thompson/claims/t1917-051_on-the-theory-of-transformations.py` L463–495

Remediation:
- Enable full_validation by default (remove --pass1 gate) with Moebius c grid sweep, shuffled-landmark TPS ablation, and KS comparison of within- vs between-species distance distributions.
- Segment alligator_mississippiensis_skull.jpg (Otsu + largest component) and report landmark-in-mask fraction against allmis mean shape.
- Expand measure_morphoeval() to export sweep, ablation, KS, and segmentation metrics; add self_test assertions for shuffled > homologous TPS RMSE.
- Overlay Thompson IA Fig. 383 crop or specimen inset on comparison figure.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Fig. 383 crocodylian skull coordinate transformations
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

Pass 1 baseline with --pass1 flag: Dryad 10.5061/dryad.14fn1 landmarks pinned; outputs docs/media/t1917_051_on_the_theory_of_transformatio_comparison.png and growth.gif.
