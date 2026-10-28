# MorphoEval report — audit `2026-10-28-t1917-041-on-the-theory-of-transformatio-audit-v2`

Rubric **0.1.0** · checkpoint `f1d96f32ba3f00517c53e9cd007634306cc8d4f4` · assessor `pkr:model/composer-2.5` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-28-t1917-041-on-the-theory-of-transformatio-audit -> 2026-10-28-t1917-041-on-the-theory-of-transformatio-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) radial-cordate leaf transformation claim T1917-041 (Pass 2) (`thompson/claims/t1917-041_on-the-theory-of-transformations.py`) | 88 | 76 | 84 | 82 | 90 | 85 | 86 | 86.7 | 78.2 | 87.1 | 83.4 | **83.4** | faithful |

## Findings and remediation

### Thompson (1917) radial-cordate leaf transformation claim T1917-041 (Pass 2)  —  overall 83.4 (faithful)

Pass 2 adds k-regime sweeps, shuffle ablation, Begonia rex CV segmentation (r, RMSE, KS), Thompson IA page ink digitization, and writes datasets/thompson_1917/t1917-041/result.json. Synthetic pairwise R² remains high; real Begonia fit is moderate (~55% R²); crude figure digitization is weak (~6% R²), yielding a mixed empirical verdict on the 1917 leaf claim.

- **Model fidelity to source: 88** — Full 8p radial-cordate map with ablation and k sweep; Thompson plate vectorized from IA page 752.  
  evidence: `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L95–120; `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L243–280
- **Morphological realism / specimen match: 76** — Segments Begonia rex Commons specimen and overlays radial fit; Thompson figure ink extracted.  
  evidence: `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L155–195
- **Parameter expressivity: 84** — Sweeps k across 9 values per pair; shuffle ablation collapses R² from ~99% to ~4%.  
  evidence: `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L205–225; `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L228–238
- **Morphometric validation: 82** — measure_morphoeval exports median R², Begonia r/RMSE/KS, and Thompson digitized R².  
  evidence: `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L310–370
- **Source and data traceability: 90** — Runtime SHA-256 verification against reference JSON for IA and Begonia assets.  
  evidence: `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L35–45
- **Code-to-model traceability: 85** — Lean module Thompson1917_T1917_041.lean linked via math_kg_loops yaml.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-041.yaml` L1–35
- **Reproducibility package: 86** — --self-test checks ablation ordering and Begonia R² floor; PNG+GIF outputs pinned.  
  evidence: `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L430–445

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) radial-cordate leaf transformation claim T1917-041 (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `2874108b4c92e842aade03eb1a10b668c06797f3484cd9de80e369cfd15118ee`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-041/t1917_041_on_the_theory_of_transformatio_reference.json` (`557e1ed2a1934ad1…`)
- `docs/media/t1917_041_on_the_theory_of_transformatio_comparison.png` — artifact_sha256 `793aa09dbe676a86…`, 393,749 B, 1680×1120 px

| measurement | value | reference | method |
|---|---:|---|---|
| median_r2_radial_8p_pct | 99.416 percent | — | 8p radial-cordate least squares on registered synthetic leaf pairs |
| median_r2_box_pct | 97.319 percent | — | Similarity (4p) null on same pairs |
| ablation_r2_true_minus_shuffled_pct | 95.8567 percent | — | Landmark shuffle ablation on lanceolate→ovate pair |
| begonia_r2_radial_8p_pct | 54.7093 percent | — | Begonia rex CV outline vs cordate template |
| begonia_rmse | 0.559161 norm | — | RMSE after Procrustes registration |
| begonia_pearson_r | 0.739657 1 | — | Pearson r on coordinates |
| begonia_ks | 0.234466 1 | — | KS on residual distance CDF |
| thompson_fig359_r2_radial_8p_pct | 6.2372 percent | — | Digitized IA page 752 ink outline vs ovate template |

## Environment at checkpoint

- python: 3.12

Pass 2 after Step 12 remediation (PASS2 enabled).
