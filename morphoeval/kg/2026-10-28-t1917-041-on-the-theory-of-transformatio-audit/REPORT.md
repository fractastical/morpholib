# MorphoEval report — audit `2026-10-28-t1917-041-on-the-theory-of-transformatio-audit`

Rubric **0.1.0** · checkpoint `f1d96f32ba3f00517c53e9cd007634306cc8d4f4` · assessor `pkr:model/composer-2.5` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) radial-cordate leaf transformation claim T1917-041 (`thompson/claims/t1917-041_on-the-theory-of-transformations.py`) | 72 | 48 | 55 | 58 | 82 | 70 | 68 | 66.3 | 51.8 | 73.2 | 62.6 | **62.6** | mostly faithful |

## Findings and remediation

### Thompson (1917) radial-cordate leaf transformation claim T1917-041  —  overall 62.6 (mostly faithful)

Pass 1 fits the 8p radial-cordate family on synthetic parametric leaf pairs with strong in-sample R², but does not segment Thompson Fig. 359–360 ink outlines, perform Begonia specimen CV validation, parameter-regime sweeps, ablation controls, or formal KS tests.

- **Model fidelity to source: 72** — Implements Thompson Fig. 359 radial-cordate 8p least squares and compares to box/affine nulls on parametric outlines; Pass 1 does not digitize landmarks from the IA page plate.  
  evidence: `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L95–120; `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L300–310
- **Morphological realism / specimen match: 48** — Vendors Begonia rex Commons JPEG and IA page 752, but Pass 1 does not segment or overlay the real Begonia outline against the radial fit.  
  evidence: `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L240–260
- **Parameter expressivity: 55** — Uses fixed k/y0 grids only; no sweep across lanceolate→cordate regimes or negative ablation.  
  evidence: `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L28–28
- **Morphometric validation: 58** — Reports median R² and RMSE on synthetic pairs but omits KS tests and Thompson-figure residuals.  
  evidence: `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L230–245
- **Source and data traceability: 82** — verify_pinned_assets checks SHA-256 against reference JSON for IA page and Begonia specimen.  
  evidence: `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L35–45
- **Code-to-model traceability: 70** — Docstring cites Thompson 1917 and links to Lean module path; lacks line-level theorem map in Pass 1.  
  evidence: `morphoeval/lean/loops/Thompson1917_T1917_041.lean` L1–60
- **Reproducibility package: 68** — --self-test and deterministic PNG/GIF outputs; Pass 1 omits ablation self-checks and Begonia metrics.  
  evidence: `thompson/claims/t1917-041_on-the-theory-of-transformations.py` L338–348

Remediation:
- Digitize / segment Thompson Fig. 359–360 leaf ink outlines from IA page 752 and fit radial vs null models on extracted boundaries.
- Enable PASS2 parameter-regime k sweep, shuffle ablation control, and Begonia rex CV segmentation with RMSE/r/KS statistics.
- Write result.json verdict using real-specimen + figure metrics, not synthetic-only R².

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) radial-cordate leaf transformation claim T1917-041
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
- numpy: 2.x
- matplotlib: 3.x

Pass 1 baseline prior to Step 12 remediation.
