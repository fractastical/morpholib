# MorphoEval report — audit `2026-10-28-t1917-033-on-leaf-arrangement-or-phyllot-audit-v2`

Rubric **0.1.0** · checkpoint `f1d96f32ba3f00517c53e9cd007634306cc8d4f4` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-28-t1917-033-on-leaf-arrangement-or-phyllot-audit -> 2026-10-28-t1917-033-on-leaf-arrangement-or-phyllot-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-033 golden divergence & Fibonacci parastichies (`thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py`) | 86 | 78 | 80 | 77 | 85 | 82 | 84 | 84.0 | 77.6 | 83.8 | 81.4 | **81.4** | faithful |

## Findings and remediation

### Thompson (1917) T1917-033 golden divergence & Fibonacci parastichies  —  overall 81.4 (faithful)

Pass 2 remediation addresses all v1 audit items. Empirical divergence alpha_emp=134.1 deg (Vogel-fit inversion) lies within 4 deg of golden angle; verdict supported in result.json.

- **Model fidelity to source: 86** — Thompson rationals, Vogel spiral, parastichy cascade, and three Lean 4 theorems aligned to claim.  
  evidence: `morphoeval/lean/loops/Thompson1917_T1917_033.lean` L28–72; `thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py` L147–155
- **Morphological realism / specimen match: 78** — DoG segmentation on pinned Helianthus photo with Vogel-fit divergence 134.1 deg and overlay in Panel A.  
  evidence: `thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py` L84–110; `datasets/thompson_1917/t1917-033/result.json` L1–20
- **Parameter expressivity: 80** — Full sweep across Thompson rationals plus golden angle with NND-CV comparison panel.  
  evidence: `thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py` L112–130
- **Morphometric validation: 77** — RMSE, Pearson r, KS p-value, and ablation golden vs 90 deg in measure_morphoeval().  
  evidence: `thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py` L157–195; `thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py` L290–330
- **Source and data traceability: 85** — Scout manifest, reference JSON, paper_citations.bib, specimen SHA-256 in result.json.  
  evidence: `datasets/thompson_1917/t1917-033/scout_manifest.json` L1–40
- **Code-to-model traceability: 82** — T1917_PASS2 flag documents Pass 1 vs Pass 2 behavior; claim_id in all artifacts.  
  evidence: `thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py` L71–73
- **Reproducibility package: 84** — Self-test verifies SHA-256, cascade, ablation; PNG, GIF, result.json emitted.  
  evidence: `thompson/claims/t1917-033_on-leaf-arrangement-or-phyllotaxis.py` L318–340

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-033 golden divergence & Fibonacci parastichies
- **spec_fingerprint** (semantic model + inputs): `76e623bd0fb2355aebb0e2e097002b099b16ff9048a213066b74c6c94c38eddc`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-033/helianthus_annuus_capitulum_specimen.jpg` (`c76483eea3e564fe…`), `datasets/thompson_1917/t1917-033/paper_citations.bib` (`37c65304ef58f6d6…`), `datasets/thompson_1917/t1917-033/t1917_033_on_leaf_arrangement_or_phyllot_reference.json` (`a83fc6bfda2f2469…`)
- `docs/media/t1917_033_on_leaf_arrangement_or_phyllot_comparison.png` — artifact_sha256 `a41a664a49552536…`, 1,185,844 B, 1800×1500 px

| measurement | value | reference | method |
|---|---:|---|---|
| empirical_divergence_angle_deg | 134.1 deg | 137.50776405003785 | Vogel-fit lattice inversion on DoG-segmented specimen centroids |
| vogel_fit_rmse_px | 11.228132456981662 px | — | Nearest-neighbor RMSE vs golden Vogel spiral aligned by radial scale |
| vogel_fit_pearson_r_radial | 0.9914348611330422 1 | — | Pearson r on sorted radial coordinates |
| within_thompson_golden_bracket | 1.0 bool | — | 135-138.5 deg or within 4 deg of golden angle |
| ablation_golden_beats_90deg_nnd_cv | 1.0 bool | — | Golden angle NND-CV lower than 90 deg negative control |
| ks_golden_vs_120deg_nnd_pvalue | 2.8044233222948308e-142 p | — | Two-sample KS on interior nearest-neighbor distances |

## Environment at checkpoint


Pass 2 with T1917_PASS2=1 — lattice-fit inversion, sweep, ablation, KS.
