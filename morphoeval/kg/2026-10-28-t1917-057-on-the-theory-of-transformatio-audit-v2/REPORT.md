# MorphoEval report — audit `2026-10-28-t1917-057-on-the-theory-of-transformatio-audit-v2`

Rubric **0.1.0** · checkpoint `a8bae15d5` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-28-t1917-057-on-the-theory-of-transformatio-audit -> 2026-10-28-t1917-057-on-the-theory-of-transformatio-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Fig. 403 human scapulae — Pass 2 (`thompson/claims/t1917-057_on-the-theory-of-transformations.py`) | 76 | 72 | 62 | 70 | 86 | 80 | 78 | 71.3 | 71.2 | 81.2 | 74.3 | **74.3** | mostly faithful |

## Findings and remediation

### Thompson (1917) Fig. 403 human scapulae — Pass 2  —  overall 74.3 (mostly faithful)

Pass 2 adds KS residual test vs identity control, 15-setting n_semi×lambda sweep, shuffled-landmark null, Commons specimen bone-fraction CV, and explicit mixed verdict (A→B supported on outline IoU; A→C not improved over identity with automatic landmarks).

- **Model fidelity to source: 76** — Pass 2 KS + sweep; still no line-by-line grid digitisation.  
  evidence: `thompson/claims/t1917-057_on-the-theory-of-transformations.py` L218–235
- **Morphological realism / specimen match: 72** — specimen_cv on Commons photo; comparison figure with overlay panels.  
  evidence: `thompson/claims/t1917-057_on-the-theory-of-transformations.py` L285–292
- **Parameter expressivity: 62** — Sweep over n_semi and lambda reported in result.json and measurements.  
  evidence: `thompson/claims/t1917-057_on-the-theory-of-transformations.py` L250–270
- **Morphometric validation: 70** — Procrustes, IoU, r, KS, shuffled null, sweep CV.  
  evidence: `datasets/thompson_1917/t1917-057/result.json` L1–80
- **Source and data traceability: 86** — Unchanged pinning from Pass 1.  
  evidence: `datasets/thompson_1917/t1917-057/t1917_057_on_the_theory_of_transformatio_reference.json` L1–55
- **Code-to-model traceability: 80** — measure_morphoeval exports KS and sweep metrics.  
  evidence: `thompson/claims/t1917-057_on-the-theory-of-transformations.py` L430–460
- **Reproducibility package: 78** — Self-test + full pipeline re-run; GIF tracked.  
  evidence: `thompson/claims/t1917-057_on-the-theory-of-transformations.py` L475–520

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Fig. 403 human scapulae — Pass 2
- **spec_fingerprint** (semantic model + inputs): `623b1de0b4ed7c847d9d960d4ae952198b27d7e7df34ef7cfb25eedd8e1212f4`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-057/ongrowthform1917thom_p769_leaf0788_fig403.jpg` (`7c57adb9fe06c75c…`), `datasets/thompson_1917/t1917-057/t1917_057_on_the_theory_of_transformatio_reference.json` (`f3c362f5577ce294…`), `morphoeval/lean/loops/Thompson1917_T1917_057.lean` (`d4758b591418639c…`)
- `docs/media/t1917_057_on_the_theory_of_transformatio_comparison.png` — artifact_sha256 `948ac31a2b981fe3…`, 148,102 B, 1440×960 px

| measurement | value | reference | method |
|---|---:|---|---|
| Outline Procrustes A→B after TPS | 0.17702785124468295 Procrustes d | 0.2622 | TPS landmarks + dense outline *(IoU=0.808 vs identity IoU 0.715; landmark d=0.0000)* |
| Outline IoU A→C after TPS | 0.6120269192890306 IoU | 0.6717 | filled outline overlap *(Indian form heavily sheared; landmark r=0.9995)* |
| Parameter sweep mean (TPS/identity) ratio | 0.8913833894824933 ratio | 1.0 | 15 settings over n_semi and lambda *(CV=0.016)* |
| Shuffled-landmark correspondence null (A→B) | 0.8932586346490007 Procrustes d | 0.177 | reverse non-center landmarks on B before TPS fit |

## Environment at checkpoint

- python: 3.12
