# MorphoEval report — audit `2026-10-28-t1917-057-on-the-theory-of-transformatio-audit`

Rubric **0.1.0** · checkpoint `a8bae15d5` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Fig. 403: Dwight human scapulae coordinate transformation (`thompson/claims/t1917-057_on-the-theory-of-transformations.py`) | 72 | 68 | 48 | 58 | 86 | 78 | 74 | 64.0 | 64.2 | 79.1 | 68.6 | **68.6** | mostly faithful |

## Findings and remediation

### Thompson (1917) Fig. 403: Dwight human scapulae coordinate transformation  —  overall 68.6 (mostly faithful)

Pass 1. Three Fig. 403 silhouettes segmented; TPS from A (Caucasian) to B and C. A→B outline Procrustes 0.177 vs identity 0.262, IoU 0.808 vs 0.715. A→C outline TPS slightly worse than identity (0.300 vs 0.275; IoU 0.612 vs 0.672). Verdict mixed.

- **Model fidelity to source: 72** — E2/E4 TPS implemented; Lean Loop_246 mirrors interpolation algebra. Grid lines not digitised.  
  evidence: `thompson/claims/t1917-057_on-the-theory-of-transformations.py` L115–145; `morphoeval/lean/loops/Thompson1917_T1917_057.lean` L1–90
- **Morphological realism / specimen match: 68** — Compared to Thompson's own Fig. 403; Commons scapula photo vendored but minimal CV.  
  evidence: `datasets/thompson_1917/t1917-057/ongrowthform1917thom_p769_leaf0788_fig403.jpg` L0–0
- **Parameter expressivity: 48** — Growth GIF for A→B; sweep code present but limited reporting in Pass 1.  
  evidence: `thompson/claims/t1917-057_on-the-theory-of-transformations.py` L250–270
- **Morphometric validation: 58** — Procrustes, IoU, Pearson r; KS and full sweep added in Pass 2.  
  evidence: `thompson/claims/t1917-057_on-the-theory-of-transformations.py` L220–248
- **Source and data traceability: 86** — SHA-256 pinned inputs; Dryad mismatch recorded; IA + Commons provenance.  
  evidence: `datasets/thompson_1917/t1917-057/scout_manifest.json` L1–60
- **Code-to-model traceability: 78** — Module docstring maps equations; KG yaml links Lean.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-057.yaml` L1–25
- **Reproducibility package: 74** — Self-test 8/8; deterministic; requirements lower bounds only.  
  evidence: `thompson/claims/t1917-057_on-the-theory-of-transformations.py` L475–495

Remediation:
- {'id': 'R1', 'item': 'Add KS test and report parameter sweep CV in measure_morphoeval'}
- {'id': 'R2', 'item': 'Expand specimen CV on Commons scapula; document A→C outline failure mode'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Fig. 403: Dwight human scapulae coordinate transformation
- **spec_fingerprint** (semantic model + inputs): `623b1de0b4ed7c847d9d960d4ae952198b27d7e7df34ef7cfb25eedd8e1212f4`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-057/ongrowthform1917thom_p769_leaf0788_fig403.jpg` (`7c57adb9fe06c75c…`), `datasets/thompson_1917/t1917-057/t1917_057_on_the_theory_of_transformatio_reference.json` (`f3c362f5577ce294…`), `morphoeval/lean/loops/Thompson1917_T1917_057.lean` (`d4758b591418639c…`)
- `datasets/thompson_1917/t1917-057/ongrowthform1917thom_p769_leaf0788_fig403.jpg` — artifact_sha256 `7c57adb9fe06c75c…`, 624,951 B, 1904×3128 px
- `docs/media/t1917_057_on_the_theory_of_transformatio_comparison.png` — artifact_sha256 `948ac31a2b981fe3…`, 148,102 B, 1440×960 px

| measurement | value | reference | method |
|---|---:|---|---|
| Outline Procrustes A→B after TPS | 0.17702785124468295 Procrustes d | 0.2622 | TPS landmarks + dense outline *(IoU=0.808 vs identity IoU 0.715; landmark d=0.0000)* |
| Outline IoU A→C after TPS | 0.6120269192890306 IoU | 0.6717 | filled outline overlap *(Indian form heavily sheared; landmark r=0.9995)* |
| Parameter sweep mean (TPS/identity) ratio | 0.8913833894824933 ratio | 1.0 | 15 settings over n_semi and lambda *(CV=0.016)* |
| Shuffled-landmark correspondence null (A→B) | 0.8932586346490007 Procrustes d | 0.177 | reverse non-center landmarks on B before TPS fit |

## Environment at checkpoint

- python: 3.12
