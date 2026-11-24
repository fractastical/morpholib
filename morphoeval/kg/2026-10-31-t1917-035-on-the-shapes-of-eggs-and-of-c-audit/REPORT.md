# MorphoEval report — audit `2026-10-31-t1917-035-on-the-shapes-of-eggs-and-of-c-audit`

Rubric **0.1.0** · checkpoint `pending` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Ch. XV Fig. 328 — sea-urchin test as pressure/tension shell (Pass 1 --pass1) (`thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py`) | 72 | 48 | 38 | 41 | 78 | 70 | 65 | 60.7 | 45.4 | 70.7 | 57.5 | **57.5** | partially faithful |

## Findings and remediation

### Thompson (1917) Ch. XV Fig. 328 — sea-urchin test as pressure/tension shell (Pass 1 --pass1)  —  overall 57.5 (partially faithful)

Pass 1 baseline (--pass1): single-specimen pipeline, no panel sweep, no KS/ablation export.

- **Model fidelity to source: 72** — Thompson envelope + Preston meridian implemented; Lean Loop_229 with 5 theorems.  
  evidence: `morphoeval/lean/loops/Thompson1917_T1917_035.lean` L1–95; `thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L95–180
- **Morphological realism / specimen match: 48** — One Commons profile photo; oblique lighting; no multi-specimen panel in Pass 1.  
  evidence: `datasets/thompson_1917/t1917-035/commons_sea_urchin_test_structure_8340336761.jpg` L0–0; `datasets/thompson_1917/t1917-035/commons_sea_urchin_tests_7_77.jpg` L0–0
- **Parameter expressivity: 38** — No closure sweep or regime table exported in Pass 1.  
  evidence: `thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L430–450
- **Morphometric validation: 41** — RMSE and closure only; no KS, no literature table, weak Pearson r in Pass 1.  
  evidence: `thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L320–380
- **Source and data traceability: 78** — Scout manifest documents Dryad MISMATCH and Commons MATCH with SHA-256.  
  evidence: `datasets/thompson_1917/t1917-035/scout_manifest.json` L1–80
- **Code-to-model traceability: 70** — math_kg_loops yaml links Lean and script.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-035.yaml` L1–40
- **Reproducibility package: 65** — Self-test and SHA pins; Pass 1 mode only.  
  evidence: `thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L470–510

Remediation:
- {'id': 'R1', 'item': 'Multi-specimen panel CV, parameter-regime sweep, sphere ablation, literature KS/RMSE (Pass 2)'}
- {'id': 'R2', 'item': 'Major-axis meridian extraction and symmetric-pole Preston constraint for urchin profiles'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Ch. XV Fig. 328 — sea-urchin test as pressure/tension shell (Pass 1 --pass1)
- **spec_fingerprint** (semantic model + inputs): `ca5163818e7027ab5d5283054726515355f4bff70768bf5884955d5df1ad41e0`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-035/commons_sea_urchin_test_structure_8340336761.jpg` (`6d2cfb0d5593a906…`), `datasets/thompson_1917/t1917-035/commons_sea_urchin_tests_7_77.jpg` (`cf443a13714d2c93…`), `datasets/thompson_1917/t1917-035/echinoid_test_literature_ratios.csv` (`d6435dd20cf9e3f4…`), `datasets/thompson_1917/t1917-035/t1917_035_on_the_shapes_of_eggs_and_of_c_reference.json` (`76a472cb9dc861b4…`)
- `datasets/thompson_1917/t1917-035/commons_sea_urchin_test_structure_8340336761.jpg` — artifact_sha256 `6d2cfb0d5593a906…`, 4,742,922 B, 3756×3000 px
- `datasets/thompson_1917/t1917-035/commons_sea_urchin_tests_7_77.jpg` — artifact_sha256 `cf443a13714d2c93…`, 1,505,461 B, 2178×2086 px
- `docs/media/t1917_035_on_the_shapes_of_eggs_and_of_c_comparison.png` — artifact_sha256 `aea4edf7c5638ffd…`, 72,693 B, 1400×448 px

| measurement | value | reference | method |
|---|---:|---|---|
| primary test Preston RMSE / breadth | 0.11274093378969872 ratio | 0.05 | CV meridian vs Preston fit on Commons primary specimen |
| axisymmetry RMS / half-breadth (primary) | 0.715543247078767 ratio | 0.0 | upper vs lower meridian on horizontal scan |
| elongation L/B (primary CV fit) | 1.3671569593866257 ratio | 1.0 | Preston elongation index |
| sphere ablation RMSE ratio vs Preston (primary) | 2.352330470566004 ratio | 1.0 | c = (max(y), 0, 0, 0) null |
| Thompson closure residual (primary descriptors) | 0.0 max abs | 0.0 | 4-constraint least squares on pole/equator descriptors |
| mean curvature profile CV (primary) | 7.603211667197825 CV | 0.0 | std/mean of 2H(x); sphere => 0 |
| meridian profile Pearson r (closure vs CV) | 0.5229821499032765 r | 1.0 | 120-point interpolated profiles |
| panel specimens segmented (Commons multi-test) | 14.0 count | 3.0 | connected components after Otsu |
| panel elongation mean | 1.4484142613122497 ratio | 1.0 | std=0.2743 |
| KS D panel vs literature elongation | 1.0 D | 0.0 | p=5.160e-05 |
| closure sweep max residual | 2.0240491505063574e-09 max abs | 0.0 | 12 delta_pole steps |

## Environment at checkpoint

- python: 3.12
- numpy: 2.x
- scipy: 1.x
- matplotlib: 3.x
- scikit-image: 0.26.x
