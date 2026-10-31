# MorphoEval report — audit `2026-10-31-t1917-035-on-the-shapes-of-eggs-and-of-c-audit-v2`

Rubric **0.1.0** · checkpoint `pending` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-31-t1917-035-on-the-shapes-of-eggs-and-of-c-audit -> 2026-10-31-t1917-035-on-the-shapes-of-eggs-and-of-c-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) Ch. XV Fig. 328 — sea-urchin test meridian under envelope pressure/tension (Pass 2) (`thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py`) | 84 | 62 | 76 | 68 | 86 | 82 | 80 | 81.3 | 64.2 | 82.5 | 74.9 | **74.9** | mostly faithful |

## Findings and remediation

### Thompson (1917) Ch. XV Fig. 328 — sea-urchin test meridian under envelope pressure/tension (Pass 2)  —  overall 74.9 (mostly faithful)

Pass 2 adds multi-specimen segmentation, closure parameter sweep, sphere ablation (2.35× worse RMSE), literature ratio table, KS statistic, and documents Dryad mismatch in result.json (verdict mixed on broad claim; urchin instance partially supported).

- **Model fidelity to source: 84** — Thompson envelope exercised via closure, mean curvature profile, and sweep; Lean laws pinned.  
  evidence: `thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L130–200; `morphoeval/lean/loops/Thompson1917_T1917_035.lean` L50–75
- **Morphological realism / specimen match: 62** — Real Commons tests segmented; profile r ~ 0.52; oblique photo limits overlay fidelity.  
  evidence: `datasets/thompson_1917/t1917-035/result.json` L1–60; `thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L188–280
- **Parameter expressivity: 76** — closure_sweep over delta_pole; GIF; literature CSV regimes.  
  evidence: `thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L290–310
- **Morphometric validation: 68** — RMSE/breadth, Pearson r, KS D, sphere ablation ratio exported in measure_morphoeval.  
  evidence: `thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L330–410
- **Source and data traceability: 86** — Scout manifest + reference JSON SHA pins; Dryad mismatch explicit.  
  evidence: `datasets/thompson_1917/t1917-035/scout_manifest.json` L20–45
- **Code-to-model traceability: 82** — math_kg_loops yaml + Lean loci.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-035.yaml` L1–35
- **Reproducibility package: 80** — Self-test, result.json, comparison PNG + growth GIF.  
  evidence: `thompson/claims/t1917-035_on-the-shapes-of-eggs-and-of-certain-other-hollo.py` L470–520

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) Ch. XV Fig. 328 — sea-urchin test meridian under envelope pressure/tension (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `ca5163818e7027ab5d5283054726515355f4bff70768bf5884955d5df1ad41e0`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-035/commons_sea_urchin_test_structure_8340336761.jpg` (`6d2cfb0d5593a906…`), `datasets/thompson_1917/t1917-035/commons_sea_urchin_tests_7_77.jpg` (`cf443a13714d2c93…`), `datasets/thompson_1917/t1917-035/echinoid_test_literature_ratios.csv` (`d6435dd20cf9e3f4…`), `datasets/thompson_1917/t1917-035/t1917_035_on_the_shapes_of_eggs_and_of_c_reference.json` (`76a472cb9dc861b4…`)
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
- Pillow: 10+
