# MorphoEval report — audit `2026-10-28-t1917-008-the-rate-of-growth-audit-v2`

Rubric **0.1.0** · checkpoint `f1d96f32ba3f00517c53e9cd007634306cc8d4f4` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-28-t1917-008-the-rate-of-growth-audit -> 2026-10-28-t1917-008-the-rate-of-growth-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) differential relative growth / Huxley allometry (`thompson/claims/t1917-008_the-rate-of-growth.py`) | 82 | 62 | 80 | 76 | 88 | 84 | 78 | 81.3 | 67.2 | 82.9 | 76.2 | **76.2** | mostly faithful |

## Findings and remediation

### Thompson (1917) differential relative growth / Huxley allometry  —  overall 76.2 (mostly faithful)

Pass 2 remediated assessment (78.4/100). Log-log fits to Thompson Figs 18–22, parameter sweep over k, isometry ablation RMSE deltas, KS tests on log residuals, Pheidole head CV segmentation, SHA-256 dataset locks, Lean 4 theorems, and result.json verdict (supported for differential allometry; Pheidole labelled analogue).

- **Model fidelity to source: 82** — Implements Huxley-Thompson relative growth ODE, log-log fits, ablation against k=1, and links three Lean laws.  
  evidence: `thompson/claims/t1917-008_the-rate-of-growth.py` L95–110; `thompson/claims/t1917-008_the-rate-of-growth.py` L175–195; `morphoeval/lean/loops/Thompson1917_T1917_008.lean` L1–75
- **Morphological realism / specimen match: 62** — CV head-fraction computed on CC Pheidole minor worker photo; comparison figure annotates CV metrics though full photo overlay remains schematic (not pixel-aligned to mesosoma CSV).  
  evidence: `thompson/claims/t1917-008_the-rate-of-growth.py` L112–145; `datasets/thompson_1917/t1917-008/pheidole_megacephala_minor_head.jpg` L0–0
- **Parameter expressivity: 80** — parameter_regime_sweep documents monotonic change in organ/body ratio vs k across 0.5–3.0 with k=1 marker.  
  evidence: `thompson/claims/t1917-008_the-rate-of-growth.py` L147–165; `docs/media/t1917_008_the_rate_of_growth_comparison.png` L0–0
- **Morphometric validation: 76** — Fitted exponents match expected regimes (brain k~0.72, plaice k~3); measure_morphoeval exports k, RMSE, head fraction.  
  evidence: `thompson/claims/t1917-008_the-rate-of-growth.py` L320–355; `datasets/thompson_1917/t1917-008/result.json` L1–30
- **Source and data traceability: 88** — IA scan pin, OpenAlex scout entries, Commons specimens, and locked digests in reference JSON.  
  evidence: `datasets/thompson_1917/t1917-008/scout_manifest.json` L1–50; `thompson/claims/t1917-008_the-rate-of-growth.py` L55–65
- **Code-to-model traceability: 84** — Each CSV maps to a Thompson figure; math_kg_loops yaml lists lean_decl/code_locus pairs.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-008.yaml` L1–35
- **Reproducibility package: 78** — Deterministic measure_morphoeval, --self-test with exponent guards, pinned dataset hashes.  
  evidence: `thompson/claims/t1917-008_the-rate-of-growth.py` L357–375

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) differential relative growth / Huxley allometry
- **spec_fingerprint** (semantic model + inputs): `b37a2dcc1ba621b31fed52cf52f6c5474ec5d50ddbbdb22578110acce07480e1`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-008/pheidole_caste_head_body.csv` (`ea73fc99f66c404b…`), `datasets/thompson_1917/t1917-008/t1917_008_the_rate_of_growth_reference.json` (`f126962ad37aed1b…`), `datasets/thompson_1917/t1917-008/thompson_fig18_brain_body.csv` (`89125925b63af4a4…`), `datasets/thompson_1917/t1917-008/thompson_fig19_heart_body.csv` (`499c9d07a4584a3d…`), `datasets/thompson_1917/t1917-008/thompson_fig22_plaice_length_weight.csv` (`748df1a60b251b7c…`), `morphoeval/lean/loops/Thompson1917_T1917_008.lean` (`4c7e5dc8c58cdc20…`)
- `datasets/thompson_1917/t1917-008/pheidole_megacephala_minor_head.jpg` — artifact_sha256 `5a30ee266fa0c029…`, 12,390,648 B, 3717×3526 px
- `docs/media/t1917_008_the_rate_of_growth_comparison.png` — artifact_sha256 `c0711f11b3b18421…`, 210,314 B, 1960×1120 px

| measurement | value | reference | method |
|---|---:|---|---|
| brain_vs_body_fig18_allometric_k | 0.7441425339866521 dimensionless | — | OLS log10-log10 |
| brain_vs_body_fig18_log_rmse | 0.1369015875431633 log10 | — | log10 residual RMSE |
| heart_vs_body_fig19_allometric_k | 0.9368047402204511 dimensionless | — | OLS log10-log10 |
| heart_vs_body_fig19_log_rmse | 0.03225442128095779 log10 | — | log10 residual RMSE |
| plaice_weight_length_fig22_allometric_k | 2.729441419967255 dimensionless | — | OLS log10-log10 |
| plaice_weight_length_fig22_log_rmse | 0.01756686603083516 log10 | — | log10 residual RMSE |
| pheidole_head_mesosoma_analogue_allometric_k | 3.7970665739026983 dimensionless | — | OLS log10-log10 |
| pheidole_head_mesosoma_analogue_log_rmse | 0.042656916126197986 log10 | — | log10 residual RMSE |
| pheidole_specimen_head_fraction | 0.46657715717866094 fraction | — | threshold segmentation minor head JPG |

## Environment at checkpoint


Pass 2 after Step 12 remediation (sweep, ablation, CV, KS).
