# MorphoEval report — audit `2026-10-28-t1917-008-the-rate-of-growth-audit`

Rubric **0.1.0** · checkpoint `f1d96f32ba3f00517c53e9cd007634306cc8d4f4` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) differential relative growth / Huxley allometry (`thompson/claims/t1917-008_the-rate-of-growth.py`) | 62 | 38 | 45 | 48 | 72 | 68 | 55 | 56.3 | 41.8 | 64.1 | 52.8 | **52.8** | partially faithful |

## Findings and remediation

### Thompson (1917) differential relative growth / Huxley allometry  —  overall 52.8 (partially faithful)

Pass 1 baseline (54.2/100). Implements log-log OLS fits to digitised Thompson Figs 18–22 CSV tables and a differential-growth animation, but omits parameter-regime sweeps, isometry ablation controls, computer-vision segmentation overlays on vendored specimen photos, and formal KS / CV goodness-of-fit reporting required for full MorphoEval biological-validation.

- **Model fidelity to source: 62** — Core Huxley law y = b x^k and exponential relative-growth trajectories are implemented with Lean 4 mirror theorems, but the script does not yet expose CLI overrides or piecewise breakpoints discussed in the 1950 Royal Society counterclaim.  
  evidence: `thompson/claims/t1917-008_the-rate-of-growth.py` L95–110; `morphoeval/lean/loops/Thompson1917_T1917_008.lean` L1–70
- **Morphological realism / specimen match: 38** — Specimen JPEGs are vendored but not drawn on the comparison figure; no visual overlay of segmented head vs mesosoma on the Pheidole photograph.  
  evidence: `thompson/claims/t1917-008_the-rate-of-growth.py` L200–240
- **Parameter expressivity: 45** — No sweep across allometric exponent k or beta regimes in Pass 1 outputs.  
  evidence: `thompson/claims/t1917-008_the-rate-of-growth.py` L1–50
- **Morphometric validation: 48** — Reports fitted k and r per series but omits RMSE/KS ablation table and CV-derived head fraction in measure_morphoeval Pass 1 hook.  
  evidence: `thompson/claims/t1917-008_the-rate-of-growth.py` L320–350
- **Source and data traceability: 72** — SHA-256-locked CSV tables and reference JSON; scout manifest documents IN_REPO verification.  
  evidence: `datasets/thompson_1917/t1917-008/t1917_008_the_rate_of_growth_reference.json` L1–60
- **Code-to-model traceability: 68** — Functions map to Huxley log-linear fit and Thompson figure CSVs; Math KG yaml links Lean theorems.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-008.yaml` L1–40
- **Reproducibility package: 55** — --self-test present but Pass 1 lacks documented CLI for ablation/sweep and environment pin in script header.  
  evidence: `thompson/claims/t1917-008_the-rate-of-growth.py` L355–375

Remediation:
- Add parameter_regime_sweep panel and isometry k=1 ablation RMSE deltas
- Run segment_ant_head_mask on pheidole_megacephala_minor_head.jpg and overlay on figure
- Report KS statistic on log residuals and expand measure_morphoeval
- Write superseding Pass 2 audit after Step 12

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) differential relative growth / Huxley allometry
- **spec_fingerprint** (semantic model + inputs): `b37a2dcc1ba621b31fed52cf52f6c5474ec5d50ddbbdb22578110acce07480e1`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-008/pheidole_caste_head_body.csv` (`ea73fc99f66c404b…`), `datasets/thompson_1917/t1917-008/t1917_008_the_rate_of_growth_reference.json` (`f126962ad37aed1b…`), `datasets/thompson_1917/t1917-008/thompson_fig18_brain_body.csv` (`89125925b63af4a4…`), `datasets/thompson_1917/t1917-008/thompson_fig19_heart_body.csv` (`499c9d07a4584a3d…`), `datasets/thompson_1917/t1917-008/thompson_fig22_plaice_length_weight.csv` (`748df1a60b251b7c…`), `morphoeval/lean/loops/Thompson1917_T1917_008.lean` (`4c7e5dc8c58cdc20…`)
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


Pass 1 baseline before Step 12 remediation (parameter sweep, CV overlay, KS statistics).
