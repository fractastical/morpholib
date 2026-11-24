# MorphoEval report — audit `2026-10-31-t1917-025-spicules-geodetics-audit-v2`

Rubric **0.1.0** · checkpoint `35d30f4cf` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-31-t1917-025-spicules-geodetics-audit -> 2026-10-31-t1917-025-spicules-geodetics-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-025 — radiolarian geodesic spherical lattices (`thompson/claims/t1917-025_spicules-geodetics.py`) | 74 | 68 | 72 | 70 | 82 | 71 | 76 | 73.3 | 68.8 | 76.7 | 72.5 | **72.5** | mostly faithful |

## Findings and remediation

### Thompson (1917) T1917-025 — radiolarian geodesic spherical lattices  —  overall 72.5 (mostly faithful)

Pass 2: freq sweep 0-3, skeleton edge extraction on Challenger crop, histogram Pearson r≈0.64, ablation random-chord CV 1.11, mixed verdict (specimen edge CV 0.375 vs model 0.179).

- **Model fidelity to source: 74** — Great-circle metric, icosa + Fibonacci kNN refs, Lean-backed chord identities.  
  evidence: `thompson/claims/t1917-025_spicules-geodetics.py` L150–178; `morphoeval/lean/loops/Thompson1917_T1917_025.lean` L24–76
- **Morphological realism / specimen match: 68** — Skeleton morphometrics on pinned Challenger crop; mixed verdict documented.  
  evidence: `thompson/claims/t1917-025_spicules-geodetics.py` L210–240; `datasets/thompson_1917/t1917-025/result.json` L1–21
- **Parameter expressivity: 72** — --pass2 enables four-level freq sweep and control panel.  
  evidence: `thompson/claims/t1917-025_spicules-geodetics.py` L268–295
- **Morphometric validation: 70** — r, RMSE, KS, CV exported in measure_morphoeval and result.json.  
  evidence: `thompson/claims/t1917-025_spicules-geodetics.py` L242–262
- **Source and data traceability: 82** — SHA-256 locked Commons plates and scout vetting of audit-named missing file.  
  evidence: `datasets/thompson_1917/t1917-025/t1917_025_spicules_geodetics_reference.json` L55–75
- **Code-to-model traceability: 71** — math_kg_loops yaml maps four equation ids to Lean theorems.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-025.yaml` L14–40
- **Reproducibility package: 76** — Self-test, pass2 CLI, GIF, result.json artefact.  
  evidence: `thompson/claims/t1917-025_spicules-geodetics.py` L398–440

## Environment at checkpoint

