# MorphoEval report — audit `2026-10-31-t1917-025-spicules-geodetics-audit`

Rubric **0.1.0** · checkpoint `35d30f4cf` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-025 — radiolarian geodesic spherical lattices (`thompson/claims/t1917-025_spicules-geodetics.py`) | 62 | 48 | 40 | 38 | 78 | 65 | 58 | 54.7 | 44.2 | 66.5 | 54.1 | **54.1** | partially faithful |

## Findings and remediation

### Thompson (1917) T1917-025 — radiolarian geodesic spherical lattices  —  overall 54.1 (partially faithful)

Pass 1 baseline: geodesic reference mesh and plate comparison figure, single subdivision frequency, no ablation controls, no result.json, incomplete morphometric GOF block.

- **Model fidelity to source: 62** — Implements arccos geodesic lengths and icosahedral baseline per reference JSON.  
  evidence: `thompson/claims/t1917-025_spicules-geodetics.py` L150–178; `morphoeval/lean/loops/Thompson1917_T1917_025.lean` L1–76
- **Morphological realism / specimen match: 48** — Verified Commons specimens rendered; limited segmentation in Pass 1.  
  evidence: `thompson/claims/t1917-025_spicules-geodetics.py` L48–62; `docs/media/t1917_025_spicules_geodetics_comparison.png` L0–0
- **Parameter expressivity: 40** — Single freq in default Pass 1 run.  
  evidence: `thompson/claims/t1917-025_spicules-geodetics.py` L268–270
- **Morphometric validation: 38** — CV only; no ablation or histogram GOF in Pass 1.  
  evidence: `thompson/claims/t1917-025_spicules-geodetics.py` L264–300
- **Source and data traceability: 78** — scout_manifest + reference JSON + Commons URLs.  
  evidence: `datasets/thompson_1917/t1917-025/scout_manifest.json` L1–80
- **Code-to-model traceability: 65** — Docstring and math_kg yaml link equations to Lean.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-025.yaml` L1–40
- **Reproducibility package: 58** — Self-test ok; Pass 1 skips result.json.  
  evidence: `thompson/claims/t1917-025_spicules-geodetics.py` L398–410

Remediation:
- Full freq sweep and specimen-matched model selection.
- Ablation controls and GOF statistics; write result.json.

## Environment at checkpoint

