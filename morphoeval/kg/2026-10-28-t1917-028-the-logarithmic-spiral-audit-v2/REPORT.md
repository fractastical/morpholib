# MorphoEval report — audit `2026-10-28-t1917-028-the-logarithmic-spiral-audit-v2`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-28-t1917-028-the-logarithmic-spiral-audit -> 2026-10-28-t1917-028-the-logarithmic-spiral-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-028 — Nautilus log-spiral & septal wall geometry (remediated) (`thompson/claims/t1917-028_the-logarithmic-spiral.py`) | 76 | 58 | 74 | 70 | 85 | 68 | 80 | 75.3 | 62.5 | 78.5 | 71.1 | **71.1** | mostly faithful |

## Findings and remediation

### Thompson (1917) T1917-028 — Nautilus log-spiral & septal wall geometry (remediated)  —  overall 71.1 (mostly faithful)

Pass 2 adds center grid search, regime/ablation panel, morphometric CV exports, and datasets/thompson_1917/t1917-028/result.json with mixed verdict (log spiral beats Archimedean on envelope; empirical W from section remains below literature W≈2.85).

- **Model fidelity to source: 76** — Full Thompson Ch. XI equation set implemented with septum involute overlay and whorl expansion; Lean Loop_222 linked via math_kg_loops YAML.  
  evidence: `thompson/claims/t1917-028_the-logarithmic-spiral.py` L45–95; `morphoeval/math_kg_loops/thompson1917-t1917-028.yaml` L1–35
- **Morphological realism / specimen match: 58** — Center-refined radial envelope (120 rays) with log-spiral overlay; r_log≈0.38 and mixed verdict — visual alignment improved vs Pass 1 but not publication-grade.  
  evidence: `thompson/claims/t1917-028_the-logarithmic-spiral.py` L135–190; `docs/media/t1917_028_the_logarithmic_spiral_comparison.png` L0–0
- **Parameter expressivity: 74** — Three literature regimes plus Archimedean negative control bar chart in full mode.  
  evidence: `thompson/claims/t1917-028_the-logarithmic-spiral.py` L320–350
- **Morphometric validation: 70** — measure_morphoeval exports r, RMSE, KS p, W_median, beta CV, Archimedean ablation r; result.json records headline_numbers for T1917-028 claim registry.  
  evidence: `thompson/claims/t1917-028_the-logarithmic-spiral.py` L380–410; `datasets/thompson_1917/t1917-028/result.json` L1–30
- **Source and data traceability: 85** — SHA-256 verified specimen, reference JSON, scout manifest, paper_citations.bib.  
  evidence: `datasets/thompson_1917/t1917-028/t1917_028_the_logarithmic_spiral_reference.json` L1–50
- **Code-to-model traceability: 68** — Docstring equations match reference JSON; math_kg YAML cites Lean loci; inline Lean name comments still sparse in Python functions.  
  evidence: `morphoeval/lean/loops/Thompson1917_T1917_028.lean` L8–52
- **Reproducibility package: 80** — self_test covers synthetic spiral, determinism; full render writes PNG/GIF/result.json.  
  evidence: `thompson/claims/t1917-028_the-logarithmic-spiral.py` L400–455

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-028 — Nautilus log-spiral & septal wall geometry (remediated)
- **spec_fingerprint** (semantic model + inputs): `08a1c83cf7770b52575a4e508d980714ec0f1886be1762236a3a8e784f1ff175`
- `docs/media/t1917_028_the_logarithmic_spiral_comparison.png` — artifact_sha256 `0d7f40e635aac342…`, 227,393 B, 2100×750 px

| measurement | value | reference | method |
|---|---:|---|---|
| log_spiral_fit_r | 0.3841 1 | — | OLS ln(r) vs θ |
| log_spiral_rmse | 105.91 px | — | radial residual RMSE |
| whorl_expansion_W | 1.488 1 | 2.85 | exp(2πk) from fit |
| equiangular_psi_deg | 86.38 deg | — | atan(1/k) |
| septum_wall_beta_deg | 3.62 deg | — | 90 - psi |
| ks_pvalue_log_residuals | 1.0 1 | — | KS on normalized residuals |
| archimedean_ablation_r | 0.30005706495543205 1 | — | negative control |
| verdict_score | 0.5 1 | — | claim verdict mapping |
| beta_angle_cv | 0.0 1 | — | septum-wall angle coefficient of variation |
| W_median_empirical | 1.072 1 | 2.85 | median successive radius ratio |

## Environment at checkpoint


Pass 2: python3 thompson/claims/t1917-028_the-logarithmic-spiral.py (full mode)
