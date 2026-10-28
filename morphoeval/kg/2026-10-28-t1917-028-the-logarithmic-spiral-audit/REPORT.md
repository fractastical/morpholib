# MorphoEval report — audit `2026-10-28-t1917-028-the-logarithmic-spiral-audit`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-028 — Nautilus log-spiral & septal wall geometry (`thompson/claims/t1917-028_the-logarithmic-spiral.py`) | 68 | 42 | 38 | 48 | 78 | 60 | 72 | 58.0 | 44.2 | 70.8 | 56.3 | **56.3** | partially faithful |

## Findings and remediation

### Thompson (1917) T1917-028 — Nautilus log-spiral & septal wall geometry  —  overall 56.3 (partially faithful)

Pass 1 baseline validates claim T1917-028 on a CC section image with radial shell sampling and log-spiral OLS; `--pass1` skips center refinement and multi-regime panel.

- **Model fidelity to source: 68** — Implements Thompson Ch. XI equiangular spiral, whorl expansion, curvature proxy, and septum involute overlays with Raup-compatible W; lacks explicit evolute locus sampling against Thompson Fig. 304 construction steps.  
  evidence: `thompson/claims/t1917-028_the-logarithmic-spiral.py` L45–95; `morphoeval/lean/loops/Thompson1917_T1917_028.lean` L1–55
- **Morphological realism / specimen match: 42** — Pass 1 `--pass1` mode disables center grid refinement; outer-envelope fit remains weak (r_log ≈ 0.38) on the section photograph despite SHA-256 verification.  
  evidence: `thompson/claims/t1917-028_the-logarithmic-spiral.py` L135–190; `docs/media/t1917_028_the_logarithmic_spiral_comparison.png` L0–0
- **Parameter expressivity: 38** — `--pass1` evaluates only the first literature regime; no Archimedean ablation bar chart or CLI `--k`/`--W` sweeps exposed.  
  evidence: `thompson/claims/t1917-028_the-logarithmic-spiral.py` L248–260
- **Morphometric validation: 48** — Reports r, RMSE, KS p-value and W_median, but Pass 1 lacks coefficient of variation on septum-wall angles and multi-regime goodness-of-fit tables.  
  evidence: `thompson/claims/t1917-028_the-logarithmic-spiral.py` L200–240
- **Source and data traceability: 78** — SHA-256 pinned specimen and reference JSON; scout manifest documents Commons source.  
  evidence: `thompson/claims/t1917-028_the-logarithmic-spiral.py` L38–43; `datasets/thompson_1917/t1917-028/scout_manifest.json` L1–40
- **Code-to-model traceability: 60** — Module docstring cites Thompson 1917 equations; functions lack line-level mapping to Lean theorem names in math_kg_loops YAML.  
  evidence: `thompson/claims/t1917-028_the-logarithmic-spiral.py` L1–22
- **Reproducibility package: 72** — `--self-test` checks synthetic spiral recovery and measure_morphoeval determinism; does not assert pass1 vs pass2 output parity or result.json verdict schema.  
  evidence: `thompson/claims/t1917-028_the-logarithmic-spiral.py` L400–430

Remediation:
- Enable full Pass 2 mode: center grid refinement, 3-regime sweep panel, Archimedean ablation bars, beta-angle CV, and result.json verdict with W_median vs literature 2.85.
- Cross-reference Lean theorems (Loop_222) in function docstrings and verify with `lean morphoeval/lean/loops/Thompson1917_T1917_028.lean` when toolchain present.
- Expand self_test to verify SHA-256, log-spiral beats Archimedean on specimen envelope, and bitwise measure_morphoeval parity.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-028 — Nautilus log-spiral & septal wall geometry
- **spec_fingerprint** (semantic model + inputs): `08a1c83cf7770b52575a4e508d980714ec0f1886be1762236a3a8e784f1ff175`
- `docs/media/t1917_028_the_logarithmic_spiral_comparison.png` — artifact_sha256 `4a85223c5be4aa90…`, 255,011 B, 1500×750 px

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


Pass 1 rendered with python3 thompson/claims/t1917-028_the-logarithmic-spiral.py --pass1
