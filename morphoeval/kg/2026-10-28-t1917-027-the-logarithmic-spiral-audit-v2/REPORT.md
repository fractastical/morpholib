# MorphoEval report — audit `2026-10-28-t1917-027-the-logarithmic-spiral-audit-v2`

Rubric **0.1.0** · checkpoint `f1d96f32` · assessor `pkr:model/composer-2.5` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-28-t1917-027-the-logarithmic-spiral-audit -> 2026-10-28-t1917-027-the-logarithmic-spiral-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-027 — logarithmic spiral claim validation (Pass 2) (`thompson/claims/t1917-027_the-logarithmic-spiral.py`) | 86 | 68 | 72 | 76 | 85 | 78 | 84 | 81.3 | 71.0 | 82.7 | 77.6 | **77.6** | mostly faithful |

## Findings and remediation

### Thompson (1917) T1917-027 — logarithmic spiral claim validation (Pass 2)  —  overall 77.6 (mostly faithful)

Pass 2 adds synthetic k recovery sweep (max error <0.05), shuffle ablation, scipy KS on residuals, and exports mixed/supported verdict on Nautilus (log pearson r≈0.96). Conus remains mixed.

- **Model fidelity to source: 86** — Thompson log-spiral law, whorl ratio, and log-vs-Archimedean discrimination implemented with ablation and synthetic recovery checks.  
  evidence: `thompson/claims/t1917-027_the-logarithmic-spiral.py` L177–195; `thompson/claims/t1917-027_the-logarithmic-spiral.py` L198–203
- **Morphological realism / specimen match: 68** — Improved outer-boundary segmentation on Commons Nautilus; growth GIF shows gnomonic expansion. Still no digitised Thompson 1917 figure overlay.  
  evidence: `thompson/claims/t1917-027_the-logarithmic-spiral.py` L105–145
- **Parameter expressivity: 72** — Synthetic sweep panel shows monotonic k recovery across 0.08-0.42; GIF animates whorl expansion.  
  evidence: `thompson/claims/t1917-027_the-logarithmic-spiral.py` L248–258
- **Morphometric validation: 76** — Nautilus log pearson r≈0.96, KS p≈0.19, measure_morphoeval with W and ΔAICc; result.json verdict mixed/supported.  
  evidence: `datasets/thompson_1917/t1917-027/result.json` L1–25; `thompson/claims/t1917-027_the-logarithmic-spiral.py` L318–373
- **Source and data traceability: 85** — Pinned SHA-256, scout manifest, bib, reference JSON, and result.json chain.  
  evidence: `datasets/thompson_1917/t1917-027/t1917_027_the_logarithmic_spiral_reference.json` L40–55
- **Code-to-model traceability: 78** — Lean Loop_221 theorems linked in math_kg yaml; compare_models documents KS/AICc.  
  evidence: `morphoeval/lean/loops/Thompson1917_T1917_027.lean` L1–45
- **Reproducibility package: 84** — --self-test covers sweep, ablation, and deterministic measure_morphoeval; outputs regenerate from pinned data.  
  evidence: `thompson/claims/t1917-027_the-logarithmic-spiral.py` L388–390

Remediation:
- Digitise Thompson 1917 Fig. 237-303 shell outlines from IA scan for direct overlay (future pass).
- Fit full Raup (W,D,T) mesh alongside 2D log-spiral centreline to address Raup (1966) qualification.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-027 — logarithmic spiral claim validation (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `2a5f73fc968e53e30990a81294c167772f366249834cc189f759b23138dd37b2`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-027/nautilus_pompilius_specimen.jpg` (`ca3ae0f147e7cdf9…`), `datasets/thompson_1917/t1917-027/t1917_027_the_logarithmic_spiral_reference.json` (`ee8f5ec994ed23ec…`)
- `docs/media/t1917_027_the_logarithmic_spiral_comparison.png` — artifact_sha256 `0a56caf122a1996d…`, 573,601 B, 1540×1400 px

| measurement | value | reference | method |
|---|---:|---|---|
| nautilus_log_spiral_pearson_r | 0.9604055170477023 dimensionless | 0.9 | Linear regression of ln r vs θ on CV-segmented outer edge |
| nautilus_whorl_expansion_W | 9.266184976340558 ratio | 3.0 | W = exp(2πk) from log-spiral fit |
| nautilus_log_rmse_px | 35.334657903727425 px | — | RMSE of log-spiral prediction on segmented radii |
| log_vs_archimedean_aicc_delta | -12.443798422278007 AICc | 0.0 | AICc comparison on same nautilus edge samples |
| conus_log_pearson_r | 0.8495571551122201 dimensionless | 0.85 | Same pipeline on conus_textile_shell.jpg |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.x
- scipy: 1.18.1
- matplotlib: 3.x
- Pillow: 12.x

Pass 2 adds parameter sweep, shuffle ablation, brightness-mask CV segmentation, KS/AICc in compare_models.
