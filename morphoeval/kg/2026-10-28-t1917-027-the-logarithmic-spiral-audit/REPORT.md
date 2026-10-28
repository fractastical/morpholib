# MorphoEval report — audit `2026-10-28-t1917-027-the-logarithmic-spiral-audit`

Rubric **0.1.0** · checkpoint `f1d96f32` · assessor `pkr:model/composer-2.5` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-027 — logarithmic spiral claim validation (`thompson/claims/t1917-027_the-logarithmic-spiral.py`) | 78 | 62 | 48 | 55 | 82 | 74 | 80 | 68.0 | 59.4 | 79.1 | 67.9 | **67.9** | mostly faithful |

## Findings and remediation

### Thompson (1917) T1917-027 — logarithmic spiral claim validation  —  overall 67.9 (mostly faithful)

Pass 1 baseline implements Thompson's log-spiral law and fits a CC Nautilus photograph, but lacks parameter-regime sweep, shuffle ablation, KS/residual reporting in the figure, and Thompson figure overlay.

- **Model fidelity to source: 78** — Core equations r=r₀exp(kθ) and W=exp(2πk) match Thompson (1917) and are implemented directly. Log vs Archimedean discrimination uses ln r linearity. Missing Raup generating-curve geometry.  
  evidence: `thompson/claims/t1917-027_the-logarithmic-spiral.py` L48–55; `thompson/claims/t1917-027_the-logarithmic-spiral.py` L68–76
- **Morphological realism / specimen match: 62** — Comparison panel overlays segmented edge on the Commons Nautilus photo and growth GIF animates gnomonic whorl expansion, but no side-by-side with Thompson 1917 Fig. 237-303 digitisation.  
  evidence: `thompson/claims/t1917-027_the-logarithmic-spiral.py` L218–235; `docs/media/t1917_027_the_logarithmic_spiral_comparison.png` L0–0
- **Parameter expressivity: 48** — Pass 1 does not sweep k or W across the documented regimes in reference JSON; GIF shows one fitted k only.  
  evidence: `datasets/thompson_1917/t1917-027/t1917_027_the_logarithmic_spiral_reference.json` L28–38
- **Morphometric validation: 55** — measure_morphoeval returns pearson r, W, and RMSE with literature anchors, but Pass 1 omits AICc/KS in exported measurements and Conus discrimination is weak.  
  evidence: `thompson/claims/t1917-027_the-logarithmic-spiral.py` L318–373; `thompson/claims/t1917-027_the-logarithmic-spiral.py` L134–175
- **Source and data traceability: 82** — SHA-256 pinned specimens, scout manifest, reference JSON and paper_citations.bib; Thompson IA scan noted.  
  evidence: `thompson/claims/t1917-027_the-logarithmic-spiral.py` L38–46; `datasets/thompson_1917/t1917-027/scout_manifest.json` L1–40
- **Code-to-model traceability: 74** — Functions map to Thompson equations in docstring; Lean module linked via math_kg_loops yaml.  
  evidence: `morphoeval/math_kg_loops/thompson1917-t1917-027.yaml` L1–35
- **Reproducibility package: 80** — --self-test passes; deterministic RNG seed; outputs written to docs/media. scipy not in cloud base image by default.  
  evidence: `thompson/claims/t1917-027_the-logarithmic-spiral.py` L376–395

Remediation:
- Add parameter-regime sweep over k and W with recovery error bounds (Pass 2).
- Add shuffle ablation control on segmented radii to prove fit is not spurious.
- Export AICc, KS p-value, and log-vs-Archimedean ΔAICc to measure_morphoeval and comparison figure.
- Improve conus segmentation or document mixed verdict when gastropod aspect ratio breaks planar log-spiral assumption.
- Add telemetry-friendly metrics CSV and Thompson 1917 figure citation overlay panel.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-027 — logarithmic spiral claim validation
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
