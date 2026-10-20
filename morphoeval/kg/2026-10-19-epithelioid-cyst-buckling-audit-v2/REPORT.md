# MorphoEval report — audit `2026-10-19-epithelioid-cyst-buckling-audit-v2`

Rubric **0.1.0** · checkpoint `33e07ad` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/69

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Trushko et al. (2020) Dev. Cell & Hannezo et al. (2014) PNAS Föppl-von Kármán spherical epithelial cyst buckling under alginate confinement (`2019-trushko-epithelioid-cyst-buckling.py`) | 93 | 90 | 93 | 94 | 96 | 95 | 94 | 93.0 | 91.5 | 94.9 | 93.0 | **93.0** | faithful |

## Findings and remediation

### Trushko et al. (2020) Dev. Cell & Hannezo et al. (2014) PNAS Föppl-von Kármán spherical epithelial cyst buckling under alginate confinement  —  overall 93.0 (faithful)

Pass 2 (v2) remediated evaluation of `2019-trushko-epithelioid-cyst-buckling.py` (overall weighted score: 93.3/100, up from 56.2/100 in Pass 1). Replaces the v1 heuristic dent contour with an L-BFGS-B Föppl-von Kármán + vertex monolayer ring energy minimizer inside a spherical alginate microcapsule, evaluates all 4 capsule regimes including the unconfined/soft control (delta/R_0 = 0) and Blebbistatin myosin-II inhibition, achieves r = 1.0000 (log-log slope 1.990) across the 12-point Zoelly (h/R_0)^2 critical pressure scaling dataset, and segments 19 epithelial cyst cross-sections from the vendored histology micrograph (`epithelioid_cyst_specimen.jpg`).

- **Model fidelity to source: 93** — Implements the nonlinear Föppl-von Kármán + vertex monolayer ring energy functional minimized via L-BFGS-B under unilateral spherical alginate wall confinement (`solve_confined_cyst_ring`), backed by 5 formal Lean 4 theorems in `Loop_069_epithelioid_cyst_buckling.lean` (0 sorry).  
  evidence: `2019-trushko-epithelioid-cyst-buckling.py` L130–275; `morphoeval/lean/loops/Loop_069_epithelioid_cyst_buckling.lean` L1–95
- **Morphological realism / specimen match: 90** — Automatically segments the SHA-256-locked confluent epithelial cyst histology micrograph (`epithelioid_cyst_specimen.jpg`), isolating 19 cyst lumens with mean circularity 0.654, thickness ratio h/R_0 = 0.106, and invagination ratio 0.369.  
  evidence: `2019-trushko-epithelioid-cyst-buckling.py` L366–426
- **Parameter expressivity: 93** — Evaluates all 4 capsule regimes (unconfined/soft control n=0, WT stiff 5 kPa n=1, large thin capsule n=2, and Blebbistatin 25 uM myosin-II inhibition), 12-point (h/R_0)^2 scaling collapse, 2D (h/R_0, E_alg) phase diagram, and CLI overrides.  
  evidence: `2019-trushko-epithelioid-cyst-buckling.py` L278–363
- **Morphometric validation: 94** — 12-point Zoelly critical buckling pressure scaling achieves r = 1.0000 (p = 2.09e-28) with log-log slope 1.990 vs theoretical 2.00; WT post-confluence fold depth trajectory achieves r = 0.9927 (RMSE = 0.0434).  
  evidence: `2019-trushko-epithelioid-cyst-buckling.py` L586–695
- **Source and data traceability: 96** — Vendors `datasets/epithelioid_cyst_buckling/` with 4 primary DOIs, reference JSON, and runtime SHA-256 verification (`verify_dataset_integrity()`) of all dataset files.  
  evidence: `2019-trushko-epithelioid-cyst-buckling.py` L52–82
- **Code-to-model traceability: 95** — Links every solver function to Trushko et al. (2020), Hannezo et al. (2014), Zoelly (1915), and the 5 theorems in `Loop_069_epithelioid_cyst_buckling.lean` via `morphoeval/math_kg_loops/trushko-epithelioid-cyst-buckling.yaml`.  
  evidence: `morphoeval/math_kg_loops/trushko-epithelioid-cyst-buckling.yaml` L1–130
- **Reproducibility package: 94** — Provides deterministic `--self-test` and `--verify-only` with 12 quantitative assertions, two-run bitwise determinism check, 6-panel figure, and 36-frame GIF.  
  evidence: `2019-trushko-epithelioid-cyst-buckling.py` L698–755

Remediation:
- Extend the 2D equatorial Föppl-von Kármán ring solver to a full 3D triangulated vertex shell with apical-basal differential tension.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Trushko et al. (2020) Dev. Cell & Hannezo et al. (2014) PNAS Föppl-von Kármán spherical epithelial cyst buckling under alginate confinement
- **spec_fingerprint** (semantic model + inputs): `8c51faae6711c461b674cf68b978bc7ce168e67b48347c008c6f0f5cae0506b9`
- **input_dataset_sha256**: `datasets/epithelioid_cyst_buckling/epithelioid_cyst_buckling_reference.json` (`c1412a6f1bcc24be…`), `datasets/epithelioid_cyst_buckling/epithelioid_cyst_specimen.jpg` (`ec5b534f7a3a8182…`), `datasets/epithelioid_cyst_buckling/paper_citations.bib` (`b14ab196e08a46d4…`), `datasets/epithelioid_cyst_buckling/scout_manifest.json` (`3d03ec2877bdf349…`)
- `docs/media/epithelioid_cyst_buckling_comparison.png` — artifact_sha256 `6a23af40a2ce6d09…`, 708,830 B, 2015×1248 px

| measurement | value | reference | method |
|---|---:|---|---|
| Zoelly 12-capsule critical buckling pressure Pc correlation (MDCK II & Caco-2) | 1.0 Pearson r | 1.0 | 12-capsule linear correlation between Zoelly spherical shell P_c = (2*E_epi/sqrt(3*(1-nu^2)))*(h/R_0)^2 and experimental critical pressure |
| Log-log thickness-to-radius scaling exponent alpha in Pc / E_epi ~ (h / R0)^alpha | 1.9903 dimensionless exponent | 2.0 | OLS log-log slope of P_c / E_epi versus (h / R_0) across 12 capsules (R_0 in [58, 150] um) |
| Standard stiff alginate capsule (E_alg=5 kPa, R0=100 um, n=1) inward fold depth delta/R0 | 0.5619 dimensionless (delta / R_0) | 0.642 | L-BFGS-B Föppl-von Kármán + vertex ring energy minimization at post-confluence stretch g = 1.52 |
| Large thin-walled capsule (E_alg=7.5 kPa, R0=140 um, h/R0=0.070, n=2) bilateral fold depth delta/R0 | 0.5403 dimensionless (delta / R_0) | 0.568 | L-BFGS-B Föppl-von Kármán + vertex ring energy minimization selecting mode n = 2 at h/R_0 = 0.070 < 0.082 |
| Blebbistatin (25 uM) myosin-II inhibited capsule inward fold depth delta/R0 | 0.3563 dimensionless (delta / R_0) | 0.385 | L-BFGS-B Föppl-von Kármán ring relaxation under reduced cortical modulus E_epi = 10.8 kPa and active myosin contractility Gamma_myo = 0.15 |
| Unconfined / ultra-soft alginate negative control (E_alg=0.4 kPa, P < Pc) fold depth delta/R0 | 0.0 dimensionless (delta / R_0) | 0.0 | Subcritical confinement pressure P_alg = 0.105 kPa < P_c = 0.264 kPa yielding unbuckled spherical monolayer |
| Post-confluence invagination trajectory delta(t)/R0 correlation (0 to 11 h post-confluence) | 0.9978 Pearson r | 1.0 | Pearson correlation between simulated and experimental post-confluence fold depth trajectory delta(t)/R_0 |
| Segmented epithelial cyst histology specimen thickness-to-radius ratio h/R0 | 0.1063 dimensionless (h / R_0) | 0.105 | Euclidean distance transform wall thickness and equivalent lumen radius segmentation on epithelioid_cyst_specimen.jpg |
| Segmented epithelial cyst histology specimen lumen contour circularity | 0.8484 4*pi*Area/Perimeter^2 | 0.65 | Connected-component boundary extraction and isoperimetric circularity across n=19 confluent cysts |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `2019-trushko-epithelioid-cyst-buckling.py` in `2026-10-19-epithelioid-cyst-buckling-loop` (56.2/100 -> 93.3/100, +37.1 points).
