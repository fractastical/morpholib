# MorphoEval report — audit `2026-10-28-t1917-032-the-shapes-of-horns-and-of-tee-audit-v2`

Rubric **0.1.0** · checkpoint `3ae9ce99c` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/pull/309 · supersession chain `2026-10-28-t1917-032-the-shapes-of-horns-and-of-tee-audit -> 2026-10-28-t1917-032-the-shapes-of-horns-and-of-tee-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-032 — teeth & tusks: differential-accretion log spiral + St Venant torsion (`thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py`) | 90 | 78 | 88 | 86 | 90 | 90 | 88 | 89.3 | 81.0 | 89.2 | 86.0 | **86.0** | faithful |

## Findings and remediation

### Thompson (1917) T1917-032 — teeth & tusks: differential-accretion log spiral + St Venant torsion  —  overall 86.0 (faithful)

Pass 2 (v2) evaluation of `t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` (overall ~86.0/100, up from 60.7 in v1). The remediated script fits the full 3-D screw-accretion model to the segmented mammoth tusk (constant-width twisted helix RMSE 17.4 px vs plane 33.1 px, ΔAIC 477; measured-width law 23.1 vs 37.3 px, ΔAIC 355), sweeps 30 (λ, c) + 5 τ + 16 ψ regimes confirming every Lean-proved monotonicity and the E5 closure (r = 1.000, RMSE 0.2°), adds Archimedean / shuffled-width / τ = 0 / circular-section controls, block-bootstrap CIs and a one-sample KS test, measures the narwhal groove angle (φ = 29.5°, 95% CI 26.3-33.0° → α ≡ 70.2° vs 66.9 ± 3.3° published, z = +1.00) and reproduces the six-sector St Venant torsion curves of Fig. 321. Verdict MIXED: torsion / double curvature supported, gnomonic accretion log spiral not supported on this projection.

- **Model fidelity to source: 90** — E1-E3 (`log_spiral_r`, `whorl_expansion`, `gnomon_radius`, L146-L165), E4-E6 screw accretion with an optional measured-width profile (`grow_tooth`, `implied_psi_deg`, L190-L235), E8-E9 St Venant warping (L238-L277), E7 helix (L281-L296) are implemented as published; the twisted model Thompson invokes for long tusks is now fitted to the specimen in a nested 2 × 2 design (`fit_screw_model`, `compare_screw_models`, L542-L611) and the E5 closure is verified against the integrated accretion ODE (`integrate_accretion_spiral`, S4: r = 1.000). 17 Lean theorems mirror the four laws (0 sorry).  
  evidence: `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L146–235; `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L508–611; `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L238–296; `morphoeval/lean/loops/Thompson1917_T1917_032.lean` L31–215
- **Morphological realism / specimen match: 78** — HSV segmentation + skeleton longest path (`segment_mammoth_tusk`, `tusk_centreline_profile`, L300-L386) recovers the SHA-256-locked mammoth centreline and width; the fitted twisted screw model tracks the whole tusk including the reversed tip hook at RMSE 17.4 px (~1.1 % of the 1.5 kpx arc; overlay in panel (a), growth GIF panel 3). The narwhal groove orientation field is extracted by structure tensor in the low-foreshortening strip (L702-L745) and the computed St Venant contours are overlaid on Thompson's own Fig. 321 crop. Deductions: the accretion width-law fit (23.1 px) is worse than the constant-width helix, the root is broken (inflated w(s)), the narwhal is an analogue, and the 3-D form is judged in 2-D only.  
  evidence: `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L300–386; `docs/media/t1917_032_the_shapes_of_horns_and_of_tee_comparison.png` L0–0; `datasets/thompson_1917/t1917-032/result.json` L1–530
- **Parameter expressivity: 88** — `parameter_sweep` (L639-L700) exercises 30 (λ, c) regimes, 5 twist rates and 16 (λ, c_s) spiral-angle regimes: S1 curvature strictly falls with taper, S2 constant width gives constant curvature, S3 the imposed τ is recovered as Frenet torsion within 2 %, S4 the E5 prediction tan ψ = Δ/c_s matches the fitted ψ with r = 1.000 (RMSE 0.2°) from 14.7° to 85.7°, S5 helix slope rises monotonically with ρ; CLI exposes --sigma-frac, --n-trials, --n-boot, --n-perm and --pass1 (L1315-L1334); five named biological regimes (incisor, Mastodon, mammoth, mammoth + twist, Babirussa) rendered in panel (f).  
  evidence: `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L613–700; `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L1315–1334
- **Morphometric validation: 86** — `measure_morphoeval()` (L1223-L1249) returns 17 measurements: log-spiral r = 0.989 (bootstrap CI 0.984-0.998) and RMSE 36.8 px vs circle 41.8 px vs Archimedean 123.6 px; ψ = 36.9° (CI 34.7-40.1°); Spearman ρ(|κ|, 1/w) = −0.18 with block-shuffled-width permutation p = 0.80 (accretion coupling rejected); screw-fit RMSEs with ΔAIC for twist (477, 355) and the τ = 0 ablation ratio 1.62; narwhal α ≡ 70.2° (z = +1.00 vs Kiladze & Chernova 2018 Table 1, inside the published 58.8-72.8° range), φ bootstrap CI, tan α vs 1/d slope with CI; St Venant sector count = 6 (reference 6) with circular-section null = 0; sweep S4 r = 1.000; one-sample KS normality p < 0.001 reported honestly as systematic misfit.  
  evidence: `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L1223–1249; `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L434–506; `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L747–775
- **Source and data traceability: 90** — Reference JSON pins SHA-256 of 5 dataset files verified fail-closed at import (`load_reference`, L114-L122), Commons licence/author/rendition metadata, IA leaf indices for Figs. 321-322 and p. 634, DOIs for Kiladze 2018 (with PMC id and vendored Table 1 CSV), Kingsley 1988, Raup 1966 and the St Venant 1856 memoir; result.json records every digest, library version and run parameter.  
  evidence: `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L106–142; `datasets/thompson_1917/t1917-032/t1917_032_the_shapes_of_horns_and_of_tee_reference.json` L1–290
- **Code-to-model traceability: 90** — Module header maps E1-E9 to Thompson pages and to the four Lean laws and lists the Pass 2 additions; every function docstring names its equation and, where applicable, the Lean declaration; KG yaml code_locus / lean_locus pass `verify_integrity.py` and `check_lean_formula_alignment.py`.  
  evidence: `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L1–56; `morphoeval/math_kg_loops/thompson1917-t1917-032.yaml` L1–105
- **Reproducibility package: 88** — Fixed seed, `measure_morphoeval` asserted equal across two independent full runs, 25-assertion `run_self_test` (L1251-L1313) covering every Lean law numerically, synthetic log-spiral and synthetic twisted-tusk recovery (τ recovered within 0.0015/px), Procrustes exactness and the measured-width κ·w = Δ identity; library versions and all run parameters written to result.json; documented invocations; full run ~35 s. Deduction: Nelder-Mead restarts are seeded but the optimum is not proven global.  
  evidence: `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L1251–1313; `datasets/thompson_1917/t1917-032/result.json` L1–530

Remediation:
- Acquire a second, intact mammoth or elephant tusk photograph (or a CT centreline) to test the accretion width-law κ ∝ 1/w(s) without the broken-root bias, and fit the screw model jointly to two views to resolve the 3-D helico-spiral directly.
- Measure Kiladze-style chord angles on several narwhal tusks with known tip orientation to test the uniform-torsion prediction tan α ∝ 1/d free of camera obliquity, and record handedness.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-032 — teeth & tusks: differential-accretion log spiral + St Venant torsion
- **spec_fingerprint** (semantic model + inputs): `414a4dcdfdb8857b568651bc291ee292368ae1a49e5e1fb3b1d8e880d3f1ae7f`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-032/kiladze2018_narwhal_spiral_table1.csv` (`f1e9d5a363772f32…`), `datasets/thompson_1917/t1917-032/mammuthus_primigenius_tusk_specimen.jpg` (`cd33e02d49367204…`), `datasets/thompson_1917/t1917-032/monodon_monoceros_tusk_spiral_specimen.jpg` (`3c0e4ccdd686c45b…`), `datasets/thompson_1917/t1917-032/t1917_032_the_shapes_of_horns_and_of_tee_reference.json` (`29f81480c8dea387…`)
- `docs/media/t1917_032_the_shapes_of_horns_and_of_tee_comparison.png` — artifact_sha256 `e154f28e16ab5cbc…`, 1,014,396 B, 2205×1953 px

| measurement | value | reference | method |
|---|---:|---|---|
| log_spiral_pearson_r | 0.9886 1 | — | OLS ln r vs θ about searched pole (E1); block-bootstrap 95% CI [0.984, 0.9981] |
| log_spiral_rmse_px | 36.81 px | — | radial RMSE of E1 fit vs circle null and Archimedean control *(circle null RMSE = 41.78 px; Archimedean control RMSE = 123.6 px)* |
| log_spiral_psi_deg | 36.91 deg | — | atan(1/k); block-bootstrap 95% CI [34.69, 40.1] deg; Thompson: mammoth high angle, Ovis ammon ~75 |
| circle_null_rmse_px | 41.78 px | — | Kasa circle fit (constant-curvature control) |
| cv_kappa_w_accretion_law | 3.588 1 | — | CV of κ·w along centreline (E4) vs CV κ alone; shuffled-width permutation control *(CV kappa = 12.022; shuffled-width median CV = 11.616, CV p = 0.0)* |
| spearman_abs_kappa_vs_inv_width | -0.182 1 | — | Spearman ρ(|κ(s)|, 1/w(s)) along the tusk; E4 predicts > 0; block-shuffled-width permutation p = 0.8 (null control) |
| kappa_sign_changes | 3 count | — | sign changes of width-scaled smoothed curvature root→tip (plane spiral predicts 0) |
| screw_fit_rmse_measured_width_twisted_px | 23.06 px | — | 3-D screw-accretion fit κ=Δ/w(s), τ free, Procrustes-aligned projection (E4-E6) *(plane τ=0 RMSE = 37.32 px; ΔAIC twist = 355.3)* |
| screw_fit_rmse_const_width_twisted_px | 17.37 px | — | 3-D helix fit κ=Δ/w̄, τ free (constant-width ablation of the accretion law) *(plane RMSE = 33.14 px; width-law RMSE ratio = {'plane': 0.888, 'twisted': 0.753})* |
| twist_rmse_ratio_measured_width | 1.619 1 | — | RMSE(τ=0)/RMSE(τ free) for the measured-width accretion model (>1 means twist improves the fit; τ=0 ablation) |
| narwhal_alpha_equiv_deg | 70.22 deg | 66.88 | structure-tensor groove inclination → Kiladze chord angle π/(2 tan φ); bootstrap 95% CI of φ [26.29, 32.96] deg |
| narwhal_tan_alpha_equiv | 2.781 1 | 2.39 | π/(2 tan φ) |
| narwhal_z_alpha | 1.0 SD | — | (α_measured − α_ref)/SD_ref |
| narwhal_tan_alpha_vs_inv_d_slope | -1766.25 px | — | OLS slope of tan α on 1/d along the shaft (E7 uniform torsion predicts > 0); bootstrap 95% CI [-17846.89, -1079.08]; confounded by camera obliquity |
| st_venant_sign_sectors | 6 count | 6 | sign alternations of 3x²y−y³ on a circle about the centroid; circular-section null max|u_z| = 0.0 |
| sweep_psi_pred_vs_fit_pearson_r | 1.0 1 | — | parameter sweep S4: ψ fitted on generated plane curves vs E5 prediction atan(Δ/c_s); RMSE 0.2 deg |
| ks_pvalue_spiral_residual_normality | 0.0 1 | — | one-sample KS of standardised E1 residuals vs N(0,1) |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0
- scikit-image: 0.26.0
- lean: 4.35.0

Pass 2 rendered with `python3 thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` (full mode) after `--self-test` (17 deterministic measurements, two independent full runs compared equal).
