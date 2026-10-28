# MorphoEval report — audit `2026-10-28-t1917-032-the-shapes-of-horns-and-of-tee-audit`

Rubric **0.1.0** · checkpoint `0e7e77bc1` · assessor `pkr:model/claude-fable-5-1` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/pull/309

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Thompson (1917) T1917-032 — teeth & tusks: differential-accretion log spiral + St Venant torsion (`thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py`) | 62 | 50 | 40 | 55 | 85 | 80 | 72 | 54.7 | 51.9 | 78.5 | 60.7 | **60.7** | mostly faithful |

## Findings and remediation

### Thompson (1917) T1917-032 — teeth & tusks: differential-accretion log spiral + St Venant torsion  —  overall 60.7 (mostly faithful)

Pass 1 baseline of `t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` (overall ~60.7/100). Implements E1-E9 (equiangular spiral, accretion curvature law, screw accretion, St Venant warping and uniform-torsion helix), segments the SHA-256-locked mammoth tusk, fits a log spiral about a searched pole against a circle null (RMSE 37.3 vs 41.8 px, r = 0.963), measures narwhal groove inclination by structure tensor (α ≡ 70.2° vs 66.9 ± 3.3° published, z = +1.0) and reproduces the six-sector St Venant torsion curves. Weaknesses: the 3-D screw model is never fitted to the specimen, curvature changes sign three times so the plane-spiral reading fails, regimes are five hand-picked points, statistics are thin (two-sample KS misuse, no CI / bootstrap, no ablation controls beyond the circle fit), no CLI parameterisation.

- **Model fidelity to source: 62** — E1-E3 (`log_spiral_r`, `whorl_expansion`, `gnomon_radius`, L112-L131), E4-E6 screw accretion (`grow_tooth`, L146-L181), E8-E9 St Venant warping (`tri_warp`, `ellipse_warp`, `st_venant_triangle_field`, L186-L223) and E7 helix (L228-L242) are implemented as published and mirrored by 17 Lean theorems, but the specimen test uses only the 2-D plane-spiral special case; the twisted (τ > 0) model that Thompson invokes for long tusks is rendered, never fitted.  
  evidence: `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L112–181; `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L186–242; `morphoeval/lean/loops/Thompson1917_T1917_032.lean` L31–215; `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L445–470
- **Morphological realism / specimen match: 50** — HSV segmentation + skeleton longest path recovers the mammoth tusk centreline and width (`segment_mammoth_tusk`, `tusk_centreline_profile`, L247-L328) and the overlay beats the circle null, but the smoothed curvature changes sign 3 times (root bend, straight shaft, reversed tip hook) so the plane log spiral mis-tracks root and tip; the narwhal panel is an analogue; no comparison against Thompson's own tusk description beyond text.  
  evidence: `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L247–328; `docs/media/t1917_032_the_shapes_of_horns_and_of_tee_comparison.png` L0–0; `datasets/thompson_1917/t1917-032/result.json` L1–57
- **Parameter expressivity: 40** — Only five hand-picked (λ, c, τ) regimes are rendered in panel (f) (L477-L483); no systematic sweep of ψ, λ, taper c or twist τ, no CLI overrides, and the monotone effects proved in Lean (curvature falls with width; helix slope rises with ρ) are not demonstrated numerically across ranges.  
  evidence: `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L477–483; `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L803–820
- **Morphometric validation: 55** — `measure_morphoeval()` (L736-L755) returns 11 measurements with one literature reference (Kiladze & Chernova 2018, z = +1.0 within 2 SD) and a St Venant sector count against theory, plus CV(κ·w) < CV(κ) (3.59 vs 12.0); but the KS statistic is a two-sample test between the spiral and circle residual sets (p = 0 is uninformative), there is no bootstrap CI on ψ or on the groove angle, no shuffled-width control for the accretion law, and no along-shaft test of tan α vs 1/d.  
  evidence: `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L736–755; `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L459–466; `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L378–391
- **Source and data traceability: 85** — Reference JSON pins SHA-256 of 5 dataset files verified at import (`load_reference`, L86-L94), Commons licence/author metadata, IA leaf index for Figs. 321-322, DOIs for Kiladze 2018, Kingsley 1988, Raup 1966 and the St Venant 1856 memoir; the Kiladze table is vendored as CSV with its PMC id.  
  evidence: `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L78–106; `datasets/thompson_1917/t1917-032/t1917_032_the_shapes_of_horns_and_of_tee_reference.json` L1–140
- **Code-to-model traceability: 80** — Module header maps E1-E9 to Thompson pages and to the four Lean laws; each function docstring cites its equation number and Lean declaration; KG yaml loci verified by `verify_integrity.py`.  
  evidence: `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L1–40; `morphoeval/math_kg_loops/thompson1917-t1917-032.yaml` L1–105
- **Reproducibility package: 72** — Fixed seed, deterministic `measure_morphoeval` (asserted twice-equal in `run_self_test`, L758-L800), 13 self-test assertions mirroring the Lean laws, documented invocation; but library versions are not pinned in the script, smoothing scale (σ = 50 px) is a hand-set constant, and `--pass1` vs full differ only in the pole grid.  
  evidence: `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L758–800; `thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py` L304–328

Remediation:
- Fit the 3-D screw-accretion model (E6: ψ or Δ/c, twist τ, taper, view rotation, scale) to the projected mammoth centreline and compare nested τ = 0 (plane spiral) vs τ > 0 (helico-spiral) by RMSE / AIC to test Thompson's 'curve of double curvature'.
- Add a systematic parameter-regime sweep over ψ (20-85°), λ, taper c and τ demonstrating the Lean-proved monotonicities numerically and rendering a morphospace panel.
- Add negative / ablation controls: Archimedean spiral fit, shuffled-width control for CV(κ·w), τ = 0 ablation, circular-section (a = b) St Venant null.
- Replace the two-sample KS misuse by a one-sample KS on standardised residuals; add bootstrap CIs for ψ, Pearson r and the narwhal groove angle; regress tan α on 1/d along the shaft for the uniform-torsion prediction.
- Width-scale the centreline smoothing, expose CLI overrides (--psi, --lam, --taper, --tau, --sigma-frac), and record library versions in result.json.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Thompson (1917) T1917-032 — teeth & tusks: differential-accretion log spiral + St Venant torsion
- **spec_fingerprint** (semantic model + inputs): `d3a3c036d2236fa5e48fed4d773ebe5efc0177f2fb7ef111e8e3fe7dfc6e5241`
- **input_dataset_sha256**: `datasets/thompson_1917/t1917-032/kiladze2018_narwhal_spiral_table1.csv` (`f1e9d5a363772f32…`), `datasets/thompson_1917/t1917-032/mammuthus_primigenius_tusk_specimen.jpg` (`cd33e02d49367204…`), `datasets/thompson_1917/t1917-032/monodon_monoceros_tusk_spiral_specimen.jpg` (`3c0e4ccdd686c45b…`), `datasets/thompson_1917/t1917-032/t1917_032_the_shapes_of_horns_and_of_tee_reference.json` (`6225b4ee87df718c…`)
- `docs/media/t1917_032_the_shapes_of_horns_and_of_tee_comparison.png` — artifact_sha256 `ce504b6b180f886e…`, 901,553 B, 2200×1265 px

| measurement | value | reference | method |
|---|---:|---|---|
| log_spiral_pearson_r | 0.9886 1 | — | OLS ln r vs θ about searched pole (E1) |
| log_spiral_rmse_px | 36.81 px | — | radial RMSE of E1 fit vs circle null RMSE *(circle null RMSE = 41.78 px)* |
| log_spiral_psi_deg | 36.92 deg | — | atan(1/k); Thompson: mammoth 'high' spiral angle, Ovis ammon ~75 |
| circle_null_rmse_px | 41.78 px | — | Kasa circle fit (constant curvature control) |
| cv_kappa_w_accretion_law | 3.588 1 | — | CV of κ·w along centreline (E4) vs CV κ alone *(CV kappa = 12.021)* |
| kappa_sign_changes | 3 count | — | sign changes of smoothed curvature root→tip (plane spiral predicts 0) |
| narwhal_alpha_equiv_deg | 70.22 deg | 66.88 | structure-tensor groove inclination → Kiladze chord angle π/(2 tan φ) |
| narwhal_tan_alpha_equiv | 2.781 1 | 2.39 | π/(2 tan φ) |
| narwhal_z_alpha | 1.0 SD | — | (α_measured − α_ref)/SD_ref |
| st_venant_sign_sectors | 6 count | 6 | sign alternations of 3x²y−y³ on a circle about the centroid |
| ks_pvalue_spiral_vs_circle_residuals | 0.0 1 | — | two-sample KS test |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0
- scikit-image: 0.26.0
- lean: 4.35.0

Pass 1 rendered with `python3 thompson/claims/t1917-032_the-shapes-of-horns-and-of-teeth-or-tusks.py --pass1` after `--self-test` (11 deterministic measurements).
