# MorphoEval report — audit `2026-10-19-limb-bud-elongation-pushing-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/17 · supersession chain `2026-10-19-limb-bud-elongation-pushing-audit -> 2026-10-19-limb-bud-elongation-pushing-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Boehm et al. (2010) & Gros et al. (2010) vertebrate limb bud oriented cell division & Wnt5a/FGF8 pushing vs proliferation-gradient hypothesis (`2015-boehm-limb-bud-elongation-pushing.py`) | 94 | 93 | 94 | 95 | 95 | 94 | 94 | 94.0 | 93.8 | 94.3 | 94.0 | **94.0** | faithful |

## Findings and remediation

### Boehm et al. (2010) & Gros et al. (2010) vertebrate limb bud oriented cell division & Wnt5a/FGF8 pushing vs proliferation-gradient hypothesis  —  overall 94.0 (faithful)

Pass 2 (v2) evaluation of `2015-boehm-limb-bud-elongation-pushing.py` (overall weighted score: 93.8/100, up from 57.3/100 in v1). The remediated script solves the 2D/3D continuum artificial-compressibility pressure-Poisson & Stokes velocity field (`nabla . v = s(x)`, `-nabla p + (1/Re)*nabla^2 v + nabla . sigma_active = 0`) coupled to Wnt5a/JNK oriented cell division active stress and AER-FGF8/MAPK velocity gradients; reproduces the 5-regime PD elongation hierarchy (Pearson r = 0.9970, RMSE = 1.232%) and E11.5 aspect-ratio hierarchy (Pearson r = 0.9920, RMSE = 0.0801); confirms 5-decade Reynolds-number insensitivity (Re in [10^-6, 10^-1], r = 0.9995, span = 8.60 percentage points); demonstrates the 11.74x Hooke-Jeeves inverse-optimization proliferation gap; and segments 5 sub-panels of the SHA-256-locked OPT specimen image `mouse_limb_bud_specimen.jpg`.

- **Model fidelity to source: 94** — Implements 3D principal strain-rate decomposition (`compute_principal_strain_rates`, L128-153), empirical dual-label IddU/BrdU cell-cycle source profiles vs Hooke-Jeeves inverse-optimized sources (`compute_iddu_brdu_atlas_profile`, L156-199), an artificial-compressibility pressure-Poisson and viscous velocity solver (`solve_continuum_pressure_velocity_field`, L205-294), and 12-substep RK2 Lagrangian boundary and marker-clone advection across all 5 experimental regimes (`simulate_limb_bud_regime`, L297-461), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_017_limb_bud_elongation_pushing.lean`.  
  evidence: `2015-boehm-limb-bud-elongation-pushing.py` L128–199; `2015-boehm-limb-bud-elongation-pushing.py` L205–461; `morphoeval/lean/loops/Loop_017_limb_bud_elongation_pushing.lean` L21–152
- **Morphological realism / specimen match: 93** — Vendors and SHA-256-verifies the CC BY 4.0 Boehm et al. (2010) Fig. 2 mouse limb bud specimen `datasets/limb_bud_elongation_pushing/mouse_limb_bud_specimen.jpg`, performs automated contour segmentation across 5 sub-panels (`segment_limb_bud_specimen`, L577-673) — extracting Panel F dorsal St1/St0 PD ratio = 1.3404, Panel F posterior St1/St0 PD ratio = 1.3587 (AR 1.0888 -> 1.3021), Panel D/E 3D OPT iso-surface AR gain = 1.1732, and Panel G FEM cyan mesh fraction = 0.3694 — and matches the 5-regime empirical hierarchy (r = 0.9970).  
  evidence: `2015-boehm-limb-bud-elongation-pushing.py` L577–673; `docs/media/limb_bud_elongation_pushing_comparison.png` L0–0
- **Parameter expressivity: 94** — Evaluates all 5 biological perturbation regimes (`evaluate_all_five_regimes`, L464-514), a 6-point Reynolds number sensitivity sweep across 5 orders of magnitude Re in [10^-6, 10^-1] (`run_reynolds_and_2d_parameter_sweeps`, L517-571), and a 25-point 5x5 2D morphospace grid over (zeta_OCD, chi_FGF), plus CLI flags `--zeta-ocd`, `--chi-fgf`, `--reynolds`, `--hours`, `--seed`, `--output`, `--gif`, and `--self-test` (L1158-1188).  
  evidence: `2015-boehm-limb-bud-elongation-pushing.py` L464–571; `2015-boehm-limb-bud-elongation-pushing.py` L1158–1188
- **Morphometric validation: 95** — `measure_morphoeval()` (L998-1090) computes 8 reference-anchored morphometric measurements: 5-regime 6h PD elongation Pearson r = 0.9970 (RMSE = 1.232%), 5-regime E11.5 aspect-ratio Pearson r = 0.9920 (RMSE = 0.0801), Model II WT 6h PD elongation = 43.3706% (ref 42.16%), Model I negative control = 10.1843% (ref 10.10%), 5-decade Reynolds sensitivity span = 8.6005 percentage points (ref 8.4, r = 0.9995), Specimen Panel F posterior St1/St0 ratio = 1.3587 (derived_from: specimen), Specimen Panel D/E 3D OPT iso-surface AR gain = 1.1732 (derived_from: specimen), and Hooke-Jeeves inverse-opt to empirical distal source ratio = 11.7357 (ref 11.7358).  
  evidence: `2015-boehm-limb-bud-elongation-pushing.py` L998–1090
- **Source and data traceability: 95** — Vendors `datasets/limb_bud_elongation_pushing/` with 6 primary DOIs in `paper_citations.bib` (`10.1371/journal.pbio.1000420`, `10.1371/journal.pbio.1000421`, `10.1016/j.cub.2010.09.063`, `10.1242/dev.046987`, `10.1007/s11538-008-9334-y`, `10.1006/jtbi.1998.0876`), `scout_manifest.json`, `limb_bud_elongation_pushing_reference.json`, and `mouse_limb_bud_specimen.jpg`, all cryptographically verified at runtime via SHA-256 (`verify_reference_dataset`, L105-122).  
  evidence: `2015-boehm-limb-bud-elongation-pushing.py` L105–122; `datasets/limb_bud_elongation_pushing/limb_bud_elongation_pushing_reference.json` L1–145
- **Code-to-model traceability: 94** — Module and function docstrings map every equation to Boehm et al. (2010) PLoS Biology Eqs. 1-11 & Figs. 2-9, Gros et al. (2010) Current Biology Figs. 1-6, Kicheva & Briscoe (2010) PLoS Biology, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_017_limb_bud_elongation_pushing.lean` (L1-55, L128-320).  
  evidence: `2015-boehm-limb-bud-elongation-pushing.py` L1–55
- **Reproducibility package: 94** — 8-point `self_test()` suite (`L1093-1155`) verifying SHA-256 dataset integrity, principal strain-rate laws (isotropic deviatoric = 0 vs anisotropic > 0), IddU/BrdU vs Hooke-Jeeves inverse-opt 11.74x distal gap, continuum pressure-Poisson convergence, 5-regime hierarchy parity (r > 0.99), 5-decade Reynolds insensitivity & 2D morphospace monotonicity, specimen CV segmentation, and two-run bitwise determinism (`np.allclose(..., atol=0.0)`).  
  evidence: `2015-boehm-limb-bud-elongation-pushing.py` L1093–1155

Remediation:
- Extend the 2D mid-plane pressure-Poisson solver to an unstructured 3D tetrahedral finite-element mesh imported directly from Optical Projection Tomography iso-surfaces.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Boehm et al. (2010) & Gros et al. (2010) vertebrate limb bud oriented cell division & Wnt5a/FGF8 pushing vs proliferation-gradient hypothesis
- **spec_fingerprint** (semantic model + inputs): `4c3f0cfc301e2c955c4f4f6c7d2e02e3996ca3b4cd571d212368d55baab36b29`
- `docs/media/limb_bud_elongation_pushing_comparison.png` — artifact_sha256 `e809b1fc6f9248c8…`, 991,910 B, 3300×1380 px

| measurement | value | reference | method |
|---|---:|---|---|
| 5-regime 6h PD elongation hierarchy Pearson r vs empirical | 0.997 Pearson r | 1.0 | evaluate_all_five_regimes (RMSE = 1.232%) *(RMSE = 1.232% across WT, FGF-only, Wnt5a-/-, Model I, and Uniform controls)* |
| 5-regime E11.5 aspect ratio L_PD/W_AP Pearson r vs empirical | 0.992 Pearson r | 1.0 | evaluate_all_five_regimes E11.5 L_PD/W_AP (RMSE = 0.0801) *(WT AR = 2.298 vs Model I bulbous AR = 1.115 and Wnt5a-/- AR = 1.286 (RMSE = 0.0801))* |
| Model II wild-type 6h PD elongation vs OPT St0->St1 benchmark | 43.3706 % | 42.16 | simulate_limb_bud_regime(model_ii_wt_directional_pushing, 6.0 h) *(Directional Wnt5a/JNK oriented cell division + AER-FGF8 velocity field reproduces +42.2% PD outgrowth)* |
| Model I isotropic proliferation 6h PD elongation at Re=10^-2 (negative control) | 10.1843 % | 10.1 | simulate_limb_bud_regime(model_i_isotropic_proliferation_gradient, Re=1e-2) *(Refutes pure proliferation-gradient hypothesis (bulbous expansion ~10.1% vs empirical 42.2%))* |
| Reynolds number 5-decade sensitivity span (Re in [10^-6, 10^-1]) | 8.6005 percentage points | 8.4 | run_reynolds_and_2d_parameter_sweeps (Pearson r = 0.9995) *(Confirms viscosity insensitivity (<9 percentage points variation across 5 decades, r = 0.9995))* |
| Specimen Panel F posterior DV-PD St1/St0 elongation ratio | 1.3587 L_PD(St1) / L_PD(St0) | 1.3587 | segment_limb_bud_specimen Panel F posterior view contour segmentation *(Aspect ratio increases from 1.0888 (St0) to 1.3021 (St1))* |
| Specimen Panel D vs Panel E 3D OPT iso-surface aspect ratio gain | 1.1732 AR(St1) / AR(St0) | 1.1732 | segment_limb_bud_specimen Panels D & E connected-component bounding-box ratio *(St0 AR = 1.0049 -> St1 AR = 1.1789)* |
| Hooke-Jeeves inverse-opt to empirical distal source ratio s_opt(1)/s_emp(1) | 11.7357 s_opt / s_emp | 11.7358 | compute_iddu_brdu_atlas_profile(x_pd_norm=1.0) inverse-opt vs empirical net rate *(Demonstrates >11x net (>8.5x gross) proliferation gap proving isotropic growth is biologically unrealistic)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2015-boehm-limb-bud-elongation-pushing.py` after completing Step 12 closed-loop remediation in `2026-10-19-limb-bud-elongation-pushing-loop`.
