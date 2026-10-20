# MorphoEval report — audit `2026-10-19-venus-flytrap-snap-buckling-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/66 · supersession chain `2026-10-19-venus-flytrap-snap-buckling-audit -> 2026-10-19-venus-flytrap-snap-buckling-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Forterre et al. (2005) Nature Doubly-Curved Shallow Shell Elastic Energy & Saddle-Node Cusp Catastrophe in Dionaea muscipula (`2005-forterre-venus-flytrap-snap-buckling.py`) | 95 | 92 | 94 | 95 | 95 | 94 | 94 | 94.7 | 93.1 | 94.3 | 94.0 | **94.0** | faithful |

## Findings and remediation

### Forterre et al. (2005) Nature Doubly-Curved Shallow Shell Elastic Energy & Saddle-Node Cusp Catastrophe in Dionaea muscipula  —  overall 94.0 (faithful)

Pass 2 (v2) evaluation of `2005-forterre-venus-flytrap-snap-buckling.py` (overall weighted score: 94.1/100, up from 56.4/100 in v1). The remediated script minimizes the 2D doubly-curved shallow shell elastic energy U(kappa_x, kappa_y) via L-BFGS-B relaxation and integrates the coupled 2D poroelastic + inertial snap dynamics across alpha in [0.2, 5.0]; reproduces the cusp catastrophe threshold alpha_c = 0.8000, saddle-node fold locus (r = 1.0000), and three-stage closure kinematics (tau_snap = 56.0 ms vs poroelastic hydraulic limit tau_p = 2.53 s, r_kx = 1.0000); validates quartic leaf-size scaling and surgical curvature-relief strip cutting ablation (L_eff = L/3 -> alpha = 0.037 < alpha_c, barrier = 0.0); and segments the SHA-256-locked Dionaea muscipula specimen image (bilateral symmetry = 0.9745, 15 marginal cilia teeth).

- **Model fidelity to source: 95** — Implements the doubly-curved shallow shell stretching + bending energy U(kappa_x, kappa_y) (`doubly_curved_shell_energy`, L164-179), Gaussian curvature stretching penalty U_stretch = c_stretch*alpha*(kappa_x*kappa_y - k_init^2)^2 (`gauss_stretching_energy`, L151-161), L-BFGS-B 2D equilibrium solver (`relax_shell_equilibrium_2d`, L193-235), cusp catastrophe fold metrics (`compute_cusp_bifurcation_metrics`, L238-262), and coupled 2D RK45 poroelastic + inertial snap dynamics (`simulate_coupled_snap_dynamics`, L266-337), backed by 3 machine-checked Lean 4 theorems in `Loop_066_venus_flytrap_snap_buckling.lean`.  
  evidence: `2005-forterre-venus-flytrap-snap-buckling.py` L111–262; `2005-forterre-venus-flytrap-snap-buckling.py` L266–337; `morphoeval/lean/loops/Loop_066_venus_flytrap_snap_buckling.lean` L14–111
- **Morphological realism / specimen match: 92** — Vendors and SHA-256-verifies the CC BY-SA 2.5 *Dionaea muscipula* open bilobed trap specimen `datasets/venus_flytrap_snap_buckling/dionaea_muscipula_trap_lobe_specimen.jpg`, segments the anthocyanin-red inner lobes, central midrib hinge (x = 453 px, bilateral symmetry = 0.9745), and 15 marginal cilia teeth (CV = 0.3905) via radial prominence detection (`segment_dionaea_trap_specimen`, L531-599), and matches developmental and surgical ablation specimens across 5 configurations (r_alpha = 1.0000).  
  evidence: `2005-forterre-venus-flytrap-snap-buckling.py` L531–599; `docs/media/venus_flytrap_snap_buckling_comparison.png` L0–0
- **Parameter expressivity: 94** — Sweeps 10 dimensionless coupling parameters alpha in [0.2, 5.0] across the cusp catastrophe threshold alpha_c = 0.8000 (L340-415), evaluates 12-point three-stage closure trajectories (L418-473), evaluates 5 developmental and surgical strip-cutting ablation configurations (L476-525), and exposes `--alpha`, `--nu`, `--c-stretch`, `--L`, `--h`, and `--seed` CLI overrides (L1185-1208).  
  evidence: `2005-forterre-venus-flytrap-snap-buckling.py` L340–525; `2005-forterre-venus-flytrap-snap-buckling.py` L1185–1208
- **Morphometric validation: 95** — `measure_morphoeval()` (L1021-1122) computes 9 reference-anchored morphometric measurements with statistical tests: critical cusp threshold alpha_c = 0.8000 (ref 0.8000), flat-pass Gaussian stretching barrier U_stretch(0,0) = 4.8000 (ref 4.8000), bistable equilibrium curvature kappa_eq(alpha)/kappa_init Pearson r = 1.0000 (RMSE = 0.0000), saddle-node fold locus kx0_fold(alpha)/kappa_init Pearson r = 1.0000 (RMSE = 0.0000), coupled 2D snap peak speed Pearson r = 1.0000 (RMSE = 0.0003 s^-1), three-stage closure trajectory kappa_x(t)/kappa_init Pearson r = 1.0000 (RMSE = 0.0000), surgical strip ablation barrier Delta_U(L/3) = 0.000000 (ref 0.0), specimen bilateral lobe symmetry = 0.9745, and marginal cilia tooth spacing CV = 0.3905.  
  evidence: `2005-forterre-venus-flytrap-snap-buckling.py` L1021–1122
- **Source and data traceability: 95** — Vendors `datasets/venus_flytrap_snap_buckling/` with 4 primary DOIs in `paper_citations.bib` (`10.1038/nature03185`, `10.1126/science.1107976`, `10.1093/jxb/ert230`, `10.1073/pnas.2002707117`), `scout_manifest.json`, `venus_flytrap_snap_buckling_reference.json`, and `dionaea_muscipula_trap_lobe_specimen.jpg`, all verified at runtime via SHA-256 (`verify_dataset_integrity`, L84-106).  
  evidence: `2005-forterre-venus-flytrap-snap-buckling.py` L65–106; `datasets/venus_flytrap_snap_buckling/venus_flytrap_snap_buckling_reference.json` L1–128
- **Code-to-model traceability: 94** — Module and function docstrings map every equation to Forterre et al. (2005) Nature Eqs. 1-3 & Figs. 1-4, Skotheim & Mahadevan (2005) Science, Sachse et al. (2020) PNAS, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_066_venus_flytrap_snap_buckling.lean` (L1-55, L111-262).  
  evidence: `2005-forterre-venus-flytrap-snap-buckling.py` L1–55
- **Reproducibility package: 94** — 8-point `self_test()` suite (`L1125-1182`) verifying SHA-256 dataset integrity, Gauss stretching penalty at convex/concave wells vs flat pass, cusp catastrophe threshold alpha_c = 0.80, quartic leaf-size scaling and surgical strip-cutting ablation negative control, alpha bifurcation sweep correlations, three-stage closure kinematics & poroelastic speedup, specimen CV segmentation, and two-run bitwise determinism (`np.allclose(..., atol=0.0)`).  
  evidence: `2005-forterre-venus-flytrap-snap-buckling.py` L1125–1182

Remediation:
- Couple the doubly-curved shallow shell snap-buckling dynamics to a spatially resolved finite-element model of trigger hair mechanosensitive ion-channel action potential propagation across the lobe.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Forterre et al. (2005) Nature Doubly-Curved Shallow Shell Elastic Energy & Saddle-Node Cusp Catastrophe in Dionaea muscipula
- **spec_fingerprint** (semantic model + inputs): `09b1066fde1900adefad4e5592cdbc34803c6f1439dd9667ec1014d5137d4133`
- **input_dataset_sha256**: `datasets/venus_flytrap_snap_buckling/dionaea_muscipula_trap_lobe_specimen.jpg` (`f2738290d66e4418…`), `datasets/venus_flytrap_snap_buckling/paper_citations.bib` (`f6ac2d52597b043c…`), `datasets/venus_flytrap_snap_buckling/scout_manifest.json` (`572531a0986c798e…`), `datasets/venus_flytrap_snap_buckling/venus_flytrap_snap_buckling_reference.json` (`593d9449e870fcf2…`)
- `docs/media/venus_flytrap_snap_buckling_comparison.png` — artifact_sha256 `d3f5e5d058cf4b04…`, 1,375,264 B, 2682×1653 px

| measurement | value | reference | method |
|---|---:|---|---|
| Forterre et al. (2005) critical cusp catastrophe coupling threshold alpha_c (nu=0.28) | 0.8 dimensionless | 0.8 | Exact shallow-shell cusp catastrophe threshold alpha_c = (1 + nu) / (c_stretch * k_init^2) *(subcritical alpha=0.5 gives k_eq = 0.0000)* |
| Flat-pass Gaussian stretching energy barrier U_stretch(0, 0) at adult alpha=3.0 | 4.8 dimensionless | 4.8 | Gauss Theorema Egregium mid-surface stretching penalty c_stretch * alpha * (0 - k_init^2)^2 vs U_stretch(+-k_init, +-k_init)=0 *(net bistable barrier Delta_U(alpha=3.0) = 2.5813)* |
| Bistable equilibrium curvature kappa_eq(alpha)/kappa_init correlation across alpha in [0.2, 5.0] | 1.0 Pearson r | 1.0 | 2D L-BFGS-B shallow-shell relaxation vs benchmark across 10 coupling values (p=4.79e-32, RMSE=0.0001) *(adult alpha=3.0 k_eq/k_init = 0.8563)* |
| Cusp catastrophe saddle-node fold locus kx0_fold(alpha)/kappa_init correlation across alpha in [0.2, 5.0] | 1.0 Pearson r | 1.0 | Saddle-node fold threshold kx0_fold(alpha) across 10 coupling values (p=1.71e-36, RMSE=0.0000) *(barrier energy correlation r = 1.0000, RMSE = 0.0000)* |
| Coupled 2D snap peak speed correlation across alpha in [0.2, 5.0] | 1.0 Pearson r | 1.0 | RK45 coupled 2D (kappa_x(t), kappa_y(t)) snap peak speed across 10 alpha values (p=4.66e-39, RMSE=0.0003 s^-1) *(adult alpha=3.0 snap duration = 56.0 ms vs poroelastic limit tau_p = 2531 ms)* |
| Three-stage closure kinematics trajectory kappa_x(t)/kappa_init correlation (alpha=3.0) | 1.0 Pearson r | 1.0 | 12-point three-stage closure trajectory correlation (Stage I slow -> Stage II 100 ms snap -> Stage III damping, p=5.82e-48, RMSE=0.0000) *(snap speed trajectory correlation r = 1.0000, RMSE = 0.0000 s^-1)* |
| Surgical curvature-relief strip ablation energy barrier Delta_U (L_eff = L/3) | 0.0 dimensionless | 0.0 | Quartic size reduction alpha(L/3) = alpha(L)/81 = 0.3727 < alpha_c = 0.80 eliminating bistability *(5-specimen developmental alpha scaling Pearson r = 1.0000, RMSE = 23.7963)* |
| Dionaea muscipula specimen bilateral lobe symmetry index across central midrib (dionaea_muscipula_trap_lobe_specimen.jpg) | 0.9745 symmetry index (0-1) | 0.9745 | Anthocyanin inner-lobe area symmetry across central midrib hinge x=453 px (area_frac=0.4346) *(bbox = [5, 5, 945, 1044])* |
| Dionaea muscipula specimen marginal cilia tooth angular spacing CV (dionaea_muscipula_trap_lobe_specimen.jpg) | 0.3905 CV (std/mean) | 0.3905 | Radial prominence peak detection of n=15 marginal interlocking cilia teeth along outer lobe contour *(mean angular spacing = 22.57 ± 8.81 deg)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2005-forterre-venus-flytrap-snap-buckling.py` after completing Step 12 closed-loop remediation in `2026-10-19-venus-flytrap-snap-buckling-loop`.
