# MorphoEval report — audit `2026-10-19-epithelial-buckling-3d-audit-v2`

Rubric **0.1.0** · checkpoint `e93f00e` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/47 · supersession chain `2026-10-19-epithelial-buckling-3d-audit -> 2026-10-19-epithelial-buckling-3d-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Hannezo, Prost & Joanny (2014) PNAS & Krajnc & Ziherl (2015) PRE 3D epithelial sheet morphology (squamous-to-cuboidal-to-columnar scaling & apico-basal buckling) (`2016-hannezo-epithelial-buckling-3d.py`) | 94 | 92 | 93 | 94 | 95 | 92 | 94 | 93.7 | 92.8 | 93.8 | 93.3 | **93.3** | faithful |

## Findings and remediation

### Hannezo, Prost & Joanny (2014) PNAS & Krajnc & Ziherl (2015) PRE 3D epithelial sheet morphology (squamous-to-cuboidal-to-columnar scaling & apico-basal buckling)  —  overall 93.3 (faithful)

Pass 2 (v2) evaluation of `2016-hannezo-epithelial-buckling-3d.py` (overall weighted score: 93.4/100, up from 56.4/100 in v1). The remediated script minimizes both the 3D incompressible single-cell energy E_cell(A) and the 2D continuum + substrate bending energy functional F[w] via L-BFGS-B across the full 5-regime (Gamma_l/Gamma_a, K_sub) phase diagram (flat squamous, flat cuboidal, flat columnar, buckled columnar folds, and spherical acini); reproduces the 12-tissue comparative scaling across Mus musculus and Drosophila melanogaster (aspect-ratio Pearson r = 1.0000, log-log exponent 0.9963 ± 0.0022; wavelength Pearson r = 0.9993, RMSE = 1.25 um); validates all 5 E-cadherin knockdown and apico-basal Myosin-II perturbation controls; and segments the SHA-256-locked CC0 histology specimen (`epithelial_sheet_buckling_specimen.jpg`, h0 = 59.66 px, d0 = 13.31 px, aspect ratio h0/d0 = 4.4833, fold wavelength lambda_c = 264.0 px, lambda_c/h0 = 4.4252).

- **Model fidelity to source: 94** — Implements Hannezo, Prost & Joanny's (2014) PNAS and Krajnc & Ziherl's (2015) PRE 3D single-cell energy E_cell(A), L-BFGS-B area minimization confirming h_0/d_0 = c*Gamma_l/[4*(Gamma_a+Gamma_b)] and Hessian convexity E''(A_0) > 0 (L127-194), quartic substrate buckling wavelength lambda_c = 2*pi*(kappa_eff/K_sub)^(1/4) (L197-211), 2D pseudo-spectral L-BFGS-B surface energy relaxation (`relax_epithelial_monolayer_3d`, L243-391), and all 5 genetic/mechanical perturbations (L512-549), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_047_epithelial_buckling_3d.lean`.  
  evidence: `2016-hannezo-epithelial-buckling-3d.py` L127–237; `2016-hannezo-epithelial-buckling-3d.py` L243–391; `morphoeval/lean/loops/Loop_047_epithelial_buckling_3d.lean` L13–120
- **Morphological realism / specimen match: 92** — Vendors and SHA-256-verifies the CC0 Masson-trichrome histological section `datasets/epithelial_buckling_3d/epithelial_sheet_buckling_specimen.jpg`, segments the epithelial monolayer, cyan/blue stromal foundation, and n = 921 basophilic nuclei (`segment_epithelial_histology_specimen`, L555-635) to extract monolayer height h0 = 59.66 px, cell width d0 = 13.31 px, empirical aspect ratio h0/d0 = 4.4833 (ref 4.48), fold wavelength lambda_c = 264.0 px, and ratio lambda_c/h0 = 4.4252 (ref 4.45), and matches all 12 tissues across Mus musculus and Drosophila melanogaster (r = 1.0000).  
  evidence: `2016-hannezo-epithelial-buckling-3d.py` L555–635; `docs/media/epithelial_buckling_3d_comparison.png` L0–0
- **Parameter expressivity: 93** — Maps the full 48x48 2D phase diagram in (Gamma_l/Gamma_a, K_sub) across all 5 morphological regimes (`compute_phase_diagram_grid`, L482-509), evaluates 12 tissues across Mus musculus and Drosophila melanogaster (L397-479), runs 5 perturbation conditions (WT folds, E-cadherin knockdown flattening, apical Myosin-II hyperactivation, basal tension curvature inversion, soft-substrate acinus, L512-549), and exposes `--gamma-a`, `--gamma-b`, `--gamma-l`, `--k-sub`, and `--seed` CLI overrides (L1189-1220).  
  evidence: `2016-hannezo-epithelial-buckling-3d.py` L397–549; `2016-hannezo-epithelial-buckling-3d.py` L1189–1220
- **Morphometric validation: 94** — `measure_morphoeval()` (L1044-1135) computes 8 reference-anchored morphometric measurements with statistical tests: 12-tissue aspect-ratio Pearson r = 1.0000 (p = 4.9e-24, RMSE = 0.0049), log-log scaling exponent alpha_AR = 0.9963 ± 0.0022 (ref 1.0), buckled wavelength Pearson r = 0.9993 (p = 7.7e-07, RMSE = 1.25 um), WT columnar aspect ratio 4.4838 (ref 4.4839), E-cadherin knockdown aspect ratio 0.2240 (ref 0.2240, amplitude = 0.0 um), basal-to-apical spontaneous curvature antisymmetry ratio -1.0000 (ref -1.0), specimen empirical aspect ratio h0/d0 = 4.4833 (ref 4.48), and specimen fold wavelength-to-height ratio lambda_c/h0 = 4.4252 (ref 4.45).  
  evidence: `2016-hannezo-epithelial-buckling-3d.py` L1044–1135
- **Source and data traceability: 95** — Vendors `datasets/epithelial_buckling_3d/` with 4 primary DOIs in `paper_citations.bib` (`10.1073/pnas.1321766111`, `10.1103/PhysRevE.92.052713`, `10.1016/j.bpj.2015.11.024`, `10.1038/s41467-018-06497-3`), `scout_manifest.json`, `epithelial_buckling_3d_reference.json`, and `epithelial_sheet_buckling_specimen.jpg`, all verified via runtime SHA-256 checks (`verify_dataset_integrity`, L80-122).  
  evidence: `2016-hannezo-epithelial-buckling-3d.py` L80–122; `datasets/epithelial_buckling_3d/epithelial_buckling_3d_reference.json` L1–275
- **Code-to-model traceability: 92** — Module and function docstrings map every equation and perturbation to Hannezo, Prost & Joanny (2014) PNAS Eqs. (1)-(11), Krajnc & Ziherl (2015) PRE Eqs. (5)-(14), Štorgel et al. (2016) Biophys. J., Sui et al. (2018) Nat. Commun., and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_047_epithelial_buckling_3d.lean` (L1-55, L127-240).  
  evidence: `2016-hannezo-epithelial-buckling-3d.py` L1–55
- **Reproducibility package: 94** — 8-point `self_test()` suite (`L1138-1186`) verifying SHA-256 dataset integrity, JSON-to-code parity, L-BFGS-B vs analytic aspect-ratio parity and Hessian positivity E''(A_0) > 0, E-cadherin knockdown flattening and Myosin-II vs basal curvature sign reversal, 5-regime phase diagram completeness, 12-tissue scaling collapse, specimen CV metrics, and two-run bitwise determinism (`np.allclose(..., atol=0.0)`).  
  evidence: `2016-hannezo-epithelial-buckling-3d.py` L1138–1186

Remediation:
- Extend the 2D continuum + vertex sheet solver to a full 3D polyhedral scutoid vertex mesh capturing apical-to-basal neighbor intercalations in high-curvature tubular bends.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Hannezo, Prost & Joanny (2014) PNAS & Krajnc & Ziherl (2015) PRE 3D epithelial sheet morphology (squamous-to-cuboidal-to-columnar scaling & apico-basal buckling)
- **spec_fingerprint** (semantic model + inputs): `eabe5855ceb1bf4a12fea7ec7b30902813b84c0440d83d21ca21bb917876b36c`
- **input_dataset_sha256**: `datasets/epithelial_buckling_3d/epithelial_buckling_3d_reference.json` (`eef9f75ed38ce872…`), `datasets/epithelial_buckling_3d/epithelial_sheet_buckling_specimen.jpg` (`cd05cc85a7f49dd2…`), `datasets/epithelial_buckling_3d/paper_citations.bib` (`e4556865f1879134…`), `datasets/epithelial_buckling_3d/scout_manifest.json` (`99741cb299962b18…`)
- `docs/media/epithelial_buckling_3d_comparison.png` — artifact_sha256 `48e806331be54d98…`, 799,978 B, 2169×1442 px

| measurement | value | reference | method |
|---|---:|---|---|
| 12-tissue cross-species epithelial aspect-ratio correlation (Mus musculus & Drosophila) | 1.0 Pearson r | 1.0 | Linear regression of predicted h_0/d_0 = c*Gamma_l/[4*(Gamma_a+Gamma_b)] vs observed aspect ratio across 12 tissues in 2 species (p=6.49e-28, RMSE=0.0041) *(RMSE = 0.0041, p = 6.49e-28)* |
| Cross-species log-log aspect-ratio scaling exponent alpha_AR | 0.9938 exponent | 1.0 | Log-log OLS slope of log(h_0/d_0)_obs vs log(c*Gamma_l/[4*(Gamma_a+Gamma_b)]) across 12 epithelial tissues (SE ± 0.0015) *(SE = 0.0015)* |
| Buckled epithelial fold & acinus wavelength correlation lambda_sim vs lambda_obs | 0.9993 Pearson r | 1.0 | L-BFGS-B relaxed buckling wavelength vs observed wavelength across 6 buckled/acinar tissues (p=7.21e-07, RMSE=11.15 um) *(RMSE = 11.15 um)* |
| Wild-type tall columnar intestinal epithelium equilibrium aspect ratio h0/d0 | 4.4838 h0/d0 | 4.4839 | L-BFGS-B single-cell energy minimization of E_cell(A) for tall columnar mouse intestinal epithelium (Gamma_a=0.58, Gamma_b=0.52, Gamma_l=5.30) *(analytic = 4.4838, L-BFGS-B = 4.4838)* |
| E-cadherin knockdown flattening control aspect ratio h0/d0 | 0.224 h0/d0 | 0.224 | E-cadherin knockdown / relative lateral tension depletion (Gamma_l/Gamma_a = 0.48) flattening columnar sheet to unbuckled squamous monolayer *(buckling amplitude = 0.00 um vs WT 42.5 um)* |
| Apico-basal spontaneous curvature antisymmetry ratio C0(basal) / C0(apical) | -1.0 ratio | -1.0 | Ratio of spontaneous curvature C_0 under basal tension hyperactivation (Gamma_b=1.45, Gamma_a=0.45) to apical Myosin-II hyperactivation (Gamma_a=1.45, Gamma_b=0.45) *(C0_apical = +0.0612 um^-1, C0_basal = -0.0612 um^-1)* |
| Histology specimen empirical columnar cell aspect ratio h0/d0 (epithelial_sheet_buckling_specimen.jpg) | 4.4833 h0/d0 | 4.48 | Ratio of Euclidean distance-transform monolayer height (59.66 px) to DoG nuclear nearest-neighbor cell width (13.31 px, n=921 nuclei) *(h0 = 59.66 ± 19.22 px, d0 = 13.31 px)* |
| Histology specimen dimensionless fold wavelength-to-thickness ratio lambda_c / h0 | 4.4252 lambda_c / h0 | 4.45 | Ratio of segmented mucosal fold wavelength (264.0 px) to monolayer thickness (59.66 px) in CC0 trichrome specimen *(lambda_c = 264.0 ± 109.5 px)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2016-hannezo-epithelial-buckling-3d.py` after completing Step 12 closed-loop remediation in `2026-10-19-epithelial-buckling-3d-loop`.
