# MorphoEval report — audit `2026-10-19-chiral-seed-pod-opening-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/65 · supersession chain `2026-10-19-chiral-seed-pod-opening-audit -> 2026-10-19-chiral-seed-pod-opening-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Armon et al. (2011) Science / Efrati et al. (2009) JMPS Incompatible Shell Elasticity & Chiral Seed Pod Opening (`2007-liang-chiral-seed-pod-opening.py`) | 95 | 91 | 93 | 94 | 95 | 93 | 94 | 94.3 | 92.1 | 94.1 | 93.4 | **93.4** | faithful |

## Findings and remediation

### Armon et al. (2011) Science / Efrati et al. (2009) JMPS Incompatible Shell Elasticity & Chiral Seed Pod Opening  —  overall 93.4 (faithful)

Pass 2 (v2) evaluation of `2007-liang-chiral-seed-pod-opening.py` (overall weighted score: 93.4/100, up from 55.8/100 in v1). The remediated script minimizes the non-Euclidean incompatible plate energy functional E = E_stretch + E_bend across dimensionless width w_tilde in [0.5, 10] via L-BFGS-B relaxation; reproduces the supercritical pitchfork bifurcation at w_tilde_c = 2.1609 separating narrow Pure-Twist Helicoids (H = 0, R = 0) from wide Cylindrical Helical Ribbons (r_H = 1.0000, r_R = 1.0000, r_P = 1.0000); validates the cutting-angle chirality sweep theta in [0 deg, 90 deg] including theta = 0 deg and 90 deg achiral circular ring controls (P = 0.000000 cm) and 4-system scaling (r_P = 0.9994); and segments the SHA-256-locked Bauhinia variegata specimen image (8 helical twist lobes, CV = 0.2405).

- **Model fidelity to source: 95** — Implements orthogonal +-45 deg sclerenchyma bilayer hygroscopic shrinkage Delta_eps_hygro -> kappa_0 (`compute_sclerenchyma_bilayer_kappa0`, L138-150), rotated saddle target curvature tensor b_0(theta) with Gauss incompatibility det(b_0) = -kappa_0^2 < 0 (`compute_target_curvature_tensor`, L152-186), exact Föppl-von Kármán critical width w_tilde_c = (16*(1+nu)/sqrt(1-nu^2))^(1/4) (`critical_fvk_width`, L188-194), and an L-BFGS-B non-Euclidean plate energy minimizer (`relax_incompatible_ribbon_equilibrium`, L253-389) with negative controls at kappa_0 = 0 and theta = 0 deg, 90 deg, backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_065_chiral_seed_pod_opening.lean`.  
  evidence: `2007-liang-chiral-seed-pod-opening.py` L138–250; `2007-liang-chiral-seed-pod-opening.py` L253–469; `morphoeval/lean/loops/Loop_065_chiral_seed_pod_opening.lean` L14–102
- **Morphological realism / specimen match: 91** — Vendors and SHA-256-verifies the CC BY-SA 3.0 *Bauhinia variegata* dehisced chiral seed pod specimen `datasets/chiral_seed_pod_opening/bauhinia_chiral_seed_pod_specimen.jpg` (Muséum de Toulouse MHNT.BOT.2011.3.22), segments the upper and lower twisted valves and 8 helical lobe apices (mean half-pitch 169.83 px, CV = 0.2405) via inner-tan vs outer-brown sclerenchyma reflectance (`segment_bauhinia_pod_specimen`, L687-770), and matches radius and pitch across 4 benchmark systems (r_R = 1.0000, r_P = 0.9994).  
  evidence: `2007-liang-chiral-seed-pod-opening.py` L687–770; `docs/media/chiral_seed_pod_opening_comparison.png` L0–0
- **Parameter expressivity: 93** — Sweeps 11 dimensionless widths w_tilde in [0.5, 10.0] across the pitchfork bifurcation w_tilde_c = 2.1609 (L472-549), sweeps 7 cutting angles theta in [0 deg, 90 deg] including achiral circular ring controls (L552-617), evaluates 4 legume and biomimetic latex bilayer systems (L620-681), and exposes `--kappa0`, `--nu`, `--W`, `--h`, `--L`, and `--seed` CLI overrides (L1224-1256).  
  evidence: `2007-liang-chiral-seed-pod-opening.py` L472–681; `2007-liang-chiral-seed-pod-opening.py` L1224–1256
- **Morphometric validation: 94** — `measure_morphoeval()` (L1053-1154) computes 9 reference-anchored morphometric measurements with uncertainty and statistical tests: critical width w_tilde_c = 2.1609 (ref 2.1609), saddle Gaussian target curvature det(b_0) = -0.1764 cm^-2 (ref -0.1764), pitchfork mean curvature H*(w_tilde)/kappa_0 Pearson r = 1.0000 (RMSE = 0.0006), helix radius R(w_tilde)*kappa_0 Pearson r = 1.0000 (RMSE = 0.0022), helical pitch P(w_tilde)*kappa_0/(2*pi) Pearson r = 1.0000 (RMSE = 0.0027), cutting-angle torsion tau(theta)/kappa_0 Pearson r = 1.0000, achiral ring pitch control P(theta=0 deg) = 0.000000 cm (ref 0.0), 4-system helical pitch Pearson r = 0.9994 (RMSE = 0.1771 cm), and *Bauhinia variegata* specimen half-pitch CV = 0.2405.  
  evidence: `2007-liang-chiral-seed-pod-opening.py` L1053–1154
- **Source and data traceability: 95** — Vendors `datasets/chiral_seed_pod_opening/` with 4 primary DOIs in `paper_citations.bib` (`10.1126/science.1203874`, `10.1016/j.jmps.2008.12.004`, `10.1039/c3sm52313f`, `10.1038/nature03185`), `scout_manifest.json`, `chiral_seed_pod_opening_reference.json`, and `bauhinia_chiral_seed_pod_specimen.jpg`, all verified at runtime via SHA-256 (`verify_dataset_integrity`, L88-127).  
  evidence: `2007-liang-chiral-seed-pod-opening.py` L88–127; `datasets/chiral_seed_pod_opening/chiral_seed_pod_opening_reference.json` L1–134
- **Code-to-model traceability: 93** — Module and function docstrings map every equation to Armon et al. (2011) Science Eqs. 1-4 & Figs. 1-4, Efrati et al. (2009) JMPS, Armon et al. (2014) Soft Matter, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_065_chiral_seed_pod_opening.lean` (L1-64, L138-285).  
  evidence: `2007-liang-chiral-seed-pod-opening.py` L1–64
- **Reproducibility package: 94** — 8-point `self_test()` suite (`L1157-1221`) verifying SHA-256 dataset integrity, Gauss saddle incompatibility det(b_0) = -kappa_0^2 < 0 across cutting angles, narrow helicoid (H=0, R=0) and achiral ring (theta=0 deg, 90 deg -> P=0) plus flat (kappa_0=0) negative controls, pitchfork bifurcation correlation, cutting-angle chirality sweep, 4-system scaling, specimen CV segmentation, and two-run bitwise determinism (`np.allclose(..., atol=0.0)`).  
  evidence: `2007-liang-chiral-seed-pod-opening.py` L1157–1221

Remediation:
- Couple the quasi-static non-Euclidean shell energy minimizer to transient elastodynamic fracture propagation along the pod suture line to model ballistic seed ejection velocities.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Armon et al. (2011) Science / Efrati et al. (2009) JMPS Incompatible Shell Elasticity & Chiral Seed Pod Opening
- **spec_fingerprint** (semantic model + inputs): `48029999be49f27cab48fdb5ee4f4512dc98d25ab053a5439aec7a9ce898cd3a`
- **input_dataset_sha256**: `datasets/chiral_seed_pod_opening/bauhinia_chiral_seed_pod_specimen.jpg` (`047f357961bc58c5…`), `datasets/chiral_seed_pod_opening/chiral_seed_pod_opening_reference.json` (`4b1fa5df06327c65…`), `datasets/chiral_seed_pod_opening/paper_citations.bib` (`b3a693ac6a538b04…`), `datasets/chiral_seed_pod_opening/scout_manifest.json` (`05711e5e7df6cb17…`)
- `docs/media/chiral_seed_pod_opening_comparison.png` — artifact_sha256 `1ecbc4345db11d57…`, 1,024,373 B, 2775×1680 px

| measurement | value | reference | method |
|---|---:|---|---|
| Armon et al. (2011) critical Föppl-von Kármán pitchfork width w_tilde_c (nu=0.30) | 2.1609 dimensionless | 2.1609 | Exact non-Euclidean shell pitchfork threshold w_tilde_c = (16*(1+nu)/sqrt(1-nu^2))^(1/4) *(narrow helicoid w_tilde=1.0 gives H/kappa_0 = 0.0000)* |
| Incompatible saddle Gaussian target curvature det(b_0) = -kappa_0^2 (kappa_0=0.42 cm^-1) | -0.1764 cm^-2 | -0.1764 | Determinant of rotated +-45 deg sclerenchyma target curvature tensor b_0(theta) vs flat reference metric K(a_0)=0 *(Tr(b_0) = 0.0000 cm^-1, bilayer kappa_0 = 0.4200 cm^-1)* |
| Supercritical pitchfork mean curvature H*(w_tilde)/kappa_0 correlation across w_tilde in [0.5, 10] | 1.0 Pearson r | 1.0 | L-BFGS-B non-Euclidean plate relaxation vs benchmark across 11 widths (p=1.59e-23, RMSE=0.0006) *(wide asymptote H(w_tilde=10)/kappa_0 = 0.3496 vs (1-nu)/2 = 0.3500)* |
| Dimensionless helix radius R(w_tilde)*kappa_0 transition correlation across w_tilde_c | 1.0 Pearson r | 1.0 | L-BFGS-B helix radius R(w_tilde)*kappa_0 across helicoid-to-ribbon transition (p=1.53e-27, RMSE=0.0009) *(narrow helicoid (w_tilde <= w_tilde_c) R*kappa_0 = 0.0000)* |
| Dimensionless helical pitch P(w_tilde)*kappa_0/(2*pi) transition correlation across w_tilde | 1.0 Pearson r | 1.0 | L-BFGS-B helical pitch P(w_tilde)*kappa_0/(2*pi) across 11 dimensionless widths (p=5.76e-22, RMSE=0.0027) *(RMSE_P = 0.0027)* |
| Cutting-angle chirality torsion tau(theta)/kappa_0 correlation across theta in [0 deg, 90 deg] | 1.0 Pearson r | 1.0 | Geodesic torsion tau(theta)/kappa_0 = (D/kappa_0)*sin(2*theta) across 7 cutting angles (p=3.41e-24, RMSE=0.0000) *(maximal twist at theta=45 deg: tau/kappa_0 = 0.3500)* |
| Achiral ring negative control helical pitch P at cutting angle theta = 0 deg | 0.0 cm | 0.0 | Helical pitch P(theta=0 deg) when principal intrinsic curvature aligns with strip axis (tau = D*sin(0) = 0) *(theta=90 deg pitch = 0.000000 cm, ring radius R(0 deg) = 3.4770 cm)* |
| 4-system legume & biomimetic latex helical pitch P correlation (Bauhinia, Erythrina, Latex) | 0.9994 Pearson r | 1.0 | Predicted vs empirical helical pitch P across 4 experimental systems (p=6.24e-04, RMSE=0.1771 cm) *(4-system helix radius Pearson r = 1.0000, RMSE = 0.0054 cm)* |
| Bauhinia variegata specimen segmented helical lobe half-pitch CV (bauhinia_chiral_seed_pod_specimen.jpg) | 0.2405 CV (std/mean) | 0.24 | Inter-lobe half-pitch spacing dispersion across n=8 segmented helical twist lobes on both dehisced valves *(mean half-pitch = 169.83 ± 40.85 px)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2007-liang-chiral-seed-pod-opening.py` after completing Step 12 closed-loop remediation in `2026-10-19-chiral-seed-pod-opening-loop`.
