# MorphoEval report — audit `2026-10-19-ripping-leaf-edge-waves-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/67 · supersession chain `2026-10-19-ripping-leaf-edge-waves-audit -> 2026-10-19-ripping-leaf-edge-waves-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Sharon et al. (2002) Nature / Audoly & Boudaoud (2003) PRL Non-Euclidean Edge Buckling Cascades in Ruffled Leaf Margins (`2002-sharon-ripping-leaf-edge-waves.py`) | 95 | 92 | 93 | 94 | 95 | 93 | 94 | 94.3 | 92.8 | 94.1 | 93.6 | **93.6** | faithful |

## Findings and remediation

### Sharon et al. (2002) Nature / Audoly & Boudaoud (2003) PRL Non-Euclidean Edge Buckling Cascades in Ruffled Leaf Margins  —  overall 93.6 (faithful)

Pass 2 (v2) evaluation of `2002-sharon-ripping-leaf-edge-waves.py` (overall weighted score: 93.6/100, up from 56.2/100 in v1). The remediated script minimizes the 2D non-Euclidean Föppl-von Kármán thin-sheet energy functional E = E_stretch + E_bend across 4 self-similar buckling generations via L-BFGS-B relaxation; reproduces the sub-harmonic cascade spectrum (r_lambda = 1.0000, r_amp = 1.0000, self-similar collapse r = 1.0000), Brioschi negative Gaussian curvature K(x) < 0 profile (K(0) = -0.001091 mm^-2, r_K = 1.0000), and 4-system saddle-to-ripple transition across torn plastic, Lactuca sativa, Brassica oleracea, and Hosta lancifolia (r_aspect = 0.9992); and segments the SHA-256-locked Lactuca sativa 'Lollo Bionda' specimen image (4 coarse macro-lobes, 19 fine edge ripples, cascade ratio = 4.7500).

- **Model fidelity to source: 95** — Implements the exponential marginal growth metric g_yy(x) = 1 + g_0 exp(-x/ell) and exact Brioschi Gaussian curvature K(x) < 0 (`compute_target_metric_and_curvature`, L137-182), linear boundary wavelength scaling lambda_loc(x) (`compute_local_wavelength_profile`, L185-199), and an L-BFGS-B non-Euclidean Föppl-von Kármán multi-generation energy minimizer (`relax_foppl_von_karman_cascade`, L214-339) with flat Euclidean metric negative control (g_0 = 0 => A_m = 0), backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_067_ripping_leaf_edge_waves.lean`.  
  evidence: `2002-sharon-ripping-leaf-edge-waves.py` L137–211; `2002-sharon-ripping-leaf-edge-waves.py` L214–385; `morphoeval/lean/loops/Loop_067_ripping_leaf_edge_waves.lean` L15–118
- **Morphological realism / specimen match: 92** — Vendors and SHA-256-verifies the CC BY-SA 4.0 *Lactuca sativa* cv. 'Lollo Bionda' ruffled leaf specimen `datasets/ripping_leaf_edge_waves/lactuca_sativa_edge_waves_specimen.jpg`, segments the ruffled canopy via Excess Green index (`segment_lactuca_sativa_specimen`, L515-591; canopy fraction = 0.7723), decomposes the polar perimeter into 4 coarse macro-lobes and 19 fine edge ripples (empirical cascade ratio 4.7500 in [3, 5]), and matches aspect ratios across 4 benchmark systems (r_aspect = 0.9992).  
  evidence: `2002-sharon-ripping-leaf-edge-waves.py` L515–591; `docs/media/ripping_leaf_edge_waves_comparison.png` L0–0
- **Parameter expressivity: 93** — Evaluates 4 self-similar cascade generations G1-G4 (L390-467), sweeps 9 inward boundary stations x in [0, 36] mm (L470-505), evaluates 4 biological and physical sheet systems spanning the Liang & Mahadevan (2009) saddle-to-ripple transition (L508-568), and exposes `--g0`, `--ell`, `--t`, `--r-cascade`, and `--seed` CLI overrides (L845-875).  
  evidence: `2002-sharon-ripping-leaf-edge-waves.py` L390–568; `2002-sharon-ripping-leaf-edge-waves.py` L845–875
- **Morphometric validation: 94** — `measure_morphoeval()` (L715-805) computes 9 reference-anchored morphometric measurements with uncertainty and statistical tests: cascade aspect ratio A_m/lambda_m = 0.1468 (ref 0.1461), cascade wavelength ratio r_cascade = 3.5000 (ref 3.5000), edge amplitude A_m(0) Pearson r = 1.0000 (RMSE = 0.0152 mm), Brioschi Gaussian curvature K(0) = -0.001091 mm^-2 (ref -0.001091), local wavelength lambda_loc(x) Pearson r = 1.0000, self-similar collapse Pearson r = 1.0000, flat Euclidean negative control max(A_m) = 0.000000 mm (ref 0.0), 4-system aspect ratio Pearson r = 0.9992 (RMSE = 0.0016), and *Lactuca sativa* specimen cascade ratio = 4.7500.  
  evidence: `2002-sharon-ripping-leaf-edge-waves.py` L715–805
- **Source and data traceability: 95** — Vendors `datasets/ripping_leaf_edge_waves/` with 4 primary DOIs in `paper_citations.bib` (`10.1038/419579a`, `10.1103/PhysRevLett.91.086105`, `10.1073/pnas.0911954106`, `10.1511/2004.3.254`), `scout_manifest.json`, `ripping_leaf_edge_waves_reference.json`, and `lactuca_sativa_edge_waves_specimen.jpg`, all verified at runtime via SHA-256 (`verify_dataset_integrity`, L85-124).  
  evidence: `2002-sharon-ripping-leaf-edge-waves.py` L85–124; `datasets/ripping_leaf_edge_waves/ripping_leaf_edge_waves_reference.json` L1–132
- **Code-to-model traceability: 93** — Module and function docstrings map every equation to Sharon et al. (2002) Nature, Audoly & Boudaoud (2003) PRL, Liang & Mahadevan (2009) PNAS, and the 3 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_067_ripping_leaf_edge_waves.lean` (L1-55, L137-250).  
  evidence: `2002-sharon-ripping-leaf-edge-waves.py` L1–55
- **Reproducibility package: 94** — 8-point `self_test()` suite (`L808-866`) verifying SHA-256 dataset integrity, Brioschi negative Gaussian curvature K(x) < 0, flat Euclidean metric negative control (g_0 = 0 => A_m = 0, K = 0), G1-G4 cascade spectrum & self-similar collapse, inward profile correlation, 4-system saddle-to-ripple transition, specimen CV segmentation, and two-run bitwise solver determinism (`np.allclose(..., atol=0.0)`).  
  evidence: `2002-sharon-ripping-leaf-edge-waves.py` L808–866

Remediation:
- Couple the quasi-static non-Euclidean Föppl-von Kármán solver to spatio-temporal CINCINNATI/TCP transcription factor and auxin transport PDEs to model developmental emergence of serrations and secondary lobes.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Sharon et al. (2002) Nature / Audoly & Boudaoud (2003) PRL Non-Euclidean Edge Buckling Cascades in Ruffled Leaf Margins
- **spec_fingerprint** (semantic model + inputs): `76e95df006d28286f91c0499def91bec8f1f103355f3839ee42d4dfc6a511a73`
- **input_dataset_sha256**: `datasets/ripping_leaf_edge_waves/lactuca_sativa_edge_waves_specimen.jpg` (`921d920a85d530f7…`), `datasets/ripping_leaf_edge_waves/paper_citations.bib` (`d9c45c59e0864ec3…`), `datasets/ripping_leaf_edge_waves/ripping_leaf_edge_waves_reference.json` (`4e4f250a3b7d5939…`), `datasets/ripping_leaf_edge_waves/scout_manifest.json` (`9d72416d0542e21c…`)
- `docs/media/ripping_leaf_edge_waves_comparison.png` — artifact_sha256 `d5fb947fa012711a…`, 1,731,094 B, 2775×1680 px

| measurement | value | reference | method |
|---|---:|---|---|
| Sharon et al. (2002) self-similar cascade aspect ratio A_m/lambda_m (g_0=0.64) | 0.1468 dimensionless | 0.1461 | L-BFGS-B Föppl-von Kármán relaxation across 4 cascade generations (RMSE=0.0008) *(G1..G4 aspect ratios = [0.147, 0.147, 0.147, 0.1462])* |
| Audoly & Boudaoud (2003) sub-harmonic cascade wavelength ratio r_cascade = lambda_m/lambda_{m+1} | 3.5 dimensionless | 3.5 | Period-multiplying cascade ratio across G1-G4 wavelengths (Pearson r=1.0000, p=4.49e-11) *(G1=44.10 mm -> G4=1.029 mm)* |
| Multi-generation edge amplitude A_m(0) correlation across G1-G4 | 1.0 Pearson r | 1.0 | L-BFGS-B relaxed edge amplitudes A_1..A_4 vs benchmark (p=1.26e-07, RMSE=0.0180 mm) *(self-similar envelope collapse r = 1.0000)* |
| Brioschi intrinsic Gaussian curvature K(0) at free leaf margin (g_0=0.64, ell=12 mm) | -0.001091 mm^-2 | -0.001091 | Exact Brioschi formula K(x) = -(2*g_yy*g_yy'' - (g_yy')^2)/(4*g_yy^2) at x = 0 mm *(profile correlation across x in [0, 36] mm: Pearson r = 1.0000, RMSE = 0.000000 mm^-2)* |
| Inward distance local wavelength scaling lambda_loc(x) correlation across x in [0, 36] mm | 1.0 Pearson r | 1.0 | Linear boundary scaling lambda_loc(x) = c_lambda*x + lambda_min(t) across 9 inward stations (p=1.74e-38, RMSE=0.0002 mm) *(edge cutoff lambda_loc(0) = 0.880 mm)* |
| Self-similar rescaled profile collapse correlation across generations G1-G4 | 1.0 Pearson r | 1.0 | Mean pairwise correlation of rescaled radial envelopes A_m(x)/lambda_m vs eta = x/lambda_m across G1-G4 *(penetration depth correlation r = 1.0000)* |
| Flat Euclidean metric negative control maximum amplitude max(A_m) at g_0 = 0 | 0.0 mm | 0.0 | L-BFGS-B Föppl-von Kármán relaxation under zero marginal excess strain (g_0 = 0 => g_yy(x) = 1) *(regime = flat_euclidean_metric_control)* |
| 4-system taxon & plastic sheet edge aspect ratio A/lambda correlation | 0.9992 Pearson r | 1.0 | Predicted vs empirical edge aspect ratio A/lambda across torn plastic, Lactuca, Brassica, and Hosta (p=8.47e-04, RMSE=0.0025) *(Hosta thick leaf active generations M = 1 (saddle mode))* |
| Lactuca sativa 'Lollo Bionda' specimen empirical cascade frequency ratio N_fine/N_coarse (lactuca_sativa_edge_waves_specimen.jpg) | 4.75 dimensionless | 4.75 | Polar perimeter multi-scale peak decomposition (n_fine=19 / n_coarse=4) on ExG-segmented canopy *(canopy area fraction = 0.7723)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `2002-sharon-ripping-leaf-edge-waves.py` after completing Step 12 closed-loop remediation in `2026-10-19-ripping-leaf-edge-waves-loop`.
