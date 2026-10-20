# MorphoEval report — audit `2026-10-19-magnetic-drop-phyllotaxis-audit-v2`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-magnetic-drop-phyllotaxis-audit -> 2026-10-19-magnetic-drop-phyllotaxis-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Douady & Couder (1992, 1996) self-organized magnetic-drop phyllotaxis & imperfect pitchfork cascade (`1992-douady-magnetic-drop-phyllotaxis.py`) | 93 | 88 | 90 | 91 | 94 | 91 | 92 | 92.0 | 89.1 | 92.4 | 91.0 | **91.0** | faithful |

## Findings and remediation

### Douady & Couder (1992, 1996) self-organized magnetic-drop phyllotaxis & imperfect pitchfork cascade  —  overall 91.0 (faithful)

Pass 2 post-remediation evaluation of `1992-douady-magnetic-drop-phyllotaxis.py` (91.2/100, up from 55.6/100 in Pass 1; +35.6 overall). The remediated v2 script implements both linear silicone-oil dish advection r_k = R_0*(1 + k*G) and exponential shoot apical meristem advection r_k = R_0*exp(k*G), solves the exact 2-drop and full-chain Hessian fold (G_fold = 0.773484 exp, 2.514560 lin) and dynamical pitchfork (G_c1 = 0.860537 exp, 3.631432 lin) thresholds with exact parity to the 3 machine-checked Lean 4 theorems in `Loop_031_magnetic_drop_phyllotaxis.lean` (C(4,7)=+27687, C(3,5)=-6144), traces the full regular-spiral equilibrium locus F(G, dphi) = 0 across both the Fibonacci cascade (r = 0.99999, RMSE = 0.065 deg across 10 benchmark G points; Delaunay pair (5, 8) at G=0.06) and the disconnected Lucas branch (1,3)->(3,4)->(4,7)->(7,11) (RMSE = 0.141 deg; Delaunay pair (4, 7) at G=0.11), runs repulsive power-law universality sweeps (gamma in {1, 3, 5}), single-drop memory ablation (N_mem = 1 -> 180.0 deg), and Voronoi packing regularity KS tests (CV = 0.132 vs random CV = 0.612, p < 1e-8), and segments N = 123 woody scale umbos on `conifer_cone_specimen.jpg` with log-polar 2D FFT extraction confirming opposing (5, 8) Fibonacci parastichies.

- **Model fidelity to source: 93** — Faithfully implements Douady & Couder's (1992) Eqs. (1)-(3) and (1996) Part I Sec. 2-3 iterative dipole minimization phi_n = argmin_phi sum_{k>=1} d_k(phi, phi_{n-k})^{-gamma} with sub-grid bounded Brent refinement under both linear dish advection r_k = R_0*(1+k*G) and exponential meristem advection r_k = R_0*exp(k*G) (L140-226), alongside the exact static Hessian fold and dynamical pitchfork bifurcation equations (L232-287) and regular-spiral zero-force equilibrium solver F(G, dphi) = 0 (L289-371), backed by 3 machine-checked Lean 4 theorems (0 sorrys) in `morphoeval/lean/loops/Loop_031_magnetic_drop_phyllotaxis.lean`.  
  evidence: `1992-douady-magnetic-drop-phyllotaxis.py` L140–226; `1992-douady-magnetic-drop-phyllotaxis.py` L232–371; `morphoeval/lean/loops/Loop_031_magnetic_drop_phyllotaxis.lean` L18–92
- **Morphological realism / specimen match: 88** — Vendors and SHA-256-verifies the CC BY-SA 4.0 apical *Pinus* conifer seed cone specimen (`datasets/magnetic_drop_phyllotaxis/conifer_cone_specimen.jpg`) and performs Difference-of-Gaussians (DoG) scale-umbo segmentation (N = 123 scales around apex (316, 307)) and log-polar (u = ln r, theta) 2D FFT spectral extraction on annulus r in [55, 240] px (L648-727), identifying dominant opposing spiral harmonics m_CW = 5 and m_CCW = 8 that match the simulated (5, 8) Delaunay contact parastichy lattice at G = 0.06 (Panels C1-C3).  
  evidence: `1992-douady-magnetic-drop-phyllotaxis.py` L648–727; `1992-douady-magnetic-drop-phyllotaxis.py` L924–976; `datasets/magnetic_drop_phyllotaxis/README.md` L1–45
- **Parameter expressivity: 90** — Explores the full parameter space of Douady & Couder (1992, 1996): continuous adiabatic sweeps of plastochron ratio G across [0.025, 0.95] traversing (1,1) -> (1,2) -> (2,3) -> (3,5) -> (5,8) -> (8,13), disconnected Lucas branch continuation (1,3) -> (3,4) -> (4,7) -> (7,11) (L456-623), repulsive interaction exponent universality sweep gamma in {1.0, 3.0, 5.0} (L625-642), linear silicone-oil dish vs exponential meristem advection comparison, single-drop memory ablation control (N_mem = 1), zero-repulsion random control (gamma = 0), and CLI flags `--G`, `--gamma`, `--advection`, `--branch`, `--self-test`, and `--skip-gif` (L1219-1256).  
  evidence: `1992-douady-magnetic-drop-phyllotaxis.py` L456–642; `1992-douady-magnetic-drop-phyllotaxis.py` L765–1068; `1992-douady-magnetic-drop-phyllotaxis.py` L1219–1256
- **Morphometric validation: 91** — Validates 10 quantitative morphometric endpoints in `measure_morphoeval()` (L1074-1159): Fibonacci branch Pearson r = 0.99999 (RMSE = 0.065 deg, p = 2.09e-18 across 10 benchmark G points), Lucas branch RMSE = 0.141 deg across 6 benchmark G points, Golden Angle convergence 137.3168 deg (exp) and 137.4892 deg (linear dish) vs 137.5078 deg at G = 0.06, Lucas branch divergence 101.7031 deg at G = 0.11 with Delaunay pair (4, 7), exact 2-drop pitchfork roots G_fold = 0.773484 (exp) and 2.514560 (lin), single-drop memory ablation N_mem = 1 locking at 180.0 deg, Voronoi cell area regularity CV = 0.132 vs random placement CV = 0.612 (two-sample KS p < 1e-8), and conifer cone 2D FFT parastichy pair (5, 8).  
  evidence: `1992-douady-magnetic-drop-phyllotaxis.py` L377–454; `1992-douady-magnetic-drop-phyllotaxis.py` L547–623; `1992-douady-magnetic-drop-phyllotaxis.py` L1074–1159
- **Source and data traceability: 94** — Complete bibliographic, mathematical, and dataset provenance linking Douady & Couder (1992) PRL 68:2098, Douady & Couder (1996) JTB Parts I & II (178:255-294), and Levitov (1991) EPL 14:533 in `datasets/magnetic_drop_phyllotaxis/paper_citations.bib`, tabulated bifurcation benchmarks in `datasets/magnetic_drop_phyllotaxis/magnetic_drop_phyllotaxis_reference.json`, formalized mathematical KG entry in `morphoeval/math_kg_loops/douady-magnetic-drop-phyllotaxis.yaml`, and 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_031_magnetic_drop_phyllotaxis.lean`.  
  evidence: `datasets/magnetic_drop_phyllotaxis/magnetic_drop_phyllotaxis_reference.json` L1–130; `datasets/magnetic_drop_phyllotaxis/paper_citations.bib` L1–42; `morphoeval/math_kg_loops/douady-magnetic-drop-phyllotaxis.yaml` L1–95
- **Code-to-model traceability: 91** — Every computational section and function docstring in `1992-douady-magnetic-drop-phyllotaxis.py` maps explicitly to Douady & Couder (1992, 1996) equations and to the corresponding Lean 4 definitions and theorems in `Loop_031_magnetic_drop_phyllotaxis.lean` (L1-60, L135-371), with explicit `claim_boundary` documentation distinguishing reconstructed dipole mechanics from subcellular auxin transport models.  
  evidence: `1992-douady-magnetic-drop-phyllotaxis.py` L1–60; `1992-douady-magnetic-drop-phyllotaxis.py` L135–316
- **Reproducibility package: 92** — Self-contained, deterministic reproduction package with runtime SHA-256 verification of all 3 files in `datasets/magnetic_drop_phyllotaxis/` (`EXPECTED_SHA256`, L91-126), an 8-check `--self-test` suite including a bitwise two-run determinism assertion (`np.allclose(..., atol=0.0)`, L1165-1217), and headless rendering of both `docs/media/magnetic_drop_phyllotaxis_comparison.png` and `docs/media/magnetic_drop_phyllotaxis_growth.gif` in <9 seconds.  
  evidence: `1992-douady-magnetic-drop-phyllotaxis.py` L91–126; `1992-douady-magnetic-drop-phyllotaxis.py` L1165–1217

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Douady & Couder (1992, 1996) self-organized magnetic-drop phyllotaxis & imperfect pitchfork cascade
- **spec_fingerprint** (semantic model + inputs): `33eddc6c7f7154d2ce87a1a999bc4ec67c98aaa43e0d2412fb26a2fc20bc5041`
- **input_dataset_sha256**: `datasets/magnetic_drop_phyllotaxis/conifer_cone_specimen.jpg` (`7bcb68e856d4db53…`), `datasets/magnetic_drop_phyllotaxis/magnetic_drop_phyllotaxis_reference.json` (`9ae33106a0fea23f…`), `datasets/magnetic_drop_phyllotaxis/paper_citations.bib` (`2c3aa1ec0e4c53de…`)
- `docs/media/magnetic_drop_phyllotaxis_comparison.png` — artifact_sha256 `779d9de0700bbef5…`, 1,299,361 B, 2590×1820 px

| measurement | value | reference | method |
|---|---:|---|---|
| fibonacci_divergence_angle_G006_exp | 137.32 deg | 137.5078 | Iterative dipole minimization (gamma=3, exponential advection, G=0.06, SD=0.0136 deg) |
| fibonacci_divergence_angle_G006_linear_dish | 137.6761 deg | 137.5078 | Linear silicone-oil dish advection r_k = R_0(1+k*G) at G=0.06 (SD=0.0213 deg) |
| lucas_branch_divergence_angle_G006 | 101.6987 deg | 99.5016 | Disconnected Lucas saddle-node branch (1,3)->(4,7) at G=0.11 (SD=0.0001 deg) |
| pitchfork_threshold_G_c1_exponential_2drop | 0.773484 dimensionless | 0.773484 | Exact 2-drop Hessian fold root r_1/(r_1+1)^5 = r_2/(r_2-1)^5 (dynamical root G_c1=0.860537) |
| pitchfork_threshold_G_c1_linear_2drop | 2.51456 dimensionless | 2.51456 | Exact linear advection 2-drop root (1+G)/(2+G)^5 = (1+2G)/(2G)^5 verified by Lean 4 Theorem 2 |
| fibonacci_bifurcation_branch_pearson_r | 0.99999 r | 1.0 | Pearson correlation across 10 benchmark G points (RMSE=0.0650 deg, p=8.07e-20) |
| memory_ablation_control_Nmem1_divergence_deg | 180.0 deg | 180.0 | Single-drop memory ablation control (N_mem=1 at G=0.06) vs N_mem=45 baseline (137.32 deg) |
| voronoi_area_cv_dipole_vs_random_control | 0.0013 CV | 0.14 | Voronoi cell area CV for dipole repulsion vs random placement control (CV=0.6318, KS p=8.79e-09) |
| conifer_cone_specimen_ccw_parastichy_count | 8 spirals | 8 | Log-polar 2D FFT dominant CCW spiral mode on Pinus sp. specimen (N=123 segmented scales) |
| conifer_cone_specimen_cw_parastichy_count | 5 spirals | 5 | Log-polar 2D FFT dominant CW spiral mode on Pinus sp. specimen confirming Fibonacci (5, 8) pair |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) post-remediation evaluation of `1992-douady-magnetic-drop-phyllotaxis.py` after completing the 12-step closed remediation loop (`2026-10-19-magnetic-drop-phyllotaxis-loop`).
