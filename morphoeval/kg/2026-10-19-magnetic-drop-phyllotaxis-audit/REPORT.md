# MorphoEval report — audit `2026-10-19-magnetic-drop-phyllotaxis-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Douady & Couder (1992, 1996) ferrofluid magnetic-drop iterative repulsive phyllotaxis (`1992-douady-magnetic-drop-phyllotaxis.py`) | 58 | 22 | 46 | 44 | 68 | 62 | 68 | 54.0 | 30.2 | 66.4 | 48.2 | **48.2** | partially faithful |

## Findings and remediation

### Douady & Couder (1992, 1996) ferrofluid magnetic-drop iterative repulsive phyllotaxis  —  overall 48.2 (partially faithful)

Pass 1 (v1) baseline evaluation of `1992-douady-magnetic-drop-phyllotaxis.py` (51.9/100). The v1 script implements iterative dipole repulsion minimization E(phi) = sum_k d_k^{-3} under exponential radial advection r_k = exp(k*G) and simulates an adiabatic Fibonacci ramp down to G=0.025 plus 4 fixed-G dishes. However, v1 omits Douady & Couder's (1992) linear dish advection r_k = R_0(1 + k*G), omits the analytical pitchfork bifurcation threshold G_{c,1} and exact regular-spiral zero-force equilibrium branches F(G, Delta_phi)=0, omits the disconnected Lucas saddle-node branch (1,3)->(3,4)->(4,7) (Delta_phi -> 99.50 deg), lacks negative/ablation controls (single-drop memory N_mem=1 or random placement), and does not segment or analyze the vendored conifer cone specimen image `datasets/magnetic_drop_phyllotaxis/conifer_cone_specimen.jpg`.

- **Model fidelity to source: 58** — Implements the iterative dipole repulsion potential E(phi) = sum_k d_k(phi, phi_{n-k})^{-3} on the meristem rim (L38-88) and adiabatic Fibonacci ramp (L142-171). However, v1 only implements exponential advection r_k = exp(k*G) without Douady & Couder's (1992) physical silicone-oil dish linear advection r_k = 1 + k*G, does not solve the analytical opposition-to-spiral pitchfork bifurcation condition, does not compute the exact regular-spiral zero-force curves F(G, Delta_phi) = 0, and omits the disconnected Lucas branch (1,3)->(3,4)->(4,7).  
  evidence: `1992-douady-magnetic-drop-phyllotaxis.py` L38–88; `1992-douady-magnetic-drop-phyllotaxis.py` L142–171
- **Morphological realism / specimen match: 22** — Although `datasets/magnetic_drop_phyllotaxis/conifer_cone_specimen.jpg` is vendored in the dataset directory, the v1 script never loads, displays, or segments the conifer cone photograph (`docs/media/magnetic_drop_phyllotaxis_comparison.png` shows only 6 synthetic plots).  
  evidence: `1992-douady-magnetic-drop-phyllotaxis.py` L174–246; `docs/media/magnetic_drop_phyllotaxis_comparison.png` L0–0
- **Parameter expressivity: 46** — Simulates 4 fixed-G values (0.92, 0.30, 0.14, 0.06) and an adiabatic ramp, but lacks a universality sweep across dipole interaction exponents gamma in {1, 3, 5}, lacks the Lucas branch initial-condition basin, lacks negative/ablation controls (such as N_mem=1 memory truncation or zero-repulsion random placement), and exposes no CLI parameters for G, gamma, or advection mode (L302-313).  
  evidence: `1992-douady-magnetic-drop-phyllotaxis.py` L177–182; `1992-douady-magnetic-drop-phyllotaxis.py` L302–313
- **Morphometric validation: 44** — `measure_morphoeval()` (L275-291) returns only 2 scalar divergence angles (at G=0.06 and G=0.92) without Delaunay contact parastichy pair extraction (m, n), without comparison to `magnetic_drop_phyllotaxis_reference.json` benchmark curves, without goodness-of-fit statistics (Pearson r, RMSE, KS test), and without empirical conifer cone parastichy measurements.  
  evidence: `1992-douady-magnetic-drop-phyllotaxis.py` L275–291
- **Source and data traceability: 68** — `datasets/magnetic_drop_phyllotaxis/` contains `magnetic_drop_phyllotaxis_reference.json`, `conifer_cone_specimen.jpg`, and `paper_citations.bib`, but the v1 script does not load or verify the SHA-256 digests of the reference JSON or specimen image at runtime.  
  evidence: `1992-douady-magnetic-drop-phyllotaxis.py` L29–36
- **Code-to-model traceability: 62** — Module docstring cites Douady & Couder (1992), but individual functions lack equation and figure mappings to Douady & Couder (1992) Eqs. 1-3 & Figs. 1-4, Douady & Couder (1996) Part I Sec. 2-3, and `Loop_031_magnetic_drop_phyllotaxis.lean`.  
  evidence: `1992-douady-magnetic-drop-phyllotaxis.py` L38–140
- **Reproducibility package: 68** — Deterministic execution and `--self-test` flag are present (L294-300), but `run_self_test()` checks only 2 loose assertions without verifying pitchfork roots, Lucas branch convergence, SHA-256 dataset locks, or bitwise repeatability.  
  evidence: `1992-douady-magnetic-drop-phyllotaxis.py` L294–300

Remediation:
- Implement both linear dish advection r_k = R_0(1 + k*G) (Douady & Couder 1992 PRL Eq. 1) and exponential meristem advection r_k = R_0*exp(k*G), solve the exact analytical 2-drop and full-chain pitchfork bifurcation thresholds G_{c,1}, and verify the integer curvature discriminant C(4,7)=+27687 vs C(3,5)=-6144 proven in `Loop_031_magnetic_drop_phyllotaxis.lean`.
- Compute the exact regular-spiral zero-force equilibrium branches F(G, Delta_phi) = 0 and Hessian stability S(G, Delta_phi) > 0 across the (G, Delta_phi) bifurcation diagram, and simulate both the main Fibonacci cascade (1,1)->(1,2)->(2,3)->(3,5)->(5,8)->(8,13) (Delta_phi -> 137.5078 deg) and the disconnected Lucas saddle-node branch (1,3)->(3,4)->(4,7)->(7,11) (Delta_phi -> 99.5016 deg).
- Add a dipole-exponent universality sweep across gamma in {1.0, 3.0, 5.0}, a single-drop memory ablation control (N_mem = 1, which eliminates the destabilizing k=2 repulsion and locks the system at Delta_phi = 180 deg for all G), and a zero-repulsion random placement control.
- Load and SHA-256-verify `datasets/magnetic_drop_phyllotaxis/conifer_cone_specimen.jpg` and `magnetic_drop_phyllotaxis_reference.json`, segment the conifer cone scale umbos via DoG filtering, extract opposing Fibonacci (5, 8) parastichy wavevectors via log-polar (ln r, theta) 2D FFT, and expand `measure_morphoeval()` with Delaunay parastichy counts, Pearson r, RMSE, and KS goodness-of-fit statistics.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Douady & Couder (1992, 1996) ferrofluid magnetic-drop iterative repulsive phyllotaxis
- **spec_fingerprint** (semantic model + inputs): `33eddc6c7f7154d2ce87a1a999bc4ec67c98aaa43e0d2412fb26a2fc20bc5041`
- **input_dataset_sha256**: `datasets/magnetic_drop_phyllotaxis/conifer_cone_specimen.jpg` (`7bcb68e856d4db53…`), `datasets/magnetic_drop_phyllotaxis/magnetic_drop_phyllotaxis_reference.json` (`9ae33106a0fea23f…`), `datasets/magnetic_drop_phyllotaxis/paper_citations.bib` (`2c3aa1ec0e4c53de…`)
- `docs/media/magnetic_drop_phyllotaxis_comparison.png` — artifact_sha256 `ef80ca944c2794c4…`, 239,042 B, 1960×1232 px

| measurement | value | reference | method |
|---|---:|---|---|
| fibonacci_divergence_angle_G006 | 137.3193 deg | 137.5078 | Iterative dipole energy minimization at G=0.06 (last 20 drops mean Δφ) |
| distichous_divergence_angle_G092 | 180.0 deg | 180.0 | Iterative dipole energy minimization at G=0.92 above pitchfork threshold G_c1=0.859 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `1992-douady-magnetic-drop-phyllotaxis.py` prior to Step 12 remediation in `2026-10-19-magnetic-drop-phyllotaxis-loop`.
