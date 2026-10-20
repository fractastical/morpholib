# MorphoEval report — audit `2026-10-19-mammalian-coat-markings-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Murray (1981) Thomas substrate-inhibition reaction-diffusion pre-pattern & conical tail spot-to-ring transition (`1981-murray-mammalian-coat-markings.py`) | 52 | 28 | 44 | 42 | 62 | 58 | 68 | 49.3 | 33.2 | 63.3 | 47.1 | **47.1** | partially faithful |

## Findings and remediation

### Murray (1981) Thomas substrate-inhibition reaction-diffusion pre-pattern & conical tail spot-to-ring transition  —  overall 47.1 (partially faithful)

Pass 1 (v1) baseline evaluation of `1981-murray-mammalian-coat-markings.py` (overall weighted score: 49.4 / 100). The v1 script solves the homogeneous steady state (u0, v0) = (9.9338, 9.2892) and linear Turing instability band [k_-^2, k_+^2] = [0.5236, 0.9648] of Thomas (1975) / Murray (1981) kinetics, but approximates pattern formation via a heuristic cosine superposition rather than integrating the nonlinear Thomas reaction-diffusion PDE with zero-flux Neumann boundaries on a 2D flank and curvilinear tapered conical tail; omits negative/ablation controls (d = 1 equal diffusion and K = 0 non-inhibitory Michaelis-Menten ablation); does not load or segment the vendored Panthera pardus specimen photograph; and exposes only 2 measurements without dispersion metrics or CLI parameter flags.

- **Model fidelity to source: 52** — Computes the exact Thomas (1975) / Murray (1981) homogeneous steady state (u0, v0) = (9.9338, 9.2892), Jacobian J0, and unstable wavenumber interval gamma*k^2 in [0.5236, 0.9648] (L24-62), but `simulate_cylinder_eigenmodes` (L65-91) synthesizes patterns using a Gaussian-weighted linear cosine sum rather than numerically integrating the nonlinear Thomas reaction-diffusion PDE du/dt = gamma*f(u,v) + Lap(u), dv/dt = gamma*g(u,v) + d*Lap(v) on a tapered conical cylinder with curvilinear Laplace-Beltrami operator and Neumann boundary conditions, and omits Liu, Liaw & Maini (2006) two-stage spot-to-rosette maturation.  
  evidence: `1981-murray-mammalian-coat-markings.py` L24–62; `1981-murray-mammalian-coat-markings.py` L65–91
- **Morphological realism / specimen match: 28** — Although `datasets/mammalian_coat_markings/panthera_pardus_specimen.jpg` is vendored in the repository, the v1 script never loads, segments, or displays the biological specimen photograph (`docs/media/mammalian_coat_markings_comparison.png` shows only 2 synthetic heatmaps), and performs no quantitative comparison against leopard flank rosette spacing or tail banding anisotropy.  
  evidence: `1981-murray-mammalian-coat-markings.py` L94–118; `docs/media/mammalian_coat_markings_comparison.png` L0–0
- **Parameter expressivity: 44** — Simulates only 2 static gamma values (8.0 and 25.0) in the comparison figure without Murray's full 5-regime scale morphospace sweep (gamma from subcritical 0.4 uniform mouse coat to 120.0 large mammal high-mode crowding), without equal-diffusion (d=1.0) or non-inhibitory (K=0.0) negative controls, and without CLI flags for a, b, alpha, rho, K, d, or gamma (L133-141).  
  evidence: `1981-murray-mammalian-coat-markings.py` L94–110; `1981-murray-mammalian-coat-markings.py` L133–141
- **Morphometric validation: 42** — `measure_morphoeval()` (L112-130) returns only 2 analytical scalar quantities (u0 and gamma*k_c^2) without measuring spatial wavelengths, proximal-vs-distal tail mode anisotropy, negative-control amplitude suppression, or biological specimen rosette spacing CV and nearest-neighbor statistics.  
  evidence: `1981-murray-mammalian-coat-markings.py` L112–130
- **Source and data traceability: 62** — Header cites Murray (1981) DOI `10.1016/0022-5193(81)90334-9` and `datasets/mammalian_coat_markings/` is present on disk, but the v1 script does not load `mammalian_coat_markings_reference.json` or verify SHA-256 digests at runtime.  
  evidence: `1981-murray-mammalian-coat-markings.py` L1–8
- **Code-to-model traceability: 58** — Basic docstrings are present for steady-state root finding (L24-62), but functions lack explicit equation, section, and figure mappings to Murray (1981) Eqs. (4)-(16), Section 4 Eqs. (26)-(32) Figs. 9-12, Liu et al. (2006) two-stage rosette equations, and the 3 Lean 4 theorems in `Loop_003_mammalian_coat_markings.lean`.  
  evidence: `1981-murray-mammalian-coat-markings.py` L24–91
- **Reproducibility package: 68** — Includes deterministic `np.random.default_rng(1981)` and `--self-test` (L123-130), but `self_test()` checks only 3 basic inequalities without verifying dataset SHA-256 locks, PDE convergence, negative-control quenching, specimen segmentation, or bitwise determinism across two runs.  
  evidence: `1981-murray-mammalian-coat-markings.py` L123–130

Remediation:
- Replace the linear cosine superposition with a genuine 2D nonlinear Thomas (1975) / Murray (1981) reaction-diffusion PDE solver (with zero-flux Neumann boundaries) on both a planar mammalian flank domain (including Liu, Liaw & Maini 2006 two-stage spot-to-rosette maturation) and a curvilinear tapered conical cylinder (Laplace-Beltrami operator (1/r) d/dx(r du/dx) + (1/r^2) d^2u/dtheta^2).
- Implement dual negative/ablation controls around the homogeneous steady state: (1) equal-diffusion control d = Dv/Du = 1.0 < d_c = 9.6292, and (2) non-inhibitory Michaelis-Menten ablation K = 0.0 (testing Lean Theorem 2 where K*u0^2 = 0 < 1 prevents self-activation).
- Load and SHA-256-verify `datasets/mammalian_coat_markings/mammalian_coat_markings_reference.json`, `panthera_pardus_specimen.jpg`, and `scout_manifest.json` at runtime; segment the biological leopard flank rosettes and distal tail rings directly from `panthera_pardus_specimen.jpg`; and compare biological vs simulated inter-spot wavelength CV, Voronoi polygonality, and proximal-to-distal tail anisotropy.
- Sweep all 5 scale regimes gamma in {0.4, 3.5, 8.0, 25.0, 120.0}, expose full CLI parameters (--a, --b, --alpha, --rho, --K, --d, --gamma, --steps, --seed), expand `measure_morphoeval()` to 8 reference-backed morphometric measurements with uncertainty/control deltas, and upgrade `docs/media/mammalian_coat_markings_comparison.png` and `docs/media/mammalian_coat_markings_growth.gif`.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Murray (1981) Thomas substrate-inhibition reaction-diffusion pre-pattern & conical tail spot-to-ring transition
- **spec_fingerprint** (semantic model + inputs): `525df88cbbed5362e036470fa7d127c713eb89280f82aab38da743ea194e4397`
- **input_dataset_sha256**: `datasets/mammalian_coat_markings/mammalian_coat_markings_reference.json` (`6d8170232f54ccce…`), `datasets/mammalian_coat_markings/panthera_pardus_specimen.jpg` (`86533a5dcf9cfee6…`), `datasets/mammalian_coat_markings/scout_manifest.json` (`28eabc9cf247b03e…`)
- `docs/media/mammalian_coat_markings_comparison.png` — artifact_sha256 `14cc11c7b1fa12eb…`, 39,199 B, 1540×630 px

| measurement | value | reference | method |
|---|---:|---|---|
| Thomas steady-state substrate concentration u0 | 9.9338 dimensionless | 9.9338 | brentq root of a - u0 - rho*u0*v0/(1 + u0 + K*u0^2) = 0 |
| Turing critical wavenumber squared gamma*k_c^2 | 0.7442 dimensionless | 0.7442 | (d * f_u + g_v) / (2 * d) at d = 10.0 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1981-murray-mammalian-coat-markings.py` prior to Step 12 remediation in `2026-10-19-mammalian-coat-markings-loop`.
