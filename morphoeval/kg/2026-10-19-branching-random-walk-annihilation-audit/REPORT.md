# MorphoEval report — audit `2026-10-19-branching-random-walk-annihilation-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/56

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Hannezo et al. (2017) Cell & Scheele et al. (2017) Nature unifying theory of branching morphogenesis via branching and annihilating random walks (BARW) (`2017-hannezo-branching-random-walk-annihilation.py`) | 56 | 20 | 44 | 46 | 58 | 64 | 68 | 52.0 | 29.8 | 63.6 | 46.6 | **46.6** | partially faithful |

## Findings and remediation

### Hannezo et al. (2017) Cell & Scheele et al. (2017) Nature unifying theory of branching morphogenesis via branching and annihilating random walks (BARW)  —  overall 46.6 (partially faithful)

Pass 1 (v1 baseline) evaluation of `2017-hannezo-branching-random-walk-annihilation.py` (51.8/100). The v1 script computes the mean-field steady-state duct density i_inf = k_b / k_a and characteristic branch length l_b = v_0 / k_b, integrates a 1D mean-field travelling-wave PDE for active tips a(x, t) and inactive ducts i(x, t), and plots an analytic S^(-3/2) curve. However, v1 lacks an off-lattice 2D spatial Branching and Annihilating Random Walk (BARW) ductal tree simulator with explicit persistent tip migration (v_0), Poisson tip bifurcation (k_b), ductal trail deposition (A -> A + I), and steric/TGF-beta tip-duct annihilation within radius R_a (A + I -> I); omits universal collapse across mouse mammary gland, kidney, and pancreas regimes; omits R_a perturbation and R_a = 0 non-annihilating explosion ablation controls; neither loads nor verifies `datasets/branching_random_walk_annihilation/branching_random_walk_annihilation_reference.json`; and performs zero computer-vision segmentation on `datasets/branching_random_walk_annihilation/mouse_branched_ductal_network_specimen.jpg`.

- **Model fidelity to source: 56** — Integrates the 1D mean-field BARW reaction-diffusion PDE for active tips a(x, t) and inactive ducts i(x, t) (L51-89) and computes basic algebraic ratios (L34-48). However, `build_simple_galton_watson_tree` (L92-106) is a synthetic exponential curve rather than an off-lattice 2D spatial BARW simulator with persistent active tips, Poisson bifurcation (A -> 2A), trail deposition (A -> A + I), and tip-duct annihilation within radius R_a (A + I -> I). Multi-organ universality (mammary, kidney, pancreas) and R_a = 0 ablation controls are absent.  
  evidence: `2017-hannezo-branching-random-walk-annihilation.py` L34–89; `2017-hannezo-branching-random-walk-annihilation.py` L92–106
- **Morphological realism / specimen match: 20** — Although `datasets/branching_random_walk_annihilation/mouse_branched_ductal_network_specimen.jpg` (whole-mount fluorescence micrograph of mouse branched glandular epithelium with cyan/green active distal buds A and blue/magenta differentiated ducts I) is vendored in the dataset directory, the v1 script never loads, segments, or displays the specimen image or extracts active tip counts, branch nodes, or topological q_branch.  
  evidence: `2017-hannezo-branching-random-walk-annihilation.py` L109–169; `docs/media/branching_random_walk_annihilation_comparison.png` L0–0
- **Parameter expressivity: 44** — Hardcodes a single parameter set (v_0=1.0, k_b=0.12, k_a=0.42, D=0.35, R_a=3.5, L27-31) without sweeping annihilation radius R_a, branching rate k_b, or organ-specific dimensionless ratio R_a / l_b across mammary gland, kidney, and pancreas regimes, and exposes only `--self-test` and `--skip-gif` on the CLI (L189-193).  
  evidence: `2017-hannezo-branching-random-walk-annihilation.py` L27–31; `2017-hannezo-branching-random-walk-annihilation.py` L189–198
- **Morphometric validation: 46** — `measure_morphoeval()` (L172-179) reports only 3 scalar numbers (l_branch_wt, i_homeostatic_wt, q_branch_baseline) without statistical dispersion metrics (RMSE, Pearson r, CV), without comparison to Hannezo et al. (2017) generational branching probabilities across mammary gland, kidney, and pancreas, and without perturbation deltas for TGF-beta / R_a variations or R_a = 0 non-annihilating explosion ablation.  
  evidence: `2017-hannezo-branching-random-walk-annihilation.py` L172–179
- **Source and data traceability: 58** — Cites Hannezo et al. (2017) and Scheele et al. (2017) in the module docstring (L5-7), but v1 does not load `datasets/branching_random_walk_annihilation/branching_random_walk_annihilation_reference.json` at runtime or verify SHA-256 cryptographic digests of the reference JSON and mouse branched ductal network fluorescence specimen.  
  evidence: `2017-hannezo-branching-random-walk-annihilation.py` L4–13
- **Code-to-model traceability: 64** — Functions have brief docstrings (L34-106), but do not map code blocks to numbered equations in Hannezo et al. (2017) STAR Methods Eqs. (1)-(10), Scheele et al. (2017) Nature, Silberstein & Daniel (1987) Science, or the 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_056_branching_random_walk_annihilation.lean`.  
  evidence: `2017-hannezo-branching-random-walk-annihilation.py` L34–106
- **Reproducibility package: 68** — Provides `--self-test` and `--skip-gif` flags and deterministic output rendering (L182-198), but `self_test()` checks only 3 algebraic assertions without verifying JSON dataset parity, 2D spatial BARW topological self-termination, specimen segmentation, or two-run bitwise determinism.  
  evidence: `2017-hannezo-branching-random-walk-annihilation.py` L182–187

Remediation:
- Implement a full 2D off-lattice spatial Branching and Annihilating Random Walk (BARW) ductal network simulator where active tips migrate persistently at speed v_0 with angular noise sigma_theta, deposit immutable differentiated ductal trail segments I, bifurcate stochastically at rate k_b (A -> 2A), and irreversibly annihilate (A + I -> I) upon approaching within sensing radius R_a of any non-sibling duct.
- Simulate the 3 Hannezo et al. (2017) organ universality regimes (Mouse Pubertal Mammary Gland, Metanephric Kidney, and Embryonic Pancreas) plus TGF-beta / R_a perturbations (reduced R_a hyper-branching, elevated R_a hypoplasia, and R_a = 0 non-annihilating exponential explosion ablation), demonstrating collapse onto the critical Galton-Watson branching probability q_branch -> 1/2 and subtree distribution P(S) ~ S^(-3/2).
- Load and SHA-256-verify `datasets/branching_random_walk_annihilation/branching_random_walk_annihilation_reference.json` and `mouse_branched_ductal_network_specimen.jpg` at runtime, segment the cyan/green active distal buds (A, N_term = 34) and blue/magenta differentiated ductal tree (I, N_branch = 33), and verify the exact topological identity N_term = N_branch + 1 (q_branch = 33/67 = 0.4925) and peripheral active-tip pulse localization (r/R_max = 0.783).
- Expand `--self-test`, CLI parameter flags (`--kb`, `--ra`, `--v0`, `--sigma-theta`, `--ka`, `--seed`), and `measure_morphoeval()` to report >= 8 quantitative metrics with reference values, dispersion (RMSE, Pearson r, CV), and negative-control ablation deltas.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Hannezo et al. (2017) Cell & Scheele et al. (2017) Nature unifying theory of branching morphogenesis via branching and annihilating random walks (BARW)
- **spec_fingerprint** (semantic model + inputs): `8396b081dd41f5428f101435bc6396b6e50c4b756e20f4883dc6ac81bd4c137b`
- **input_dataset_sha256**: `datasets/branching_random_walk_annihilation/branching_random_walk_annihilation_reference.json` (`d85ad5c91bd80050…`), `datasets/branching_random_walk_annihilation/mouse_branched_ductal_network_specimen.jpg` (`d8946f47dcc4d313…`), `datasets/branching_random_walk_annihilation/paper_citations.bib` (`521cc284892baf9f…`), `datasets/branching_random_walk_annihilation/scout_manifest.json` (`8b04ea70e9991ba0…`)
- `docs/media/branching_random_walk_annihilation_comparison.png` — artifact_sha256 `a8730e1f88a086fe…`, 111,422 B, 1890×588 px

| measurement | value | reference | method |
|---|---:|---|---|
| BARW characteristic branch length l_b = v_0 / k_b (WT) | 8.333333 dimensionless length | 8.333333 | compute_barw_steady_state(v0=1.0, kb=0.12) |
| BARW mean-field homeostatic duct density i_inf = k_b / k_a (WT) | 0.285714 dimensionless duct density | 0.285714 | compute_barw_steady_state(kb=0.12, ka=0.42) |
| Uncoupled binary tree branching fraction q_branch (N_branch = 30) | 0.491803 probability | — | N_branch / (N_branch + N_term) with N_term = N_branch + 1 |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1 baseline) evaluation of `2017-hannezo-branching-random-walk-annihilation.py` prior to Step 12 remediation in `2026-10-19-branching-random-walk-annihilation-loop`.
