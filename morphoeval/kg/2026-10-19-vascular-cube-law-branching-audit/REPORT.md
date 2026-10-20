# MorphoEval report — audit `2026-10-19-vascular-cube-law-branching-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Murray (1926a, 1926b), Zamir (1976) & Sherman (1981) minimum-work vascular cube law & arterial branching angles (`1959-murray-vascular-cube-law-branching.py`) | 62 | 22 | 42 | 48 | 58 | 62 | 68 | 55.3 | 31.8 | 63.1 | 48.2 | **48.2** | partially faithful |

## Findings and remediation

### Murray (1926a, 1926b), Zamir (1976) & Sherman (1981) minimum-work vascular cube law & arterial branching angles  —  overall 48.2 (partially faithful)

Pass 1 (v1 baseline) evaluation of `1959-murray-vascular-cube-law-branching.py` (51.4/100). The v1 script implements Murray's (1926a) Poiseuille viscous power + metabolic blood-volume cost functional P(r) = 8*eta*L*Q^2/(pi*r^4) + b*pi*r^2*L, stationarity radius r*, symmetric cube law r_0^3 = 2*r_1^3 (r_1/r_0 = 2^(-1/3)), and symmetric optimal branching angle (theta_1 + theta_2 = 74.93 deg), but omits asymmetric daughter bifurcations (alpha = r_2/r_1 < 1), omits alternative optimality ablations (Da Vinci area-preserving gamma=2, Zamir 90 deg surface-area minimization, Steiner 120 deg length minimization), neither loads nor verifies `datasets/vascular_cube_law_branching/vascular_cube_law_branching_reference.json`, and performs zero biological specimen segmentation on `datasets/vascular_cube_law_branching/retinal_vasculature_specimen.jpg`.

- **Model fidelity to source: 62** — Evaluates Murray's (1926a) Poiseuille + metabolic power functional P(r) and stationarity radius r* (L35-50) and computes the symmetric bifurcation angle theta_1 + theta_2 = 74.9346 deg (L53-68). However, asymmetric daughter branching (r_1 != r_2, cos(theta_2) for side branches) from Murray (1926b) Eqs. 3-6, wall shear stress invariance tau_w = 4*eta*Q/(pi*r^3) = sqrt(eta*b) across vessel generations, and Zamir's (1976) competing optimality regimes (surface-area drag 90 deg, Steiner minimum length 120 deg, Da Vinci gamma=2 area conservation) are not implemented in v1.  
  evidence: `1959-murray-vascular-cube-law-branching.py` L35–68; `1959-murray-vascular-cube-law-branching.py` L71–96
- **Morphological realism / specimen match: 22** — Although `datasets/vascular_cube_law_branching/retinal_vasculature_specimen.jpg` is present in the dataset folder, the v1 script never loads, segments, or displays the human retinal fundus vasculature specimen or extracts vessel skeleton bifurcations and caliber ratios from the photograph.  
  evidence: `1959-murray-vascular-cube-law-branching.py` L99–160; `docs/media/vascular_cube_law_branching_comparison.png` L0–0
- **Parameter expressivity: 42** — Hardcodes a single symmetric ratio 2^(-1/3) and root radius r_0 = 82 um (L29-31) without sweeping daughter asymmetry alpha = r_2/r_1 in (0, 1], vascular scaling exponent gamma in [1.5, 4.0], or blood viscosity / metabolic cost parameters, and exposes only `--self-test` and `--skip-gif` on the CLI (L253-257).  
  evidence: `1959-murray-vascular-cube-law-branching.py` L29–32; `1959-murray-vascular-cube-law-branching.py` L253–263
- **Morphometric validation: 48** — `measure_morphoeval()` reports 3 exact algebraic scalars (r_1/r_0, symmetric angle 74.9346 deg, P_met/P_visc = 2.0, L199-236), but lacks statistical dispersion metrics (RMSE, Pearson r, CV), lacks comparison to Zamir (1979) / Sherman (1981) empirical human retinal bifurcation measurements (J01-J08), and lacks negative-control ablation deltas against Da Vinci (gamma=2) or Steiner (120 deg) branching.  
  evidence: `1959-murray-vascular-cube-law-branching.py` L199–236
- **Source and data traceability: 58** — Cites Murray (1926a, 1926b) in the docstring (L5-7), but v1 does not load `datasets/vascular_cube_law_branching/vascular_cube_law_branching_reference.json` at runtime or verify SHA-256 cryptographic digests of the reference JSON and retinal fundus specimen image.  
  evidence: `1959-murray-vascular-cube-law-branching.py` L4–15
- **Code-to-model traceability: 62** — Functions have brief one-line docstrings (L35-75), but do not map code blocks to specific equations in Murray (1926a) Eqs. 1-8, Murray (1926b) Eqs. 1-6, Zamir (1976) Table 1, Sherman (1981) Table 1, or the 3 Lean 4 theorems in `morphoeval/lean/loops/Loop_052_vascular_cube_law_branching.lean`.  
  evidence: `1959-murray-vascular-cube-law-branching.py` L35–96
- **Reproducibility package: 68** — Includes `--self-test` and `--skip-gif` flags and renders deterministic PNG/GIF outputs (L239-263), but `--self-test` checks only 4 scalar assertions without verifying JSON dataset parity, specimen segmentation, or two-run bitwise determinism.  
  evidence: `1959-murray-vascular-cube-law-branching.py` L239–250

Remediation:
- Implement full asymmetric arterial bifurcation laws cos(theta_1), cos(theta_2), and cos(theta_1 + theta_2) across alpha = r_2/r_1 in [0.1, 1.0] (Murray 1926b Eqs. 3-6) and verify scale-invariant wall shear stress tau_w = 4*eta*Q/(pi*r^3) = sqrt(eta*b) across all 7 Sherman (1981) arteriole generations.
- Implement competing optimality negative controls and ablations: Da Vinci area-preserving branching (gamma = 2.0, r_0^2 = r_1^2 + r_2^2), Zamir (1976) lumen-surface-minimizing branching (theta_1 + theta_2 = 90 deg), and Steiner length-minimizing branching (theta_1 + theta_2 = 120 deg), quantifying total hydraulic + metabolic power penalty (% excess work) relative to Murray's gamma = 3.0 optimum.
- Load and SHA-256-verify `datasets/vascular_cube_law_branching/vascular_cube_law_branching_reference.json` and `retinal_vasculature_specimen.jpg` at runtime, segment the human retinal vasculature fundus panels (Frangi/multi-scale matched-filter vessel enhancement, medial-axis skeletonization, distance-transform vessel caliber estimation, and Y-junction angle/radius ratio extraction), and compare against empirical human retinal junctions J01-J08 (RMSE, Pearson r, CV).
- Expand `--self-test`, CLI parameter flags (`--alpha`, `--gamma`, `--eta`, `--b`, `--generations`, `--seed`), and `measure_morphoeval()` to report >= 8 quantitative metrics with reference values, dispersion (RMSE, CV, std), and negative-control deltas.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Murray (1926a, 1926b), Zamir (1976) & Sherman (1981) minimum-work vascular cube law & arterial branching angles
- **spec_fingerprint** (semantic model + inputs): `28bf2bb7d9ebb4ea9c5721cc29a875189747c8344d5828934425a1601bbeb74e`
- **input_dataset_sha256**: `datasets/vascular_cube_law_branching/retinal_vasculature_specimen.jpg` (`bc548d4150e070a5…`), `datasets/vascular_cube_law_branching/vascular_cube_law_branching_reference.json` (`450775cc5fc7ee81…`)
- `docs/media/vascular_cube_law_branching_comparison.png` — artifact_sha256 `a251584c0520dfda…`, 153,263 B, 1960×644 px

| measurement | value | reference | method |
|---|---:|---|---|
| symmetric daughter radius ratio r_1 / r_0 | 0.793701 ratio | 0.793701 | Murray (1926a) cube law r_0^3 = 2 * r_1^3 => r_1/r_0 = 2^(-1/3) |
| symmetric total bifurcation angle (theta_1 + theta_2) | 74.9346 deg | 74.9347 | 2 * arccos(2^(-1/3)) from Murray (1926b) Eq. 5 |
| metabolic-to-viscous power ratio at optimal radius P_met(r*) / P_visc(r*) | 2.0 ratio | 2.0 | stationarity dP/dr = 0 evaluated at r* = murray_optimal_radius(Q) |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1 baseline) evaluation of `1959-murray-vascular-cube-law-branching.py` prior to Step 12 remediation in `2026-10-19-vascular-cube-law-branching-loop`.
