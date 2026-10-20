# MorphoEval report — audit `2026-10-19-vascular-cube-law-branching-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-vascular-cube-law-branching-audit -> 2026-10-19-vascular-cube-law-branching-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Murray (1926a, 1926b), Zamir (1976) & Sherman (1981) minimum-work vascular cube law & arterial branching angles (`1959-murray-vascular-cube-law-branching.py`) | 94 | 87 | 89 | 89 | 94 | 92 | 92 | 92.3 | 87.8 | 92.7 | 90.6 | **90.6** | faithful |

## Findings and remediation

### Murray (1926a, 1926b), Zamir (1976) & Sherman (1981) minimum-work vascular cube law & arterial branching angles  —  overall 90.6 (faithful)

Pass 2 (v2 remediated) evaluation of `1959-murray-vascular-cube-law-branching.py` (90.9/100, up from 51.4/100 in Pass 1; +39.5 overall). The remediated script implements Murray's (1926a) Poiseuille + metabolic cost functional P(r), stationarity partition P_met(r*)/P_visc(r*) = 2.0, scale-invariant wall shear stress tau_w = sqrt(eta*b) = 8.2189 dyn/cm^2 across all 7 Sherman (1981) arteriole generations, and Murray's (1926b) full asymmetric bifurcation cosine laws cos(theta_1), cos(theta_2) across alpha = r_2/r_1 in (0, 1]; compares against competing Zamir (1976) surface-drag (90 deg), Steiner length-minimum (120 deg), and Da Vinci area-preserving (gamma=2, +134.8% excess tree power) ablation controls; segments N=19 Y-junctions on the SHA-256-verified human retinal fundus specimen (`retinal_vasculature_specimen.jpg`, mean angle 79.28 deg, cube-law ratio 0.917, tortuosity B/A = 1.015x); and is backed by 3 machine-checked Lean 4 theorems in `morphoeval/lean/loops/Loop_052_vascular_cube_law_branching.lean`.

- **Model fidelity to source: 94** — Implements Murray's (1926a) Poiseuille + metabolic power functional P(r), stationarity radius r*, and exact wall shear stress invariance tau_w = sqrt(eta*b) = 8.2189 dyn/cm^2 across 7 vascular generations (L100-147, L284-338), plus Murray's (1926b) full asymmetric bifurcation angle laws cos(theta_1) and cos(theta_2) alongside Zamir's (1976) surface-drag (90 deg), Steiner's length-minimizing (120 deg), and Da Vinci's area-preserving (gamma=2) negative controls (L154-279, L340-421). Backed by 3 machine-checked Lean 4 theorems (0 `sorry`s) in `Loop_052_vascular_cube_law_branching.lean`.  
  evidence: `1959-murray-vascular-cube-law-branching.py` L100–219; `1959-murray-vascular-cube-law-branching.py` L284–378; `morphoeval/lean/loops/Loop_052_vascular_cube_law_branching.lean` L33–110
- **Morphological realism / specimen match: 87** — Segments the SHA-256-locked human retinal fundus vasculature specimen (`datasets/vascular_cube_law_branching/retinal_vasculature_specimen.jpg`, Panels A and B) using green-channel hemoglobin top-hat enhancement, Zhang-Suen 1-pixel skeletonization, and Euclidean distance transform (EDT) lumen caliber mapping (L427-616), extracting N=19 Y-junction bifurcations (mean angle 79.28 ± 17.34 deg vs Murray 74.93-80 deg theoretical range; mean cube-law ratio 0.917 ± 0.246; Panel B/A tortuosity ratio 1.015x) and displaying both skeleton and EDT caliber overlays in Panel E of the 6-panel synthesis figure.  
  evidence: `1959-murray-vascular-cube-law-branching.py` L427–616; `docs/media/vascular_cube_law_branching_comparison.png` L0–0
- **Parameter expressivity: 89** — Sweeps daughter radius asymmetry alpha = r_2/r_1 across [0.15, 1.0], sweeps branching exponent gamma across [1.8, 4.2] (121 grid points, confirming global power minimum at gamma* = 3.000), compares three angle optimality regimes (`murray_volume`, `zamir_surface`, `steiner_length`), and exposes CLI flags `--alpha`, `--gamma`, `--eta`, `--b`, `--r0-um`, `--generations`, and `--seed` (L1057-1086).  
  evidence: `1959-murray-vascular-cube-law-branching.py` L340–421; `1959-murray-vascular-cube-law-branching.py` L1057–1086
- **Morphometric validation: 89** — `measure_morphoeval()` reports 8 quantitative morphometrics with reference values, uncertainty/dispersion statistics (RMSE, Pearson r, p-value, SD, CV), and negative-control ablation deltas (L891-985): empirical human retinal junctions J01-J08 achieve total-angle RMSE = 0.134 deg and branch-angle Pearson r = 0.9999 (p = 2.4e-26), cube-law conservation ratio (r_1^3+r_2^3)^(1/3)/r_0 = 0.9956 ± 0.0033 (CV = 0.0033) vs Da Vinci gamma=2 control 1.1091 (+11.4% bias and +134.8% excess tree power), and segmented fundus Y-junction angle mean = 79.28 deg (N = 19).  
  evidence: `1959-murray-vascular-cube-law-branching.py` L221–279; `1959-murray-vascular-cube-law-branching.py` L891–985
- **Source and data traceability: 94** — Vendors `datasets/vascular_cube_law_branching/` with `vascular_cube_law_branching_reference.json`, `retinal_vasculature_specimen.jpg` (Wikimedia Commons CC BY 4.0), `paper_citations.bib` (4 primary DOIs: `10.1073/pnas.12.3.207`, `10.1085/jgp.9.6.835`, `10.1016/0022-5193(76)90058-8`, `10.1085/jgp.78.4.431`), `scout_manifest.json`, and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L72-93).  
  evidence: `1959-murray-vascular-cube-law-branching.py` L72–93; `datasets/vascular_cube_law_branching/vascular_cube_law_branching_reference.json` L1–96
- **Code-to-model traceability: 92** — Every computational section and docstring in `1959-murray-vascular-cube-law-branching.py` maps explicitly to numbered equations in Murray (1926a) Eqs. 1-8, Murray (1926b) Eqs. 3-6, Zamir (1976) Table 1, Sherman (1981) Table I, and the 3 Lean 4 theorems in `Loop_052_vascular_cube_law_branching.lean` (L1-35, L95-220).  
  evidence: `1959-murray-vascular-cube-law-branching.py` L1–35; `morphoeval/math_kg_loops/murray-vascular-cube-law-branching.yaml` L1–74
- **Reproducibility package: 92** — `self_test()` (L988-1054) executes 7 automated verification suites covering SHA-256 dataset integrity, symmetric & asymmetric angle parity, stationarity P_met/P_visc = 2.0, gamma* = 3.000 global minimum, empirical retinal junction RMSE/r, fundus image Y-junction segmentation, and two-run bitwise determinism.  
  evidence: `1959-murray-vascular-cube-law-branching.py` L988–1054

Remediation:
- Extend 2D retinal vascular tree synthesis to 3D curved hemispherical retinal cup geometry with coupled arteriovenous interdigitation.
- Incorporate non-Newtonian Fahraeus-Lindqvist apparent blood viscosity eta(r) in pre-capillary arterioles (r < 25 um) to quantify the shift in optimal exponent from gamma = 3.0 toward gamma ≈ 2.7.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Murray (1926a, 1926b), Zamir (1976) & Sherman (1981) minimum-work vascular cube law & arterial branching angles
- **spec_fingerprint** (semantic model + inputs): `d50b49ef447bf68e29d604bf8fac7d65aec5afb9d6827a3f7ef26e8dd37a2ab8`
- **input_dataset_sha256**: `datasets/vascular_cube_law_branching/retinal_vasculature_specimen.jpg` (`bc548d4150e070a5…`), `datasets/vascular_cube_law_branching/vascular_cube_law_branching_reference.json` (`6d1761f2e01e4fa7…`)
- `docs/media/vascular_cube_law_branching_comparison.png` — artifact_sha256 `6f5c640c92a28928…`, 807,479 B, 2537×1537 px

| measurement | value | reference | method |
|---|---:|---|---|
| symmetric daughter radius ratio r_1 / r_0 | 0.793701 ratio | 0.793701 | Murray (1926a) cube law r_0^3 = 2 * r_1^3 => r_1/r_0 = 2^(-1/3) *(area expansion ratio beta = (r_1^2 + r_2^2)/r_0^2 = 1.259921 (ref 2^(1/3) = 1.259921))* |
| symmetric total bifurcation angle (theta_1 + theta_2) | 74.9346 deg | 74.9346 | 2 * arccos(2^(-1/3)) from Murray (1926b) Eq. 5 vs Zamir 90 deg and Steiner 120 deg controls *(half-angle theta_1 = theta_2 = 37.4673 deg; Steiner length-minimizing control = 120.0 deg)* |
| metabolic-to-viscous power ratio at optimal radius P_met(r*) / P_visc(r*) | 2.0 ratio | 2.0 | stationarity dP/dr = 0 evaluated at r* = murray_optimal_radius(Q) (Lean 4 theorem murray_stationarity_viscous_twice_metabolic) |
| empirical human retinal bifurcation angle RMSE (Murray prediction vs J01-J08) | 0.1342 deg | 0.0 | RMSE and Pearson r across N=8 asymmetric human retinal arterial junctions (alpha in 0.56..0.99) *(branch angle Pearson r = 1.0000 (p=6.18e-30), branch RMSE = 0.119 deg)* |
| empirical retinal cube-law conservation ratio (r_1^3 + r_2^3)^(1/3) / r_0 | 0.9956 ratio | 1.0 | mean ± SD (CV) of (r_1^3 + r_2^3)^(1/3) / r_0 vs Da Vinci square-law ablation control (r_1^2 + r_2^2)^(1/2) / r_0 *(std = 0.0034, CV = 0.0034; Da Vinci gamma=2 control gives 1.1088 (+11.7% bias))* |
| Da Vinci area-preserving control (gamma=2) excess vascular tree power penalty | 134.796 % | 0.0 | 100 * (P_tree(gamma=2.0) - P_tree(gamma=3.0)) / P_tree(gamma=3.0) across 7-generation arterial tree *(numerical minimum found at gamma* = 3.000; wall shear CV at gamma=3 is 2.95e-16 vs +100% drift at gamma=2)* |
| segmented retinal fundus specimen Y-junction bifurcation angle (mean) | 79.279 deg | 74.9346 | green-channel top-hat vessel segmentation + medial-axis skeletonization across N=19 Y-junctions in retinal_vasculature_specimen.jpg *(std = 17.60 deg, CV = 0.222, segmented cube ratio = 0.917 ± 0.247, tortuosity B/A ratio = 1.015)* |
| generational wall shear stress tau_w across 7 Sherman (1981) arteriole levels | 8.2189 dyn/cm^2 | 8.2189 | tau_w = 4*eta*Q_g / (pi*r_g^3) = sqrt(eta*b) across generations g=0..6 *(generational shear CV = 2.95e-16 (exact scale invariance))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2 remediated) evaluation of `1959-murray-vascular-cube-law-branching.py` after completing the 12-step closed remediation loop (`2026-10-19-vascular-cube-law-branching-loop`).
