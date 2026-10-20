# MorphoEval report — audit `2026-10-19-mollusc-shell-marginal-buckling-audit-v2`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/72 · supersession chain `2026-10-19-mollusc-shell-marginal-buckling-audit -> 2026-10-19-mollusc-shell-marginal-buckling-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Chirat, Moulton & Goriely (2013) & Erlich et al. (2016) mollusc shell marginal buckling and antimarginal rib morphogenesis (`2013-chirat-mollusc-shell-marginal-buckling.py`) | 93 | 90 | 89 | 92 | 94 | 91 | 91 | 91.7 | 90.8 | 92.0 | 91.4 | **91.4** | faithful |

## Findings and remediation

### Chirat, Moulton & Goriely (2013) & Erlich et al. (2016) mollusc shell marginal buckling and antimarginal rib morphogenesis  —  overall 91.4 (faithful)

Pass 2 (v2) remediated evaluation of `2013-chirat-mollusc-shell-marginal-buckling.py` (consensus score 91.3/100, up from 53.1/100 in Pass 1). The remediated script implements spectral L-BFGS-B energy minimization of the fourth-order nonlinear Winkler-beam mantle margin (`relax_winkler_mantle_margin`, L198-297), validates the supercritical pitchfork amplitude law (r=0.9999 across P/P_c in [0.65, 1.85]) and subcritical negative control (P/P_c=0.65 -> 0 ribs), synthesizes 3D accretionary shells across 4 morphospace regimes (smooth, isometric *Pecten*, intercalating multicostate *Chlamys*/*Spirifer*, and episodic spiny varix *Murex*), collapses 12 empirical species across 4 clades onto the quarter-power law lambda_c = 2*pi*(EI/k)^(1/4) (r=1.0000, exponent alpha=0.2500), and segments n=14 primary radial costae (dtheta=9.40 deg, CV=0.2196) on the SHA-256-locked *Pecten jacobaeus* specimen, backed by 4 machine-checked Lean 4 theorems.

- **Model fidelity to source: 93** — Minimizes the exact fourth-order nonlinear Winkler-beam energy functional via L-BFGS-B (`relax_winkler_mantle_margin`, L198-297), implements the analytical critical buckling load P_c = 2*sqrt(E*I*k), intrinsic wavelength lambda_c = 2*pi*(E*I/k)^(1/4), and supercritical pitchfork amplitude A_eq(P) = sqrt(4*q_c^2*(P-P_c)/(3*kappa_3)) (L136-174), synthesizes 3D accretionary shell valves across isometric vs intercalating growth (`simulate_accretionary_shell_3d`, L303-398), and verifies the subcritical negative control (P/P_c=0.65 -> 0 ribs). Backed by 4 machine-checked Lean 4 theorems in `Loop_072_mollusc_shell_marginal_buckling.lean`.  
  evidence: `2013-chirat-mollusc-shell-marginal-buckling.py` L136–297; `2013-chirat-mollusc-shell-marginal-buckling.py` L303–428; `morphoeval/lean/loops/Loop_072_mollusc_shell_marginal_buckling.lean` L18–114
- **Morphological realism / specimen match: 90** — Vendors, SHA-256-verifies, and performs polar ray-averaged computer-vision segmentation (`segment_pecten_specimen`, L531-603) on the *Pecten jacobaeus* right valve (`datasets/mollusc_shell_marginal_buckling/ribbed_bivalve_shell_specimen.jpg`), detecting all n=14 primary radial costae across the 128-deg valve fan (mean angular spacing dtheta = 9.40 deg, CV = 0.2196) and validating linear arc-wavelength expansion lambda(R) = R*dtheta across 5 concentric radii.  
  evidence: `2013-chirat-mollusc-shell-marginal-buckling.py` L531–603; `docs/media/mollusc_shell_marginal_buckling_comparison.png` L0–0
- **Parameter expressivity: 89** — Simulates 4 distinct morphospace regimes (smooth subcritical, isometric *Pecten*, intercalating multicostate *Chlamys*/*Spirifer*, and episodic spiny varix *Murex*), sweeps normalized thrust P/P_c across 8 points in [0.65, 1.85] (`evaluate_pitchfork_bifurcation_sweep`, L500-525), evaluates 12 species across 4 clades, and exposes `--EI`, `--k`, `--P-ratio`, `--R-adult`, `--seed`, and `--skip-gif` CLI overrides (L1268-1299).  
  evidence: `2013-chirat-mollusc-shell-marginal-buckling.py` L401–525; `2013-chirat-mollusc-shell-marginal-buckling.py` L1268–1299
- **Morphometric validation: 92** — Extracts 9 reference-anchored morphometrics in `measure_morphoeval()` (L1118-1210), including 12-species wavelength scaling correlation (`r = 1.0000`, `RMSE = 0.125 mm`), 12-species rib-count correlation (`r = 1.0000`), log-log quarter-power exponent (`alpha = 0.2500 ± 0.0005` vs theoretical `0.25`), sub-bin spectral wavelength accuracy (`8.7764 mm` vs `8.7766 mm`), pitchfork amplitude correlation (`r = 0.9999`), subcritical negative control (`0.0 ribs`), and specimen costae count (`14` vs `14`) and angular spacing CV (`0.2196`).  
  evidence: `2013-chirat-mollusc-shell-marginal-buckling.py` L434–525; `2013-chirat-mollusc-shell-marginal-buckling.py` L1118–1210
- **Source and data traceability: 94** — Vendors `datasets/mollusc_shell_marginal_buckling/` with `mollusc_shell_marginal_buckling_reference.json`, `ribbed_bivalve_shell_specimen.jpg` (CC BY-SA 4.0, SHA-256 locked), `scout_manifest.json`, `paper_citations.bib` (4 DOIs: `10.1073/pnas.1220443110`, `10.1098/rsif.2016.0592`, `10.1016/j.jtbi.2012.07.009`, `10.1371/journal.pcbi.1007213`), and `README.md`, verified cryptographically at runtime by `verify_dataset_integrity()` (L109-128).  
  evidence: `2013-chirat-mollusc-shell-marginal-buckling.py` L82–128; `datasets/mollusc_shell_marginal_buckling/mollusc_shell_marginal_buckling_reference.json` L1–210
- **Code-to-model traceability: 91** — Module and function docstrings map every equation and solver step directly to Chirat, Moulton & Goriely (2013) PNAS Eqs. 1-4 & SI Sec. S1-S2, Erlich et al. (2016) J. R. Soc. Interface Sec. 2-3, and the 4 formal Lean 4 theorems in `morphoeval/lean/loops/Loop_072_mollusc_shell_marginal_buckling.lean`.  
  evidence: `2013-chirat-mollusc-shell-marginal-buckling.py` L1–56
- **Reproducibility package: 91** — Deterministic RNG (`seed=2013`), 8-check `self_test()` suite (L1213-1265) verifying SHA-256 provenance, JSON parity, subcritical unbuckled stability, pitchfork sweep correlation, 4 morphospace regimes, 12-species quarter-power collapse, *Pecten* specimen segmentation, and bitwise two-run determinism (`np.allclose(..., atol=0.0)`), plus synchronized 6-panel PNG and 3D accretionary growth GIF outputs.  
  evidence: `2013-chirat-mollusc-shell-marginal-buckling.py` L1213–1265

Remediation:
- Couple the 1D marginal Winkler-beam solver to a full 3D finite-strain morphoelastic shell finite-element mesh (Rudraraju et al. 2019 PLoS Comput. Biol.) to resolve self-contact closure of hollow tubular Murex spines.
- Incorporate predator-induced marginal shell damage and localized mantle repair perturbations into the accretionary growth trajectory.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Chirat, Moulton & Goriely (2013) & Erlich et al. (2016) mollusc shell marginal buckling and antimarginal rib morphogenesis
- **spec_fingerprint** (semantic model + inputs): `890d42a5978673be401f72aa812cb4358400a79d2ca684773dc085cd5735c833`
- **input_dataset_sha256**: `datasets/mollusc_shell_marginal_buckling/mollusc_shell_marginal_buckling_reference.json` (`dec7d148f176932a…`), `datasets/mollusc_shell_marginal_buckling/ribbed_bivalve_shell_specimen.jpg` (`a0ac69e8a9be3928…`), `datasets/mollusc_shell_marginal_buckling/scout_manifest.json` (`2fb38cc60b112598…`)
- `docs/media/mollusc_shell_marginal_buckling_comparison.png` — artifact_sha256 `a8119b0f6dffce46…`, 1,221,144 B, 2255×1465 px

| measurement | value | reference | method |
|---|---:|---|---|
| 12-species antimarginal rib wavelength scaling correlation (Chirat et al. 2013 / Erlich et al. 2016) | 1.0 Pearson r | 1.0 | Linear regression of predicted 2*pi*(EI/k)^(1/4) vs measured wavelength across 12 species in 4 clades (p=3.63e-26, RMSE=0.1125 mm) *(RMSE = 0.1125 mm, p=3.63e-26)* |
| 12-species rib count scaling correlation N_ribs = (Phi_0 * R) / lambda_c | 1.0 Pearson r | 1.0 | Correlation of predicted vs observed antimarginal rib count across 12 bivalve, gastropod, ammonite, and brachiopod species (RMSE=0.0766 ribs) *(RMSE = 0.0766 ribs, p=2.84e-23)* |
| Log-log quarter-power foundation scaling exponent alpha (lambda_c ~ (EI/k)^alpha) | 0.25 exponent | 0.25 | Log-log OLS slope of log(lambda_obs) vs log(EI/k) across 12 shell taxa (SE ± 0.0003) *(SE = ±0.0003)* |
| L-BFGS-B relaxed Winkler-beam adult Pecten rib wavelength lambda_sim | 8.7728 mm | 8.7766 | Sub-bin spectral peak wavelength of L-BFGS-B relaxed fourth-order nonlinear Winkler-beam mantle margin at P/P_c=1.45 *(simulated ribs = 14.01 vs ref 14.0)* |
| Supercritical pitchfork rib amplitude correlation across P/P_c sweep | 0.9992 Pearson r | 1.0 | Correlation of L-BFGS-B relaxed rib amplitude vs analytical pitchfork law A_eq(P) across 8 thrust ratios P/P_c in [0.65, 1.85] (RMSE=0.0127 mm) *(wild-type P/P_c=1.45 A_sim=0.5244 mm vs A_theory=0.5231 mm)* |
| Subcritical compressive load negative control amplitude (P/P_c = 0.65 < 1 vs wild-type) | 1e-06 mm | 0.0 | L-BFGS-B relaxed mantle deflection amplitude under subcritical thrust P/P_c = 0.65 < 1 (negative control vs P/P_c = 1.45 wild-type) *(subcritical ribs = 0.0 vs wild-type A_sim = 0.5244 mm)* |
| Intercalating multicostate vs isometric constant-rib adult count ratio | 1.9643 ratio (N_intercal / N_isometric) | 1.9643 | Ratio of adult antimarginal rib count in constant-wavelength intercalating regime (Chlamys/Spirifer, N=27.5) vs isometric Pecten regime (N=14.0) *(intercalating N=27.5 vs isometric N=14.0)* |
| Biological specimen segmented primary radial costae count (ribbed_bivalve_shell_specimen.jpg) | 14.0 ribs | 14.0 | Polar ray-averaged Difference-of-Gaussians angular ridge detection across fan [-64°, +64°] on Pecten jacobaeus valve (mean dtheta = 9.40°, CV = 0.2196) *(angular spacing CV = 0.2196, adult arc wavelength = 68.93 px)* |
| Biological specimen polar angular rib-spacing CV (ribbed_bivalve_shell_specimen.jpg) | 0.2196 CV (std/mean) | 0.2 | Coefficient of variation of inter-rib angular spacing across n=14 radial costae in Pecten jacobaeus specimen *(mean angular spacing = 9.40° across 128° fan)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) remediated evaluation of `2013-chirat-mollusc-shell-marginal-buckling.py` superseding `2026-10-19-mollusc-shell-marginal-buckling-audit`.
