# MorphoEval report — audit `2026-10-19-mollusc-shell-marginal-buckling-audit`

Rubric **0.1.0** · checkpoint `HEAD` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/72

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Chirat, Moulton & Goriely (2013) & Erlich et al. (2016) mollusc shell marginal buckling and antimarginal rib morphogenesis (`2013-chirat-mollusc-shell-marginal-buckling.py`) | 58 | 24 | 45 | 50 | 64 | 62 | 69 | 53.7 | 33.8 | 65.5 | 49.2 | **49.2** | partially faithful |

## Findings and remediation

### Chirat, Moulton & Goriely (2013) & Erlich et al. (2016) mollusc shell marginal buckling and antimarginal rib morphogenesis  —  overall 49.2 (partially faithful)

Pass 1 (v1) baseline evaluation of `2013-chirat-mollusc-shell-marginal-buckling.py` (consensus score 53.1/100). The v1 script integrates the 1D fourth-order Winkler-beam mantle-margin PDE at a single adult scallop margin (L=122.87 mm, P/P_c=1.45), but omits 3D accretionary shell surface morphogenesis, isometric vs intercalating ontogenetic rib scaling, supercritical pitchfork bifurcation sweeps across P/P_c, subcritical negative controls (P < P_c), 12-species cross-taxa scaling collapse, and computer-vision polar segmentation of the vendored *Pecten jacobaeus* specimen.

- **Model fidelity to source: 58** — Solves a 1D periodic Fourier relaxation of E*I*w_ssss + P*w_ss + k*w + kappa_3*w^3 = 0 at a single fixed arc length (L28-78), but lacks L-BFGS-B energy minimization of the exact Winkler-beam functional, omits the analytical supercritical pitchfork law A_eq(P) = sqrt(4*q_c^2*(P-P_c)/(3*kappa_3)), omits 3D accretionary shell surface deposition across isometric vs intercalating growth regimes, and lacks a subcritical negative control (P/P_c = 0.65 < 1).  
  evidence: `2013-chirat-mollusc-shell-marginal-buckling.py` L28–78
- **Morphological realism / specimen match: 24** — Renders a 2-panel synthetic 1D profile and dispersion curve without loading, SHA-256-verifying, or segmenting the vendored *Pecten jacobaeus* ribbed bivalve specimen (`datasets/mollusc_shell_marginal_buckling/ribbed_bivalve_shell_specimen.jpg`).  
  evidence: `2013-chirat-mollusc-shell-marginal-buckling.py` L81–120; `docs/media/mollusc_shell_marginal_buckling_comparison.png` L0–0
- **Parameter expressivity: 45** — Hardcodes a single parameter tuple (EI=1.085, k=0.285, P/P_c=1.45, L=122.87) with no CLI parameter overrides, no supercritical pitchfork sweep across P/P_c, and no comparison across the 4 morphospace regimes (smooth subcritical, isometric *Pecten*, intercalating multicostate *Chlamys*, and episodic spiny varix *Murex*).  
  evidence: `2013-chirat-mollusc-shell-marginal-buckling.py` L28–38
- **Morphometric validation: 50** — Uses integer FFT bin `k_bin` (L63-66) without parabolic sub-bin interpolation and omits the 12-species cross-taxa scaling collapse (Bivalvia, Gastropoda, Ammonitida, Brachiopoda), pitchfork amplitude validation, and specimen radial rib angular CV.  
  evidence: `2013-chirat-mollusc-shell-marginal-buckling.py` L63–66
- **Source and data traceability: 64** — Reference dataset `datasets/mollusc_shell_marginal_buckling/mollusc_shell_marginal_buckling_reference.json` and specimen image exist on disk, but v1 does not load or verify their SHA-256 digests at runtime.  
  evidence: `2013-chirat-mollusc-shell-marginal-buckling.py` L1–27
- **Code-to-model traceability: 62** — Module docstring cites Chirat et al. (2013), but functions lack explicit equation mappings to Erlich et al. (2016) and the 4 Lean 4 theorems in `morphoeval/lean/loops/Loop_072_mollusc_shell_marginal_buckling.lean`.  
  evidence: `2013-chirat-mollusc-shell-marginal-buckling.py` L28–78
- **Reproducibility package: 69** — Deterministic RNG (`seed=2013`) and `--self-test` flag are present (L148-162), but `self_test()` checks only 3 basic inequalities without testing SHA-256 dataset provenance, JSON parity, subcritical unbuckled stability, or bitwise two-run determinism.  
  evidence: `2013-chirat-mollusc-shell-marginal-buckling.py` L148–155

Remediation:
- Implement exact L-BFGS-B energy minimization of the fourth-order nonlinear Winkler-beam mantle energy functional U[w] = integral [0.5*EI*w_ss^2 - 0.5*P*w_s^2 + 0.5*k*w^2 + 0.25*kappa_3*w^4] ds alongside the analytical supercritical pitchfork amplitude law A_eq(P) = sqrt(4*q_c^2*(P-P_c)/(3*kappa_3)) and a subcritical negative control (P/P_c = 0.65 < 1 -> 0 ribs, RMS amplitude = 0).
- Simulate 3D accretionary shell valve morphogenesis across ontogeny (R = 12 -> 55 mm) for all 4 Chirat-Erlich morphospace regimes: (1) subcritical smooth valve, (2) isometric constant-rib-count *Pecten jacobaeus* (N_ribs = 14), (3) constant-wavelength secondary rib-intercalating multicostate shell (*Chlamys* / *Spirifer*, N_ribs: 6 -> 27.5), and (4) episodic orthomarginal + antimarginal cancellate/spiny varix flaring.
- Load and SHA-256-verify `datasets/mollusc_shell_marginal_buckling/` at runtime, validate the universal quarter-power rib wavelength scaling law lambda_c = 2*pi*(E*I/k)^(1/4) and rib-number law across 12 empirical bivalve, gastropod, ammonite, and brachiopod species, and perform polar ray-averaged computer-vision segmentation of the 14 primary radial costae and linear wavelength expansion lambda(R) = R*dtheta on the *Pecten jacobaeus* specimen (`ribbed_bivalve_shell_specimen.jpg`).
- Expose full CLI parameter overrides (`--EI`, `--k`, `--P-ratio`, `--R-adult`, `--seed`, `--skip-gif`), upgrade `self_test()` to an 8-point verification suite with bitwise two-run determinism, and render a 6-panel publication-grade synthesis figure and co-evolving 3D accretionary growth GIF.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Chirat, Moulton & Goriely (2013) & Erlich et al. (2016) mollusc shell marginal buckling and antimarginal rib morphogenesis
- **spec_fingerprint** (semantic model + inputs): `890d42a5978673be401f72aa812cb4358400a79d2ca684773dc085cd5735c833`
- **input_dataset_sha256**: `datasets/mollusc_shell_marginal_buckling/mollusc_shell_marginal_buckling_reference.json` (`dec7d148f176932a…`), `datasets/mollusc_shell_marginal_buckling/ribbed_bivalve_shell_specimen.jpg` (`a0ac69e8a9be3928…`), `datasets/mollusc_shell_marginal_buckling/scout_manifest.json` (`2fb38cc60b112598…`)
- `docs/media/mollusc_shell_marginal_buckling_comparison.png` — artifact_sha256 `ea04f8c7bdce3d6c…`, 120,133 B, 1430×572 px

| measurement | value | reference | method |
|---|---:|---|---|
| Simulated antimarginal rib wavelength lambda_sim (v1 integer FFT bin) | 8.7764 mm | 8.7766 | Integer FFT peak wavenumber on 1D periodic Winkler beam (L=122.87 mm) *(n_ribs_sim = 14.0)* |
| Simulated adult Pecten antimarginal rib count | 14.0 ribs | 14.0 | Dominant Fourier mode count along adult scallop margin L=122.87 mm *(RMS amplitude = 0.3679 mm)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `2013-chirat-mollusc-shell-marginal-buckling.py` prior to Step 12 remediation in `2026-10-19-mollusc-shell-marginal-buckling-loop`.
