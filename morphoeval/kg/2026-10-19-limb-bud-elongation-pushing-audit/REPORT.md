# MorphoEval report — audit `2026-10-19-limb-bud-elongation-pushing-audit`

Rubric **0.1.0** · checkpoint `317bc3c` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Boehm et al. (2010) & Gros et al. (2010) vertebrate limb bud oriented cell division & Wnt5a/FGF8 pushing vs proliferation-gradient hypothesis (`2015-boehm-limb-bud-elongation-pushing.py`) | 66 | 35 | 46 | 54 | 64 | 68 | 68 | 59.3 | 42.1 | 66.7 | 54.6 | **54.6** | partially faithful |

## Findings and remediation

### Boehm et al. (2010) & Gros et al. (2010) vertebrate limb bud oriented cell division & Wnt5a/FGF8 pushing vs proliferation-gradient hypothesis  —  overall 54.6 (partially faithful)

Pass 1 (v1) evaluation of `2015-boehm-limb-bud-elongation-pushing.py` (57.3/100). The v1 baseline simulates a 1D/2D boundary contour comparing Model I (isotropic proliferation gradient) against Model II (directional Wnt5a/FGF8 pushing), but omits a full 2D/3D continuum artificial-compressibility pressure-Poisson & Stokes velocity solver (`-nabla p + (1/Re)*nabla^2 v + nabla . sigma_active = 0`, `nabla . v = s(x)`), omits the 5-order-of-magnitude Reynolds number sensitivity sweep (`Re in [10^-6, 10^-1]`), omits the Wnt5a-/- bulbous mutant and Hooke-Jeeves inverse-optimization analysis, does not load or segment `mouse_limb_bud_specimen.jpg`, and exposes no CLI flags beyond `--self-test`.

- **Model fidelity to source: 66** — Implements the IddU/BrdU cell-cycle source profile (`compute_brdu_iddu_source_rate`, L29-40) and a kinematic 2D boundary contour integration (`simulate_limb_bud_v1`, L43-85) comparing isotropic proliferation (`zeta_ocd=0, chi_fgf=0`) against directional pushing (`zeta_ocd=1, chi_fgf=1`). However, v1 omits the 2D/3D continuum artificial-compressibility Poisson pressure solver (Boehm et al. 2010 Eqs. 3-5, 10-11), omits the 5-regime perturbation suite (including the Wnt5a-/- PCP mutant, uniform-Tc control, and FGF-only motility regime), and omits the Hooke-Jeeves inverse-optimization demonstration showing why isotropic growth requires unphysical distal rates (`s > 0.50 h^-1`) and negative DV shrinkage.  
  evidence: `2015-boehm-limb-bud-elongation-pushing.py` L29–40; `2015-boehm-limb-bud-elongation-pushing.py` L43–85
- **Morphological realism / specimen match: 35** — Although `datasets/limb_bud_elongation_pushing/mouse_limb_bud_specimen.jpg` is vendored on disk, the v1 script never loads, verifies, or segments the E11.0 (`St0`) and E11.25 (`St1`) OPT 3D iso-surfaces or dorsal/posterior overlays (`render_outputs_v1`, L88-160), rendering only a 3-panel synthetic contour plot.  
  evidence: `2015-boehm-limb-bud-elongation-pushing.py` L88–160; `docs/media/limb_bud_elongation_pushing_comparison.png` L0–0
- **Parameter expressivity: 46** — Evaluates only 2 hardcoded parameter points (Model I vs Model II) without a Reynolds number sweep across `Re in [10^-6, 10^-1]`, without a 2D parameter sweep across `(zeta_ocd, chi_fgf_wnt5a)`, and without CLI flags for `--zeta-ocd`, `--chi-fgf`, `--reynolds`, `--hours`, `--seed`, or `--output` (L179-189).  
  evidence: `2015-boehm-limb-bud-elongation-pushing.py` L90–92; `2015-boehm-limb-bud-elongation-pushing.py` L179–189
- **Morphometric validation: 54** — Computes 4 synthetic measurements in `measure_morphoeval()` (L163-171), but does not validate the 5-regime PD elongation and E11.5 aspect-ratio hierarchy against empirical benchmarks (Pearson r, RMSE), does not verify Reynolds-number insensitivity, and does not extract empirical `St0` vs `St1` contours or aspect ratios from `mouse_limb_bud_specimen.jpg`.  
  evidence: `2015-boehm-limb-bud-elongation-pushing.py` L163–171
- **Source and data traceability: 64** — Module docstring cites Boehm et al. (2010), Kicheva & Briscoe (2010), and Gros et al. (2010) DOIs (L11-13) and `datasets/limb_bud_elongation_pushing/` exists on disk, but the v1 script does not load `limb_bud_elongation_pushing_reference.json` or verify SHA-256 digests at runtime.  
  evidence: `2015-boehm-limb-bud-elongation-pushing.py` L11–14
- **Code-to-model traceability: 68** — Functions correspond to IddU/BrdU proliferation source rates and limb bud contour evolution, and `morphoeval/lean/loops/Loop_017_limb_bud_elongation_pushing.lean` proves 3 theorems, but v1 docstrings lack equation-by-equation mappings to Boehm et al. (2010) Eqs. 1-11 and Gros et al. (2010) Figs. 1-6.  
  evidence: `2015-boehm-limb-bud-elongation-pushing.py` L29–85; `morphoeval/lean/loops/Loop_017_limb_bud_elongation_pushing.lean` L1–145
- **Reproducibility package: 68** — Deterministic execution with `--self-test` and `measure_morphoeval()` (L163-177), but `self_test()` checks only 4 basic assertions without SHA-256 dataset verification, 5-regime hierarchy parity, or two-run bitwise determinism checks.  
  evidence: `2015-boehm-limb-bud-elongation-pushing.py` L173–177

Remediation:
- Implement a 2D/3D continuum viscoelastic/Stokes limb bud growth solver with artificial-compressibility pressure-Poisson coupling (`nabla^2 p = (nabla . v - s(x))/(beta * dtau)`), comparing Model I (isotropic proliferation gradient), Model II (Wnt5a/JNK oriented cell division/intercalation active stress + AER-FGF8/MAPK chemotactic velocity field), Wnt5a-/- bulbous mutant, uniform-Tc control, and FGF-only motility across E11.0-E11.25 (6h) and E9.5-E11.5 (`L_PD/W_AP > 2.1`), plus the 5-decade Reynolds number sweep (`Re in [10^-6, 10^-1]`) and Hooke-Jeeves inverse-optimization demonstration.
- Load and SHA-256-verify `datasets/limb_bud_elongation_pushing/limb_bud_elongation_pushing_reference.json` and `mouse_limb_bud_specimen.jpg`, segment the E11.0 (`St0`) and E11.25 (`St1`) OPT 3D iso-surfaces (Panels D, E), Dorsal/Posterior `St0`-in-`St1` overlays (Panel F), and 3D tetrahedral mesh (Panel G), and quantify empirical PD elongation and aspect-ratio changes.
- Validate the 5-regime PD elongation and E11.5 aspect-ratio hierarchy against Boehm et al. (2010) and Gros et al. (2010) empirical benchmarks (Pearson r, RMSE), expand `measure_morphoeval()` to 8 reference-anchored metrics, and expose full CLI flags.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Boehm et al. (2010) & Gros et al. (2010) vertebrate limb bud oriented cell division & Wnt5a/FGF8 pushing vs proliferation-gradient hypothesis
- **spec_fingerprint** (semantic model + inputs): `4c3f0cfc301e2c955c4f4f6c7d2e02e3996ca3b4cd571d212368d55baab36b29`
- `docs/media/limb_bud_elongation_pushing_comparison.png` — artifact_sha256 `bd430e6fba9f2cdc…`, 119,117 B, 1890×602 px

| measurement | value | reference | method |
|---|---:|---|---|
| Model I isotropic proliferation 6h PD elongation vs control | 9.6201 % | 10.1 | simulate_limb_bud_v1(zeta_ocd=0.0, chi_fgf=0.0) *(Isotropic proliferation gradient alone yields bulbous expansion (~10% PD growth))* |
| Model II directional Wnt5a/FGF8 pushing 6h PD elongation | 51.9466 % | 42.2 | simulate_limb_bud_v1(zeta_ocd=1.0, chi_fgf=1.0) *(Directional active stress + chemotactic velocity gradient drives PD outgrowth)* |
| Model II final PD/AP aspect ratio after 6h | 1.6858 L_PD / W_AP | 1.35 | x_final.max() / (y_final.max() - y_final.min()) *(Anisotropic PD/AP aspect ratio vs isotropic baseline)* |
| Distal-to-proximal net proliferation source ratio | 4.2899 s(1) / s(0) | 4.22 | compute_brdu_iddu_source_rate(x_pd_norm=[0, 1]) *(Empirical IddU/BrdU source ratio across PD axis)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) evaluation of `2015-boehm-limb-bud-elongation-pushing.py` prior to Pass 2 remediation in `2026-10-19-limb-bud-elongation-pushing-loop`.
