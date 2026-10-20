# MorphoEval report — audit `2026-10-19-limit-cycle-turing-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg (1979) tri-molecular limit-cycle & Turing reaction-diffusion morphogenesis (`1979-schnakenberg-limit-cycle-turing.py`) | 52 | 18 | 35 | 34 | 48 | 52 | 64 | 46.3 | 24.0 | 55.5 | 40.1 | **40.1** | partially faithful |

## Findings and remediation

### Schnakenberg (1979) tri-molecular limit-cycle & Turing reaction-diffusion morphogenesis  —  overall 40.1 (partially faithful)

Pass 1 (v1) baseline evaluation of `1979-schnakenberg-limit-cycle-turing.py` (overall score 44.5/100). The v1 script implements the homogeneous equilibrium (u_*, v_*) = (a+b, b/(a+b)^2), 2x2 reaction Jacobian J_0, and a 5-point finite difference 2D solver at a single Turing point (a=0.10, b=0.90, d=18.0), but omits Schnakenberg's (1979) Hopf bifurcation curve b - a = (a+b)^3 and autonomous limit-cycle oscillator, omits the spatial dispersion relation sigma(k^2) and critical diffusivity ratio d_c(a,b), lacks an equal-diffusion negative control or diffusivity sweep, and performs zero biological specimen segmentation on `datasets/limit_cycle_turing/dictyostelium_specimen.jpg`.

- **Model fidelity to source: 52** — Implements Schnakenberg's (1979) cubic autocatalytic PDE du/dt = gamma*(a - u + u^2*v) + Lap(u), dv/dt = gamma*(b - u^2*v) + d*Lap(v) and homogeneous equilibrium (u_*, v_*) = (a+b, b/(a+b)^2) (L26-66). However, Schnakenberg's central Hopf bifurcation locus b - a = (a+b)^3, Poincare-Bendixson limit-cycle ODE oscillator, spatial dispersion relation sigma(k^2), critical diffusivity ratio d_c(a,b), and equal-diffusion negative control (d=1) are all absent in v1.  
  evidence: `1979-schnakenberg-limit-cycle-turing.py` L26–66
- **Morphological realism / specimen match: 18** — `datasets/limit_cycle_turing/dictyostelium_specimen.jpg` is not loaded or segmented in v1, and `docs/media/limit_cycle_turing_comparison.png` displays only a 1x2 synthetic activator/substrate heatmap without biological specimen overlay, sorocarp head/aggregate segmentation, or limit-cycle phase portrait.  
  evidence: `1979-schnakenberg-limit-cycle-turing.py` L99–126; `docs/media/limit_cycle_turing_comparison.png` L0–0
- **Parameter expressivity: 35** — Hardcodes a single parameter point (a=0.10, b=0.90, gamma=180, d=18.0) on a 64x64 grid (L41-66) without exposing CLI flags (--a, --b, --gamma, --d, --N, --steps, --dt, --seed) and without sweeping across the Hopf boundary b - a = (a+b)^3 or Turing threshold d_c = 8.5676.  
  evidence: `1979-schnakenberg-limit-cycle-turing.py` L41–66; `1979-schnakenberg-limit-cycle-turing.py` L137–146
- **Morphometric validation: 34** — `measure_morphoeval()` (L69-96) checks analytical equilibrium values (u_*=1.0, v_*=0.9, det(J_0)=1.0) and raw activator standard deviation std_u, but computes no 2D FFT radial power spectrum, no sub-bin wavenumber peak k* vs analytical k_c, no limit-cycle oscillation period T_LC, and no specimen nearest-neighbor or sorocarp morphometrics.  
  evidence: `1979-schnakenberg-limit-cycle-turing.py` L69–96
- **Source and data traceability: 48** — Cites Schnakenberg (1979) (`10.1016/0022-5193(79)90042-0`) in the module docstring (L6-9), but does not load or SHA-256-verify `datasets/limit_cycle_turing/limit_cycle_turing_reference.json`, `scout_manifest.json`, or `dictyostelium_specimen.jpg` at runtime.  
  evidence: `1979-schnakenberg-limit-cycle-turing.py` L6–9
- **Code-to-model traceability: 52** — Basic docstrings identify the equilibrium and Jacobian formulas (L26-39), but lack detailed equation-by-equation mappings to Schnakenberg (1979) Eqs. (15)-(24), Turing (1952) dispersion theory, and the 3 Lean 4 theorems in `morphoeval/lean/loops/Loop_002_limit_cycle_turing.lean`.  
  evidence: `1979-schnakenberg-limit-cycle-turing.py` L26–66
- **Reproducibility package: 64** — Uses deterministic `np.random.default_rng(1979)` and includes `--self-test` and `--skip-gif` (L129-146), but `self_test()` checks only 3 basic assertions without verifying two-run bitwise determinism, Hopf trace vanishing, dispersion thresholds, or dataset SHA-256 integrity.  
  evidence: `1979-schnakenberg-limit-cycle-turing.py` L129–135

Remediation:
- Implement Schnakenberg's (1979) exact Hopf bifurcation curve b - a = (a+b)^3, Turing instability boundary d_c(a,b), spatial dispersion relation sigma(k^2), critical wavenumber k_c, and autonomous Poincare-Bendixson limit-cycle ODE integration with period T_LC and phase-plane limit cycle vs stable spiral comparison.
- Simulate all 5 canonical regimes (turing_spots_standard, turing_labyrinth_high_d, turing_hopf_codim2, hopf_limit_cycle, equal_diffusion_control) plus a 7-point diffusivity-ratio sweep across d in {1, 4, 7, d_c, 12, 18, 25} and expose full CLI flags (--a, --b, --gamma, --d, --N, --steps, --dt, --seed).
- Load and SHA-256-verify `datasets/limit_cycle_turing/{scout_manifest.json,limit_cycle_turing_reference.json,dictyostelium_specimen.jpg}` at runtime, and implement computer-vision segmentation of `dictyostelium_specimen.jpg` (sorocarp spore-head sori area/diameter parity, inter-sorus spacing, interior DoG aggregation centers & NND distribution, and 2D FFT agar wavelength).
- Upgrade `measure_morphoeval()` and `--self-test` with sub-bin parabolic 2D FFT wavenumber peak estimation vs analytical k_c, limit-cycle period verification, equal-diffusion ablation ratio, specimen segmentation metrics, and two-run bitwise determinism.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg (1979) tri-molecular limit-cycle & Turing reaction-diffusion morphogenesis
- **spec_fingerprint** (semantic model + inputs): `e7a48d2f9b6eadc84256e32af6d90edcc5a9823751407e82e8ce1b2293da2e1e`
- **input_dataset_sha256**: `datasets/limit_cycle_turing/dictyostelium_specimen.jpg` (`e2d2b739211bda97…`), `datasets/limit_cycle_turing/limit_cycle_turing_reference.json` (`d46a92b16200c35b…`), `datasets/limit_cycle_turing/scout_manifest.json` (`e71deffb9f317a88…`)
- `docs/media/limit_cycle_turing_comparison.png` — artifact_sha256 `eff14c72d4713c77…`, 30,690 B, 1170×520 px

| measurement | value | reference | method |
|---|---:|---|---|
| homogeneous equilibrium activator u_star | 1.0 dimensionless | 1.0 | Analytical equilibrium u_* = a + b at (a, b) = (0.10, 0.90) *(Schnakenberg (1979) Eq. 20 baseline equilibrium check)* |
| homogeneous equilibrium substrate v_star | 0.9 dimensionless | 0.9 | Analytical equilibrium v_* = b / (a + b)^2 at (a, b) = (0.10, 0.90) *(Schnakenberg (1979) Eq. 20 baseline substrate check)* |
| reaction Jacobian determinant det(J_0) | 1.0 dimensionless | 1.0 | det(J_0) = (a + b)^2 evaluated at (a, b) = (0.10, 0.90) *(Positive Jacobian determinant at Turing base state)* |
| Turing activator spatial std_u (d=18) | 2.9094 concentration units (std) | — | Standard deviation of 2D activator field u(x,y) after 1200 steps *(Pass 1 uncalibrated spatial amplitude std)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) baseline evaluation of `1979-schnakenberg-limit-cycle-turing.py` prior to Step 12 remediation in `2026-10-19-limit-cycle-turing-loop`.
