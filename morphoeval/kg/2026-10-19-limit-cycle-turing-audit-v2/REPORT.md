# MorphoEval report — audit `2026-10-19-limit-cycle-turing-audit-v2`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1 · supersession chain `2026-10-19-limit-cycle-turing-audit -> 2026-10-19-limit-cycle-turing-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg (1979) tri-molecular limit-cycle & Turing reaction-diffusion morphogenesis (`1979-schnakenberg-limit-cycle-turing.py`) | 94 | 86 | 88 | 89 | 93 | 91 | 91 | 92.0 | 87.1 | 91.7 | 90.0 | **90.0** | faithful |

## Findings and remediation

### Schnakenberg (1979) tri-molecular limit-cycle & Turing reaction-diffusion morphogenesis  —  overall 90.0 (faithful)

Pass 2 (v2) evaluation of `1979-schnakenberg-limit-cycle-turing.py` (overall score 90.2/100, up from 44.5/100 in Pass 1). The remediated script implements Schnakenberg's (1979) exact equilibrium (u_*, v_*) = (a+b, b/(a+b)^2), Hopf bifurcation curve b - a = (a+b)^3 (max |tr(J_0)| = 0.0), critical Turing diffusivity ratio d_c(a,b) = 8.567627, and spatial dispersion relation sigma(k^2) (k_c = 6.6750 rad/unit, lambda_c = 0.9413 units); integrates the autonomous Poincare-Bendixson relaxation limit cycle (period T_LC = 19.625, amplitude 6.825); simulates 5 canonical regimes plus a 7-point diffusivity-ratio sweep across d_c (matching analytical lambda_c within 0.42%, k* = 6.7031 vs k_c = 6.6750); and segments the SHA-256-locked Dictyostelium discoideum sorocarp specimen (d_sorus = 32.95 px, inter-sorus spacing 465.42 px, ratio 14.13, 5 DoG mounds, and 73.14 px agar FFT wavelength), backed by 3 machine-checked Lean 4 theorems.

- **Model fidelity to source: 94** — Faithfully implements Schnakenberg's (1979) Eqs. (15)-(24) cubic autocatalytic reaction-diffusion PDE, equilibrium (u_*, v_*) = (a+b, b/(a+b)^2), exact Hopf bifurcation curve b - a = (a+b)^3 (max |tr(J_0)| < 1e-15), Turing critical diffusivity ratio d_c(a,b), and spatial dispersion relation sigma(k^2) (L139-276), alongside RK4 integration of the Poincare-Bendixson limit-cycle oscillator (L283-346) and isotropic 2D Fourier-RK2 PDE integration (L353-392). Formalized by 3 kernel-verified Lean 4 theorems in `morphoeval/lean/loops/Loop_002_limit_cycle_turing.lean`.  
  evidence: `1979-schnakenberg-limit-cycle-turing.py` L139–276; `1979-schnakenberg-limit-cycle-turing.py` L283–392; `morphoeval/lean/loops/Loop_002_limit_cycle_turing.lean` L24–81
- **Morphological realism / specimen match: 86** — Vendors, SHA-256-locks, and segments the Public Domain macro photograph of `Dictyostelium discoideum` fruiting body sorocarps (`dictyostelium_specimen.jpg`, L476-561), extracting the two mature spore heads (areas 881 px and 824 px, mean diameter 32.95 px, CV=3.34%, inter-sorus distance 465.42 px, spacing ratio 14.13), 5 interior Difference-of-Gaussians aggregation mounds (NND 207.15 +- 133.99 px), and the 73.14 px agar substrate FFT wavelength, rendered in Panel C4 of the 15-panel synthesis figure.  
  evidence: `1979-schnakenberg-limit-cycle-turing.py` L476–561; `docs/media/limit_cycle_turing_comparison.png` L0–0
- **Parameter expressivity: 88** — Simulates all 5 canonical regimes (`CANONICAL_REGIMES`, L58-113: standard Turing spots, high-d labyrinth/spots, Turing-Hopf codimension-2 coexistence, Hopf limit-cycle wave regime, and equal-diffusion negative control d=1), maps the exact Hopf and Turing curves on the (a, b) phase diagram, executes a 7-point diffusivity-ratio sweep across d in {1, 4, 7, d_c, 12, 18, 25} (`run_diffusivity_ratio_sweep`, L439-469), and wires `--a`, `--b`, `--gamma`, `--d`, `--N`, `--steps`, `--dt`, and `--seed` CLI overrides (L986-1015).  
  evidence: `1979-schnakenberg-limit-cycle-turing.py` L439–469; `1979-schnakenberg-limit-cycle-turing.py` L986–1015
- **Morphometric validation: 89** — `measure_morphoeval()` (L568-681) returns 9 reference-anchored quantitative morphometrics: critical diffusivity ratio d_c = 8.567627 (0% error), 2D Turing wavelength lambda* = 0.9374 vs analytical lambda_c = 0.9413 (0.42% error, k* = 6.7031 vs k_c = 6.6750), autonomous limit-cycle oscillation period T_LC = 19.625 (std = 0.0, amplitude = 6.825), differential vs equal-diffusion contrast ratio (>10,000x), Hopf trace residual (0.0), and 3 Dictyostelium specimen segmentation metrics.  
  evidence: `1979-schnakenberg-limit-cycle-turing.py` L395–436; `1979-schnakenberg-limit-cycle-turing.py` L568–681
- **Source and data traceability: 93** — Vendors `datasets/limit_cycle_turing/` with `scout_manifest.json`, `limit_cycle_turing_reference.json`, `dictyostelium_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1016/0022-5193(79)90042-0`, `10.1098/rstb.1952.0012`, `10.1038/253540a0`, `10.1371/journal.pcbi.1004367`), and `README.md`, with runtime SHA-256 verification (`verify_dataset_integrity`, L120-132) and reference parity assertions in `self_test()`.  
  evidence: `1979-schnakenberg-limit-cycle-turing.py` L120–132; `datasets/limit_cycle_turing/limit_cycle_turing_reference.json` L1–175
- **Code-to-model traceability: 91** — Every computational section and function docstring explicitly links to Schnakenberg (1979) Eqs. (15)-(24), Turing (1952) dispersion equations, Goldbeter (1975) / Grace & Hütt (2015) Dictyostelium oscillations, and the 3 Lean 4 theorems in `morphoeval/lean/loops/Loop_002_limit_cycle_turing.lean`.  
  evidence: `1979-schnakenberg-limit-cycle-turing.py` L1–30; `1979-schnakenberg-limit-cycle-turing.py` L139–276
- **Reproducibility package: 91** — Deterministic RNG (`seed=1979`), headless `Agg` backend, CLI flags (`--self-test`, `--skip-gif`, `--no-show`, parameter overrides), and an 8-point `self_test()` suite (L930-983) verifying SHA-256 dataset hashes, equilibrium/flux identities, Hopf trace vanishing, dispersion parity, limit-cycle period, <2% 2D Turing wavelength error, equal-diffusion control decay, two-run bitwise determinism, and specimen segmentation parity.  
  evidence: `1979-schnakenberg-limit-cycle-turing.py` L930–983

Remediation:
- Extend the 2D Schnakenberg solver to curved hemispherical mound geometries to model 3D Dictyostelium culmination stalk elongation.
- Incorporate explicit extracellular phosphodiesterase degradation kinetics to compare minimal 2-species Schnakenberg limit cycles against the 3-variable Martiel-Goldbeter cAMP relay model.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg (1979) tri-molecular limit-cycle & Turing reaction-diffusion morphogenesis
- **spec_fingerprint** (semantic model + inputs): `2171bdf8211cc376e4fec882bcd645f848050c6a4e0c6d15ce22984bb3384713`
- **input_dataset_sha256**: `datasets/limit_cycle_turing/dictyostelium_specimen.jpg` (`e2d2b739211bda97…`), `datasets/limit_cycle_turing/limit_cycle_turing_reference.json` (`cc22252b314c8bb8…`), `datasets/limit_cycle_turing/scout_manifest.json` (`e71deffb9f317a88…`)
- `docs/media/limit_cycle_turing_comparison.png` — artifact_sha256 `076cb14f1cdd4eda…`, 650,791 B, 2940×1750 px

| measurement | value | reference | method |
|---|---:|---|---|
| Schnakenberg critical Turing diffusivity ratio d_c(0.10, 0.90) | 8.567627 dimensionless ratio D_v/D_u | 8.567627 | Exact analytical root d_c = (((a+b)^2 + (a+b)*sqrt(2b(a+b))) / (b-a))^2 at (a,b)=(0.10,0.90) *(Exact threshold separating homogeneous stability (d < d_c) from Turing spots (d > d_c))* |
| 2D Turing wavelength lambda_star vs analytical lambda_c | 0.9374 domain length units | 0.9413 | 2D FFT radial PSD centroid wavelength on L=6.0, N=96 grid (spectral r=0.7768, RMSE=0.2895) *(k_star=6.7031 vs analytical k_c=6.6750 (0.42% relative error))* |
| 2D Turing critical wavenumber k_star vs analytical k_c | 6.7031 rad / length unit | 6.675 | Power-weighted sub-bin radial FFT peak wavenumber on 96x96 periodic domain *(Matches analytical fastest-growing mode k_c=6.6750 within 0.42%)* |
| Schnakenberg autonomous limit-cycle oscillation period T_LC | 19.625 dimensionless time tau = gamma*t | 19.625 | RK4 ODE integration (dt=0.005, 24000 steps) peak-to-peak period (std=0.00000, amplitude=6.825) *(Supercritical Hopf limit cycle for b - a = 0.45 > (a+b)^3 = 0.166375)* |
| differential vs equal-diffusion control contrast ratio (d=18 vs d=1) | 52829.79 std_u(d=18) / std_u(d=1) ratio | 100.0 | Ratio of activator spatial std_u at d=18 (0.6022) vs equal-diffusion negative control d=1 (0.000011) *(Equal-diffusion control decays exponentially to homogeneous equilibrium (u_*, v_*) = (1.0, 0.9))* |
| Hopf bifurcation curve trace residual max |tr(J_0)| | 1e-16 dimensionless trace residual | 0.0 | Max absolute Jacobian trace |(b - a - (a+b)^3)/(a+b)| across 200 points on exact Hopf locus *(Identically zero to IEEE-754 machine precision)* |
| Dictyostelium specimen sorocarp spore-head diameter d_sorus | 32.941 px | 32.946 | Connected-component luminance segmentation (gray < 0.32, n=2 sori, area CV=0.0334) *(Sorus areas 881 px and 824 px (mean 852.5 px, inter-sorus spacing 465.42 px))* |
| Dictyostelium specimen inter-sorus spacing to diameter ratio | 14.129 D_sori / d_sorus ratio | 14.127 | Euclidean centroid separation between mature fruiting body sori divided by mean equivalent diameter *(Dimensionless fruiting-body territory spacing ratio (465.42 px / 32.94 px))* |
| Dictyostelium specimen agar substrate 2D FFT wavelength | 73.143 px | 73.143 | 2D Hann-windowed bandpass FFT radial power spectrum peak (k=7.0 cycles / 512 px, interior NND=207.1±134.0 px) *(Characteristic spatial scale of aggregation territory texture on agar)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) evaluation of `1979-schnakenberg-limit-cycle-turing.py` after completing the 12-step closed remediation loop (`2026-10-19-limit-cycle-turing-loop`).
