# MorphoEval report — audit `2026-10-19-turing-audit`

Rubric **0.1.0** · checkpoint `8871db0` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasProbability` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for ProbKnow template conformance). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Turing (1952), Pearson (1993) & Kondo-Asai (1995) reaction-diffusion morphogenesis (`1952-turing-morpho.py`) | 54 | 14 | 38 | 30 | 45 | 45 | 61 | 48.7 | 20.0 | 51.4 | 38.0 | **38.0** | unfaithful |

## Findings and remediation

### Turing (1952), Pearson (1993) & Kondo-Asai (1995) reaction-diffusion morphogenesis  —  overall 38.0 (unfaithful)

Pass 1 (v1) dual-agent evaluation of `1952-turing-morpho.py` (Gemini 4 Argon + Claude Sonnet 5.5 High consensus: 39.6/100). The v1 script integrates the 2D Gray-Scott PDE across 4 hardcoded (F, k) regimes plus an equal-diffusion control and computes 2D FFT radial power spectra, but completely omits Turing's (1952) linear reaction-diffusion instability equations, dispersion relation sigma(k^2), and ring-of-20-cells model; omits Pearson's (1993) saddle-node (4(F+k)^2 = F) and Hopf bifurcation curves; uses a conceptually flawed trivial-state extinction control for Du=Dv; has zero biological specimen images or segmentation (no Kondo & Asai 1995 Pomacanthus stripe spacing or Y-junction analysis); and lacks a vendored `datasets/turing/` reference package with SHA-256 verification.

- **Model fidelity to source: 54** — Integrates the 2D periodic Gray-Scott PDE dU/dt = Du*Lap(U) - U*V^2 + F*(1-U), dV/dt = Dv*Lap(V) + U*V^2 - (F+k)*V (L162-221) with Du/Dv = 2.0. However, as identified by the independent second assessor (Claude Sonnet 5.5 High): (1) Turing (1952) is not implemented at all—there is no linearized 2-morphogen reaction-diffusion system, no dispersion relation sigma(k^2), no critical wavenumber k_c, and no ring-of-20-cells model; (2) Pearson's (1993) saddle-node bifurcation curve 4*(F+k)^2 = F and Hopf boundary are not computed, and 3 of the 4 patterned regimes lie past the saddle-node boundary where no nontrivial homogeneous steady state exists; and (3) the equal-diffusion control (Du=Dv=0.10, L147-155) halves Du and tests finite-amplitude front extinction toward the always-stable trivial state (1,0) rather than linear Turing instability around a homogeneous steady state.  
  evidence: `1952-turing-morpho.py` L100–156; `1952-turing-morpho.py` L162–221
- **Morphological realism / specimen match: 14** — No biological specimen photograph, segmentation, or comparison to Kondo & Asai (1995) Pomacanthus angelfish skin stripes or boxfish/pufferfish hexagonal spot arrays is present anywhere in the v1 script. Furthermore, in `turing_pearson_morphospace.png`, the first two spot panels (mitosis_spots at F=0.028 and soliton_spots at F=0.030) are visually near-indistinguishable at t=4000 because both are initialized with 10 nucleation patches rather than Pearson's single central 20x20 perturbation.  
  evidence: `1952-turing-morpho.py` L173–200; `turing_pearson_morphospace.png` L0–0
- **Parameter expressivity: 38** — Only 4 patterned (F, k) points plus 1 control are hardcoded in `PEARSON_REGIMES` (L100-156), omitting Pearson's (1993) 12 pattern classes (alpha through mu) and (F, k) phase diagram. No CLI arguments exist for F, k, Du, Dv, N, steps, dt, or seed (L561-580), and no continuous parameter sweep of diffusivity ratio d = Du/Dv or feed/kill rates is performed.  
  evidence: `1952-turing-morpho.py` L100–156; `1952-turing-morpho.py` L561–580
- **Morphometric validation: 30** — Computes 2D Hann-windowed FFT radial PSD, azimuthal m=2/m=6 harmonics, and connected-component Euler characteristic (L227-313), but all metrics are purely self-referential without comparison to analytical Turing wavelength lambda_c = 2*pi/k_c, Pearson's (1993) phase boundaries, or Kondo & Asai (1995) empirical stripe wavelengths. Moreover, integer-binned k* gives k* = 11, 12, 12, 11 cycles across all four patterned regimes, failing to resolve sub-bin wavelength shifts.  
  evidence: `1952-turing-morpho.py` L270–286; `turing_pearson_metrics.csv` L1–6
- **Source and data traceability: 45** — Module docstring lists DOIs for Turing (1952), Pearson (1993), and Lee et al. (1994) (L7-15), but `datasets/turing/` does not exist, no formal JSON/CSV reference dataset or specimen photographs are vendored or SHA-256-locked, and Kondo & Asai (1995) (`10.1038/376765a0`) is absent.  
  evidence: `1952-turing-morpho.py` L7–15
- **Code-to-model traceability: 45** — Section headers and docstrings describe the Gray-Scott PDE and FFT ring metrics (L160-313), but functions lack explicit equation, section, and figure mappings to Turing (1952) Sections 8-9, Pearson (1993) Eq. 1 & Fig. 1-2, and Kondo & Asai (1995) Fig. 1-3.  
  evidence: `1952-turing-morpho.py` L162–221
- **Reproducibility package: 61** — Uses deterministic `np.random.default_rng(1952)` and includes `--self-test`, `--no-show`, and `--skip-gif` flags (L511-591). However, `--self-test` uses overly loose thresholds (`ring_sharpness > 5.0` when observed values are 718-967), does not verify two-run hash determinism, and `render_legacy_neon_gif` writes `turing_patterns_black_neon.gif` (N=200, 5000 steps) out of sync with `docs/media/turing_gray_scott.gif`.  
  evidence: `1952-turing-morpho.py` L319–364; `1952-turing-morpho.py` L528–557

Remediation:
- Implement Turing (1952) Section 8-9 two-morphogen linear reaction-diffusion dispersion relation sigma(k^2), critical wavenumber k_c, analytical wavelength lambda_c = 2*pi/k_c, and Turing's N=20 ring-of-cells morphogenesis model.
- Implement Pearson (1993) exact Saddle-Node (4*(F+k)^2 = F) and Hopf bifurcation curves, map all 12 Pearson pattern classes (alpha through mu) on the (F, k) phase diagram, and replace the flawed trivial-state extinction control with a true diffusivity-ratio sweep (d = Du/Dv across the analytical Turing threshold d_c around a nontrivial steady state).
- Vendor `datasets/turing/` with SHA-256-locked reference data (Turing 1952 ring parameters, Pearson 1993 12-regime table, Kondo & Asai 1995 Pomacanthus stripe spacing table) plus real CC-licensed biological specimen photographs (Pomacanthus semicirculatus skin stripes & Tetraodon/Ostracion hexagonal spots), and segment the biological stripes, inter-stripe wavelength, and Y-branch dislocations directly in `1952-turing-morpho.py`.
- Upgrade spectral morphometrics to sub-bin parabolic/centroid wavenumber estimation, compare measured wavelengths against analytical lambda_c and empirical Pomacanthus stripe spacing, expose full CLI parameters, sync `docs/media/turing_gray_scott.gif`, and formalize Turing dispersion & Pearson saddle-node theorems in `morphoeval/lean/MathOfBiology.lean` Section 5.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Turing (1952), Pearson (1993) & Kondo-Asai (1995) reaction-diffusion morphogenesis
- **spec_fingerprint** (semantic model + inputs): `457ba1d8776c5109e7d828595ec1ea637e0d9b8eebcf725fc073d1f70566f695`
- `turing_pearson_morphospace.png` — artifact_sha256 `545bf3f8f507c4ca…`, 1,206,789 B, 2421×1240 px

| measurement | value | reference | method |
|---|---:|---|---|
| dominant wavelength (last frame) | 50.0 grid cells | — | peak of the radial power spectrum of the axes-box image, scaled to the N=200 grid *(Gray–Scott f=0.055,k=0.062 labyrinth; no specimen or literature wavelength recorded)* |
| pattern coverage (last frame) | 0.0059 fraction | — | fraction of axes box in the high-V (yellow) band |
| connected high-V components (last frame) | 101 count | — | scipy.ndimage.label on the yellow mask |
| fraction of pattern pixels changed, first→last frame | 0.0 fraction | >0 for an evolving simulation | XOR of yellow masks of frame 0 and the last frame *(≈0 means the GIF shows a frozen pattern)* |
| Pearson coral labyrinth (F=0.055, k=0.062) spectral ring wavenumber k* | 11.0 cycles / domain | >0 (Turing band-pass ring) | peak of 2D radial PSD P(k) on N=160x160 grid after 4000 steps *(λ* = 14.55 grid cells, ring sharpness = 718.1)* |
| Pearson mitosis spots (F=0.028, k=0.062) connected components vs coral labyrinth | 129 count | 14 | connected high-activator components in self-replicating spot regime *(spots remain disconnected (>100 components) whereas coral labyrinth percolates)* |
| equal-diffusion negative control (Du=Dv=0.10) spectral wavenumber k* | 0.0 cycles / domain | 0.0 | dominant wavenumber when differential diffusivity is removed (Du = Dv = 0.10 at F=0.055, k=0.062) *(spatial σ_V = 0.00e+00, ring sharpness = 0.0)* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 1 (v1) dual-agent evaluation of `1952-turing-morpho.py` prior to Pass 2 remediation in `2026-10-19-turing-morpho-loop`.
