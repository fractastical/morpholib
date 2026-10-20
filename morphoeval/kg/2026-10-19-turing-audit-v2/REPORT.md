# MorphoEval report — audit `2026-10-19-turing-audit-v2`

Rubric **0.1.0** · checkpoint `b9976f8` · assessor `pkr:model/gemini-4-argon` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the ProbKnow rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Turing (1952), Pearson (1993) & Kondo-Asai (1995) reaction-diffusion morphogenesis (`1952-turing-morpho.py`) | 92 | 82 | 81 | 75 | 94 | 89 | 89 | 88.3 | 79.4 | 90.7 | 85.5 | **85.5** | faithful |

## Findings and remediation

### Turing (1952), Pearson (1993) & Kondo-Asai (1995) reaction-diffusion morphogenesis  —  overall 85.5 (faithful)

Pass 2 (v2) dual-agent evaluation of `1952-turing-morpho.py` (Gemini 4 Argon 91.9/100 + Claude Sonnet 5.5 High 79.1/100; consensus 85.5/100, up from 38.0/100 in v1). The remediated script implements Turing's (1952) Sections 8-10 two-morphogen linear dispersion relation sigma(k^2) and N=20 ring-of-cells morphogenesis (mode s*=4, lambda_c=5.62 cells); computes Pearson's (1993) exact Saddle-Node (4*(F+k)^2=F) and Hopf (max |tr(J_0)| = 2.36e-16) bifurcation curves with all 12 pattern classes (alpha through mu); pairs a genuine steady-state Turing labyrinth (F=0.054, k=0.0616, Du=0.20, Dv=0.10, sigma=+0.00164) with an equal-diffusion control (Du=Dv=0.20, std_V=1.91e-7) around the same nontrivial steady state; and segments SHA-256-locked photographs of Pomacanthus imperator skin stripes (spacing 9.55 px, CV=0.149, 22 Y-junctions) and Ostracion meleagris hexagonal spots (121 spots) alongside a white-noise-initialized 1D PDE test of Kondo & Asai's (1995) stripe-insertion law (r=0.9981).

- **Model fidelity to source: 92** — Implements both Turing's (1952) Sections 8-10 linearized two-morphogen reaction-diffusion dispersion relation sigma(k^2) + discrete N=20 ring-of-cells model (L163-244, d_c=3.927051, k_c=1.118034 rad/cell, dominant mode s*=4) and Pearson's (1993) 2D periodic Gray-Scott PDE with exact Saddle-Node (4*(F+k)^2 = F) and Hopf (F_Hopf(k) where max |tr(J_0)| = 2.36e-16) bifurcation curves (L327-426). The equal-diffusion negative control (Du=Dv=0.20 at F=0.054, k=0.0616) is initialized around the exact nontrivial homogeneous steady state (U_*, V_*) and verified against the supercritical Turing state (Du=0.20, Dv=0.10, sigma=+0.00164). Backed by 4 Lean 4 theorems in `MathOfBiology.lean` Section 5.  
  evidence: `1952-turing-morpho.py` L163–244; `1952-turing-morpho.py` L327–426; `morphoeval/lean/MathOfBiology.lean` L195–283
- **Morphological realism / specimen match: 82** — Vendors and SHA-256-verifies real CC BY-SA 4.0 biological specimen photographs of adult `Pomacanthus imperator` skin stripes (`pomacanthus_imperator_stripes_specimen.jpg`) and `Ostracion meleagris` boxfish spots (`ostracion_meleagris_spots_specimen.jpg`), segments the yellow/blue angelfish stripes and medial-axis Y-branch dislocations (mean inter-stripe spacing 9.55 px, CV=0.149, 22 macroscopic Y-junctions) and 121 boxfish Turing spots (L706-845), and differentiates Pearson's single-patch self-replicating spot regime (A1, `pearson_center`) from domain-filling quasi-hexagonal spots (A2), labyrinthine stripes (A3), and steady-state Turing labyrinth (A4).  
  evidence: `1952-turing-morpho.py` L706–845; `turing_pearson_morphospace.png` L0–0
- **Parameter expressivity: 81** — Maps all 12 Pearson (1993) pattern classes (alpha through mu) onto the (k, F) bifurcation diagram (Panel C2), simulates 5 distinct initialization/parameter regimes in `PEARSON_REGIMES`, executes a 5-point diffusivity-ratio sweep across d = Du/Dv in {1.0, 1.25, 1.5, 2.0, 2.5} (`run_diffusivity_ratio_sweep`, L515-536) spanning the analytical Turing threshold, and wires `--F`, `--k`, `--Du`, `--Dv`, `--N`, `--steps`, and `--seed` CLI arguments into the solver (L939-954, L1385-1425).  
  evidence: `1952-turing-morpho.py` L515–536; `1952-turing-morpho.py` L939–954
- **Morphometric validation: 75** — Upgrades 2D FFT radial power spectrum estimation with 3-point parabolic sub-bin interpolation (`_parabolic_subbin_peak`, L539-550) resolving distinct wavelengths across all four patterned regimes (k* = 10.89, 12.10, 11.89, 11.50 cycles/box; lambda* = 14.69, 13.23, 13.46, 13.92 px), compares the steady-state Turing labyrinth wavelength lambda* = 13.92 px directly against analytical lambda_c = 11.92 px (16.7% difference), quantifies Pomacanthus inter-stripe spacing (9.55 px, CV = 0.149) and Ostracion nearest-neighbor spot spacing (21.5 px), and runs a white-noise-initialized 1D Turing PDE across growing domains (`r = 0.9981` against Kondo & Asai 1995 empirical stripe counts, L672-703).  
  evidence: `1952-turing-morpho.py` L539–646; `1952-turing-morpho.py` L672–703
- **Source and data traceability: 94** — Vendors `datasets/turing/` with `turing_reference.json`, `pomacanthus_imperator_stripes_specimen.jpg`, `ostracion_meleagris_spots_specimen.jpg`, `paper_citations.bib` (4 primary DOIs: `10.1098/rstb.1952.0012`, `10.1126/science.261.5118.189`, `10.1038/369215a0`, `10.1038/376765a0`), and `datasets/turing/README.md`, with runtime SHA-256 cryptographic verification (`EXPECTED_SHA256`, L91-125) and JSON-to-code parity assertions in `self_test()` (L1288-1308).  
  evidence: `1952-turing-morpho.py` L91–125; `datasets/turing/turing_reference.json` L1–145
- **Code-to-model traceability: 89** — Every computational section and function docstring maps directly to numbered sections, equations, and figures in Turing (1952) Sections 8-10, Pearson (1993) Eq. 1 & Figs. 1-2, Lee et al. (1994) Fig. 2, and Kondo & Asai (1995) Figs. 1-2, as well as the 4 Lean 4 theorems in `morphoeval/lean/MathOfBiology.lean` Section 5.  
  evidence: `1952-turing-morpho.py` L18–66
- **Reproducibility package: 89** — Deterministic RNG (`seed=1952`), pinned `requirements.lock.txt`, comprehensive `--self-test` suite verifying Turing (1952) dispersion + JSON parity, exact Hopf curve `|tr(J_0)| < 1e-12`, diffusivity-ratio sweep monotonicity, sub-bin spectral rings, analytical vs. nonlinear wavelength agreement, biological specimen segmentation, and two-run bitwise determinism (`np.allclose(..., atol=0.0)`), plus byte-synchronized animated GIF outputs at `docs/media/turing_gray_scott.gif` and `turing_patterns_black_neon.gif`.  
  evidence: `1952-turing-morpho.py` L1263–1383

Remediation:
- Calibrate a 2D growing-domain PDE simulation with adiabatic domain expansion L(t) to observe spontaneous Y-branch stripe insertion events in 2D space-time.
- Add manual ground-truth landmark annotations for Pomacanthus imperator Y-junction coordinates to benchmark automated skeleton junction recall and precision.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Turing (1952), Pearson (1993) & Kondo-Asai (1995) reaction-diffusion morphogenesis
- **spec_fingerprint** (semantic model + inputs): `457ba1d8776c5109e7d828595ec1ea637e0d9b8eebcf725fc073d1f70566f695`
- `turing_pearson_morphospace.png` — artifact_sha256 `4dce6f8427cdf23e…`, 1,674,289 B, 2617×1806 px

| measurement | value | reference | method |
|---|---:|---|---|
| dominant wavelength (last frame) | 40.0 grid cells | — | peak of the radial power spectrum of the axes-box image, scaled to the N=200 grid *(Gray–Scott f=0.055,k=0.062 labyrinth; no specimen or literature wavelength recorded)* |
| pattern coverage (last frame) | 0.0095 fraction | — | fraction of axes box in the high-V (yellow) band |
| connected high-V components (last frame) | 560 count | — | scipy.ndimage.label on the yellow mask |
| fraction of pattern pixels changed, first→last frame | 0.0 fraction | >0 for an evolving simulation | XOR of yellow masks of frame 0 and the last frame *(≈0 means the GIF shows a frozen pattern)* |
| Pearson coral labyrinth (F=0.054, k=0.0616) spectral ring wavenumber k* | 11.487 cycles / domain | >0 (Turing band-pass ring) | sub-bin parabolic peak of 2D radial PSD P(k) on N=160x160 grid *(λ* = 13.93 grid cells (analytical λ_c = 11.92), ring sharpness = 968.1)* |
| Pearson mitosis spots (F=0.028, k=0.062) connected components vs coral labyrinth | 52 count | 1 | connected high-activator components in self-replicating spot regime *(spots remain disconnected whereas coral labyrinth percolates)* |
| steady-state equal-diffusion negative control (Du=Dv=0.20) spectral wavenumber k* | 0.0 cycles / domain | 0.0 | dominant wavenumber when differential diffusivity is removed around nontrivial steady state (U_*, V_*) *(spatial σ_V = 4.70e-07, ring sharpness = 0.0)* |
| Turing (1952) Sec. 8-9 analytical critical wavelength λ_c = 2π/k_c | 5.62 cells | 5.62 | 2*pi / sqrt((d*D_X + a*D_Y) / (2*D_X*D_Y)) on 2-morphogen activator-inhibitor system *(d = 4.00 > d_c = 3.927, N=20 ring dominant mode s* = 4)* |
| Pomacanthus imperator segmented skin stripe wavelength | 9.55 px | 8–16 px (CV < 0.35) | mean dorsoventral inter-stripe peak spacing on SHA-256-locked specimen ROI *(CV = 0.149, topological Y-branch junctions = 22, ontogenetic r = 0.9981)* |
| Ostracion meleagris segmented Turing spots (DoG + Delaunay) | 121 spots | >=15 hexagonal spots | Difference-of-Gaussians segmented spots on SHA-256-locked boxfish specimen ROI *(mean Delaunay nearest-neighbor spacing = 10.71 px (CV = 0.394))* |

## Environment at checkpoint

- python: 3.14.7
- numpy: 2.4.2
- scipy: 1.17.0
- matplotlib: 3.10.8
- Pillow: 12.1.1

Pass 2 (v2) dual-agent evaluation of `1952-turing-morpho.py` after completing the 12-step closed remediation loop (`2026-10-19-turing-morpho-loop`).
