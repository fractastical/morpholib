# MorphoEval report — audit `2026-10-27-drosophila-planitibia-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `614364984b077999780547c54c33040355b80944` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila planitibia detail-tier Turing wavelength (5.13 µm, Holocene 0 Ma) (`models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py`) | 72 | 78 | 64 | 68 | 88 | 80 | 84 | 69.3 | 74.2 | 84.3 | 75.8 | **75.8** | mostly faithful |

## Findings and remediation

### Drosophila planitibia detail-tier Turing wavelength (5.13 µm, Holocene 0 Ma)  —  overall 75.8 (mostly faithful)

Pass 1 overall is the pre-correction rating. The Schnakenberg PDE rendered a 12-frame 800×600 GIF whose specimen CV similarity was already 87.6, but the dispersion self-test returned Pearson r = -0.125 because activator-only initial data mixed a complex long-wave pair into the log-slope. The diffusion-ratio sweep asserted the wrong monotonicity, and the specimen radial peak locked onto the vignette at 180 µm instead of the spiral-band ridge. Those three failures are the remediation list. Dataset SHA-256 pins, the 0 Ma chronoseries statement, and the Lean statements were already in place.

- **Model fidelity to source: 72** — The Jacobian, the Turing threshold, and the periodic Schnakenberg Euler step match Schnakenberg (1979) and Turing (1952). The Pass 1 dispersion comparison did not: seeding u = ε cos(kx), v = 0 produced Pearson r = -0.125 because modes 1–3 are a complex conjugate pair and the activator projection is not a pure exponential. Model fidelity stays at 72 until that probe is restricted to the real Turing band.  
  evidence: `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L95–149; `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L301–320; `morphoeval/lean/loops/Loop_3199_drosophila_planitibia_turing_wavelength_s0.lean` L39–47
- **Morphological realism / specimen match: 78** — The growth GIF is 800×600 and 12 frames, full-bleed, colored with the specimen pigment LUT, and scored CV similarity 87.6 against drosophila_planitibia_specimen.jpg. The Pass 1 carrier measurement nevertheless reported the vignette (180 µm) rather than the spiral ridge, so the specimen metric was not yet the band the figure claims to compare. Spot geometry is also not a pixel copy of the spiral.  
  evidence: `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L449–482; `docs/media/drosophila_planitibia_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_planitibia_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 64** — A diffusion-ratio sweep and an equal-diffusion ablation exist, but the Pass 1 monotonicity test included ratios whose sigma_max was still negative and asserted that lambda falls as d rises. The self-test raised AssertionError on that claim. Expressivity is not credited until the unstable branch is isolated and the measured direction is the one that is tested.  
  evidence: `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L229–298
- **Morphometric validation: 68** — Scale-bar detection (111 px = 100 µm), Otsu circularity, and a KS comparison were implemented, but the headline specimen wavelength was the 180 µm envelope and the dispersion Pearson r was -0.125, below the 0.98 gate. Quantitative validation is incomplete until both numbers are recomputed on the corrected definitions.  
  evidence: `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L383–447; `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L638–717
- **Source and data traceability: 88** — OpenAlex/Crossref resolved Turing 1952, Schnakenberg 1979 (10.1016/0022-5193(79)90042-0, not the non-resolving 90258-0 suffix), Edwards 2007, Kambysellis 1995 Current Biology (10.1016/s0960-9822(95)00229-6), Carson 1982, and Kondo & Miura 2010. Wikimedia Commons returned no D. planitibia file; the gallery JPEG is SHA-256 pinned. Slot 3199 is absent from campaign_100_slots.yaml, which is recorded in the scout manifest rather than papered over.  
  evidence: `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L83–92; `datasets/drosophila_planitibia_turing_wavelength_s0/drosophila_planitibia_turing_wavelength_s0_reference.json` L1–76; `datasets/drosophila_planitibia_turing_wavelength_s0/paper_citations.bib` L1–87
- **Code-to-model traceability: 80** — Docstring, math-kg loci, and Lean namespace MathOfBiology.Loop_3199 name the same wavelength, age, and diffusion-ratio integers. Pass 1 could not yet trace a passing dispersion fit to those declarations because the numerical probe disagreed with the symbol.  
  evidence: `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L1–42; `morphoeval/math_kg_loops/delphi10000-drosophila-planitibia-turing-wavelength-s0.yaml` L1–75
- **Reproducibility package: 84** — --self-test is deterministic (seed 3199), checks dataset hashes, and correctly failed closed on Pass 1 (Pearson r and sweep direction). Failure was reported rather than relaxed. A known-good invocation exists; the numerical gates were not yet green.  
  evidence: `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L720–753; `datasets/drosophila_planitibia_turing_wavelength_s0/README.md` L1–24

Remediation:
- Reseed each Turing-band Fourier mode on its leading real eigenvector, drop the complex long-wave pair, and require Pearson r > 0.98 against the discrete dispersion symbol. Pass 1 measured r = -0.125.
- Restrict the diffusion-ratio sweep to ratios with sigma_max > 0 and test the direction the fastest mode actually moves. Pass 1 asserted a decreasing lambda and the assertion failed.
- Measure the specimen spiral-band ridge above the vignette bins of the radial FFT. Pass 1 reported the envelope at 180 µm from a 111 px / 100 µm scale bar.
- Re-run image fitness, measure_outputs, and the knowledge-graph build on the corrected probe.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila planitibia detail-tier Turing wavelength (5.13 µm, Holocene 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `ae50fdfd76557b94257607e992d8d85891de9e571d0334c02e62f7bd23852ae1`
- **input_dataset_sha256**: `datasets/drosophila_planitibia_turing_wavelength_s0/README.md` (`5bb5fa3afb09444b…`), `datasets/drosophila_planitibia_turing_wavelength_s0/drosophila_planitibia_specimen.jpg` (`4e1280925b1e63af…`), `datasets/drosophila_planitibia_turing_wavelength_s0/drosophila_planitibia_turing_wavelength_s0_reference.json` (`5fa45424d1e6731e…`), `datasets/drosophila_planitibia_turing_wavelength_s0/paper_citations.bib` (`7e60555a08d73b71…`), `datasets/drosophila_planitibia_turing_wavelength_s0/scout_manifest.json` (`0045b52532f7ea49…`)
- `docs/media/drosophila_planitibia_turing_wavelength_s0_comparison.png` — artifact_sha256 `9d34dea3222423fb…`, 568,974 B, 1632×1008 px
- `docs/media/drosophila_planitibia_turing_wavelength_s0_growth.gif` — artifact_sha256 `94245e816e2126ef…`, 1,556,551 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing dispersion Pearson r (linear PDE vs discrete symbol) | 0.99999993 Pearson r | 1.0 | log-slope of cosine projections on integer modes versus eig(J(k)) of the 5-point Laplacian *(RMSE = 0.00007, p = 6.389e-39, N = 13 modes)* |
| linear-regime stripe wavelength | 5.0972 µm | 5.13 | parabolic interpolation of the measured σ(k) peak, scaled by dx so the analytic discrete k_star equals 5.13 µm *(continuous-symbol wavelength on the same grid is 5.1769 µm)* |
| specimen scale-bar spiral carrier wavelength | 67.5676 µm | 5.13 | radial FFT peak of the square crop, calibrated by the printed 100 µm bar (111 px) *(Otsu pigment mask on drosophila_planitibia_specimen.jpg)* |
| specimen largest-contour circularity | 0.6619 4πA/P² | 1.0 | Otsu dark-pigment mask, largest connected component, perimeter from one binary erosion *(pigment area fraction = 0.9996, coherency = 0.2287)* |
| specimen vs simulation radial-spectrum KS statistic | 0.719 KS distance | 0.0 | two-sample KS on inverse-CDF draws of the 48-bin radial power spectra *(p = 2.346e-107; simulation coherency = 0.8526)* |
| equal-diffusion ablation amplitude ratio | 45.7569 std(d=18.6) / std(d=1) | 1.0 | same noise seed integrated on the linearized Schnakenberg operator *(σ_max(d=18.6) = 0.1798; equal-diffusion probe σ = -0.1628)* |
| activator diffusivity ratio versus D. persimilis | 520.16 D_u(117 µm) / D_u(5.13 µm) | 1.0 | λ ∝ sqrt(D_u) at fixed ρ, a, b, and d; age delta is 0 Ma *(versus D. pseudoobscura the ratio is 493.8272; both horizons are 0 Ma)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 of loop 3199. Schnakenberg PDE, specimen LUT render, and dataset pins were in place. The dispersion correlation, the diffusion-ratio monotonicity test, and the specimen radial-peak definition were not.
