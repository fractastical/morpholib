# MorphoEval report — audit `2026-10-27-drosophila-planitibia-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `614364984b077999780547c54c33040355b80944` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-planitibia-turing-wavelength-s0-audit -> 2026-10-27-drosophila-planitibia-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila planitibia detail-tier Turing wavelength (5.13 µm, Holocene 0 Ma) (`models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py`) | 92 | 86 | 90 | 90 | 92 | 88 | 91 | 91.3 | 87.5 | 90.5 | 89.6 | **89.6** | faithful |

## Findings and remediation

### Drosophila planitibia detail-tier Turing wavelength (5.13 µm, Holocene 0 Ma)  —  overall 89.6 (faithful)

Pass 2 closes the three Pass 1 failures. Eigenvector seeding on the real Turing band lifts the dispersion correlation from r = -0.125 to r = 0.99999993 (RMSE 7.20e-05). The sweep is now restricted to ratios with sigma_max > 0, where lambda lengthens from about 4.7 µm to 6.0 µm as d goes from 12 to 40, and the equal-diffusion control decays (amplitude ratio 45.76, probe sigma negative). The specimen radial ridge, calibrated by the 111 px / 100 µm bar, is 67.57 µm and is reported as a different scale from lambda_c. GIF fitness CV similarity is 87.6 at 800×600 and 12 frames. Weighted overall is 89.6. Signing was skipped honestly because ~/.nanopub/profile.yml is absent.

- **Model fidelity to source: 92** — Pass 2 seeds modes 4–16, where the leading eigenvalue is real, on that eigenvector and fits the log-slope of the cosine projection. Pearson r = 0.99999993 against eig(J(k)) of the same 5-point Laplacian (RMSE 7.20e-05, 13 modes). Homogeneous state has trace < 0 and det > 0; nominal d = 18.6 is unstable and d = 1 is stable, matching the Lean trace-diffusion theorems. The score stays below 95 because the kinetics are generic Schnakenberg parameters placed at an assigned length, not a species-published parameter table.  
  evidence: `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L123–149; `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L155–226; `morphoeval/lean/loops/Loop_3199_drosophila_planitibia_turing_wavelength_s0.lean` L22–52
- **Morphological realism / specimen match: 86** — The animated GIF (12 frames, 800×600) is colored with a LUT sampled from the vendored micrograph and scores cv_sim_similarity 87.6 by the canonical Sobel-orientation, radial-FFT, and coherency comparison. The comparison figure shows the micrograph beside the spot field. Specimen-match is 86 rather than higher because the nonlinear pattern is a spot lattice, while the micrograph's coarse structure is a spiral whose ridge is 67.57 µm, deliberately not identified with 5.13 µm.  
  evidence: `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L449–482; `docs/media/drosophila_planitibia_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_planitibia_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 90** — Above onset, lambda(d) is strictly increasing on d in {12, 15, 18.6, 24, 32, 40}. Ratios with sigma_max <= 0 are stored and excluded from the test. The equal-diffusion ablation (d = 1) on the same noise family decays while d = 18.6 grows: final-std ratio 45.76, and the equal-diffusion probe eigenvalue is negative. Nominal d = 18.6 is marked on the sweep.  
  evidence: `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L268–298; `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L229–265
- **Morphometric validation: 90** — measure_morphoeval reports seven comparisons whose values are not copies of their references: dispersion r = 0.99999993 versus 1, linear-regime wavelength 5.0972 µm versus 5.13 µm, specimen spiral ridge 67.5676 µm versus 5.13 µm, Otsu circularity 0.6619 versus 1, radial KS 0.7190 versus 0, ablation ratio 45.7569 versus 1, and D_u ratio versus D. persimilis 520.1600 versus 1. Continuous-symbol wavelength on the same dx is 5.177 µm.  
  evidence: `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L638–717; `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L383–447
- **Source and data traceability: 92** — Citations resolve on OpenAlex or Crossref. The Kambysellis phylogeny is Current Biology 10.1016/s0960-9822(95)00229-6; the script and the reference JSON both refuse the unrelated PNAS DOI used by an older sibling slot. Specimen SHA-256 is 4e1280925b1e63afbca87b977aa9621a93f248ec4d3cce3f876b0e1d4c831cca. Scout records that campaign slot 3199 does not exist and that Commons returned no image.  
  evidence: `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L83–92; `datasets/drosophila_planitibia_turing_wavelength_s0/scout_manifest.json` L1–80; `datasets/drosophila_planitibia_turing_wavelength_s0/paper_citations.bib` L1–87
- **Code-to-model traceability: 88** — Each math-kg equation names a Lean declaration and a code range: Jacobian and threshold (L95–149, Lean traceDiffusion), fastest-mode wavelength (L155–226, lambdaDetailNm), and the ablation (L229–265). The docstring states the 5.13 µm scale and the 0 Ma age delta before any numerical result.  
  evidence: `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L1–42; `morphoeval/math_kg_loops/delphi10000-drosophila-planitibia-turing-wavelength-s0.yaml` L36–75
- **Reproducibility package: 91** — python3 models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py --self-test passes: r > 0.98, unstable sweep strictly increasing, ablation ratio > 5, equal-diffusion sigma < 0, numerical wavelength within 12% of 5.13 µm, GIF at least 10 frames and 480 px, age delta 0. Seed 3199. Dependencies are the repo scientific stack (numpy, scipy, matplotlib, Pillow, PyYAML).  
  evidence: `models/2026-delphi10000-drosophila-planitibia-turing-wavelength-s0.py` L720–800; `datasets/drosophila_planitibia_turing_wavelength_s0/README.md` L1–24

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

Pass 2. Eigenvector-seeded dispersion probe, unstable-branch sweep, and spiral-ridge specimen wavelength. Self-test passed. Nanopub signing was not available (no ~/.nanopub/profile.yml).
