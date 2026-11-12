# MorphoEval report — audit `2026-10-27-cavia-porcellus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `cc4b7fb74d981b19349d4808476e5b8b2b12f75a` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cavia porcellus Holocene (0 Ma) Turing wavelength at 126.7 µm (tier-5 detail s0) (`models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py`) | 86 | 80 | 68 | 78 | 86 | 80 | 84 | 80.0 | 79.2 | 83.6 | 80.8 | **80.8** | faithful |

## Findings and remediation

### Cavia porcellus Holocene (0 Ma) Turing wavelength at 126.7 µm (tier-5 detail s0)  —  overall 80.8 (faithful)

Pass 1 scores the dispersion law, the spiral render, the epsilon and substrate sweeps, the ablation, and the specimen scale-bar segmentation. It does not credit the charge sweep or the seed ensemble; those are remediation for the v2 audit. Analytical lambda_c = 126.7 µm, numerical eigenvalue Pearson r = 1 (roundoff RMSE = 0), simulated radial peak 128.55 µm, gallery-field wavelength 78.18 µm.

- **Model fidelity to source: 86** — The script implements the Schnakenberg steady state and Jacobian (a = 0.08, b = 1.15), the Turing wavenumber k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v), and the Swift-Hohenberg amplitude equation at that k_c. Homogeneous trace is negative, h_min is negative, and sigma(k_c) = 0.301 while sigma(0) = -0.321. The picture is the amplitude equation, not a nonlinear two-species PDE, so this is not a line-by-line Murray simulation.  
  evidence: `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L94–161; `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L251–312; `morphoeval/lean/loops/Loop_3141_cavia_porcellus_turing_wavelength_s0.lean` L18–40
- **Morphological realism / specimen match: 80** — The GIF is a 12-frame 640x480 lavender spiral whose last frame scores cv_sim_similarity 88.2 against the gallery pigment field. The same field, calibrated by its 110 px / 100 µm bar, has a dominant wavelength of 78.18 µm, not 126.7 µm. Pass 1 therefore does not treat the visual match as a wavelength validation. The gallery image is not a DOI histology plate.  
  evidence: `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L397–489; `docs/media/cavia_porcellus_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 68** — Epsilon from -0.20 to 0.45 raises saturated RMS from 0.0001 to 0.524. Holding D fixed, lambda_c(b) moves from 176.95 µm at b = 0.20 to 126.7 µm at b = 1.15, and the mode is no longer unstable at b = 2.5 and b = 4 (h_min > 0). The plain-diffusion ablation shifts the radial peak from 0.986 to 0.261. Charge and seed variation are not credited on this pass.  
  evidence: `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L184–203; `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L337–362
- **Morphometric validation: 78** — Closed-form sigma(k) matches numpy.linalg.eigvals with Pearson r = 1 and RMSE = 0; that is an eigensolver identity, not an independent time-integration residual, and Pass 1 does not treat it as one. The gallery wavelength 78.18 µm, coherency 0.198, orientation KS 0.077, and simulated peak 128.55 µm are reported against references they do not copy.  
  evidence: `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L228–249; `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L598–700
- **Source and data traceability: 86** — Crossref-confirmed DOIs for Turing 1952, Murray 1981, Schnakenberg 1979, Swift & Hohenberg 1977, Cross & Hohenberg 1993, and Sick et al. 2006 are in paper_citations.bib and the reference JSON. The specimen SHA-256 is 42014eaf5b445ec569739b4b14be9f65dea72da7126e6dc92702e66dbbbf2acd, the gallery file. OpenAlex returned no morphogenesis hits for the guinea-pig query; that empty result is kept in the scout manifest.  
  evidence: `datasets/cavia_porcellus_turing_wavelength_s0/cavia_porcellus_turing_wavelength_s0_reference.json` L1–40; `datasets/cavia_porcellus_turing_wavelength_s0/paper_citations.bib` L1–40
- **Code-to-model traceability: 80** — The module docstring, the math-KG loop YAML, and Loop_3141 name the same k_c formula and the 10x tier relation (1267 µm = 10 x 126.7 µm, 126.7 µm = 10 x 12.67 µm). The Lean theorems encode those integer relations and the sign of the cross term. They were not machine-checked in this image.  
  evidence: `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L1–48; `morphoeval/math_kg_loops/delphi10000-cavia-porcellus-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 84** — --self-test checks the SHA-256 pin, the stability signs, Pearson r > 0.98, the ablation shift, the decay ratio, GIF frame count (>= 10) and minimum side (>= 480). Seed 3141 and the grid (128, dt = 0.10, 480 steps) are fixed. Signing keys are absent, and Lean was not run.  
  evidence: `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L759–812; `docs/media/cavia_porcellus_turing_wavelength_s0_comparison.png` L0–0

Remediation:
- Credit the topological-charge sweep (m = 1, 2, 3) only after reporting that all three charges fall in the same radial bin k = 0.9856 (lambda = 128.55 µm).
- Report the four-seed wavenumber ensemble. At this bin width the CV is 0 because every seed shares that bin; do not describe it as a sub-bin robustness proof.
- State in the morphometric rationale that Pearson r = 1 and RMSE = 0 are the eigensolver identity of the quadratic formula, not an independent integrator residual.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=2.5]** `parameter_sweep` → `parameter-expressivity`: Score the charge sweep and the seed-ensemble CV, including the bin-width caveat.
- **[OPEN · P=2.5]** `morphometric_validation` → `morphometric-validation`: Label the Pearson/RMSE pair as an eigensolver identity.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cavia porcellus Holocene (0 Ma) Turing wavelength at 126.7 µm (tier-5 detail s0)
- **spec_fingerprint** (semantic model + inputs): `6023b337ad820eb039944062d27303cbeeca1ff69b22a1423b528112c7a5bbc6`
- **input_dataset_sha256**: `datasets/cavia_porcellus_turing_wavelength_s0/cavia_porcellus_turing_wavelength_s0_reference.json` (`6cf1e5c9b49da4a2…`), `datasets/cavia_porcellus_turing_wavelength_s0/cavia_porcellus_turing_wavelength_s0_specimen.jpg` (`42014eaf5b445ec5…`)
- `docs/media/cavia_porcellus_turing_wavelength_s0_comparison.png` — artifact_sha256 `ecf63291fef200a1…`, 567,960 B, 1344×1008 px
- `docs/media/cavia_porcellus_turing_wavelength_s0_growth.gif` — artifact_sha256 `0057ed24394e027d…`, 2,459,286 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytical Turing wavelength lambda_c of the s0 Schnakenberg mode | 126.7 µm | 126.7 | k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v); lambda_c = 2 pi / k_c after diffusivity scaling *(D_u = 169.1753 µm^2, D_v/D_u = 40.0, h_min = -5.4109)* |
| specimen pigment-field dominant wavelength from the 100 µm scale bar | 78.1818 µm | 126.7 | radial FFT peak of the caption-cropped gallery field, converted with the detected 100 µm bar *(scale bar 110 px = 100 µm (0.9091 µm/px); foreground fraction 0.6287)* |
| specimen structure-tensor coherency of the pigment field | 0.1983 dimensionless | 0.5 | mean (λ1-λ2)/(λ1+λ2) of the Gaussian-smoothed structure tensor inside the foreground mask *(contour circularity 0.0263 (far from a round blob))* |
| dispersion-curve Pearson r of analytical sigma(k) vs numerical eigenvalues | 1.0 Pearson r | 0.98 | 28 wavenumbers, closed-form quadratic versus numpy.linalg.eigvals of the Fourier-mode Jacobian *(RMSE = 0.000000; KS statistic between the two sigma samples = 0.0357)* |
| dispersion RMSE between analytical and numerical eigenvalue growth rates | 0.0 1/time | 0.05 | root mean square of sigma_closed_form(k) - sigma_eigvals(k) on k in [0.01, 0.12] µm^-1 *(sigma(k_c) = 0.3006, sigma(0) = -0.3215 (homogeneous state stable))* |
| simulated stripe wavelength on the eight-wave primordium | 128.5469 µm | 126.7 | radial power spectrum peak of the saturated Swift-Hohenberg spiral, scaled by 126.7 µm at k_hat = 1 *(k_peak (hat) = 0.9856; charge m = 2)* |
| orientation KS distance between specimen gradients and the simulated spiral | 0.077 KS statistic | 0.0 | magnitude-weighted two-sample KS on Sobel orientations of the gallery field and the SH spiral |
| negative-control RMS ratio at epsilon = -0.2 | 0.0026 RMS_final / RMS_initial | 1.0 | same spiral initial condition integrated below onset; pattern must collapse *(RMS0 = 0.0877, RMS_final = 0.0002)* |
| ablation wavenumber shift after removing finite-k selection | 0.7241 hat wavenumber | 0.0 | replace (k_c^2 + Laplacian)^2 by plain diffusion and remeasure the radial peak *(k_full = 0.9856, k_ablate = 0.2615)* |
| epsilon-sweep monotonic amplitude flag for epsilon > 0 | 1.0 boolean | 1.0 | saturated RMS at epsilon in {0.05, 0.15, 0.25, 0.45} is nondecreasing *(RMS sequence -0.2:0.000, 0.05:0.137, 0.15:0.287, 0.25:0.381, 0.45:0.524)* |
| substrate-sweep wavelength at b = 0.20 with diffusivities held at the s0 calibration | 176.9486 µm | 126.7 | recompute k_c(b) at fixed D_u, D_v calibrated for b = 1.15; b = 0.20 is the low-substrate end of the sweep *(sweep b=0.2:λ=176.9, b=0.4:λ=142.2, b=1.15:λ=126.7, b=2.5:λ=131.6, b=4.0:λ=156.6)* |
| specimen radial-spectrum cosine against the simulated spiral | 0.987 cosine | 1.0 | log radial FFT bins 2:24 of the gallery pigment field versus the Swift-Hohenberg field |
| charge-sweep wavelength at topological charge m = 1 | 128.5469 µm | 126.7 | Swift-Hohenberg spiral at m = 1, radial-spectrum wavelength; m = 2 is the rendered field *(charges m=1:λ=128.55, m=2:λ=128.55, m=3:λ=128.55)* |
| seed-ensemble coefficient of variation of the selected wavenumber | 0.0 CV | 0.0 | sample standard deviation over mean of k_peak on seeds 3141-3144 *(k_peaks 0.9856, 0.9856, 0.9856, 0.9856)* |

## Environment at checkpoint

- python: 3.12
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 of loop 3141. The cloud image has no Lean 4 binary, so the five theorems were not executed here. Nanopub signing was not available (~/.nanopub/profile.yml absent).
