# MorphoEval report — audit `2026-10-27-cavia-porcellus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `cc4b7fb74d981b19349d4808476e5b8b2b12f75a` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-cavia-porcellus-turing-wavelength-s0-audit -> 2026-10-27-cavia-porcellus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cavia porcellus Holocene (0 Ma) Turing wavelength at 126.7 µm (tier-5 detail s0) (`models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py`) | 90 | 86 | 88 | 90 | 89 | 85 | 90 | 89.3 | 87.5 | 88.3 | 88.3 | **88.3** | faithful |

## Findings and remediation

### Cavia porcellus Holocene (0 Ma) Turing wavelength at 126.7 µm (tier-5 detail s0)  —  overall 88.3 (faithful)

Pass 2 overall is the weighted rubric mean after crediting the charge sweep, the seed-bin ensemble, and an explicit eigensolver reading of the dispersion fit. lambda_c = 126.7 µm, simulated bin 128.55 µm, gallery field 78.18 µm, negative-control RMS ratio 0.00264, ablation peak shift 0.724.

- **Model fidelity to source: 90** — Pass 2 keeps the Pass 1 equation set and states the limit more sharply. The chemical law is the Schnakenberg Jacobian wavenumber, scaled onto 126.7 µm (D_u = 169.175, D_v = 6767.013, ratio 40, h_min = -5.411). The rendered dynamics are Swift-Hohenberg, which is the amplitude equation of that bifurcation, not a substitute chemical mechanism. Gierer & Meinhardt (1972) is cited as the activator-inhibitor precedent and is not the coded kinetics.  
  evidence: `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L94–182; `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L267–312; `morphoeval/lean/loops/Loop_3141_cavia_porcellus_turing_wavelength_s0.lean` L9–40
- **Morphological realism / specimen match: 86** — Direct computer-vision similarity of the 12-frame 640x480 GIF against the gallery field is 88.2. Radial-spectrum cosine between the field and the spiral is 0.987. The scale-bar wavelength of that same field is 78.18 µm (110 px = 100 µm, 0.909 µm/px), so the match is textural and orientational (KS = 0.077, segmented coherency 0.198) and is not a claim that the gallery peak sits at 126.7 µm.  
  evidence: `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L419–489; `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L205–226; `docs/media/cavia_porcellus_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — Three controls move the pattern as they should. Epsilon raises RMS monotonically (0.137, 0.287, 0.381, 0.524 for epsilon = 0.05, 0.15, 0.25, 0.45) and epsilon = -0.20 collapses the spiral (RMS ratio 0.00264). Removing finite-k selection moves k from 0.986 to 0.261 (wavelength 128.55 µm to 484.5 µm). Raising b at fixed D moves lambda from 176.95 µm (b = 0.20) through 126.7 µm (b = 1.15) and restabilizes the homogeneous state at b = 2.5 and b = 4. Charges m = 1, 2, 3 and four seeds all select the same bin; that is reported as bin-stability, not as a continuous CV estimate.  
  evidence: `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L337–388; `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L184–203
- **Morphometric validation: 90** — Paired comparison of the quadratic growth rate and numpy.linalg.eigvals on 28 wavenumbers gives Pearson r = 1 and RMSE = 0. This pass labels that pair as an eigensolver identity (the Pass 1 remediation), with a distributional KS of 0.036 between the two sigma samples. Independent of that identity, the simulated bin is 128.55 µm versus analytical 126.7 µm, the gallery wavelength is 78.18 µm, orientation KS is 0.077, and radial cosine is 0.987. None of those specimen or stripe values is copied from its reference.  
  evidence: `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L163–249; `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L598–758
- **Source and data traceability: 89** — Seven Crossref-checked DOIs, the scout manifest (including the empty OpenAlex morphogenesis result and the CC BY-SA whole-animal portraits that were rejected as the wrong spatial scale), and the SHA-256 pin of the gallery field. Age 0 Ma and length 0.0001267 m are written in the README and the reference JSON.  
  evidence: `datasets/cavia_porcellus_turing_wavelength_s0/README.md` L1–24; `datasets/cavia_porcellus_turing_wavelength_s0/scout_manifest.json` L1–30; `datasets/cavia_porcellus_turing_wavelength_s0/paper_citations.bib` L1–70
- **Code-to-model traceability: 85** — Each scored block is a named function whose docstring points at the formula it implements, and the math-KG YAML repeats those formulas with Lean declaration names. The Lean file has five theorems and zero sorry. Code traceability stops short of a machine check because lean is not installed in this image.  
  evidence: `morphoeval/math_kg_loops/delphi10000-cavia-porcellus-turing-wavelength-s0.yaml` L1–54; `morphoeval/lean/loops/Loop_3141_cavia_porcellus_turing_wavelength_s0.lean` L18–39
- **Reproducibility package: 90** — --self-test passed on this checkpoint: stability signs, r > 0.98, ablation shift > 0.25, decay ratio < 0.5, substrate sweep, charge bin, seed CV < 0.08, GIF frames and resolution, and the specimen SHA-256. The invocation is deterministic given seed 3141.  
  evidence: `models/2026-delphi10000-cavia-porcellus-turing-wavelength-s0.py` L759–812; `docs/media/cavia_porcellus_turing_wavelength_s0_comparison.png` L0–0

Remediation:
- Integrate the nonlinear Schnakenberg PDE, not only its amplitude equation, on the same 1.0136 mm window.
- Replace the radial-bin wavenumber with a continuous Fourier centroid so a seed ensemble can report a non-zero CV.
- Run the Lean file under a Lean 4 toolchain, and sign the nanopubs once a profile exists.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=2.5]** `model_equation` → `model-fidelity`: The rendered field is Swift-Hohenberg. A two-species nonlinear integration is still open.
- **[OPEN · P=2.5]** `provenance_lock` → `code-traceability`: Loop_3141 was not executed; this image has no lean binary.

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

Pass 2 of loop 3141 after the Pass 1 remediation. The charge sweep, seed ensemble, and eigensolver metric titles are in this checkpoint. Lean was not executed (no lean binary). Nanopub profile absent, so nothing was signed.
