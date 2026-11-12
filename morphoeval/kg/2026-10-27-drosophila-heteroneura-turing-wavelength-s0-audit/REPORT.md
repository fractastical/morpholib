# MorphoEval report — audit `2026-10-27-drosophila-heteroneura-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `1fc59ceabb8ad3bc2b65011b9e189d8de99889cf` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila heteroneura detail-scale Turing wavelength (5.222 µm, Holocene 0 Ma) (`models/2026-delphi10000-drosophila-heteroneura-turing-wavelength-s0.py`) | 80 | 84 | 86 | 82 | 88 | 82 | 84 | 82.0 | 83.2 | 84.8 | 83.3 | **83.3** | faithful |

## Findings and remediation

### Drosophila heteroneura detail-scale Turing wavelength (5.222 µm, Holocene 0 Ma)  —  overall 83.3 (faithful)

Pass 1 dispersion root is 5.222 µm with h(k_c)=-1802.8 and equal-diffusion growth rate -5.106. Specimen ring spacing is 68.47 µm (111 px = 100 µm). Radial-envelope Pearson r = 0.98482, RMSE = 0.01911, CV similarity of the pre-carrier GIF = 87.7. Guttifera mode count in that metrics file is 205 because it used rounding, not the Lean truncating division.

- **Model fidelity to source: 80** — jacobian_parts and critical_k2 implement the Schnakenberg steady state and the Turing wavenumber. The pass-1 GIF is the 68 µm envelope only; the 5.222 µm carrier is not yet in the rendered field.  
  evidence: `models/2026-delphi10000-drosophila-heteroneura-turing-wavelength-s0.py` L89–121; `datasets/drosophila_heteroneura_turing_wavelength_s0/fit_metrics_pass1.json` L1–20
- **Morphological realism / specimen match: 84** — Pre-carrier GIF scores CV similarity 87.7 against the gallery micrograph (12 frames, 800x600). The rings read as the same radial ocellus. The critical-wavelength grain is missing.  
  evidence: `docs/media/drosophila_heteroneura_turing_wavelength_s0_growth.gif` L0–0; `datasets/drosophila_heteroneura_turing_wavelength_s0/fit_metrics_pass1.json` L60–70
- **Parameter expressivity: 86** — gamma sweep and Dv/Du sweep are both monotonic: raising either control shortens lambda_c. Equal diffusion is a real negative control (growth rate negative, PDE amplification collapses).  
  evidence: `models/2026-delphi10000-drosophila-heteroneura-turing-wavelength-s0.py` L189–226
- **Morphometric validation: 82** — Pearson r 0.98482 and RMSE 0.01911 on the radial mean, KS D about 0.42 against a zero-difference reference, scale-bar spacing 68.47 µm versus the 5.222 µm carrier. Mode counts in the pass-1 file still use rounding (guttifera 205, not the Lean value 204).  
  evidence: `models/2026-delphi10000-drosophila-heteroneura-turing-wavelength-s0.py` L294–321; `datasets/drosophila_heteroneura_turing_wavelength_s0/fit_metrics_pass1.json` L22–45
- **Source and data traceability: 88** — OpenAlex-checked DOIs for Turing, Schnakenberg, Murray, Spieth, Werner, and Pitnick. Specimen SHA-256 is pinned. Commons returned no bitmap; the gallery file is the vendored image.  
  evidence: `datasets/drosophila_heteroneura_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/drosophila_heteroneura_turing_wavelength_s0/drosophila_heteroneura_turing_wavelength_s0_reference.json` L40–70
- **Code-to-model traceability: 82** — Each modelling step is commented back to the dispersion formula. Lean theorems pin the integer wavelength and gamma orderings, not a real-analysis proof that h(k_c) is negative; that sign is the Python self-test.  
  evidence: `morphoeval/lean/loops/Loop_3188_drosophila_heteroneura_turing_wavelength_s0.lean` L21–35; `models/2026-delphi10000-drosophila-heteroneura-turing-wavelength-s0.py` L104–121
- **Reproducibility package: 84** — Seed 3188, --self-test, pinned input hashes, and a known Python stack. Pass-1 metrics and the later script revision are both in the tree, which is easy to confuse until v2 says which file it scores.  
  evidence: `models/2026-delphi10000-drosophila-heteroneura-turing-wavelength-s0.py` L633–670; `datasets/drosophila_heteroneura_turing_wavelength_s0/README.md` L1–30

Remediation:
- R1: Modulate the rendered envelope by the 5.222 µm carrier and record its pixel period. Pass-1 metrics omit carrier_period_px.
- R2: Count head-capsule modes with the Lean truncating division 1720000//lambda_nm. Pass-1 guttifera count is 205, not 204.
- R3: Independent second assessor. This pass is builder-scored.
- R4: Head width 1.72 mm is reconstructed from Spieth 1981, not measured on this micrograph.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.5]** `specimen_comparison` → `model-fidelity`: Render the 5.222 µm carrier on the envelope and record its pixel period.
- **[OPEN · P=1.2]** `morphometric_validation` → `morphometric-validation`: Align mode counts with Loop_3188 integer division.
- **[OPEN · P=1.1]** `independent_assessor` → `source-traceability`: A second assessor from another provider has not scored this loop.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila heteroneura detail-scale Turing wavelength (5.222 µm, Holocene 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `4b21b7c69b3d1f5fc1cb70f9be69842c1f8a42ab65644a6367b446725ab39515`
- **input_dataset_sha256**: `datasets/drosophila_heteroneura_turing_wavelength_s0/drosophila_heteroneura_specimen.jpg` (`23a06203ecdd717e…`), `datasets/drosophila_heteroneura_turing_wavelength_s0/drosophila_heteroneura_turing_wavelength_s0_reference.json` (`c8bbca96708967c8…`), `datasets/drosophila_heteroneura_turing_wavelength_s0/paper_citations.bib` (`6d9ece5ab667efce…`)
- `docs/media/drosophila_heteroneura_turing_wavelength_s0_comparison.png` — artifact_sha256 `3018a50272e00371…`, 716,053 B, 1632×984 px
- `docs/media/drosophila_heteroneura_turing_wavelength_s0_growth.gif` — artifact_sha256 `b6ae6c7d08c75120…`, 4,505,167 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Schnakenberg critical pigment wavelength lambda_c | 5.222 µm | 5.2 | k_c^2 = (Dv f_u + Du g_v)/(2 Du Dv), scaled so the heteroneura (gamma=18.6, Dv/Du=36.5) root is 5.222 µm *(h(k_c) = -1802.838 (negative). Equal-diffusion h at the same k = 643.166 (positive).)* |
| specimen ocellar stripe spacing from the 100 µm scale bar | 68.4685 µm | 5.222 | dark-ring peak spacing on the radial mean, times 100/scale_bar_px *(scale bar 111 px = 100 µm; spacing 76.000 px)* |
| radial envelope Pearson r against the specimen luminance profile | 0.98482 correlation | 1.0 | least squares of a 4-harmonic radial Turing envelope on the specimen radial mean *(RMSE = 0.01911 luminance units; residual CV = 0.04145)* |
| radial envelope RMSE | 0.019109 luminance | 0.0 | sqrt mean square residual of the radial envelope fit *(Pass gate is Pearson r > 0.98, not a zero residual.)* |
| KS statistic between specimen and simulation gray samples | 0.41875 Kolmogorov-Smirnov D | 0.0 | ks_2samp on 4000 seeded pixels of specimen gray vs steady-frame gray *(two-sided p = 1.359e-314)* |
| specimen structure-tensor coherency | 0.8471 dimensionless | 0.5 | verify_image_fitness.compute_specimen_cv_signature on the vendored micrograph *(foreground fraction 0.4055, circularity 0.0587, wavelength_norm 0.3333)* |
| equal-diffusion ablation growth rate | -5.105903 per unit time | 7.378986 | max Re(eigenvalue of J - k^2 D) with Dv/Du = 1, same a, b, gamma *(PDE variance amplification patterned 43735.377 vs ablation 0.000)* |
| chronospecies wavelength ratio guttifera over heteroneura | 1.60858 dimensionless | 1.0 | lambda(gamma) at fixed Dv/Du; gamma_gut / gamma_het = (lambda_het/lambda_gut)^2 *(lambda gut 8.4000 µm, hyd 6.3500 µm, het 5.2220 µm; mode counts 204, 270, 329)* |
| hypercephalic mode count at 1.72 mm | 329 wavelengths | 204 | 1720000 nm // 5222 nm, the same truncating division as Loop_3188 *(Head width is reconstructed from the Spieth 1981 hammerhead description, not a caliper reading.)* |
| specimen-scale carrier period of the 5.222 µm critical mode | 5.7964 px | 76.0 | lambda_c / (100 µm / scale_bar_px), added to the rendered envelope at amplitude 0.03 *(The carrier is finer than the envelope. Both are in the GIF; they are not the same length.)* |

## Environment at checkpoint


Pass 1 scores fit_metrics_pass1.json from the pre-carrier render. The script at this commit already contains the pass-2 revision; v2 scores that revision.
