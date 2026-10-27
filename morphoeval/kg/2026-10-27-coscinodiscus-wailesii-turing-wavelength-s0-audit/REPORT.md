# MorphoEval report — audit `2026-10-27-coscinodiscus-wailesii-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Coscinodiscus wailesii Turing wavelength at 4.3 mm (Holocene, chronoseries 1/1) (`models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py`) | 84 | 76 | 74 | 76 | 90 | 86 | 86 | 80.7 | 76.0 | 87.3 | 80.8 | **80.8** | faithful |

## Findings and remediation

### Coscinodiscus wailesii Turing wavelength at 4.3 mm (Holocene, chronoseries 1/1)  —  overall 80.8 (faithful)

Pass 1 implements spatial Schnakenberg kinetics and reports k_c = argmax sigma_+(k), calibrated so the nominal wavelength is 4.3 mm at age 0 Ma. The early-time power spectrum tracks lambda_sigma with Pearson r = 0.9899 and RMSE = 1.453 length units. The determinant-minimizer proxy anti-correlates (r = -0.919) and is rejected. Equal diffusivities decay (amplitude ratio 0.038). The upper-left SEM panel yields 718 areolae at pitch 10.15 px. The animated GIF is 12 frames at 640×480 and scores cv_sim_similarity 84.8 against the vendored plate.

- **Model fidelity to source: 84** — The PDE, Jacobian, and sigma_+ maximizer are implemented (sigma_plus, sigma_max_wavenumber). The det-minimizer is kept only as a negative control and anti-correlates with the spectrum. Pass 1 still headlines the power-spectrum peak, which is systematically longer than sigma_+ (RMSE 1.453 length units), so the reported comparison is not yet the linear growth rate.  
  evidence: `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L15–28; `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L143–165; `morphoeval/lean/loops/Loop_3156_coscinodiscus_wailesii_turing_wavelength_s0.lean` L26–48
- **Morphological realism / specimen match: 76** — The comparison figure places the vendored six-panel SEM beside the spot field, and the GIF scores cv_sim_similarity 84.8 at 640×480 with 12 frames. Pass 1 does not yet treat that score as a closed specimen match: the animation is a periodic spot lattice, while the micrograph is a labelled plate montage, and the disc-clamped valve is unmeasured.  
  evidence: `docs/media/coscinodiscus_wailesii_turing_wavelength_s0_comparison.png` L0–0; `docs/media/coscinodiscus_wailesii_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L451–466
- **Parameter expressivity: 74** — An eight-point D_v sweep and an equal-diffusivity ablation are computed, and lambda_sigma lengthens with inhibitor diffusivity (14.086, 16.328, 18.895 length units at D_v = 16, 32, 64). Pass 1 does not yet record a one-point CLI probe or the interior standard deviation of the disc-clamped margin, so the boundary condition is shown rather than measured.  
  evidence: `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L321–364; `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L367–390
- **Morphometric validation: 76** — Power-spectrum Pearson r = 0.9899 against lambda_sigma, with RMSE = 1.453 length units. Upper-left areola pitch is 10.15 px (CV 0.268, n = 718) against simulated pitch 14.06 px (CV 0.086). KS distance on median-normalised diameters is 0.317. The RMSE is a real bias, so this pass does not treat the fit as closed.  
  evidence: `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L251–263; `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L696–780
- **Source and data traceability: 90** — Five DOIs are vendored in paper_citations.bib and the reference JSON. The specimen JPEG is SHA-256 cf556dab9e116068…, matching the species-gallery file and the CC BY-SA 4.0 Wikimedia plate. Slot 3156 is absent from campaign_100_slots.yaml; the scout used the OpenAlex/Crossref/Wikimedia query path and found no colliding slug.  
  evidence: `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L101–112; `datasets/coscinodiscus_wailesii_turing_wavelength_s0/coscinodiscus_wailesii_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 86** — The module docstring maps each symbol onto Turing 1952 and Schnakenberg 1979, and the Lean namespace MathOfBiology.Loop_3156 restates the integer band conditions. The spatial step is explicitly a transfer of a well-mixed paper.  
  evidence: `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L1–39
- **Reproducibility package: 86** — Seed 3156, a self-test covering the Turing band, ablation decay, bitwise rerun, the power-spectrum correlation, and panel pitch order. Dependencies are the scientific stack already used by MorphoEval. The Lean kernel and nanopub signer were not available.  
  evidence: `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L858–917

Remediation:
- Headline the goodness of fit on the two-time Fourier growth-rate wavelength (empirical_growth_wavelength) and report its RMSE beside the 1.453 length-unit power-spectrum residual.
- Expose a --dv/--seed probe so inhibitor diffusivity is a command-line regime, and print lambda_sigma against lambda_growth at that point.
- Measure the disc-clamped valve interior standard deviation against the ablation amplitude so the margin boundary condition is a number, not only a panel.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Coscinodiscus wailesii Turing wavelength at 4.3 mm (Holocene, chronoseries 1/1)
- **spec_fingerprint** (semantic model + inputs): `7caaeeacf26efa87e18b214d7dcc8b05a2350b3e3c52eadcca8cd98bd3b11f89`
- **input_dataset_sha256**: `datasets/coscinodiscus_wailesii_turing_wavelength_s0/coscinodiscus_wailesii_specimen.jpg` (`cf556dab9e116068…`), `datasets/coscinodiscus_wailesii_turing_wavelength_s0/coscinodiscus_wailesii_turing_wavelength_s0_reference.json` (`e1fca8ca30c9782c…`), `datasets/coscinodiscus_wailesii_turing_wavelength_s0/scout_manifest.json` (`c6bf02023a4c27e5…`)
- `docs/media/coscinodiscus_wailesii_turing_wavelength_s0_comparison.png` — artifact_sha256 `249a0ac9549fd79e…`, 594,009 B, 1512×1008 px
- `docs/media/coscinodiscus_wailesii_turing_wavelength_s0_growth.gif` — artifact_sha256 `998305546cc851c9…`, 2,764,030 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| turing_wavenumber_k_c | 0.3848 1/length | 0.5 | k_c = sqrt(argmax_q sigma_+(q)) at D_u=1.60, D_v=32, a=0.10, b=0.90 *(reference is a round marker; the value is the computed argmax)* |
| calibrated_turing_wavelength_mm | 4.3 mm | 4.661512293505827 | nominal lambda_sigma mapped onto the Delphi detail scale *(value is the 4.3 mm slot scale; reference is the simulated peak in mm and is not equal to it)* |
| pearson_r_sigma_vs_simulated_wavelength | 0.9899 1 | 1.0 | Pearson r across the D_v sweep between lambda(argmax sigma_+) and the early-time radial FFT peak *(RMSE_length_units recorded alongside; r > 0.98 is the goodness-of-fit gate)* |
| rmse_wavelength_length_units | 1.4534 length | 0.0 | RMSE of the early-time power-spectrum peak against lambda_sigma (pass-1 residual) *(systematic long bias of the power peak; pass 2 reports the growth-rate RMSE separately)* |
| pearson_r_growth_rate_vs_sigma | 0.9997 1 | 1.0 | Pearson r between lambda(argmax sigma_+) and the two-time Fourier growth-rate peak *(RMSE of this estimator is rmse_growth_rate_length_units)* |
| rmse_growth_rate_length_units | 0.7392 length | 1.4534456459318923 | RMSE of the Fourier growth-rate wavelength against lambda_sigma over 8 values of D_v *(reference is the larger pass-1 residual and is not equal to this value)* |
| disc_interior_std_vs_ablation | 0.7187 concentration | 0.001899324739550657 | standard deviation of u inside the clamped valve disc at nominal D_v *(margin boundary condition still patterns the interior; the ablation does not)* |
| pearson_r_det_proxy_vs_simulated_wavelength | -0.9186 1 | 0.98991946317729 | Negative-control correlation of the det-minimizer wavelength against the same spectral peaks *(control: the det proxy anti-correlates; it is not the reported wavenumber)* |
| ablation_amplitude_decay_ratio | 0.038 1 | 6.024039457818228 | std(u) final/initial at D_v = D_u = 1.60 (Turing drive negative) *(negative control versus the patterning branch)* |
| turing_drive_nominal | 24.0 1 | -0.31999999999999984 | D_v f_u + D_u g_v at the Holocene nominal diffusivities |
| specimen_areola_pitch_px | 10.1498 px | 14.05991630588931 | mean nearest-neighbour pitch of dark pores on the upper-left SEM panel (Otsu, circularity >= 0.4) *(specimen image metric; value is not the simulated reference)* |
| specimen_areola_pitch_cv | 0.2677 1 | 0.0860576378628658 | coefficient of variation of nearest-neighbour pitch on the upper-left SEM panel *(CV goodness-of-fit term; self-refential equality is not used)* |
| specimen_pore_circularity | 0.9448 1 | 1.0 | mean 4*pi*area/perimeter^2 of segmented upper-left areolae |
| ks_specimen_vs_sim_diameter_shape | 0.3171 1 | 0.0 | Kolmogorov-Smirnov distance between median-normalised specimen and simulated pore diameters *(KS goodness-of-fit; diameters are scaled out so magnification does not dominate)* |
| specimen_contour_circularity | 0.166 1 | 1.0 | compute_specimen_cv_signature circularity of the full six-panel plate (non-circular composite) *(non-circular specimen metric; the vendored image is a rectangular plate montage)* |
| sem_magnification_pitch_ratio_coarse_over_mid | 2.0414 1 | 2.3349299223721474 | left-column pitch ratio between the coarsest and middle SEM panels (figure magnification, not a biological tier) |
| holocene_age_Ma | 0.0 Ma | 0.001977 | chronoseries 1/1 extant point *(age of the slot is 0 Ma; the 1977 introduction date is a historical record inside the Holocene)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 of slot 3156. The Lean file is integer `decide` proofs; this environment has no `lean` binary, so the kernel was not executed. Nanopub signing profile ~/.nanopub/profile.yml is absent.
