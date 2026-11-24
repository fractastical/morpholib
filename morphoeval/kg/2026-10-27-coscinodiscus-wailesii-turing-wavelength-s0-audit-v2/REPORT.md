# MorphoEval report — audit `2026-10-27-coscinodiscus-wailesii-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-coscinodiscus-wailesii-turing-wavelength-s0-audit -> 2026-10-27-coscinodiscus-wailesii-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Coscinodiscus wailesii Turing wavelength at 4.3 mm (Holocene, chronoseries 1/1) (`models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py`) | 92 | 88 | 90 | 93 | 93 | 91 | 92 | 91.3 | 89.9 | 92.1 | 91.0 | **91.0** | faithful |

## Findings and remediation

### Coscinodiscus wailesii Turing wavelength at 4.3 mm (Holocene, chronoseries 1/1)  —  overall 91.0 (faithful)

Pass 2 keeps k_c = argmax sigma_+(k) calibrated to 4.3 mm at 0 Ma and replaces the headline residual. Two-time Fourier growth rates match lambda_sigma with Pearson r = 0.9997 and RMSE = 0.739 length units, against the pass-1 power-spectrum residual of r = 0.9899 and RMSE = 1.453. A --dv 48 probe returns lambda_sigma = 17.78 and lambda_growth = 18.56. The disc-clamped interior standard deviation is 0.719 against an ablation final standard deviation near 0.002. Upper-left SEM pitch is 10.15 px (n = 718, CV = 0.268) versus simulated pitch 14.06 px (CV = 0.086); KS = 0.317. The GIF remains 12 frames at 640×480 with cv_sim_similarity 84.8.

- **Model fidelity to source: 92** — sigma_plus and sigma_max_wavenumber implement the reported law k_c = argmax sigma_+. empirical_growth_wavelength estimates the same peak from exp(2 σ t) growth of Fourier power between two linear-regime snapshots. The det-proxy remains a negative control (r = -0.919). Integer band conditions are in Loop_3156. The spatial PDE is a transfer of Schnakenberg's well-mixed kinetics, which caps the score below a line-by-line transcription of a spatial paper.  
  evidence: `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L143–165; `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L266–318; `morphoeval/lean/loops/Loop_3156_coscinodiscus_wailesii_turing_wavelength_s0.lean` L37–66
- **Morphological realism / specimen match: 88** — The vendored CC BY-SA plate is segmented (718 pores on the upper-left panel) and shown beside the periodic spots and the disc-clamped valve. The growth GIF is 12 frames at 640×480. Canonical cv_sim_similarity against that plate is 84.8 (Sobel orientation, radial FFT, structure-tensor coherency, occupancy). The montage still contains panel labels and scale bars the spot field does not draw, so the score stays below the mid-90s.  
  evidence: `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L451–466; `docs/media/coscinodiscus_wailesii_turing_wavelength_s0_growth.gif` L0–0; `docs/media/coscinodiscus_wailesii_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 90** — lambda_sigma increases across D_v in {16, 20, 24, 28, 32, 40, 48, 64}. The equal-diffusivity ablation decays (ratio 0.038) while the nominal branch grows. --dv 48 --probe-only returns lambda_sigma = 17.781 and lambda_growth = 18.563, longer than the D_v = 32 nominal, which is the monotonic control the parameter is supposed to have. Disc interior std is 0.719 against the near-zero ablation amplitude.  
  evidence: `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L321–364; `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L919–931; `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L367–390
- **Morphometric validation: 93** — Growth-rate Pearson r = 0.9997 and RMSE = 0.739 length units, improving on the power-spectrum residual r = 0.9899, RMSE = 1.453. Specimen pitch 10.150 px (CV 0.268) is not equal to simulated pitch 14.060 px (CV 0.086). KS distance on median-normalised diameters is 0.317. Contour circularity of the plate montage is 0.166 against a circular reference of 1. The early-time simulated wavelength converts to 4.66 mm beside the calibrated 4.3 mm.  
  evidence: `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L266–364; `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L696–830
- **Source and data traceability: 93** — Reference JSON states 4.3 mm and age 0 Ma, pins the specimen SHA-256, and names the Wikimedia CC BY-SA 4.0 source. BibTeX holds the five DOIs. Scout manifest records the OpenAlex/Wikimedia query after slot 3156 missed campaign_100_slots.yaml. No slug collision with the existing Sumper Cahn-Hilliard diatom model.  
  evidence: `datasets/coscinodiscus_wailesii_turing_wavelength_s0/paper_citations.bib` L1–54; `datasets/coscinodiscus_wailesii_turing_wavelength_s0/coscinodiscus_wailesii_turing_wavelength_s0_reference.json` L1–80
- **Code-to-model traceability: 91** — Docstrings name the dispersion branch, the rejected proxy, and the growth-rate window. Lean theorems homogeneous_mode_stable, nominal_holocene_turing_band, equal_diffusivity_ablation_suppresses_pattern, and inhibitor_diffusion_lengthens_selected_wavelength match the integer checks in self_test.  
  evidence: `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L1–33; `morphoeval/lean/loops/Loop_3156_coscinodiscus_wailesii_turing_wavelength_s0.lean` L1–78
- **Reproducibility package: 92** — --self-test checks the band inequalities, Lean milli-wavelengths 14086/16328/18895, ablation ratio < 0.2, bitwise reruns, both Pearson thresholds, RMSE improvement, and left-column pitch order. Seed 3156. Canonical invocation is the script with no arguments. Signing could not be run: no nanopub profile.  
  evidence: `models/2026-delphi10000-coscinodiscus-wailesii-turing-wavelength-s0.py` L858–916

Remediation:
- The Lean 4 kernel was not executed here (no lean binary on the image). The proofs are decidable integer statements and remain to be checked by `lean` when a toolchain is present.
- SEM scale-bar labels were not transcribed to nanometres; panel pitches stay in pixels.
- Growth-rate RMSE is 0.739 length units, a remaining systematic offset between the continuous argmax and the discrete ring estimator.

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

Pass 2 of slot 3156 after replacing the headline residual with the growth-rate wavelength, adding the --dv probe, and measuring the disc interior. No lean binary and no ~/.nanopub/profile.yml in this environment.
