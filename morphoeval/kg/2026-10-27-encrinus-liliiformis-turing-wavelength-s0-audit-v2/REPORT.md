# MorphoEval report — audit `2026-10-27-encrinus-liliiformis-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `cd3c0ff20113e2850a5dd72a24dfd8d4e782b322` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-encrinus-liliiformis-turing-wavelength-s0-audit -> 2026-10-27-encrinus-liliiformis-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Encrinus liliiformis Schnakenberg wavelength at 12 nm (Muschelkalk, 240 Ma) (`models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py`) | 88 | 76 | 88 | 88 | 92 | 86 | 88 | 88.0 | 80.5 | 88.8 | 85.2 | **85.2** | faithful |

## Findings and remediation

### Encrinus liliiformis Schnakenberg wavelength at 12 nm (Muschelkalk, 240 Ma)  —  overall 85.2 (faithful)

Same 12 nm, 240 Ma Schnakenberg point. Radial-bin sign agreement is 0.972, Pearson r is 0.9996, RMSE is 0.0031, KS is 0.042, equal-diffusion amplitude ratio is about 1.3e5, and GIF-to-slab CV similarity is 84.6. Pixel IoU 0.17 is retained as a failed mask, not as the segmentation.

- **Model fidelity to source: 88** — The Jacobian, onset formula k_c^2 = (D_v f_u + D_u g_v)/(2 D_u D_v), dispersion eigenvalue, and periodic integrator are all in the script and the self-test checks them. The rendered disk uses harmonics of the image fundamental whose physical radius is n_wave times 12 nm. Above threshold the fastest mode shifts; that shift is reported rather than hidden.  
  evidence: `models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py` L106–163; `models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py` L260–302
- **Morphological realism / specimen match: 76** — Last GIF frame versus the CC BY-SA Ingelfingen slab has CV similarity 84.6 (orientation, radial FFT, coherency, occupancy). The comparison figure shows the slab and the rings side by side. They are not the same silhouette: one is a crown on limestone, the other is the radial stereom mode. 76 credits the shared calcite palette and the radial profile overlay without calling them the same picture.  
  evidence: `docs/media/encrinus_liliiformis_turing_wavelength_s0_comparison.png` L0–0; `docs/media/encrinus_liliiformis_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — Once delta exceeds the Turing threshold, lambda_c falls monotonically from about 25 nm toward 12 nm. delta=22.4 is the Encrinus point. Equal diffusion (delta=1) is stable. Holocrinus (delta 18, 18.72 nm, 248 Ma) and Chelocrinus (delta 15.6, 15.64 nm, 232 Ma) are separate parameter points, not rescalings of this one.  
  evidence: `models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py` L50–76; `models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py` L165–187
- **Morphometric validation: 88** — Stereom radial Pearson r = 0.9996, RMSE = 0.0031, KS = 0.042 (reference is the 5% critical value, not the statistic). Radial-bin sign agreement is 0.972 against a chance reference of 0.5. Fossil coherency 0.612 is not the render coherency 0.943. Pixel IoU remains 0.17 and is not used as a pass. The radial fit is in-sample.  
  evidence: `models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py` L381–386; `models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py` L539–722; `datasets/encrinus_liliiformis_turing_wavelength_s0/encrinus_liliiformis_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 92** — DOIs were checked on Crossref. The specimen file is the Ghedoghedo CC BY-SA 4.0 Muschelkalkmuseum photograph, SHA-256 pinned. The gallery stereom image is labeled as a 100 um rendering, not as the fossil.  
  evidence: `datasets/encrinus_liliiformis_turing_wavelength_s0/paper_citations.bib` L1–70; `datasets/encrinus_liliiformis_turing_wavelength_s0/README.md` L1–38; `datasets/encrinus_liliiformis_turing_wavelength_s0/scout_manifest.json` L0–0
- **Code-to-model traceability: 86** — Docstring, regime(), and growth_rate() are the onset formula. Lean theorems muschelkalk_wavelength_is_chronoseries_minimum, schnakenberg_trace_negative, and inhibitor_drive_positive_and_equal_diffusion_fails are the integer forms of those claims. They use decide on concrete Nats, not mathlib analysis.  
  evidence: `models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py` L95–146; `morphoeval/math_kg_loops/delphi10000-encrinus-liliiformis-turing-wavelength-s0.yaml` L1–63
- **Reproducibility package: 88** — python3 models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py --self-test passes: r>0.98, radial agreement >0.9, CV>=58, GIF >=10 frames and >=480 px, seed 3218. Signing was not available (no ~/.nanopub/profile.yml).  
  evidence: `models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py` L775–812

Remediation:
- Independent second assessor was not run. Scores are single-agent.
- The radial Pearson fit is in-sample. A holdout of the outer radius does not extrapolate (Fourier harmonics are a poor extrapolant); that limit is recorded rather than patched.
- Pixel dark-mask IoU remains 0.17. It is no longer the segmentation used for the score.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=0.65]** `independent_assessor` → `model-fidelity`: A second model provider has not scored this audit.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Encrinus liliiformis Schnakenberg wavelength at 12 nm (Muschelkalk, 240 Ma)
- **spec_fingerprint** (semantic model + inputs): `a01502a8a9ccbc4b5251448c56820781f74c49be8d38d864003cd60c668e74f1`
- **input_dataset_sha256**: `datasets/encrinus_liliiformis_turing_wavelength_s0/encrinus_liliiformis_specimen.jpg` (`9bb90bc584ae7e85…`), `datasets/encrinus_liliiformis_turing_wavelength_s0/encrinus_liliiformis_stereom_gyroid_canonical.jpg` (`d19f3b6e48018c2d…`), `datasets/encrinus_liliiformis_turing_wavelength_s0/encrinus_liliiformis_turing_wavelength_s0_reference.json` (`64d02f82520a1b3a…`), `datasets/encrinus_liliiformis_turing_wavelength_s0/paper_citations.bib` (`160e70968df139fb…`)
- `datasets/encrinus_liliiformis_turing_wavelength_s0/encrinus_liliiformis_turing_wavelength_s0_metrics.json` — artifact_sha256 `007986656ea2e031…`, 1,845 B, 
- `datasets/encrinus_liliiformis_turing_wavelength_s0/scout_manifest.json` — artifact_sha256 `78145824bb27b836…`, 3,426 B, 
- `docs/media/encrinus_liliiformis_turing_wavelength_s0_comparison.png` — artifact_sha256 `f172842057f58457…`, 725,947 B, 1456×1092 px
- `docs/media/encrinus_liliiformis_turing_wavelength_s0_growth.gif` — artifact_sha256 `09eee2d7b66bd8fe…`, 5,571,040 B, 800×600 px, 16 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytical_turing_wavelength_m | 1.238e-08 m | 6.12e-07 | lambda_c = 2*pi/k_c from Schnakenberg Jacobian at a=0.047, b=1.153, delta=22.4 *(value is the 12 nm law; reference is the macromolecular sibling, not this run)* |
| dispersion_sigma_max | 0.29622236 1/s | 0.0 | max_k of the Turing growth rate; RMSE not applicable to a scalar rate; std across the k-grid is implicit |
| k_at_sigma_max_over_k_c | 0.8004002 1 | 1.0 | wavenumber of peak growth divided by the onset k_c; above threshold the fastest mode shifts *(reference is the onset wavenumber ratio; the measured ratio is the supercritical shift)* |
| specimen_stereom_radial_pearson_r | 0.99964129 1 | 0.98 | Pearson r of azimuthal stereom luminance vs Galerkin harmonics of k_c; in-sample fit |
| specimen_stereom_radial_rmse | 0.003129998 luminance | 0.05 | RMSE of the same radial Pearson fit |
| specimen_stereom_radial_ks_statistic | 0.041666667 1 | 0.16027754 | two-sample KS statistic between empirical radial bins and the harmonic prediction *(reference is the approximate alpha=0.05 critical value, not the statistic)* |
| specimen_radial_profile_cv | 0.24953188 1 | 0.0 | coefficient of variation (std/mean) of the empirical radial luminance bins |
| ablation_sigma_max_equal_diffusion | -0.25916667 1/s | 0.29622236 | negative control: D_v set equal to D_u, Turing drive lost *(control vs patterned sigma_max)* |
| pattern_vs_ablation_amplitude_ratio | 130704.85 1 | 1.0 | numerical Schnakenberg transect amplitude vs equal-diffusion ablation control |
| specimen_fossil_coherency | 0.6121 1 | 0.9454 | structure-tensor coherency of the Muschelkalkmuseum slab (real specimen segmentation) *(fossil vs rendered frame; the two coherencies are not forced equal)* |
| specimen_fossil_foreground_fraction | 0.675 1 | 0.4552 | optical-density foreground fraction of the real Ingelfingen slab |
| specimen_fossil_circularity | 0.1178 1 | 1.0 | contour circularity of the segmented Ingelfingen slab |
| specimen_stereom_dark_annulus_iou | 0.17105591 1 | 1.0 | pixel IoU of quantile dark masks; lowered by the caption strip and center crop *(1 would be a copied mask; pixel IoU is not the radial segmentation)* |
| specimen_stereom_radial_sign_agreement | 0.97222222 1 | 0.5 | fraction of radial bins on the same side of the median, empirical vs Galerkin prediction *(0.5 is chance agreement; the measured fraction is the caption-free segmentation)* |
| chronoseries_predecessor_wavelength_m | 1.872e-08 m | 1.238e-08 | Holocrinus dubius at 248 Ma, delta=18.0; vs Encrinus 12 nm |
| chronoseries_successor_wavelength_m | 1.564e-08 m | 1.238e-08 | Chelocrinus cassianus at 232 Ma, delta=15.6; vs Encrinus 12 nm |
| delta_sweep_unstable_count | 8.0 1 | 10.0 | count of D_v/D_u ratios above the Turing threshold in the parameter sweep |
| gif_frame_count | 16.0 frames | 6.0 | animated GIF frames written for the 12 nm growth sequence |
| gif_min_resolution_px | 600.0 px | 340.0 | minimum edge of the growth GIF |

## Environment at checkpoint


Pass 2 re-score of the same commit after adopting radial-bin segmentation.
