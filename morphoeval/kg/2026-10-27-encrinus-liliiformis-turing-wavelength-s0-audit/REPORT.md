# MorphoEval report — audit `2026-10-27-encrinus-liliiformis-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `cd3c0ff20113e2850a5dd72a24dfd8d4e782b322` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Encrinus liliiformis Schnakenberg wavelength at 12 nm (Muschelkalk, 240 Ma) (`models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py`) | 84 | 70 | 76 | 74 | 90 | 80 | 82 | 81.3 | 71.5 | 84.1 | 78.2 | **78.2** | mostly faithful |

## Findings and remediation

### Encrinus liliiformis Schnakenberg wavelength at 12 nm (Muschelkalk, 240 Ma)  —  overall 78.2 (mostly faithful)

Onset k_c is placed at 1.238e-8 m with a=0.047, b=1.153, delta=22.4. Self-test Pearson r of the stereom radial profile is 0.9996 and the GIF versus the Ingelfingen slab scores CV 84.6. Pixel dark-mask IoU is only 0.17.

- **Model fidelity to source: 84** — regime() and growth_rate() implement the Schnakenberg Jacobian and the Turing dispersion relation, and the onset wavelength is exactly 1.238e-8 m. The picture is a harmonic projection, not the PDE field itself, and above threshold the fastest wavenumber shifts off onset k_c.  
  evidence: `models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py` L106–146; `morphoeval/lean/loops/Loop_3218_encrinus_liliiformis_turing_wavelength_s0.lean` L34–42
- **Morphological realism / specimen match: 70** — The comparison figure places the real Muschelkalkmuseum slab beside sepia rings. CV similarity of the GIF's last frame to that slab is 84.6, but a viewer still sees a crown-on-matrix versus a bullseye. The rings do track the stereom target's radial profile.  
  evidence: `docs/media/encrinus_liliiformis_turing_wavelength_s0_comparison.png` L0–0; `docs/media/encrinus_liliiformis_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 76** — A delta sweep and an equal-diffusion ablation are in the script. Pass 1 does not yet treat the sweep as demonstrated morphospace control because the segmentation metric (pixel IoU) failed.  
  evidence: `models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py` L156–187
- **Morphometric validation: 74** — In-sample Pearson r is 0.9996 and RMSE is 0.0031, but the pixel dark-mask IoU against the stereom target is 0.17. That mask is the wrong segmentation for a captioned image. Fossil coherency is measured and is not equal to the render.  
  evidence: `datasets/encrinus_liliiformis_turing_wavelength_s0/encrinus_liliiformis_turing_wavelength_s0_metrics.json` L0–0; `models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py` L539–722
- **Source and data traceability: 90** — Crossref-verified DOIs, CC BY-SA 4.0 Ingelfingen photograph with artist and SHA-256, and an explicit statement that 12 nm is an assigned detail scale.  
  evidence: `datasets/encrinus_liliiformis_turing_wavelength_s0/paper_citations.bib` L1–70; `datasets/encrinus_liliiformis_turing_wavelength_s0/encrinus_liliiformis_specimen.jpg` L0–0
- **Code-to-model traceability: 80** — The module docstring and regime() name the onset formula. Lean proofs are integer specializations of the trace, the drive, and the chronoseries, not a general real-analysis development.  
  evidence: `models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py` L1–20; `morphoeval/lean/loops/Loop_3218_encrinus_liliiformis_turing_wavelength_s0.lean` L18–48
- **Reproducibility package: 82** — --self-test passed (r=0.9996, cv=84.6, 16 frames, 800x600, seed 3218). No second assessor and no separate environment lock for this slot.  
  evidence: `models/2026-delphi10000-encrinus-liliiformis-turing-wavelength-s0.py` L775–812

Remediation:
- Replace the pixel dark-mask IoU (0.17) with the radial-bin sign agreement as the segmentation metric; the caption strip makes the pixel mask the wrong object.
- Re-score the delta sweep and the equal-diffusion ablation once that segmentation is the one the audit uses.
- Independent second assessor was not run.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.5]** `specimen_comparison` → `morphometric-validation`: Pixel IoU 0.17 is not a registered annulus. Use the radial-bin sign agreement already computed on the same profile.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Encrinus liliiformis Schnakenberg wavelength at 12 nm (Muschelkalk, 240 Ma)
- **spec_fingerprint** (semantic model + inputs): `a01502a8a9ccbc4b5251448c56820781f74c49be8d38d864003cd60c668e74f1`
- **input_dataset_sha256**: `datasets/encrinus_liliiformis_turing_wavelength_s0/encrinus_liliiformis_specimen.jpg` (`9bb90bc584ae7e85…`), `datasets/encrinus_liliiformis_turing_wavelength_s0/encrinus_liliiformis_stereom_gyroid_canonical.jpg` (`d19f3b6e48018c2d…`), `datasets/encrinus_liliiformis_turing_wavelength_s0/encrinus_liliiformis_turing_wavelength_s0_reference.json` (`64d02f82520a1b3a…`), `datasets/encrinus_liliiformis_turing_wavelength_s0/paper_citations.bib` (`160e70968df139fb…`)
- `datasets/encrinus_liliiformis_turing_wavelength_s0/encrinus_liliiformis_specimen.jpg` — artifact_sha256 `9bb90bc584ae7e85…`, 3,638,217 B, 1635×3847 px
- `datasets/encrinus_liliiformis_turing_wavelength_s0/encrinus_liliiformis_turing_wavelength_s0_metrics.json` — artifact_sha256 `007986656ea2e031…`, 1,845 B, 
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


Pass 1. Cloud origin URL carries a token, so build_morphoeval_kg.py skips blob verification.
