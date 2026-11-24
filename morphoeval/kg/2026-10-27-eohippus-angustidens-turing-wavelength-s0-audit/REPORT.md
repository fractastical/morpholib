# MorphoEval report — audit `2026-10-27-eohippus-angustidens-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `b6fee985871268f23e54cff46ae5985cc017a4e8` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Eohippus angustidens Schnakenberg wavelength at 78 µm (Wasatchian, 52.0 Ma) (`models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py`) | 86 | 78 | 80 | 82 | 88 | 78 | 84 | 84.0 | 79.5 | 83.7 | 82.1 | **82.1** | faithful |

## Findings and remediation

### Eohippus angustidens Schnakenberg wavelength at 78 µm (Wasatchian, 52.0 Ma)  —  overall 82.1 (faithful)

Pass 1 implements Schnakenberg kinetics and Murray's k_c so that lambda_c is 7.752e-05 m (78 µm) at D_v/D_u = 8.22, on a Wasatchian (52.0 Ma) Eohippus angustidens epidermal patch. Dispersion Pearson r = 0.997 against the analytical eigenvalue. Equal diffusion collapses the field (std ratio 0.0005). The specimen plate's scale-bar wavelength is 86.5 µm, not 77.5 µm. The growth GIF is 640×480 and 12 frames; pass-1 CV similarity was 87.3 at pattern weight 0.72. Weighted overall 82.1. Gaps closed in pass 2 are listed under remediation.

- **Model fidelity to source: 86** — The steady state, Jacobian, sigma(k), critical ratio, and Murray k_c formula are implemented and self-tested (Pearson r 0.997 on the binned growth rates). Seven Lean theorems encode the integer shadow, including the equal-diffusion failure and the ratio-8.22 band inequality. Deduction: D is fit to the assigned 78 µm scale rather than copied from a measured diffusivity, and pass 1 still carried an unused probe stub.  
  evidence: `models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py` L64–118; `morphoeval/lean/loops/Loop_3219_eohippus_angustidens_turing_wavelength_s0.lean` L19–49
- **Morphological realism / specimen match: 78** — The comparison figure places the purple spiral plate beside a labyrinthine Schnakenberg field in the same palette. They share a cellular texture class and the pass-1 GIF scored CV 87.3, but the topologies differ (one spiral versus a spot-labyrinth). That gap is why this is not a high-90 match.  
  evidence: `docs/media/eohippus_angustidens_turing_wavelength_s0_comparison.png` L0–0; `docs/media/eohippus_angustidens_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 80** — With D_u fixed, Murray lambda falls from 81.9 µm at ratio 5 to 74.3 µm at ratio 18, and sigma_max rises once the band is unstable. Equal diffusion and ratio 6 are negative controls (final std near 0 versus 0.119). Pass 1 did not yet report that shift as its own measurement or a second seed.  
  evidence: `models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py` L415–458
- **Morphometric validation: 82** — Dispersion Pearson r = 0.997 (reference 1), RMSE = 0.020 (reference 0), specimen dominant wavelength 86.5 µm versus model 77.5 µm, specimen coherency 0.83 versus an isotropic reference 0.5, contour circularity 0.075 versus 1. Pass 1 had not yet added the shared-lattice Pearson/KS.  
  evidence: `models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py` L257–323; `datasets/eohippus_angustidens_turing_wavelength_s0/eohippus_angustidens_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 88** — Six DOIs were resolved on OpenAlex on 2026-10-27. The specimen file is SHA-256 4f492b7fff0d285dc090002b29f9aca4a757bad6f4c08a8f5dd93b8a1bcfc0bf, copied from the gallery plate after Commons returned no file. The 100 µm bar is 111 px.  
  evidence: `datasets/eohippus_angustidens_turing_wavelength_s0/paper_citations.bib` L1–70; `datasets/eohippus_angustidens_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 78** — The module docstring names Schnakenberg, Turing, and Murray, and the Lean file cites the integer shadow of this Jacobian. Pass 1 still had a dead probe helper, so not every declared routine was on the executed path.  
  evidence: `models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py` L1–20
- **Reproducibility package: 84** — --self-test checks r > 0.98, the Turing inequalities, the ablation ratio, and the specimen pin. Seed 3219, dt 0.0035, grid 112. Dependencies are numpy, scipy, pillow, and matplotlib, already used by the repo. No per-script lockfile.  
  evidence: `models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py` L689–716

Remediation:
- {'id': 'R1', 'item': 'Delete the unused linear-probe stub so every helper on the dispersion path is the one that is executed.'}
- {'id': 'R2', 'item': 'Recompute Pearson r and KS on one shared 160x120 lattice between the specimen plate and the rendered field.'}
- {'id': 'R3', 'item': 'Record the pattern-weight CV table and keep the delivered frame at pattern weight >= 0.55.'}
- {'id': 'R4', 'item': 'Report a second-seed centroid and the Murray lambda shift from diffusion ratio 5 to 18 as their own measurements (reference 0).'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Eohippus angustidens Schnakenberg wavelength at 78 µm (Wasatchian, 52.0 Ma)
- **spec_fingerprint** (semantic model + inputs): `0efa2111ea371920360cef53e69ded09282e4fac6685c513b8c1093c17921e13`
- **input_dataset_sha256**: `datasets/eohippus_angustidens_turing_wavelength_s0/eohippus_angustidens_specimen.jpg` (`4f492b7fff0d285d…`), `datasets/eohippus_angustidens_turing_wavelength_s0/eohippus_angustidens_turing_wavelength_s0_reference.json` (`db5bbb9f7e1d38d4…`), `datasets/eohippus_angustidens_turing_wavelength_s0/paper_citations.bib` (`aa36e656b9c744bd…`), `datasets/eohippus_angustidens_turing_wavelength_s0/scout_manifest.json` (`543e8435e637b43f…`)
- `datasets/eohippus_angustidens_turing_wavelength_s0/eohippus_angustidens_turing_wavelength_s0_metrics.json` — artifact_sha256 `a6f95339df8f2238…`, 3,722 B, 
- `docs/media/eohippus_angustidens_turing_wavelength_s0_comparison.png` — artifact_sha256 `a7a42bf2bfffbfbc…`, 709,694 B, 1764×1148 px
- `docs/media/eohippus_angustidens_turing_wavelength_s0_growth.gif` — artifact_sha256 `78ca2b8fdb9d7ca9…`, 3,175,254 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion_sigma_pearson_r | 0.9971233615039029 1 | 1.0 | Pearson r of binned numerical vs analytical Turing growth rate σ(k); RMSE reported separately *(value is the measured correlation, not the ideal reference 1)* |
| dispersion_sigma_rmse | 0.02036612180734322 1/time | 0.0 | RMSE between numerical and analytical σ(k) on the same radial bins |
| specimen_dominant_wavelength_um | 86.48648648648648 µm | 77.52000000000001 | radial FFT of the specimen interior, calibrated by the 111 px = 100 µm scale bar *(specimen plate wavelength is not forced equal to the assigned cellular scale)* |
| specimen_structure_tensor_coherency | 0.8272348563939591 1 | 0.5 | Sobel structure-tensor coherency on the contrast-segmented specimen plate |
| specimen_contour_circularity | 0.07487262531902233 1 | 1.0 | 4π area/perimeter² of the specimen foreground mask |
| ks_radial_spectrum | 0.75 1 | 0.0 | two-sample KS statistic between specimen and simulation radial power spectra |
| equal_diffusion_ablation_std_ratio | 0.000508735637310571 1 | 1.0 | final activator std at D_v=D_u divided by the patterned std (negative control) |
| cells_per_critical_wavelength | 6.46 cells | 6.0 | λ_c / 12 µm keratinocyte diameter |
| specimen_simulation_aligned_pearson_r | -0.027655521569292234 1 | 1.0 | Pearson r of grayscale pixels on a shared 160x120 lattice (specimen plate vs rendered field) *(this is a specimen-image comparison and is intentionally not the dispersion-curve Pearson r)* |
| specimen_simulation_aligned_ks | 0.16375000000000006 1 | 0.0 | KS statistic of grayscale values on the shared 160x120 lattice |
| formula_lambda_shift_um | 7.590717748996752 µm | 0.0 | Murray k_c wavelength at D_v/D_u = 5 minus the wavelength at D_v/D_u = 18, D_u fixed |
| seed_to_seed_wavelength_difference_um | 0.3487948156439131 µm | 0.0 | absolute difference of power-weighted spectral centroids at seeds 3219 and 3230 |
| detail_to_organ_wavenumber_ratio | 42.54385964912281 1 | 1.0 | organ-tier sibling wavelength 0.003298 m divided by this slot's 7.752e-05 m |

## Environment at checkpoint


Slot 3219 files, including the specimen JPEG, are in b6fee985. This audit record is updated in the following commit so the nanopubs cite that tree.
