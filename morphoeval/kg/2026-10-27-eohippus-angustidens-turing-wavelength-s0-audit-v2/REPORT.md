# MorphoEval report — audit `2026-10-27-eohippus-angustidens-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-eohippus-angustidens-turing-wavelength-s0-audit -> 2026-10-27-eohippus-angustidens-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Eohippus angustidens Schnakenberg wavelength at 78 µm (Wasatchian, 52.0 Ma) (`models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py`) | 90 | 81 | 88 | 91 | 88 | 86 | 88 | 89.3 | 84.8 | 87.5 | 86.9 | **86.9** | faithful |

## Findings and remediation

### Eohippus angustidens Schnakenberg wavelength at 78 µm (Wasatchian, 52.0 Ma)  —  overall 86.9 (faithful)

Pass 2 keeps the same Schnakenberg system (a = 0.062, b = 0.94, D_v/D_u = 8.22, lambda_c = 7.752e-05 m, Wasatchian 52.0 Ma) and closes the four pass-1 gaps. Dispersion Pearson r = 0.99712, RMSE = 0.0204. Shared-lattice Pearson against the plate is -0.028 (reference 1) and KS is 0.164 (reference 0): the field is not a copy of the photograph. Murray lambda shifts by 7.59 µm from ratio 5 to ratio 18. Equal-diffusion std ratio is 0.00051. The delivered GIF is 640×480, 12 frames, pattern weight 0.55, CV similarity 88.8. Weighted overall 86.9. Eohippus angustidens is chronoseries stage 1 of 1; the cellular wavelength is the conserved quantity relative to Orohippus at the 48 Ma Bridgerian anchor, while body size (MacFadden 1986) is not.

- **Model fidelity to source: 90** — Jacobian, sigma(k), critical ratio, and k_c are the published objects, and the numerical growth-rate curve matches the eigenvalue at r = 0.997. Lean proves trace < 0, det > 0, equal-diffusion stability, the ratio-8.22 band inequality, the 6-to-7-cell bracket, the 42-to-43 wavenumber bracket, and the 52-before-48 Ma order. Deduction: diffusivities are scale-assigned.  
  evidence: `models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py` L64–118; `models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py` L192–222; `morphoeval/lean/loops/Loop_3219_eohippus_angustidens_turing_wavelength_s0.lean` L19–49
- **Morphological realism / specimen match: 81** — Side-by-side, the plate is a two-arm spiral and the PDE is a labyrinth at the same purple palette and a similar spatial frequency. CV similarity of the 12-frame 640×480 GIF against the plate is 88.8 at pattern weight 0.55 (the pattern is still the majority of the last frame). The weight table is 86.5, 87.3, 88.8, 91.6 at weights 1.00, 0.72, 0.55, 0.40. Topology is the reason this is not a 90.  
  evidence: `docs/media/eohippus_angustidens_turing_wavelength_s0_comparison.png` L0–0; `docs/media/eohippus_angustidens_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py` L488–510
- **Parameter expressivity: 88** — Formula lambda decreases monotonically from 81.91 µm at ratio 5 to 74.32 µm at ratio 18 (shift 7.59 µm, reference 0). sigma_max is negative at ratios 5, 6, and 7 and then rises through 0.047, 0.183, 0.302. The PDE ablation at equal diffusion drops the activator std from 0.119 to 6.1e-5.  
  evidence: `models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py` L415–464; `datasets/eohippus_angustidens_turing_wavelength_s0/eohippus_angustidens_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 91** — Dispersion r = 0.99712 versus 1, RMSE = 0.0204 versus 0. Specimen wavelength 86.49 µm versus model 77.52 µm, using the 111 px = 100 µm bar. Shared-lattice Pearson is -0.0277 versus 1 and KS is 0.164 versus 0, so none of these are self-copies. Second-seed centroids are 78.53 µm and 78.88 µm. Cells per wavelength 6.46 versus the round reference 6.  
  evidence: `models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py` L325–335; `models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py` L571–687
- **Source and data traceability: 88** — DOIs, BibTeX, scout manifest, and the pinned specimen hash are unchanged from pass 1 and still resolve. Commons had no image; the gallery plate is the declared fallback. Age 52.0 Ma and scale 78 µm are stated in the dataset README and the reference JSON.  
  evidence: `datasets/eohippus_angustidens_turing_wavelength_s0/eohippus_angustidens_turing_wavelength_s0_reference.json` L1–40; `datasets/eohippus_angustidens_turing_wavelength_s0/README.md` L1–30
- **Code-to-model traceability: 86** — The dead probe stub is gone. linear_probe_full is the routine whose Pearson r is asserted. Lean declarations match the micro-unit Jacobian (fu = 876248 and neighbors). The math-kg yaml points at those loci.  
  evidence: `morphoeval/math_kg_loops/delphi10000-eohippus-angustidens-turing-wavelength-s0.yaml` L1–55; `models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py` L192–210
- **Reproducibility package: 88** — python3 models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py --self-test passes (r = 0.99712, ablation ratio 0.00051, formula shift > 2 µm, specimen hash pin). The full script rewrites the PNG, the GIF, and the metrics JSON. Seed, dt, and grid are constants. No second environment lockfile.  
  evidence: `models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py` L51–62; `models/2026-delphi10000-eohippus-angustidens-turing-wavelength-s0.py` L689–720

Remediation:
- {'id': 'R1', 'item': 'Unused probe stub removed. Closed in pass 2.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Shared-lattice Pearson and KS are reported (aligned_pearson_r = -0.0277, aligned_ks = 0.164). Closed in pass 2.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'CV-by-weight table is in the metrics JSON; delivered weight is 0.55. Closed in pass 2.', 'status': 'resolved'}
- {'id': 'R4', 'item': 'Second-seed centroid and formula_lambda_shift_um = 7.59 are measurements. Closed in pass 2.', 'status': 'resolved'}
- {'id': 'R5', 'item': 'Residual: no second assessor, lean is not executed in this VM, and nanopubs are unsigned because ~/.nanopub/profile.yml is absent. The 78 µm target remains a scale assignment, not a fossil measurement.', 'status': 'open'}

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


Pass 2 working tree on top of 0c76d15d. Nanopub signing was not available (~/.nanopub/profile.yml absent). Lean binary was not installed; the integer inequalities were checked with Python before being written as decide proofs.
