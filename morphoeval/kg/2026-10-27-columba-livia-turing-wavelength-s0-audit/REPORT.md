# MorphoEval report — audit `2026-10-27-columba-livia-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in https://github.com/fractastical/morpholib/issues/1

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Columba livia Turing wavelength at 256 um (Holocene, chronoseries 1/1) (`models/2026-delphi10000-columba-livia-turing-wavelength-s0.py`) | 90 | 66 | 88 | 90 | 88 | 86 | 88 | 89.3 | 75.0 | 87.5 | 83.0 | **83.0** | faithful |

## Findings and remediation

### Columba livia Turing wavelength at 256 um (Holocene, chronoseries 1/1)  —  overall 83.0 (faithful)

Pass 1 scores 83.0/100. The Schnakenberg critical wavelength is 256.000 um, only mode n=4 is unstable, and the IMEX growth rates match the analytic dispersion with Pearson r = 0.999889 (unscaled RMSE 0.0511). Equal diffusion drops the field standard deviation from 0.291 to 0.00516. The unfitted GIF (12 frames, 640x480) scores CV similarity 57.8 against the CC BY-SA 2.5 rock-dove photograph, so image fitness is suppressed at 63.0.

- **Model fidelity to source: 90** — Implements the Schnakenberg linearization (sigma_plus), the critical-wavenumber formula that pins lambda_c at 256 um, and an IMEX Fourier step. Four Lean theorems with zero sorry cover the determinant sign, the equal-diffusion ablation, the four-wavelength domain, and the Holocene 1/1 scale succession. Parameters are reconstructed for this scale rather than copied from a pigeon-specific published table, which is why the score is not higher.  
  evidence: `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L94–141; `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L173–220; `morphoeval/lean/loops/Loop_3151_columba_livia_turing_wavelength_s0.lean` L50–95
- **Morphological realism / specimen match: 66** — The comparison figure shows the Alan D. Wilson flight photograph and a melanin registration, and the script segments that file. The fitness test reads the GIF, which pass 1 rendered unfitted: 12 frames at 640x480, CV similarity 57.8, below the 58 gate. Image fitness is therefore suppressed (63.0). The photograph is a real rock dove with the two wing bars; it is not a micrometre micrograph.  
  evidence: `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L311–336; `docs/media/columba_livia_turing_wavelength_s0_growth.gif` L0–0; `docs/media/columba_livia_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 88** — At fixed gamma, analytic lambda falls monotonically as Dv rises from 9 to 24 (258.1 um to 246.0 um), and the number of unstable grid modes is non-decreasing (1, 1, 3, 3, 4). Anisotropy Dv_y/Du = 2.5 (subcritical) selects stripes; the isotropic tensor at the same Dv/Du = 10 selects spots. Both lock to mode n = 4 because that is the only unstable mode on this grid.  
  evidence: `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L222–244; `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L143–168
- **Morphometric validation: 90** — measure_morphoeval reports Pearson r = 0.999889 against the analytic dispersion (reference 1; unscaled RMSE 0.0511), stripe spacing CV, an ablation ratio of 0.0177 against a reference of 0, specimen foreground fraction 0.5577 against a balanced-frame null of 0.5, contour circularity 0.0145 against a circle null of 1, and a luminance KS of 0.387 against identical-distribution 0. The saturated stripe mode is n = 4, so the realized wavelength is 256 um on the 1024 um domain.  
  evidence: `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L471–591; `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L246–293
- **Source and data traceability: 88** — Six Crossref-checked DOIs are vendored in paper_citations.bib and the reference JSON. The specimen is Alan D. Wilson's CC BY-SA 2.5 file, SHA-256 74dfe533e6b396ac7c607cf39ec48ce3b867de1c26b527b027f86f0b61dc357e. The first broad OpenAlex query returned no works; the DOIs were resolved by follow-up title queries. The gallery JPEG labeled Columba livia is a synthetic spiral and was not used.  
  evidence: `datasets/columba_livia_turing_wavelength_s0/columba_livia_turing_wavelength_s0_reference.json` L1–90; `datasets/columba_livia_turing_wavelength_s0/paper_citations.bib` L1–60; `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L81–83
- **Code-to-model traceability: 86** — The module docstring names each equation's source. sigma_plus is the Turing growth rate, analytical_kc2 is the critical-wavenumber formula, and the Lean declarations are listed in the math-KG YAML with line loci. The time-scale factor S is documented as a clock, not a biological rate.  
  evidence: `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L1–24; `morphoeval/math_kg_loops/delphi10000-columba-livia-turing-wavelength-s0.yaml` L28–66
- **Reproducibility package: 88** — --self-test checks the specimen hash, lambda = 256 um, the sign of sigma at k_c, zero, and equal diffusion, that the only unstable mode is n = 4, Pearson r > 0.98, monotonic Dv sweep, bitwise repeatability, stripe mode 4, and the ablation ratio. Seed is 3151. Pass 1 was invoked as --render-mode unfitted.  
  evidence: `models/2026-delphi10000-columba-livia-turing-wavelength-s0.py` L594–626; `docs/media/columba_livia_turing_wavelength_s0_growth.gif` L0–0

Remediation:
- Re-render docs/media/columba_livia_turing_wavelength_s0_growth.gif with --render-mode fitted so the last frames register the simulated melanin on the rock-dove photograph. Pass 1 CV similarity was 57.8 (gate 58) and image_fitness was 63.0 (gate 70).
- Re-score specimen-match from that fitted GIF. Do not retune k_c or the audit weights to manufacture the pass; the dispersion Pearson r = 0.999889 already clears 0.98.
- Record signing as unavailable if ~/.nanopub/profile.yml is still absent. Do not invent a trusty URI.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Columba livia Turing wavelength at 256 um (Holocene, chronoseries 1/1)
- **spec_fingerprint** (semantic model + inputs): `19ee9eebdc9d074f8b89f48e2acdf67ee528590e0dcb3f1d1d367a19a14b1e0e`
- **input_dataset_sha256**: `datasets/columba_livia_turing_wavelength_s0/columba_livia_specimen.jpg` (`74dfe533e6b396ac…`), `datasets/columba_livia_turing_wavelength_s0/columba_livia_turing_wavelength_s0_reference.json` (`d0639e66b6546900…`), `datasets/columba_livia_turing_wavelength_s0/paper_citations.bib` (`2bafbef715549e81…`)
- `docs/media/columba_livia_turing_wavelength_s0_comparison.png` — artifact_sha256 `046b0d0f8827d6ba…`, 513,643 B, 1512×960 px
- `docs/media/columba_livia_turing_wavelength_s0_growth.gif` — artifact_sha256 `5c7eda50cc291565…`, 1,121,478 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| critical_wavelength_um | 256.0 um | 256.0 | analytic k_c at Dv/Du = 10; lambda = 2*pi/k_c * 16 um *(Linear law, not the saturated Eckhaus shift.)* |
| dispersion_pearson_r | 0.999889 1 | 1.0 | Pearson r of analytic vs IMEX growth rates on real eigenmodes n = 1..15 *(RMSE of scaled sigma = 1.5333; unscaled RMSE = 0.05111)* |
| dispersion_rmse_unscaled | 0.051109 1/time | 0.0 | RMSE of Re(sigma) after dividing out the time-scale factor S = 30 *(control comparison against the closed-form dispersion)* |
| saturated_stripe_wavelength_um | 256.0 um | 256.0 | dominant kx mode of the anisotropic saturated field; domain 1024 um / n *(selected mode n = 4; spacing CV = 0.0000)* |
| stripe_spacing_cv | 0.0 1 | 0.0 | CV of row-wise zero-crossing spacings on the stripe field *(dispersion of the realized bar spacing)* |
| ablation_to_stripe_std_ratio | 0.017726 1 | 0.0 | std(u) at Dv = Du divided by std(u) of the stripe regime, same seed and duration *(negative control vs the supercritical stripe field)* |
| specimen_foreground_fraction | 0.5577 1 | 0.5 | border-median contrast mask on columba_livia_specimen.jpg resized to 160x120 *(non-circular specimen segmentation metric)* |
| specimen_contour_circularity | 0.0145 1 | 1.0 | 4*pi*area/perimeter^2 of the border-contrast silhouette *(the flying bird is a thin, non-circular contour)* |
| specimen_luminance_ks | 0.387 1 | 0.0 | two-sample KS between specimen luminance and stripe melanin, 4000 pixels each *(goodness of fit of the pigment histogram against the photograph; not expected to be zero)* |
| ocellar_peak_count | 20 count | 16 | strict local maxima of the isotropic field above the 75th percentile *(spot/ocellus regime at the same k_c; count need not equal the square lattice)* |
| diffusion_length_predecessor_um | 24.1043 um | 256.0 | sqrt(Du0/gamma0) * 16 um *(predecessor scale at 0 Ma; not a chronospecies. Ratio to 256 um is about 10.6)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 of loop 3151. The solver, dispersion test, ablation, and specimen segmentation are in the script. The animated GIF was written with --render-mode unfitted, so its last frame is the bare stripe field. Signing keys were not present (~/.nanopub/profile.yml missing); nanopubs from this audit are unsigned. An independent second-company assessor was not run in this session.
