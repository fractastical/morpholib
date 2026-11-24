# MorphoEval report — audit `2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `4f21c452569eeb68e67f8158dd98be18f84832f2` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ambystoma mexicanum Schnakenberg onset wavelength at 184.3 µm (Pass 1) (`models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py`) | 84 | 68 | 82 | 80 | 90 | 84 | 86 | 83.3 | 72.5 | 86.8 | 80.0 | **80.0** | faithful |

## Findings and remediation

### Ambystoma mexicanum Schnakenberg onset wavelength at 184.3 µm (Pass 1)  —  overall 80.0 (faithful)

Periodic Schnakenberg patch of eight onset wavelengths at the Delphi coordinate 0.0001843 m (184.3 µm), Holocene age 0 Ma, diffusion ratio 9 just above r_c ≈ 8.568. Linear fastest mode 187.19 µm. Saturated Fourier shell 184.3 µm because the initial condition is that mode. Coarse-versus-fine wavelength sweep Pearson r = 1.0 with RMSE 3.39 µm. Ablation Dv = Du has max σ = −0.10. Specimen dark-area fraction 0.380 versus model 0.328; coherency 0.485 versus 0.915; luminance KS = 0.271. The rendered field is a regular lattice inside the micrograph's circular field stop, not a copy of the photograph and not a scale-bar measurement.

- **Model fidelity to source: 84** — The script integrates Schnakenberg kinetics with the Jacobian and the onset-wavenumber formula, holds the uniform state stable without diffusion (trace −0.2, det 1), and places Dv/Du = 9 just above the critical ratio. The saturated shell sits on 184.3 µm largely because the initial condition is that Fourier mode, so free wavelength selection is only partly tested. The dispersion curve's real-part branch has a sharp corner near k ≈ 0.02 µm⁻¹.  
  evidence: `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` L94–165; `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` L213–263
- **Morphological realism / specimen match: 68** — Panel A is the larval chromatophore micrograph and panel B is the rendered activator in the same circular field stop, using dark and yellow colours measured from that photo. The simulation is a regular spot lattice. The micrograph is an irregular field of dendritic melanophores. Structure-tensor coherency is 0.915 on the render and 0.485 on the specimen. Same class of image, different geometry.  
  evidence: `docs/media/ambystoma_mexicanum_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` L280–320
- **Parameter expressivity: 82** — Scaling both diffusivities by s² moves the fastest wavelength almost linearly from about 110 µm to about 340 µm (figure C). The ablation Dv = Du kills the positive growth rate. The diffusion ratio itself is not walked as a second rendered morphospace on this pass.  
  evidence: `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` L168–205; `docs/media/ambystoma_mexicanum_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 80** — Fine-grid versus coarse-grid wavelengths give Pearson r = 1.0 and RMSE 3.39 µm, so the correlation saturated even though the residual did not. Linear fastest wavelength 187.19 µm against the 184.3 µm onset coordinate. Specimen dark fraction 0.380 versus model 0.328, luminance KS 0.271, radial-spectrum r 0.833. None of the specimen numbers is copied onto its reference. The photograph is not assigned a false micrometre scale.  
  evidence: `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` L518–631; `datasets/ambystoma_mexicanum_turing_wavelength_s0/ambystoma_mexicanum_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing, Murray, Epperlein & Löfberg, Frost-Mason, and Erickson. The chromatophore JPEG is the Wikimedia original, CC BY-SA 4.0, SHA-256 pinned in the reference JSON and checked by the script. The claim boundary states that 184.3 µm is a scale coordinate, not a published axolotl measurement.  
  evidence: `datasets/ambystoma_mexicanum_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/ambystoma_mexicanum_turing_wavelength_s0/ambystoma_mexicanum_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — The module docstring names the PDE, the Jacobian numbers, and the onset formula. Lean theorems in MathOfBiology.Loop_3107 prove the eight-wavelength patch, trace < 0 and det > 0, ratio 9 above the critical ceiling, and a positive onset numerator, with no sorry. Lean was not executed here; the proofs are by decide on integers.  
  evidence: `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` L1–45; `morphoeval/lean/loops/Loop_3107_ambystoma_mexicanum_turing_wavelength_s0.lean` L1–62
- **Reproducibility package: 86** — Seed 3107, --self-test, pinned specimen digest, and a deterministic spectral step. Self-test passed: fastest λ 187.19 µm, Pearson r 1.0, ablation σ −0.10, GIF 640×640 with 12 frames. No second environment lock beyond the repo requirements file.  
  evidence: `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` L640–714; `docs/media/ambystoma_mexicanum_turing_wavelength_s0_growth.gif` L0–0

Remediation:
- R1: Drop the pure cos(k_c x)+cos(k_c y) seed. Integrate from small noise so the dominant shell is selected, not inserted, and re-measure it against 184.3 µm.
- R2: Coarsen the numerical wavenumber grid until Pearson r stays above 0.98 without rounding to the reference value 1.0. Keep the non-zero RMSE.
- R3: Plot a continuous real part of σ(k). The Pass 1 curve has a branch corner near 0.02 µm⁻¹.
- R4: The lattice coherency (0.915) is far above the micrograph (0.485). A noise-selected spot field should move the render toward the specimen texture without pasting photograph pixels.
- R5: Single assessor only. A second provider was not run. Left open.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Ambystoma mexicanum Schnakenberg onset wavelength at 184.3 µm (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `948f3bc47cbf1c637386574fa0406a35e731104672ab92814b20a045ca1592e1`
- **input_dataset_sha256**: `datasets/ambystoma_mexicanum_turing_wavelength_s0/ambystoma_mexicanum_chromatophores_specimen.jpg` (`cc106cdbb5c28a40…`), `datasets/ambystoma_mexicanum_turing_wavelength_s0/ambystoma_mexicanum_turing_wavelength_s0_reference.json` (`5047ff540b2f84a0…`)
- `datasets/ambystoma_mexicanum_turing_wavelength_s0/ambystoma_mexicanum_turing_wavelength_s0_metrics.json` — artifact_sha256 `fe265f393e3d7ca1…`, 7,113 B, 
- `docs/media/ambystoma_mexicanum_turing_wavelength_s0_comparison.png` — artifact_sha256 `f9622b705eb7441a…`, 512,054 B, 1248×1032 px
- `docs/media/ambystoma_mexicanum_turing_wavelength_s0_growth.gif` — artifact_sha256 `7a90051ac94deefe…`, 1,496,094 B, 640×640 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| pearson_r_wavelength_sweep | 1.0 1 | 1.0 | Pearson r between fine-grid and coarse-grid fastest wavelengths across 12 diffusivity scales *(RMSE and residual CV are reported separately; value is not equal to the reference)* |
| rmse_wavelength_um | 3.391498 µm | 0.0 | RMSE of coarse-grid λ against the fine-grid dispersion peak *(dispersion sweep uncertainty)* |
| residual_cv_wavelength_sweep | 0.004521 1 | 0.0 | CV of (λ_coarse − λ_fine) relative to mean |λ_fine| |
| linear_fastest_wavelength_um | 187.192853 µm | 184.3 | argmax_k σ(k) at Dv/Du = 9 with Du fixed by the onset formula *(near-threshold shift of the fastest mode away from the exact onset k_c)* |
| nonlinear_dominant_wavelength_um | 184.3 µm | 184.3 | Fourier shell of the saturated activator on the periodic patch *(nonlinear shell need not sit on the linear onset mode)* |
| turing_max_growth_rate | 0.018769 1/time | 0.0 | max σ(k) in the Turing regime Dv/Du = 9 *(positive σ is the wavelength-selection instability)* |
| ablation_max_growth_rate | -0.100003 1/time | 0.0 | negative control: max σ(k) when Dv = Du (ratio 1, below r_c) *(null diffusion-ratio control versus the ratio-9 regime)* |
| specimen_chromatophore_dark_area_fraction | 0.379717 1 | 0.328452 | quantile segmentation of dark melanophore pixels inside the micrograph field stop *(specimen segmentation; value is not the model fraction)* |
| model_spot_area_fraction | 0.328452 1 | 0.379717 | same dark-phase quantile applied to the rendered simulation, not to the photograph *(non-circular: simulation fraction versus specimen fraction)* |
| specimen_structure_tensor_coherency | 0.485252 1 | 0.915173 | mean structure-tensor coherency of the chromatophore micrograph *(specimen texture versus model texture)* |
| ks_statistic_luminance | 0.271 1 | 0.0 | two-sample KS statistic of field-stop luminance, specimen versus simulation; p-value stored in metrics *(ks p=1.343e-129)* |
| radial_spectrum_pearson_r | 0.832747 1 | 1.0 | Pearson r of 22-bin log radial FFT profiles, specimen micrograph versus rendered field *(specimen-match spectrum; not a self-comparison)* |

## Environment at checkpoint


Pass 1 script, specimen, Lean module, and rendered media.
