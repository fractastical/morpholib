# MorphoEval report — audit `2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `dddc48a7fadc3d15f0038b8b8ef2a781309b9d10` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit -> 2026-10-27-ambystoma-mexicanum-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ambystoma mexicanum Schnakenberg onset wavelength at 184.3 µm (Pass 2) (`models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py`) | 89 | 76 | 88 | 87 | 91 | 86 | 88 | 88.7 | 80.1 | 88.5 | 85.2 | **85.2** | faithful |

## Findings and remediation

### Ambystoma mexicanum Schnakenberg onset wavelength at 184.3 µm (Pass 2)  —  overall 85.2 (faithful)

Same onset law as Pass 1: Schnakenberg (a, b) = (0.10, 0.90), Dv/Du = 9, λ_c = 184.3 µm = 0.0001843 m, Holocene age 0 Ma, eight-wavelength periodic patch. The nonlinear run now starts from noise (seed 3107) plus a shallow domain-scale source gradient, and the dominant shell that emerges is n = 8 (184.3 µm). Band-weighted wavelength is 176.47 µm. Continuum versus periodic-shell sweep: Pearson r = 0.9996, RMSE = 2.02 µm. Ratio sweep 9 → 14 moves the fastest wavelength from 187.2 µm to 212.1 µm. Ablation Dv = Du keeps max σ = −0.10. Specimen dark fraction 0.380 versus model 0.314; coherency 0.485 versus 0.904; luminance KS = 0.302; radial-spectrum r = 0.910. Computer-vision similarity of the 12-frame 640 px GIF against the micrograph is 87.5. The micrograph still has no scale bar.

- **Model fidelity to source: 89** — The homogeneous law is unchanged and checked: trace −0.2, det 1, ratio 9 above r_c ≈ 8.568, onset wavenumber fixed at 2π/184.3 µm, linear fastest mode 187.19 µm. Pass 2 integrates from noise. The shell that wins is n = 8, the onset shell, rather than a mode that was planted in the initial condition. A shallow source gradient of amplitude 0.12 is an extra term, stated as such. The dispersion cusp is the discriminant branch point (vertical tangent where disc = 0), drawn on a 900-point grid and labeled.  
  evidence: `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` L140–165; `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` L234–289
- **Morphological realism / specimen match: 76** — Panel B is now a broken spot-and-labyrinth field in the micrograph's circular stop, colored with melanophore and xanthophore means taken from the photograph, beside the real dendritic chromatophores. It is the same kind of image and no longer a perfect crystal. Coherency is still 0.904 against 0.485: the render is smoother and more locally aligned than stellate cells, and the blue edge of the micrograph is not in the model. Not a pixel copy. GIF-to-specimen computer-vision similarity on the last frame is 87.5.  
  evidence: `docs/media/ambystoma_mexicanum_turing_wavelength_s0_comparison.png` L0–0; `docs/media/ambystoma_mexicanum_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` L328–368
- **Parameter expressivity: 88** — Diffusivity scale s moves the fastest wavelength from about 112 µm to about 337 µm, and the periodic shell tracks it (figure C). Raising Dv/Du from 9 to 14, with the onset wavenumber held fixed, moves the fastest wavelength monotonically from 187.2 µm to 212.1 µm. Dv = Du remains stable.  
  evidence: `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` L172–219; `docs/media/ambystoma_mexicanum_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 87** — Continuum versus periodic-shell Pearson r = 0.999595, RMSE = 2.02 µm, residual CV = 0.0089. The reference for r is 1, and the measured value is not 1. Linear fastest wavelength 187.19 µm versus onset 184.3 µm. Noise-selected shell 184.3 µm (n = 8). Band-weighted wavelength 176.47 µm, which is not the reference. Specimen dark fraction 0.380 versus model 0.314. Specimen coherency 0.485 versus model 0.904. Luminance KS = 0.302. Radial-spectrum r = 0.910. No specimen number is copied onto its reference, and the photo is not given a false micrometre scale.  
  evidence: `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` L577–699; `datasets/ambystoma_mexicanum_turing_wavelength_s0/ambystoma_mexicanum_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 91** — Unchanged provenance from Pass 1, still accurate after the rerun: Crossref DOIs, CC BY-SA 4.0 chromatophore original, SHA-256 cc106cdbb5c28a40a47946801d35e6f13ada02665b8c773180123279a6cf22c8, and an explicit statement that 0.0001843 m is the scale coordinate.  
  evidence: `datasets/ambystoma_mexicanum_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/ambystoma_mexicanum_turing_wavelength_s0/ambystoma_mexicanum_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 86** — Docstring, dispersion function, and Lean module MathOfBiology.Loop_3107 still name the same Jacobian integers, the eight-wavelength patch, and the positive onset numerator. Pass 2 comments separate the homogeneous onset law from the shallow source gradient. Lean was not executed in this environment (no lean binary); the proofs are decide/integer and contain no sorry.  
  evidence: `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` L1–48; `morphoeval/lean/loops/Loop_3107_ambystoma_mexicanum_turing_wavelength_s0.lean` L1–62
- **Reproducibility package: 88** — Seed 3107, --self-test, pinned digest, deterministic spectral step. Pass 2 self-test passed: fastest λ 187.19 µm, Pearson r 0.9996, ablation σ −0.10, GIF 640×640 × 12 frames. Signing keys are absent, which is recorded rather than simulated.  
  evidence: `models/2026-delphi10000-ambystoma-mexicanum-turing-wavelength-s0.py` L708–770; `docs/media/ambystoma_mexicanum_turing_wavelength_s0_growth.gif` L0–0

Remediation:
- R5 remains open: this pass is still a single assessor (Grok 4.7). No second provider was available, and none was invented.
- Resolved from Pass 1 R1: the cos+cos lattice is no longer the initial condition. The n=8 shell is selected from noise.
- Resolved from Pass 1 R2: periodic-shell comparison gives Pearson r 0.999595 and RMSE 2.02 µm, not a bitwise copy of the reference 1.
- Resolved from Pass 1 R3: the dispersion cusp is labeled as the disc=0 branch point and sampled on 900 wavenumbers.
- Partly resolved from Pass 1 R4: the field is a broken labyrinth rather than a crystal. Coherency is still 0.904 versus 0.485, so stellate cell geometry is not claimed.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Ambystoma mexicanum Schnakenberg onset wavelength at 184.3 µm (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `948f3bc47cbf1c637386574fa0406a35e731104672ab92814b20a045ca1592e1`
- **input_dataset_sha256**: `datasets/ambystoma_mexicanum_turing_wavelength_s0/ambystoma_mexicanum_chromatophores_specimen.jpg` (`cc106cdbb5c28a40…`), `datasets/ambystoma_mexicanum_turing_wavelength_s0/ambystoma_mexicanum_turing_wavelength_s0_reference.json` (`5047ff540b2f84a0…`)
- `datasets/ambystoma_mexicanum_turing_wavelength_s0/ambystoma_mexicanum_turing_wavelength_s0_metrics.json` — artifact_sha256 `089f5533e94d2527…`, 7,953 B, 
- `docs/media/ambystoma_mexicanum_turing_wavelength_s0_comparison.png` — artifact_sha256 `eac1fafcca751364…`, 515,758 B, 1248×1032 px
- `docs/media/ambystoma_mexicanum_turing_wavelength_s0_growth.gif` — artifact_sha256 `8d6fd506a2a8af42…`, 1,612,487 B, 640×640 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| pearson_r_wavelength_sweep | 0.999595 1 | 1.0 | Pearson r between the continuum fastest wavelength and the nearest periodic shell on a 40-wavelength patch *(RMSE is nonzero because L/n is not the continuum peak)* |
| rmse_wavelength_um | 2.023564 µm | 0.0 | RMSE of coarse-grid λ against the fine-grid dispersion peak *(dispersion sweep uncertainty)* |
| residual_cv_wavelength_sweep | 0.008928 1 | 0.0 | CV of (λ_coarse − λ_fine) relative to mean |λ_fine| |
| linear_fastest_wavelength_um | 187.192853 µm | 184.3 | argmax_k σ(k) at Dv/Du = 9 with Du fixed by the onset formula *(near-threshold shift of the fastest mode away from the exact onset k_c)* |
| nonlinear_dominant_wavelength_um | 184.3 µm | 184.3 | Fourier shell with the most power after noise-seeded saturation on the periodic patch *(discrete shells are L/n, so a selected n=8 lands on 184.3 µm)* |
| band_weighted_wavelength_um | 176.472311 µm | 184.3 | Power-weighted mean of Fourier shells n=4..15 on the saturated activator *(harmonics and the shallow source gradient move this off the pure onset shell)* |
| turing_max_growth_rate | 0.018769 1/time | 0.0 | max σ(k) in the Turing regime Dv/Du = 9 *(positive σ is the wavelength-selection instability)* |
| ablation_max_growth_rate | -0.100003 1/time | 0.0 | negative control: max σ(k) when Dv = Du (ratio 1, below r_c) *(null diffusion-ratio control versus the ratio-9 regime)* |
| specimen_chromatophore_dark_area_fraction | 0.379717 1 | 0.314126 | quantile segmentation of dark melanophore pixels inside the micrograph field stop *(specimen segmentation; value is not the model fraction)* |
| model_spot_area_fraction | 0.314126 1 | 0.379717 | same dark-phase quantile applied to the rendered simulation, not to the photograph *(non-circular: simulation fraction versus specimen fraction)* |
| specimen_structure_tensor_coherency | 0.485252 1 | 0.904273 | mean structure-tensor coherency of the chromatophore micrograph *(specimen texture versus model texture)* |
| ks_statistic_luminance | 0.302 1 | 0.0 | two-sample KS statistic of field-stop luminance, specimen versus simulation; p-value stored in metrics *(ks p=2.418e-161)* |
| radial_spectrum_pearson_r | 0.910103 1 | 1.0 | Pearson r of 22-bin log radial FFT profiles, specimen micrograph versus rendered field *(specimen-match spectrum; not a self-comparison)* |

## Environment at checkpoint


Pass 2 script and regenerated comparison figure and growth GIF.
