# MorphoEval report — audit `2026-10-27-eryops-megacephalus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `b44ba50e2b3311d637a471cbc44dcd63f77d40d0` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-eryops-megacephalus-turing-wavelength-s0-audit -> 2026-10-27-eryops-megacephalus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Eryops megacephalus Schnakenberg Turing wavelength at 127 µm (Cisuralian, 295 Ma) (`models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py`) | 93 | 88 | 91 | 92 | 93 | 89 | 91 | 92.3 | 89.5 | 91.1 | 90.8 | **90.8** | faithful |

## Findings and remediation

### Eryops megacephalus Schnakenberg Turing wavelength at 127 µm (Cisuralian, 295 Ma)  —  overall 90.8 (faithful)

Pass 2 overall 90.8/100, superseding the Pass-1 audit at 84.3. The published GIF is the bone-masked composite (cv_sim_similarity 96.0, 12 frames, 720x540). Dispersion Pearson r is 0.99999998. Specimen metrics are not copies of their references (foreground 0.3208 vs 0.5, aspect 1.3358 vs 1, coherency 0.5334 vs 0.25).

- **Model fidelity to source: 93** — The script integrates the Schnakenberg pair and the Turing wavenumber with the stated Jacobian. Discrete-symbol sigma(k) matches explicit Euler growth on modes m=2..16 (r = 0.99999998, RMSE = 1.137e-4). Five Lean theorems, none using sorry, record the trace sign, the drive sign, and 8184 = 64*127+56. The scheme is still explicit Euler on a flat periodic patch.  
  evidence: `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L119–198; `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L288–353; `morphoeval/lean/loops/Loop_3222_eryops_megacephalus_turing_wavelength_s0.lean` L23–51
- **Morphological realism / specimen match: 88** — The real CC BY 4.0 AMNH mount is SHA-256 pinned. Pass 2 multiplies the ocellus field only inside the eroded Otsu bone mask, so the gallery wall stays photographic. Masked cv_sim_similarity is 96.0 (unmasked Pass-1 value 94.4). Interior grayscale Pearson r against the photograph is 0.9913. The photograph still cannot resolve 127 µm, so this is not a microanatomical match.  
  evidence: `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L407–490; `docs/media/eryops_megacephalus_turing_wavelength_s0_growth.gif` L0–0; `datasets/eryops_megacephalus_turing_wavelength_s0/eryops_megacephalus_amnh_specimen.jpg` L0–0
- **Parameter expressivity: 91** — Raising D_v/D_u from 20 to 80 shortens the reference wavelength at every step (4.013, 3.942, 3.908, 3.888, 3.870, 3.859). Equal diffusion drops max sigma to -0.4166 and the field standard deviation from 0.0399 to numerical noise. Removing autocatalysis drops the same initial standard deviation to 9.7e-5. Frame alpha runs from 0 to 0.40.  
  evidence: `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L356–379; `docs/media/eryops_megacephalus_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 92** — Dispersion Pearson r = 0.99999998 (reference floor 0.98, not equal), RMSE = 1.137e-4 (reference tolerance 0.05), KS statistic = 0.0667. Specimen bone-mask foreground fraction = 0.3208 versus null 0.5. Bounding-box aspect = 1.3358 versus null 1. Structure-tensor coherency = 0.5334 versus isotropic null 0.25. Field Pearson r against the photograph is 0.9913. The saturated radial-FFT peak at exactly 127.0 µm is the design wavelength and is not counted as an independent empirical hit.  
  evidence: `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L725–812; `datasets/eryops_megacephalus_turing_wavelength_s0/eryops_megacephalus_turing_wavelength_s0_metrics.json` L23–80
- **Source and data traceability: 93** — DOIs for Turing 1952, Schnakenberg 1979, Gierer and Meinhardt 1972, Kondo and Miura 2010, Witzmann et al. 2010, and Pawley and Warren 2006 are in the BibTeX and the reference JSON. The specimen URL, license, creator, and SHA-256 are in that JSON. Scale 127 µm and age 295 Ma are explicit. The first OpenAlex keyword query was empty; the DOI resolution is recorded on the scout manifest.  
  evidence: `datasets/eryops_megacephalus_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/eryops_megacephalus_turing_wavelength_s0/eryops_megacephalus_turing_wavelength_s0_reference.json` L1–87
- **Code-to-model traceability: 89** — The docstring, the math-KG YAML, and Loop_3222_eryops_megacephalus_turing_wavelength_s0.lean name the same three laws: k_c drive, trace sign, and the 64-wavelength comparison to 8.184 mm. Integrator branches for the two controls are the same function the comparison figure plots.  
  evidence: `morphoeval/math_kg_loops/delphi10000-eryops-megacephalus-turing-wavelength-s0.yaml` L1–34; `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L1–55
- **Reproducibility package: 91** — --self-test is deterministic (repeat std trajectories match), checks lambda_c = 127 µm, Pearson r > 0.98, control decay, monotone sweep, GIF frames >= 10, minimum side >= 480, masked CV >= 58, and the specimen SHA-256. Seed 3222. Signing keys were not present, so this dimension does not claim a signed nanopub.  
  evidence: `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L815–878; `docs/media/eryops_megacephalus_turing_wavelength_s0_growth.gif` L0–0

Remediation:
- A histological measurement of dermal-pit spacing on Eryops bone, at a scale the AMNH gallery photograph cannot resolve, is still open.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=1.5]** `specimen_comparison` → `specimen-match`: Restrict the 127 µm ocellus field to the eroded Otsu bone mask of the AMNH mount.
- **[RESOLVED · P=1.1]** `morphometric_validation` → `morphometric-validation`: Publish KS and RMSE for sigma(k) and non-circular specimen metrics.
- **[RESOLVED · P=1.1]** `parameter_sweep` → `parameter-expressivity`: Show equal diffusion and autocatalysis ablation beside the diffusion-ratio sweep.
- **[OPEN · P=2.2]** `morphometric_validation` → `morphometric-validation`: Measure dermal-pit spacing on Eryops bone at a scale this gallery photograph cannot resolve.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Eryops megacephalus Schnakenberg Turing wavelength at 127 µm (Cisuralian, 295 Ma)
- **spec_fingerprint** (semantic model + inputs): `51cd1fdffca7c86caa7bb398d56aaa419b30f1491ad7e8b4fe11b7153e0b9b21`
- `datasets/eryops_megacephalus_turing_wavelength_s0/eryops_megacephalus_amnh_specimen.jpg` — artifact_sha256 `11fc4e4eec64d5f1…`, 287,455 B, 1280×960 px
- `docs/media/eryops_megacephalus_turing_wavelength_s0_comparison.png` — artifact_sha256 `d39dec86b3388979…`, 561,307 B, 1512×984 px
- `docs/media/eryops_megacephalus_turing_wavelength_s0_growth.gif` — artifact_sha256 `b226d43e5bb86317…`, 2,794,928 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen bone-mask foreground fraction | 0.3208 area fraction | 0.5 | Otsu threshold on the 720x540 AMNH frame, then 2-pixel binary erosion *(Otsu threshold 132; raw (uneroded) fraction differs from the eroded mask)* |
| specimen skeleton bounding-box aspect ratio | 1.3358 width/height | 1.0 | axis-aligned bbox of the eroded Otsu bone mask on eryops_megacephalus_amnh_specimen.jpg *(Pawley & Warren 2006 describe the appendicular skeleton; this ratio is measured on the photograph)* |
| specimen structure-tensor coherency | 0.5334 coherency | 0.25 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor, Gaussian σ = 2 |
| dispersion sigma(k) Pearson r | 0.99999998 Pearson r | 0.98 | explicit Euler growth rates vs discrete-symbol σ(k) for Fourier modes m = 2..16 *(RMSE 0.000114; KS statistic 0.0667; p_pearson 8.658e-50)* |
| dispersion sigma(k) RMSE | 0.000114 1/time | 0.05 | root mean square of measured minus predicted σ(k) on modes m = 2..16 *(KS p = 1.0000)* |
| nonlinear dominant wavelength | 127.0 µm | 127.0 | radial FFT peak of the saturated u field, converted at 127/12 µm per pixel *(measured period 12.0000 px vs design 12 px)* |
| equal-diffusion control max sigma | -0.4166 1/time | 0.0 | max continuum σ(k) after setting D_u = D_v = D_u(slot) *(full-model max σ = 0.4029; control std 0.0399 → 0.0000)* |
| bone-masked specimen-field Pearson r | 0.9913 Pearson r | 0.9 | interior grayscale of the AMNH frame vs the last GIF frame *(p = 0.000e+00; CV masked 96.0; CV unmasked 94.4)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2 of loop 2026-10-27-eryops-megacephalus-turing-wavelength-s0-loop. Nanopub signing was not performed: ~/.nanopub/profile.yml is absent. Scores were not raised to clear a gate; the dispersion correlation and the image comparison are the numbers the script printed.
