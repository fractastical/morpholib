# MorphoEval report — audit `2026-10-27-eryops-megacephalus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `b44ba50e2b3311d637a471cbc44dcd63f77d40d0` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Eryops megacephalus Schnakenberg Turing wavelength at 127 µm (Cisuralian, 295 Ma) (`models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py`) | 90 | 78 | 84 | 80 | 90 | 86 | 88 | 88.0 | 78.8 | 88.1 | 84.3 | **84.3** | faithful |

## Findings and remediation

### Eryops megacephalus Schnakenberg Turing wavelength at 127 µm (Cisuralian, 295 Ma)  —  overall 84.3 (faithful)

Pass 1 overall 84.3/100. The Schnakenberg critical-wavenumber law is in the script and the dispersion correlation on modes m=2..16 is r=0.99999998, but the published composite still modulates the gallery wall (unmasked cv_sim_similarity 94.4) and the audit does not yet credit the bone mask, the KS/RMSE report, or the negative controls as closed remediation.

- **Model fidelity to source: 90** — steady_state, jacobian, and scale_diffusivities implement k_c^2 = gamma (D_v f_u + D_u g_v) / (2 D_u D_v) with a=0.085, b=1.22, and a pixel scale that makes lambda_c = 12 px = 127 µm. Trace is negative and the reference drive h is positive. The explicit Euler scheme is a reconstruction, not a line-by-line copy of Schnakenberg 1979.  
  evidence: `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L112–176; `morphoeval/lean/loops/Loop_3222_eryops_megacephalus_turing_wavelength_s0.lean` L36–51
- **Morphological realism / specimen match: 78** — The CC BY 4.0 AMNH mount is the specimen (SHA-256 pinned). Pass 1 composites the ocellus field over the whole frame, including the gallery wall and labels. Unmasked cv_sim_similarity is 94.4, but the placement is not anatomically restricted to bone. That is the main Pass-1 specimen gap.  
  evidence: `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L459–490; `docs/media/eryops_megacephalus_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 84** — D_v/D_u from 20 to 80 shortens the reference wavelength monotonically (4.013 to 3.859 reference px). Alpha grows across the 12 frames. The equal-diffusion and autocatalysis runs exist, but Pass 1 does not yet treat them as a closed control panel.  
  evidence: `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L356–379; `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L211–286
- **Morphometric validation: 80** — Dispersion Pearson r on modes m=2..16 is 0.99999998 with RMSE 1.14e-4. Pass 1 records that fit but does not yet score the KS statistic (0.0667) or the non-circular specimen measurements (bone fraction 0.3208 vs null 0.5; aspect 1.3358 vs null 1) as closed validation. The nonlinear FFT peak landing on exactly 127 µm is the design mode, not an independent histological measurement.  
  evidence: `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L288–353; `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L725–812
- **Source and data traceability: 90** — Scout manifest, reference JSON, BibTeX, and the CC BY 4.0 AMNH file with SHA-256 11fc4e4eec64d5f1cd5cc143b6ac1832f8ea67ac18f4d0a5ee25c2af919f0b66. OpenAlex keyword search returned no papers; DOIs were resolved by identifier lookup. Scale 127 µm and age 295 Ma are stated in the reference.  
  evidence: `datasets/eryops_megacephalus_turing_wavelength_s0/eryops_megacephalus_turing_wavelength_s0_reference.json` L1–80; `datasets/eryops_megacephalus_turing_wavelength_s0/paper_citations.bib` L1–70
- **Code-to-model traceability: 86** — Module docstring maps the Jacobian and k_c to Turing 1952 and Schnakenberg 1979. The math-KG loop file points at the same Lean declarations. Pass 1 does not yet tie every control panel to a scored claim.  
  evidence: `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L1–55; `morphoeval/math_kg_loops/delphi10000-eryops-megacephalus-turing-wavelength-s0.yaml` L1–34
- **Reproducibility package: 88** — --self-test checks the wavelength, the dispersion correlation above 0.98, control decay, GIF size, and the specimen hash. Seed is 3222. Pass 1 still owes a measurements sidecar and a knowledge-graph build.  
  evidence: `models/2026-delphi10000-eryops-megacephalus-turing-wavelength-s0.py` L815–869; `docs/media/eryops_megacephalus_turing_wavelength_s0_comparison.png` L0–0

Remediation:
- Restrict the ocellus modulation to the eroded Otsu bone mask so the gallery wall is not painted.
- Report the KS statistic and RMSE of sigma(k) next to the Pearson correlation, and keep value != reference on the specimen metrics.
- Put the equal-diffusion and no-autocatalysis trajectories on the comparison figure.
- Write the measurements sidecar and rebuild the knowledge graph from this audit.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.5]** `specimen_comparison` → `specimen-match`: Restrict the 127 µm ocellus field to the eroded Otsu bone mask of the AMNH mount.
- **[OPEN · P=1.1]** `morphometric_validation` → `morphometric-validation`: Publish KS and RMSE for sigma(k) and non-circular specimen metrics.
- **[OPEN · P=1.1]** `parameter_sweep` → `parameter-expressivity`: Show equal diffusion and autocatalysis ablation beside the diffusion-ratio sweep.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Eryops megacephalus Schnakenberg Turing wavelength at 127 µm (Cisuralian, 295 Ma)
- **spec_fingerprint** (semantic model + inputs): `51cd1fdffca7c86caa7bb398d56aaa419b30f1491ad7e8b4fe11b7153e0b9b21`
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

Pass 1 of loop 2026-10-27-eryops-megacephalus-turing-wavelength-s0-loop. Checkpoint is the loop commit that adds the script, specimen, and renders. Nanopub signing was not available (~/.nanopub/profile.yml absent).
