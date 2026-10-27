# MorphoEval report — audit `2026-10-27-dactylioceras-commune-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `8c95fb91d41c76d640c8216d24030d8fd3768487` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dactylioceras commune Turing k_c at 24 µm (pass 1, stage 1) (`models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py`) | 86 | 74 | 54 | 58 | 90 | 80 | 82 | 75.3 | 68.0 | 84.1 | 75.0 | **75.0** | mostly faithful |

## Findings and remediation

### Dactylioceras commune Turing k_c at 24 µm (pass 1, stage 1)  —  overall 75.0 (mostly faithful)

Pass 1 solves the Schnakenberg critical wavenumber so lambda_c is 24.19 µm (2.419e-05 m) at diffusion ratio 7.2, just above onset (6.724). Sigma at k_c is +0.0297. The stage-1 animation is a synthetic 44-rib coil, 12 frames at 640 px, computer-vision similarity 79.5 against the Whitby photograph. Pearson r of the dispersion curve, the equal-diffusion ablation, the diffusivity sweep, and the rib measurement on the real shell were not run.

- **Model fidelity to source: 86** — Stage 1 implements the Schnakenberg Jacobian and the textbook k_c formula and solves Du, Dv so lambda_c is 24.19 µm with sigma(k_c) = +0.0297, above the onset ratio 6.724. The numerical growth-rate comparison to sigma(k) is not in this pass.  
  evidence: `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L101–119; `datasets/dactylioceras_commune_turing_wavelength_s0/pass1_metrics.json` L1–19
- **Morphological realism / specimen match: 74** — The stage-1 GIF is a radial fibulate coil at 640 px and 12 frames. CV similarity to the Whitby photograph is 79.5. The coil has the right rib family and the wrong outline, matrix, and pyrite.  
  evidence: `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L410–431; `datasets/dactylioceras_commune_turing_wavelength_s0/pass1_growth.gif` L0–0
- **Parameter expressivity: 54** — The diffusion ratio and the wavelength are fixed constants in stage 1. No diffusivity scale was varied, so monotonic control of lambda_c was not shown.  
  evidence: `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L56–68; `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L585–592
- **Morphometric validation: 58** — Stage 1 reports lambda_c and one growth rate. It does not measure ribs on the photograph, does not report Pearson r or RMSE, and does not run a KS test.  
  evidence: `datasets/dactylioceras_commune_turing_wavelength_s0/pass1_metrics.json` L3–14
- **Source and data traceability: 90** — Crossref-checked DOIs for Turing, Schnakenberg, Meinhardt and Klingler, Fowler, García-Ruiz, and Lutz and Boyajian. The Whitby photograph is CC BY 2.0 with artist, URL, 57 mm width, and a SHA-256 pin.  
  evidence: `datasets/dactylioceras_commune_turing_wavelength_s0/paper_citations.bib` L1–74; `datasets/dactylioceras_commune_turing_wavelength_s0/dactylioceras_commune_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 80** — The module docstring and kc_sq map onto Schnakenberg and the k_c formula. Stage 1 does not yet call the dispersion, ablation, or segmentation functions those comments describe.  
  evidence: `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L1–48
- **Reproducibility package: 82** — Stage 1 checks the specimen SHA-256, uses seed 3159, and writes pass1_metrics.json. The --self-test gates (Pearson r, ablation, sweep, rib bounds) belong to stage 2 and were not the pass-1 command.  
  evidence: `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L241–248; `models/2026-delphi10000-dactylioceras-commune-turing-wavelength-s0.py` L560–592

Remediation:
- R1 parameter-regime sweep: scale both diffusivities and show lambda_c moves monotonically.
- R2 negative control: equal-diffusion ablation must be stable at k_c while the Turing ratio is unstable.
- R3 real specimen CV segmentation: measure outer-whorl rib harmonic, rib spacing, coherency, and foreground on the Whitby photograph.
- R4 goodness-of-fit: Pearson r and RMSE of numerical versus analytical sigma(k), plus a KS test of orientations.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=2.5]** `parameter_sweep` → `parameter-expressivity`: Diffusivity scale sweep with monotonic lambda_c.
- **[OPEN · P=3.0]** `morphometric_validation` → `morphometric-validation`: Dispersion Pearson r, RMSE, ablation, and specimen rib segmentation.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Dactylioceras commune Turing k_c at 24 µm (pass 1, stage 1)
- **spec_fingerprint** (semantic model + inputs): `250d9efaee72b1c988aa7b69152735c73f811d95d1843ab2b2c5433409d97e1d`
- **input_dataset_sha256**: `datasets/dactylioceras_commune_turing_wavelength_s0/dactylioceras_commune_specimen.jpg` (`c309b077fb0c3b68…`), `datasets/dactylioceras_commune_turing_wavelength_s0/dactylioceras_commune_turing_wavelength_s0_reference.json` (`a8f717691a831459…`)
- `datasets/dactylioceras_commune_turing_wavelength_s0/pass1_growth.gif` — artifact_sha256 `b8cc961dfa5bc20d…`, 936,417 B, 640×640 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing stripe critical wavelength | 24.19 µm | 24.0 | lambda_c = 2 pi / k_c with k_c^2 = (Dv f_u + Du g_v) / (2 Du Dv) *(reference is the integer catalogue label, not the calibrated SI wavelength)* |
| dispersion growth-rate Pearson r | 0.99991 correlation | 0.98 | corr(analytical sigma(k), numerical Schnakenberg growth rate) across the band around k_c *(RMSE = 0.000948)* |
| dispersion growth-rate RMSE | 0.000948 1/time | 0.05 | root mean square of analytical minus numerical sigma(k) |
| equal-diffusion ablation growth rate | -0.313981 1/time | 0.029661 | numerical sigma at k_c with Dv reset to Du *(negative control versus the positive Turing growth rate)* |
| segmented specimen shell outer-whorl rib harmonic | 44 ribs per revolution | 22 | angular FFT on the 0.75-0.95 radius band of the CC BY Whitby photograph *(outer harmonic and the halved primary prediction are not the same number)* |
| segmented specimen shell rib spacing | 3.4841 mm | 1.088 | outer circumference at 0.85 shell radius using the photographer's 57 mm width, divided by the rib harmonic *(measured Whitby rib spacing versus the predecessor organ-scale wavelength)* |
| specimen image structure-tensor coherency | 0.3619 coherency | 0.35 | mean structure-tensor coherency on the shell crop |
| segmented specimen foreground fraction | 0.5357 fraction | 0.5 | largest connected component below the 55th percentile of gray value |
| specimen versus Turing stripe orientation KS statistic | 0.4716 KS distance | 0.05 | two-sample KS test of Sobel orientations, photograph versus the Schnakenberg patch *(p = 1.451e-275)* |

## Environment at checkpoint


Pass 1 stage-1 path. Dispersion fit, ablation, sweep, and specimen segmentation are not in this run.
