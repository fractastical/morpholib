# MorphoEval report — audit `2026-10-27-drosophila-subobscura-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `a89be184699d71a03a1673e1f49d8998944a483a` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila subobscura Turing wavelength at 3.0 µm (Pass 1) (`models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py`) | 82 | 64 | 58 | 74 | 88 | 80 | 84 | 74.0 | 67.8 | 84.3 | 74.6 | **74.6** | mostly faithful |

## Findings and remediation

### Drosophila subobscura Turing wavelength at 3.0 µm (Pass 1)  —  overall 74.6 (mostly faithful)

Pass 1 integrates Schnakenberg kinetics on a periodic 6-wavelength patch pinned to 2.97e-6 m (3.0 µm, Holocene, 0 Ma). The dispersion sample already correlates with the discrete eigenvalue (Pearson r = 0.999998), and the chronospecies ratios versus Drosophila simulans (27.80) and Drosophila suzukii (33.15) are computed. The Pass 1 path does not sweep Du, does not run the equal-diffusion ablation, and replaces the specimen mask with an all-ones field, so the rendered frame is the activator alone. Huey et al. (2000) measured a latitudinal wing-size cline, not a 3 µm pigment spacing.

- **Model fidelity to source: 82** — The Jacobian, the critical-wavenumber formula, and the explicit integrator match the cited reaction-diffusion law. Pass 1 does not yet show that equal diffusivities close the Turing band.  
  evidence: `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L85–99; `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L497–499
- **Morphological realism / specimen match: 64** — Pass 1 paints the activator field by itself (pass_id == 1). The Cooper collage is loaded but the mask is discarded, so the frame is not a specimen composite.  
  evidence: `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L415–416; `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L503–509
- **Parameter expressivity: 58** — A single (Du, Dv) pair is integrated. The Du sweep that should lengthen λ_c as sqrt(Du) is not run when pass_id is 1.  
  evidence: `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L497–498
- **Morphometric validation: 74** — Linear-regime growth rates already match the discrete eigenvalue (r = 0.999998, RMSE = 0.000495 on 12 modes). Specimen fractions are placeholders (both 1.0) and there is no ablation contrast.  
  evidence: `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L131–210
- **Source and data traceability: 88** — Huey 2000, Turing 1952, Schnakenberg 1979, and Maini 1997 are pinned by DOI. The CC BY 2.0 Cooper collage is SHA-256 pinned. The wing-size papers are not asked to carry the 3 µm claim.  
  evidence: `datasets/drosophila_subobscura_turing_wavelength_s0/paper_citations.bib` L1–10; `datasets/drosophila_subobscura_turing_wavelength_s0/paper_citations.bib` L34–43
- **Code-to-model traceability: 80** — critical_mode and chronospecies_contrast cite the formulae they implement. Pass 1's skipped controls are visible as the empty branch, which is traceable but incomplete.  
  evidence: `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L325–340
- **Reproducibility package: 84** — Seed 3206, --self-test, and the dataset README name the 3.0 µm scale and 0 Ma age. Pass 1 self-test does not require the sweep or the ablation.  
  evidence: `datasets/drosophila_subobscura_turing_wavelength_s0/README.md` L5–8; `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L651–680

Remediation:
- {'id': 'R1', 'item': 'Sweep Du at fixed Dv/Du and show that λ_c increases with sqrt(Du).'}
- {'id': 'R2', 'item': 'Add the equal-diffusion ablation and require its maximum growth rate to be negative.'}
- {'id': 'R3', 'item': 'Segment the Cooper 2015 collage and composite the activator into the GIF and the comparison figure.'}
- {'id': 'R4', 'item': 'Report two non-identical specimen estimators (Otsu versus border contrast) plus the KS distance of the growth-rate sample.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila subobscura Turing wavelength at 3.0 µm (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `ddf700f45c258daa80240c53b23e60ad30c6cfe8bec1c0e8b9940daf13a758b9`
- **input_dataset_sha256**: `datasets/drosophila_subobscura_turing_wavelength_s0/drosophila_subobscura_specimen.jpg` (`5366640b506b5317…`), `datasets/drosophila_subobscura_turing_wavelength_s0/drosophila_subobscura_turing_wavelength_s0_reference.json` (`0b9b69eda941b675…`)
- `docs/media/drosophila_subobscura_turing_wavelength_s0_comparison.png` — artifact_sha256 `243f026cc7f99290…`, 876,428 B, 1344×1008 px

| measurement | value | reference | method |
|---|---:|---|---|
| specimen image Otsu foreground fraction | 0.4649 area fraction | 0.7715 | Otsu threshold on the 480×270 specimen luminance *(value is the Otsu fraction; reference is the border-contrast fraction; the two are not forced equal)* |
| specimen image structure-tensor coherency at sigma 1.6 | 0.6623 coherency | 0.6823 | mean structure-tensor coherency, Gaussian sigma 1.6, on the downscaled specimen *(same photograph, different scale and smoothing, so the coherency is not the fitness-script value)* |
| emergent Turing wavelength on the cuticle patch | 2.97 µm | 3.0 | radial FFT peak of the Schnakenberg activator, scaled so analytic λ_c = 2.97 µm *(relative error against 2.97 µm is 0.0000; patch length 17.82 µm)* |
| dispersion Pearson r of PDE growth versus discrete eigenvalue | 1.0 Pearson r | 0.98 | adaptive-horizon explicit Euler on grid modes 1..12 versus the five-point Laplacian eigenvalue *(RMSE 0.00050; KS statistic 0.0833; KS p 1.0000; n = 12)* |
| dispersion RMSE between PDE and discrete growth rates | 0.0005 1/time | 0.0 | root mean square residual of the linear-regime growth-rate sample |
| equal-diffusion ablation maximum growth rate | -0.2334 1/time | 0.0879 | continuum σ(k) with Dv reset to Du; negative means the Turing band is closed *(ablation field std after 1200 steps = 0.000000; patterned std = 0.1000)* |
| wavenumber ratio versus predecessor Drosophila simulans stripe | 27.798 k_detail / k_simulans | 27.0 | 2π/λ with λ_detail = 2.97e-6 m and λ_simulans = 8.256e-5 m |
| wavenumber ratio versus successor Drosophila suzukii wing spot | 33.1481 k_detail / k_suzukii | 33.0 | 2π/λ with λ_suzukii = 9.845e-5 m |
| Du sweep Pearson r of lambda_c against sqrt(Du) | 1.0 Pearson r | 0.99 | five Du values at fixed Dv/Du; analytic k_c^2 ∝ 1/Du so λ_c ∝ sqrt(Du) *(monotonic increasing = True)* |

## Environment at checkpoint


Pass 1 scores the --pass 1 path: Schnakenberg integration and dispersion only. Sweep, ablation, and specimen segmentation are absent on that path.
