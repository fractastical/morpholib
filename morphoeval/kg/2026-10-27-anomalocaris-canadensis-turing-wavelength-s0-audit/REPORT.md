# MorphoEval report — audit `2026-10-27-anomalocaris-canadensis-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `63ace896bba72dd95b1823861b5b9936be9cf066` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Anomalocaris canadensis Turing wavenumber k_c at 24 µm (508 Ma) (`models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py`) | 80 | 72 | 58 | 60 | 90 | 82 | 84 | 72.7 | 67.5 | 85.5 | 74.4 | **74.4** | mostly faithful |

## Findings and remediation

### Anomalocaris canadensis Turing wavenumber k_c at 24 µm (508 Ma)  —  overall 74.4 (mostly faithful)

Pass 1 integrates Schnakenberg kinetics on a no-flux corneal patch whose continuous k_c is pinned to 2.397e-05 m at 508 Ma, and places that field beside the Tyrrell cast TMP 2023.003.0003. The sweep, ablation, specimen segmentation, and dispersion goodness-of-fit are not yet accepted as closing evidence.

- **Model fidelity to source: 80** — The Jacobian, the k_c formula, and the Neumann integrator are the published Turing/Schnakenberg objects. Pass 1 does not yet treat the numerical growth-rate comparison as part of the fidelity score.  
  evidence: `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` L61–90; `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` L174–214
- **Morphological realism / specimen match: 72** — The plate shows the real Tyrrell cast beside the activator. The 24 µm spots are a registration, not a feature a viewer can find in the shale. Pass 1 does not yet score the cast segmentation.  
  evidence: `docs/media/anomalocaris_canadensis_turing_wavelength_s0_comparison.png` L0–0; `docs/media/anomalocaris_canadensis_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 58** — Du is fixed to the slot length. Pass 1 does not yet accept a sweep in which lambda_c moves monotonically, nor the equal-diffusion ablation, as expressivity evidence.  
  evidence: `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` L74–90
- **Morphometric validation: 60** — The slot length and the 508 Ma age are explicit. Pass 1 does not yet accept Pearson r, RMSE, KS, or specimen-mask morphometrics.  
  evidence: `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` L48–59; `datasets/anomalocaris_canadensis_turing_wavelength_s0/anomalocaris_canadensis_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Crossref and Europe PMC resolve the seed papers. The cast is SHA-256 pinned with catalogue number, licence, and artist. OpenAlex search returned HTTP 429; the dossier says so.  
  evidence: `datasets/anomalocaris_canadensis_turing_wavelength_s0/paper_citations.bib` L1–79; `datasets/anomalocaris_canadensis_turing_wavelength_s0/anomalocaris_canadensis_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 82** — The module docstring and the law functions name Turing 1952 and Schnakenberg 1979. Pass 1 does not yet tie the dispersion fitter and the self-test assertions into the score.  
  evidence: `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` L1–30
- **Reproducibility package: 84** — Seed 3113 and --self-test are present. Pass 1 does not yet treat the self-test's sweep, ablation, and Pearson gate as the reproducibility package.  
  evidence: `models/2026-delphi10000-anomalocaris-canadensis-turing-wavelength-s0.py` L567–610

Remediation:
- R1: Accept a Du diffusivity sweep and show that lambda_c increases monotonically across the stated range.
- R2: Accept the equal-diffusion negative control and the linear-reaction ablation, with a pattern-std ratio against the Turing run.
- R3: Segment the real Tyrrell cast (foreground fraction, circularity, structure-tensor coherency) and keep those values distinct from their references.
- R4: Report goodness-of-fit of the stencil growth rate against the numerical growth rate (Pearson r, RMSE, KS) and require Pearson r > 0.98.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Anomalocaris canadensis Turing wavenumber k_c at 24 µm (508 Ma)
- **spec_fingerprint** (semantic model + inputs): `6e60ae9de13486d021af674ae24c3fe415cc4c359fddd14174f41704efc81a34`
- `datasets/anomalocaris_canadensis_turing_wavelength_s0/anomalocaris_canadensis_turing_wavelength_s0_metrics.json` — artifact_sha256 `3aec9942cf43eb06…`, 977 B, 
- `docs/media/anomalocaris_canadensis_turing_wavelength_s0_comparison.png` — artifact_sha256 `8c788c0b684071e1…`, 661,168 B, 1344×1008 px
- `docs/media/anomalocaris_canadensis_turing_wavelength_s0_growth.gif` — artifact_sha256 `fdc6e92b6f8b2b81…`, 3,313,375 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion_growth_rate_pearson_r | 0.999989 Pearson r | 1.0 | 1D Schnakenberg cosine projected for 120 steps versus stencil growth rate σ(q), 14 wavenumbers *(Reference 1 is perfect numerical-analytic agreement. The measured r is lower.)* |
| dispersion_growth_rate_rmse | 0.006263 1/time | 0.0 | RMSE of numerical minus stencil growth rates on q in [0.25, 1.25] *(Reference 0 would be a perfect residual. RMSE is the KS companion scale.)* |
| dispersion_growth_rate_ks_statistic | 0.071429 KS D | 0.714286 | Two-sample KS of numerical versus stencil growth rates on the same 14 wavenumbers *(The reference D is the no-growth null. The fit D is the stencil comparison and is smaller.)* |
| specimen_image_foreground_fraction | 0.5885 fraction | 0.45 | border-median contrast mask on the Tyrrell cast, 160x120 *(0.45 is the mask prior used only as a non-equal comparator, not a published area.)* |
| specimen_contour_circularity | 0.3026 4πA/P² | 1.0 | erosion perimeter of the border-contrast mask *(The slab-plus-animal silhouette is not a circle.)* |
| specimen_stripe_coherency | 0.5589 (λ1-λ2)/(λ1+λ2) | 0.0 | mean structure-tensor coherency inside the specimen mask *(Shale cracks and flap margins are oriented, so coherency is above the isotropic null.)* |
| specimen_subfacet_to_facet_spacing_ratio | 0.0901 ratio | 1.0 | slot λ 23.97 µm divided by reconstructed 2.5 cm hemisphere spacing at 16000 lenses *(A ratio of 1 would mean this slot's wavelength was the facet spacing. It is not.)* |
| equal_diffusion_ablation_std_ratio | 0.0026 ratio | 1.0 | std(u | Du=Dv) / std(u | Dv/Du=20) after 700 steps on a 64x48 Neumann patch *(Reference 1 would mean the ablation still formed the Turing pattern.)* |
| du_sweep_lambda_monotonic | 1.0 boolean | 0.0 | eight diffusivity scales from 0.55 to 1.80; analytical λ_c must increase *(0 would record a non-monotonic sweep.)* |
| specimen_vs_modulated_luminance_ks | 0.1044 KS D | 0.0233 | KS of cast luminance versus the 0.16-alpha activator modulation *(The reference is the scrambled image, not a copy of the modulated KS.)* |

## Environment at checkpoint


Slot 3113 Pass 1. Evidence paths are in this commit.
