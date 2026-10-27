# MorphoEval report — audit `2026-10-27-drosophila-subobscura-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `a89be184699d71a03a1673e1f49d8998944a483a` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-subobscura-turing-wavelength-s0-audit -> 2026-10-27-drosophila-subobscura-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila subobscura Turing wavelength at 3.0 µm (Pass 2) (`models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py`) | 90 | 82 | 88 | 90 | 92 | 88 | 90 | 89.3 | 85.0 | 90.1 | 87.8 | **87.8** | faithful |

## Findings and remediation

### Drosophila subobscura Turing wavelength at 3.0 µm (Pass 2)  —  overall 87.8 (faithful)

Pass 2 keeps the Schnakenberg field at λ_c = 2.97 µm (3.0 µm detail scale, Holocene, 0 Ma) and closes the Pass 1 gaps. Du at fixed Dv/Du lengthens λ_c monotonically and in proportion to sqrt(Du) (Pearson r = 1 on five values). Setting Dv = Du makes the maximum growth rate -0.233 and the seeded field decays to std 7.7e-9, against a patterned std of 0.100. The 720×480, 12-frame GIF ends on the Cooper 2015 collage with the activator as a 14% pigment modulation inside a border-contrast mask. PDE growth versus the discrete eigenvalue has Pearson r = 0.999998, RMSE = 0.000495, and KS statistic 0.083 (12 modes). Otsu foreground fraction is 0.465 against a border-contrast fraction of 0.771. Wavenumber ratios are 27.80 versus Drosophila simulans and 33.15 versus Drosophila suzukii. The wing-size cline remains a natural-history citation, not a measured 3 µm lattice.

- **Model fidelity to source: 90** — Schnakenberg steady state, Jacobian, dispersion determinant, and k_c are implemented as published and checked by --self-test (trace < 0, det > 0, discriminant > 0, σ(k_c) = 0.0879). The ablation shows the band requires Dv > Du.  
  evidence: `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L71–99; `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L290–299
- **Morphological realism / specimen match: 82** — The comparison figure places the Cooper collage beside the spot field and beside the masked pigment composite. The GIF's last frame is that composite, so the silhouette matches the photograph by construction of the mask. The reaction-diffusion panel is a spot lattice at the pinned wavelength, not a redrawing of the wing veins. Score reflects that distinction.  
  evidence: `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L415–422; `docs/media/drosophila_subobscura_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_subobscura_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 88** — Five Du values at fixed Dv/Du give λ_c = 1.451, 1.777, 2.077, 2.406, 2.809. The sequence is strictly increasing and correlates with sqrt(Du) at r = 1. σ(k_c) stays 0.0879, so the sweep changes wavelength rather than merely rescaling time.  
  evidence: `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L302–323
- **Morphometric validation: 90** — PDE versus discrete growth: r = 0.999998, RMSE = 0.000495, KS = 0.083, n = 12. Ablation σ_max = -0.233 versus σ(k_c) = 0.088. Otsu foreground 0.465 versus border-contrast 0.771. Coherency at sigma 1.6 is 0.662 versus the fitness script's sigma-2.5 value 0.682. FFT wavelength 2.97 µm versus the 3.0 µm tier label.  
  evidence: `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L553–648; `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L343–376
- **Source and data traceability: 92** — Six DOIs, the Wikimedia file URL, Martin Cooper as creator, CC BY 2.0, and SHA-256 5366640b506b531749072d3e4097e68630b8132fd433df786b50e2e16c89711d. Scout recorded that slot 3206 does not collide with the 124 µm sibling.  
  evidence: `datasets/drosophila_subobscura_turing_wavelength_s0/paper_citations.bib` L1–54; `datasets/drosophila_subobscura_turing_wavelength_s0/README.md` L12–14
- **Code-to-model traceability: 88** — Each scored step points at a function: critical_mode, sample_dispersion, parameter_sweep, ablation_control, segment_specimen, render_gif. Lean theorems pin the integer wavenumber inequalities the chronospecies block evaluates.  
  evidence: `morphoeval/lean/loops/Loop_3206_drosophila_subobscura_turing_wavelength_s0.lean` L27–39; `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L131–210
- **Reproducibility package: 90** — python3 models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py --self-test exits 0 on the Pass 2 assertions, including r > 0.98, ablation, monotonic sweep, and unequal specimen fractions. Seed 3206. NumPy, SciPy, Pillow, Matplotlib.  
  evidence: `datasets/drosophila_subobscura_turing_wavelength_s0/README.md` L16–22; `models/2026-delphi10000-drosophila-subobscura-turing-wavelength-s0.py` L651–687

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila subobscura Turing wavelength at 3.0 µm (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `ddf700f45c258daa80240c53b23e60ad30c6cfe8bec1c0e8b9940daf13a758b9`
- **input_dataset_sha256**: `datasets/drosophila_subobscura_turing_wavelength_s0/drosophila_subobscura_specimen.jpg` (`5366640b506b5317…`), `datasets/drosophila_subobscura_turing_wavelength_s0/drosophila_subobscura_turing_wavelength_s0_reference.json` (`0b9b69eda941b675…`)
- `docs/media/drosophila_subobscura_turing_wavelength_s0_comparison.png` — artifact_sha256 `243f026cc7f99290…`, 876,428 B, 1344×1008 px
- `docs/media/drosophila_subobscura_turing_wavelength_s0_growth.gif` — artifact_sha256 `0040e9b46892f41d…`, 2,516,450 B, 720×480 px, 12 frames

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


Pass 2 closes R1–R4 on the default path: Du sweep, equal-diffusion ablation, specimen-mask composite, and non-circular specimen metrics.
