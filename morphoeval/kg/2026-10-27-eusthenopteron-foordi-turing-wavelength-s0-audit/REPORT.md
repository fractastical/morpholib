# MorphoEval report — audit `2026-10-27-eusthenopteron-foordi-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Turing k_c at 104 µm on Eusthenopteron foordi (Frasnian, 379 Ma) (`models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py`) | 86 | 76 | 74 | 80 | 90 | 82 | 84 | 82.0 | 77.5 | 85.5 | 81.2 | **81.2** | faithful |

## Findings and remediation

### Turing k_c at 104 µm on Eusthenopteron foordi (Frasnian, 379 Ma)  —  overall 81.2 (faithful)

Schnakenberg kinetics with the Turing wavenumber set so λ_c = 104.2 µm. Pass 1 confirms the integrator runs, the Brussels mount is the display domain, and the realized wavelength is 102.0 µm. Sweep monotonicity, ablation stds, mask morphometrics, and the dispersion residual are listed as remediation rather than fully credited.

- **Model fidelity to source: 86** — The Fourier step is the Schnakenberg reaction plus diffusion, and k_c is the standard onset formula. Pass 1 does not yet treat the dispersion residual as a completed fidelity test.  
  evidence: `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L12–20; `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L101–108
- **Morphological realism / specimen match: 76** — The comparison figure places the real Brussels mount beside the segmented body. Pass 1 does not yet credit the ocellus tint as a specimen match, because fossil color is not preserved.  
  evidence: `docs/media/eusthenopteron_foordi_turing_wavelength_s0_comparison.png` L0–0; `docs/media/eusthenopteron_foordi_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 74** — Dv/Du is the morphospace parameter, but Pass 1 does not yet accept the sweep: the ratio-10 run is barely patterned (std 0.08) and λ is not strictly monotone.  
  evidence: `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L271–294
- **Morphometric validation: 80** — A wavelength is reported (102.0 µm versus formula 104.2 µm). Pass 1 still requires the Pearson/RMSE dispersion table, the KS orientation test, and the mask aspect to be the scored comparison.  
  evidence: `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L161–176
- **Source and data traceability: 90** — Turing, Schnakenberg, Kondo & Asai, and Andrews & Westoll are cited by DOI. The specimen file is SHA-256 pinned and CC BY-SA 3.0.  
  evidence: `datasets/eusthenopteron_foordi_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/eusthenopteron_foordi_turing_wavelength_s0/eusthenopteron_foordi_turing_wavelength_s0_reference.json` L55–66
- **Code-to-model traceability: 82** — The module docstring states the PDE and the wavenumber formula that schnakenberg_step implements. Pass 1 does not yet trace the dispersion harness line by line.  
  evidence: `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L62–78
- **Reproducibility package: 84** — Seed 3225, --self-test, and a pinned specimen hash. Pass 1 does not yet treat the control and ablation paths as part of the reproducibility gate.  
  evidence: `models/2026-delphi10000-eusthenopteron-foordi-turing-wavelength-s0.py` L631–658

Remediation:
- Parameter-regime sweep: re-score Dv/Du against realized wavelength, including the near-threshold ratio and the non-monotone step.
- Negative control and ablation: report equal-diffusion std and autocatalysis-ablated std beside the patterned std.
- Real specimen CV segmentation: score the body mask (aspect, circularity) against the gallery reconstruction and record the orientation KS.
- Goodness-of-fit: Pearson r, RMSE on σ(k), and the KS/CV specimen comparison must be the morphometric evidence, with value ≠ reference.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Turing k_c at 104 µm on Eusthenopteron foordi (Frasnian, 379 Ma)
- **spec_fingerprint** (semantic model + inputs): `7d5c5c4a83fa8f6c063e1e197796aa4e97931ea29ced33c300363ee59ac62e99`
- `docs/media/eusthenopteron_foordi_turing_wavelength_s0_comparison.png` — artifact_sha256 `721d8d4672825d00…`, 631,334 B, 1584×888 px
- `docs/media/eusthenopteron_foordi_turing_wavelength_s0_growth.gif` — artifact_sha256 `960a6f94384c58dc…`, 2,695,211 B, 974×480 px, 10 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen_dispersion_pearson_r | 0.9999997928438427 1 | 1.0 | Pearson r between analytical Turing σ(k) and a linearized Fourier step on the real-eigenvalue band |
| specimen_dispersion_rmse | 0.0026052316732888803 1/time | 0.0 | RMSE of numerical versus analytical growth rates σ(k) |
| ocellus_wavelength_um | 102.00978980708321 µm | 104.2 | Radial FFT peak of the saturated Schnakenberg field, converted by the 104.2 µm formula-wavelength unit |
| specimen_contour_aspect_ratio | 3.6198347107438016 1 | 3.9055555555555554 | Bounding-box width/height of the dark fossil body segmented from the Brussels mount |
| specimen_mask_circularity | 0.18330998977798435 1 | 0.2117580600263819 | 4πA/P² of the segmented fossil body |
| specimen_orientation_ks | 0.02400000000000002 1 | 0.0 | Two-sample KS statistic of Sobel orientations, fossil crop versus last GIF frame |
| pattern_versus_equal_diffusion_std_ratio | 286241.3553406803 1 | 1.0 | std(u) of the Turing run divided by std(u) of the equal-diffusion negative control |
| pattern_versus_ablated_autocatalysis_std_ratio | 483305978204.98065 1 | 1.0 | std(u) of the full model divided by std(u) after dropping u²v autocatalysis |
| wavelength_sweep_spearman_r | 0.8999999999999998 1 | 0.0 | Spearman rank correlation of realized λ against Dv/Du over the sweep |

## Environment at checkpoint


Pass 1 working tree. New slot files are not on this commit; signing profile is absent.
