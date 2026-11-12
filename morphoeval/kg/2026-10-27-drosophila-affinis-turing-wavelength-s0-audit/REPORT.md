# MorphoEval report — audit `2026-10-27-drosophila-affinis-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila affinis Turing wavenumber at 2.4 µm (Holocene, detail tier) (`models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py`) | 84 | 78 | 84 | 76 | 90 | 82 | 86 | 84.0 | 77.2 | 86.3 | 82.0 | **82.0** | faithful |

## Findings and remediation

### Drosophila affinis Turing wavenumber at 2.4 µm (Holocene, detail tier)  —  overall 82.0 (faithful)

Pass 1 integrated Schnakenberg kinetics with Du chosen so the determinant-minimizing wavenumber is 2.444 µm at Dv/Du = 10. The fastest linear mode was 2.652 µm and the nonlinear FFT selected 2.619 µm (metrics-pass1.json). Pearson r between continuum and discrete growth rates was 0.9997 and the equal-diffusivity ablation stayed negative (sigma_max = -0.099). The habitus frame matches the CC BY-SA photograph; the 2.444 µm field is not resolved by that photograph's 250 µm scale bar. Weighted overall on the Pass-1 gap is 82.0. Calibration of Du onto the fastest mode is the Pass-2 fix.

- **Model fidelity to source: 84** — The Jacobian, classical k_c formula, periodic Schnakenberg update, and equal-diffusivity ablation implement the published law. Pass 1 left Du at the determinant condition, so the mode that actually grows (2.652 µm) is not the assigned 2.444 µm.  
  evidence: `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L90–139; `morphoeval/runs/2026-10-27-drosophila-affinis-turing-wavelength-s0-loop-metrics-pass1.json` L6–21
- **Morphological realism / specimen match: 78** — The comparison figure places the CC BY-SA female habitus beside the activator field, and the GIF resolves to that habitus. The generative spots are a melanin-ramped field, not a segmentation of the abdominal stripes, and the photograph cannot show 2.444 µm structure.  
  evidence: `docs/media/drosophila_affinis_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_affinis_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 84** — Above onset, peak growth rate increases with Dv/Du while the classical wavelength is held at 2.444 µm. Setting Dv = Du removes the instability. Pass 1 did not yet use Du as a control that places the fastest mode on the assigned scale.  
  evidence: `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L198–226
- **Morphometric validation: 76** — Continuum versus discrete sigma(k) Pearson r = 0.9997 with RMSE 0.0089, which clears 0.98. The nonlinear wavelength error against 2.444 µm is 7.2 percent (2.619 vs 2.444). That is the quantity Pass 2 has to close. Specimen segmentation is computed but was not yet the basis of the wavelength claim.  
  evidence: `morphoeval/runs/2026-10-27-drosophila-affinis-turing-wavelength-s0-loop-metrics-pass1.json` L6–18; `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L182–196
- **Source and data traceability: 90** — Crossref-confirmed DOIs for Turing, Schnakenberg, Gierer-Meinhardt, Sturtevant, Miller, Wittkopp, and Shoup. The habitus is Wikimedia Commons File:Daff-fem.tif, CC BY-SA 4.0, SHA-256 pinned.  
  evidence: `datasets/drosophila_affinis_turing_wavelength_s0/paper_citations.bib` L1–108; `datasets/drosophila_affinis_turing_wavelength_s0/drosophila_affinis_turing_wavelength_s0_reference.json` L1–30
- **Code-to-model traceability: 82** — The module docstring and the kinetics, dispersion, and ablation functions name the equations they implement. Lean theorems encode the integer form of stability, instability, and the ablation. Pass-1 Du was not yet traced to the fastest-mode condition.  
  evidence: `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L1–40; `morphoeval/lean/loops/Loop_3174_drosophila_affinis_turing_wavelength_s0.lean` L1–62
- **Reproducibility package: 86** — Seed 3174, --self-test, and SHA-256 pins on the reference, specimen, scout manifest, bibliography, and README. Nanopub signing is unavailable because ~/.nanopub/profile.yml is absent. One builder scored the audit.  
  evidence: `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L55–88; `models/2026-delphi10000-drosophila-affinis-turing-wavelength-s0.py` L763–795

Remediation:
- R1: Retune Du so the fastest-growing mode, which the nonlinear FFT follows, sits on 2.444 µm. Pass 1 left it at 2.652 µm (FFT 2.619 µm).
- R2: Redraw the chronoseries panel so the 2.444 µm bar is readable on a log axis next to 98.43 µm and 138.7 µm.
- R3: Report specimen-mask foreground, circularity, coherency, and the orientation-histogram KS against the habitus, with reference values that are not copies of the measurements.
- R4: Nanopub signing was not performed. ~/.nanopub/profile.yml is not on this machine.
- R5: One habitus photograph. The 250 µm scale bar does not resolve 2.444 µm, and there is no second assessor.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila affinis Turing wavenumber at 2.4 µm (Holocene, detail tier)
- **spec_fingerprint** (semantic model + inputs): `1d549b7322f4ffdb1233c8d9a23519c69dfd24ec8235a0131e1cb9b06ccb909d`
- **input_dataset_sha256**: `datasets/drosophila_affinis_turing_wavelength_s0/drosophila_affinis_specimen.jpg` (`83062d706a605b86…`), `datasets/drosophila_affinis_turing_wavelength_s0/drosophila_affinis_turing_wavelength_s0_reference.json` (`9939928504f37754…`), `datasets/drosophila_affinis_turing_wavelength_s0/paper_citations.bib` (`368c8326ecee5c1b…`), `datasets/drosophila_affinis_turing_wavelength_s0/scout_manifest.json` (`2164e988b14e5446…`)
- `docs/media/drosophila_affinis_turing_wavelength_s0_comparison.png` — artifact_sha256 `50fd3a8949bcb8a6…`, 280,979 B, 1380×984 px
- `docs/media/drosophila_affinis_turing_wavelength_s0_growth.gif` — artifact_sha256 `7605e3d489c1c227…`, 2,087,759 B, 960×720 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| classical_turing_wavelength | 2.444 µm | 2.4 | k_c^2 = (Dv f_u + Du g_v) / (2 Du Dv); lambda = 2 pi / k_c *(value is the solved classical wavelength, reference is the rounded tier label)* |
| fastest_mode_wavelength | 2.442842 µm | 2.251481424903267 | argmax_k σ(k) after Du is calibrated so this peak sits on the assigned 2.444 µm scale *(the two wavenumbers differ above onset; the pattern follows the fastest mode)* |
| nonlinear_stripe_wavelength | 2.431871 µm | 2.444 | dominant Fourier bin of the periodic Schnakenberg activator after the seeded run |
| dispersion_sigma_pearson_r | 0.999142 1 | 0.98 | Pearson r between continuum σ(k) and discrete-Laplacian Euler growth, 12 wavenumbers *(RMSE of the same residuals is reported separately)* |
| dispersion_sigma_rmse | 0.011665 1/time | 0.0 | RMSE of numerical minus analytical σ(k) |
| ablation_max_growth_rate | -0.098997 1/time | 0.0 | max_k σ(k) with Dv = Du (equal diffusivity negative control) *(control versus the patterned ratio-10 run)* |
| patterned_max_growth_rate | 0.10944 1/time | -0.09899657029629944 | max_k σ(k) at Dv/Du = 10 and lambda_c = 2.444 µm |
| specimen_foreground_fraction | 0.221338 1 | — | border-median contrast mask on the CC BY-SA habitus |
| specimen_contour_circularity | 0.10793 1 | 1.0 | 4 pi Area / Perimeter^2 on the segmented habitus |
| specimen_structure_tensor_coherency | 0.495974 1 | 0.5 | mean structure-tensor coherency inside the habitus mask |
| wavelength_ratio_vs_berryi | 0.02483 1 | 1.0 | 2.444 µm / 98.43 µm |
| wavelength_ratio_vs_americana | 0.017621 1 | 1.0 | 2.444 µm / 138.7 µm |
| rendered_habitus_foreground_fraction | 0.299285 1 | 0.221338 | same segmentation applied to the last GIF frame (habitus plus activator strip) *(the bottom activator strip changes the mask, so this is not equal to the specimen value)* |
| orientation_histogram_ks | 0.002469 1 | 0.05 | KS distance between 12-bin Sobel orientation histograms of specimen and last GIF frame |

## Environment at checkpoint


Pass 1 scored on the working tree of loop 3174 before the loop commit. Parent commit 0c76d15d does not yet contain these paths. publisher_orcid is the repository URI because no personal ORCID profile is configured; the nanopubs are unsigned.
