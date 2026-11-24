# MorphoEval report — audit `2026-10-27-bombyx-mori-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bombyx mori Turing wavenumber selection at 41 µm (Holocene, 0 Ma) (`models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py`) | 86 | 82 | 80 | 72 | 90 | 84 | 86 | 84.0 | 78.2 | 86.8 | 82.5 | **82.5** | faithful |

## Findings and remediation

### Bombyx mori Turing wavenumber selection at 41 µm (Holocene, 0 Ma)  —  overall 82.5 (faithful)

Pass 1. Schnakenberg linear dispersion is calibrated so argmax σ(k) = 41.016 µm at age 0 Ma (Du = 10.659 µm², Dv = 191.861 µm², ratio 18). Analytic versus numeric growth-rate Pearson r = 0.9923 (RMSE 0.051). The axisymmetric Galerkin fit to the ocellus card has Pearson r = 0.9997. The scale helper returns 100/width (µm per px, 0.9009) but callers treat it as px per µm, so the 41 µm carrier carries only 0.14% of the wave amplitude and the dark-ring pitch is reported as 82.14 µm instead of about 67 µm. σ(96 µm) = -0.490 (mandarina contrast damped) and the Du = Dv ablation growth rate is -1.315.

- **Model fidelity to source: 86** — Implements the Schnakenberg Jacobian, σ(k), and a calibration that puts the maximum-growth wavelength at 41.016 µm. The rendered carrier for that wavelength is then placed with an inverted scale factor, so the picture is not yet the 41 µm mode.  
  evidence: `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` L74–118; `morphoeval/lean/loops/Loop_3134_bombyx_mori_turing_wavelength_s0.lean` L21–49
- **Morphological realism / specimen match: 82** — The growth GIF is an axisymmetric rendering of the ocellus card with synthetic grain, 12 frames at 640×480. The side-by-side figure shows the card. The 41 µm ring is not yet on the card's pixel scale.  
  evidence: `docs/media/bombyx_mori_turing_wavelength_s0_growth.gif` L0–0; `docs/media/bombyx_mori_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 80** — dispersion_sweep varies wavelength from 28 to 88 µm and Dv by factors 0.55 to 2.05. λ(Dv) is monotonic. The ablation is a growth-rate number plus a flat gray panel, not a field evolved at Du = Dv.  
  evidence: `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` L147–183
- **Morphometric validation: 72** — Dispersion Pearson r = 0.9923 and radial Pearson r = 0.9997, with RMSE 0.669 gray levels and KS D = 0.0265. The reported dark-ring pitch is 82.14 µm because measure_scale_bar returns µm/px (0.9009) and segment_card divides by it. The 41 µm mode power fraction is 0.0014. KS p-value is computed and dropped.  
  evidence: `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` L191–208; `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` L489–574
- **Source and data traceability: 90** — Crossref-checked DOIs for Yamaguchi 2013, Yoda 2014, Turing 1952, and Gierer & Meinhardt 1972. Ocellus card SHA-256 2adf5122… and CC0 adult SHA-256 76144516…. Age 0 Ma and scale 41 µm are stated in the reference JSON.  
  evidence: `datasets/bombyx_mori_turing_wavelength_s0/paper_citations.bib` L1–39; `datasets/bombyx_mori_turing_wavelength_s0/bombyx_mori_turing_wavelength_s0_reference.json` L1–90
- **Code-to-model traceability: 84** — Module docstring maps Schnakenberg, k_c, the 96 µm predecessor, and the Galerkin card fit to Turing, Gierer-Meinhardt, and Yamaguchi. Lean theorems are integer witnesses of the same identities, not a real-analysis formalization of σ(k).  
  evidence: `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` L1–42
- **Reproducibility package: 86** — Seed 3134, --self-test, pinned specimen digest, deterministic dispersion correlation. Passing the self-test does not catch the inverted scale factor.  
  evidence: `models/2026-delphi10000-bombyx-mori-turing-wavelength-s0.py` L577–614

Remediation:
- {'id': 'R1', 'item': 'measure_scale_bar returns micrometres per pixel (100/width) while fit_galerkin and segment_card treat the value as pixels per micrometre. Return width/100, re-place the 41 µm carrier, and recompute the dark-ring pitch.'}
- {'id': 'R2', 'item': 'Replace the constant-gray ablation panel with a field evolved at Du = Dv from the same seeded noise, and keep the numeric growth rate σ < 0.'}
- {'id': 'R3', 'item': 'Emit the Kolmogorov-Smirnov p-value as its own measurement (value not equal to the reference) alongside D, Pearson r, and RMSE.'}
- {'id': 'R4', 'item': 'Re-render the GIF after the scale correction and re-score computer-vision similarity against the ocellus card.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bombyx mori Turing wavenumber selection at 41 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `900e8c4cd551803050526b60486b8aa4c0338d7c662a7b9812395080ca4e0324`
- **input_dataset_sha256**: `datasets/bombyx_mori_turing_wavelength_s0/bombyx_mori_adult_cc0.jpg` (`76144516fb810472…`), `datasets/bombyx_mori_turing_wavelength_s0/bombyx_mori_ocellus_specimen.jpg` (`2adf51228d2da684…`), `datasets/bombyx_mori_turing_wavelength_s0/bombyx_mori_turing_wavelength_s0_reference.json` (`0a11d56294fb484c…`), `datasets/bombyx_mori_turing_wavelength_s0/paper_citations.bib` (`08ab9cb9580075e6…`), `datasets/bombyx_mori_turing_wavelength_s0/scout_manifest.json` (`9e4a322221a85be7…`)
- `docs/media/bombyx_mori_turing_wavelength_s0_comparison.png` — artifact_sha256 `462d0dff8415d976…`, 560,478 B, 1632×1008 px
- `docs/media/bombyx_mori_turing_wavelength_s0_growth.gif` — artifact_sha256 `d768ef9b85c6ae76…`, 3,132,697 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion analytic vs numeric growth-rate Pearson r | 0.9923 Pearson r | 1.0 | Pearson correlation of σ(λ) on 12 wavelengths from 28 to 88 µm |
| ocellus-card radial profile Galerkin Pearson r | 0.9997 Pearson r | 1.0 | Degree-4 envelope plus 8 Fourier rings, including the 41 µm carrier, versus the card radial mean |
| ocellus-card radial profile RMSE | 0.6692 gray levels | 0.0 | RMSE of the Galerkin fit on radii 0–340 px |
| specimen radial-profile KS statistic | 0.0265 KS D | 0.05 | Two-sample Kolmogorov-Smirnov between card radial mean and Galerkin prediction |
| specimen ocellus dominant dark-ring pitch | 82.14 µm | 41.0 | Median spacing of radial-mean minima, converted with the 100 µm scale bar |
| specimen ocellus structure-tensor coherency | 0.2386 coherency | 0.85 | Mean structure-tensor coherency on the contrast foreground of the ocellus card |
| specimen ocellus foreground fraction | 0.4812 fraction | 0.5 | Border-contrast mask, morphological closing, hole fill |
| selected wavelength lambda_c | 41.0161 µm | 41.0 | argmax_k σ(k) after joint Du, Dv calibration |
| ablation growth rate at 41 µm when Du = Dv | -1.3152 1/time | 0.0 | Numeric growth of a 41 µm cosine with Dv reset to Du |
| ancestral 96 µm mode growth rate | -0.4899 1/time | 0.0 | Analytic σ(2π/96 µm) at the domestic diffusivity pair |

## Environment at checkpoint


Pass 1 working tree. New slot-3134 files are not in this commit yet.
