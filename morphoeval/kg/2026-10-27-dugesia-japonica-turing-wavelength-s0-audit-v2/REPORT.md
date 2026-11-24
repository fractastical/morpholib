# MorphoEval report — audit `2026-10-27-dugesia-japonica-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `80fe3e3968850cd5826acf65ba6a24cab2e95424` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-dugesia-japonica-turing-wavelength-s0-audit -> 2026-10-27-dugesia-japonica-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Saturated Schnakenberg Turing wavelength for Dugesia japonica dorsal pigment at 6.912 µm (Holocene, Pass 2) (`models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py`) | 88 | 86 | 90 | 90 | 92 | 86 | 90 | 88.7 | 87.5 | 89.6 | 88.5 | **88.5** | faithful |

## Findings and remediation

### Saturated Schnakenberg Turing wavelength for Dugesia japonica dorsal pigment at 6.912 µm (Holocene, Pass 2)  —  overall 88.5 (faithful)

Pass 2 keeps the same kinetics and the same 6.912 µm linear wavelength. The saturated steady field peaks at Fourier mode 6, λ = 8.064 µm, which is reported beside λ_c rather than tuned away. Diffusion ratio R from 36 to 120 shortens λ monotonically, and equal diffusion (R = 1) has max σ = -0.565. Interior gray KS distance against the Nogi plate is D = 0.434 (reference null 0.5). The Kalininsky wild-type voucher yields two bodies with mean circularity 0.307. Linear dispersion Pearson r remains 0.99998.

- **Model fidelity to source: 88** — Same saturated Schnakenberg system and wavenumber formula as Pass 1. Pass 2 states the nonlinear shift (8.064 µm versus 6.912 µm) instead of treating the linear wavelength as the steady-state peak. κ and R are still chosen for this slot, not taken from a Dugesia pigment paper.  
  evidence: `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L122–156; `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L458–470
- **Morphological realism / specimen match: 86** — The 640 px, 16-frame GIF still paints the field through the Nogi body mask (CV similarity 94.7). Pass 2 does not pretend the wild-type rocks-and-worms photo is the same object; it measures that voucher separately. Merged outlines on the regeneration plate remain visible.  
  evidence: `docs/media/dugesia_japonica_turing_wavelength_s0_growth.gif` L0–0; `docs/media/dugesia_japonica_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 90** — The figure plots λ(R) from 36 to 120, falling from 7.002 µm to 6.758 µm, and the equal-diffusion bar is negative while σ(k_c) at R = 48 is positive. Self-test asserts the monotonicity.  
  evidence: `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L582–602; `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L344–360
- **Morphometric validation: 90** — Pearson r = 0.99998 on the linear modes. Nonlinear peak 8.064 µm versus reference 6.912 µm. KS D = 0.434 versus reference 0.5. Nogi circularity 0.261 versus 0.55. Ocelli 7 versus 2. Wild-type circularity 0.307 versus 0.55. None of these values is copied onto its reference.  
  evidence: `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L494–508; `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L729–756
- **Source and data traceability: 92** — Pass 1 provenance stands. The wild-type voucher is now actually measured, and both SHA-256 digests remain in the reference JSON.  
  evidence: `datasets/dugesia_japonica_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/dugesia_japonica_turing_wavelength_s0/README.md` L1–28
- **Code-to-model traceability: 86** — Docstring, Lean integer shadow, and the new measurement functions each point at the formula they implement. Lean was not executed in this VM.  
  evidence: `morphoeval/lean/loops/Loop_3213_dugesia_japonica_turing_wavelength_s0.lean` L14–39; `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L139–171
- **Reproducibility package: 90** — --self-test now also checks a non-flat patch, a nonlinear wavelength off 6.912 µm, and at least one wild-type body. Seed, grid, and commands are unchanged. Still one assessor, and no nanopub signature because this VM has no ~/.nanopub/profile.yml.  
  evidence: `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L759–808; `datasets/dugesia_japonica_turing_wavelength_s0/dugesia_japonica_turing_wavelength_s0_metrics.json` L1–20

Remediation:
- {'id': 'R5', 'item': 'Still one assessor. A second model has not scored this audit.'}
- {'id': 'R6', 'item': 'The Nogi brightness mask still merges some adjacent animals. A watershed split was not required for the image-fitness gate and was not added.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Saturated Schnakenberg Turing wavelength for Dugesia japonica dorsal pigment at 6.912 µm (Holocene, Pass 2)
- **spec_fingerprint** (semantic model + inputs): `69b07407b41111e3df409d09d284c8ed47d97c6e78075bf3f4f35e0a79d838da`
- **input_dataset_sha256**: `datasets/dugesia_japonica_turing_wavelength_s0/dugesia_japonica_field_kalininsky_ccby4.jpg` (`e40acb76c9286a8d…`), `datasets/dugesia_japonica_turing_wavelength_s0/dugesia_japonica_ocellus_pigment_nogi2009.png` (`6abf2ae940aa3115…`), `datasets/dugesia_japonica_turing_wavelength_s0/dugesia_japonica_turing_wavelength_s0_reference.json` (`2bc1220e0880d8c0…`)
- `docs/media/dugesia_japonica_turing_wavelength_s0_comparison.png` — artifact_sha256 `e752de071e812756…`, 458,681 B, 1496×946 px
- `docs/media/dugesia_japonica_turing_wavelength_s0_growth.gif` — artifact_sha256 `810e44644592871f…`, 1,099,037 B, 640×640 px, 16 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing critical wavelength lambda_c of Dugesia japonica pigment field | 6.912 µm | 6.9 | k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) with D_u solved at diffusion ratio 48, κ = 0.15 *(log10(length/m) = -5.160; age 0 Ma; chronoseries 1/1)* |
| linear dispersion growth-rate Pearson r (numeric modes vs analytic sigma) | 0.99998 Pearson_r | 0.98 | 1D periodic Schnakenberg, modes 2–12, log-amplitude slope versus σ(k_disc) *(RMSE = 0.00539; p = 4.694e-19; σ(k_c) = 0.0566)* |
| equal-diffusion ablation max growth rate versus operating Turing gap | -0.56518 1/time | 0.05657 | Analytic dispersion over k in [0, 3] /µm with D_u = D_v *(ablation gap numerator = -1.1304 (negative control); operating gap = 20.8679)* |
| specimen image body-mask circularity of Dugesia japonica Nogi 2009 plate | 0.2609 4πA/P² | 0.55 | Otsu-style brightness mask on the CC-BY Nogi plate, filled, circularity 4πA/P² *(foreground fraction 0.319; bodies 2; ocelli 7)* |
| specimen image ocellus pigment-cup count on the Nogi plate | 7 count | 2 | Dark connected components (area 8–800 px) inside the eroded body mask *(value differs from the wild-type pair on purpose: the plate shows regenerated heads)* |
| nonlinear stripe wavelength of the saturated steady field | 8.064 µm | 6.912 | Radial FFT peak of the periodic activator patch after spectral integration (modes 2–15) *(dominant Fourier mode 6 on a 48.384 µm patch; saturation shifts the peak off mode 7)* |
| specimen image versus render interior gray KS distance | 0.4343 KS D | 0.5 | ks_2samp on 4000 gray samples inside the Nogi body mask versus the rendered field *(two-sided p = 0.000e+00)* |
| specimen image wild-type body circularity on the Kalininsky voucher | 0.3074 4πA/P² | 0.55 | Pale-tan components larger than 8000 px on the CC BY 4.0 iNaturalist frame; mean circularity *(bodies kept = 2; foreground fraction of the pale mask = 0.074)* |

## Environment at checkpoint

- python: 3.12
- lean: not installed in this VM; theorems are Lean 4 prelude, unchecked here

Pass 2. Sweep and ablation are on the figure. Nonlinear peak is 8.064 µm (mode 6) against linear λ_c = 6.912 µm. KS D = 0.434. iNaturalist mean body circularity = 0.307.
