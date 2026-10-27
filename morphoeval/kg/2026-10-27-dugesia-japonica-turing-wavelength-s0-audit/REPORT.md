# MorphoEval report — audit `2026-10-27-dugesia-japonica-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `e4ebf5f02021c5cb9f6779addaebe15739877255` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Saturated Schnakenberg Turing wavelength for Dugesia japonica dorsal pigment at 6.912 µm (Holocene, Pass 1) (`models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py`) | 86 | 82 | 72 | 80 | 90 | 84 | 88 | 81.3 | 81.2 | 87.6 | 83.2 | **83.2** | faithful |

## Findings and remediation

### Saturated Schnakenberg Turing wavelength for Dugesia japonica dorsal pigment at 6.912 µm (Holocene, Pass 1)  —  overall 83.2 (faithful)

Pass 1 selects λ_c = 6.912 µm by solving D_u at diffusion ratio 48 for the saturated Schnakenberg Jacobian (κ = 0.15, a = 0.05, b = 1.40). Linear-mode growth matches analytic σ(k) at Pearson r = 0.99998 (RMSE 0.0054). Equal diffusion is stable (max σ = -0.565). The 16-frame, 640 px GIF paints that field through the segmented Nogi et al. 2009 plate. Not yet shown: the diffusion-ratio sweep, a KS comparison, the nonlinear peak wavelength, and the iNaturalist wild-type voucher.

- **Model fidelity to source: 86** — The script integrates the saturated Schnakenberg system and the Turing wavenumber formula, and D_u is solved so λ_c is 6.912 µm. Saturation is a numerical bound, not a measured pigment Michaelis constant. The wavelength is the assigned scale, not a fitted biological constant.  
  evidence: `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L122–156; `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L198–243
- **Morphological realism / specimen match: 82** — The GIF and the comparison panel place the pigment field inside segmented bodies from the Nogi plate, with a bright rim and dark cups. The mask merges animals that are separate on the plate, and the field is a magnified motif, not a resolution-matched micrograph. Still clearly a planarian pigment plate rather than an abstract plot.  
  evidence: `docs/media/dugesia_japonica_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L365–456
- **Parameter expressivity: 72** — Self-test checks that λ(R) falls from R = 36 to R = 120 and that R = 1 has a negative gap. The published figure does not show that sweep or the ablation, so a reader of the figure alone cannot see the control.  
  evidence: `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L344–360; `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L495–537
- **Morphometric validation: 80** — Linear dispersion Pearson r = 0.99998 against analytic σ(k_disc), ablation max σ = -0.565, body-mask circularity 0.261 versus an ellipse reference 0.55, and 7 ocelli versus the wild-type pair of 2. Missing a KS histogram test and the nonlinear steady-state wavelength.  
  evidence: `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L294–340; `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L586–636
- **Source and data traceability: 90** — Crossref-checked DOIs, two CC plates with SHA-256, and an explicit statement that 6.912 µm is the slot scale. Wikimedia fetch succeeded; the synthetic gallery bullseye was not used as the fitness target.  
  evidence: `datasets/dugesia_japonica_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/dugesia_japonica_turing_wavelength_s0/dugesia_japonica_turing_wavelength_s0_reference.json` L1–30
- **Code-to-model traceability: 84** — Docstring states the PDE, the wavenumber formula, the chronoseries, and the display caveat. Lean theorems are the integer shadow of the gap, the trace, seven wavelengths, the ablation, and the empty predecessor count. Lean was not executed here.  
  evidence: `morphoeval/lean/loops/Loop_3213_dugesia_japonica_turing_wavelength_s0.lean` L14–39; `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L1–70
- **Reproducibility package: 88** — --self-test checks the Turing inequalities, r > 0.98, ablation, monotonic sweep, and a non-flat patch. Seed 3213, grid 84, and the invocation are in the dataset README. Dependencies are the repo scientific stack. No second assessor.  
  evidence: `models/2026-delphi10000-dugesia-japonica-turing-wavelength-s0.py` L638–684; `datasets/dugesia_japonica_turing_wavelength_s0/README.md` L1–30

Remediation:
- {'id': 'R1', 'item': 'Report the nonlinear steady-state dominant wavelength next to the linear λ_c = 6.912 µm. The saturated field shifts off the linear mode.'}
- {'id': 'R2', 'item': 'Add a two-sample KS (or equivalent) comparison of the specimen interior histogram and the simulated interior, with the statistic not equal to its reference.'}
- {'id': 'R3', 'item': 'Draw the diffusion-ratio sweep and the equal-diffusion ablation on the comparison figure. They are computed and not shown.'}
- {'id': 'R4', 'item': 'Segment the CC BY 4.0 iNaturalist wild-type voucher and record a body metric that is not copied from the Nogi plate.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Saturated Schnakenberg Turing wavelength for Dugesia japonica dorsal pigment at 6.912 µm (Holocene, Pass 1)
- **spec_fingerprint** (semantic model + inputs): `69b07407b41111e3df409d09d284c8ed47d97c6e78075bf3f4f35e0a79d838da`
- **input_dataset_sha256**: `datasets/dugesia_japonica_turing_wavelength_s0/dugesia_japonica_field_kalininsky_ccby4.jpg` (`e40acb76c9286a8d…`), `datasets/dugesia_japonica_turing_wavelength_s0/dugesia_japonica_ocellus_pigment_nogi2009.png` (`6abf2ae940aa3115…`), `datasets/dugesia_japonica_turing_wavelength_s0/dugesia_japonica_turing_wavelength_s0_reference.json` (`2bc1220e0880d8c0…`)
- `docs/media/dugesia_japonica_turing_wavelength_s0_comparison.png` — artifact_sha256 `a9ac178d5b9463ac…`, 552,219 B, 1248×1104 px
- `docs/media/dugesia_japonica_turing_wavelength_s0_growth.gif` — artifact_sha256 `810e44644592871f…`, 1,099,037 B, 640×640 px, 16 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing critical wavelength lambda_c of Dugesia japonica pigment field | 6.912 µm | 6.9 | k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) with D_u solved at diffusion ratio 48, κ = 0.15 *(log10(length/m) = -5.160; age 0 Ma; chronoseries 1/1)* |
| linear dispersion growth-rate Pearson r (numeric modes vs analytic sigma) | 0.99998 Pearson_r | 0.98 | 1D periodic Schnakenberg, modes 2–12, log-amplitude slope versus σ(k_disc) *(RMSE = 0.00539; p = 4.694e-19; σ(k_c) = 0.0566)* |
| equal-diffusion ablation max growth rate versus operating Turing gap | -0.56518 1/time | 0.05657 | Analytic dispersion over k in [0, 3] /µm with D_u = D_v *(ablation gap numerator = -1.1304 (negative control); operating gap = 20.8679)* |
| specimen image body-mask circularity of Dugesia japonica Nogi 2009 plate | 0.2609 4πA/P² | 0.55 | Otsu-style brightness mask on the CC-BY Nogi plate, filled, circularity 4πA/P² *(foreground fraction 0.319; bodies 2; ocelli 7)* |
| specimen image ocellus pigment-cup count on the Nogi plate | 7 count | 2 | Dark connected components (area 8–800 px) inside the eroded body mask *(value differs from the wild-type pair on purpose: the plate shows regenerated heads)* |

## Environment at checkpoint

- python: 3.12
- lean: not installed in this VM; theorems are Lean 4 prelude, unchecked here

Pass 1. Saturated Schnakenberg field on the Nogi 2009 plate. Sweep and ablation are computed in the self-test but not drawn. Nonlinear wavelength, KS, and the iNat voucher are not yet in the figure.
