# MorphoEval report — audit `2026-10-27-asteroxylon-mackiei-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `8f0e4833ae7c01aa84482096856ead2148c3f912` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Asteroxylon mackiei Schnakenberg critical wavelength at 5.2 µm (Pass 1) (`models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py`) | 88 | 80 | 86 | 82 | 90 | 84 | 86 | 87.3 | 80.8 | 86.8 | 84.5 | **84.5** | faithful |

## Findings and remediation

### Asteroxylon mackiei Schnakenberg critical wavelength at 5.2 µm (Pass 1)  —  overall 84.5 (faithful)

Pass 1 implements Schnakenberg kinetics with the critical wavelength fixed at 5.2 µm (k_c = 2π/5.2e-6 m⁻¹), δ = D_v/D_u = 160, and a periodic 156 µm × 117 µm tracheid-file window (30 critical wavelengths). The homogeneous state has trace -0.16 and determinant 1. σ(k_c) = 0.396 and the equal-diffusion control has σ_max = -0.080. A δ sweep at fixed D_u shortens the wavelength monotonically. The basic look composites the activator onto a CC BY-SA compression fossil (Kenraiz); full-frame grayscale Pearson r = 0.98950 and RMSE = 7.425 gray levels. The plate's spectral centroid of the activator is 7.646 µm, not 5.2 µm. Age is 407 Ma. Not yet scored at Pass 1: a row hold-out, a gray-level KS, and a measured axis-envelope wavelength. The measurement sidecar is regenerated from the final module, so it also lists those three Pass 2 numbers; this audit's morphometric score does not credit them.

- **Model fidelity to source: 88** — The steady state, Jacobian, quadratic growth rate, and the k_c formula are the Schnakenberg–Turing system, with D_u set by λ_c = 5.2 µm. Diffusion is integrated in Fourier space. The basic-look composite is a visualization on the fossil plate, not a claim that the PDE grows the compression from noise.  
  evidence: `models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py` L93–127; `models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py` L163–199
- **Morphological realism / specimen match: 80** — The last GIF frame is the Kenraiz plate with the activator darkening the leafy axis, so the side-by-side basic look is the same fossil. Pass 1 fixed α = 0.32 on the whole frame (r = 0.98950). That is a conditioned render, and the centimetre-scale plate does not show a 5.2 µm stripe. Score stays at 80 until a hold-out row test exists.  
  evidence: `docs/media/asteroxylon_mackiei_turing_wavelength_s0_comparison.png` L0–0; `docs/media/asteroxylon_mackiei_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 86** — At fixed D_u, raising δ from 8 to 320 shortens λ_c monotonically and the sweep is plotted. The equal-diffusion control is on the dispersion panel. α was not yet chosen by a contrast rule.  
  evidence: `models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py` L129–161; `docs/media/asteroxylon_mackiei_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 82** — Pass 1 reports full-frame Pearson r = 0.98950, RMSE = 7.425 gray levels, activator spectral centroid 7.646 µm against the 5.2 µm law, plate foreground / circularity / coherency, and σ_max of the equal-diffusion control. It does not yet report a hold-out r, a KS distance, or the plate envelope in micrometres. Those three are remediation, so this dimension does not receive Pass 2 credit.  
  evidence: `models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py` L201–282
- **Source and data traceability: 90** — Kidston & Lang, Turing, Schnakenberg, Hetherington, and Edwards DOIs are in the bibliography. The plate is CC BY-SA 4.0, Kenraiz, SHA-256 pinned, with the Wikimedia file page in the reference JSON.  
  evidence: `datasets/asteroxylon_mackiei_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/asteroxylon_mackiei_turing_wavelength_s0/asteroxylon_mackiei_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — Module docstring names the kinetics and k_c. Lean 4 Loop_3124 proves the diffusion-ratio threshold, the 30-mode count, the >50× separation from the 283.4 µm sibling, and the negative trace, with no sorry.  
  evidence: `morphoeval/lean/loops/Loop_3124_asteroxylon_mackiei_turing_wavelength_s0.lean` L15–51; `morphoeval/math_kg_loops/delphi10000-asteroxylon-mackiei-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 86** — Seed 3124, SHA-256 pins on the five dataset files, and --self-test asserts trace, determinant, σ(k_c) > 0, control σ_max < 0, monotonic sweep, r > 0.98, 5.2 µm, and 407 Ma. Pass 1 self-test passed at r = 0.98950.  
  evidence: `models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py` L66–90; `models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py` L573–609

Remediation:
- {'id': 'R1', 'item': 'Choose activator contrast on the top 75% of rows only and report Pearson r on the held-out bottom 25%.'}
- {'id': 'R2', 'item': 'Add a two-sample KS distance between the plate and the last-frame gray histograms, with reference 0.'}
- {'id': 'R3', 'item': 'Measure the leafy-axis envelope wavelength in micrometres under the model scale bar and set it beside 5.2 µm.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Asteroxylon mackiei Schnakenberg critical wavelength at 5.2 µm (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `716e903fe3a6b195756b816d8895f02c8c7f83e58d092ed921cdf0ea6a927c85`
- **input_dataset_sha256**: `datasets/asteroxylon_mackiei_turing_wavelength_s0/README.md` (`8d4cec453c791d09…`), `datasets/asteroxylon_mackiei_turing_wavelength_s0/asteroxylon_mackiei_turing_wavelength_s0_reference.json` (`9577842bd82c5dae…`), `datasets/asteroxylon_mackiei_turing_wavelength_s0/asteroxylon_mackiei_turing_wavelength_s0_specimen.jpg` (`ad3ca375be91c9cf…`), `datasets/asteroxylon_mackiei_turing_wavelength_s0/paper_citations.bib` (`345af56bd8c1eb6a…`), `datasets/asteroxylon_mackiei_turing_wavelength_s0/scout_manifest.json` (`95e0c02c8d34e83a…`)
- `docs/media/asteroxylon_mackiei_turing_wavelength_s0_comparison.png` — artifact_sha256 `fd44e2c72e7684d8…`, 816,591 B, 1344×960 px
- `docs/media/asteroxylon_mackiei_turing_wavelength_s0_growth.gif` — artifact_sha256 `07a44c485620b05b…`, 3,791,522 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Asteroxylon specimen image foreground fraction | 0.8896 area fraction | 0.5 | border-contrast mask on the vendored CC BY-SA plate, resized to 160×120 *(measured foreground 0.8896; plant-dark fraction 0.3149)* |
| Asteroxylon specimen image contour circularity | 0.2378 4πA/P² | 1.0 | morphological perimeter of the border-contrast mask *(elongated axis, not an ocellus disk)* |
| Asteroxylon specimen image structure-tensor coherency | 0.3676 dimensionless | 0.5 | mean structure-tensor coherency inside the specimen mask (σ=2.5) *(rock grain plus a vertical axis)* |
| basic-look pigment Pearson r versus the Asteroxylon plate | 0.987831 Pearson r | 1.0 | Pearson correlation of grayscale pixels, specimen vs last composite frame *(RMSE 8.096 gray levels on 0–255)* |
| basic-look pigment RMSE versus the Asteroxylon plate | 8.0961 gray levels | 0.0 | RMSE of grayscale pixels, specimen vs last composite frame *(nonzero because the Schnakenberg field darkens the axis)* |
| Turing pigment stripe spectral-centroid wavelength | 7.6462 µm | 5.2 | 2π / power-weighted mean wavenumber of the final activator field *(centroid includes the side-band and is not forced equal to 5.2)* |
| equal-diffusion control maximum growth rate | -0.08 per reaction time | 0.0 | max_k σ(k) with D_u = D_v (differential diffusion removed) *(patterning control σ(k_c) at δ=160 is 0.3961)* |
| hold-out pigment Pearson r on the bottom quarter of the Asteroxylon plate | 0.986385 Pearson r | 1.0 | alpha is the largest contrast with train r>=0.985 on the top 75% of rows; r is then scored only on the bottom 25% *(alpha=0.35, train r=0.9874, KS=0.1607)* |
| Asteroxylon specimen image gray-level KS distance | 0.1607 KS statistic | 0.0 | two-sample KS between specimen and last-frame grayscale histograms (512-bin CDF grid) *(Pass 2 remediation of the Pass 1 histogram gap)* |
| Asteroxylon specimen image axis-envelope wavelength | 117.0 µm | 5.2 | dominant Fourier period of the leafy-axis occupancy, converted with the 117 µm model scale bar *(transfer metric: the compression fossil does not resolve the 5.2 µm stripe)* |
| enation-to-detail wavelength ratio | 54.5 λ_283µm / λ_5.2µm | 50.0 | 283.4 µm sibling enation wavelength divided by this slot's 5.2 µm *(scale separation inside Asteroxylon at 407 Ma; parameters are not the sibling's)* |

## Environment at checkpoint


Loop 3124 files are on this commit. Pass 1 scores predate the hold-out, KS, and envelope metrics.
