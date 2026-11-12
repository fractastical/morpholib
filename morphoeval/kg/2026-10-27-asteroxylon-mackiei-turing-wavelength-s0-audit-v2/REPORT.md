# MorphoEval report — audit `2026-10-27-asteroxylon-mackiei-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `8f0e4833ae7c01aa84482096856ead2148c3f912` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-asteroxylon-mackiei-turing-wavelength-s0-audit -> 2026-10-27-asteroxylon-mackiei-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Asteroxylon mackiei Schnakenberg critical wavelength at 5.2 µm (Pass 2) (`models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py`) | 90 | 84 | 88 | 90 | 90 | 86 | 88 | 89.3 | 86.2 | 88.1 | 87.7 | **87.7** | faithful |

## Findings and remediation

### Asteroxylon mackiei Schnakenberg critical wavelength at 5.2 µm (Pass 2)  —  overall 87.7 (faithful)

Pass 2 keeps the 5.2 µm, 407 Ma Schnakenberg calibration (δ = 160, 156 µm × 117 µm window, σ(k_c) = 0.3961, equal-diffusion σ_max = -0.0800). Activator contrast α is now the largest value whose Pearson r on the top 75% of rows stays at or above 0.985. That rule selects α = 0.35. Held-out bottom 25% scores r = 0.98639. Full-frame r = 0.98783, RMSE = 8.096 gray levels, gray-level KS = 0.161. The leafy-axis envelope on the model scale bar is 117 µm, against the 5.2 µm stripe; the spectral centroid of the activator is 7.646 µm. The δ sweep remains monotonic. Asteroxylon chronoseries is 1/1 at 407 Ma; the 283.4 µm enation sibling is 54.5 times longer and is not re-parameterized here.

- **Model fidelity to source: 90** — Pass 1's Jacobian, quadratic dispersion, and pseudospectral integrator are unchanged and still match Schnakenberg at λ_c = 5.2 µm. Pass 2 does not alter the PDE to chase the plate; it only changes how contrast is chosen and how the residual is reported.  
  evidence: `models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py` L93–127; `models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py` L163–199
- **Morphological realism / specimen match: 84** — The re-rendered last frame still shows the Kenraiz leafy axis. Hold-out rows, which did not choose α, correlate at 0.98639, so the match is not confined to the rows used to set contrast. The plate remains a compression fossil; the 117 µm envelope versus the 5.2 µm stripe is why this score is 84 rather than a micrograph overlay.  
  evidence: `docs/media/asteroxylon_mackiei_turing_wavelength_s0_comparison.png` L0–0; `docs/media/asteroxylon_mackiei_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 88** — δ still shortens λ_c monotonically at fixed D_u, and the equal-diffusion curve is on the figure. α is now the largest contrast that keeps the train-row correlation at or above 0.985, which is a one-dimensional expressivity rule with a held-out check.  
  evidence: `models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py` L138–161; `models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py` L310–335
- **Morphometric validation: 90** — Full-frame Pearson r = 0.98783 (reference 1), RMSE = 8.096 gray levels (reference 0), hold-out r = 0.98639 (reference 1), KS = 0.161 (reference 0), stripe centroid 7.646 µm (reference 5.2), axis envelope 117 µm (reference 5.2), control σ_max = -0.080 (reference 0), scale ratio 54.5 (reference 50). None of these equal their references. Plate foreground, circularity, and coherency are measured on the vendored image.  
  evidence: `models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py` L284–335; `models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py` L465–571
- **Source and data traceability: 90** — Unchanged from Pass 1, plus the Gierer & Meinhardt 1972 DOI for the long-range inhibitor. Plate SHA-256 and CC BY-SA attribution remain pinned.  
  evidence: `datasets/asteroxylon_mackiei_turing_wavelength_s0/paper_citations.bib` L1–80; `datasets/asteroxylon_mackiei_turing_wavelength_s0/README.md` L1–30
- **Code-to-model traceability: 86** — Pass 2 functions gray_ks, envelope_wavelength_um, and select_alpha_holdout are named for R1–R3. Lean theorems are unchanged and still have no sorry.  
  evidence: `morphoeval/lean/loops/Loop_3124_asteroxylon_mackiei_turing_wavelength_s0.lean` L18–51; `models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py` L284–335
- **Reproducibility package: 88** — --self-test now also requires hold-out r > 0.98, 0 < KS < 0.5, and envelope > 20 µm. The Pass 2 run passed: r = 0.98783, hold-out = 0.98639, KS = 0.161, envelope = 117 µm, 12 frames at 640×480. Dataset SHA-256 pins still match.  
  evidence: `models/2026-delphi10000-asteroxylon-mackiei-turing-wavelength-s0.py` L573–609

Remediation:
- {'id': 'R1', 'item': 'Closed in Pass 2. α = 0.35 chosen on the top 75% of rows; bottom 25% Pearson r = 0.98639.'}
- {'id': 'R2', 'item': 'Closed in Pass 2. Gray-level KS = 0.161 against reference 0.'}
- {'id': 'R3', 'item': 'Closed in Pass 2. Leafy-axis envelope = 117 µm against the 5.2 µm stripe.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Asteroxylon mackiei Schnakenberg critical wavelength at 5.2 µm (Pass 2)
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


Pass 2 of loop 3124. Supersedes the Pass 1 audit. Nanopubs were not signed: ~/.nanopub/profile.yml is absent.
