# MorphoEval report — audit `2026-10-27-cooksonia-pertoni-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `d80cc9da2650b08c89dee19e5506350ec29b0f2e` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-cooksonia-pertoni-turing-wavelength-s0-audit -> 2026-10-27-cooksonia-pertoni-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cooksonia pertoni Turing wavelength at 1.3 µm (425 Ma) (`models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py`) | 86 | 84 | 88 | 89 | 90 | 87 | 86 | 86.7 | 85.9 | 87.6 | 86.6 | **86.6** | faithful |

## Findings and remediation

### Cooksonia pertoni Turing wavelength at 1.3 µm (425 Ma)  —  overall 86.6 (faithful)

Same Schnakenberg system as pass 1 (k_c=2.356, lambda_c=1.339 µm at 425 Ma, predecessor sigma_max=-0.086, Rhynia successor lambda_c=1.695 µm). Pass 2 imprints the axial Turing cut on the axis after spectral matching, and replaces the 2.5 degree angle grid with a quadratic peak. Specimen radial Pearson r is 0.988. GIF computer-vision similarity to the Amgueddfa compression is 95.8 at 720x540 and 12 frames. Dispersion Pearson r remains 1.000. Still one assessor, and still unsigned.

- **Model fidelity to source: 86** — Unchanged from pass 1. Jacobian trace -0.172 and det 0.922, dispersion Pearson r = 1.000 (RMSE 1.4e-4) on modes 3-8 against the discrete Laplacian, equal-diffusion predecessor stable. The new imprint uses the axial cut of the same activator field; it does not add a second law. The 1.339 µm scale is still imposed by the length unit.  
  evidence: `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L95–144; `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L187–240; `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L457–475; `morphoeval/lean/loops/Loop_3153_cooksonia_pertoni_turing_wavelength_s0.lean` L5–33
- **Morphological realism / specimen match: 84** — Pass 1 CV similarity was 95.4 with a smooth axis. Pass 2 CV similarity is 95.8 after the wall-band imprint, at 12 frames and 720x540. The bands are low contrast: they read as a periodic darkening along the branches, not as a resolved micrograph of 1.3 µm thickenings. Amgueddfa template angle is now 69.97 degrees against the drawn 64.37 degrees. The matrix is still a spectral match, not a pixel registration, so the score stays in the mid-80s.  
  evidence: `docs/media/cooksonia_pertoni_turing_wavelength_s0_growth.gif` L0–0; `docs/media/cooksonia_pertoni_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L508–516
- **Parameter expressivity: 88** — Unchanged sweep. D_v/D_u in {8, 9, 10, 12, 16, 24} moves sigma_max monotonically from -0.037 to +0.282 and lambda_c from 1.271 to 1.617 µm. Ratio 8 is stable; ratio 10 is the 425 Ma operating point. a and b are still fixed.  
  evidence: `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L622–631; `docs/media/cooksonia_pertoni_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 89** — Dispersion Pearson r = 1.000, RMSE 1.4e-4. Specimen radial Pearson r rose from 0.9865 to 0.9882. Gray KS remains about 0.028. Branch angles are no longer snapped to a 2.5 degree grid: Amgueddfa 69.97 degrees, NMNH 68.03 degrees, drawn axis 64.37 degrees, so value is not equal to reference. Successor wavelength 1.695 µm. Stripe-spacing CV of the pure mode is 0, reported against a 0.05 tracing tolerance rather than copied onto the value. Still one photograph pair and no published tracheid spacing to hit.  
  evidence: `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L381–433; `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L755–884; `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Same five DOIs and the same CC BY-SA Amgueddfa file, SHA-256 aa48bf1c9b50b41dc7f1ef6c6fc8b73b112b59e09f2a8d6ebb4d2aa8f97a0d57. Nature 1992 is not open access, so no tracheid diameter was read from a figure.  
  evidence: `datasets/cooksonia_pertoni_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 87** — Docstring, Lean inequalities, and the new imprint function are each tied to a stated step. The imprint is labelled as a display of the axial cut, not as a second PDE. Shell-power matching remains a rendering calibration outside Turing 1952, which is why this is not higher.  
  evidence: `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L1–70; `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L457–475
- **Reproducibility package: 86** — --self-test passes on the pass-2 tree: 1.339 µm, dispersion r > 0.98, predecessor stable, successor longer, GIF >= 10 frames and >= 480 px. Seed 3153. Lean is still not executed in this image, and dependencies are the repo requirements file.  
  evidence: `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L887–916; `datasets/cooksonia_pertoni_turing_wavelength_s0/README.md` L1–40

Remediation:
- {'id': 'R1', 'item': 'Closed in pass 2. imprint_wall_bands runs after match_shell_power. CV similarity 95.8. Bands are low contrast.', 'status': 'closed'}
- {'id': 'R2', 'item': 'Closed in pass 2. Template is 141 angles and a quadratic peak: Amgueddfa 69.97 degrees, NMNH 68.03 degrees.', 'status': 'closed'}
- {'id': 'R3', 'item': 'Open. No independent second assessor. This pass is still builder-scored.', 'status': 'open'}
- {'id': 'R4', 'item': 'Open. ~/.nanopub/profile.yml is absent. Nanopubs were built for conformance and were not signed.', 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cooksonia pertoni Turing wavelength at 1.3 µm (425 Ma)
- **spec_fingerprint** (semantic model + inputs): `0469116f4f2b9848b31bd188b5d42585042d519d8571ea93be197af7958cd169`
- **input_dataset_sha256**: `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_amgueddfa_specimen.jpg` (`aa48bf1c9b50b41d…`), `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_nmnh_specimen.jpg` (`6c6d03579c80de68…`), `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_turing_wavelength_s0_reference.json` (`93a1a0f57016c745…`)
- `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_turing_wavelength_s0_metrics.json` — artifact_sha256 `4ae844cc3b57ed7d…`, 4,337 B, 
- `docs/media/cooksonia_pertoni_turing_wavelength_s0_comparison.png` — artifact_sha256 `93e88f9e5d0b1916…`, 624,221 B, 1344×1008 px
- `docs/media/cooksonia_pertoni_turing_wavelength_s0_growth.gif` — artifact_sha256 `cf9f5da25f5ea446…`, 5,887,468 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing critical wavenumber k_c | 2.356077 1/dimensionless length | 2.4 | argmax_k Re sigma_+(k) of Schnakenberg J - k^2 D at D_v/D_u = 10 *(selected mode at 425 Ma; physical λ_c fixed at 1.339 µm)* |
| selected wavelength | 1.339 µm | 1.3 | 2π/k_c times the length unit ξ = 1.339e-6 m / λ_dim |
| predecessor equal-diffusion sigma_max | -0.085937 1/time | 0.0 | sigma_plus with D_u = D_v = 0.055 (pre-vascular control) *(negative value is the pre-425 Ma absence of a selected band)* |
| successor Rhynia wavelength | 1.695151 µm | 1.3389999999999997 | same kinetics, D_u = 0.088, D_v/D_u = 10, same ξ as the 425 Ma calibration *(reference is the focal wavelength so the delta is visible; value is the successor)* |
| dispersion Pearson r | 1.0 correlation | 0.98 | finite-difference growth of Fourier modes vs analytic sigma_+(k) |
| dispersion RMSE | 0.00014 1/time | 0.05 | RMSE of measured vs analytic sigma across 6 real-eigenvalue circumferential modes |
| specimen radial spectrum Pearson r | 0.988177 correlation | 0.98 | 22-bin log radial FFT of Amgueddfa crop vs growth-frame crop *(spectrum match of the rendered compression, not a µm-scale overlay)* |
| specimen radial spectrum RMSE | 0.235284 log power | 0.15 | RMSE of the same 22-bin radial profiles |
| specimen image gray KS statistic | 0.029586 KS distance | 0.05 | two-sample KS between specimen and rendered gray values |
| specimen axis branch angle | 64.365399 degrees | 69.96807044554417 | opening angle of the reconstructed isotomous axis *(model geometry and ridge PCA are different estimators)* |
| stripe spacing CV | 0.0 coefficient of variation | 0.05 | CV of peak gaps along the circumferential mean of activator u |
| control amplitude ratio | 241.627139 ratio | 1.0 | std(u) of the Turing run divided by std(u) of the equal-diffusion ablation |

## Environment at checkpoint


Pass 2 of slot 3153. Wall bands are imprinted after the shell-power match. Branch angles use a quadratic peak on a 0.5 degree grid. Unsigned: no ~/.nanopub/profile.yml.
