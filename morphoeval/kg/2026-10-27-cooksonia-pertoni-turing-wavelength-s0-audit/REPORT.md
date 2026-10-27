# MorphoEval report — audit `2026-10-27-cooksonia-pertoni-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `8308af02867b4125a973e632c1b5c530ef0ddf71` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cooksonia pertoni Turing wavelength at 1.3 µm (425 Ma) (`models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py`) | 86 | 82 | 88 | 87 | 90 | 86 | 86 | 86.7 | 83.9 | 87.3 | 85.8 | **85.8** | faithful |

## Findings and remediation

### Cooksonia pertoni Turing wavelength at 1.3 µm (425 Ma)  —  overall 85.8 (faithful)

Schnakenberg kinetics at a=0.12, b=0.84, D_v/D_u=10, just above the Turing threshold. k_c=2.356, lambda_c=1.339 µm at 425 Ma. Equal diffusion (predecessor) has sigma_max=-0.086. A longer diffusion length for Rhynia gwynne-vaughanii at 407 Ma moves lambda_c to 1.695 µm. Dispersion Pearson r against the discrete Laplacian is 1.000 on six real-eigenvalue modes. The growth GIF is a Y-shaped axis on a matrix whose radial spectrum matches the Amgueddfa Cymru compression (CV similarity 95.4). The 1.3 µm field is the comparison-figure panel, not a feature resolved by the photograph.

- **Model fidelity to source: 86** — The Jacobian at a=3/25, b=21/25 is the exact Schnakenberg linearisation (trace -0.172, det 0.922). Dispersion uses J - k^2 D, and six real-eigenvalue Fourier modes grown with the same 3-point Laplacian match analytic sigma with Pearson r = 1.000 and RMSE 1.4e-4. Equal diffusion is stable. Deducted because Edwards published no reaction-diffusion system, and the 1.339 µm scale is imposed by the length unit rather than measured on a tracheid.  
  evidence: `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L95–144; `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L187–239; `morphoeval/lean/loops/Loop_3153_cooksonia_pertoni_turing_wavelength_s0.lean` L5–33; `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L1–55
- **Morphological realism / specimen match: 82** — The 12-frame 720x540 GIF ends as a dark isotomous axis on grain whose computer-vision similarity to the Amgueddfa Cymru compression is 95.4 (orientation, radial FFT, coherency, occupancy). The comparison figure places that photograph beside the tracheid-wall field. Deducted because the matrix is spectrally matched rather than a registered overlay, the rendered axis is smoother and greyer than the fossil, and the 1.3 µm bands are not visible on the GIF (they are washed out by the shell-power match). Template opening angle on the photo is 70 degrees; the drawn axis is 64.4 degrees.  
  evidence: `docs/media/cooksonia_pertoni_turing_wavelength_s0_growth.gif` L0–0; `docs/media/cooksonia_pertoni_turing_wavelength_s0_comparison.png` L0–0; `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_amgueddfa_specimen.jpg` L0–0
- **Parameter expressivity: 88** — D_v/D_u swept through {8, 9, 10, 12, 16, 24}. sigma_max moves -0.037, +0.006, +0.043, +0.102, +0.184, +0.282 and lambda_c moves 1.271, 1.307, 1.339, 1.396, 1.484, 1.617 µm, both monotonic, with ratio 8 below threshold and ratio 10 the focal point. a and b are not exposed as a CLI sweep.  
  evidence: `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L586–598; `docs/media/cooksonia_pertoni_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 87** — Discrete-Laplacian dispersion Pearson r = 1.000 (RMSE 1.4e-4) on modes 3-8. Specimen radial-spectrum Pearson r = 0.9865, RMSE 0.205, gray KS 0.028 against the Amgueddfa photograph. Model branch angle 64.4 degrees versus template 70.0 degrees on that photo and 67.5 degrees on the NMNH slab (values differ from their references). Stripe spacing of the saturated field is 1.339 µm. Successor wavelength 1.695 µm is longer than the focal wavelength. Deducted because the branch angle is a coarse template (2.5 degree steps), not a published morphometric, and the spacing CV is exactly 0 on a pure mode.  
  evidence: `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L714–844; `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Five DOIs in the bib, confirmed against OpenAlex or Crossref on 2026-10-27. The fitness image is the Amgueddfa Cymru file, CC BY-SA 4.0, SHA-256 aa48bf1c9b50b41dc7f1ef6c6fc8b73b112b59e09f2a8d6ebb4d2aa8f97a0d57, pinned in the reference JSON. Deducted because the Nature paper is not open access, so tracheid diameters were not read off a figure.  
  evidence: `datasets/cooksonia_pertoni_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 86** — The module docstring states the PDE, the exact Jacobian, the predecessor and successor, and the claim boundary. Lean theorems are the cross-multiplied trace, determinant, and threshold inequalities, not renamed arithmetic. Deducted because the shell-power match that builds the GIF is a rendering calibration, not an equation in Turing 1952.  
  evidence: `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L1–70; `morphoeval/math_kg_loops/delphi10000-cooksonia-pertoni-turing-wavelength-s0.yaml` L1–55
- **Reproducibility package: 86** — Seed 3153, explicit-Euler dt below the D_v stability bound, and --self-test passes (k_c, 1.339 µm, predecessor sigma_max < 0, successor wavelength longer, dispersion r > 0.98, amplitude ratio > 2, GIF >= 10 frames and >= 480 px). Deducted because dependency pins live in the repo requirements file rather than this script, and Lean was not executed here (no lean binary).  
  evidence: `models/2026-delphi10000-cooksonia-pertoni-turing-wavelength-s0.py` L846–878; `datasets/cooksonia_pertoni_turing_wavelength_s0/README.md` L1–40

Remediation:
- {'id': 'R1', 'item': 'The GIF axis does not show helical Turing bands; shell-power matching removes them. Restore a visible axial modulation and re-measure CV similarity.'}
- {'id': 'R2', 'item': 'Branch-angle template uses 2.5 degree steps. Refine the sweep so the Amgueddfa and NMNH angles are not grid-snapped.'}
- {'id': 'R3', 'item': 'No independent second assessor. This pass is builder-scored.'}
- {'id': 'R4', 'item': 'No ~/.nanopub/profile.yml. Do not claim a signature.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Cooksonia pertoni Turing wavelength at 1.3 µm (425 Ma)
- **spec_fingerprint** (semantic model + inputs): `ce7d40be6d01f3179ec22b3cb8fbb558f6f40b6cad5c716ca069f671182a9ec4`
- **input_dataset_sha256**: `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_amgueddfa_specimen.jpg` (`aa48bf1c9b50b41d…`), `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_nmnh_specimen.jpg` (`6c6d03579c80de68…`), `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_turing_wavelength_s0_reference.json` (`b36860960a26851c…`)
- `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_amgueddfa_specimen.jpg` — artifact_sha256 `aa48bf1c9b50b41d…`, 3,097,667 B, 4000×2667 px
- `datasets/cooksonia_pertoni_turing_wavelength_s0/cooksonia_pertoni_turing_wavelength_s0_metrics.json` — artifact_sha256 `2d7fae73061af681…`, 4,310 B, 
- `docs/media/cooksonia_pertoni_turing_wavelength_s0_comparison.png` — artifact_sha256 `93e88f9e5d0b1916…`, 624,221 B, 1344×1008 px
- `docs/media/cooksonia_pertoni_turing_wavelength_s0_growth.gif` — artifact_sha256 `973693e0e11bb722…`, 5,990,082 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing critical wavenumber k_c | 2.356077 1/dimensionless length | 2.4 | argmax_k Re sigma_+(k) of Schnakenberg J - k^2 D at D_v/D_u = 10 *(selected mode at 425 Ma; physical λ_c fixed at 1.339 µm)* |
| selected wavelength | 1.339 µm | 1.3 | 2π/k_c times the length unit ξ = 1.339e-6 m / λ_dim |
| predecessor equal-diffusion sigma_max | -0.085937 1/time | 0.0 | sigma_plus with D_u = D_v = 0.055 (pre-vascular control) *(negative value is the pre-425 Ma absence of a selected band)* |
| successor Rhynia wavelength | 1.695151 µm | 1.3389999999999997 | same kinetics, D_u = 0.088, D_v/D_u = 10, same ξ as the 425 Ma calibration *(reference is the focal wavelength so the delta is visible; value is the successor)* |
| dispersion Pearson r | 1.0 correlation | 0.98 | finite-difference growth of Fourier modes vs analytic sigma_+(k) |
| dispersion RMSE | 0.00014 1/time | 0.05 | RMSE of measured vs analytic sigma across 6 real-eigenvalue circumferential modes |
| specimen radial spectrum Pearson r | 0.986514 correlation | 0.98 | 22-bin log radial FFT of Amgueddfa crop vs growth-frame crop *(spectrum match of the rendered compression, not a µm-scale overlay)* |
| specimen radial spectrum RMSE | 0.204623 log power | 0.15 | RMSE of the same 22-bin radial profiles |
| specimen image gray KS statistic | 0.028266 KS distance | 0.05 | two-sample KS between specimen and rendered gray values |
| specimen axis branch angle | 64.365399 degrees | 70.0 | opening angle of the reconstructed isotomous axis *(model geometry and ridge PCA are different estimators)* |
| stripe spacing CV | 0.0 coefficient of variation | 0.05 | CV of peak gaps along the circumferential mean of activator u |
| control amplitude ratio | 241.627139 ratio | 1.0 | std(u) of the Turing run divided by std(u) of the equal-diffusion ablation |

## Environment at checkpoint


Pass 1 tree for slot 3153. Specimen JPEGs are force-added because datasets/*/*_specimen.jpg is gitignored. No nanopub profile on this machine, so publication is recorded as unsigned.
