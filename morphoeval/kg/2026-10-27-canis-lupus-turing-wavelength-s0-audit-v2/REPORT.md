# MorphoEval report — audit `2026-10-27-canis-lupus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-canis-lupus-turing-wavelength-s0-audit -> 2026-10-27-canis-lupus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 1.781 µm on a Holocene Canis lupus (MDCK) apical patch (`models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py`) | 90 | 84 | 90 | 88 | 88 | 86 | 88 | 90.0 | 85.5 | 87.5 | 87.4 | **87.4** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 1.781 µm on a Holocene Canis lupus (MDCK) apical patch  —  overall 87.4 (faithful)

Pass 2 turns on the diffusivity sweep, the equal-diffusion and gamma=0 controls, and the KS area comparison that pass 1 left gated. Ratios D_v/D_u of 6 and 8 are linearly stable; 12, 18 and 30 are unstable and lambda_c falls from 1.781 µm to 1.697 µm. Pattern activator std is 0.259; equal diffusion falls to 2.9e-15 and gamma=0 to 6.9e-4. Dispersion Pearson r remains 0.9935 (RMSE 1.002). Specimen-versus-spot area KS is 0.378 (p = 9.2e-9): the micrograph's domain areas are not the spot lattice, and that mismatch is reported rather than fitted away. The GIF is unchanged in kind: 16 frames, 640×480, CV similarity 96.0 against the canonical micrograph. Age 0 Ma, scale 1.781e-6 m. Signing keys are not on this machine.

- **Model fidelity to source: 90** — Schnakenberg kinetics, the Jacobian, and the Turing k_c formula are unchanged and still integrate on the periodic patch with a clip fraction of 0. Pass 2 adds the two controls the dispersion relation predicts: equal diffusion quenches (std 2.9e-15) and gamma=0 only diffuses the seed (std 6.9e-4) against a patterned std of 0.259. The 1 s time unit used to quote D_phys remains a reconstruction.  
  evidence: `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L72–96; `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L383–387
- **Morphological realism / specimen match: 84** — The GIF last frame is the canonical micrograph modulated by the activator (CV similarity 96.0, 16 frames, 640×480). The still figure shows the micrograph, the spot lattice, and the product, and now labels the micrograph as having no scale bar. Deducted because the polygons are the photograph, and because the KS test says their area distribution is not the lattice's.  
  evidence: `docs/media/canis_lupus_turing_wavelength_s0_growth.gif` L0–0; `docs/media/canis_lupus_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L438–438
- **Parameter expressivity: 90** — The sweep is now executed and drawn. D_v/D_u = 6 and 8 have a negative Turing margin and no wavelength; 12, 18 and 30 are unstable and lambda_c decreases from 1.781 µm to 1.733 µm to 1.697 µm. Equal diffusion and gamma=0 are separate switches and both collapse the activator standard deviation, in the direction the mechanism requires.  
  evidence: `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L239–269; `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L456–483
- **Morphometric validation: 88** — Dispersion Pearson r = 0.99352 against a reference of 1, RMSE = 1.002 against 0, saturated k_meas/k_c = 0.976 against 1, and peak spacing 1.754e-6 m against 1.781e-6 m. Specimen nearest-neighbour spacing is 8.94 px against the model wavelength in micrometres (different quantity, stated as such). KS = 0.378 against a reference of 0, so the area shapes do not match; that is the score's ceiling.  
  evidence: `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L128–157; `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L390–421; `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L579–700
- **Source and data traceability: 88** — Same Crossref-checked DOIs and the same SHA-256-pinned gallery specimen as pass 1. The reference JSON states 1.781e-6 m and 0 Ma. The pass-2 metrics file records the sweep, controls, and KS beside those sources. No signature is claimed.  
  evidence: `datasets/canis_lupus_turing_wavelength_s0/paper_citations.bib` L1–88; `datasets/canis_lupus_turing_wavelength_s0/canis_lupus_turing_wavelength_s0_reference.json` L1–48
- **Code-to-model traceability: 86** — Docstring equations, named functions, and the four Lean theorems still match. Pass 2's sweep and controls are the same functions the pass-1 audit named as gated; the figure now calls them. The coat-scale ratio remains an order-of-magnitude transfer from Murray, labelled as such in the footer.  
  evidence: `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L1–36; `morphoeval/lean/loops/Loop_3139_canis_lupus_turing_wavelength_s0.lean` L19–50
- **Reproducibility package: 88** — --self-test still passes (lambda exactly 1.781e-6 m, Pearson r > 0.98, equal-diffusion eigenvalue negative). Seed 3139. REMEDIATION_PASS = 2 is the switch that reproduces this figure. Nanopub signing was not run because ~/.nanopub/profile.yml is absent.  
  evidence: `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L49–49; `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L709–743

Remediation:
- {'id': 'R1', 'item': 'Closed in this pass. Sweep is executed, plotted (ratios 6 and 8 stable; 12, 18, 30 unstable), and written to the sidecar.', 'status': 'closed'}
- {'id': 'R2', 'item': 'Closed in this pass. Equal-diffusion std 2.9e-15 and gamma=0 std 6.9e-4 are drawn against the patterned std 0.259.', 'status': 'closed'}
- {'id': 'R3', 'item': 'Closed in this pass. KS = 0.378 (p = 9.2e-9) on median-normalised areas; value is not the reference 0.', 'status': 'closed'}
- {'id': 'R4', 'item': "Open. No independent second assessor. This score is still the builder's.", 'status': 'open'}
- {'id': 'R5', 'item': 'Closed in this pass. The specimen panel and the footer state that nearest-neighbour spacing is in pixels and is not 1.781 µm.', 'status': 'closed'}
- {'id': 'R6', 'item': "Open residual, not a pass-1 miss. KS = 0.38 means the micrograph's domain areas are a different shape from the spot lattice. Forcing the PDE to copy those polygons would erase the claim boundary.", 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 1.781 µm on a Holocene Canis lupus (MDCK) apical patch
- **spec_fingerprint** (semantic model + inputs): `1384d9b17e356fe071ee14fb12c57b4d3761c15343bddda445ec92854a2f9659`
- **input_dataset_sha256**: `datasets/canis_lupus_turing_wavelength_s0/canis_lupus_turing_wavelength_s0_reference.json` (`d16ba21863080c9e…`), `datasets/canis_lupus_turing_wavelength_s0/canis_lupus_turing_wavelength_s0_specimen.jpg` (`8dd4a5b3caa19a6f…`)
- `docs/media/canis_lupus_turing_wavelength_s0_comparison.png` — artifact_sha256 `38a5f6da5be08515…`, 622,920 B, 1512×984 px
- `docs/media/canis_lupus_turing_wavelength_s0_growth.gif` — artifact_sha256 `d1a71525f7f75bfc…`, 3,945,362 B, 640×480 px, 16 frames

| measurement | value | reference | method |
|---|---:|---|---|
| Turing critical wavelength lambda_c | 1.781e-06 m | 1.781e-06 | lambda_c = 2*pi/k_c with k_c^2 = (D_v f_u + D_u g_v)/(2 D_u D_v), length unit fixed so lambda_c = 1.781e-6 m *(age 0.0 Ma; log10 = -5.7493; domain 14.2480 µm)* |
| Turing wavenumber k_c | 3527897.42121257 1/m | 3527897.4212125693 | k_c = 2*pi/lambda_c at lambda_c = 1.781e-6 m *(dimensionless k_c = 4.222953)* |
| dispersion goodness-of-fit Pearson r | 0.9935158009629034 correlation | 1.0 | Pearson r between continuum sigma(k) and the eigenvalue of a coarse 8-point finite-difference Laplacian, 12 wavenumbers *(RMSE = 1.00213; p = 8.93e-11; value is not equal to the reference)* |
| dispersion RMSE | 1.0021308147510768 1/time | 0.0 | root mean square of continuum sigma(k) minus finite-difference sigma(k) *(same 12-point k grid as the Pearson statistic)* |
| nonlinear dominant wavenumber over k_c | 0.9762812094883317 dimensionless | 1.0 | argmax of the 2D Fourier power of the saturated activator, divided by analytical k_c *(pattern std = 0.2593; clip fraction = 0)* |
| segmented specimen domain nearest-neighbour spacing | 8.94427190999916 px | 1.781 | Otsu on the green channel of the canonical Canis lupus micrograph, nearest-peak spacing of the distance transform *(n_regions = 2813; NN CV = 0.7545334650650951; not equated to 1.781 µm)* |
| simulated activator peak spacing | 1.7540815744998853e-06 m | 1.781e-06 | median nearest-neighbour distance of saturated activator maxima, times metres_per_unit *(spacing/lambda = 0.9849; n_peaks = 60)* |
| equal-diffusion negative-control activator std | 2.9125873420833343e-15 concentration | 0.25929148786790934 | same initial noise, D_v reset to D_u, short spectral integration *(gamma=0 ablation std = 0.000687873)* |
| specimen versus simulation domain-area KS statistic | 0.377581359273503 KS distance | 0.0 | two-sample KS on median-normalised area distributions (specimen watershed vs activator peaks) *(p = 9.151e-09; n_spec = 2813; n_sim = 66)* |
| diffusivity-ratio sweep Pearson r of lambda_c | -0.9625514167733088 correlation | -1.0 | Pearson r between D_v/D_u and lambda_c across the unstable ratios in the sweep *(ratios 6.0, 8.0, 12.0, 18.0, 30.0)* |

## Environment at checkpoint


Pass 2 working tree. REMEDIATION_PASS = 2. Unsigned: no ~/.nanopub/profile.yml.
