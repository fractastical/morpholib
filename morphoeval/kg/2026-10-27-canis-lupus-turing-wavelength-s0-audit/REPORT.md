# MorphoEval report — audit `2026-10-27-canis-lupus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 1.781 µm on a Holocene Canis lupus (MDCK) apical patch (`models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py`) | 88 | 80 | 68 | 80 | 86 | 82 | 84 | 81.3 | 80.0 | 84.1 | 81.6 | **81.6** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 1.781 µm on a Holocene Canis lupus (MDCK) apical patch  —  overall 81.6 (faithful)

Pass 1 integrates the Schnakenberg system so that the determinant formula for k_c lands on 1.781 µm (Holocene, 0 Ma, domain 14.248 µm = 8 wavelengths). Continuum versus coarse finite-difference dispersion curves give Pearson r = 0.9935 (RMSE 1.002). The saturated Fourier peak sits at 0.976 k_c. The growth GIF is 16 frames at 640×480; its last frame is the canonical micrograph modulated by the activator (CV similarity 96.0 against that micrograph). Specimen nearest-neighbour spacing is 8.94 px and is not converted to micrometres. The diffusivity sweep, equal-diffusion figure, gamma=0 panel, and KS area test are implemented but gated off (REMEDIATION_PASS = 1), so this pass does not score them as done.

- **Model fidelity to source: 88** — The script implements Schnakenberg kinetics, the homogeneous Jacobian, and the Turing determinant formula for k_c, then integrates them with a Fourier operator split on a periodic patch. Clip bounds are present and recorded a hit fraction of 0 on the audited run, so they did not move the trajectory. Deducted because the physical diffusivities use a reconstructed 1 s time unit, and because the pass-1 figure does not yet show the equal-diffusion ablation that the self-test checks analytically.  
  evidence: `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L72–96; `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L159–214
- **Morphological realism / specimen match: 80** — The comparison figure places the canonical Canis lupus micrograph beside the activator lattice and beside the micrograph multiplied by that lattice. The 16-frame GIF ends on that same modulated micrograph and scores CV similarity 96.0. Deducted because the histological polygons are the photograph, not an emergent output of the PDE, and because the micrograph has no scale bar linking those polygons to 1.781 µm.  
  evidence: `docs/media/canis_lupus_turing_wavelength_s0_growth.gif` L0–0; `docs/media/canis_lupus_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L347–381
- **Parameter expressivity: 68** — a, b, gamma, and the diffusivity ratio are named constants and the self-test shows that equal diffusivities quench the instability while the slot ratio does not. The sweep over D_v/D_u and the control bars are coded but not executed or drawn while REMEDIATION_PASS is 1, so this pass does not yet show monotonic control of wavelength.  
  evidence: `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L49–65; `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L239–269
- **Morphometric validation: 80** — measure_morphoeval reports lambda_c, k_c, dispersion Pearson r = 0.9935 against a reference of 1 (not equal), RMSE = 1.002 against 0, saturated k_meas/k_c = 0.976 against 1, activator peak spacing 1.754 µm against 1.781 µm, and segmented specimen nearest-neighbour spacing 8.94 px against the model wavelength in micrometres. The pixel spacing is a real segmentation and is deliberately not treated as a micrometre measurement. KS and the control stds are absent from this pass's sidecar.  
  evidence: `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L128–157; `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L271–339; `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L562–640
- **Source and data traceability: 86** — Crossref-checked DOIs for Turing 1952, Schnakenberg 1979, Murray 1981, Kondo and Miura 2010, Martín-Belmonte 2007, and Candille 2007 are in the BibTeX and the reference JSON. The specimen SHA-256 matches the species-gallery file. OpenAlex keyword search was off-topic; records were kept by DOI. No nanopub signature is claimed.  
  evidence: `datasets/canis_lupus_turing_wavelength_s0/paper_citations.bib` L1–88; `datasets/canis_lupus_turing_wavelength_s0/canis_lupus_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 82** — The module docstring states the PDE, the steady state, and the k_c formula, and each is implemented under a named function. Lean theorems in MathOfBiology.Loop_3139 encode the same sign conditions. Deducted because the pass-1 figure's text panel, rather than a cited equation number in Murray, carries the coat-scale comparison.  
  evidence: `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L1–36; `morphoeval/lean/loops/Loop_3139_canis_lupus_turing_wavelength_s0.lean` L19–50
- **Reproducibility package: 84** — --self-test checks trace, determinant, Turing margin, equal-diffusion quenching, lambda_c = 1.781e-6 m, log10 ≈ -5.749, and Pearson r > 0.98. The integrator seed is 3139. Invocation is the script path with no extra data beyond the vendored specimen. Deducted because pass 2 is a hand-flipped constant rather than a CLI flag, and because signing keys are absent on this machine.  
  evidence: `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L692–727; `models/2026-delphi10000-canis-lupus-turing-wavelength-s0.py` L63–63

Remediation:
- {'id': 'R1', 'item': 'Execute sweep_diffusion_ratio, plot lambda_c against D_v/D_u, and write the sweep into the measurement sidecar (set REMEDIATION_PASS = 2).'}
- {'id': 'R2', 'item': 'Run the equal-diffusion and gamma=0 integrations and show their activator standard deviations next to the patterned regime.'}
- {'id': 'R3', 'item': 'Compute the median-normalised two-sample KS statistic between specimen domain areas and simulated activator-peak areas, with value not equal to the reference.'}
- {'id': 'R4', 'item': 'Independent second assessor was not run. Left open; this pass is builder-scored only.'}
- {'id': 'R5', 'item': 'Keep the explicit statement that specimen spacing is in pixels and is not a 1.781 µm measurement once the pass-2 panels replace the text block.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 1.781 µm on a Holocene Canis lupus (MDCK) apical patch
- **spec_fingerprint** (semantic model + inputs): `62f18d9df76b5263186739999d39e30372650ea68bead990595950ce9e6644ae`
- **input_dataset_sha256**: `datasets/canis_lupus_turing_wavelength_s0/canis_lupus_turing_wavelength_s0_reference.json` (`d16ba21863080c9e…`), `datasets/canis_lupus_turing_wavelength_s0/canis_lupus_turing_wavelength_s0_specimen.jpg` (`8dd4a5b3caa19a6f…`)
- `docs/media/canis_lupus_turing_wavelength_s0_comparison.png` — artifact_sha256 `747326cd4a188994…`, 686,761 B, 1512×984 px
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

## Environment at checkpoint


Pass 1 working tree on top of this commit. REMEDIATION_PASS = 1.
