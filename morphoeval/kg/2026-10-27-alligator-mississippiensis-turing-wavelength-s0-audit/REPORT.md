# MorphoEval report — audit `2026-10-27-alligator-mississippiensis-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `uncommitted-working-tree` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Alligator mississippiensis Holocene 1.498 mm Turing wavelength (`models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py`) | 86 | 74 | 88 | 80 | 92 | 86 | 88 | 86.7 | 76.2 | 88.8 | 83.1 | **83.1** | faithful |

## Findings and remediation

### Alligator mississippiensis Holocene 1.498 mm Turing wavelength  —  overall 83.1 (faithful)

Pass 1. Schnakenberg RD on a 17.976 by 11.984 mm flank (exactly 12 by 8 wavelengths of 1.498 mm). gamma = 2.5909 is solved so k_c = 2 pi / 1.498 mm. Discrete-mode Euler growth matches the dispersion eigenvalues with Pearson r = 0.999999998 and RMSE = 5.0e-5 on the real-eigenvalue band m = 5..20. Equal diffusion gives sigma(m=12) = -1.197 and the control standard deviation falls from 0.0120 to 0.00042. The stripe FFT wavelength is 1.498 mm. gamma multipliers 0.64, 1, 1.96 lock to integer modes at 1.798, 1.498 and 1.057 mm. The Sévi juveniles show cream transverse bands; the rear-tail ROI has 13 peaks at 73 px (cv 0.126) along a 12 degree normal. The pass-1 GIF overlays that field at alpha 0.62 through a dilated mask that also paints rock and reeds. Overall 83.1/100.

- **Model fidelity to source: 86** — The script implements Schnakenberg kinetics, the Turing dispersion relation, and gamma such that k_c = 2 pi / 1.498 mm. Self-test and the pass-1 run both give Pearson r = 0.999999998 between Euler growth and the discrete eigenvalues on modes with a real discriminant (m = 5..20), RMSE about 5e-5. Equal diffusion is stable (sigma = -1.197). Eight Lean theorems with zero sorry cover the trace, determinant, diffusion margin, Turing inequality, the 12-by-8 tiling, and the single-point chronoseries. Deduction: the kinetics are the standard Schnakenberg scheme, not a parameter table published for Alligator pigment, and 1.498 mm is the slot scale rather than a fitted biological constant.  
  evidence: `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L84–147; `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L304–361; `morphoeval/lean/loops/Loop_3105_alligator_mississippiensis_turing_wavelength_s0.lean` L15–49
- **Morphological realism / specimen match: 74** — The comparison figure places the Sévi juveniles, a rear-tail crop with 13 cream bands, and a vertical stripe field of wavelength 1.498 mm on the same page. The motif (repeated light bands on a dark ground) is the same class as the tail. The isotropic neighbour breaks into 47 components rather than stripes. The pass-1 GIF does grow (12 frames, 720x480, frame-to-frame mean absolute difference 0.0375) and the computer-vision score against the photograph is 96.0, but alpha 0.62 is applied through a dilated mask, so stripes also fall on rock and reeds. The photograph's band pitch is not 1.498 mm. Those two limits keep the score below the mid-80s.  
  evidence: `datasets/alligator_mississippiensis_turing_wavelength_s0/pass1_comparison.png` L0–0; `datasets/alligator_mississippiensis_turing_wavelength_s0/pass1_growth.gif` L0–0; `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L444–462
- **Parameter expressivity: 88** — Raising the gamma multiplier from 0.64 to 1 to 1.96 shortens the dominant wavelength from 1.798 mm to 1.498 mm to 1.057 mm. Those are the integer modes nearest to lambda_c / sqrt(gamma) (predicted 1.873, 1.498, 1.070; the grid locks to m = 10, 12, 17). Stripe coherency is 0.997 with profile-to-cosine Pearson r = 0.997. The isotropic run is a different morphology (47 components, not a single stripe family). The equal-diffusion control decays. Theta is not yet varied; that is remediation R2, so the score is not 95.  
  evidence: `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L210–257; `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L871–881; `datasets/alligator_mississippiensis_turing_wavelength_s0/alligator_mississippiensis_turing_wavelength_s0_metrics_pass1.json` L0–0
- **Morphometric validation: 80** — The simulated stripe wavelength is 1.498 mm against the nominal 1.5 mm slot scale (they are not forced equal). Dispersion Pearson r = 0.999999998, RMSE = 5.01e-5. Rear-tail cream peaks: 13, median spacing 73 px, cv = 0.126, normal 12 degrees. Fore-flank: 19 peaks at 39.5 px. KS distance between median-normalised spacing distributions is 0.5 (they are not the same distribution). The photograph has no scale bar, so a millimetre reading of the 73 px pitch is not yet reported. That conversion is remediation R3.  
  evidence: `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L409–442; `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L839–860; `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L1028–1110
- **Source and data traceability: 92** — Eight DOIs were checked against Crossref on 2026-10-27, including the real Milinkovitch 2013 cracking paper (10.1126/science.1226265), which is cited as a boundary rather than as a fake Turing DOI. The specimen is Ianaré Sévi, CC BY-SA 3.0, SHA-256 a4a5ed7f280ee2de43a726ab16d9d681bbb4dd6ac032c141591dc8a7aff276de, checked at runtime. The species-gallery spiral was rejected in the dataset README. Deduction: OpenAlex's first hit on the naive query was not a morphogenesis paper; the scout record says so.  
  evidence: `datasets/alligator_mississippiensis_turing_wavelength_s0/paper_citations.bib` L1–97; `datasets/alligator_mississippiensis_turing_wavelength_s0/alligator_mississippiensis_turing_wavelength_s0_reference.json` L0–0; `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L1117–1130
- **Code-to-model traceability: 86** — The module docstring states the PDE, the wavenumber formula, the periodic patch, and the pass-1 versus pass-2 split. derive_gamma, step_nonlinear, dispersion_check and the Lean theorems are the same identities. There is no equation number from Murray 1981 printed in the source, which is the deduction.  
  evidence: `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L1–28; `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L173–179
- **Reproducibility package: 88** — --self-test checks the fixed point, the signs of the trace, determinant and both diffusion margins, the k_c identity, the theta=0 tensor, a dispersion Pearson r above 0.98, growth of the stripe seed, decay of the control, and the Euler CFL bound. Seed 3105 and dt = 0.0015 are fixed. The pass-1 run finished in about 22 s. Dependencies are the repo scientific stack (numpy, scipy, pillow, matplotlib) rather than a per-script lock file.  
  evidence: `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L1117–1183; `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L1186–1194

Remediation:
- {'id': 'R1', 'item': 'Confine the GIF overlay to tight_animal_mask so the 1.498 mm field sits on the juveniles rather than on rock and reeds.'}
- {'id': 'R2', 'item': 'Rotate the diffusion tensor from theta = 0 onto the measured rear-tail band normal (12 degrees).'}
- {'id': 'R3', 'item': 'Report photo-band pitch in millimetres under assumed total lengths 300, 450 and 600 mm, using the integument major axis rather than the dilated mask.'}
- {'id': 'R4', 'item': 'Measure the upper-neck ROI on the second juvenile in the same frame.'}
- {'id': 'R5', 'item': "This pass is single-assessor. A second provider has not scored it; that limit stays visible and is not filled in by copying the builder's numbers into a second file."}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Alligator mississippiensis Holocene 1.498 mm Turing wavelength
- **spec_fingerprint** (semantic model + inputs): `fae26d87967d4118683e0f0b8d83dcefcf0e38b2299c821d238a0bfda7bc67c0`
- **input_dataset_sha256**: `datasets/alligator_mississippiensis_turing_wavelength_s0/Alligator_mississippiensis_2_babies.jpg` (`a4a5ed7f280ee2de…`), `datasets/alligator_mississippiensis_turing_wavelength_s0/alligator_mississippiensis_turing_wavelength_s0_reference.json` (`b935d3f5134b0b50…`)
- `datasets/alligator_mississippiensis_turing_wavelength_s0/alligator_mississippiensis_turing_wavelength_s0_metrics_pass1.json` — artifact_sha256 `358804aec45683a7…`, 2,128 B, 
- `datasets/alligator_mississippiensis_turing_wavelength_s0/alligator_mississippiensis_turing_wavelength_s0_reference.json` — artifact_sha256 `b935d3f5134b0b50…`, 6,189 B, 
- `datasets/alligator_mississippiensis_turing_wavelength_s0/pass1_comparison.png` — artifact_sha256 `91799bcef508e95b…`, 610,280 B, 1474×946 px
- `datasets/alligator_mississippiensis_turing_wavelength_s0/pass1_growth.gif` — artifact_sha256 `7199a03959a9442b…`, 2,616,426 B, 720×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| simulated stripe wavelength | 1.498 mm | 1.5 | radial FFT peak of the final Schnakenberg activator on the 17.976 mm flank *(analytic lambda_c = 1.498; gamma = 2.590901)* |
| dispersion pearson r numeric vs analytic | 0.9999999983380724 correlation | 1.0 | Pearson r of Euler growth rates against discrete-Laplacian eigenvalues for modes m = 3..20 *(rmse = 0.000050; ks not this row; p = 9.39e-61)* |
| equal-diffusion negative control sigma at m=12 | -1.19654556243601 1/time | 0.0 | discrete dispersion with D_u = D_v = 0.05 and anisotropy 1 *(control std 0.0120 -> 0.0004)* |
| specimen image rear-tail stripe spacing | 73.0 px | 1.498 | cream-index projection along the structure-tensor normal of the rear-tail ROI *(cv = 0.12623783718073947; n_peaks = 13)* |
| specimen vs simulated stripe spacing shape KS | 0.5 KS distance | 0.0 | ks_2samp on median-normalised inter-peak spacings of the specimen tail and the simulated field *(ks p = 0.09270296267200293; sim spacing cv = 0.0)* |
| specimen image flank cream-peak count | 19 count | 13 | peak count on the fore-flank ROI cream index *(fore flank versus rear tail, same Miami-Dade frame)* |

## Environment at checkpoint


Pass-1 working tree for slot 3105. Commit omitted so nanopub loci are not pinned to a parent commit that does not yet contain these new files.


## Nanopublication status

`python3 morphoeval/build_morphoeval_kg.py <this-audit> --conform` validated the audit against rubric 0.1.0 and then exited before writing TriG. The builder's message was: `no publisher ORCID: set audit.publisher_orcid or ~/.nanopub/profile.yml`. No ORCID was invented, and `~/.nanopub/profile.yml` is not on this machine, so nothing was signed and `morphoeval/sign_morphoeval.py` / `node morphoeval/check_signed.mjs` were not run. This report is the builder's `report()` text for the scored audit; it is not a claim that nanopublications were published.
