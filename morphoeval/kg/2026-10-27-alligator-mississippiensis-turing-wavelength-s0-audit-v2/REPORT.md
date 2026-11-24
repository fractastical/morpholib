# MorphoEval report — audit `2026-10-27-alligator-mississippiensis-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `uncommitted-working-tree` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-alligator-mississippiensis-turing-wavelength-s0-audit -> 2026-10-27-alligator-mississippiensis-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Alligator mississippiensis Holocene 1.498 mm Turing wavelength (`models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py`) | 88 | 82 | 90 | 90 | 92 | 88 | 90 | 88.7 | 85.0 | 90.1 | 87.6 | **87.6** | faithful |

## Findings and remediation

### Alligator mississippiensis Holocene 1.498 mm Turing wavelength  —  overall 87.6 (faithful)

Pass 2 closes R1-R4. The diffusion tensor is rotated onto the measured rear-tail band normal (12 degrees). The GIF overlay uses the tight integument mask, so the 1.498 mm field is no longer painted across the rock. Total-length sensitivity at 300, 450 and 600 mm, using a 1528 px major axis on that mask, puts the photographed cream bands at 14.3, 21.5 and 28.7 mm (9.6, 14.4 and 19.1 times lambda_c). The upper-neck ROI on the second juvenile has 19 peaks at 57 px (cv 0.42), less regular than the tail. Dispersion Pearson r remains 0.999999998 (RMSE 5.0e-5). Stripe FFT wavelength remains 1.498 mm. Computer-vision similarity of the 12-frame 720x480 GIF to the Sévi photograph is 96.0. Overall 87.6/100. The chronoseries is still a single Holocene point at 0 Ma.

- **Model fidelity to source: 88** — Same Schnakenberg stepper and dispersion test as pass 1 (Pearson r = 0.999999998, RMSE = 5.01e-5, equal-diffusion sigma = -1.197). Pass 2 additionally points the slow diffusion axis along the measured band normal rather than leaving theta at 0. Lean theorems are unchanged and still have zero sorry. The deduction that remains is that 1.498 mm and the rate constants are the slot's reconstruction, not a published alligator parameter table.  
  evidence: `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L95–147; `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L982–1018; `morphoeval/lean/loops/Loop_3105_alligator_mississippiensis_turing_wavelength_s0.lean` L15–49
- **Morphological realism / specimen match: 82** — The comparison PNG shows the Sévi juveniles, the rear-tail crop (13 cream bands, normal 12 degrees), and a stripe field at theta = 12 degrees with wavelength 1.498 mm and a drawn 1.498 mm bar. The isotropic panel is a different pattern (47 components). The GIF is 12 frames at 720x480. Frame difference is 0.0135. Computer-vision similarity to the photograph is 96.0 (orientation 0.963, radial FFT 0.9998, coherency 0.996) with the tight mask, so the rock is no longer striped. The score stays short of 90 because the body bands in the photograph are centimetre-scale under every total-length assumption, and the overlay opacity is still 0.62 on the integument.  
  evidence: `docs/media/alligator_mississippiensis_turing_wavelength_s0_comparison.png` L0–0; `docs/media/alligator_mississippiensis_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L465–484
- **Parameter expressivity: 90** — Gamma multipliers 0.64, 1 and 1.96 still shorten the wavelength monotonically (1.798, 1.498, 1.057 mm), locking to the integer mode nearest lambda_c/sqrt(gamma). Pass 2 reruns that sweep at the measured 12 degree angle. Stripe coherency stays above 0.98. The isotropic neighbour remains a spotted field. Equal diffusion still decays (std 0.0120 to 0.00042).  
  evidence: `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L982–1004; `datasets/alligator_mississippiensis_turing_wavelength_s0/alligator_mississippiensis_turing_wavelength_s0_metrics_pass2.json` L0–0
- **Morphometric validation: 90** — Simulated wavelength 1.498 mm versus nominal 1.5 mm. Dispersion Pearson r = 0.999999998 against reference 1, RMSE = 5.01e-5. Stripe profile versus cos(k_c x) has Pearson r = 0.997. KS distance between specimen and simulation spacing shapes is 0.5. Rear tail: 13 peaks, 73 px, cv = 0.126. Fore flank: 19 peaks at 39.5 px. Upper neck: 19 peaks at 57 px, cv = 0.42. With the tight-mask major axis of 1528 px, assumed total lengths 300, 450 and 600 mm convert the 73 px pitch to 14.34, 21.50 and 28.67 mm, which is 9.57, 14.35 and 19.14 times lambda_c. None of these reference comparisons is a copied value.  
  evidence: `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L515–531; `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L839–860; `datasets/alligator_mississippiensis_turing_wavelength_s0/alligator_mississippiensis_turing_wavelength_s0_metrics_pass2.json` L0–0
- **Source and data traceability: 92** — Unchanged from pass 1 and still accurate. Crossref-checked DOIs, Sévi CC BY-SA 3.0 specimen with a runtime SHA-256 check, and an explicit rejection of the synthetic species-gallery spiral. The Milinkovitch citation is the real 2013 Science paper and is used as a negative boundary.  
  evidence: `datasets/alligator_mississippiensis_turing_wavelength_s0/paper_citations.bib` L1–97; `datasets/alligator_mississippiensis_turing_wavelength_s0/alligator_mississippiensis_turing_wavelength_s0_reference.json` L0–0
- **Code-to-model traceability: 88** — Pass 2's rotation, tight mask and total-length table are separate functions called from run_pass2_extension, and the pass-1 branch still returns theta = 0 so the two renders stay distinguishable. The docstring states which pass does which job. Still no Murray equation number in a comment.  
  evidence: `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L1–28; `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L957–980
- **Reproducibility package: 90** — --self-test still passes after the pass-2 code path was added. --pass-id 1 and --pass-id 2 both ran. Pass 1 artifacts are archived under the dataset directory so the supersession is inspectable. Seed, dt and the CFL check are unchanged. No per-script dependency lock file.  
  evidence: `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L1117–1183; `models/2026-delphi10000-alligator-mississippiensis-turing-wavelength-s0.py` L1021–1026

Remediation:
- {'id': 'R1', 'item': 'Closed in pass 2. tight_animal_mask confines the overlay to the juveniles.', 'status': 'resolved'}
- {'id': 'R2', 'item': 'Closed in pass 2. theta = 12 degrees, the measured rear-tail band normal.', 'status': 'resolved'}
- {'id': 'R3', 'item': 'Closed in pass 2. Bands are 14.3, 21.5 and 28.7 mm at assumed total lengths 300, 450 and 600 mm.', 'status': 'resolved'}
- {'id': 'R4', 'item': 'Closed in pass 2. Upper-neck ROI: 19 peaks, 57 px, cv 0.42.', 'status': 'resolved'}
- {'id': 'R5', 'item': 'Still open. One assessor (grok-4.7) scored both passes. No second provider.', 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Alligator mississippiensis Holocene 1.498 mm Turing wavelength
- **spec_fingerprint** (semantic model + inputs): `fae26d87967d4118683e0f0b8d83dcefcf0e38b2299c821d238a0bfda7bc67c0`
- **input_dataset_sha256**: `datasets/alligator_mississippiensis_turing_wavelength_s0/Alligator_mississippiensis_2_babies.jpg` (`a4a5ed7f280ee2de…`), `datasets/alligator_mississippiensis_turing_wavelength_s0/alligator_mississippiensis_turing_wavelength_s0_reference.json` (`b935d3f5134b0b50…`)
- `datasets/alligator_mississippiensis_turing_wavelength_s0/alligator_mississippiensis_turing_wavelength_s0_metrics_pass2.json` — artifact_sha256 `d54ce95afc18ef0e…`, 3,197 B, 
- `datasets/alligator_mississippiensis_turing_wavelength_s0/alligator_mississippiensis_turing_wavelength_s0_reference.json` — artifact_sha256 `b935d3f5134b0b50…`, 6,189 B, 
- `docs/media/alligator_mississippiensis_turing_wavelength_s0_comparison.png` — artifact_sha256 `a9456e5361be0b2c…`, 615,880 B, 1474×946 px
- `docs/media/alligator_mississippiensis_turing_wavelength_s0_growth.gif` — artifact_sha256 `d6eedc6e7271affa…`, 2,705,418 B, 720×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| simulated stripe wavelength | 1.498 mm | 1.5 | radial FFT peak of the final Schnakenberg activator on the 17.976 mm flank *(analytic lambda_c = 1.498; gamma = 2.590901)* |
| dispersion pearson r numeric vs analytic | 0.9999999983380724 correlation | 1.0 | Pearson r of Euler growth rates against discrete-Laplacian eigenvalues for modes m = 3..20 *(rmse = 0.000050; ks not this row; p = 9.39e-61)* |
| equal-diffusion negative control sigma at m=12 | -1.19654556243601 1/time | 0.0 | discrete dispersion with D_u = D_v = 0.05 and anisotropy 1 *(control std 0.0120 -> 0.0004)* |
| specimen image rear-tail stripe spacing | 73.0 px | 1.498 | cream-index projection along the structure-tensor normal of the rear-tail ROI *(cv = 0.12623783718073947; n_peaks = 13)* |
| specimen vs simulated stripe spacing shape KS | 0.5 KS distance | 0.0 | ks_2samp on median-normalised inter-peak spacings of the specimen tail and the simulated field *(ks p = 0.09270296267200293; sim spacing cv = 0.0)* |
| specimen image flank cream-peak count | 19 count | 13 | peak count on the fore-flank ROI cream index *(fore flank versus rear tail, same Miami-Dade frame)* |
| specimen band pitch over lambda_c at assumed TL 450 mm | 14.354275914548877 ratio | 1.0 | rear-tail spacing_px * (450 mm / foreground major-axis px) / 1.498 mm *(total-length sensitivity; cv of the assumption is the 300/450/600 mm bracket)* |

## Environment at checkpoint


Pass-2 working tree. No publisher ORCID and no ~/.nanopub/profile.yml, so the knowledge-graph builder cannot emit signed or unsigned nanopubs. Commit omitted until this loop's files are the commit being published.


## Nanopublication status

`python3 morphoeval/build_morphoeval_kg.py <this-audit> --conform` validated the audit against rubric 0.1.0 and then exited before writing TriG. The builder's message was: `no publisher ORCID: set audit.publisher_orcid or ~/.nanopub/profile.yml`. No ORCID was invented, and `~/.nanopub/profile.yml` is not on this machine, so nothing was signed and `morphoeval/sign_morphoeval.py` / `node morphoeval/check_signed.mjs` were not run. This report is the builder's `report()` text for the scored audit; it is not a claim that nanopublications were published.
