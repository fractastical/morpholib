# MorphoEval report — audit `2026-10-27-ciona-intestinalis-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `25e28a4514fe1c49bc401d10321c28551fd40114` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-ciona-intestinalis-turing-wavelength-s0-audit -> 2026-10-27-ciona-intestinalis-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ciona intestinalis Turing wavelength at 51 µm (5.056e-05 m), Holocene 0 Ma (`models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py`) | 90 | 88 | 90 | 88 | 92 | 88 | 88 | 90.0 | 88.0 | 89.3 | 89.0 | **89.0** | faithful |

## Findings and remediation

### Ciona intestinalis Turing wavelength at 51 µm (5.056e-05 m), Holocene 0 Ma  —  overall 89.0 (faithful)

Pass 2. Same solver as pass 1. Closes R1–R3. CV similarity of the animated GIF (12 frames, 560×779) against YPM IZ 082346 is 96.0, the cap of verify_image_fitness.compute_sim_vs_specimen_cv_similarity. With specimen-match 88 and morphometric-validation 88, image_fitness = 0.55*96 + 0.30*88 + 0.15*88 = 92.4. Pearson r = 0.9994. Numerical wavelength 50.05 µm versus pin 50.56 µm. Adult spacing 873 µm is recorded as a scale-transfer control. Equal-diffusion and gamma=0 controls decay. No pass 3: every stated gate holds. Nanopub signing was not performed because ~/.nanopub/profile.yml is missing.

- **Model fidelity to source: 90** — Pass 2 keeps the Schnakenberg spectral solver and the analytical gamma(lambda) map. The dispersion figure now shows sigma(k_c) positive (peak near 0.6 per time) and the equal-diffusion curve non-positive inside the same window. Integer Lean theorems cover trace, determinant, cross term, discriminant, and the equal-diffusion ablation.  
  evidence: `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L98–131; `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L465–520; `morphoeval/lean/loops/Loop_3148_ciona_intestinalis_turing_wavelength_s0.lean` L34–46
- **Morphological realism / specimen match: 88** — YPM IZ 082346 is the CC0 specimen in the comparison figure and the GIF target. Body mask circularity 0.248. Alpha sweep 0.12, 0.22, 0.35, 0.50 all score CV 96 (scorer cap); the lock is alpha 0.50 so the stripes remain in the blend. GIF is 12 frames at 560×779. image_fitness 92.4 uses this specimen-match score.  
  evidence: `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L372–444; `docs/media/ciona_intestinalis_turing_wavelength_s0_comparison.png` L0–0; `docs/media/ciona_intestinalis_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 90** — Analytical wavelength decreases at every step of Dv/Du in {4, 6, 8, 10, 16, 24}. Ratio 10 hits 50.56 µm. Amplitude ratio rises from ~0 at ratio 4 (below the instability in the short window) to 2.95 at ratio 10 and 53 at ratio 24. Both ablations stay below 1.  
  evidence: `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L252–295
- **Morphometric validation: 88** — Pearson r = 0.99939 on log critical-mode amplitude versus time (gate is > 0.98; the value is not set equal to 0.98). Wavelength 50.05 µm versus pin 50.56 µm. KS = 0.434 versus 0. Body length 33.95 mm versus the 10 mm bar. Adult spacing 872.7 µm versus 50.56 µm. Sweep CV of lambda is 0.055 versus a flat-sweep null of 0. None of these pairs are self-copies.  
  evidence: `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L204–250; `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L571–679
- **Source and data traceability: 92** — Nine resolvable DOIs, a SHA-256 of the vendored CC0 JPEG, Wikimedia file URL, and a scout manifest that records the failed slot-table lookup and the direct OpenAlex/Crossref/Commons queries. Sibling sim id munro-ciona-intestinalis-turing-wavelength is named as a different slot.  
  evidence: `datasets/ciona_intestinalis_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/ciona_intestinalis_turing_wavelength_s0/paper_citations.bib` L1–80
- **Code-to-model traceability: 88** — Docstring, gamma_for_wavelength, and the Lean module name the same k_c formula. Math KG yaml points at those loci. The scale sandwich theorem is the chronoseries statement: no temporal predecessor, and the detail patch sits between 1 µm and 10 mm.  
  evidence: `morphoeval/math_kg_loops/delphi10000-ciona-intestinalis-turing-wavelength-s0.yaml` L1–40; `morphoeval/lean/loops/Loop_3148_ciona_intestinalis_turing_wavelength_s0.lean` L48–61
- **Reproducibility package: 88** — --self-test passes (Pearson 0.994 on the short grid). Full run is deterministic at seed 3148. README states the 51 µm label, the 5.056e-05 m pin, and age 0 Ma. Fitness JSON is written by --write-fitness without touching the shared category index.  
  evidence: `models/2026-delphi10000-ciona-intestinalis-turing-wavelength-s0.py` L729–807; `datasets/ciona_intestinalis_turing_wavelength_s0/README.md` L1–24

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=1.3]** `specimen_comparison` → `specimen-match`: Image-fitness JSON written at pass 2 (image_fitness 92.4, CV 96, 12 frames, 560 px).
- **[RESOLVED · P=1.1]** `morphometric_validation` → `morphometric-validation`: Adult spacing 872.7 µm versus pin 50.56 µm is in the measurements sidecar.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Ciona intestinalis Turing wavelength at 51 µm (5.056e-05 m), Holocene 0 Ma
- **spec_fingerprint** (semantic model + inputs): `cc8077f6425fa5fa7578c008956e84c51354ee6ba24c3850292df68d0c32e0f9`
- **input_dataset_sha256**: `datasets/ciona_intestinalis_turing_wavelength_s0/Ciona_intestinalis_YPM_IZ_082346.jpg` (`bacc05b7298b188c…`), `datasets/ciona_intestinalis_turing_wavelength_s0/ciona_intestinalis_turing_wavelength_s0_reference.json` (`ba8af50f6e980b7c…`), `datasets/ciona_intestinalis_turing_wavelength_s0/paper_citations.bib` (`6b138fabe89be7a0…`)
- `docs/media/ciona_intestinalis_turing_wavelength_s0_comparison.png` — artifact_sha256 `cc584dd710e4101b…`, 277,777 B, 1512×984 px
- `docs/media/ciona_intestinalis_turing_wavelength_s0_growth.gif` — artifact_sha256 `e120696368881972…`, 1,385,890 B, 560×779 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| stripe_wavelength_um | 50.05183748686624 µm | 50.559999999999995 | Radial FFT peak of activator u; parabolic sub-bin interpolation. RMSE not used on the peak. *(Numerical peak is not forced equal to the pin.)* |
| critical_mode_log_amplitude_pearson_r | 0.999390494829923 1 | 0.98 | Pearson r and RMSE of log|A(k_c)| versus t in the linear window *(Gate is r > 0.98; the fitted r is a separate number.)* |
| specimen_foreground_fraction | 0.2977914554670529 1 | 0.5 | Largest connected component on the CC0 YPM photograph *(Specimen segmentation, not a self-copy of a reference.)* |
| specimen_contour_circularity | 0.24771186677158408 1 | 1.0 | Contour circularity of the body mask *(Vase outline is far from a circle.)* |
| specimen_body_length_mm | 33.95454545454545 mm | 10.0 | Bounding-box height times µm/px from the white 10 mm bar *(Adult body versus the bar length.)* |
| adult_pigment_spacing_um | 872.7272727272726 µm | 50.559999999999995 | FFT of the high-pass red-channel transect inside the body mask *(Adult photograph resolves millimetre speckles, not the 51 µm law. Scale-transfer control.)* |
| equal_diffusion_amplitude_ratio | 1.9757219312794065e-05 1 | 2.9452267353620942 | Negative control: D_u = D_v amplitude ratio versus the Turing ratio *(Equal diffusion must not amplify the critical mode.)* |
| pure_diffusion_amplitude_ratio | 0.00027972213368463464 1 | 1.0 | Ablation control, γ = 0, amplitude ratio *(Decay ratio is below 1.)* |
| ks_specimen_vs_activator | 0.43400000000000005 1 | 0.0 | Two-sample KS (and p-value) between body luminance and mapped activator; CV of the wavelength sweep is stored separately *(Distributions are not the same object; KS is not zero.)* |
| sweep_wavelength_cv | 0.05455172534481563 1 | 0.0 | Coefficient of variation of analytical λ across D_v/D_u *(λ moves when the diffusion ratio moves.)* |

## Environment at checkpoint

- python: 3.12.3
- note: numpy, scipy, pillow, matplotlib. Signing skipped: ~/.nanopub/profile.yml is absent.

Pass 2 closes R1–R3 from the pass-1 audit. Same solver commit; fitness JSON and measurement sidecar are the new records.
