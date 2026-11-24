# MorphoEval report — audit `2026-10-27-daphnia-magna-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `86cc0374f85f7f6fea93d271291a5192c8f44eb3` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Daphnia magna ocellus-pigment Turing wavelength at 2.736 µm (Holocene, detail scale) (`models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py`) | 88 | 74 | 86 | 88 | 91 | 84 | 86 | 87.3 | 79.2 | 87.1 | 84.0 | **84.0** | faithful |

## Findings and remediation

### Daphnia magna ocellus-pigment Turing wavelength at 2.736 µm (Holocene, detail scale)  —  overall 84.0 (faithful)

Schnakenberg reaction-diffusion on a periodic patch whose length unit puts the analytical critical wavelength at 2.736 µm (log10 = -5.563), the Holocene detail tier for Daphnia magna. The chronoseries is 1/1 at 0 Ma: there is no predecessor chronospecies and no successor. The solver's growth rates match σ(k) at Pearson r = 0.99999995 on the band around k_c (RMSE 0.00013). Raising D_u from 0.008 to 0.036 lengthens λ_c by a factor 2.21. Setting D_v = D_u drops the activator standard deviation to 0.286 of its initial value and the analytical max σ is negative. The CC BY-SA head SEM (Paulo Corgosinho, 2016) has a 100 µm bar measured at 156 px. Its cuticle mesh has an interior spectral peak at 20.17 µm, about seven times the model wavelength, so the polygons are not the Turing field. The saturated pattern's dominant wavelength is 3.72 µm, a nonlinear shift that Pass 1 does not yet treat as a held-out claim. The growth GIF has 12 frames at 720×540; its last frame is the SEM, which is the registration image for the fitness test.

- **Model fidelity to source: 88** — The script integrates Schnakenberg kinetics with a spectral Laplacian and compares growth rates to the larger eigenvalue of J − k²D. The length unit is imposed so λ_c equals the assigned 2.736 µm; it is not fitted to a published stripe period. The strongly damped mode is excluded from the scored band after a burn-in, which is disclosed.  
  evidence: `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L87–141; `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L155–162
- **Morphological realism / specimen match: 74** — The comparison figure places the Corgosinho head SEM beside the Otsu mask and the activator. The GIF's last frame is that SEM, so the registration image is the animal. The activator is a subcellular field, and the cuticle peak at 20.17 µm shows the polygons are a coarser structure. Specimen-match is the anatomical registration plus that disclosed mismatch, not a claim that the spots reproduce the mesh.  
  evidence: `docs/media/daphnia_magna_turing_wavelength_s0_comparison.png` L0–0; `docs/media/daphnia_magna_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 86** — D_u from 0.008 to 0.036, the term that dominates k_c² ≈ f_u/(2 D_u), lengthens λ_c by a factor 2.21 and the series is strictly increasing. The equal-diffusion ablation decays (amplitude ratio 0.286, max σ < 0).  
  evidence: `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L299–322; `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L274–297
- **Morphometric validation: 88** — Dispersion Pearson r = 0.99999995 against a reference of 1, RMSE = 0.000125 against 0. The SEM foreground, aspect ratio, coherency, and orientation KS differ from their references. The cuticle peak is 20.17 µm against the model wavelength 2.736 µm. The saturated dominant wavelength is 3.72 µm and is not yet given its own coarsening ratio. Outline circularity is 0.018 because the perimeter follows the antennule and carapace folds.  
  evidence: `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L171–229; `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L358–426; `datasets/daphnia_magna_turing_wavelength_s0/run_results.json` L0–0
- **Source and data traceability: 91** — Six Crossref-checked DOIs, a scout manifest, and a SHA-256-pinned CC BY-SA 4.0 SEM (Paulo Corgosinho, File:Daphnia magna head.jpg). An OpenAlex DOI that did not resolve at Crossref is stored and unused. The reference JSON states 2.736e-06 m and 0 Ma.  
  evidence: `datasets/daphnia_magna_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/daphnia_magna_turing_wavelength_s0/daphnia_magna_turing_wavelength_s0_reference.json` L1–40; `datasets/daphnia_magna_turing_wavelength_s0/scout_manifest.json` L1–30
- **Code-to-model traceability: 84** — The module docstring, the Jacobian, and the Lean witnesses (disc numerator 5350516939, margin numerator 22679272279285929721, D_v = 40 D_u, 500 < 2736 < 22000 nm) name the same rationals. The nonlinear 3.72 µm peak is plotted but not yet tied back to k_c by a ratio in the scored measurements' narrative beyond the raw comparison.  
  evidence: `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L1–20; `morphoeval/lean/loops/Loop_3165_daphnia_magna_turing_wavelength_s0.lean` L20–54
- **Reproducibility package: 86** — Pinned SHA-256 inputs, seed 3165, and a --self-test that requires r > 0.98, a monotone D_u sweep with ratio > 1.4, a decaying ablation, a 156 px scale bar, and a GIF of at least 10 frames and 480 px. The self-test passed. Signing is unavailable. One assessor scored the pass.  
  evidence: `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L46–70; `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L682–724; `datasets/daphnia_magna_turing_wavelength_s0/README.md` L1–32

Remediation:
- {'id': 'R1', 'item': 'Hold out the odd-indexed Fourier modes and require Pearson r > 0.98 on that withheld set, not only on the full scored band.'}
- {'id': 'R2', 'item': 'Report the saturated wavelength divided by the analytical 2.736 µm as a coarsening ratio, so 3.72 µm is read as a nonlinear shift rather than a second target.'}
- {'id': 'R3', 'item': 'Nanopub signing is blocked: ~/.nanopub/profile.yml is absent. Do not invent a signature.'}
- {'id': 'R4', 'item': 'State beside the circularity 0.018 that the perimeter follows the antennule and carapace folds, and keep the aspect ratio as the shape number.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Daphnia magna ocellus-pigment Turing wavelength at 2.736 µm (Holocene, detail scale)
- **spec_fingerprint** (semantic model + inputs): `5cc7a733d6eb0f2c8ca309d68942c0dd30e27b59c80a745c4958e462d76f9c31`
- `datasets/daphnia_magna_turing_wavelength_s0/run_results.json` — artifact_sha256 `bce0c6d82e18c990…`, 6,704 B, 
- `docs/media/daphnia_magna_turing_wavelength_s0_comparison.png` — artifact_sha256 `2ca968226229d471…`, 343,840 B, 1512×960 px
- `docs/media/daphnia_magna_turing_wavelength_s0_growth.gif` — artifact_sha256 `50d8f2ffb0e2ca69…`, 2,671,805 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion_pearson_r | 0.99999995 1 | 1.0 | Pearson r of solver growth rates against σ(k) on nine Fourier modes |
| dispersion_rmse | 0.000125 1/time | 0.0 | RMSE of the same nine-mode dispersion comparison |
| simulated_dominant_wavelength_um | 3.7212 µm | 2.736 | Radial Fourier peak of the saturated activator, scaled so analytical λ_c is 2.736 µm |
| specimen_foreground_fraction | 0.504 1 | 0.2636 | Otsu foreground of the Corgosinho head SEM versus the saturated field |
| specimen_contour_circularity | 0.018 1 | 1.0 | 4πA/P² of the SEM Otsu mask; a circle would be 1 |
| specimen_aspect_ratio | 1.4272 1 | 1.0 | Bounding-box width/height of the SEM foreground |
| specimen_structure_tensor_coherency | 0.5772 1 | 0.1878 | Mean structure-tensor coherency of the SEM versus the saturated activator |
| specimen_cuticle_reticulation_wavelength_um | 20.1658 µm | 2.736 | Interior local maximum of the cuticle radial spectrum, 100 µm = 156 px, 8–22 µm |
| specimen_vs_field_orientation_ks | 0.0925 1 | 0.0 | Two-sample KS statistic of Sobel orientation samples, SEM versus saturated field |
| ablation_amplitude_ratio | 0.286 1 | 1.0 | Final/initial activator std when D_v is set equal to D_u |
| sweep_lambda_coefficient_of_variation | 0.2297 1 | 0.0 | CV of λ_c across the D_v sweep; zero would mean the parameter does not move the wavelength |

## Environment at checkpoint

- python: 3.12
- numpy: 1.26.4
- note: scipy 1.18 and Pillow 13 installed for this run; no virtualenv

Pass 1. Sources committed before this audit. Nanopub profile is absent, so nothing is signed.
