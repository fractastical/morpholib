# MorphoEval report — audit `2026-10-27-daphnia-magna-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `5b213d4525d0e9f71d5fc4e385d270b87ac07ec8` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-daphnia-magna-turing-wavelength-s0-audit -> 2026-10-27-daphnia-magna-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Daphnia magna ocellus-pigment Turing wavelength at 2.736 µm (Holocene, detail scale) (`models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py`) | 90 | 76 | 88 | 92 | 92 | 88 | 88 | 89.3 | 82.0 | 89.3 | 86.4 | **86.4** | faithful |

## Findings and remediation

### Daphnia magna ocellus-pigment Turing wavelength at 2.736 µm (Holocene, detail scale)  —  overall 86.4 (faithful)

Pass 2 of the Holocene Daphnia magna detail-tier model. Analytical λ_c remains 2.736 µm at 0 Ma on a chronoseries of length 1. The full scored band still has Pearson r = 0.99999995 (RMSE 0.000125). The odd indexed modes, withheld from the fitting narrative, have Pearson r = 0.99999978 (RMSE 0.000123). The saturated peak is 3.7212 µm, a coarsening ratio of 1.3601 against a no-shift reference of 1. The D_u sweep still lengthens λ_c by 2.21, and equal diffusion still decays (ratio 0.286). The cuticle mesh peak stays at 20.17 µm. Outline circularity 0.018 is the antennule and carapace folds; aspect ratio 1.43 is the shape number. The GIF is 12 frames at 720×540 and its last frame is the CC BY-SA SEM. Signing is still blocked: there is no nanopub profile.

- **Model fidelity to source: 90** — Same Schnakenberg spectral solver as Pass 1, now checked on modes that were not the ones used to narrate the fit. Held-out r = 0.99999978. The length unit is still imposed at the assigned tier rather than fitted to a measured stripe.  
  evidence: `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L87–141; `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L211–229
- **Morphological realism / specimen match: 76** — The comparison figure now labels the saturated peak as coarsening ×1.36 beside the SEM and the Otsu mask. The GIF last frame is the Corgosinho SEM. The field is still not the 20.17 µm mesh, and the score stays below the registration-image similarity for that reason.  
  evidence: `docs/media/daphnia_magna_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L496–499
- **Parameter expressivity: 88** — D_u from 0.008 to 0.036 lengthens λ_c monotonically by a factor 2.21. Equal diffusion remains a negative control (amplitude ratio 0.286, max σ < 0).  
  evidence: `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L299–322; `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L274–297
- **Morphometric validation: 92** — Full-band r = 0.99999995 and held-out r = 0.99999978, both against a reference of 1. RMSE values are 0.000125 and 0.000123 against 0. Coarsening ratio 1.3601 against 1. Cuticle peak 20.17 µm against 2.736 µm. SEM foreground, coherency, aspect ratio, and orientation KS differ from their references. Circularity 0.018 is explained as the folded outline, with aspect ratio 1.43 carrying the shape.  
  evidence: `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L569–574; `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L690–699; `datasets/daphnia_magna_turing_wavelength_s0/run_results.json` L0–0
- **Source and data traceability: 92** — Unchanged provenance from Pass 1: Crossref DOIs, pinned CC BY-SA SEM, scout manifest, and an explicit statement that 2.736e-06 m and 0 Ma are the assigned tier on a 1/1 chronoseries. The OpenAlex DOI that failed Crossref stays unused.  
  evidence: `datasets/daphnia_magna_turing_wavelength_s0/paper_citations.bib` L1–62; `datasets/daphnia_magna_turing_wavelength_s0/daphnia_magna_turing_wavelength_s0_reference.json` L14–48
- **Code-to-model traceability: 88** — The coarsening ratio in the measurements is the same division drawn on the figure, and the holdout flag is the same branch the self-test asserts. Lean witnesses are unchanged and still match the rational numerators.  
  evidence: `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L68–70; `morphoeval/lean/loops/Loop_3165_daphnia_magna_turing_wavelength_s0.lean` L20–54
- **Reproducibility package: 88** — --self-test passed on the held-out correlation, the D_u ratio, the ablation, the 156 px scale bar, and the 12-frame 720×540 GIF. Inputs remain SHA-256 pinned. The package is still unsigned, and only one assessor scored it.  
  evidence: `models/2026-delphi10000-daphnia-magna-turing-wavelength-s0.py` L690–722; `datasets/daphnia_magna_turing_wavelength_s0/README.md` L1–32

Remediation:
- {'id': 'R3', 'item': 'Nanopub signing is still blocked. ~/.nanopub/profile.yml does not exist, and no signature was created. Pass 1 items R1 (held-out modes), R2 (coarsening ratio), and R4 (circularity wording) are closed.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Daphnia magna ocellus-pigment Turing wavelength at 2.736 µm (Holocene, detail scale)
- **spec_fingerprint** (semantic model + inputs): `5cc7a733d6eb0f2c8ca309d68942c0dd30e27b59c80a745c4958e462d76f9c31`
- `datasets/daphnia_magna_turing_wavelength_s0/run_results.json` — artifact_sha256 `fbc453a12d32bf1a…`, 7,081 B, 
- `docs/media/daphnia_magna_turing_wavelength_s0_comparison.png` — artifact_sha256 `9fac22d3c08e135d…`, 345,784 B, 1512×960 px

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion_pearson_r | 0.99999995 1 | 1.0 | Pearson r of solver growth rates against σ(k) on nine Fourier modes |
| dispersion_rmse | 0.000125 1/time | 0.0 | RMSE of the same nine-mode dispersion comparison |
| simulated_dominant_wavelength_um | 3.7212 µm | 2.736 | Radial Fourier peak of the saturated activator, scaled so analytical λ_c is 2.736 µm |
| nonlinear_coarsening_ratio | 1.3601 1 | 1.0 | Saturated wavelength divided by the analytical 2.736 µm. A ratio of 1 would mean no nonlinear shift. |
| specimen_foreground_fraction | 0.504 1 | 0.2636 | Otsu foreground of the Corgosinho head SEM versus the saturated field |
| specimen_contour_circularity | 0.018 1 | 1.0 | 4πA/P² of the opened SEM mask. Far below 1 because the perimeter follows the antennule and carapace folds; the aspect ratio is the shape number. |
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

Pass 2. Odd Fourier modes are held out. No nanopub profile, so the graph is unsigned. The ORCID is the repository publisher used by the other MorphoEval graphs.
