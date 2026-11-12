# MorphoEval report — audit `2026-10-27-gasterosteus-doryssus-mid-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-gasterosteus-doryssus-mid-turing-wavelength-s0-audit -> 2026-10-27-gasterosteus-doryssus-mid-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 42 µm on Gasterosteus doryssus_mid (Truckee Fm, 10.01 Ma) (`models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py`) | 88 | 82 | 90 | 88 | 91 | 86 | 88 | 88.7 | 84.2 | 88.5 | 86.8 | **86.8** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 42 µm on Gasterosteus doryssus_mid (Truckee Fm, 10.01 Ma)  —  overall 86.8 (faithful)

Pass 2 closes the three Pass 1 remediations. Du scaled by 0.9 and 1.1 moves λ from 39.80 µm to 44.02 µm around the calibrated 41.94 µm. The unguided activator GIF scores CV 58.1; the specimen-registered growth GIF (12 frames, 640×480) scores CV 92.3. The darkest 40% of the plate (bone) has radial KS 0.333 against the render; the sediment complement has KS 0.542. Dispersion Pearson r remains 0.9897, RMSE 0.070. Early/mid/late wavelengths remain 32.14 / 41.94 / 56.90 µm at 10.02 / 10.01 / 9.99 Ma. Overall 86.8/100. Nanopubs are unsigned: ~/.nanopub/profile.yml is absent.

- **Model fidelity to source: 88** — The Pass 1 linearisation is unchanged and still matches the Schnakenberg Jacobian, with k_c defined as argmax σ(k) rather than the large-ratio algebraic approximation (those two differ by about two pixels here, and the code uses the exact argmax). Pass 2 adds du_sensitivity, pure_field_cv and bone_mask_spectra without changing the PDE. The amplitude clip at 8 and the fossil composite remain render/numerics choices, so this is not a 100. Four Lean theorems still have zero sorry.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L97–140; `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L435–493; `morphoeval/lean/loops/Loop_3230_gasterosteus_doryssus_mid_turing_wavelength_s0.lean` L18–44
- **Morphological realism / specimen match: 82** — Registered GIF versus the CC BY Voje, Bell & Stuart 2022 plate: CV similarity 92.3, 12 frames, 640×480. The same fitness function on the unguided stripe field, now stored as pure_field_cv, is 58.1. The comparison PNG keeps the pure stripes visible next to the fossil and the equal-diffusion control. The score stays below the CV number because the plate is a centimetre skeleton and the 42 µm claim is a modulation of that plate, which the code states in composite_on_specimen.  
  evidence: `docs/media/gasterosteus_doryssus_mid_turing_wavelength_s0_growth.gif` L0–0; `docs/media/gasterosteus_doryssus_mid_turing_wavelength_s0_comparison.png` L0–0; `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/gasterosteus_doryssus_mid_turing_wavelength_s0_metrics.json` L0–0
- **Parameter expressivity: 90** — Ratio sweep at γ = 1: λ_pix rises 12.91 → 15.73 from Dv/Du = 12 to 32. Chronoseries, which also drops γ: 32.14 → 41.94 → 56.90 µm. New Du band: 0.9 Du gives 39.80 µm, 1.0 Du gives 41.94 µm, 1.1 Du gives 44.02 µm, monotone, as λ ∝ sqrt(Du) near the calibrated mode. The self-test checks that order.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L435–448; `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L159–171; `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/gasterosteus_doryssus_mid_turing_wavelength_s0_metrics.json` L0–0
- **Morphometric validation: 88** — Dispersion: Pearson r = 0.9897, RMSE = 0.070 against the eigenvalue. Equal-diffusion σ_max = -0.324. Registered-frame coherency 0.603 versus specimen 0.596; foreground 0.821 versus 0.589. Whole-frame radial KS D = 0.333. Bone mask (darkest 40%, fraction 0.400) radial KS = 0.333; sediment complement KS = 0.542, so the two regions are not the same comparison. Gray-level CV ratio sim/specimen = 0.948. None of these value/reference pairs are copies.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L173–206; `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L473–493; `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/gasterosteus_doryssus_mid_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 91** — Unchanged sources from Pass 1, still SHA-pinned. The metrics sidecar now records the pure-field CV and the bone-mask split, so a reader can see which number came from the plate and which from the PDE. Crossref DOIs and the Commons CC BY 4.0 file are the same as Pass 1. Shapiro and Chan author lists remain fuller in the BibTeX than in the short audit reference labels.  
  evidence: `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/paper_citations.bib` L1–74; `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/gasterosteus_doryssus_mid_turing_wavelength_s0_reference.json` L48–56; `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L89–95
- **Code-to-model traceability: 86** — Pass 2 functions are named for the remediations they close: du_sensitivity, pure_field_cv, bone_mask_spectra. Each is called from run_pipeline and asserted in self_test. The docstring claim boundary still matches does_not_claim in this audit, including the sentence that the unguided field is not the skeleton.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L1–52; `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L697–726
- **Reproducibility package: 88** — --self-test passed after the Pass 2 additions: r = 0.98974, wavelengths 32.14/41.94/56.90 µm, control σ = -0.324, Du band monotone, pure CV 58.1 below registered CV 92.3, GIF 12 frames at 640×480. Seed 3230. No second machine, and signing was not possible without a nanopub profile.  
  evidence: `models/2026-delphi10000-gasterosteus-doryssus-mid-turing-wavelength-s0.py` L697–726; `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/README.md` L1–40

Remediation:
- {'id': 'R4', 'item': 'Residual, not a failed numerical gate: no second assessor, and ~/.nanopub/profile.yml is absent so the nanopubs stay unsigned.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 42 µm on Gasterosteus doryssus_mid (Truckee Fm, 10.01 Ma)
- **spec_fingerprint** (semantic model + inputs): `dc4c1601bb00a6d80d4baed16fe77c8243511e825b49d7d785e454e5f8378e0b`
- **input_dataset_sha256**: `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/gasterosteus_doryssus_mid_turing_wavelength_s0_reference.json` (`38c0a5455a7e482a…`), `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/gasterosteus_doryssus_mid_turing_wavelength_s0_specimen.jpg` (`a1d587e4264868ea…`)
- `datasets/gasterosteus_doryssus_mid_turing_wavelength_s0/gasterosteus_doryssus_mid_turing_wavelength_s0_metrics.json` — artifact_sha256 `d47d3627a1ab343c…`, 10,227 B, 
- `docs/media/gasterosteus_doryssus_mid_turing_wavelength_s0_comparison.png` — artifact_sha256 `3a2189b6a31a4953…`, 428,769 B, 1512×984 px
- `docs/media/gasterosteus_doryssus_mid_turing_wavelength_s0_growth.gif` — artifact_sha256 `101ffda1755cf4e1…`, 2,323,139 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| mid-stage critical wavelength | 41.94 µm | 42.0 | argmax_k sigma(k) of the Schnakenberg Jacobian, scaled so the mid mode is the assigned 42 µm *(pixel wavelength 13.9946; sigma_max 0.1893)* |
| early-to-late wavelength ratio | 1.7704 λ_late / λ_early | 1.0 | same Du, stage-specific (Dv/Du, γ) at T = 0 and T = +22 kyr *(early 32.139 µm, mid 41.940 µm, late 56.898 µm)* |
| dispersion Pearson r analytic vs RK2 | 0.9897 Pearson r | 1.0 | 16 wavenumbers; closed-form σ(k) versus RK2 integration of the linear mode ODE *(RMSE 0.070349)* |
| dispersion RMSE | 0.070349 1/time | 0.0 | sqrt mean square gap between analytic and RK2 growth rates *(rmse)* |
| equal-diffusion control sigma_max | -0.3236 1/time | 0.0 | fastest_mode with Dv/Du = 1 at γ = 1 *(negative control; pattern must decay)* |
| ratio-sweep wavelength slope | 2.8229 pixels | 0.0 | λ(ratio=32) − λ(ratio=12) at γ = 1, unstable band only *(positive slope: inhibitor range lengthens the flank wavelength)* |
| specimen structure-tensor coherency | 0.5957 coherency | 0.6031 | verify_image_fitness.compute_specimen_cv_signature on the CC BY fossil plate *(specimen vs simulation; value is not copied from the reference)* |
| specimen foreground fraction | 0.5894 area fraction | 0.8207 | border-contrast mask on the Voje, Bell & Stuart 2022 plate *(non-circular: fossil sediment mask versus the filled flank field)* |
| radial-spectrum KS statistic specimen vs simulation | 0.3333 KS D | 0.0 | scipy.stats.ks_2samp on 24-bin radial FFT profiles *(orientation-histogram Pearson r = 0.6685)* |
| simulation gray-level CV versus specimen | 0.9481 std_sim / std_specimen | 1.0 | grayscale standard-deviation ratio after 160×120 resize *(cv of luminance, not a copied scalar)* |
| Du plus-or-minus 10 percent wavelength span | 4.2278 µm | 0.0 | λ(1.1 Du) − λ(0.9 Du) at the mid-stage ratio and gamma, nominal dx held fixed *(λ(0.9 Du) = 39.795 µm, λ(Du) = 41.940 µm, λ(1.1 Du) = 44.023 µm)* |
| unguided pure-field CV similarity | 58.1 CV similarity | 92.3 | verify_image_fitness on a 12-frame GIF of the activator alone, no fossil composite *(the pure stripe field is not the skeleton; this number is kept separate from the registered render)* |
| bone-mask radial KS versus rendered frame | 0.3333 KS D | 0.5417 | darkest 40 percent of the plate (bone) versus the registered frame, 24-bin radial FFT *(bone area fraction 0.400)* |

## Environment at checkpoint


Pass 2 files are new on top of 0c76d15d. Built with --no-verify-loci.
