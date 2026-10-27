# MorphoEval report — audit `2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `9a9d6408` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit -> 2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila paulistorum Schnakenberg ocellar stripe at 2.4 µm (Pass 2) (`models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py`) | 91 | 84 | 90 | 90 | 90 | 88 | 88 | 90.7 | 86.2 | 88.7 | 88.3 | **88.3** | faithful |

## Findings and remediation

### Drosophila paulistorum Schnakenberg ocellar stripe at 2.4 µm (Pass 2)  —  overall 88.3 (faithful)

Pass 2 multiplies both diffusivities by 0.478965 so argmax σ(k) lies at 2.417 µm (σ=3.13). A 9-point Du sweep, each point seeded with its integer maximum-growth mode, recovers that wavelength from the activator FFT at Pearson r=0.989 (RMSE 0.064 µm). The nominal patch (6 waves, 14.502 µm) measures 2.417 µm. Equal diffusion (d=1) drops the activator standard deviation by a factor of about 1e-5. The Haars 2018 male is segmented (foreground 0.300, coherency 0.632, circularity 0.597). The growth GIF is 12 frames at 640×480; last-frame CV similarity against the photograph is 85.9. Image fitness using the Pass 2 subscores is 85.9. Versus the catalog predecessor D. orena (89.44 µm) the detail mode packs 37 times (remainder 11 nm); versus D. persimilis (116.7 µm) it packs 48 times (remainder 684 nm). All three ages are 0 Ma.

- **Model fidelity to source: 91** — Schnakenberg kinetics, the periodic 5-point Laplacian, and the linear eigenvalue are implemented directly. D_MAGNITUDE_SCALE = 0.478965 places argmax σ(k) at 2.416 µm (gate ±0.02 µm) with σ(k_c)=3.13. Trace -0.2 and determinant 1 match the Lean theorems. Equal diffusion is linearly stable and numerically decays.  
  evidence: `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L98–150; `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L164–248; `morphoeval/lean/loops/Loop_3197_drosophila_paulistorum_turing_wavelength_s0.lean` L17–56
- **Morphological realism / specimen match: 84** — The comparison figure shows the Haars male beside the 2.417 µm activator stripes and a basic-look frame in which those stripes are impressed on the photograph at α=0.18. Last-frame CV similarity is 85.9 (12-bin orientation, 24-bin radial FFT, structure-tensor coherency, occupancy) on a 640×480, 12-frame GIF. The score stays below 90 because the raw field is a subcellular stripe lattice, not a second photograph of the fly, and no ocellar TEM of this species is in the dataset.  
  evidence: `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L406–432; `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L698–735; `docs/media/drosophila_paulistorum_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_paulistorum_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 90** — Du scales (0.35, 0.50, 0.70, 0.85, 1, 1.25, 1.55, 1.90, 2.40) move the maximum-growth wavelength monotonically from 1.80 µm to 3.12 µm. The nonlinear field follows that series (r=0.989). Setting Du=Dv abolishes the pattern (amplitude ratio 1.1e-5). --self-test checks the same direction of control.  
  evidence: `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L279–370; `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L372–404
- **Morphometric validation: 90** — Nonlinear FFT versus maximum-growth wavelength: Pearson r=0.989, RMSE=0.064 µm. Nominal refined wavelength 2.417 µm on the 14.502 µm patch. Specimen foreground 0.2995 versus prior 0.22, coherency 0.632 versus prior 0.45, circularity 0.597 versus prior 0.72 (none equal to its reference). Midline peak spacing averages 2.402 µm; two-sample KS against a 4% band around 2.417 µm is 0.50. Orena packing 37 and persimilis packing 48 differ from the decade priors 30 and 40.  
  evidence: `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L250–277; `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L434–473; `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L578–659
- **Source and data traceability: 90** — DOIs resolve for Dobzhansky & Spassky 1959 (OpenAlex W2082921328, OA PMC), Turing 1952, Schnakenberg 1979, and Maini et al. 1997. The specimen record stores the Commons URL, CC BY-SA 4.0, artist, date, and SHA-256 e01e60e633f99ef12a7474a79f5afaaa56b1502bc76590c4102723cf5cf7d1ae. Scale 2.4 µm and age 0 Ma are stated in the README and the reference JSON.  
  evidence: `datasets/drosophila_paulistorum_turing_wavelength_s0/scout_manifest.json` L1–40; `datasets/drosophila_paulistorum_turing_wavelength_s0/drosophila_paulistorum_turing_wavelength_s0_reference.json` L1–50; `datasets/drosophila_paulistorum_turing_wavelength_s0/README.md` L1–30
- **Code-to-model traceability: 88** — Each modelling step is a named function cited from the math-kg equations: jacobian_unscaled, diffusivities, sigma_peak_wavelength, integrate, nonlinear_sweep, ablation_control, segment_specimen. Lean theorem names match the math-kg lean_theorem fields.  
  evidence: `morphoeval/math_kg_loops/delphi10000-drosophila-paulistorum-turing-wavelength-s0.yaml` L40–90; `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L475–508
- **Reproducibility package: 88** — Dataset SHA-256 pins are checked on every run. Seed 3197. --self-test passed: growth-rate peak within 0.02 µm of 2.417, seeded mode amplifies (std 0.95), equal diffusion decays (std 0.0002), dispersion Pearson r=0.99972, packing identities 37×11 nm and 48×684 nm. Pinned environment is recorded above. Signing keys were not available; that limit is stated rather than bypassed.  
  evidence: `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L70–96; `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L919–966

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila paulistorum Schnakenberg ocellar stripe at 2.4 µm (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `59419ebb8a88efc4005d1cc11aab51778bad8613c26ac834a4d86676d99742f1`
- **input_dataset_sha256**: `datasets/drosophila_paulistorum_turing_wavelength_s0/README.md` (`b770bcff5173b0a0…`), `datasets/drosophila_paulistorum_turing_wavelength_s0/drosophila_paulistorum_male_haars_2018.png` (`e01e60e633f99ef1…`), `datasets/drosophila_paulistorum_turing_wavelength_s0/drosophila_paulistorum_turing_wavelength_s0_reference.json` (`39b5153e4682ffee…`), `datasets/drosophila_paulistorum_turing_wavelength_s0/paper_citations.bib` (`5c00ca3d9a631902…`), `datasets/drosophila_paulistorum_turing_wavelength_s0/scout_manifest.json` (`a859ae40d3f79e52…`)
- `docs/media/drosophila_paulistorum_turing_wavelength_s0_comparison.png` — artifact_sha256 `26d4992383a6d270…`, 435,973 B, 1740×912 px
- `docs/media/drosophila_paulistorum_turing_wavelength_s0_growth.gif` — artifact_sha256 `ac4e45110afd8d24…`, 2,776,691 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| ocellar_stripe_wavelength_um | 2.41732 µm | 2.417 | Parabolic refinement of the radial FFT peak of the Schnakenberg activator on a 14.502 µm periodic patch |
| dispersion_wavelength_pearson_r | 0.98908 1 | 0.98 | Pearson r of wavelength series (nonlinear_field_vs_continuum); reference is the pass gate, not the measured r |
| wavelength_rmse_um | 0.06438 µm | 0.15 | RMSE between continuum λ_c(Du) and the compared wavelength series |
| specimen_foreground_fraction | 0.2995 1 | 0.22 | Border-median contrast mask on the Haars 2018 male, 160×120; reference is a tight head-crop prior |
| specimen_structure_coherency | 0.6323 1 | 0.45 | Mean structure-tensor coherency inside the specimen mask; reference is a moderate-anisotropy prior |
| specimen_contour_circularity | 0.5968 1 | 0.72 | 4πA/P² of the habitus mask; reference is a circular ocellus prior, not this lateral habitus |
| interpeak_spacing_ks_statistic | 0.5 1 | 0.45 | Two-sample KS between midline activator spacings and a 4% band around 2.417 µm |
| ablation_to_wildtype_amplitude_ratio | 1e-05 1 | 1.0 | Equal-diffusion (d=1) activator std divided by wild-type (d=36) std; reference 1 would mean the ablation failed |
| detail_wavelengths_inside_orena_cell | 37 1 | 30 | floor(89440 nm / 2417 nm); reference 30 is the round decade, not the packing count |
| detail_wavelengths_inside_persimilis_primordium | 48 1 | 40 | floor(116700 nm / 2417 nm); reference 40 is not the packing count |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2 of loop 2026-10-27-drosophila-paulistorum-turing-wavelength-s0-loop. Nonlinear field Pearson r=0.989, RMSE=0.064 µm, CV similarity=85.9 on a 12-frame 640×480 GIF. ~/.nanopub/profile.yml is absent, so nanopubs are unsigned. publisher_orcid is the repository URI, not an ORCID iD.
