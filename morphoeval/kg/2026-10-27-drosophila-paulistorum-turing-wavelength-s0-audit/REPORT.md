# MorphoEval report — audit `2026-10-27-drosophila-paulistorum-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `9a9d6408` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila paulistorum Schnakenberg ocellar stripe at 2.4 µm (Pass 1) (`models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py`) | 82 | 80 | 70 | 68 | 88 | 84 | 86 | 78.0 | 75.5 | 86.1 | 79.4 | **79.4** | mostly faithful |

## Findings and remediation

### Drosophila paulistorum Schnakenberg ocellar stripe at 2.4 µm (Pass 1)  —  overall 79.4 (mostly faithful)

Pass 1 implements periodic Schnakenberg kinetics (a=0.10, b=0.90, γ=8, Dv/Du=36) on a 14.502 µm patch, segments the CC BY-SA Haars 2018 male, and renders a 12-frame 640×480 growth GIF. The classical determinant wavenumber was pinned at 2.417 µm, but the maximum of σ(k) sat near 3.5 µm. The nonlinear Du sweep therefore correlated only at r=0.877 (RMSE 1.18 µm). Image CV against the Haars frame was 88.5.

- **Model fidelity to source: 82** — The Jacobian, periodic Laplacian, and Schnakenberg update match the cited kinetics, and the homogeneous state has tr=-0.2 and det=1. The emergent stripe was not yet the 2.417 µm maximum-growth mode, so the run does not get full credit for the law at this rung.  
  evidence: `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L98–176; `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L190–248; `morphoeval/lean/loops/Loop_3197_drosophila_paulistorum_turing_wavelength_s0.lean` L17–56
- **Morphological realism / specimen match: 80** — The Haars 2018 male is vendored and SHA-256 pinned, and the 12-frame GIF's last frame scored CV 88.5 against that photograph. The raw activator panel was still the wrong wavelength, so the biological match of the stripe itself is incomplete.  
  evidence: `datasets/drosophila_paulistorum_turing_wavelength_s0/drosophila_paulistorum_male_haars_2018.png` L0–0; `docs/media/drosophila_paulistorum_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L406–432
- **Parameter expressivity: 70** — A Du sweep and an equal-diffusion ablation were executed. Activator amplitude collapsed under d=1, but measured wavelength did not move cleanly with Du (nonlinear r=0.877), so the sweep did not yet demonstrate control of λ.  
  evidence: `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L332–404; `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L919–966
- **Morphometric validation: 68** — Specimen foreground 0.300 (prior 0.22), coherency 0.632 (prior 0.45), and circularity 0.597 (prior 0.72) are non-circular. The wavelength goodness-of-fit on the nonlinear field failed the 0.98 gate (r=0.877, RMSE 1.18 µm). Midline spacing was already near 2.40 µm (KS 0.50 against a 4% band).  
  evidence: `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L434–473; `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L578–659
- **Source and data traceability: 88** — Dobzhansky & Spassky 1959, Turing 1952, Schnakenberg 1979, and Maini et al. 1997 are cited with DOIs. The Haars file URL, license, and SHA-256 are in the reference JSON. The 2.417 µm length is identified as the campaign rung, not a 1959 measurement.  
  evidence: `datasets/drosophila_paulistorum_turing_wavelength_s0/paper_citations.bib` L1–55; `datasets/drosophila_paulistorum_turing_wavelength_s0/drosophila_paulistorum_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — Integrator, Jacobian, and Lean declarations are mapped in the math-kg loop file. The Pass 1 dispersion curve annotated the determinant wavenumber rather than the wavenumber the field selected.  
  evidence: `morphoeval/math_kg_loops/delphi10000-drosophila-paulistorum-turing-wavelength-s0.yaml` L1–80; `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L1–30
- **Reproducibility package: 86** — SHA-256 pins, seed 3197, --self-test, and a deterministic explicit scheme are present. Self-test passed (dispersion r=0.999 on the linear grid, ablation std near 0). The nonlinear wavelength failure is scored under morphometric-validation, not hidden here.  
  evidence: `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L70–84; `models/2026-delphi10000-drosophila-paulistorum-turing-wavelength-s0.py` L919–966

Remediation:
- The nonlinear activator FFT did not track the stated wavelength (Pearson r=0.877, RMSE 1.18 µm) because σ(k) peaked near 3.5 µm while the determinant formula had been pinned at 2.417 µm, and broadband noise did not lock to one mode.
- Rescale both diffusivities so argmax σ(k) is 2.417 µm, seed each Du-sweep point with that integer mode, and remeasure λ from the activator field until Pearson r > 0.98.
- Re-render the growth GIF from the corrected field and re-score image fitness.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila paulistorum Schnakenberg ocellar stripe at 2.4 µm (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `59419ebb8a88efc4005d1cc11aab51778bad8613c26ac834a4d86676d99742f1`
- **input_dataset_sha256**: `datasets/drosophila_paulistorum_turing_wavelength_s0/README.md` (`b770bcff5173b0a0…`), `datasets/drosophila_paulistorum_turing_wavelength_s0/drosophila_paulistorum_male_haars_2018.png` (`e01e60e633f99ef1…`), `datasets/drosophila_paulistorum_turing_wavelength_s0/drosophila_paulistorum_turing_wavelength_s0_reference.json` (`39b5153e4682ffee…`), `datasets/drosophila_paulistorum_turing_wavelength_s0/paper_citations.bib` (`5c00ca3d9a631902…`), `datasets/drosophila_paulistorum_turing_wavelength_s0/scout_manifest.json` (`a859ae40d3f79e52…`)
- `datasets/drosophila_paulistorum_turing_wavelength_s0/drosophila_paulistorum_male_haars_2018.png` — artifact_sha256 `e01e60e633f99ef1…`, 981,339 B, 800×800 px
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

Pass 1 of loop 2026-10-27-drosophila-paulistorum-turing-wavelength-s0-loop. The cited commit contains the model, the Haars specimen, and the Lean module. Nanopub signing was not available (~/.nanopub/profile.yml absent). publisher_orcid is the repository URI, not an ORCID iD, so the unsigned graph can be emitted.
