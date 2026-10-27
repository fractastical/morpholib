# MorphoEval report — audit `2026-10-27-dasypus-novemcinctus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 4.108 µm on Holocene Dasypus novemcinctus dermis (Pass 1) (`models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py`) | 86 | 76 | 58 | 64 | 90 | 82 | 84 | 76.7 | 71.5 | 85.5 | 77.2 | **77.2** | mostly faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 4.108 µm on Holocene Dasypus novemcinctus dermis (Pass 1)  —  overall 77.2 (mostly faithful)

Pass 1 overall 77.2/100. The script implements Schnakenberg kinetics and the Turing wavenumber identity at 4.108 µm (0 Ma, Holocene, chronoseries length 1) and renders a 12-frame 640×480 GIF next to a CC BY-SA specimen photograph. The saturated lattice measures 4.56 µm, longer than the linear target. No sweep, ablation, specimen segmentation, or goodness-of-fit is present yet.

- **Model fidelity to source: 86** — Fourier-split Schnakenberg integration and the critical-wavenumber formula are implemented as published. Diffusivities are reconstructed to hit 4.108 µm rather than taken from a Dasypus experiment, so this is not a direct paper reproduction.  
  evidence: `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L77–124; `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L126–165; `morphoeval/lean/loops/Loop_3166_dasypus_novemcinctus_turing_wavelength_s0.lean` L20–54
- **Morphological realism / specimen match: 76** — The comparison figure places the real La Fortuna photograph beside the activator and an overlay. The GIF modulates that carapace with the growing spot field. There is no segmented osteoderm measurement yet, and the photo is not a 4.1 µm micrograph.  
  evidence: `docs/media/dasypus_novemcinctus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/dasypus_novemcinctus_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L213–261
- **Parameter expressivity: 58** — a, b, and the diffusion ratio are named constants and the dispersion identity is checked in --self-test. No range is exercised, so monotonic control of wavelength is not demonstrated.  
  evidence: `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L42–55; `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L361–390
- **Morphometric validation: 64** — The saturated wavelength (4.56 µm) is reported against the linear target 4.108 µm, and σ(k_c) is positive. Pearson r, RMSE, KS, and a specimen segmentation are absent, so the dimension cannot score higher on this pass.  
  evidence: `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L167–199; `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L322–359
- **Source and data traceability: 90** — Chen, Vickaryous, Turing, Schnakenberg, and Murray are cited with real DOIs. The specimen SHA-256 and Wikimedia creator, licence, and locality are recorded.  
  evidence: `datasets/dasypus_novemcinctus_turing_wavelength_s0/paper_citations.bib` L1–68; `datasets/dasypus_novemcinctus_turing_wavelength_s0/dasypus_novemcinctus_turing_wavelength_s0_reference.json` L1–70
- **Code-to-model traceability: 82** — Equations 1-3 are commented on the coefficient function and the Lean file checks the integer form of the same Jacobian and packing count.  
  evidence: `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L77–114; `morphoeval/math_kg_loops/delphi10000-dasypus-novemcinctus-turing-wavelength-s0.yaml` L12–48
- **Reproducibility package: 84** — Seed 3166, --self-test, and a pinned specimen digest. The run is deterministic. Pass 1 does not yet expose a sweep CLI or record multi-seed dispersion.  
  evidence: `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L361–430

Remediation:
- {'id': 'R1', 'item': 'Parameter-regime sweep: vary the diffusion scale and show that measured wavelength tracks the analytic square-root law.'}
- {'id': 'R2', 'item': 'Negative control: set D_u = D_v and show that the activator amplitude collapses.'}
- {'id': 'R3', 'item': 'Segment the real specimen photograph (foreground, contour circularity, structure-tensor coherency, movable-band count, osteoderm spacing).'}
- {'id': 'R4', 'item': 'Goodness-of-fit on the sweep: Pearson r, RMSE, a KS statistic, and the coefficient of variation across seeds.'}
- {'id': 'R5', 'item': 'Independent second assessor. This pass is single-assessor.'}
- {'id': 'R6', 'item': 'Decompose the 4.56 µm versus 4.108 µm red-shift (finite amplitude versus the linear dispersion curve).'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 4.108 µm on Holocene Dasypus novemcinctus dermis (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `0ed1ba81083be6f146345ad2f0ef47d7ab333edbbcd8e4d2c467f1b152acb7c7`
- **input_dataset_sha256**: `datasets/dasypus_novemcinctus_turing_wavelength_s0/dasypus_novemcinctus_specimen.jpg` (`7f1ba0e3ab8bdb13…`), `datasets/dasypus_novemcinctus_turing_wavelength_s0/dasypus_novemcinctus_turing_wavelength_s0_reference.json` (`e8ce5113e12fa07c…`)
- `docs/media/dasypus_novemcinctus_turing_wavelength_s0_comparison.png` — artifact_sha256 `57b258b8330bd9d0…`, 850,647 B, 1512×528 px
- `docs/media/dasypus_novemcinctus_turing_wavelength_s0_growth.gif` — artifact_sha256 `41326f640f21a2ef…`, 2,752,116 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| saturated_activator_wavelength_um | 4.5617 µm | 4.108 | Parabolic peak of the radial Fourier power of the saturated activator *(Finite-amplitude spots sit slightly long of the linear k_c. value != reference.)* |
| linear_growth_rate_at_kc | 0.2185 1/τ | 0.0 | Larger eigenvalue of J - k_c^2 D |
| specimen_file_bytes | 440984 byte | 0 | Vendored CC BY-SA photograph byte length |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 working tree. One Schnakenberg integration, analytic dispersion identity, 12-frame GIF.
