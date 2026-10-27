# MorphoEval report — audit `2026-10-27-euperipatoides-rowelli-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-euperipatoides-rowelli-turing-wavelength-s0-audit -> 2026-10-27-euperipatoides-rowelli-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg k_sigma at 1.12 µm in the E. rowelli ocellus (Holocene, 0 Ma) (`models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py`) | 88 | 84 | 90 | 88 | 90 | 86 | 88 | 88.7 | 85.5 | 88.1 | 87.2 | **87.2** | faithful |

## Findings and remediation

### Schnakenberg k_sigma at 1.12 µm in the E. rowelli ocellus (Holocene, 0 Ma)  —  overall 87.2 (faithful)

Pass 2 keeps the same Schnakenberg calibration (a=0.12, b=0.88, Du=1, Dv=22, γ=34122.25342711872, seed 3223, λ=1.12 µm, age 0 Ma). Dispersion Pearson r stays 0.9915 (RMSE 1171, KS 0.069). The saturated wavenumber is now a parabolic sub-bin DCT peak at 14.072 cycles, so λ_2d = 1.114 µm (0.5% under 1.12 µm; Pass 1's integer ring said 1.100 µm). In-band controls separate: full mode +8571, equal diffusion −5662, reversed ratio −22114. Specimen-versus-last-frame 12-bin orientation L1 is 0.0134, not zero. GIF unchanged at 12 frames and 640×480; CV similarity 96.0.

- **Model fidelity to source: 88** — The PDE, Jacobian, Neumann integrator, and dispersion calibration are unchanged and still match Schnakenberg/Turing. Pass 2 replaces the half-step ring with a parabolic sub-bin peak: 14.072 cycles, 1.114 µm, 0.5% under the calibrated 1.12 µm. The kinetics remain a transfer into the ocellus, not a published onychophoran RD fit.  
  evidence: `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L60–80; `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L249–263; `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L215–247
- **Morphological realism / specimen match: 84** — Last-frame CV similarity to the CC BY Couch photograph is 96.0 on a 12-frame 640×480 GIF. Orientation L1 is 0.0134, so the frame is not a byte-identical copy, but the 0.16 body gain is still a visualization of a whole animal rather than a 1.12 µm micrograph.  
  evidence: `docs/media/euperipatoides_rowelli_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L310–324; `docs/media/euperipatoides_rowelli_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 90** — Wavelength still falls monotonically as γ increases. In-band max Re σ is now +8571 (full), −5662 (equal diffusion), and −22114 (reversed ratio), so the two stable controls are no longer the same number.  
  evidence: `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L195–212; `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L162–167
- **Morphometric validation: 88** — Pearson r = 0.9915, RMSE = 1171, KS = 0.069 on the dispersion curves. Saturated wavelength 1.114 µm versus the 1.12 µm calibration and versus the 2.5 µm receptor pitch. Body aspect 0.610 versus a circular null of 1. Orientation L1 0.0134 versus an identical-histogram null of 0. Still one specimen, and the growth-rate RMSE is large on an absolute scale.  
  evidence: `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L102–192; `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L486–554; `datasets/euperipatoides_rowelli_turing_wavelength_s0/euperipatoides_rowelli_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Same five checked DOIs and the SHA-256-pinned CC BY 2.0 Couch photograph as Pass 1. No new source was required for the numerical remediation.  
  evidence: `datasets/euperipatoides_rowelli_turing_wavelength_s0/paper_citations.bib` L1–54; `datasets/euperipatoides_rowelli_turing_wavelength_s0/euperipatoides_rowelli_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 86** — Pass 2 comments mark the in-band control and the sub-bin peak in place. The docstring still states the PDE and the scale transfer. Lean theorems still pin 1120 nm < 2500 nm and N = 89.  
  evidence: `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L1–30; `morphoeval/lean/loops/Loop_3223_euperipatoides_rowelli_turing_wavelength_s0.lean` L31–52
- **Reproducibility package: 88** — --self-test now also requires the reversed-ratio control to be strictly more stable in-band than equal diffusion, λ_2d within 2% of 1.12 µm, and a non-zero orientation L1. Seed, γ, and the specimen hash are unchanged. Deterministic re-run.  
  evidence: `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L568–584; `datasets/euperipatoides_rowelli_turing_wavelength_s0/README.md` L1–29

Remediation:
- {'id': 'R1', 'item': 'Independent second assessor — still open. This run had only the builder-assessor.'}
- {'id': 'R2', 'item': 'Resolved: sub-bin peak is 14.072 cycles, λ = 1.114 µm.'}
- {'id': 'R3', 'item': 'Resolved: orientation-histogram L1 = 0.0134 against a null of 0.'}
- {'id': 'R4', 'item': 'Resolved: in-band controls are +8571, −5662, and −22114.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=2.5]** `independent_assessor` → `model-fidelity`: Still one assessor. No second provider was available in this run.
- **[RESOLVED · P=5.0]** `morphometric_validation` → `morphometric-validation`: Sub-bin DCT peak reports 14.072 cycles and 1.114 µm.
- **[RESOLVED · P=5.0]** `specimen_comparison` → `specimen-match`: 12-bin Sobel orientation L1 between specimen and GIF last frame is 0.0134.
- **[RESOLVED · P=3.5]** `parameter_sweep` → `parameter-expressivity`: In-band controls are +8571, −5662, and −22114 rather than one shared k=0 eigenvalue.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg k_sigma at 1.12 µm in the E. rowelli ocellus (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `345373ab42e0e02cf6b6faa2dc71e527f6cadf8f798b162c003fe87164eb8759`
- **input_dataset_sha256**: `datasets/euperipatoides_rowelli_turing_wavelength_s0/euperipatoides_rowelli_specimen.jpg` (`29166a767a52ae55…`), `datasets/euperipatoides_rowelli_turing_wavelength_s0/euperipatoides_rowelli_turing_wavelength_s0_reference.json` (`953c4f019dafd397…`), `datasets/euperipatoides_rowelli_turing_wavelength_s0/paper_citations.bib` (`4bc7e5b005ed9e1a…`)
- `datasets/euperipatoides_rowelli_turing_wavelength_s0/euperipatoides_rowelli_turing_wavelength_s0_metrics.json` — artifact_sha256 `6d4c1fdf5c5266ac…`, 1,864 B, 
- `docs/media/euperipatoides_rowelli_turing_wavelength_s0_comparison.png` — artifact_sha256 `c75a57997907af22…`, 506,526 B, 1512×984 px
- `docs/media/euperipatoides_rowelli_turing_wavelength_s0_growth.gif` — artifact_sha256 `450e8f6e9205d790…`, 1,841,776 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| continuum vs explicit-PDE dispersion Pearson r | 0.9915 Pearson r | 1.0 | corrcoef of σ(k) on modes 3..31; eigenvector-seeded explicit periodic linearization *(RMSE of growth rate = 1170.7; KS statistic = 0.0690)* |
| specimen stripe wavelength versus ocellus receptor pitch | 1.1143 µm | 2.5 | dominant DCT ring of the saturated activator, converted by domain length 14 × 1.12 µm *(Detail wavelength is required to differ from the 2.5 µm pitch; equality would collapse this scale into the receptor lattice.)* |
| segmented specimen body aspect ratio | 0.6099 height/width | 1.0 | Otsu dark-body mask on the CC BY specimen, largest component bounding box *(foreground fraction = 0.1782; circularity = 0.1124)* |
| segmented specimen structure-tensor coherency | 0.5216 coherency | 0.4965 | mean (λ1−λ2)/(λ1+λ2) of the Sobel structure tensor on the specimen *(Specimen and simulation coherencies are paired, not copied: the reference is the simulation measurement.)* |
| Holocene countable detail wavelengths across the 100 µm eye | 89 wavelengths | 0 | floor(100000 nm / 1120 nm) *(Chronoseries 1/1 at 0 Ma versus the unresolved predecessor. No successor.)* |
| specimen versus simulation orientation-histogram L1 | 0.0134 L1 | 0.0 | 12-bin Sobel orientation histograms on the 240×180 interior of the specimen and the GIF last frame *(Pass 2. Non-zero L1 is required; the body-masked gain moves orientations off the raw photograph.)* |

## Environment at checkpoint


Pass 2 working tree. Signing was not performed: ~/.nanopub/profile.yml is absent, so trusty URIs were not minted.
