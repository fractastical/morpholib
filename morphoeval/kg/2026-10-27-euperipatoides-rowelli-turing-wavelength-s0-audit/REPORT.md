# MorphoEval report — audit `2026-10-27-euperipatoides-rowelli-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg k_sigma at 1.12 µm in the E. rowelli ocellus (Holocene, 0 Ma) (`models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py`) | 84 | 80 | 86 | 82 | 90 | 82 | 84 | 84.7 | 80.8 | 85.5 | 83.3 | **83.3** | faithful |

## Findings and remediation

### Schnakenberg k_sigma at 1.12 µm in the E. rowelli ocellus (Holocene, 0 Ma)  —  overall 83.3 (faithful)

Pass 1 calibrates Schnakenberg (a=0.12, b=0.88, Du=1, Dv=22, seed 3223) so the fastest Turing mode is 14 cycles on a 15.68 µm patch (λ = 1.12 µm). Explicit linear PDE versus continuum dispersion: Pearson r = 0.9915, RMSE = 1171 on the growth-rate scale, KS = 0.069. Saturated DCT peak is quantized at 14.25 cycles (λ_2d = 1.100 µm). Equal-diffusion and reversed-ratio controls are both stable, but both report the same k=0 eigenvalue (−4094.7), so the ablation panel does not yet separate them. The GIF is the Alan Couch CC BY 2.0 animal with a 0.16 body-masked gain.

- **Model fidelity to source: 84** — Jacobian, Neumann DCT integrator, and Turing dispersion are the Schnakenberg/Turing equations, with γ solved for k_sigma at 1.12 µm. The saturated field is still reported on a 0.5-step ring (14.25 cycles). The kinetics are transferred into the ocellus; they are not a published onychophoran RD model.  
  evidence: `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L60–80; `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L71–99; `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L215–247
- **Morphological realism / specimen match: 80** — The 12-frame 640×480 GIF is the CC BY 2.0 Couch photograph with a 0.16 luminance gain inside the Otsu body mask. Computer-vision similarity of the last frame to that photograph is 96.0. The match is to the whole animal, not to a 1.12 µm micrograph; the gain is a declared visualization.  
  evidence: `docs/media/euperipatoides_rowelli_turing_wavelength_s0_growth.gif` L0–0; `docs/media/euperipatoides_rowelli_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L336–356
- **Parameter expressivity: 86** — Raising γ shortens the selected wavelength monotonically across scales 0.55, 0.75, 1, 1.40, 1.90. Both diffusion ablations are linearly stable. They are not yet separated from each other inside the unstable band (R4).  
  evidence: `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L195–212; `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L162–169
- **Morphometric validation: 82** — Dispersion Pearson r = 0.9915, RMSE = 1171, KS = 0.069 against the continuum law. Segmented body aspect 0.610 versus a circular null of 1, and the 1.100 µm saturated wavelength versus the 2.5 µm receptor pitch, are non-circular. The ring estimator is still quantized (R2) and there is no orientation-histogram KS (R3). One specimen.  
  evidence: `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L102–194; `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L275–307; `datasets/euperipatoides_rowelli_turing_wavelength_s0/euperipatoides_rowelli_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 90** — Five DOIs were checked on OpenAlex or Crossref. The specimen is Alan Couch, CC BY 2.0, Wikimedia File:Euperipatoides_rowelli.jpg, SHA-256 pinned in the reference JSON.  
  evidence: `datasets/euperipatoides_rowelli_turing_wavelength_s0/paper_citations.bib` L1–52; `datasets/euperipatoides_rowelli_turing_wavelength_s0/euperipatoides_rowelli_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 82** — The module docstring states the PDE, the boundary condition, and the scale transfer. Dispersion, sweep, and render functions are separate. The metrics JSON is a derived log, not a second derivation of k_sigma.  
  evidence: `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L1–30; `morphoeval/lean/loops/Loop_3223_euperipatoides_rowelli_turing_wavelength_s0.lean` L31–52
- **Reproducibility package: 84** — --self-test checks r>0.98, age 0, both controls stable, monotonic sweep, 12 frames, and min side 480. Seed 3223 and γ are pinned in the reference JSON. Dependencies are the repo scientific stack (numpy, scipy, pillow, matplotlib).  
  evidence: `models/2026-delphi10000-euperipatoides-rowelli-turing-wavelength-s0.py` L557–585; `datasets/euperipatoides_rowelli_turing_wavelength_s0/README.md` L1–29

Remediation:
- {'id': 'R1', 'item': 'Independent second assessor — still builder-scored.'}
- {'id': 'R2', 'item': 'Replace the integer DCT-ring peak with a sub-bin estimate and report λ against 1.12 µm.'}
- {'id': 'R3', 'item': 'Add a Sobel 12-bin orientation-histogram KS (or L1) between the specimen and the GIF last frame.'}
- {'id': 'R4', 'item': 'Report in-band max Re σ for the two controls so they are not identical copies of the k=0 eigenvalue.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=2.5]** `independent_assessor` → `model-fidelity`: Single builder-assessor. A second provider has not scored this audit.
- **[OPEN · P=5.0]** `morphometric_validation` → `morphometric-validation`: Saturated wavenumber is an integer DCT ring (14.25 cycles, λ=1.100 µm) rather than a sub-bin peak against the calibrated 14 cycles / 1.12 µm.
- **[OPEN · P=5.0]** `specimen_comparison` → `specimen-match`: No Sobel orientation-histogram comparison between the specimen crop and the GIF last frame is reported, only a scalar coherency pair.
- **[OPEN · P=3.5]** `parameter_sweep` → `parameter-expressivity`: Equal-diffusion and reversed-ratio controls both return the k=0 eigenvalue −4094.7, so the ablation panel does not show an in-band difference.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg k_sigma at 1.12 µm in the E. rowelli ocellus (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `345373ab42e0e02cf6b6faa2dc71e527f6cadf8f798b162c003fe87164eb8759`
- **input_dataset_sha256**: `datasets/euperipatoides_rowelli_turing_wavelength_s0/euperipatoides_rowelli_specimen.jpg` (`29166a767a52ae55…`), `datasets/euperipatoides_rowelli_turing_wavelength_s0/euperipatoides_rowelli_turing_wavelength_s0_reference.json` (`953c4f019dafd397…`), `datasets/euperipatoides_rowelli_turing_wavelength_s0/paper_citations.bib` (`4bc7e5b005ed9e1a…`)
- `datasets/euperipatoides_rowelli_turing_wavelength_s0/euperipatoides_rowelli_turing_wavelength_s0_metrics.json` — artifact_sha256 `9393707a60c9576a…`, 1,797 B, 
- `docs/media/euperipatoides_rowelli_turing_wavelength_s0_comparison.png` — artifact_sha256 `289aff418d717fbf…`, 521,196 B, 1512×984 px
- `docs/media/euperipatoides_rowelli_turing_wavelength_s0_growth.gif` — artifact_sha256 `450e8f6e9205d790…`, 1,841,776 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| continuum vs explicit-PDE dispersion Pearson r | 0.9915 Pearson r | 1.0 | corrcoef of σ(k) on modes 2..35; explicit periodic linearization of Schnakenberg *(RMSE of growth rate = 1170.7; KS statistic = 0.0690)* |
| specimen stripe wavelength versus ocellus receptor pitch | 1.1004 µm | 2.5 | dominant DCT ring of the saturated activator, converted by domain length 14 × 1.12 µm *(Detail wavelength is required to differ from the 2.5 µm pitch; equality would collapse this scale into the receptor lattice.)* |
| segmented specimen body aspect ratio | 0.6099 height/width | 1.0 | Otsu dark-body mask on the CC BY specimen, largest component bounding box *(foreground fraction = 0.1782; circularity = 0.1124)* |
| segmented specimen structure-tensor coherency | 0.5216 coherency | 0.4965 | mean (λ1−λ2)/(λ1+λ2) of the Sobel structure tensor on the specimen *(Specimen and simulation coherencies are paired, not copied: the reference is the simulation measurement.)* |
| Holocene countable detail wavelengths across the 100 µm eye | 89 wavelengths | 0 | floor(100000 nm / 1120 nm) *(Chronoseries 1/1 at 0 Ma versus the unresolved predecessor. No successor.)* |

## Environment at checkpoint


Pass 1 working tree on 0c76d15d. Loop files are new; commit will be updated to the loop commit that contains the script before the final knowledge-graph build.
