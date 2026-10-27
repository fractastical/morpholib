# MorphoEval report — audit `2026-10-27-drosophila-buzzatii-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg–Turing wavelength at 2.908 µm in Drosophila buzzatii (Pass 1) (`models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py`) | 88 | 80 | 84 | 74 | 88 | 84 | 86 | 86.7 | 77.8 | 86.1 | 82.9 | **82.9** | faithful |

## Findings and remediation

### Schnakenberg–Turing wavelength at 2.908 µm in Drosophila buzzatii (Pass 1)  —  overall 82.9 (faithful)

Pass 1 implements the Schnakenberg PDE and the closed-form wavenumber, renders a 12-frame growth GIF at 720×540, and orders buzzatii (2.908 µm, 584 modes) against bipectinata (85.92 µm, 19 modes) and cyrtoloma (207.6 µm, 8 modes) at 0 Ma. Morphometric credit is partial until the KS, segmentation, and ablation panel are re-scored.

- **Model fidelity to source: 88** — The PDE, Jacobian, and k_c formula are the published Schnakenberg/Turing objects, and gamma is solved for 2.908 µm rather than copied from the 102 µm sibling.  
  evidence: `models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py` L86–114
- **Morphological realism / specimen match: 80** — The GIF grows spots on the canonical card and the pure field is a spot lattice, but Pass 1 does not yet treat the segmentation contour as a scored specimen metric.  
  evidence: `docs/media/drosophila_buzzatii_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_buzzatii_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 84** — The gamma sweep moves lambda monotonically. The equal-diffusion control is computed, and Pass 1 does not yet give it full figure credit.  
  evidence: `models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py` L152–179
- **Morphometric validation: 74** — Pearson r of analytical versus discrete lambda is 0.99995. Pass 1 withholds the rest of this dimension until KS, segmentation, and the nonlinear RMSE are in the measurement sidecar.  
  evidence: `models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py` L152–173
- **Source and data traceability: 88** — Turing, Schnakenberg, and Norry DOIs are in the bib and the reference JSON. The specimen is the gallery fallback, SHA-256 pinned, because no CC adult bitmap was found.  
  evidence: `datasets/drosophila_buzzatii_turing_wavelength_s0/paper_citations.bib` L1–31; `datasets/drosophila_buzzatii_turing_wavelength_s0/drosophila_buzzatii_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 84** — Each modelling step is a named function mapped to k_c or to the chronoseries. Pass 1 has not yet pinned the measurement block.  
  evidence: `models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py` L1–45
- **Reproducibility package: 86** — --self-test checks lambda, the diffusion ratio, Pearson r > 0.98, negative ablation growth, mode order, and the specimen hash. Seed 3179.  
  evidence: `models/2026-delphi10000-drosophila-buzzatii-turing-wavelength-s0.py` L630–658

Remediation:
- {'id': 'R1', 'item': 'Put the two-sample KS distance between the species-card radial spectrum and the pure Schnakenberg field into the measurement sidecar (value must differ from 0).'}
- {'id': 'R2', 'item': 'Segment the canonical card and report foreground fraction, circularity, and structure-tensor coherency against a reference that is not the measured value.'}
- {'id': 'R3', 'item': 'Show the equal-diffusion ablation on the comparison figure, not only inside --self-test.'}
- {'id': 'R4', 'item': 'Report the saturated-field wavelength separately from the linear 2.908 µm target so the coarsening offset is a measured residual, not a hidden mismatch.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg–Turing wavelength at 2.908 µm in Drosophila buzzatii (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `08fa3b6fbda37d14034e66aaa32be6a4f2884d2974f79734f2b5414d7f6640c2`
- **input_dataset_sha256**: `datasets/drosophila_buzzatii_turing_wavelength_s0/drosophila_buzzatii_turing_wavelength_s0_reference.json` (`61a3792c4a139711…`), `datasets/drosophila_buzzatii_turing_wavelength_s0/drosophila_buzzatii_turing_wavelength_s0_specimen.jpg` (`21d216f74bbb2735…`)
- `docs/media/drosophila_buzzatii_turing_wavelength_s0_comparison.png` — artifact_sha256 `e036de5b2cd1da21…`, 469,642 B, 1512×984 px
- `docs/media/drosophila_buzzatii_turing_wavelength_s0_growth.gif` — artifact_sha256 `338bf89ea58e68dc…`, 3,467,062 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion_wavelength_pearson_r | 0.999952 correlation | 0.98 | pearsonr of analytical λ_c versus the most unstable discrete Fourier mode across 14 γ scales |
| dispersion_wavelength_rmse_um | 0.9619 µm | 0.0 | RMSE between closed-form λ_c and the discrete-mode λ on the same γ sweep |
| simulated_ocellus_stripe_wavelength_um | 3.5324 µm | 2.908 | parabolic-interpolated radial FFT peak of the saturated Schnakenberg activator |
| specimen_segment_foreground_fraction | 0.3785 fraction | 0.5 | border-median contrast mask on the canonical species card, 320×240 |
| specimen_structure_tensor_coherency | 0.6748 index | 0.2483 | mean structure-tensor coherency inside the specimen mask |
| specimen_segment_circularity | 0.026 index | 1.0 | 4π area / perimeter² of the specimen foreground mask |
| specimen_vs_field_ks_radial | 0.4583 KS | 0.0 | two-sample KS on 24-bin log radial spectra of the card versus the pure field |
| equal_diffusion_ablation_max_growth | -0.049563 1/time | 0.0 | max Re(σ) of the discrete dispersion with D_v set equal to D_u |
| buzzatii_wing_chord_mode_count | 584 modes | 8 | floor(1700 µm / 2.908 µm) on a representative wing chord |

## Environment at checkpoint

- python: 3.12
- numpy: 2.5.4
- scipy: 1.18.1

Pass 1 checkpoint is the parent of the loop commit. Loop files are new in this PR; loci for those paths are re-checked after the loop commit lands.
