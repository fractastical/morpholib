# MorphoEval report — audit `2026-10-27-dickinsonia-costata-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dickinsonia costata cellular Turing wavelength, 56 µm, White Sea 558 Ma (Pass 1) (`models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py`) | 82 | 52 | 55 | 55 | 90 | 84 | 86 | 73.0 | 53.1 | 86.8 | 69.2 | **69.2** | mostly faithful |

## Findings and remediation

### Dickinsonia costata cellular Turing wavelength, 56 µm, White Sea 558 Ma (Pass 1)  —  overall 69.2 (mostly faithful)

Pass 1 runs Schnakenberg kinetics at Du=0.05, Dv=0.55 on a periodic septal patch and anchors the measured spectral peak at 5.564e-5 m (55.64 µm, the 56 µm cellular tier). k_c=2.587, linear fastest k_m=2.467, measured k=2.558 (ratio 1.037). The animation is a 12-frame spot field at 720x540, not the fossil. No diffusivity sweep, no equal-diffusion control, and no specimen ellipse are in the Pass 1 metrics.

- **Model fidelity to source: 82** — sigma(k), k_c, and the spectral Schnakenberg step match Turing 1952 and Schnakenberg 1979. The Pass 1 artifact exercises only the reference diffusivity, so the dispersion law is checked at one point (k_meas/k_m = 1.037).  
  evidence: `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` L83–100; `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` L133–168
- **Morphological realism / specimen match: 52** — The comparison figure places the Verisimilus fossil beside the field, and they do not match: the fossil is an oval of parallel modules and the field is a spot lattice. Coarse CV similarity of the Pass 1 GIF is 83.4, which overstates the visual resemblance.  
  evidence: `docs/media/dickinsonia_costata_turing_wavelength_s0_comparison.png` L0–0; `docs/media/dickinsonia_costata_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 55** — a, b, Du, Dv and the sweep list are defined, but --pass-id 1 does not run the sweep or the ablation, so the delivered figure has a single point.  
  evidence: `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` L414–430; `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` L46–58
- **Morphometric validation: 55** — Selected wavelength 55.64 µm is reported against the 2210 µm organ-tier pitch, and age 558 Ma against 635 Ma. Both differ from their references. Pearson, RMSE, KS, and specimen-contour metrics are absent.  
  evidence: `datasets/dickinsonia_costata_turing_wavelength_s0/dickinsonia_costata_turing_wavelength_s0_metrics.json` L13–32
- **Source and data traceability: 90** — BibTeX DOIs for Turing, Schnakenberg, Bobrovskiy, Evans 2017, Evans 2021, and Hoekzema. Specimen SHA-256 and CC BY 2.5 Commons URL are pinned.  
  evidence: `datasets/dickinsonia_costata_turing_wavelength_s0/paper_citations.bib` L1–65; `datasets/dickinsonia_costata_turing_wavelength_s0/dickinsonia_costata_turing_wavelength_s0_reference.json` L1–20
- **Code-to-model traceability: 84** — Dispersion, integrator, and render functions are separated and named against the source laws. Pass 1 does not yet call the sweep from the canonical artifact.  
  evidence: `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` L1–28; `morphoeval/lean/loops/Loop_3167_dickinsonia_costata_turing_wavelength_s0.lean` L15–53
- **Reproducibility package: 86** — Seed 3167, pinned specimen digest, and --pass-id 1 reproduce the baseline. --self-test is the Pass 2 gate and is not what this artifact ran.  
  evidence: `models/2026-delphi10000-dickinsonia-costata-turing-wavelength-s0.py` L663–690; `datasets/dickinsonia_costata_turing_wavelength_s0/README.md` L1–20

Remediation:
- {'id': 'R1', 'item': 'Terminal GIF frame is a spot field. Fit the animation to the CC BY 2.5 fossil and draw the specimen ellipse.'}
- {'id': 'R2', 'item': 'Pass 1 metrics omit the sqrt(D) sweep and its Pearson r.'}
- {'id': 'R3', 'item': 'No equal-diffusion negative control in the Pass 1 artifact.'}
- {'id': 'R4', 'item': 'No KS/RMSE/CV and no real-specimen contour metrics yet.'}

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=2.3]** `specimen_comparison` → `specimen-match`: Render the growth GIF so the terminal frame is the real fossil and overlay a segmented ellipse on the comparison figure.
- **[OPEN · P=1.7]** `parameter_sweep` → `parameter-expressivity`: Deliver the diffusivity-factor sweep and the Pearson correlation of measured versus linear wavenumbers.
- **[OPEN · P=1.4]** `parameter_sweep` → `model-fidelity`: Report the equal-diffusion ablation on the same grid (noise must decay).
- **[OPEN · P=1.9]** `morphometric_validation` → `morphometric-validation`: Add RMSE, KS, and CV for the wavenumber sweep, plus specimen aspect, circularity, coherency, and ridge spacing with references that are not the measured values.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Dickinsonia costata cellular Turing wavelength, 56 µm, White Sea 558 Ma (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `fd86eb9ab09aa054555305e9405e0ca0d623d01f76b8fb3f7e4bc2a4741557ec`
- **input_dataset_sha256**: `datasets/dickinsonia_costata_turing_wavelength_s0/README.md` (`2eab73577d2f595b…`), `datasets/dickinsonia_costata_turing_wavelength_s0/dickinsonia_costata_turing_wavelength_s0_reference.json` (`654790d69ff50dc8…`), `datasets/dickinsonia_costata_turing_wavelength_s0/dickinsonia_costata_turing_wavelength_s0_specimen.jpg` (`9cde902e8577ea73…`), `datasets/dickinsonia_costata_turing_wavelength_s0/paper_citations.bib` (`5eef4d2dd1042991…`), `datasets/dickinsonia_costata_turing_wavelength_s0/scout_manifest.json` (`7d02cb51fc7fd66f…`)
- `docs/media/dickinsonia_costata_turing_wavelength_s0_comparison.png` — artifact_sha256 `fd889af7f5a33470…`, 700,140 B, 1440×984 px
- `docs/media/dickinsonia_costata_turing_wavelength_s0_growth.gif` — artifact_sha256 `830616083e586cdc…`, 1,751,800 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| selected_wavelength_um | 55.64 µm | 2210.0 | Length unit anchored so the reference-run spectral peak equals the assigned cellular pitch *(value is the cellular pitch; reference_value is the 2.2 mm module pitch and must differ)* |
| geological_age_ma | 558.0 Ma | 635.0 | White Sea Dickinsonia horizon used by Bobrovskiy et al. 2018 *(558 != 635, so the age comparison is not self-referential)* |

## Environment at checkpoint


Pass 1 working tree on top of origin/main. New slot files are not in that commit yet.
