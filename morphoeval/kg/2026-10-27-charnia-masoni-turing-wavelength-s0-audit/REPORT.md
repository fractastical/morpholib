# MorphoEval report — audit `2026-10-27-charnia-masoni-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `96a891cfffcbf7fb81617c1df76c424d1d7359a5` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Charnia masoni apical Turing wavelength at 140 µm (565 Ma), pass 1 (`models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py`) | 80 | 72 | 74 | 76 | 90 | 84 | 86 | 78.0 | 73.5 | 86.8 | 78.8 | **78.8** | mostly faithful |

## Findings and remediation

### Charnia masoni apical Turing wavelength at 140 µm (565 Ma), pass 1  —  overall 78.8 (mostly faithful)

Pass 1 of the 140 µm Charnia masoni apical Turing model. The dispersion law already matched (Pearson r 0.998) but the render was a landscape disk at 85° (CV 83.9) and the wavelength sweep mixed k_c with the fastest mode (RMSE 34.5 µm). Those three items are the remediation list.

- **Model fidelity to source: 80** — Eqs. 1–7 are the Schnakenberg Turing law and the Euler probe already tracked analytic σ (r = 0.998). The pass-1 sweep then compared that σ-peak with a different wavenumber, so the wavelength test was not yet the law it named.  
  evidence: `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L83–142
- **Morphological realism / specimen match: 72** — CV against the Tübingen cast was 83.9, but the frame was a circle of near-horizontal stripes with a midline seam. It did not read as a rangeomorph frond.  
  evidence: `docs/media/charnia_masoni_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 74** — The chevron-angle search minimised an orientation L1 that kept falling through 85°, so the operating point sat on the boundary. Argmax-bin versus angle correlated only at 0.58.  
  evidence: `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L365–399
- **Morphometric validation: 76** — Dispersion RMSE was already 0.020. The wavelength sweep RMSE of 34.5 µm was an apples-to-oranges gap, not a measured biological residual, and no KS against the cast had been interpreted.  
  evidence: `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L144–191
- **Source and data traceability: 90** — Ford, Dunn, Brasier, Turing and Schnakenberg DOIs were checked on Crossref, and the cast's CC BY-SA 3.0 attribution (Ghedoghedo, Tübingen, 2013) was already pinned.  
  evidence: `datasets/charnia_masoni_turing_wavelength_s0/paper_citations.bib` L1–40
- **Code-to-model traceability: 84** — Equation numbers in the module docstring match the functions, and the Lean integers are checked at runtime. The pass-1 figure did not yet label which wavenumber it was plotting.  
  evidence: `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L413–423
- **Reproducibility package: 86** — Seed 3145, --self-test, and SHA-256 pins for the cast, README and bibliography. Pass 1 had not yet frozen the portrait canvas size.  
  evidence: `models/2026-delphi10000-charnia-masoni-turing-wavelength-s0.py` L67–81

Remediation:
- Replace the 720×540 disk at 85° with a portrait chevron whose angle sits at the CV maximum (58° on the probe grid), not on the edge of the search.
- Sweep the fastest-mode wavelength against itself. The pass-1 sweep mixed the determinant-critical k_c with argmax σ and reported RMSE 34.5 µm.
- Draw the equal-diffusion control as a panel. Pass 1 showed it only as a decaying curve.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Charnia masoni apical Turing wavelength at 140 µm (565 Ma), pass 1
- **spec_fingerprint** (semantic model + inputs): `571668419f3b61d8829b66959046ef157ab530e8516cc89e6ab7b1925a5766c7`
- **input_dataset_sha256**: `datasets/charnia_masoni_turing_wavelength_s0/README.md` (`4488c865883388ab…`), `datasets/charnia_masoni_turing_wavelength_s0/charnia_masoni_specimen.jpg` (`c7e8d470af175675…`), `datasets/charnia_masoni_turing_wavelength_s0/charnia_masoni_turing_wavelength_s0_reference.json` (`262080091420a03a…`), `datasets/charnia_masoni_turing_wavelength_s0/paper_citations.bib` (`f4cb4c63fbfbc0fd…`), `datasets/charnia_masoni_turing_wavelength_s0/scout_manifest.json` (`ee8179dfab5e793f…`)
- `docs/media/charnia_masoni_turing_wavelength_s0_comparison.png` — artifact_sha256 `261c01363f55e4ae…`, 342,160 B, 1452×902 px

| measurement | value | reference | method |
|---|---:|---|---|
| fastest-growing apical stripe wavelength | 167.44 µm | 140.4 | 2*pi/k_m where k_m maximises analytic sigma(k); the determinant-critical k_c is scaled so its wavelength is 140.4 µm, and above onset k_m is smaller |
| dispersion-curve Pearson r (Euler versus analytic sigma) | 0.9984 Pearson r | 0.98 | 12-point σ(k) from Eq. 7 versus short explicit-Euler growth of the same cosine *(RMSE of sigma = 0.02007; control sigma_max = -0.9004)* |
| wavelength-sweep Pearson r across inhibitor diffusivity | 1.0 Pearson r | 0.98 | Du and Dv scaled together by {0.35, 0.6, 1, 1.7, 2.8, 4.5}; analytic argmax of σ(k) versus the Euler argmax *(RMSE = 3.264 µm)* |
| equal-diffusion control peak growth rate | -0.9004 1/time | 0.0 | same Euler probe with Du = Dv = 0.062 *(patterned sigma_max = 0.1964; profile amplitude ratio control/pattern = 0.0005)* |
| segmented specimen slab foreground fraction | 0.5936 area fraction | 0.5 | largest connected component of (R - B) above the 42nd percentile on the Tübingen cast resized to 720×540 |
| specimen versus render orientation-histogram L1 | 0.2825 L1 | 0.0 | 12-bin Sobel orientation histograms inside the cast mask and the rendered slab *(KS statistic = 0.0932, KS p = 7.215e-10, chevron angle = 58.0 deg, angle-sweep Pearson = 0.9953)* |
| specimen structure-tensor coherency | 0.456 coherency | 0.3803 | mean (λ1-λ2)/(λ1+λ2) of the Gaussian-smoothed structure tensor on the cast *(render coherency = 0.3803. Occupancy Pearson r = 0.8005, occupancy RMSE = 0.2809)* |
| detail-to-organ wavenumber ratio | 45.0285 k_detail / k_organ | 45.0 | 0.006322 m / 0.0001404 m *(D_u = 1.426661e-12 m^2/s at tau = 120 s; age = 565 Ma)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- matplotlib: 3.11.2
- Pillow: 13.0.0
- scipy: 1.18.1

Pass-1 pixels were a 720×540 disk of near-horizontal stripes at a chevron angle pegged at 85°, CV 83.9, with a wavelength sweep that compared the determinant-critical λ to the σ-peak (RMSE 34.5 µm). Those three defects are fixed in the committed tree; this record does not rescore them.
