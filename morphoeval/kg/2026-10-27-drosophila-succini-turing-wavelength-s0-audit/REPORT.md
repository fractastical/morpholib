# MorphoEval report — audit `2026-10-27-drosophila-succini-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila succini Turing wavenumber at 2.8 µm (44 Ma, detail scale s0) (`models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py`) | 82 | 70 | 72 | 74 | 80 | 84 | 82 | 78.7 | 71.5 | 81.9 | 76.8 | **76.8** | mostly faithful |

## Findings and remediation

### Drosophila succini Turing wavenumber at 2.8 µm (44 Ma, detail scale s0)  —  overall 76.8 (mostly faithful)

Pass 1 implements Schnakenberg kinetics and the Turing wavenumber at 2.759e-06 m for Drosophila succini at 44 Ma. The dispersion band already fits, but this scoring withholds specimen and sweep credit until the sidecar and the image-fitness test exist. Overall is the rubric weighted mean, not a separate number.

- **Model fidelity to source: 82** — The script states and codes Schnakenberg kinetics, the Jacobian, sigma(k), and k_c = argmax sigma(k), and it solves D_u so the peak sits on 2.759 µm at r = 18.72. Deducted because the kinetics are not a Drosophila-specific published system, and because the textbook determinant wavenumber k_h is not the same as k_c above onset.  
  evidence: `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L20–28; `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L88–162; `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L165–177
- **Morphological realism / specimen match: 70** — The comparison figure places the gallery card next to the amber-coloured spot field, and the GIF is twelve frames of that field at 960 by 720. Pass 1 does not yet credit a scored CV similarity. The card is a synthetic spiral, so anatomical match to an amber inclusion is not claimed.  
  evidence: `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L376–442; `docs/media/drosophila_succini_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_succini_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 72** — r = 8.40 and r = 1 are negative controls with sigma_max < 0, and a D_u scale sweep is drawn. Pass 1 does not yet treat the monotonicity table as a scored measurement.  
  evidence: `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L245–277; `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L405–433
- **Morphometric validation: 74** — A 1D spectral check of sigma(k) exists. Pass 1 does not yet score the band Pearson residual, the nonlinear wavelength offset, or the specimen KS distance as sidecar rows.  
  evidence: `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L187–234; `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L597–712
- **Source and data traceability: 80** — Turing, Schnakenberg, Kondo, Hennig (Zenodo/BHL), and Grimaldi (Zenodo/BHL) are cited. The species-registry DOI 10.1163/187631265X00018 did not resolve and is not used. The specimen is a gallery fallback after an empty Commons search.  
  evidence: `datasets/drosophila_succini_turing_wavelength_s0/paper_citations.bib` L1–50; `datasets/drosophila_succini_turing_wavelength_s0/README.md` L1–25; `datasets/drosophila_succini_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 84** — The module docstring names the equations and the chronoseries contrast, and the Lean file checks the integer form of the onset crossing, the berryi compression, and the mode count. The Lean theorems are Nat comparisons, not a real-analysis proof of sigma(k).  
  evidence: `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L30–43; `morphoeval/lean/loops/Loop_3172_drosophila_succini_turing_wavelength_s0.lean` L1–45
- **Reproducibility package: 82** — --self-test checks the wavelength, the band correlation, both negative controls, the monotone D_u sweep, and a GIF of at least 10 frames and 480 px. Dataset hashes beyond the specimen digest are not all pinned inside the script.  
  evidence: `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L70–86; `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L715–755

Remediation:
- Add the D_u sweep monotonicity table to the scored measurements.
- Add specimen segmentation metrics that are not self-referential.
- Report Turing-band Pearson r and the weaker full-spectrum r side by side.
- Score the animated GIF against the specimen with the canonical CV function.

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[OPEN · P=1.3]** `parameter_sweep` → `parameter-expressivity`: Tabulate lambda versus D_u at fixed r and confirm it is monotone in the measurements sidecar.
- **[OPEN · P=1.5]** `specimen_comparison` → `specimen-match`: Segment the gallery card and report foreground fraction, coherency, circularity, and an orientation KS distance against the spot field, with value not equal to the reference.
- **[OPEN · P=1.5]** `morphometric_validation` → `morphometric-validation`: Publish the band-restricted Pearson r and RMSE, and state the full-spectrum r so the band choice is not hidden.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila succini Turing wavenumber at 2.8 µm (44 Ma, detail scale s0)
- **spec_fingerprint** (semantic model + inputs): `d5250d5d69797419c7dcbd44bd49a2f156290e47fdf9d65ee96983d3232db5d9`
- **input_dataset_sha256**: `datasets/drosophila_succini_turing_wavelength_s0/drosophila_succini_turing_wavelength_s0_reference.json` (`e227ba771980817d…`), `datasets/drosophila_succini_turing_wavelength_s0/drosophila_succini_turing_wavelength_s0_specimen.jpg` (`1a0de2972916cbfa…`)
- `docs/media/drosophila_succini_turing_wavelength_s0_comparison.png` — artifact_sha256 `8b3f9e4fcc75332b…`, 448,421 B, 1512×1008 px
- `docs/media/drosophila_succini_turing_wavelength_s0_growth.gif` — artifact_sha256 `c5d87de63a9f808b…`, 4,958,003 B, 960×720 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| succini critical wavelength from argmax sigma(k) | 2.7592e-06 m | 2.759e-06 | k_c = argmax sigma(k); lambda = 2 pi / k_c * 2.759e-6 m *(linear prediction; reference is the assigned scale, value is the solved peak)* |
| nonlinear spot-field dominant wavelength | 2.4581e-06 m | 2.7592e-06 | radial FFT peak of the saturated 2D Schnakenberg field *(finite-domain FFT binning shifts this off the linear peak)* |
| Turing-band Pearson r of PDE growth vs analytic sigma | 1.0 correlation | 0.98 | modes with analytic sigma > -0.25; spectral 1D PDE, T=1.2 *(RMSE = 2.98334e-05; full-spectrum r = 0.7931 (stable modes mix the fast eigenvalue))* |
| Turing-band RMSE of sigma | 3e-05 1/time | 0.05 | sqrt mean square of numeric minus analytic growth rates on the band *(value is the residual, reference is the tolerance, they are not the same quantity)* |
| stem predecessor sigma_max at r=8.40 | -0.110821 1/time | 0.0 | max_k sigma(k) at r_stem below onset, succini D_u *(negative value means the 2.8 µm ocellar mode is absent before 44 Ma)* |
| equal-diffusivity ablation sigma_max | -0.321374 1/time | 0.0 | max_k sigma(k) at r=1 *(ablation control)* |
| berryi transferred detail wavelength | 2.3823e-06 m | 2.3822e-06 | same r, D scaled by (98.43/114)^2; k_c remeasured *(successor Drosophila berryi, 19 Ma)* |
| specimen foreground fraction | 0.4043 area fraction | 0.62 | border-median contrast mask on the gallery card, 160x120 *(reference 0.62 is a generic filled-body prior, not this card)* |
| specimen structure-tensor coherency | 0.7458 coherency | 0.8826 | mean (λ1-λ2)/(λ1+λ2) on the contrast mask *(specimen value and simulation reference differ; KS orientation is separate)* |
| KS distance of gradient orientations specimen vs simulation | 0.0939 KS statistic | 0.0 | two-sample KS on Sobel orientation angles *(non-circular specimen metric)* |
| specimen contour circularity | 0.0478 4πA/P² | 0.8 | mask perimeter from binary erosion *(low circularity is expected for a full-bleed spiral card)* |

## Environment at checkpoint


Parent of the loop commit. New slot-3172 paths are not in 0c76d15d; the KG build uses --no-verify-loci for that reason. Scale 2.759e-06 m, age 44 Ma.
