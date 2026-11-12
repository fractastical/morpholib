# MorphoEval report — audit `2026-10-27-drosophila-succini-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-succini-turing-wavelength-s0-audit -> 2026-10-27-drosophila-succini-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila succini Turing wavenumber at 2.8 µm (44 Ma, detail scale s0) (`models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py`) | 88 | 80 | 90 | 88 | 86 | 90 | 91 | 88.7 | 83.0 | 89.1 | 86.5 | **86.5** | faithful |

## Findings and remediation

### Drosophila succini Turing wavenumber at 2.8 µm (44 Ma, detail scale s0)  —  overall 86.5 (faithful)

Pass 2 keeps the Schnakenberg wavelength law at 2.759e-06 m and 44 Ma, and scores the D_u sweep, the stem and equal-diffusivity controls, the specimen segmentation, and the band-restricted dispersion fit. The remaining limits are the synthetic gallery card, the transferred berryi wavelength, the 11 percent nonlinear-wavelength binning offset, and the unsigned nanopubs.

- **Model fidelity to source: 88** — Schnakenberg kinetics, the Jacobian trace and determinant (stable without diffusion), sigma(k), and k_c = argmax sigma are implemented, and D_u is solved so the peak is within 0.01 percent of 2.759e-06 m. The textbook k_h is computed and is not silently substituted for k_c. Deducted because no Drosophila morphogen pair is identified in Hennig 1965, and because k_h (7.30) and k_c (6.28) differ above onset.  
  evidence: `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L20–28; `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L88–177; `morphoeval/lean/loops/Loop_3172_drosophila_succini_turing_wavelength_s0.lean` L16–40
- **Morphological realism / specimen match: 80** — The GIF is the Schnakenberg field, not a paste of the card: twelve frames, 960 by 720, amber spot palette. The canonical CV comparator (12-bin Sobel orientation, 24-bin radial FFT, structure-tensor coherency, occupancy) scores the last frame against the gallery card in the mid-80s, above the 58 gate. The card itself is a synthetic spiral with a 100 µm bar, so the score stays at 80 rather than a holotype match.  
  evidence: `docs/media/drosophila_succini_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_succini_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L345–373; `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L445–476
- **Parameter expressivity: 90** — At fixed r = 18.72, lambda increases at each step of the D_u scale set {0.25, 0.50, 0.75, 1, 1.5, 2, 3}. The ratio axis is a separate control: r = 1 and r = 8.40 sit below onset and sigma_max is negative; r = 18.72 is unstable. Both are drawn on the comparison figure.  
  evidence: `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L245–277; `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L405–433; `datasets/drosophila_succini_turing_wavelength_s0/drosophila_succini_turing_wavelength_s0_reference.json` L1–30
- **Morphometric validation: 88** — On the pre-registered band sigma > -0.25 (10 modes) the spectral PDE matches analytic sigma with Pearson r = 0.999999997 and RMSE = 2.98e-5. The full spectrum is reported at r = 0.793 and is not used as the gate. The nonlinear radial peak is 2.46 µm against the linear 2.76 µm (FFT binning on a 3.2-wavelength box). Specimen foreground fraction 0.404 versus a 0.62 prior, coherency 0.746 versus the field's 0.883, circularity 0.048 versus 0.80, and orientation KS 0.094 versus 0. None of those pairs are equal.  
  evidence: `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L597–712; `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L490–493; `datasets/drosophila_succini_turing_wavelength_s0/drosophila_succini_turing_wavelength_s0_reference.json` L40–80
- **Source and data traceability: 86** — Five DOIs were resolved on OpenAlex or Crossref (Turing, Schnakenberg, Kondo, Hennig Zenodo 10.5281/zenodo.10748923, Grimaldi Zenodo 10.5281/zenodo.10747237). The registry DOI 10.1163/187631265X00018 returned 404 and is recorded as unused. Commons had no specimen bitmap; the gallery card SHA-256 is pinned.  
  evidence: `datasets/drosophila_succini_turing_wavelength_s0/paper_citations.bib` L1–50; `datasets/drosophila_succini_turing_wavelength_s0/scout_manifest.json` L1–80; `datasets/drosophila_succini_turing_wavelength_s0/README.md` L14–25
- **Code-to-model traceability: 90** — Each modelling step in run_model follows the docstring: Jacobian, onset, D_u solve, numeric dispersion, 2D field, chronoseries scaling. Lean checks the integer shadow of the onset inequality, the 377 nm compression, and the mode-count increase. It does not formalise the real eigenvalue.  
  evidence: `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L496–590; `morphoeval/math_kg_loops/delphi10000-drosophila-succini-turing-wavelength-s0.yaml` L1–40
- **Reproducibility package: 91** — python3 models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py --self-test passed: wavelength error 7.5e-5 relative, band r > 0.98, both controls negative, sweep monotone, GIF 12 frames at 960x720, specimen SHA-256 matched, seed 3172. Dependencies are the repo scientific stack (numpy, scipy, matplotlib, Pillow).  
  evidence: `models/2026-delphi10000-drosophila-succini-turing-wavelength-s0.py` L715–755; `datasets/drosophila_succini_turing_wavelength_s0/README.md` L20–25

Structured open needs (pressure $P = 2n + c + 0.5d + 0.2\ln(1+a)$):
- **[RESOLVED · P=1.3]** `parameter_sweep` → `parameter-expressivity`: D_u sweep is monotone in lambda and is in the metrics JSON and the comparison figure.
- **[RESOLVED · P=1.5]** `specimen_comparison` → `specimen-match`: Gallery card segmented; foreground, coherency, circularity, and orientation KS differ from their reference values. GIF CV similarity is recorded in the per-loop fitness JSON.
- **[RESOLVED · P=1.5]** `morphometric_validation` → `morphometric-validation`: Band Pearson r > 0.98 with RMSE ~ 3e-5; full-spectrum r ~ 0.79 is reported beside it. Nonlinear FFT wavelength is 2.46 µm against the linear 2.76 µm.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila succini Turing wavenumber at 2.8 µm (44 Ma, detail scale s0)
- **spec_fingerprint** (semantic model + inputs): `e5db01a3e13f63a6317e9540b0b493892dc3fdac609c17758cbc181cf60cf311`
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


Same parent commit as pass 1. Slot files are new in the loop commit. Scale 2.8 µm (2.759e-06 m), Middle Eocene, 44 Ma. Signing profile ~/.nanopub/profile.yml is absent, so nanopubs are not signed.
