# MorphoEval report — audit `2026-10-27-ceratites-nodosus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `040c297c78d4c53dd775f1fcd23b9440ae0558db` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-ceratites-nodosus-turing-wavelength-s0-audit -> 2026-10-27-ceratites-nodosus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ceratites nodosus Schnakenberg Turing wavenumber at 26 µm (Ladinian, 240 Ma) (`models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py`) | 90 | 78 | 88 | 88 | 88 | 84 | 90 | 89.3 | 81.8 | 87.7 | 85.8 | **85.8** | faithful |

## Findings and remediation

### Ceratites nodosus Schnakenberg Turing wavenumber at 26 µm (Ladinian, 240 Ma)  —  overall 85.8 (faithful)

Pass 2, superseding the Pass-1 audit (overall 81.1). Activator diffusivity stays at the 26 µm solve. --gamma-factor 1.55 holds Du fixed and moves the selected wavelength from 25.6 µm to 29.3 µm. Resolved modes match the discrete Jacobian eigenvalue at Pearson r = 1.000 (RMSE ~ 1e-13). The continuous-symbol residual on that band is r = 0.976, RMSE 0.0037, which is the second-difference truncation and is reported rather than hidden. The full 30-mode continuous comparison remains r = 0.990. Equal diffusion and a 0.45-wavelength domain both decay. Specimen orientation cosine against the ocellus field is 0.995; row-wise wavelength CV is 0.675. Shell segmentation is unchanged (circularity 0.659). Chronoseries counts at 26 µm remain 703, 1250, and 1641. Still one photograph, and still no second assessor.

- **Model fidelity to source: 90** — The nonlinear periodic Schnakenberg system and the onset wavenumber k_c^2 = sqrt(det J / (Du Dv)) are unchanged. Pass 2 separates solver error from stencil error: resolved modes match the discrete eigenvalue at r = 1.000, and the continuous residual (r = 0.976, RMSE 0.0037) is the five-point Laplacian, not a hidden mismatch.  
  evidence: `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L105–134; `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L502–554
- **Morphological realism / specimen match: 78** — The comparison figure still sets the Muschelkalk shell beside the 25.60 µm ocellus field. The 12-bin orientation cosine is 0.995, so the edge-angle mix is broad in both images, but the radial KS remains 0.94: the photograph is not a picture of the cellular pigment. The GIF still carries the pattern as chroma on the shell.  
  evidence: `docs/media/ceratites_nodosus_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L592–605
- **Parameter expressivity: 88** — The sweep at fixed Du lengthens lambda from 25.1 to 28.2 µm as gamma/gamma_onset goes from 1.08 to 1.90. --gamma-factor 1.55 keeps Du and moves the simulated wavelength to 29.3 µm (ratio 1.143). Equal diffusion and a subcritical domain both decay.  
  evidence: `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L254–304; `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L556–581; `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L896–923
- **Morphometric validation: 88** — Continuous 30-mode Pearson r is 0.990 and the 2D FFT peak is 2.56e-05 m. Discrete unstable-band r is 1.000. Specimen circularity 0.659 (reference 1), foreground 0.505 (reference 0.70), coherency 0.522 (reference 0.25), radial KS 0.936, orientation cosine 0.995 (reference 1), row-wise wavelength CV 0.675 (reference 0). None of these specimen comparisons copy the reference onto the value.  
  evidence: `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L318–370; `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L607–619; `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L808–848
- **Source and data traceability: 88** — Unchanged from Pass 1. Six sources, SHA-256-pinned gallery specimen, Wikimedia CC candidates in the scout manifest. The gallery JPEG has no embedded license tag.  
  evidence: `datasets/ceratites_nodosus_turing_wavelength_s0/paper_citations.bib` L1–55; `datasets/ceratites_nodosus_turing_wavelength_s0/README.md` L1–40
- **Code-to-model traceability: 84** — Pass-2 functions name the discrete eigenvalue, the subcritical length, and the orientation histogram in their docstrings. Lean still checks the integer wavelength counts and the sign of the Turing numerator, not the real dispersion relation.  
  evidence: `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L502–620; `morphoeval/lean/loops/Loop_3142_ceratites_nodosus_turing_wavelength_s0.lean` L10–51
- **Reproducibility package: 90** — Seed 3142, pinned dataset hashes, and --self-test now also requires discrete-band r > 0.98, both ablations to decay, and a non-default --gamma-factor to move lambda by more than 2 percent. GIF is 16 frames at 650×720.  
  evidence: `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L42–78; `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L915–931

Remediation:
- {'id': 'R5', 'item': 'Still one shell photograph. It cannot confirm a 26 µm pigment; that limit stays open.'}
- {'id': 'R6', 'item': 'Still no independent second assessor.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Ceratites nodosus Schnakenberg Turing wavenumber at 26 µm (Ladinian, 240 Ma)
- **spec_fingerprint** (semantic model + inputs): `cef6ede1e830551276a963350aa3ce5c67a0cb0dabe081d68214d961e5f27b65`
- **input_dataset_sha256**: `datasets/ceratites_nodosus_turing_wavelength_s0/README.md` (`ae965a155c4d31c3…`), `datasets/ceratites_nodosus_turing_wavelength_s0/ceratites_nodosus_specimen.jpg` (`3c37b42ed1bd3fde…`), `datasets/ceratites_nodosus_turing_wavelength_s0/ceratites_nodosus_turing_wavelength_s0_reference.json` (`256f3874e2f667a0…`), `datasets/ceratites_nodosus_turing_wavelength_s0/paper_citations.bib` (`c4b089031d26801a…`), `datasets/ceratites_nodosus_turing_wavelength_s0/scout_manifest.json` (`54469d3595f1c16c…`)
- `docs/media/ceratites_nodosus_turing_wavelength_s0_comparison.png` — artifact_sha256 `1b02be065e72e6fa…`, 490,995 B, 1848×1176 px

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion Pearson r, continuous σ(k) vs Euler mode | 0.990228 correlation | 1.0 | one-step explicit Euler on periodic Fourier modes versus analytic Turing growth rate *(unstable-band r = 0.9644; RMSE band = 0.0026)* |
| dispersion RMSE | 0.500082 1/s | 0.0 | root mean square of analytic minus measured growth rate over 30 modes *(stiff decaying modes dominate RMSE; the unstable band is tighter)* |
| selected wavelength on the 96² mantle patch | 2.6e-05 m | 2.56e-05 | radial FFT peak of the saturated Schnakenberg activator *(ratio to target = 1.0000)* |
| specimen shell contour circularity | 0.659446 4 pi A / P^2 | 1.0 | border-contrast mask on the vendored Ceratites nodosus photograph, 160×120 *(aspect ratio = 1.2667)* |
| segmented specimen foreground fraction | 0.505365 fraction | 0.7 | same mask as the circularity measurement |
| segmented specimen structure-tensor coherency | 0.521862 dimensionless | 0.25 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor inside the shell mask |
| shell stripe radial-spectrum KS distance | 0.93625 KS statistic | 0.0 | two-sample KS between radial Fourier densities of the photograph and the simulated mantle field |
| equal-diffusion ablation final/initial amplitude | 0.0 ratio | 1.0 | Schnakenberg with Dv = Du, 1500 steps, seed 3143 *(sigma_max = -0.2819)* |
| cellular wavelengths across the 32 mm shell | 1250 count | 703 | N = round(0.032 / 2.56e-05) *(successor Dactylioceras commune holds 1641; organ-scale rib holds 42)* |
| unstable-band Pearson r versus discrete Jacobian eigenvalue | 1.0 correlation | 0.98 | one Euler step on resolved modes of a 32-wavelength patch versus ln(1+dt*sigma_discrete)/dt *(RMSE = 8.432e-14; continuous-symbol band r = 0.9759 (stencil truncation, RMSE 0.0037))* |
| subcritical-domain ablation amplitude ratio | 0.050901 ratio | 1.0 | Schnakenberg on a patch 0.45 wavelengths across, so the fundamental mode is stable *(sigma(k_min) = -0.9725)* |
| specimen versus field 12-bin orientation cosine | 0.994843 cosine | 1.0 | Sobel orientation histograms (12 bins) of the shell photograph and the simulated activator *(the fossil's ribs and the cellular ocelli do not share an orientation spectrum)* |
| row-wise wavelength coefficient of variation | 0.67514 CV | 0.0 | std/mean of the dominant 1D wavelength on each row of the saturated mantle patch *(row mean = 4.148063e-05 m over 96 rows)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2. Discrete-band dispersion, subcritical-domain ablation, orientation cosine, wavelength CV, CLI gamma-factor.
