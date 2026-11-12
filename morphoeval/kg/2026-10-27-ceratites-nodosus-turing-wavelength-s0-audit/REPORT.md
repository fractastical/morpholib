# MorphoEval report — audit `2026-10-27-ceratites-nodosus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0669e59d85ea16ffd68400afdedd6a2fd1cb346e` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ceratites nodosus Schnakenberg Turing wavenumber at 26 µm (Ladinian, 240 Ma) (`models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py`) | 84 | 74 | 80 | 82 | 88 | 80 | 86 | 82.7 | 77.0 | 85.1 | 81.1 | **81.1** | faithful |

## Findings and remediation

### Ceratites nodosus Schnakenberg Turing wavenumber at 26 µm (Ladinian, 240 Ma)  —  overall 81.1 (faithful)

Pass 1. Periodic Schnakenberg activator-inhibitor on an 8-wavelength mantle patch. Diffusivities are solved so k_c sits at 2.56e-05 m. The saturated field selects that wavelength (ratio 1.000). Dispersion Pearson r against one Euler step is 0.990; full-curve RMSE is 0.50 because stiff modes are poorly resolved, and the unstable-band r is 0.964. Equal diffusion decays. A gamma sweep from 1.08 to 1.90 times onset lengthens lambda from 25.1 to 28.2 µm. The shell photograph is segmented (circularity 0.659, coherency 0.522). At fixed 26 µm, Goniatites (333 Ma, 18 mm) holds 703 cellular wavelengths, Ceratites (240 Ma, 32 mm) holds 1250, and Dactylioceras (181 Ma, 42 mm) holds 1641. The fossil does not resolve 26 µm pigment.

- **Model fidelity to source: 84** — The script linearises Schnakenberg kinetics, sets k_c^2 = sqrt(det J / (Du Dv)), and integrates the nonlinear PDE on a periodic patch. Operating Du is solved for 26 µm rather than copied from the 1.1 mm sibling. The one-step Euler dispersion test uses a large diffusive timestep, so stiff modes miss the continuous curve (RMSE 0.50).  
  evidence: `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L84–134; `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L136–252
- **Morphological realism / specimen match: 74** — The comparison figure places the vendored Muschelkalk shell beside an ocellus field at 25.60 µm. The growth GIF keeps shell luminance and carries the activator as chroma, so the fitness crop sees the fossil rather than the spots. The fossil's ribs and nodes are not a 26 µm pigment pattern.  
  evidence: `docs/media/ceratites_nodosus_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L387–428
- **Parameter expressivity: 80** — Raising the inhibitor/activator ratio from 1.08 to 1.90 times onset lengthens the selected wavelength monotonically (25.1 to 28.2 µm). Setting Dv = Du drives sigma_max to -0.28 and the amplitude to numerical zero. Gamma is a constant, not a command-line argument.  
  evidence: `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L254–304
- **Morphometric validation: 82** — Full-curve Pearson r is 0.990 and the radial FFT peak is exactly 2.56e-05 m. Specimen circularity is 0.659 against a circle null of 1, foreground fraction 0.505 against 0.70, coherency 0.522 against 0.25. Radial KS versus the photograph is 0.94, which is the expected mismatch at a scale the photo cannot resolve. Unstable-band r is only 0.964.  
  evidence: `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L318–370; `datasets/ceratites_nodosus_turing_wavelength_s0/simulation_metrics.json` L0–0
- **Source and data traceability: 88** — Six Crossref-checked DOIs are in the bib and the reference JSON. The shell file is SHA-256 pinned and copied from the campaign gallery. Wikimedia CC candidates are listed in the scout manifest. The gallery file's own license record is not in EXIF.  
  evidence: `datasets/ceratites_nodosus_turing_wavelength_s0/paper_citations.bib` L1–55; `datasets/ceratites_nodosus_turing_wavelength_s0/ceratites_nodosus_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 80** — The module docstring and the diffusivity solver name the Turing wavenumber. Lean theorems check the integer wavelength counts (1250, and 42 per organ-scale rib) and the sign of the integer Turing numerator. They are not a real-analysis proof of the dispersion relation.  
  evidence: `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L1–20; `morphoeval/lean/loops/Loop_3142_ceratites_nodosus_turing_wavelength_s0.lean` L1–51
- **Reproducibility package: 86** — Seed 3142, pinned SHA-256 of the five dataset sources, and a --self-test that checks r > 0.98, the wavelength ratio, ablation decay, sweep monotonicity, and a GIF of at least 10 frames at least 480 px. The run is deterministic given that seed.  
  evidence: `models/2026-delphi10000-ceratites-nodosus-turing-wavelength-s0.py` L42–78

Remediation:
- {'id': 'R1', 'item': 'Re-measure the dispersion with a smaller timestep so the unstable-band Pearson r exceeds 0.98 and the stiff-mode RMSE drops.'}
- {'id': 'R2', 'item': 'Add a second negative control: a domain shorter than one wavelength, where the lowest mode is stable.'}
- {'id': 'R3', 'item': 'Report a 12-bin Sobel orientation cosine between the specimen and the simulated field, and a transect bootstrap CV of the selected wavelength.'}
- {'id': 'R4', 'item': 'Expose the diffusion-ratio factor on the command line and show that it moves lambda.'}
- {'id': 'R5', 'item': 'Single photograph. The fossil cannot confirm a 26 µm pigment; that limit stays open.'}
- {'id': 'R6', 'item': 'No independent second assessor.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Ceratites nodosus Schnakenberg Turing wavenumber at 26 µm (Ladinian, 240 Ma)
- **spec_fingerprint** (semantic model + inputs): `cef6ede1e830551276a963350aa3ce5c67a0cb0dabe081d68214d961e5f27b65`
- **input_dataset_sha256**: `datasets/ceratites_nodosus_turing_wavelength_s0/README.md` (`ae965a155c4d31c3…`), `datasets/ceratites_nodosus_turing_wavelength_s0/ceratites_nodosus_specimen.jpg` (`3c37b42ed1bd3fde…`), `datasets/ceratites_nodosus_turing_wavelength_s0/ceratites_nodosus_turing_wavelength_s0_reference.json` (`256f3874e2f667a0…`), `datasets/ceratites_nodosus_turing_wavelength_s0/paper_citations.bib` (`c4b089031d26801a…`), `datasets/ceratites_nodosus_turing_wavelength_s0/scout_manifest.json` (`54469d3595f1c16c…`)
- `datasets/ceratites_nodosus_turing_wavelength_s0/simulation_metrics.json` — artifact_sha256 `4589e1b86a4e3aed…`, 5,480 B, 
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

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 of slot 3142. Schnakenberg wavenumber at 26 µm on a periodic mantle patch.
