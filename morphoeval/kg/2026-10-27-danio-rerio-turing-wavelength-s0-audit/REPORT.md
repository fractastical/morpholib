# MorphoEval report — audit `2026-10-27-danio-rerio-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Danio rerio Holocene 24.08 µm anisotropic Schnakenberg wavelength (pass 1) (`models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py`) | 86 | 74 | 88 | 86 | 90 | 84 | 88 | 86.7 | 78.5 | 87.6 | 83.7 | **83.7** | faithful |

## Findings and remediation

### Danio rerio Holocene 24.08 µm anisotropic Schnakenberg wavelength (pass 1)  —  overall 83.7 (faithful)

Pass 1 integrates an anisotropic Schnakenberg system whose dorsoventral dispersion peak sits on k_c = 2π/24.08 µm. Inhibitor anisotropy 3.85 (extant D. rerio, 0 Ma) produces horizontal stripes; anisotropy 1.12 (predecessor D. nigrofasciatus, same cellular scale) does not. Dispersion Pearson r = 0.9975, RMSE = 0.0192, selected wavelength = 24.08 µm on three noise seeds. The GIF is a 12-frame, 960×540 magnification of the 144.48 µm window inside the segmented specimen outline (CV similarity 74.5). The rendered stripes are still a gray sinusoid, and the comparison figure does not show that composite.

- **Model fidelity to source: 86** — sigma_mode is the larger eigenvalue of gamma*J - k^2 D for the Schnakenberg Jacobian, and integrate steps that operator with spectral diffusion. Anisotropy keeps Dv_y on the calibrated peak and lowers Dv_x. gamma is a scale calibration, not a number copied from Nakamasu et al. 2009.  
  evidence: `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L85–94; `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L113–161
- **Morphological realism / specimen match: 74** — The GIF uses the segmented adult outline and draws horizontal stripes, and the computer-vision similarity against the photograph is 74.5. The stripes are a gray sinusoid without the specimen's blue-black and silver-gold bands, the eye is absent, and the comparison PNG shows the unwrapped field rather than the fish composite.  
  evidence: `docs/media/danio_rerio_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L254–286; `docs/media/danio_rerio_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 88** — The anisotropy sweep 1.12, 2.0, 3.85, 6.0 raises gradient anisotropy from 1.20 to 6.00 monotonically, and the equal-diffusion control collapses to amplitude ~1e-11. The predecessor regime is the low end of the same sweep.  
  evidence: `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L358–390; `docs/media/danio_rerio_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 86** — PDE growth versus analytical sigma(k) has Pearson r = 0.9975 and RMSE = 0.0192. The rendered-GIF coherency (0.4399) is not the specimen coherency (0.3887). The predecessor wavelength on the dedicated run is 20.64 µm against the rerio 24.08 µm target. The KS sample has only 12 points, so D = 0.25 sits on the reference ceiling and is a weak goodness-of-fit statement.  
  evidence: `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L179–227; `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L420–505
- **Source and data traceability: 90** — Five Crossref-checked DOIs are in the audit and in paper_citations.bib. The specimen, scout manifest, README, and BibTeX are SHA-256 pinned and checked at runtime. Slot 3164 was not in the 100-slot campaign file; the scout query and that fact are recorded in the manifest.  
  evidence: `datasets/danio_rerio_turing_wavelength_s0/paper_citations.bib` L1–70; `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L101–111
- **Code-to-model traceability: 84** — The module docstring states the PDE, the 24.08 µm peak condition, and the predecessor anisotropy. Lean theorems check the integer Turing margins for both axes and both species. The Lean integers are rounded to 0.01, and gamma's calibration is described in prose rather than derived in a cited equation number from Nakamasu 2009.  
  evidence: `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L1–32; `morphoeval/lean/loops/Loop_3164_danio_rerio_turing_wavelength_s0.lean` L14–44
- **Reproducibility package: 88** — --self-test checks Pearson r > 0.98, RMSE < 0.02, wavelength, the equal-diffusion control, GIF geometry, three seeds, and the sweep direction. Seed 3164 and dt = 0.25 are fixed. The run completes in about 12 s. There is no second machine record and no signed nanopub.  
  evidence: `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L520–536

Remediation:
- Render the comparison PNG with the fish-mask composite beside the photograph, and sharpen the GIF pigment so stripes use the specimen dark and light palette instead of a gray sinusoid.
- Recompute the Kolmogorov-Smirnov statistic on a finer wavenumber grid so D is not forced onto 0.25 by a 12-point sample.
- A second assessor on a different model has not scored this audit.
- gamma = 0.1989 is a scale calibration to 24.08 µm, not a value transcribed from a published table.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Danio rerio Holocene 24.08 µm anisotropic Schnakenberg wavelength (pass 1)
- **spec_fingerprint** (semantic model + inputs): `d7d1a6ee4915a8532ac3ba145d5dd7b6436e0f7006bd83ec8bd0ab39b939e0f7`
- **input_dataset_sha256**: `datasets/danio_rerio_turing_wavelength_s0/README.md` (`dbe49b6901b94381…`), `datasets/danio_rerio_turing_wavelength_s0/danio_rerio_specimen.jpg` (`850b2d4a94c7cbcb…`), `datasets/danio_rerio_turing_wavelength_s0/danio_rerio_turing_wavelength_s0_reference.json` (`8456a78d5a6e1822…`), `datasets/danio_rerio_turing_wavelength_s0/paper_citations.bib` (`e77f417b131e9ec8…`), `datasets/danio_rerio_turing_wavelength_s0/scout_manifest.json` (`7dfbf8b27236cb36…`)
- `docs/media/danio_rerio_turing_wavelength_s0_comparison.png` — artifact_sha256 `bce4762d3451a391…`, 469,820 B, 1488×1032 px
- `docs/media/danio_rerio_turing_wavelength_s0_growth.gif` — artifact_sha256 `edf0c48b8af20dfe…`, 353,698 B, 960×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion-curve Pearson r of PDE growth versus analytical sigma(k) | 0.997544 Pearson r | 0.98 | log-amplitude growth between t=8 and t=28 versus sigma_mode on 12 wavenumbers *(RMSE = 0.019188)* |
| dispersion RMSE between analytical and numerical growth rates | 0.019188 1/time | 0.02 | root mean square of sigma_analytical - sigma_numerical |
| Kolmogorov-Smirnov statistic of analytical versus numerical growth-rate samples | 0.25 KS D | 0.25 | scipy.stats.ks_2samp on the 12-point sigma samples *(two-sided p = 0.869)* |
| selected dorsoventral wavelength of the Holocene Danio rerio field | 24.08 µm | 24.08 | argmax of the x-averaged Fourier profile on the 144.48 µm window *(six wavelengths span Ly when the mode index is 6)* |
| predecessor Danio nigrofasciatus dominant wavelength at anisotropy 1.12 | 20.64 µm | 24.08 | same grid and seed, inhibitor anisotropy 1.12, no phase cue *(predecessor gradient anisotropy = 1.181)* |
| equal-diffusion negative-control field amplitude | 0.0 concentration std | 0.0 | Dv_x = Dv_y = Du = 0.80, same gamma, 800 steps |
| specimen-image structure-tensor coherency of the rendered GIF | 0.4399 coherency | 0.3887 | Sobel structure tensor on the last GIF frame *(simulation and specimen values are measured separately)* |
| segmented specimen foreground fraction versus GIF occupancy | 0.4107 area fraction | 0.4104 | fraction of GIF pixels darker than 0.90 after fish-mask rendering *(specimen aspect ratio (length/height) = 2.347)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 1 working tree on cursor/danio-rerio-turing-wavelength-1215. Signing unavailable: ~/.nanopub/profile.yml is absent.
