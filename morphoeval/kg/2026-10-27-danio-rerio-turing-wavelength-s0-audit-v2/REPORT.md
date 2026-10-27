# MorphoEval report — audit `2026-10-27-danio-rerio-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-danio-rerio-turing-wavelength-s0-audit -> 2026-10-27-danio-rerio-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Danio rerio Holocene 24.08 µm anisotropic Schnakenberg wavelength (pass 2) (`models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py`) | 87 | 80 | 90 | 88 | 92 | 86 | 90 | 88.0 | 83.0 | 89.6 | 86.5 | **86.5** | faithful |

## Findings and remediation

### Danio rerio Holocene 24.08 µm anisotropic Schnakenberg wavelength (pass 2)  —  overall 86.5 (faithful)

Pass 2 keeps the same Schnakenberg system (a=0.11, b=1.24, Du=0.80, Dv_y=20.80, gamma=0.1989, anisotropy 3.85). The GIF pigment is now a steep sigmoid of the activator, colored with the specimen dark and light palette, and the comparison figure shows that fish-mask composite beside the photograph. Dispersion growth rates are measured on the 11 exact dorsoventral Fourier modes: Pearson r = 0.9805, RMSE = 0.0191, KS D = 0.273 against a 0.40 ceiling. Selected wavelength remains 24.08 µm. Computer-vision similarity of the 960×540, 12-frame GIF against the photograph is 77.8.

- **Model fidelity to source: 87** — The integrator and the analytical eigenvalue are unchanged from pass 1. Pass 2 measures growth on the exact dorsoventral DFT modes of the window, which are the eigenfunctions of the discrete Laplacian the solver uses. gamma remains a scale calibration rather than a transcribed Nakamasu parameter.  
  evidence: `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L85–94; `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L179–190
- **Morphological realism / specimen match: 80** — The comparison figure now places the fish-mask composite next to the photograph, and the GIF stripes are bimodal in the specimen dark/light palette rather than a gray sinusoid. Computer-vision similarity rose from 74.5 to 77.8. The eye, blue iridescence, and the adult count of four or five stripes are still absent; the composite shows the six wavelengths of the 144.48 µm window.  
  evidence: `docs/media/danio_rerio_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L270–286; `docs/media/danio_rerio_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 90** — Gradient anisotropy rises monotonically across inhibitor ratios 1.12, 2.0, 3.85, and 6.0 (about 1.2 to 6.0). The predecessor sits at the isotropic end and the equal-diffusion ablation stays at numerical roundoff. The sweep is the same experiment as pass 1 and is plotted with both chronospecies marked.  
  evidence: `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L386–420; `docs/media/danio_rerio_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 88** — On the 11 exact modes, Pearson r = 0.9805 and RMSE = 0.0191. The KS statistic is 0.273 against a ceiling of 0.40, so the pass-1 coincidence of D with its own reference is gone. GIF coherency 0.4399 is not the specimen coherency 0.3887, and the predecessor wavelength is 20.64 µm against the rerio target 24.08 µm. The correlation clears 0.98 with little room; one low-k mode still sits off the analytical curve.  
  evidence: `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L179–232; `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L448–530
- **Source and data traceability: 92** — DOIs for Turing 1952, Nakamasu 2009, Hiscock & Megason 2015, Quigley 2004, and Patterson 2014 were checked on Crossref and written to the BibTeX. Specimen, scout, README, and BibTeX SHA-256 pins are verified before integration. The scout manifest records that slot 3164 is outside the 100-slot campaign file.  
  evidence: `datasets/danio_rerio_turing_wavelength_s0/paper_citations.bib` L1–84; `datasets/danio_rerio_turing_wavelength_s0/danio_rerio_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 86** — The docstring, reference JSON, math-KG laws, and Lean theorems name the same Jacobian margins and the 3.85 versus 1.12 anisotropy split. Pass 2's sigmoid is commented as a display map, not a change to the PDE. Lean integers are rounded to 0.01.  
  evidence: `morphoeval/lean/loops/Loop_3164_danio_rerio_turing_wavelength_s0.lean` L14–47; `morphoeval/math_kg_loops/delphi10000-danio-rerio-turing-wavelength-s0.yaml` L1–55
- **Reproducibility package: 90** — --self-test still requires Pearson r > 0.98, RMSE < 0.02, wavelength error below 0.75 µm, a dead equal-diffusion control, three seeds, a rising anisotropy sweep, and a GIF of at least 10 frames and 480 px. The pass-2 command completes in about 11 s from a fixed seed.  
  evidence: `models/2026-delphi10000-danio-rerio-turing-wavelength-s0.py` L549–564

Remediation:
- No second assessor has scored this pass.
- gamma = 0.1989 is calibrated to the assigned 24.08 µm scale rather than copied from a published table.
- The composite has no eye and no iridescent blue; the stripe count is the cellular window, not the adult flank.
- Nanopubs are unsigned because ~/.nanopub/profile.yml is absent.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Danio rerio Holocene 24.08 µm anisotropic Schnakenberg wavelength (pass 2)
- **spec_fingerprint** (semantic model + inputs): `d7d1a6ee4915a8532ac3ba145d5dd7b6436e0f7006bd83ec8bd0ab39b939e0f7`
- **input_dataset_sha256**: `datasets/danio_rerio_turing_wavelength_s0/README.md` (`dbe49b6901b94381…`), `datasets/danio_rerio_turing_wavelength_s0/danio_rerio_specimen.jpg` (`850b2d4a94c7cbcb…`), `datasets/danio_rerio_turing_wavelength_s0/danio_rerio_turing_wavelength_s0_reference.json` (`8456a78d5a6e1822…`), `datasets/danio_rerio_turing_wavelength_s0/paper_citations.bib` (`e77f417b131e9ec8…`), `datasets/danio_rerio_turing_wavelength_s0/scout_manifest.json` (`7dfbf8b27236cb36…`)
- `docs/media/danio_rerio_turing_wavelength_s0_comparison.png` — artifact_sha256 `71b3259f10bb6add…`, 386,615 B, 1716×924 px
- `docs/media/danio_rerio_turing_wavelength_s0_growth.gif` — artifact_sha256 `2f1a761269b6d515…`, 271,334 B, 960×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion-curve Pearson r of PDE growth versus analytical sigma(k) | 0.980483 Pearson r | 0.98 | log-amplitude growth between t=2 and t=14 versus sigma_mode on the 11 exact DV Fourier modes *(RMSE = 0.019055)* |
| dispersion RMSE between analytical and numerical growth rates | 0.019055 1/time | 0.02 | root mean square of sigma_analytical - sigma_numerical |
| Kolmogorov-Smirnov statistic of analytical versus numerical growth-rate samples | 0.272727 KS D | 0.4 | scipy.stats.ks_2samp on the 11 exact-mode sigma samples *(two-sided p = 0.8326)* |
| selected dorsoventral wavelength of the Holocene Danio rerio field | 24.08 µm | 24.08 | argmax of the x-averaged Fourier profile on the 144.48 µm window *(six wavelengths span Ly when the mode index is 6)* |
| predecessor Danio nigrofasciatus dominant wavelength at anisotropy 1.12 | 20.64 µm | 24.08 | same grid and seed, inhibitor anisotropy 1.12, no phase cue *(predecessor gradient anisotropy = 1.181)* |
| equal-diffusion negative-control field amplitude | 0.0 concentration std | 0.0 | Dv_x = Dv_y = Du = 0.80, same gamma, 800 steps |
| specimen-image structure-tensor coherency of the rendered GIF | 0.4396 coherency | 0.3887 | Sobel structure tensor on the last GIF frame *(simulation and specimen values are measured separately)* |
| segmented specimen foreground fraction versus GIF occupancy | 0.4107 area fraction | 0.4104 | fraction of GIF pixels darker than 0.90 after fish-mask rendering *(specimen aspect ratio (length/height) = 2.347)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2 working tree. ~/.nanopub/profile.yml is absent, so nanopubs are built and conformance-checked but not signed.
