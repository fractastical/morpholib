# MorphoEval report — audit `2026-10-27-drosophila-eugracilis-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 1.5 µm in Drosophila eugracilis (Pass 1 focal run) (`models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py`) | 86 | 72 | 60 | 66 | 90 | 84 | 82 | 77.3 | 69.8 | 85.2 | 76.7 | **76.7** | mostly faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 1.5 µm in Drosophila eugracilis (Pass 1 focal run)  —  overall 76.7 (mostly faithful)

Pass 1 integrates Schnakenberg kinetics at gamma=1 on a 128^2 periodic grid (12 focal wavelengths, 17.820 µm window) and fixes the length unit so the analytic lambda_c is 1.485 µm. The saturated field is an ocellar spot lattice (activator std 1.26, 93 foci). Analytic lambda 1.485 µm versus FFT lambda 1.809 µm. Raw Otsu arm fraction on the gallery plate is 0.141 and KS luminance distance to the render is 0.343. No gamma sweep, no equal-diffusion control, and no series Pearson/RMSE are stored in pass1_metrics.json.

- **Model fidelity to source: 86** — Pass 1 implements the Schnakenberg PDE and the Turing wavenumber formula, and the gamma=1 length unit hits 1.485 µm by construction. The nonlinear FFT wavelength (1.809 µm) is not yet reconciled with k_c across a parameter series.  
  evidence: `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L79–111; `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L139–195
- **Morphological realism / specimen match: 72** — The 12-frame 720x540 GIF ends in a brown ocellar lattice rendered with the plate LUT. The plate itself is a spiral, so the geometry does not match even though both fill the frame. The plate is the gallery fallback, not a photograph of the fly.  
  evidence: `docs/media/drosophila_eugracilis_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L247–257
- **Parameter expressivity: 60** — The dispersion panel draws three gamma curves, but pass1_metrics.json stores only gamma=1. Monotonic control of the nonlinear wavelength by gamma is not yet measured, and the equal-diffusion control is not in this pass.  
  evidence: `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L127–129; `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L652–653
- **Morphometric validation: 66** — Single-point comparison only: analytic 1.485 µm versus FFT 1.809 µm, raw Otsu fraction 0.141 (reference fitness-script fraction 0.354), KS 0.343, focus spacing CV 0.294. No Pearson r, RMSE series, or ablation contrast in the pass-1 record.  
  evidence: `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L197–223; `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L259–288; `datasets/drosophila_eugracilis_turing_wavelength_s0/pass1_metrics.json` L0–0
- **Source and data traceability: 90** — Nine Crossref-checked DOIs are in paper_citations.bib. The sibling primordium DOI 10.1016/j.ydbio.2009.04.012 was rejected because Crossref shows it is a mouse germ-cell paper. Wikimedia returned no D. eugracilis bitmap; the gallery fallback and its SHA-256 are recorded.  
  evidence: `datasets/drosophila_eugracilis_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/drosophila_eugracilis_turing_wavelength_s0/scout_manifest.json` L0–0
- **Code-to-model traceability: 84** — The module docstring and jacobian docstring name Turing 1952, Schnakenberg 1979 and Murray gamma. Lean Loop_3184 has six theorems and no sorry, encoding the 2100 > 1485 > 1180 nm order. Pass 1 does not yet exercise the mode-count comparison on simulated fields.  
  evidence: `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L1–28; `morphoeval/lean/loops/Loop_3184_drosophila_eugracilis_turing_wavelength_s0.lean` L20–43
- **Reproducibility package: 82** — Dataset SHA-256 locks, seed 3184, and --self-test are present. Pass 1 itself does not record the sweep that --self-test checks, and there is no pinned per-script requirements file beyond the repo requirements.txt.  
  evidence: `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L64–77; `models/2026-delphi10000-drosophila-eugracilis-turing-wavelength-s0.py` L745–759

Remediation:
- {'id': 'R1', 'item': 'Run a gamma sweep spanning D. erecta (0.50), D. eugracilis (1), and D. ficusphila, and record analytic versus FFT wavelength.'}
- {'id': 'R2', 'item': 'Add the equal-diffusion negative control D_u = D_v and report final activator std against the Turing run.'}
- {'id': 'R3', 'item': 'Replace the grain-dominated raw Otsu perimeter with a blurred arm segmentation and keep value distinct from the reference.'}
- {'id': 'R4', 'item': 'Report series goodness-of-fit: Pearson r, RMSE, and the plate-versus-render KS statistic, with value not equal to reference.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 1.5 µm in Drosophila eugracilis (Pass 1 focal run)
- **spec_fingerprint** (semantic model + inputs): `4cd159ca0e54b6b9882667ea66ee1c36e85c25ecfd3d9cd25e35efca3e35cc7f`
- **input_dataset_sha256**: `datasets/drosophila_eugracilis_turing_wavelength_s0/drosophila_eugracilis_turing_wavelength_s0_reference.json` (`aeae6e2ee094bbd3…`), `datasets/drosophila_eugracilis_turing_wavelength_s0/drosophila_eugracilis_turing_wavelength_s0_specimen.jpg` (`f3705e558598c7f5…`), `datasets/drosophila_eugracilis_turing_wavelength_s0/paper_citations.bib` (`963fd37391fe7bc7…`)
- `datasets/drosophila_eugracilis_turing_wavelength_s0/pass1_metrics.json` — artifact_sha256 `540dfdb9eebf0598…`, 1,180 B, 
- `datasets/drosophila_eugracilis_turing_wavelength_s0/scout_manifest.json` — artifact_sha256 `65cbd08f09bd7095…`, 4,666 B, 
- `docs/media/drosophila_eugracilis_turing_wavelength_s0_comparison.png` — artifact_sha256 `6d16c18dee30d5d1…`, 476,053 B, 1792×1148 px
- `docs/media/drosophila_eugracilis_turing_wavelength_s0_growth.gif` — artifact_sha256 `b571e2432db179c4…`, 2,461,602 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| analytic_turing_wavelength_um | 1.485 µm | 1.808868 | Schnakenberg dispersion k_c at gamma=1, length unit fixed to the assigned 1.485 µm; reference is the nonlinear FFT wavelength of the same run |
| simulated_stripe_fft_wavelength_um | 1.808868 µm | 1.485 | Radial Fourier ring of the saturated activator field, parabolic peak, converted by the focal length unit |
| predecessor_erecta_wavelength_um | 2.100107 µm | 1.485 | lambda(gamma=0.5) = 1.485/sqrt(0.5); predecessor contrast, not a published erecta SEM |
| successor_ficusphila_wavelength_um | 1.18 µm | 1.485 | Assigned successor wavelength 1.180 µm used to set gamma_ficusphila |
| specimen_stripe_arm_foreground_fraction | 0.140679 1 | 0.3543 | Otsu dark-arm fraction of the vendored gallery plate; reference is verify_image_fitness foreground_fraction (different estimator) |
| specimen_structure_tensor_coherency | 0.205514 1 | 0.8014 | Mean structure-tensor coherency of the full plate; reference is the fitness-script crop signature |
| specimen_vs_simulation_ks_luminance | 0.342917 1 | 0.0 | Two-sample KS statistic between plate luminance and the rendered activator; 0 would be identical histograms |
| ocellus_nearest_neighbor_spacing_cv | 0.29424 1 | 0.0 | CV of periodic nearest-neighbour distances among high-activator foci; 0 is a perfect lattice |

## Environment at checkpoint


Pass 1 working tree. Media paths are later overwritten by pass 2; pass1_metrics.json keeps this pass.
