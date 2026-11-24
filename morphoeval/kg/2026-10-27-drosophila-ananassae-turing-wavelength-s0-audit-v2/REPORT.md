# MorphoEval report — audit `2026-10-27-drosophila-ananassae-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `94b693bd2928b4d8996239b3f400b320fcf30723` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-ananassae-turing-wavelength-s0-audit -> 2026-10-27-drosophila-ananassae-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila ananassae Turing wavenumber at 2.1 µm (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py`) | 90 | 86 | 88 | 90 | 86 | 84 | 86 | 89.3 | 87.5 | 85.5 | 87.4 | **87.4** | faithful |

## Findings and remediation

### Drosophila ananassae Turing wavenumber at 2.1 µm (Holocene, 0 Ma)  —  overall 87.4 (faithful)

Pass 2 keeps the Schnakenberg field (a=0.11, b=1.15, Dv/Du=19.7) at 2.106 µm. Scaling both diffusivities by s² moves the FFT wavelength with the linear prediction (Pearson r = 0.99483, RMSE = 8.05e-8 m, KS = 0.167, six scales). Holding Du fixed, λ rises from 1.75 µm at Dv/Du = 8 to 2.26 µm at Dv/Du = 28, and σ changes from negative to positive across that grid. Equal diffusion and linear kinetics, integrated for 2000 steps, decay (std ratios 2.0e-4 and 6.5e-9) while the nominal system grows (1.73). The plate and the GIF last frame are segmented separately: circularity 0.136 versus 0.018, gray-level KS 0.222. Computer-vision similarity of the 12-frame 960×640 GIF to the plate is 94.3.

- **Model fidelity to source: 90** — The integrator is the Schnakenberg PDE. Removing either the diffusion difference or the nonlinear term, at the same seed and step count, collapses the activator variance, which is the integrated form of the Turing condition already checked on σ(k).  
  evidence: `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L79–124; `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L362–372
- **Morphological realism / specimen match: 86** — The GIF last frame is the public-domain plate with a cuticle-only pigment modulation. CV similarity to that plate is 94.3 at 960×640 and 12 frames. Segmentation shows the render is not a pixel copy (circularity 0.018 versus 0.136 on the plate; gray KS 0.222). The plate is a 225×112 illustration, so this is not a match to a 2.1 µm micrograph.  
  evidence: `docs/media/drosophila_ananassae_turing_wavelength_s0_comparison.png` L0–0; `docs/media/drosophila_ananassae_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L419–442
- **Parameter expressivity: 88** — Multiplying both diffusivities by s² moves analytic and FFT wavelengths together across s = 0.75 to 1.50. Holding Du fixed, raising Dv/Du from 8 to 28 lengthens λ from 1.75 µm to 2.26 µm and carries σ from negative (no Turing) to positive.  
  evidence: `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L317–360; `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L626–630
- **Morphometric validation: 90** — Pearson r = 0.99483 and RMSE = 8.05e-8 m on six scales, with KS statistic 0.167 between the analytic and FFT samples. Plate circularity 0.136 is scored against a circle null of 1, and the render coherency gap is scored against 0. The same sweep is used to fit and to score.  
  evidence: `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L479–600; `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L626–634
- **Source and data traceability: 86** — Eight Crossref-checked DOIs, the Wikimedia public-domain plate URL, and SHA-256 pins checked at runtime. OpenAlex returned no works for the slot query; that empty result is kept in the scout manifest.  
  evidence: `datasets/drosophila_ananassae_turing_wavelength_s0/paper_citations.bib` L1–84; `datasets/drosophila_ananassae_turing_wavelength_s0/drosophila_ananassae_turing_wavelength_s0_reference.json` L77–86; `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L67–77
- **Code-to-model traceability: 84** — Each pass-2 control is its own function (scale_sweep, ratio_sweep, integrated_ablations, render_versus_plate), and the Lean file states the 2106 nm wavelength and the milli-unit h-condition. Paper equation numbers are not pinned, and the Lean kernel was not run.  
  evidence: `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L317–442; `morphoeval/lean/loops/Loop_3176_drosophila_ananassae_turing_wavelength_s0.lean` L24–40
- **Reproducibility package: 86** — Seed 3176, pinned hashes, and --self-test now fails unless r > 0.98 and both ablations decay. The run is deterministic. No Lean binary is installed, so the theorems were not kernel-checked.  
  evidence: `models/2026-delphi10000-drosophila-ananassae-turing-wavelength-s0.py` L604–648; `datasets/drosophila_ananassae_turing_wavelength_s0/README.md` L1–29

Remediation:
- Kernel-check the Lean file when a Lean 4 binary is available.
- A second assessor has not scored this audit.
- The vendored plate is a 225×112 public-domain illustration, not a micrograph resolved at 2.1 µm.
- The wavelength Pearson coefficient is computed on the same six scales used to choose the comparison, so it is a fit rather than a held-out test.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila ananassae Turing wavenumber at 2.1 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `cb48fb2dc3693d0cac7caeab627f7cf256ca041c37d25c1fbbe642b593156101`
- **input_dataset_sha256**: `datasets/drosophila_ananassae_turing_wavelength_s0/drosophila_ananassae_specimen.png` (`e3f07821c201b2b5…`), `datasets/drosophila_ananassae_turing_wavelength_s0/drosophila_ananassae_turing_wavelength_s0_reference.json` (`2768c7dcb983c315…`)
- `docs/media/drosophila_ananassae_turing_wavelength_s0_comparison.png` — artifact_sha256 `0fb2a26b4b6381d5…`, 549,266 B, 1200×800 px
- `docs/media/drosophila_ananassae_turing_wavelength_s0_growth.gif` — artifact_sha256 `87c0dd9a019df28b…`, 3,444,176 B, 960×640 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| fft_wavelength_m | 2.0301e-06 m | 2.106e-06 | dominant non-DC Fourier mode of the saturated activator, times dx |
| specimen_contour_circularity | 0.136 1 | 1.0 | 4πA/P² on the border-contrast mask of the Wikimedia plate |
| specimen_aspect_ratio | 2.018 1 | 1.0 | bounding-box width/height of the plate foreground |
| wavelength_ratio_americana_over_s0 | 65.8594 1 | 1.0 | 139 µm predecessor / 2.106 µm detail wavelength |
| growth_gif_frame_count | 12 frames | 6 | PIL n_frames of the rendered GIF |
| growth_gif_min_resolution_px | 640 px | 340 | min(width, height) of the GIF |
| pearson_r_wavelength_law | 0.99483 1 | 1.0 | Pearson r of analytic λ(s) versus FFT λ(s) on six diffusion scales |
| rmse_wavelength_m | 8.054e-08 m | 0.0 | RMSE of FFT wavelength against the linear-theory wavelength |
| ks_statistic_wavelength | 0.1667 1 | 0.0 | two-sample KS statistic between analytic and FFT wavelength samples |
| equal_diffusion_std_ratio | 0.0002 1 | 1.7262 | final/initial activator std with Du = Dv, 2000 steps |
| specimen_render_coherency_gap | 0.0099 1 | 0.0 | absolute gap in structure-tensor coherency, plate versus GIF last frame |

## Environment at checkpoint


Pass 2 adds the diffusion-scale sweep, ratio sweep, integrated ablations, and plate-versus-render segmentation. publisher_orcid is the creator URI already used on this repository's MorphoEval nanopubs. ~/.nanopub/profile.yml is absent, so these nanopubs are not signed. Lean was not kernel-executed.
