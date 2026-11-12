# MorphoEval report — audit `2026-10-27-dasypus-novemcinctus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-dasypus-novemcinctus-turing-wavelength-s0-audit -> 2026-10-27-dasypus-novemcinctus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength at 4.108 µm on Holocene Dasypus novemcinctus dermis (Pass 2) (`models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py`) | 90 | 80 | 90 | 86 | 92 | 86 | 90 | 90.0 | 82.2 | 89.6 | 86.8 | **86.8** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength at 4.108 µm on Holocene Dasypus novemcinctus dermis (Pass 2)  —  overall 86.8 (faithful)

Pass 2 overall 86.8/100, superseding the Pass 1 audit at 77.2. The diffusion-scale sweep tracks the analytic wavelength with Pearson r = 0.999999 and RMSE = 0.524 µm. Equal diffusivities collapse the activator. The specimen mask is non-circular (circularity 0.144 versus 1) and the dorsal Sobel counter returns 12 peaks versus the nominal nine bands. Chronoseries length is 1 at 0 Ma; 66 detail wavelengths fit in the 273 µm primordium interval. No second assessor.

- **Model fidelity to source: 90** — The script integrates Schnakenberg kinetics with the Turing wavenumber identity and an equal-diffusion ablation that removes the pattern. Parameters are still reconstructed for the 4.108 µm rung rather than fitted to a Dasypus reaction-diffusion experiment.  
  evidence: `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L79–126; `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L128–167; `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L342–348; `morphoeval/lean/loops/Loop_3166_dasypus_novemcinctus_turing_wavelength_s0.lean` L20–54
- **Morphological realism / specimen match: 80** — The comparison figure shows the La Fortuna photograph, the activator, an overlay, and the sweep. The animation is 12 frames at 640×480. The render keeps the photograph's mid-frequency edges, and the photo is not a 4.1 µm micrograph, so this is not a pixel-level histological match.  
  evidence: `docs/media/dasypus_novemcinctus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/dasypus_novemcinctus_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L386–448
- **Parameter expressivity: 90** — Six diffusion scales from 0.64 to 1.69 move the measured wavelength monotonically with the analytic square-root law. The equal-diffusion control is the off state.  
  evidence: `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L312–340; `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L265–310
- **Morphometric validation: 86** — Pearson r = 0.999999 and RMSE = 0.524 µm on the sweep. Replicate CV = 0.029. KS statistic between measured and analytic samples is 0.333 with p = 0.93 at n = 6, so the offset is visible but not a significant two-sample rejection. Specimen circularity 0.144 versus 1, band count 12 versus 9, spacing 72 px versus 0. The band counter is a Sobel heuristic, not a histological count.  
  evidence: `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L360–384; `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L494–605
- **Source and data traceability: 92** — Real DOIs for Chen, Vickaryous, Turing, Schnakenberg, and Murray. Specimen SHA-256, creator Rhododendrites, CC BY-SA 4.0, La Fortuna, 2024-12-11.  
  evidence: `datasets/dasypus_novemcinctus_turing_wavelength_s0/paper_citations.bib` L1–75; `datasets/dasypus_novemcinctus_turing_wavelength_s0/dasypus_novemcinctus_turing_wavelength_s0_reference.json` L1–71; `datasets/dasypus_novemcinctus_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 86** — Equations 1-3 are commented on the coefficient function. The sweep, ablation, and fit functions name the Pass 1 remediation they close. Lean checks the integer packing of 66 and the discriminant.  
  evidence: `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L79–116; `morphoeval/math_kg_loops/delphi10000-dasypus-novemcinctus-turing-wavelength-s0.yaml` L12–55
- **Reproducibility package: 90** — Seed 3166, four replicate seeds, pinned specimen digest, and --self-test requiring Pearson r > 0.98, a collapsed ablation, packing 66, and a 12-frame GIF of at least 480 px. Dependencies are the repo scientific stack.  
  evidence: `models/2026-delphi10000-dasypus-novemcinctus-turing-wavelength-s0.py` L607–651

Remediation:
- {'id': 'R5', 'item': 'Independent second assessor. Both passes are single-assessor (Grok 4.7).', 'status': 'open'}
- {'id': 'R7', 'item': 'Dorsal Sobel band count is 12 against the nominal nine movable bands. The counter is not a manual shield census.', 'status': 'open'}
- {'id': 'R8', 'item': 'Saturated wavelength remains about 1.12 times the linear 4.108 µm (RMSE 0.524 µm) even though the sweep correlation is above 0.98.', 'status': 'open'}
- {'id': 'R9', 'item': 'No 4.1 µm histological micrograph of Dasypus dermis is in the dataset. The specimen is an organism-scale photograph.', 'status': 'open'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength at 4.108 µm on Holocene Dasypus novemcinctus dermis (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `0ed1ba81083be6f146345ad2f0ef47d7ab333edbbcd8e4d2c467f1b152acb7c7`
- **input_dataset_sha256**: `datasets/dasypus_novemcinctus_turing_wavelength_s0/dasypus_novemcinctus_specimen.jpg` (`7f1ba0e3ab8bdb13…`), `datasets/dasypus_novemcinctus_turing_wavelength_s0/dasypus_novemcinctus_turing_wavelength_s0_reference.json` (`e8ce5113e12fa07c…`)
- `docs/media/dasypus_novemcinctus_turing_wavelength_s0_comparison.png` — artifact_sha256 `aeacb08e61f21379…`, 1,008,876 B, 1248×984 px
- `docs/media/dasypus_novemcinctus_turing_wavelength_s0_growth.gif` — artifact_sha256 `41326f640f21a2ef…`, 2,752,116 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| saturated_activator_wavelength_um | 4.5617 µm | 4.108 | Parabolic peak of the radial Fourier power of the saturated activator *(Finite-amplitude spots sit long of linear k_c. value != reference.)* |
| sweep_pearson_r | 0.999999 1 | 0.98 | Pearson correlation of measured versus analytic wavelength across six diffusion scales *(Reference is the gate, not the measurement.)* |
| sweep_rmse_um | 0.5243 µm | 0.0 | RMSE of measured versus analytic wavelength on the diffusion-scale sweep *(The offset is the nonlinear red-shift, reported rather than removed.)* |
| wavelength_replicate_cv | 0.0293 1 | 0.0 | Sample CV of the saturated wavelength across four RNG seeds *(Non-zero CV is the replicate scatter.)* |
| ablation_ks_statistic | 0.3333 1 | 0.0 | Two-sample KS statistic between measured and analytic wavelengths on the sweep *(n = 6, so this is a distance, not a precise p-value.)* |
| ablation_amplitude_ratio | 51196.6493 1 | 1.0 | std(u) with D_v/D_u = 20 divided by std(u) with D_u = D_v *(The patterned state must be much larger than the ablated state.)* |
| specimen_foreground_fraction | 0.3502 1 | 1.0 | Largest warm-tone component on the CC BY-SA photograph after moss rejection *(The animal does not fill the frame.)* |
| specimen_contour_circularity | 0.1436 1 | 1.0 | 4 pi Area / Perimeter^2 of the smoothed foreground mask *(The carapace silhouette is not circular.)* |
| specimen_structure_tensor_coherency | 0.4108 1 | 0.0 | Mean structure-tensor coherency on the foreground mask *(Bands and tile edges raise coherency above 0.)* |
| specimen_movable_band_count | 12 count | 9 | Peaks in the vertical Sobel profile of the dorsal carapace crop *(The peak counter is not a manual band census. value is whatever the profile yields.)* |
| specimen_osteoderm_spacing_px | 72.0 px | 0.0 | First autocorrelation peak beyond 8 px on the shoulder-tile crop *(Spacing is in photo pixels and is not converted to 4.1 µm.)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2. Diffusion-scale sweep, equal-diffusion ablation, specimen segmentation, Pearson/RMSE/KS/CV.
