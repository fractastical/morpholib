# MorphoEval report — audit `2026-10-27-baculites-compressus-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `8f0ebbb7583b60dab933158a86eda8e291baab3e` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength of Baculites compressus at 94.16 µm (Pass 1) (`models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py`) | 88 | 76 | 82 | 84 | 90 | 82 | 86 | 86.0 | 79.0 | 86.3 | 83.3 | **83.3** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength of Baculites compressus at 94.16 µm (Pass 1)  —  overall 83.3 (faithful)

Cell-frame Schnakenberg system (a=0.10, b=0.90, r=9.2 just above r_c≈8.568) is calibrated so k_c = 2π/94.16 µm. Numeric growth rates track the analytic dispersion with Pearson r = 0.99966 and RMSE = 0.00884 per τ. Equal diffusion decays (σ≈−0.77). The fastest mode is 96.54 µm, not forced onto 94.16. Aperture factor χ is the series parameter: Placenticeras 141.24 µm, Baculites 94.16 µm at 73.5 Ma, Nipponites 62.15 µm. The flank GIF is a band-limited suture packet on a capsule, compared with Kevmin's CC BY-SA polished section. The comparison figure prints orientation KS, radial-FFT r, and coherency. The eigenmode step is still 0.12 of the diffusion stability bound (RMSE 0.00884 per τ).

- **Model fidelity to source: 88** — Jacobian, k_c formula, analytic eigenvalue, and the periodic integrator implement Turing 1952 / Schnakenberg 1979. r = 9.2 is held just above the computed threshold so the selected band sits on the assigned wavelength. The flank image is a seeded Fourier packet, not the time-integrated field.  
  evidence: `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L66–84; `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L99–118
- **Morphological realism / specimen match: 76** — Side by side, the capsule has the dark field, metallic ridges, and horizontal shaft of the polished section, but the outline is a smooth oval and the sutures are a regular labyrinth rather than the specimen's broken dendritic sutures. Texture statistics are closer than the silhouette.  
  evidence: `docs/media/baculites_compressus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/baculites_compressus_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 82** — Multiplying both diffusivities moves λ_max as √D across a 16-fold range, and χ orders the three series members. The sweep's Pearson r of 1 is a three-point probe centered on the analytic peak, so it cannot miss.  
  evidence: `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L205–242; `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L244–265
- **Morphometric validation: 84** — Dispersion r = 0.99966 with RMSE 0.00884 at dt = 0.12 of the stability bound; ablation σ is negative; λ_max = 96.54 µm against the 94.16 µm reference; the Kevmin section is segmented (area 0.427, aspect 2.62). KS, radial-FFT r, and coherency are printed on the figure. The timestep has not yet been tightened.  
  evidence: `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L161–189; `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L373–440
- **Source and data traceability: 90** — Crossref DOIs for Turing, Schnakenberg, Gierer–Meinhardt, Meinhardt–Klingler, the 1892 coiled-stage note, and Fatherree et al. 1998. The specimen is Kevmin's CC BY-SA 3.0 file, SHA-256 pinned. OpenAlex returned HTTP 429 and that failure is recorded.  
  evidence: `datasets/baculites_compressus_turing_wavelength_s0/paper_citations.bib` L1–70; `datasets/baculites_compressus_turing_wavelength_s0/baculites_compressus_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 82** — k_c, σ(k), the ablation, and χ are each a named function. The flank wavenumber k0 = 0.12 rad/px is a photograph-scale packet, not the 94.16 µm chemical wavenumber, and that distinction lives in the docstring rather than a cited equation number.  
  evidence: `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L274–294; `morphoeval/lean/loops/Loop_3131_baculites_compressus_turing_wavelength_s0.lean` L34–63
- **Reproducibility package: 86** — Seed 3131, SHA-256 gate, --self-test rejecting r ≤ 0.98, and one command writes the PNG, the 12-frame GIF, and the metrics JSON. Dependencies are the repo scientific stack, not a per-script lock.  
  evidence: `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L647–673; `datasets/baculites_compressus_turing_wavelength_s0/README.md` L20–32

Remediation:
- {'id': 'R1', 'item': 'Tighten the eigenmode timestep from 0.12 to 0.06 of the explicit diffusion bound and report the new dispersion RMSE against the Pass-1 value 0.00884.'}
- {'id': 'R2', 'item': 'Put the Bearpaw-versus-Pierre-Shale transfer and the 94.16 µm versus 2.7 cm distinction in the comparison-figure caption, not only in the reference JSON.'}
- {'id': 'R3', 'item': 'Keep the diffusion-sweep panel labeled as a local probe after the timestep change, so the dispersion curve remains the independent fit.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength of Baculites compressus at 94.16 µm (Pass 1)
- **spec_fingerprint** (semantic model + inputs): `987c95abfd36be5a754c6bd9b1cc8b19b4c29acfc2ed8558194b88bb00b620d6`
- **input_dataset_sha256**: `datasets/baculites_compressus_turing_wavelength_s0/baculites_compressus_specimen.jpg` (`b4eabd655977c5ee…`), `datasets/baculites_compressus_turing_wavelength_s0/baculites_compressus_turing_wavelength_s0_reference.json` (`1e794dec0e5bf0ec…`), `datasets/baculites_compressus_turing_wavelength_s0/paper_citations.bib` (`03254933877482eb…`)
- `docs/media/baculites_compressus_turing_wavelength_s0_comparison.png` — artifact_sha256 `11bd13ddcc65dc8a…`, 359,866 B, 1512×888 px
- `docs/media/baculites_compressus_turing_wavelength_s0_growth.gif` — artifact_sha256 `7784e16f96ef0f74…`, 1,152,113 B, 720×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion Pearson r of numeric versus analytic growth rate | 0.999659 Pearson r | 1.0 | 1D periodic Schnakenberg eigenmode integrated with explicit Euler; σ = log(A1/A0)/t compared with the larger Jacobian eigenvalue *(RMSE = 0.00884 per τ across 8 wavenumbers)* |
| fastest-mode wavelength versus critical wavelength | 96.5433 µm | 94.16 | argmax_k σ(k) on a 240-point grid at r = 9.2 *(near threshold λ_max sits a few micrometres above λ_c; they are not forced equal)* |
| equal-diffusion ablation growth rate at k_c | -0.7739 1/τ | 0.0 | same eigenmode integrator with D_v = D_u *(analytic ablation σ = -0.4457; Turing σ(k_c) = 0.0264)* |
| shell foreground fraction segmented from the Kevmin specimen | 0.4267 area fraction | 0.55 | border-median threshold, closing, largest connected component *(circularity = 0.1019, aspect = 2.620)* |
| specimen versus simulation orientation-histogram KS statistic | 0.4167 KS distance | 0.0 | 12-bin Sobel orientation histograms on the fitness crops; scipy ks_2samp *(radial-FFT Pearson r = 0.8542, RMSE = 1.2557)* |
| simulation structure-tensor coherency on the flank frame | 0.3716 coherency | 0.3986 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor *(reference is the specimen measurement; the two numbers are computed separately)* |
| Placenticeras predecessor critical wavelength | 141.24 µm | 94.16 | λ_c(χ=3/2) with the Baculites chemistry held fixed *(aperture stretch, not a new reaction scheme)* |
| Nipponites successor critical wavelength | 62.15 µm | 94.16 | λ_c(χ=6215/9416) with the Baculites chemistry held fixed *(whorl recontact shortens the free margin)* |

## Environment at checkpoint


Source commit for the script, specimen, and Lean module. The audit graph is committed after this hash.
