# MorphoEval report — audit `2026-10-27-baculites-compressus-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `8f0ebbb7583b60dab933158a86eda8e291baab3e` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-baculites-compressus-turing-wavelength-s0-audit -> 2026-10-27-baculites-compressus-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg Turing wavelength of Baculites compressus at 94.16 µm (Pass 2) (`models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py`) | 90 | 78 | 88 | 88 | 92 | 86 | 88 | 89.3 | 81.8 | 88.8 | 86.1 | **86.1** | faithful |

## Findings and remediation

### Schnakenberg Turing wavelength of Baculites compressus at 94.16 µm (Pass 2)  —  overall 86.1 (faithful)

Pass 2 closes the three Pass-1 remediation items. The eigenmode step is 0.06 of the diffusion bound instead of 0.12; dispersion Pearson r moves from 0.99966 to 0.99994 and RMSE from 0.00884 to 0.00802 per τ. The comparison-figure caption states that 94.16 µm is the chemical wavelength at the 73.5 Ma Pierre Shale zone, not a caliper reading of the 2.7 cm Bearpaw section. The diffusion sweep stays labeled as a local probe. λ_c remains 94.16 µm, λ_max 96.54 µm, and equal diffusion still decays. The flank GIF is unchanged in construction: 12 frames, 720×480.

- **Model fidelity to source: 90** — The same Schnakenberg Jacobian and k_c formula as Pass 1, now integrated at half the previous timestep fraction. Dispersion r = 0.99994. The flank image remains a seeded Fourier packet rather than the time-integrated 2D field, which is why this is not higher.  
  evidence: `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L121–132; `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L99–118
- **Morphological realism / specimen match: 78** — The GIF is the same capsule-and-labyrinth comparison as Pass 1. The new caption stops the figure from being read as a measured 94 µm caliper on the Bearpaw section. The outline is still a smooth oval against a broken shell, so the side-by-side score stays moderate.  
  evidence: `docs/media/baculites_compressus_turing_wavelength_s0_comparison.png` L0–0; `docs/media/baculites_compressus_turing_wavelength_s0_growth.gif` L0–0; `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L501–509
- **Parameter expressivity: 88** — λ_max still scales as √D across the 16-fold diffusion multipliers, and χ still orders Nipponites < Baculites < Placenticeras. The panel is explicitly a local probe. The independent wavenumber test is the dispersion curve, which survived the tighter step.  
  evidence: `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L205–242; `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L475–478
- **Morphometric validation: 88** — Pass-2 dispersion r = 0.99994, RMSE = 0.00802 per τ, down from 0.00884. λ_max = 96.54 µm versus the 94.16 µm reference. Ablation σ stays negative. Segmented shell area, orientation KS, and radial-FFT r are unchanged and still not equal to their references.  
  evidence: `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L161–189; `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L373–440
- **Source and data traceability: 92** — Same DOI set and SHA-256-pinned CC BY-SA specimen as Pass 1. The figure caption now repeats the Bearpaw / Pierre Shale distinction that the reference JSON already stated.  
  evidence: `datasets/baculites_compressus_turing_wavelength_s0/paper_citations.bib` L1–70; `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L501–503
- **Code-to-model traceability: 86** — The timestep comment records the Pass-1 fraction and the Pass-2 fraction on the integrator itself. k_c, σ(k), the ablation, and χ remain named functions. The flank packet is still the photograph-scale projection, documented as such.  
  evidence: `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L128–132; `morphoeval/lean/loops/Loop_3131_baculites_compressus_turing_wavelength_s0.lean` L34–63
- **Reproducibility package: 88** — Same seed, hash gate, and --self-test. Re-running the script after the timestep change regenerated the PNG, the 12-frame 720×480 GIF, and the metrics JSON.  
  evidence: `models/2026-delphi10000-baculites-compressus-turing-wavelength-s0.py` L647–673; `datasets/baculites_compressus_turing_wavelength_s0/README.md` L26–36

Remediation:
- {'id': 'R4', 'item': 'One polished Bearpaw section stands in for a Pierre Shale population. χ for Placenticeras and Nipponites is still a geometric prediction, and the flank GIF is still not a 2D shell-mesh integration.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg Turing wavelength of Baculites compressus at 94.16 µm (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `987c95abfd36be5a754c6bd9b1cc8b19b4c29acfc2ed8558194b88bb00b620d6`
- **input_dataset_sha256**: `datasets/baculites_compressus_turing_wavelength_s0/baculites_compressus_specimen.jpg` (`b4eabd655977c5ee…`), `datasets/baculites_compressus_turing_wavelength_s0/baculites_compressus_turing_wavelength_s0_reference.json` (`1e794dec0e5bf0ec…`), `datasets/baculites_compressus_turing_wavelength_s0/paper_citations.bib` (`03254933877482eb…`)
- `docs/media/baculites_compressus_turing_wavelength_s0_comparison.png` — artifact_sha256 `82db17e156350666…`, 360,713 B, 1512×888 px
- `docs/media/baculites_compressus_turing_wavelength_s0_growth.gif` — artifact_sha256 `7784e16f96ef0f74…`, 1,152,113 B, 720×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion Pearson r of numeric versus analytic growth rate | 0.999943 Pearson r | 1.0 | 1D periodic Schnakenberg eigenmode integrated with explicit Euler; σ = log(A1/A0)/t compared with the larger Jacobian eigenvalue *(RMSE = 0.00802 per τ across 8 wavenumbers)* |
| fastest-mode wavelength versus critical wavelength | 96.5433 µm | 94.16 | argmax_k σ(k) on a 240-point grid at r = 9.2 *(near threshold λ_max sits a few micrometres above λ_c; they are not forced equal)* |
| equal-diffusion ablation growth rate at k_c | -0.4057 1/τ | 0.0 | same eigenmode integrator with D_v = D_u *(analytic ablation σ = -0.4457; Turing σ(k_c) = 0.0264)* |
| shell foreground fraction segmented from the Kevmin specimen | 0.4267 area fraction | 0.55 | border-median threshold, closing, largest connected component *(circularity = 0.1019, aspect = 2.620)* |
| specimen versus simulation orientation-histogram KS statistic | 0.4167 KS distance | 0.0 | 12-bin Sobel orientation histograms on the fitness crops; scipy ks_2samp *(radial-FFT Pearson r = 0.8542, RMSE = 1.2557)* |
| simulation structure-tensor coherency on the flank frame | 0.3716 coherency | 0.3986 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor *(reference is the specimen measurement; the two numbers are computed separately)* |
| Placenticeras predecessor critical wavelength | 141.24 µm | 94.16 | λ_c(χ=3/2) with the Baculites chemistry held fixed *(aperture stretch, not a new reaction scheme)* |
| Nipponites successor critical wavelength | 62.15 µm | 94.16 | λ_c(χ=6215/9416) with the Baculites chemistry held fixed *(whorl recontact shortens the free margin)* |

## Environment at checkpoint


Source commit for the script, specimen, and Lean module. Pass 2 tightens the integrator that this commit contains.
