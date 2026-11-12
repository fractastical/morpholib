# MorphoEval report — audit `2026-10-27-drosophila-cyrtoloma-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila cyrtoloma Turing carrier at 4.3 µm (Pass 1 diagnostic) (`models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py`) | 82 | 74 | 72 | 74 | 90 | 84 | 86 | 78.7 | 74.0 | 86.8 | 79.2 | **79.2** | mostly faithful |

## Findings and remediation

### Drosophila cyrtoloma Turing carrier at 4.3 µm (Pass 1 diagnostic)  —  overall 79.2 (mostly faithful)

Pass 1 integrates the diffusivity sweep only to γt = 0.55. The activator has not selected a wavelength, and Pearson r against the linear λ(σ) is 0.906. The equal-diffusion control, Otsu segmentation, spacing CV/KS, and the domain-length count are specified as remediation and scored in v2.

- **Model fidelity to source: 82** — Schnakenberg kinetics and the Turing dispersion formula are implemented (trace < 0, det > 0, k_c). Pass 1 does not yet separate the linear wavelength (3.70 µm on the calibrated grid) from the 4.315 µm nonlinear calibration in the scored fit.  
  evidence: `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L95–133; `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L147–197; `morphoeval/lean/loops/Loop_3180_drosophila_cyrtoloma_turing_wavelength_s0.lean` L50–55
- **Morphological realism / specimen match: 74** — The GIF is 640×480 and 12 frames and the plate is pinned, but Pass 1 has not yet scored Otsu circularity or structure-tensor coherency of the plate against the last frame. The plate is the project gallery spiral, not a pinned adult.  
  evidence: `docs/media/drosophila_cyrtoloma_turing_wavelength_s0_growth.gif` L0–0; `datasets/drosophila_cyrtoloma_turing_wavelength_s0/drosophila_cyrtoloma_turing_wavelength_s0_reference.json` L7–22
- **Parameter expressivity: 72** — The σ sweep at γt = 0.55 moves the analytic wavelength but the simulated transect does not follow it (r = 0.906). Species diffusivities are coded and not yet the scored comparison.  
  evidence: `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L247–268
- **Morphometric validation: 74** — Pass 1 Pearson r = 0.906 on the short sweep, below the 0.98 gate. RMSE, KS, spacing CV, and specimen circularity are not yet the reported comparison.  
  evidence: `datasets/drosophila_cyrtoloma_turing_wavelength_s0/simulation_metrics.json` L0–0
- **Source and data traceability: 90** — DOIs were checked on Crossref or OpenAlex. The Zootaxa 2400 DOI used by slot 3061 is rejected because Crossref assigns it to Macroscytus, not this fly. The gallery-plate fallback is stated.  
  evidence: `datasets/drosophila_cyrtoloma_turing_wavelength_s0/paper_citations.bib` L1–124; `datasets/drosophila_cyrtoloma_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 84** — The module docstring names Turing 1952, Murray 1981, and Schnakenberg 1979 next to the equations. Species-specific micrometre values are labelled as reconstructions.  
  evidence: `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L1–36
- **Reproducibility package: 86** — Seed 3180, SHA-256 pins, and --self-test are present. Pass 1 self-test was not yet the domain-length assertion that clears r > 0.98.  
  evidence: `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L71–79; `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L697–730

Remediation:
- Replace the γt=0.55 diffusivity sweep (Pearson r = 0.906) with the developed domain-length peak count and require r > 0.98.
- Report the equal-diffusion ablation (Dv = Du) beside the focal activator standard deviation.
- Segment the vendored plate with Otsu and report circularity and coherency with value unequal to the reference.
- Report inter-peak spacing CV and a KS statistic for the 4.315 µm carrier.
- Simulate the buzzatii and deflecta carriers on the same grid spacing and check the wavelength order.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila cyrtoloma Turing carrier at 4.3 µm (Pass 1 diagnostic)
- **spec_fingerprint** (semantic model + inputs): `9a348d1b6c0648f573bdd67fbb5ddd58de163dee9a214d6c7b1eca5585a2eda8`
- **input_dataset_sha256**: `datasets/drosophila_cyrtoloma_turing_wavelength_s0/drosophila_cyrtoloma_specimen.jpg` (`a274390f57aa205d…`), `datasets/drosophila_cyrtoloma_turing_wavelength_s0/drosophila_cyrtoloma_turing_wavelength_s0_reference.json` (`dbe207001ca2facb…`), `datasets/drosophila_cyrtoloma_turing_wavelength_s0/paper_citations.bib` (`bd4d7e6a882797ef…`), `datasets/drosophila_cyrtoloma_turing_wavelength_s0/scout_manifest.json` (`6de58dd5861c376c…`)
- `datasets/drosophila_cyrtoloma_turing_wavelength_s0/simulation_metrics.json` — artifact_sha256 `839cf809a8ad98af…`, 4,410 B, 
- `docs/media/drosophila_cyrtoloma_turing_wavelength_s0_comparison.png` — artifact_sha256 `f9e516d6baf1e715…`, 823,511 B, 1344×1008 px
- `docs/media/drosophila_cyrtoloma_turing_wavelength_s0_growth.gif` — artifact_sha256 `486121e471e2b3aa…`, 2,826,530 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| stripe-count Pearson r versus domain length | 0.9917040947249601 Pearson r | 0.9055782723766085 | Count activator peaks along x on widths 48..144 px and correlate with L/λ_linear *(RMSE = 0.5492 stripes; pass-1 short-sweep r = 0.9056)* |
| stripe-count RMSE versus L/lambda_c | 0.549169647365276 stripes | 1.0 | Root mean square of (counted peaks − L/λ_linear) across seven domain widths |
| segmented specimen-plate Otsu circularity | 0.7720191371683136 4πA/P² | 1.0 | Otsu threshold, closing, fill, on the 640×480 plate *(foreground fraction 0.9927; aspect 0.7492)* |
| specimen-plate structure-tensor coherency | 0.2978936598814845 (λ1-λ2)/(λ1+λ2) | 0.5350061496960096 | Mean structure-tensor coherency inside the Otsu mask of the vendored plate *(value is the plate; reference_value is the simulation frame)* |
| inter-peak spacing CV of the 4.3 µm carrier | 0.0829156197386925 CV | 0.0 | std/mean of peak-to-peak distances on the y-averaged activator *(KS statistic against a 2% jitter cloud = 0.5000, p = 0.474)* |
| linear k_c wavelength on the calibrated 4.315 µm grid | 3.695638951339883 µm | 4.315 | λ = 2π/k_c with k_c from the Schnakenberg Jacobian, times dx = 4.315/λ_nonlinear *(Linear and nonlinear wavelengths are reported as different numbers on purpose)* |
| equal-diffusion ablation activator std | 0.0021102062657923225 concentration | 0.5802143748679993 | Same kinetics and Du with Dv = Du; Turing threshold fails *(equal-diffusion Turing flag = False)* |
| buzzatii predecessor simulated carrier | 8.0652 µm | 7.84 | Same dx as the focal grid; Dv/Du = 12.5; diffusivity scale (7.84/4.315)^2 |
| deflecta successor simulated carrier | 6.8293 µm | 6.05 | Dv/Du = 26, x/y inhibitor anisotropy 1.55, diffusivity scale (6.05/4.315)^2 |

## Environment at checkpoint


Pass 1 readout is the under-developed sweep recorded in simulation_metrics.json. Loop files are new relative to this commit.
