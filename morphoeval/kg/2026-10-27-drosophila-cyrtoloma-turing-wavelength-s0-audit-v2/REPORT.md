# MorphoEval report — audit `2026-10-27-drosophila-cyrtoloma-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-cyrtoloma-turing-wavelength-s0-audit -> 2026-10-27-drosophila-cyrtoloma-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila cyrtoloma Turing carrier at 4.3 µm (Pass 2) (`models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py`) | 88 | 86 | 90 | 88 | 92 | 86 | 90 | 88.7 | 86.8 | 89.6 | 88.2 | **88.2** | faithful |

## Findings and remediation

### Drosophila cyrtoloma Turing carrier at 4.3 µm (Pass 2)  —  overall 88.2 (faithful)

At 0 Ma the focal carrier is the nonlinear Schnakenberg pattern calibrated to 4.315 µm (Dv/Du = 18). Linear k_c on the same grid is 3.70 µm. Stripe counts on widths 48–144 px correlate with L/λ_linear at r = 0.9917 (RMSE 0.549). Equal diffusion stays at activator std 0.0021 against 0.580 for the focal run. Simulated neighbours, same dx: buzzatii 8.07 µm, deflecta 6.83 µm, both longer than cyrtoloma. GIF last-frame CV similarity to the vendored plate is 92.8 (640×480, 12 frames).

- **Model fidelity to source: 88** — The update is Schnakenberg's (a − u + u²v, b − u²v) and k_c is the Turing/Murray quotient. The Jacobian trace is negative and the equal-diffusion threshold fails, matching the Lean witness. The 4.315 µm value is a grid calibration of the nonlinear wavelength, disclosed as such, which keeps this below the low 90s.  
  evidence: `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L95–133; `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L147–197; `morphoeval/lean/loops/Loop_3180_drosophila_cyrtoloma_turing_wavelength_s0.lean` L35–55
- **Morphological realism / specimen match: 86** — Last GIF frame versus the pinned plate scores CV similarity 92.8 (orientation, radial FFT, coherency, occupancy) at 640×480 with 12 frames. The plate is a gallery spiral with a 100 µm bar, not a located Commons photograph, so the score stays short of a verified micrograph overlay.  
  evidence: `docs/media/drosophila_cyrtoloma_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_cyrtoloma_turing_wavelength_s0_comparison.png` L0–0; `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L386–411
- **Parameter expressivity: 90** — Raising γt from 0.55 to about 55 turns a failed sweep (r = 0.906) into a domain-length count that tracks L/λ. Equal diffusion abolishes the pattern. Buzzatii, cyrtoloma, and deflecta keep the order long > mid > short.  
  evidence: `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L223–245; `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L270–287; `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L289–319
- **Morphometric validation: 88** — Pearson r = 0.9917 and RMSE = 0.549 stripes on seven widths. Spacing CV = 0.083 against a zero reference. Plate Otsu circularity is 0.772 against a disk reference of 1. Plate coherency is not set equal to the simulation coherency. Deflecta is 6.83 µm against a 6.05 µm hypothesis, which is why this is not 95.  
  evidence: `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L332–384; `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L595–694; `datasets/drosophila_cyrtoloma_turing_wavelength_s0/simulation_metrics.json` L0–0
- **Source and data traceability: 92** — Twelve bibliography DOIs were resolved on Crossref or OpenAlex, including the corrected Werner 2010 DOI 10.1038/nature08896. Slot 3061's Zootaxa 2400 DOI is explicitly rejected. Specimen SHA-256 is the gallery byte copy.  
  evidence: `datasets/drosophila_cyrtoloma_turing_wavelength_s0/paper_citations.bib` L1–124; `datasets/drosophila_cyrtoloma_turing_wavelength_s0/README.md` L1–40
- **Code-to-model traceability: 86** — Each modelling step in the docstring maps to Turing, Murray, or Schnakenberg. The Lean file witnesses the same inequalities. The floating-point (a, b) pair is not re-derived from a cyrtoloma table, which caps the score.  
  evidence: `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L1–36; `morphoeval/math_kg_loops/delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.yaml` L24–70
- **Reproducibility package: 90** — python3 script --self-test checks the Turing flag, the equal-diffusion ablation, the wavelength order, and Pearson r > 0.98. Seed 3180 and SHA-256 pins are fixed. Nanopub signing is unavailable in this environment and is not pretended.  
  evidence: `models/2026-delphi10000-drosophila-cyrtoloma-turing-wavelength-s0.py` L697–731; `datasets/drosophila_cyrtoloma_turing_wavelength_s0/README.md` L28–42

Remediation:
- A real CC photograph of a pinned D. cyrtoloma adult is still missing; Commons had none on 2026-10-27.
- Deflecta simulated carrier is 6.83 µm against the 6.05 µm hypothesis. The order versus cyrtoloma and buzzatii holds; the micrometre does not.
- Signing was not run: ~/.nanopub/profile.yml is absent.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila cyrtoloma Turing carrier at 4.3 µm (Pass 2)
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


Pass 2 readout is the γt≈55 domain-length protocol in the same script. Loop files are new relative to this commit.
