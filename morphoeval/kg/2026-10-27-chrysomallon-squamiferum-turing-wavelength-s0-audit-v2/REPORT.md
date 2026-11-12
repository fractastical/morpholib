# MorphoEval report — audit `2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `6f01b4c5e4c9fa53e8bf2c8825be8b7ccb7fba3c` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit -> 2026-10-27-chrysomallon-squamiferum-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Schnakenberg fastest-mode wavelength at 2.0 nm on Chrysomallon squamiferum (Pass 2) (`models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py`) | 86 | 83 | 88 | 90 | 92 | 86 | 90 | 86.7 | 85.6 | 89.6 | 87.1 | **87.1** | faithful |

## Findings and remediation

### Schnakenberg fastest-mode wavelength at 2.0 nm on Chrysomallon squamiferum (Pass 2)  —  overall 87.1 (faithful)

Pass 2 scales D_u and D_v so the maximum of σ(k) lies on the requested wavelength, then applies one nonlinear correction learned at 2.0 nm. Sweep targets 1.35, 1.70, 2.00, 2.35, 2.80, 3.30 nm give PDE peaks 1.229, 1.559, 1.935, 2.211, 2.620, 2.970 nm (Pearson r=0.9964, RMSE=0.182 nm, residual CV=0.043). Equal-diffusion ablation ratio is 6.5e-12. Paratype mask foreground fraction 0.394 (prior 0.55), circularity 0.118 (circle would be 1), specimen coherency 0.310 versus composite 0.345, intensity KS=0.248. GIF is 12 frames at 640×480. Chronoseries 1/1 at 0 Ma; successor none; comparative predecessor is sclerite-less Gigantopelta.

- **Model fidelity to source: 86** — Pass 2 keeps Schnakenberg kinetics and distinguishes the onset wavenumber from argmax σ(k). Diffusivities are scaled so the linear peak tracks the requested wavelength, with one nonlinear correction. The biology sources still do not contain this PDE; the transfer is stated rather than hidden.  
  evidence: `models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py` L416–438; `models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py` L124–132
- **Morphological realism / specimen match: 83** — The GIF last frame is the Hennion paratype with a copper activator overlay on the dark sclerites and shell, and the comparison figure shows the spot field beside the photograph. The spots are a model, not a photograph of 2 nm nuclei. CV similarity of the GIF against the paratype is 96.0.  
  evidence: `docs/media/chrysomallon_squamiferum_turing_wavelength_s0_growth.gif` L0–0; `docs/media/chrysomallon_squamiferum_turing_wavelength_s0_comparison.png` L0–0
- **Parameter expressivity: 88** — Six target wavelengths from 1.35 to 3.30 nm move the PDE peak monotonically (1.23 to 2.97 nm). Equal diffusion at the same D_u collapses the pattern. a and b stay at the slot Jacobian so the Lean thousandths remain the ones that were proved.  
  evidence: `models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py` L506–535
- **Morphometric validation: 90** — Pearson r=0.9964 and RMSE=0.182 nm on the sweep; the 2.0 nm run peaks at 1.935 nm. Ablation ratio 6.5e-12. Specimen foreground 0.394 vs prior 0.55, circularity 0.118 vs 1, coherency 0.310 vs composite 0.345, intensity KS=0.248. These references are not copies of the measured values. One paratype only.  
  evidence: `models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py` L525–563; `datasets/chrysomallon_squamiferum_turing_wavelength_s0/chrysomallon_squamiferum_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 92** — Same pinned paratype and Crossref DOIs as Pass 1, plus Wong et al. 2025 on sclerite co-option relative to Gigantopelta. SHA-256 of the specimen, bibliography, scout manifest, and README are in the reference JSON.  
  evidence: `datasets/chrysomallon_squamiferum_turing_wavelength_s0/paper_citations.bib` L55–90; `datasets/chrysomallon_squamiferum_turing_wavelength_s0/scout_manifest.json` L1–30
- **Code-to-model traceability: 86** — run_pass2 cites the fastest-mode correction in place. Loop_3147 proves the integer wavenumber identity, the greigite window, equal-diffusion trace stability, and the Holocene singleton, with zero sorry.  
  evidence: `models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py` L495–514; `morphoeval/lean/loops/Loop_3147_chrysomallon_squamiferum_turing_wavelength_s0.lean` L19–47
- **Reproducibility package: 90** — --self-test checks r>0.98, PDE wavelength within 8%, ablation ratio <0.25, GIF frames and size, and a non-equal specimen metric. Seed 3147. Signing keys are absent, which is recorded rather than simulated.  
  evidence: `models/2026-delphi10000-chrysomallon-squamiferum-turing-wavelength-s0.py` L687–698; `datasets/chrysomallon_squamiferum_turing_wavelength_s0/README.md` L1–22

Remediation:
- {'id': 'R5', 'item': 'Only one paratype photograph. A population of sclerite outlines would test whether foreground fraction and circularity are specimen-specific.'}
- {'id': 'R6', 'item': 'No transmission-electron measurement at 2.0 nm. The wavelength is the calibrated critical mode, about two greigite cells, not a counted nucleation spacing.'}
- {'id': 'R7', 'item': 'Builder-assessor only. An independent second scorer has not re-read the transfer boundary.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Schnakenberg fastest-mode wavelength at 2.0 nm on Chrysomallon squamiferum (Pass 2)
- **spec_fingerprint** (semantic model + inputs): `e0865ee451716d4a3fa5bd6b6983a8391e5da7f8dc195fece7fe187c823b6b31`
- `datasets/chrysomallon_squamiferum_turing_wavelength_s0/chrysomallon_squamiferum_turing_wavelength_s0_metrics.json` — artifact_sha256 `1e5b0c6aa1335788…`, 3,586 B, 
- `docs/media/chrysomallon_squamiferum_turing_wavelength_s0_comparison.png` — artifact_sha256 `f78b2d8abd16f668…`, 505,249 B, 1248×960 px
- `docs/media/chrysomallon_squamiferum_turing_wavelength_s0_growth.gif` — artifact_sha256 `894a567dd3d326cd…`, 1,370,524 B, 640×480 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| specimen_foreground_fraction | 0.3936 1 | 0.55 | gray<0.78 mask on paratype MNHN-IM-2000-30072 *(non-circular: measured area fraction differs from the prior)* |
| specimen_contour_circularity | 0.1183 1 | 1.0 | 4*pi*area/perimeter^2 on the sclerite-and-shell mask *(non-circular specimen shape metric)* |
| specimen_structure_tensor_coherency | 0.3097 1 | 0.3455 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor *(KS intensity statistic 0.2475 (p=0); orientation KS 0.0265)* |
| pde_versus_target_wavelength_pearson_r | 0.996439 1 | 0.98 | Pearson r of 6 target wavelengths vs PDE radial-FFT peaks after fastest-mode calibration *(RMSE 0.1823 nm; residual CV 0.0365; sweep [1.35, 1.7, 2.0, 2.35, 2.8, 3.3] -> [1.229, 1.559, 1.935, 2.211, 2.62, 2.97])* |
| calibrated_wavelength_nm | 1.9347 nm | 2.0 | dominant radial-FFT wavelength of the 2.0 nm Schnakenberg run *(equal-diffusion ablation std ratio 6.546e-12)* |
| equal_diffusion_ablation_ratio | 0.0 1 | 1.0 | std(u | D_u=D_v) / std(u | D_v=36 D_u) at the calibrated 2.0 nm scale *(negative control; trace f_u+g_v < 0 so equal diffusion is linearly stable)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- scipy: 1.18.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

Pass 2 default path is run_pass2. Nanopub signing was not available (~/.nanopub/profile.yml absent).
