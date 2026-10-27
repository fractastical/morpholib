# MorphoEval report — audit `2026-10-27-drosophila-grimshawi-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `0c76d15d14afcec6c366e1b9274a6586a0de48fc` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-drosophila-grimshawi-turing-wavelength-s0-audit -> 2026-10-27-drosophila-grimshawi-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila grimshawi Turing wavelength at 2.8 µm (detail) (`models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py`) | 89 | 87 | 88 | 90 | 89 | 86 | 88 | 88.7 | 88.1 | 87.8 | 88.2 | **88.2** | faithful |

## Findings and remediation

### Drosophila grimshawi Turing wavelength at 2.8 µm (detail)  —  overall 88.2 (faithful)

Pass 2 closes the three pass-1 remediation items. The comparison figure now plots λ_c against δ from 9 to 20 (monotone lengthening; grimshawi at δ = 16.3186, λ = 2.766 µm). Sobel-orientation KS between the masked gallery specimen and the spiral is 0.0418; the zero-arm ablation KS is 0.0191, so orientation distance alone does not rank the spiral first. The full CV score does: the zero-arm ablation, which is not histogram-matched, scores CV 71.4 against 93.8 for the spiral (drop 22.4). Specimen Otsu fraction is referenced to that ablation, not to a copy of itself. Dispersion growth Pearson r remains 0.99974. Chronoseries at 0 Ma, same kinetics: ficusphila 2.512 µm, guttifera 2.629 µm, grimshawi 2.766 µm. Residual limit: the canonical image is the project gallery spiral, not a 2.8 µm wing micrograph, and Lean was not executed because no Lean binary is installed.

- **Model fidelity to source: 89** — The eigenvalue, the cosine-seeded PDE (r = 0.99974), equal-diffusion σ_max = -0.717, and a zero-arm ablation whose CV falls from 93.8 to 71.4 are all in the script. The animation is still the near-onset spiral, not a coarsened 2D spot field.  
  evidence: `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L67–79; `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L607–627
- **Morphological realism / specimen match: 87** — GIF last frame CV is 93.8 at 800×600 with 12 frames. Removing the spiral carrier drops CV by 22.4 (ablation 71.4). The gallery frame is not a retinal micrograph, which caps this score.  
  evidence: `docs/media/drosophila_grimshawi_turing_wavelength_s0_growth.gif` L0–0; `docs/media/drosophila_grimshawi_turing_wavelength_s0_comparison.png` L1–1
- **Parameter expressivity: 88** — λ_c increases monotonically with δ from 9 to 20 on the pass-2 panel. The focal, predecessor, and successor ratios are marked by the dispersion curves and the bar chart.  
  evidence: `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L335–347; `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L110–120
- **Morphometric validation: 90** — Growth Pearson r = 0.99974 (gate 0.98). Radial envelope r = 0.99542. Orientation KS is 0.0418 for the spiral and does not by itself beat the ablation; the CV drop of 22.4 does. Specimen Otsu fraction is referenced to the unmatched ablation, not copied from the specimen.  
  evidence: `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L559–604; `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L122–173
- **Source and data traceability: 89** — DOIs for Turing, Schnakenberg, Edwards 2007, Clark 2007, Werner 2010, and Koshikawa 2020 are in the bib and reference JSON. Specimen SHA-256 is pinned. The rejected Commons envelopes are recorded in the scout manifest.  
  evidence: `datasets/drosophila_grimshawi_turing_wavelength_s0/paper_citations.bib` L1–48; `datasets/drosophila_grimshawi_turing_wavelength_s0/scout_manifest.json` L1–40
- **Code-to-model traceability: 86** — Dispersion, sweep, ablation, and morphometrics are separate functions. Lean states the trace sign, equal-diffusion suppression, wavelength order, and stripe count with zero sorry. The kernel was not run here.  
  evidence: `morphoeval/lean/loops/Loop_3186_drosophila_grimshawi_turing_wavelength_s0.lean` L16–39; `morphoeval/math_kg_loops/delphi10000-drosophila-grimshawi-turing-wavelength-s0.yaml` L36–70
- **Reproducibility package: 88** — Self-test now also requires the ablation CV drop and a finite KS. Seed 3186 and the specimen digest are unchanged. Invocation is in the dataset README.  
  evidence: `models/2026-delphi10000-drosophila-grimshawi-turing-wavelength-s0.py` L515–557; `datasets/drosophila_grimshawi_turing_wavelength_s0/README.md` L14–32

Remediation:
- {'id': 'R1', 'item': 'RESOLVED in pass 2: continuous δ sweep is panel (2,0) of the comparison figure.'}
- {'id': 'R2', 'item': 'RESOLVED in pass 2: orientation KS and Otsu foreground fractions are in measure_morphoeval.'}
- {'id': 'R3', 'item': 'RESOLVED in pass 2: zero-arm ablation is rendered and its CV drop is a measurement.'}
- {'id': 'R4', 'item': 'No anatomical 2.8 µm micrograph was available; Commons files were envelope labels. The gallery spiral remains the fitness target.'}
- {'id': 'R5', 'item': 'Lean theorems are sorry-free but were not kernel-checked; lean is not installed in this environment.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila grimshawi Turing wavelength at 2.8 µm (detail)
- **spec_fingerprint** (semantic model + inputs): `e889fa230e849b3a3849aace67edfdc63e0c1f6e6b4678385e1b298ef5811b4a`
- **input_dataset_sha256**: `datasets/drosophila_grimshawi_turing_wavelength_s0/drosophila_grimshawi_turing_wavelength_s0_reference.json` (`ac4d59314c82a9c0…`), `datasets/drosophila_grimshawi_turing_wavelength_s0/drosophila_grimshawi_turing_wavelength_s0_specimen.jpg` (`4ea3740942a29423…`), `datasets/drosophila_grimshawi_turing_wavelength_s0/paper_citations.bib` (`ebf2b96514b75c28…`)
- `docs/media/drosophila_grimshawi_turing_wavelength_s0_comparison.png` — artifact_sha256 `5e740d6d5a9e0aac…`, 521,128 B, 1144×1232 px
- `docs/media/drosophila_grimshawi_turing_wavelength_s0_growth.gif` — artifact_sha256 `7b664426b7a9d089…`, 6,713,095 B, 800×600 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| grimshawi fastest-mode wavelength | 2.7659 µm | 2.8 | k_c = argmax sigma(k) for Schnakenberg δ=16.3186; lambda = 2π/k_c *(k_c=2.2717 1/µm; σ_max=1.8705)* |
| numerical mode growth rate | 1.8841 1/time | 1.8705 | log-amplitude slope of a cosine-seeded Schnakenberg integration *(discrete Laplacian shifts σ slightly below the continuum value)* |
| dispersion growth Pearson r | 0.99974 Pearson r | 0.98 | pearsonr(t, log amplitude) on the linear-regime cosine mode *(log-amplitude RMSE=0.01503)* |
| specimen radial pigment profile Pearson r | 0.99542 Pearson r | 0.9 | degree-5 polynomial amplitude R(ρ) versus azimuthally averaged gallery luminance *(RMSE=1.5835 gray levels)* |
| segmented specimen structure-tensor coherency | 0.2122 coherency | 0.5 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor inside the caption-excluded mask |
| predecessor ficusphila detail wavelength | 2.5123 µm | 2.7659 | same kinetics, δ=10.40 *(shorter than grimshawi)* |
| successor guttifera detail wavelength | 2.6291 µm | 2.5123 | same kinetics, δ=12.85 *(intermediate between ficusphila and grimshawi)* |
| equal-diffusion control max sigma | -0.7167 1/time | 0.0 | max_k σ(k) at D_u = D_v *(negative control)* |
| specimen orientation histogram KS versus spiral | 0.0418 KS distance | 0.0191 | two-sample KS on Sobel orientations inside the caption mask *(pass-2 segmentation comparison; lower is closer)* |
| segmented specimen Otsu foreground fraction | 0.1637 area fraction | 0.0843 | Otsu threshold on caption-excluded gallery luminance *(spiral Otsu fraction=0.1637 (histogram match can equal the specimen))* |
| ablation CV drop versus Turing spiral | 22.4 CV points | 0.0 | verify_image_fitness CV of the spiral GIF ramp minus the zero-arm ablation *(model CV=93.8; ablation CV=71.4)* |

## Environment at checkpoint


Pass 2 remediation of R1–R3. Origin URL carries credentials, so commit-pinned loci are skipped by the checker.
