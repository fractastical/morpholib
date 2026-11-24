# MorphoEval report — audit `2026-10-27-ambulocetus-natans-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `dff4f2b5` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ambulocetus natans Turing wavenumber at 224 µm (47.5 Ma, detail primordium) (`models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py`) | 90 | 84 | 82 | 88 | 92 | 86 | 90 | 87.3 | 85.5 | 89.6 | 87.3 | **87.3** | faithful |

## Findings and remediation

### Ambulocetus natans Turing wavenumber at 224 µm (47.5 Ma, detail primordium)  —  overall 87.3 (faithful)

Pass 1 implements Schnakenberg kinetics at a=0.05, b=0.90, Du=1, Dv=50, with the physical length chosen so the fastest analytic mode is 224 µm. Forward-Euler mode growth matches the eigenvalue curve at Pearson r=0.9997 (RMSE=0.0034). The nonlinear field on a 50x50 periodic patch selects about 178 µm, short of the 224 µm target, which is reported rather than forced. Equal diffusion is stable. The GIF (12 frames, 720x540) paints that field into a flank window of the Naturalis mount (Ghedoghedo, CC BY-SA 3.0). Hydration factors for Pakicetus (0.72) and Rodhocetus (1.35) are a transfer hypothesis, not fossil measurements. ANALYSIS_PASS is still 1.

- **Model fidelity to source: 90** — The script integrates the Schnakenberg system and the Turing dispersion eigenvalue. Onset k_c and the fastest-mode k are both computed and are not treated as the same number. Gamma rescales time only. The integer Lean file pins the trace-diffusion sign and the 190 < 224 < 260 µm order.  
  evidence: `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L58–88; `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L216–251; `morphoeval/lean/loops/Loop_3106_ambulocetus_natans_turing_wavelength_s0.lean` L55–63
- **Morphological realism / specimen match: 84** — The comparison figure places the Naturalis mount beside the simulated field and the last GIF frame. The mount is recognizable. The dermal pattern is an inset, not a claim that the bones are striped. Soft-tissue pigment is absent from the fossil, so this cannot be a pigment-photograph match.  
  evidence: `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L345–372; `docs/media/ambulocetus_natans_turing_wavelength_s0_comparison.png` L0–0; `docs/media/ambulocetus_natans_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 82** — Hydration h moves the analytic wavelength as sqrt(h), and the bar chart shows Pakicetus < Ambulocetus < Rodhocetus. Equal diffusion flips the sign of the maximum growth rate. Pass 1 does not yet re-simulate the nonlinear patch at the predecessor and successor hydrations.  
  evidence: `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L148–183; `docs/media/ambulocetus_natans_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 88** — Analytic versus Euler growth rates give Pearson r=0.9997 and RMSE=0.0034. The nonlinear radial peak is about 178 µm against the 224 µm slot target. Otsu foreground fraction, structure-tensor coherency and a KS test compare the fossil photograph with the field, and those pairs are not value==reference.  
  evidence: `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L116–145; `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L522–598
- **Source and data traceability: 92** — Primary DOIs are pinned for Turing, Schnakenberg, Thewissen 1994, Kondo and Miura, Gingerich 1994 and Thewissen 2001. The specimen file carries artist, date, Commons URL, CC BY-SA 3.0 and a SHA-256 checked at runtime.  
  evidence: `datasets/ambulocetus_natans_turing_wavelength_s0/ambulocetus_natans_turing_wavelength_s0_reference.json` L63–73; `datasets/ambulocetus_natans_turing_wavelength_s0/paper_citations.bib` L1–40
- **Code-to-model traceability: 86** — onset_kc, sigma_of_k, run_schnakenberg and lambda_of_h are named for the expressions in the reference JSON. Lean theorems are the integer skeleton of the same sign and ordering statements, not a rewrite of the floats.  
  evidence: `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L68–88; `morphoeval/math_kg_loops/delphi10000-ambulocetus-natans-turing-wavelength-s0.yaml` L14–45
- **Reproducibility package: 90** — --self-test checks r>0.98, the 224 µm target, ablation sign, chronoseries order, GIF frame count and the 480 px floor. Seed 3106 and the specimen digest are fixed. ANALYSIS_PASS=1 is the pass-1 gate.  
  evidence: `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L36–45; `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L613–628

Remediation:
- {'id': 'R1', 'item': 'ANALYSIS_PASS is 1, so extra seeds are skipped and no cross-seed wavelength CV is reported.'}
- {'id': 'R2', 'item': 'Specimen segmentation is global Otsu. Largest-component cleanup is gated off, so dark-background specks can enter the mask.'}
- {'id': 'R3', 'item': 'Pakicetus and Rodhocetus wavelengths are analytic sqrt(h) only. The nonlinear patch is not re-run at h=0.72 and h=1.35.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Ambulocetus natans Turing wavenumber at 224 µm (47.5 Ma, detail primordium)
- **spec_fingerprint** (semantic model + inputs): `82d5550289ea52c71a2cf9496547413e4cf4267d76c981bb4448f33443a3774d`
- **input_dataset_sha256**: `datasets/ambulocetus_natans_turing_wavelength_s0/ambulocetus_natans_naturalis_specimen.jpg` (`738e2188c8bb2936…`), `datasets/ambulocetus_natans_turing_wavelength_s0/ambulocetus_natans_turing_wavelength_s0_reference.json` (`0246234fefb8db6a…`)
- `docs/media/ambulocetus_natans_turing_wavelength_s0_comparison.png` — artifact_sha256 `13ea89dadbd2be7a…`, 246,374 B, 1008×656 px
- `docs/media/ambulocetus_natans_turing_wavelength_s0_growth.gif` — artifact_sha256 `d0ad8d52ca0864ae…`, 3,240,023 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion Pearson r (analytic eigenvalue vs forward Euler) | 0.999699 correlation | 1.0 | pearsonr of σ(k) on 28 wavenumbers *(RMSE=0.00336)* |
| dispersion RMSE | 0.003356 growth-rate units | 0.0 | root mean square of analytic minus Euler σ(k) |
| specimen image foreground fraction | 0.1783 area fraction | 0.1136 | Otsu threshold on the Naturalis mount luminance *(bbox aspect=1.478)* |
| specimen image structure-tensor coherency | 0.3591 coherency | 0.1136 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor on the fossil photograph |
| specimen image vs model orientation KS statistic | 0.066 KS distance | 0.0 | ks_2samp on Sobel orientation samples from the fossil and the simulated field *(p=0.00182)* |
| emergent stripe wavelength | 177.8 µm | 224.0 | radial FFT peak of the nonlinear Schnakenberg field, scaled so the analytic fastest mode is 224 µm *(analytic fastest mode mapped to 224 µm; onset k_c=0.662, fastest k=0.394.)* |
| equal-diffusion ablation final activator std | 0.00719 concentration | 1.34247 | nonlinear Schnakenberg with Du=Dv, same noise amplitude *(control initial std=0.0403; sigma_max equal=-0.0103)* |

## Environment at checkpoint

- python: 3.12
- numpy: 2.5
- scipy: 1.18

Pass 1 working tree. Commit hash updated to the loop commit that contains these paths before the knowledge-graph build.
