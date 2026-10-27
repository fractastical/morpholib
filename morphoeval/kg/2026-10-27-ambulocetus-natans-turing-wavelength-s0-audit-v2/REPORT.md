# MorphoEval report — audit `2026-10-27-ambulocetus-natans-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `dff4f2b5` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-ambulocetus-natans-turing-wavelength-s0-audit -> 2026-10-27-ambulocetus-natans-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ambulocetus natans Turing wavenumber at 224 µm (47.5 Ma, detail primordium) (`models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py`) | 91 | 84 | 90 | 91 | 92 | 87 | 91 | 90.7 | 86.6 | 90.3 | 88.9 | **88.9** | faithful |

## Findings and remediation

### Ambulocetus natans Turing wavenumber at 224 µm (47.5 Ma, detail primordium)  —  overall 88.9 (faithful)

Pass 2 sets ANALYSIS_PASS=2. Three seeds give emergent wavelengths 172.8, 175.9 and 177.4 µm (the pass-1 single-seed value was 177.8 µm). Re-running the nonlinear patch at the chronoseries hydrations gives 127.7 µm (Pakicetus, h=0.72), 177.4 µm (Ambulocetus, h=1) and 239.5 µm (Rodhocetus, h=1.35), monotone and in the same order as 190, 224 and 260 µm from lambda = 224 µm * sqrt(h). The fossil mask now keeps the largest Otsu component (foreground fraction 0.178 to 0.161). Dispersion fit is unchanged: Pearson r=0.9997, RMSE=0.0034. The nonlinear peak remains shorter than the analytic fastest mode; that gap is reported. GIF is 12 frames at 720x540. Direct CV similarity of the last frame to the Naturalis photograph is 91.6.

- **Model fidelity to source: 91** — Same Schnakenberg system and eigenvalue dispersion as pass 1 (r=0.9997). Pass 2 adds a log-parabola radial peak so the reported wavelength is not an integer FFT bin, and it re-integrates the PDE at the predecessor and successor hydrations instead of only scaling the closed form.  
  evidence: `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L68–88; `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L192–251; `morphoeval/lean/loops/Loop_3106_ambulocetus_natans_turing_wavelength_s0.lean` L31–53
- **Morphological realism / specimen match: 84** — Unchanged composite: the Naturalis mount fills the frame and a flank window shows the growing activator field. The mount is recognizable. Pigment is not preserved on the fossil, so the inset is a model overlay.  
  evidence: `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L345–372; `docs/media/ambulocetus_natans_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 90** — Nonlinear emergent wavelength increases with hydration: 127.7, 177.4, 239.5 µm at h=0.72, 1.0, 1.35. That is the same order as the analytic sqrt(h) law (190, 224, 260 µm). Equal diffusion still suppresses the band. The yellow points on the chronoseries panel are those re-runs.  
  evidence: `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L457–463; `docs/media/ambulocetus_natans_turing_wavelength_s0_comparison.png` L0–0
- **Morphometric validation: 91** — Dispersion Pearson r=0.9997 and RMSE=0.0034 stand. Three seeds span 172.8–177.4 µm. Specimen coherency 0.359 versus model coherency 0.114, KS=0.066, foreground fraction 0.161 after largest-component cleanup. Emergent 177.4 µm is not overwritten to equal the 224 µm reference.  
  evidence: `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L290–308; `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L522–598; `datasets/ambulocetus_natans_turing_wavelength_s0/ambulocetus_natans_turing_wavelength_s0_metrics.json` L0–0
- **Source and data traceability: 92** — Unchanged provenance: six DOIs, Commons photograph by Ghedoghedo (2011), CC BY-SA 3.0, SHA-256 checked before the run.  
  evidence: `datasets/ambulocetus_natans_turing_wavelength_s0/ambulocetus_natans_turing_wavelength_s0_reference.json` L63–73; `datasets/ambulocetus_natans_turing_wavelength_s0/paper_citations.bib` L1–70
- **Code-to-model traceability: 87** — The pass-2 gate is the ANALYSIS_PASS branch in run_all, next to lambda_of_h and the Lean statements of the same ordering. Float parameters stay in the script; Lean keeps the integer sign conditions.  
  evidence: `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L43–46; `morphoeval/math_kg_loops/delphi10000-ambulocetus-natans-turing-wavelength-s0.yaml` L14–48
- **Reproducibility package: 91** — --self-test still requires r>0.98, ablation sign, chronoseries order, at least 10 frames and 480 px. Pass 2 adds the extra seeds and hydration runs under the same seed family (3106, 3123, 3147) without changing the pinned specimen digest.  
  evidence: `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L613–628; `models/2026-delphi10000-ambulocetus-natans-turing-wavelength-s0.py` L436–443

Remediation:
- {'id': 'R4', 'item': 'Emergent wavelength stays near 177 µm against the analytic fastest-mode target of 224 µm (ratio about 0.79 at h=1). Reported as a nonlinear shift, not adjusted to force agreement.'}

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Ambulocetus natans Turing wavenumber at 224 µm (47.5 Ma, detail primordium)
- **spec_fingerprint** (semantic model + inputs): `82d5550289ea52c71a2cf9496547413e4cf4267d76c981bb4448f33443a3774d`
- **input_dataset_sha256**: `datasets/ambulocetus_natans_turing_wavelength_s0/ambulocetus_natans_naturalis_specimen.jpg` (`738e2188c8bb2936…`), `datasets/ambulocetus_natans_turing_wavelength_s0/ambulocetus_natans_turing_wavelength_s0_reference.json` (`0246234fefb8db6a…`)
- `datasets/ambulocetus_natans_turing_wavelength_s0/ambulocetus_natans_turing_wavelength_s0_metrics.json` — artifact_sha256 `448043f686b7c5db…`, 2,310 B, 
- `docs/media/ambulocetus_natans_turing_wavelength_s0_comparison.png` — artifact_sha256 `efd6b1735e46c2ca…`, 248,114 B, 1008×656 px
- `docs/media/ambulocetus_natans_turing_wavelength_s0_growth.gif` — artifact_sha256 `d0ad8d52ca0864ae…`, 3,240,023 B, 720×540 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| dispersion Pearson r (analytic eigenvalue vs forward Euler) | 0.999699 correlation | 1.0 | pearsonr of σ(k) on 28 wavenumbers *(RMSE=0.00336)* |
| dispersion RMSE | 0.003356 growth-rate units | 0.0 | root mean square of analytic minus Euler σ(k) |
| specimen image foreground fraction | 0.1613 area fraction | 0.1136 | Otsu threshold on the Naturalis mount luminance plus largest-component cleanup *(bbox aspect=1.382)* |
| specimen image structure-tensor coherency | 0.3591 coherency | 0.1136 | mean (λ1-λ2)/(λ1+λ2) of the Sobel structure tensor on the fossil photograph |
| specimen image vs model orientation KS statistic | 0.066 KS distance | 0.0 | ks_2samp on Sobel orientation samples from the fossil and the simulated field *(p=0.00182)* |
| emergent stripe wavelength | 177.374 µm | 224.0 | radial FFT peak of the nonlinear Schnakenberg field, scaled so the analytic fastest mode is 224 µm *(analytic fastest mode mapped to 224 µm; onset k_c=0.662, fastest k=0.394. multi-seed λ µm = [175.94, 172.8, 177.37], CV=0.0109)* |
| equal-diffusion ablation final activator std | 0.00719 concentration | 1.34247 | nonlinear Schnakenberg with Du=Dv, same noise amplitude *(control initial std=0.0403; sigma_max equal=-0.0103)* |

## Environment at checkpoint

- python: 3.12
- numpy: 2.5
- scipy: 1.18

Pass 2 working tree on the loop branch. Origin URL in this checkout does not match the plain repository URL, so commit-pinned loci are not git-verified here.
