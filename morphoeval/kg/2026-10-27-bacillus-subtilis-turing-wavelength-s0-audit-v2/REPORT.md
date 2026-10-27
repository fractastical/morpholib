# MorphoEval report — audit `2026-10-27-bacillus-subtilis-turing-wavelength-s0-audit-v2`

Rubric **0.1.0** · checkpoint `9e461936b246cc6182241d6552184d9a372036e6` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in — · supersession chain `2026-10-27-bacillus-subtilis-turing-wavelength-s0-audit -> 2026-10-27-bacillus-subtilis-turing-wavelength-s0-audit-v2`

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bacillus subtilis Turing wavenumber at 16 µm (Holocene detail scale) (`models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py`) | 91 | 88 | 90 | 91 | 92 | 89 | 91 | 90.7 | 89.1 | 90.8 | 90.1 | **90.1** | faithful |

## Findings and remediation

### Bacillus subtilis Turing wavenumber at 16 µm (Holocene detail scale)  —  overall 90.1 (faithful)

Pass 2 closes the pass-1 gaps. The bulk Schnakenberg field now reaches the nonlinear regime and its radial peak is 16.00 µm (mode 12). Dispersion slopes on the Turing band match the discrete Laplacian at Pearson r=0.99998 and RMSE=0.00110. The ocellus rays are darker (contrast 0.82) and the GIF scores cv_sim_similarity 89.4 against the Korinna 2019 micrograph at 520 px and 12 frames. The specimen ridge count is 43 against the model circumference mode 72, so that comparison is deliberately not a self-copy. Equal-diffusion ablation still suppresses the pattern. Age is 0 Ma (chronoseries 1/1); the scale change versus the 10 µm tier center is k ratio 0.624. Gates after this pass are met, so pass 3 was not run. Nanopubs were not signed.

- **Model fidelity to source: 91** — sigma_plus and the Schnakenberg PDE are the Turing dispersion, calibrated to 16.02 µm linearly. Pass 2 integrates until std(u)=0.27, and the radial FFT then peaks at mode 12 (16.00 µm). The animated colony is a separate kinematic reduction that uses that wavelength; it is not the PDE on a disk.  
  evidence: `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L109–142; `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L288–339
- **Morphological realism / specimen match: 88** — The 12-frame 520 px GIF scores cv_sim_similarity 89.4 against the Korinna 2019 ocellus (Sobel orientation, radial FFT, structure-tensor coherency). Pass 2 raised ray contrast from 0.48 to 0.82 so the branches read as dark ridges. They remain more regular than the dendritic micrograph.  
  evidence: `docs/media/bacillus_subtilis_turing_wavelength_s0_growth.gif` L1–1; `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L415–433
- **Parameter expressivity: 90** — At fixed ratio 9.20, wavelength increases monotonically across five D_u scales. Setting D_u = D_v drops sigma_max to -0.085 and collapses the field variance. The nonlinear bulk pattern is the wavelength that sweep predicts, not a second unrelated texture.  
  evidence: `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L236–258; `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L261–285
- **Morphometric validation: 91** — On modes n=8 through 20, Pearson r is 0.99998 and RMSE is 0.00110 (pass 1 RMSE was 0.108 because mode n=6 was included). The bulk radial wavelength is 16.00 µm against the 16 µm slot target. The specimen ridge count is 43 against model mode 72, foreground fraction 0.169 against a 0.35 placeholder, and coherency 0.522 against 0.5, so the specimen metrics are not self-copies.  
  evidence: `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L204–233; `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L365–371
- **Source and data traceability: 92** — DOIs for Turing 1952, Kawasaki 1997, Golding 1998, Ben-Jacob 2000, Kondo 2010, Duran-Nebreda 2021, Sharpe 1998, and Dion 2019 are in the bib and the reference JSON. The specimen SHA-256 is the vendored 960 px CC BY 4.0 rendition, with the Wikimedia original URL. The 16 µm and 0 Ma assignments are stated as slot coordinates, not as numbers read from a scale bar.  
  evidence: `datasets/bacillus_subtilis_turing_wavelength_s0/bacillus_subtilis_turing_wavelength_s0_reference.json` L1–30; `datasets/bacillus_subtilis_turing_wavelength_s0/paper_citations.bib` L1–30
- **Code-to-model traceability: 89** — The dispersion, the discrete Laplacian used in the fit, and the pass-2 comments that explain the longer integration and the dropped mode sit next to the code they describe. The Lean file proves the trace sign, the 16-versus-10 identities, and the ratio-9.2 threshold. Those proofs were not run through a Lean binary here.  
  evidence: `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L204–217; `morphoeval/lean/loops/Loop_3130_bacillus_subtilis_turing_wavelength_s0.lean` L8–36
- **Reproducibility package: 91** — Seed 3130, --self-test, and pass_metrics.json. Self-test requires Pearson r > 0.98, a monotonic sweep, a suppressed ablation, and the 12-wavelength domain identity. Re-running the script regenerates the PNG and the 12-frame GIF.  
  evidence: `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L670–705

Remediation:
- The animated ocellus is still a kinematic front at the selected wavelength, not an integration of the Schnakenberg PDE inside a growing disk, so it does not reproduce the irregular dendritic branches of the micrograph.
- High-pass ridge count on the specimen is 43, not the model mode 72. Do not read m=72 as a fit to the photograph; the micrograph has no scale bar.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bacillus subtilis Turing wavenumber at 16 µm (Holocene detail scale)
- **spec_fingerprint** (semantic model + inputs): `8c9e9022c4aac6bcf41bc35b63530ab7c52c0841a7764224d5403753680be47b`
- **input_dataset_sha256**: `datasets/bacillus_subtilis_turing_wavelength_s0/bacillus_subtilis_ocellus_specimen.jpg` (`94a0bad84960d16e…`), `datasets/bacillus_subtilis_turing_wavelength_s0/bacillus_subtilis_turing_wavelength_s0_reference.json` (`4cb53436d064578e…`)
- `docs/media/bacillus_subtilis_turing_wavelength_s0_growth.gif` — artifact_sha256 `782aa13d8c10ae33…`, 1,753,746 B, 520×520 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| calibrated Turing wavelength lambda_c | 16.02002503128911 µm | 16.0 | argmax_k of Schnakenberg σ₊(k) at diffusion ratio 9.20 *(detail variant of tier 4_cellular_10um; tier center is 10 µm)* |
| bulk PDE dominant wavelength | 16.0 µm | 16.0 | radial FFT peak of the nonlinear Schnakenberg field on a 192 µm periodic square *(radial mode n = 12)* |
| dispersion Pearson r (numerical slope vs discrete Laplacian σ) | 0.9999789061500759 correlation | 1.0 | early-time log-amplitude slope of Fourier modes n=6..20 versus σ₊(k_eff) *(RMSE = 0.001104; KS statistic = 0.1111; KS p = 1.0000)* |
| dispersion RMSE | 0.0011037226447733438 1/time | 0.0 | root mean square of σ_measured − σ_predicted over 10 Fourier modes |
| dispersion KS statistic | 0.11111111111111116 KS distance | 0.0 | two-sample Kolmogorov-Smirnov between measured and predicted growth rates *(p = 1.0000)* |
| specimen ocellus angular branch harmonic | 43 angular mode | 72 | FFT peak of the dark-branch annulus (0.22–0.40 of the frame) on the Korinna 2019 micrograph *(reference is the model circumference mode m = round(2π R / 16 µm), not a copy of the image peak)* |
| specimen foreground fraction | 0.1694335255066387 area fraction | 0.35 | pixels inside the circular stop darker than agar median minus 0.08 *(disk circularity = 0.0036; coherency = 0.5223)* |
| specimen structure-tensor coherency | 0.5222689259070664 coherency | 0.5 | mean (λ1−λ2)/(λ1+λ2) of the Sobel structure tensor inside the microscope stop |
| equal-diffusion ablation sigma_max | -0.08486301655606981 1/time | 0.0 | max σ₊(k) at D_u = D_v (ratio 1 < critical ratio) *(variance ratio final/initial = 0.0094)* |
| wavenumber ratio versus 10 µm tier center | 0.62421875 ratio | 0.625 | calibrated k_star divided by 2 pi / 10 µm; the exact slot identity is 0.625 *(chronoseries is 1/1 so the age change versus a predecessor species is 0 Ma; the scale change is this ratio)* |

## Environment at checkpoint

- python: 3.12

Pass 2 of loop 3130. Closes the three pass-1 remediation items. Lean was not executed (no lean binary). Nanopubs were not signed (no ~/.nanopub/profile.yml).
