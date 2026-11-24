# MorphoEval report — audit `2026-10-27-bacillus-subtilis-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `e9fd00d16157d4244d56076d8b43ea8013dc6dec` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Bacillus subtilis Turing wavenumber at 16 µm (Holocene detail scale) (`models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py`) | 84 | 86 | 86 | 74 | 90 | 86 | 88 | 84.7 | 81.5 | 88.1 | 84.4 | **84.4** | faithful |

## Findings and remediation

### Bacillus subtilis Turing wavenumber at 16 µm (Holocene detail scale)  —  overall 84.4 (faithful)

Pass 1 implements the Schnakenberg dispersion and calibrates the linear wavenumber to 16.02 µm (target 16 µm) at age 0 Ma. Pearson r of early-time mode slopes against the discrete Laplacian dispersion is 0.985. The bulk field is still near the noise floor (std 0.017) and its radial peak is 17.45 µm. The ocellus GIF matches the specimen at CV 91.9 but the branches are faint regular rays, and the specimen angular detector peaks at harmonic 20 rather than the model count 72. Equal-diffusion ablation suppresses growth. Signing was not available (no ~/.nanopub/profile.yml).

- **Model fidelity to source: 84** — The script implements the Schnakenberg PDE, the Jacobian at (u0, v0)=(1, 0.916), and sigma_plus(k), and it places the linear peak at 16.02 µm with diffusion ratio 9.20 above the threshold 8.00. The bulk integration (1800 steps) never leaves the linear noise floor, so the nonlinear pattern the dispersion is supposed to select is not yet visible.  
  evidence: `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L109–142; `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L286–330
- **Morphological realism / specimen match: 86** — The comparison figure places the Korinna 2019 ocellus beside the reconstruction, and the 12-frame 520 px GIF scores cv_sim_similarity 91.9 against that photograph. The rays are much fainter and more regular than the dendritic branches in the micrograph, so the match is textural rather than anatomical.  
  evidence: `docs/media/bacillus_subtilis_turing_wavelength_s0_comparison.png` L1–1; `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L383–423
- **Parameter expressivity: 86** — A five-point sweep of D_u at fixed ratio 9.20 increases wavelength monotonically, which is the control the wavenumber law claims. The equal-diffusion ablation drives sigma_max to -0.085 and the field variance down by about 100-fold. The sweep is of the linear dispersion, not of the rendered ocellus.  
  evidence: `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L234–257; `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L259–284
- **Morphometric validation: 74** — Early-time Fourier slopes correlate with the discrete-Laplacian dispersion at Pearson r = 0.9847, above 0.98. RMSE is 0.108 because mode n=6 is a poor fit; modes 11 through 20 agree to about 1 percent. The bulk wavelength 17.45 µm is not the 16 µm target. The specimen angular harmonic is 20 against a model branch count of 72, so that comparison does not yet validate the front.  
  evidence: `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L204–232; `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L332–377
- **Source and data traceability: 90** — Primary DOIs resolve on OpenAlex (Turing 1952, Kawasaki 1997, Golding 1998, Ben-Jacob 2000, Kondo 2010, Duran-Nebreda 2021, Sharpe 1998, Dion 2019). The specimen file is SHA-256 pinned and attributed to Korinna, CC BY 4.0. The 16 µm radius is explicitly a reconstruction because the micrograph has no scale bar.  
  evidence: `datasets/bacillus_subtilis_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/bacillus_subtilis_turing_wavelength_s0/bacillus_subtilis_turing_wavelength_s0_reference.json` L68–80
- **Code-to-model traceability: 86** — calibrate_diffusivity, sigma_plus, and the module docstring name the dispersion relation and the 16 µm target. The Lean file states the same integer threshold and the 16/10 scale identities. Lean was not executed in this environment.  
  evidence: `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L1–45; `morphoeval/lean/loops/Loop_3130_bacillus_subtilis_turing_wavelength_s0.lean` L24–32
- **Reproducibility package: 88** — Seed 3130, --self-test, and a deterministic renderer. Self-test checks the trace, the diffusion threshold, Pearson r > 0.98, monotonic sweep, and the ablation. Pass 1 metrics are written to pass_metrics.json.  
  evidence: `models/2026-delphi10000-bacillus-subtilis-turing-wavelength-s0.py` L655–690

Remediation:
- Integrate the bulk Schnakenberg field into the nonlinear regime so the dominant radial wavelength locks to the linearly most unstable mode near 16 µm (pass 1 std(u)=0.017, lambda_meas=17.45 µm).
- Restrict the dispersion goodness-of-fit to modes with a real eigenvalue. Mode n=6 (deeply stable) inflates RMSE to 0.108 while the Turing band itself agrees to about 1 percent.
- Replace the specimen angular-FFT harmonic (peak 20, an envelope mode) with a band-passed ridge count in the branch annulus, and darken the reconstructed rays so they read as colony branches rather than faint pinstripes.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Bacillus subtilis Turing wavenumber at 16 µm (Holocene detail scale)
- **spec_fingerprint** (semantic model + inputs): `8c9e9022c4aac6bcf41bc35b63530ab7c52c0841a7764224d5403753680be47b`
- **input_dataset_sha256**: `datasets/bacillus_subtilis_turing_wavelength_s0/bacillus_subtilis_ocellus_specimen.jpg` (`94a0bad84960d16e…`), `datasets/bacillus_subtilis_turing_wavelength_s0/bacillus_subtilis_turing_wavelength_s0_reference.json` (`4cb53436d064578e…`)
- `docs/media/bacillus_subtilis_turing_wavelength_s0_growth.gif` — artifact_sha256 `af51b19a9d998054…`, 1,616,817 B, 520×520 px, 12 frames

| measurement | value | reference | method |
|---|---:|---|---|
| calibrated Turing wavelength lambda_c | 16.02002503128911 µm | 16.0 | argmax_k of Schnakenberg σ₊(k) at diffusion ratio 9.20 *(detail variant of tier 4_cellular_10um; tier center is 10 µm)* |
| bulk PDE dominant wavelength | 17.454545454545453 µm | 16.0 | radial FFT peak of the nonlinear Schnakenberg field on a 192 µm periodic square *(radial mode n = 11)* |
| dispersion Pearson r (numerical slope vs discrete Laplacian σ) | 0.9846945786141443 correlation | 1.0 | early-time log-amplitude slope of Fourier modes n=6..20 versus σ₊(k_eff) *(RMSE = 0.108146; KS statistic = 0.1000; KS p = 1.0000)* |
| dispersion RMSE | 0.10814619859536843 1/time | 0.0 | root mean square of σ_measured − σ_predicted over 10 Fourier modes |
| dispersion KS statistic | 0.10000000000000009 KS distance | 0.0 | two-sample Kolmogorov-Smirnov between measured and predicted growth rates *(p = 1.0000)* |
| specimen ocellus angular branch harmonic | 20 angular mode | 72 | FFT peak of the dark-branch annulus (0.22–0.40 of the frame) on the Korinna 2019 micrograph *(reference is the model circumference mode m = round(2π R / 16 µm), not a copy of the image peak)* |
| specimen foreground fraction | 0.1694335255066387 area fraction | 0.35 | pixels inside the circular stop darker than agar median minus 0.08 *(disk circularity = 0.0036; coherency = 0.5223)* |
| specimen structure-tensor coherency | 0.5222689259070664 coherency | 0.5 | mean (λ1−λ2)/(λ1+λ2) of the Sobel structure tensor inside the microscope stop |
| equal-diffusion ablation sigma_max | -0.08486301655606981 1/time | 0.0 | max σ₊(k) at D_u = D_v (ratio 1 < critical ratio) *(variance ratio final/initial = 0.0094)* |
| wavenumber ratio versus 10 µm tier center | 0.625 ratio | 0.625 | k(16 µm)/k(10 µm) = 10/16 *(chronoseries is 1/1 so the age change versus a predecessor species is 0 Ma; the scale change is this ratio)* |

## Environment at checkpoint

- python: 3.12

Pass 1 of loop 3130. Lean binary was not installed in this environment, so the Loop_3130 theorems were not machine-checked here.
