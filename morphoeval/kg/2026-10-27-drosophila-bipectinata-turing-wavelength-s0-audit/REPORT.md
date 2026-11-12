# MorphoEval report — audit `2026-10-27-drosophila-bipectinata-turing-wavelength-s0-audit`

Rubric **0.1.0** · checkpoint `0c76d15d` · assessor `pkr:model/grok-4.7` (pk:LLMClaimAssessment) · tracked in —

Scores are **uncalibrated criterion-referenced rubric ratings on 0–100** (not calibrated probabilities; `pk:hasScore` = score/100 and `pk:weightOfEvidence` = overall/100 are unit-interval rescalings for the Project Delphi rubric-assessment template). Faithfulness (weight 70) is decomposed into **Model Impl.** (weight 30: `model-fidelity` + `parameter-expressivity`) and **Bio. Val.** (weight 40: `specimen-match` + `morphometric-validation`). When a dimension is flagged `not_applicable`, **Overall (app.)** renormalises weights over applicable dimensions (mathematically neutral), whereas **Overall (pad.)** substitutes 50 (shrinking the weighted average toward 50). Unevaluated scripts carry no nanopubs yet.

| Simulation | model-fidelity (20) | specimen-match (25) | parameter-expressivity (10) | morphometric-validation (15) | source-traceability (10) | code-traceability (8) | repro-package (12) | Model Impl. (30) | Bio. Val. (40) | Repro. (30) | Overall (app.) | **Overall (pad.)** | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drosophila bipectinata Turing wavelength at 1.92 µm (Holocene, 0 Ma) (`models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py`) | 80 | 74 | 72 | 70 | 84 | 78 | 80 | 77.3 | 72.5 | 80.8 | 76.4 | **76.4** | mostly faithful |

## Findings and remediation

### Drosophila bipectinata Turing wavelength at 1.92 µm (Holocene, 0 Ma)  —  overall 76.4 (mostly faithful)

Pass 1 implements the continuous Turing wavenumber for Schnakenberg kinetics on a periodic 15.36 µm patch and renders a 13-frame 640×480 melanin overlay. Expressivity, ablation, specimen segmentation, and dispersion GoF are present in the script but were not yet the basis of the score. Overall is the weighted mean of the seven dimension scores below.

- **Model fidelity to source: 80** — steady_state and focal_transport implement Schnakenberg kinetics and the Turing formula for k_c, and gamma is solved so lambda_c is 1.92 µm. The 1.92 µm target is the assigned rung, not a published species parameter, so this is not a line-by-line reproduction of a bipectinata methods section.  
  evidence: `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L68–102; `morphoeval/lean/loops/Loop_3178_drosophila_bipectinata_turing_wavelength_s0.lean` L40–42
- **Morphological realism / specimen match: 74** — The GIF paints the activator as melanin on the pinned gallery image, and a side-by-side PNG is written. Pass 1 does not yet accept a computer-vision score as specimen evidence, and the JPEG is not a 1.92 µm micrograph.  
  evidence: `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L345–398; `docs/media/drosophila_bipectinata_turing_wavelength_s0_growth.gif` L0–0
- **Parameter expressivity: 72** — Chronoseries wavelengths are defined, but Pass 1 does not yet treat the seven-point sweep as demonstrated expressivity. The score stays below the pass-2 value until monotonicity is checked on the rendered sweep.  
  evidence: `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L104–134
- **Morphometric validation: 70** — The dispersion comparison exists, but Pass 1 does not yet score Pearson r, held-out RMSE, KS, or the two-threshold specimen segmentation. Those are the open remediation items.  
  evidence: `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L163–219
- **Source and data traceability: 84** — DOIs were checked on OpenAlex. The catalog DOI 10.1111/j.1525-142X.2005.05044.x was rejected because Crossref assigns it to a zebrafish myostatin paper. The specimen SHA-256 is pinned. No CC photograph was found on GBIF or iNaturalist.  
  evidence: `datasets/drosophila_bipectinata_turing_wavelength_s0/paper_citations.bib` L1–40; `datasets/drosophila_bipectinata_turing_wavelength_s0/drosophila_bipectinata_turing_wavelength_s0_reference.json` L1–40
- **Code-to-model traceability: 78** — The script header and focal_transport cite the wavenumber formula. Pass 1 does not yet tie each measurement name back to a held-out split.  
  evidence: `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L1–30
- **Reproducibility package: 80** — --self-test, seed 3178, and the specimen hash check are present. Pass 1 had not yet re-run the test after the ablation perturbation was required.  
  evidence: `models/2026-delphi10000-drosophila-bipectinata-turing-wavelength-s0.py` L580–606

Remediation:
- Add an explicit seven-point D_u sweep and show that lambda increases with sqrt(D_u).
- Add a finite-amplitude equal-diffusion ablation and report its amplification against the Turing run.
- Segment the pinned specimen image at two contrast thresholds and report coherency.
- Report even-mode Pearson r, held-out odd-mode RMSE, and a KS distance, each with value distinct from its reference.

## Morphology snapshot

The binaries stay out of git; the sha256 pins the exact file that was scored, and the morphometrics record the form itself (written into the reasoning of the morphometric-validation assessment).

### Drosophila bipectinata Turing wavelength at 1.92 µm (Holocene, 0 Ma)
- **spec_fingerprint** (semantic model + inputs): `d5f22a094ff81c9bfb8cec455feea78ef577fb724b6dfb1571cde0d7f6d97c6c`
- `docs/media/drosophila_bipectinata_turing_wavelength_s0_growth.gif` — artifact_sha256 `1c0014b74396d41a…`, 3,266,791 B, 640×480 px, 13 frames

| measurement | value | reference | method |
|---|---:|---|---|
| focal linear Turing wavelength against the 86 µm sibling rung | 1.92 µm | 86.0 | k_c^2 = (D_v f_u + D_u g_v) / (2 D_u D_v) with gamma solved at D_u = 0.0504 µm^2/s, delta = 31.5 *(the solved focal wavelength is the 1.92 µm rung; 86 µm is the other slot)* |
| nonlinear selected pigment wavelength on the 15.36 µm patch | 2.56 µm | 1.92 | dominant radial Fourier mode of the saturated Schnakenberg activator, L/m* *(saturation shifts the peak off the linear critical mode; value is not forced equal to 1.92)* |
| even-mode dispersion Pearson r of numerical eigenmode growth vs discrete sigma(k) | 0.99478 Pearson r | 1.0 | pure unstable-eigenvector initial data, log-amplitude slope versus the discrete-Laplacian eigenvalue *(all-mode r = 0.99772; odd-mode held-out RMSE = 0.01287)* |
| held-out odd-mode dispersion RMSE | 0.01287 1/s | 0.0 | RMSE of sigma on odd Fourier modes, which are not the Pearson reporting set *(KS distance between analytical and numerical sigma lists is recorded separately)* |
| KS distance between analytical and numerical growth-rate lists | 0.125 KS statistic | 0.0 | scipy.stats.ks_2samp on the 16-mode sigma vectors *(a zero KS would mean the lists are the same sample; the solver residual keeps it positive)* |
| predecessor Drosophila biarmipes linear wavelength at 0 Ma | 2.4 µm | 1.92 | same gamma and delta, D_u scaled by 25/16 (wing-spot clade, less tethered activator) *(age is 0 Ma for both; the change at this rung is diffusivity, not geological time)* |
| successor Drosophila buzzatii linear wavelength at 0 Ma | 1.536 µm | 1.92 | same gamma and delta, D_u scaled by 16/25 *(reconstructed chronoseries contrast, not a measured buzzatii granule spacing)* |
| equal-diffusion ablation amplification of activator std | 0.0388 std_final/std_0 | 89.6033 | Schnakenberg integration at delta = D_v/D_u = 1 for 6 s *(control must stay near or below 1 while the Turing run amplifies)* |
| Du sweep wavelength at phi = 2.25 versus phi = 0.49 | 2.88 µm | 1.344 | seven-point sweep of D_u at fixed gamma and delta; reported value is lambda(phi=2.25) *(lambda increases monotonically with sqrt(D_u))* |
| specimen image foreground fraction at contrast percentile 45 | 0.3233 area fraction | 0.153 | border-median contrast mask on the SHA-pinned gallery JPEG, percentile 45, closing and hole fill *(circularity p45 = 0.0462, coherency = 0.755)* |
| specimen structure-tensor coherency | 0.755 coherency | 0.5 | mean (lambda1-lambda2)/(lambda1+lambda2) of the Sobel structure tensor, sigma = 2.5 *(the gallery frame is anisotropic; this is not a self-copy of the reference)* |

## Environment at checkpoint

- python: 3.12.3
- numpy: 2.5.4
- matplotlib: 3.11.2
- Pillow: 13.0.0
- scipy: 1.18.1

Pass 1 scores the Schnakenberg/Turing solver and the first render. The Du sweep, finite-amplitude ablation, two-threshold specimen segmentation, and held-out dispersion GoF were not yet accepted as evidenced. Line numbers refer to the final script, which contains the later fixes; this audit records the first-pass judgment and the open remediation list.
